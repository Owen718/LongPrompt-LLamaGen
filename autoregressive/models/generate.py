# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    """
    Sample a token from the logits using specified temperature, top-k, and top-p values.
    
    Args:
        logits: The logits distribution shape (batch size, vocabulary size).
        temperature: Controls the randomness of the sampling. Higher values make the output more random.
        top_k: If greater than 0, only the top k tokens with the highest probability are kept for sampling.
        top_p: If less than 1.0, only the top tokens with cumulative probability >= top_p are kept for sampling.
        sample_logits: If True, samples from the logits using a multinomial distribution. If False, takes the top token.
    
    Returns:
        idx: The sampled token index.
        probs: The probability distribution of the tokens.
    """
    # Adjust the logits by dividing by the temperature, ensuring the temperature is at least 1e-5 to avoid division by zero.
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    
    # Apply top-k and top-p filtering if specified.
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    # Convert the logits to a probability distribution.
    probs = F.softmax(logits, dim=-1)
    
    # Sample a token index from the probability distribution if sample_logits is True, otherwise take the top token.
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    # Adjust the logits by dividing by the temperature, ensuring the temperature is at least 1e-5 to avoid division by zero.
    logits = logits / max(temperature, 1e-5)
    
    # Apply top-k and top-p filtering if specified.
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    # Check if the classifier-free guidance scale is greater than 1.0
    if cfg_scale > 1.0:
        # Generate logits using the model with no input tokens but with conditioning and input positions
        logits, _ = model(None, cond_idx, input_pos)
        # Combine the logits into a single tensor
        logits_combined = logits
        # Split the combined logits into conditional and unconditional logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        # Apply classifier-free guidance by scaling the difference between conditional and unconditional logits
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        # If classifier-free guidance scale is not greater than 1.0, generate logits without scaling
        logits, _ = model(None, cond_idx, input_pos)

    # Sample the next token from the logits using the provided sampling kwargs
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    # Initialize lists to store the new tokens and their probabilities
    new_tokens, new_probs = [], []
    # Initialize the classifier-free guidance flag to True
    cfg_flag = True
    # Loop through the number of new tokens to generate
    for i in range(num_new_tokens):
        # Use specific CUDA SDP kernel settings for better performance with Inductor
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            # Check if the classifier-free guidance interval is set and if the current token index exceeds it
            if cfg_interval > -1 and i > cfg_interval:
                # Disable classifier-free guidance after the specified interval
                cfg_flag = False
            # Generate the next token and its probability using the decode_one_token function
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            # Increment the input position for the next token
            input_pos += 1
            # Append the new token and its probability to their respective lists
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            # Update the current token to the newly generated token for the next iteration
            cur_token = next_token.view(-1, 1)
    
    # Return the lists of new tokens and their probabilities
    return new_tokens, new_probs

# @torch.no_grad()


@torch.inference_mode()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    # Check the model type to determine the appropriate conditioning strategy
    if model.model_type == 'c2i':
        # For 'c2i' model type, if cfg_scale is greater than 1.0, create a null conditioning tensor
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            # Combine the original conditioning tensor with the null conditioning tensor
            cond_combined = torch.cat([cond, cond_null])
        else:
            # If cfg_scale is not greater than 1.0, use the original conditioning tensor
            cond_combined = cond
        # Set the sequence length T to 1 for 'c2i' model type
        T = 1
    elif model.model_type == 't2i':
        # For 't2i' model type, if cfg_scale is greater than 1.0, create a null conditioning tensor
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            # Combine the original conditioning tensor with the null conditioning tensor
            cond_combined = torch.cat([cond, cond_null])
        else:
            # If cfg_scale is not greater than 1.0, use the original conditioning tensor
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")
    # Calculate the new sequence length by adding the maximum number of new tokens to the current sequence length
    T_new = T + max_new_tokens
    # Set the maximum sequence length to the new sequence length
    max_seq_length = T_new
    # Determine the maximum batch size based on the shape of the conditioning tensor
    max_batch_size = cond.shape[0]

    # Get the device where the conditioning tensor is located
    device = cond.device
    # Set the device context to ensure all subsequent operations are performed on the correct device
    with torch.device(device):
        # If CFG scale is greater than 1.0, double the maximum batch size to accommodate combined conditioning tensors
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        # Set up model caches with the appropriate maximum batch size and sequence length, and the same data type as the token embeddings
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    # Check if embedding masks are provided
    if emb_masks is not None:
        # Ensure the first dimension of embedding masks matches the maximum batch size
        assert emb_masks.shape[0] == max_batch_size
        # Ensure the last dimension of embedding masks matches the current sequence length T
        assert emb_masks.shape[-1] == T
        # If CFG scale is greater than 1.0, concatenate the embedding masks with themselves
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            # If CFG scale is not greater than 1.0, use the original embedding masks
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        # Create an identity matrix of the same size as the causal mask
        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        # Update the causal mask by combining it with the identity matrix
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    # Create a tensor representing the input positions from 0 to T (current sequence length)
    input_pos = torch.arange(0, T, device=device)
    # Generate the next token using the prefill function with combined conditioning, input positions, and sampling kwargs
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    # Place the generated next token into the sequence tensor at the position starting from T to T+1
    seq[:, T:T+1] = next_token

    # Create a tensor representing the next input position, which is T (current sequence length)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    # Generate the remaining tokens by decoding n tokens using the model, next token, input position, and sampling kwargs
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    # Place the generated tokens into the sequence tensor starting from position T+1
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    # Return the generated sequence starting from position T
    return seq[:, T:]






import torch
from tqdm import tqdm
import time

@torch.inference_mode()
def generate_tqdm(model, cond, max_new_tokens, emb_masks=None,negative_cond_masks=None, cfg_scale=1.0, cfg_interval=-1,negative_cond=None,cls_token_num=120, **sampling_kwargs):
    # Check the model type to determine the appropriate conditioning strategy
    if model.model_type == 'c2i':
        # For 'c2i' model type, if cfg_scale is greater than 1.0, create a null conditioning tensor
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            # Combine the original conditioning tensor with the null conditioning tensor
            cond_combined = torch.cat([cond, cond_null])
        else:
            # If cfg_scale is not greater than 1.0, use the original conditioning tensor
            cond_combined = cond
        # Set the sequence length T to 1 for 'c2i' model type
        T = 1
    elif model.model_type == 't2i':
        # For 't2i' model type, if cfg_scale is greater than 1.0, create a null conditioning tensor
        if cfg_scale > 1.0:
            if negative_cond is None:
                cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            else:
                b = cond.size()[0]
                # import pdb
                # pdb.set_trace()
                #if cls_token_num==120:
                cond_null = negative_cond #+ model.cls_embedding.uncond_embedding  
                # else:
                #     cond_null = negative_cond 
                negative_cond_masks = negative_cond_masks#.repeat(b,1)
            # Combine the original conditioning tensor with the null conditioning tensor
            cond_combined = torch.cat([cond, cond_null])
        else:
            # If cfg_scale is not greater than 1.0, use the original conditioning tensor
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")
    # Calculate the new sequence length by adding the maximum number of new tokens to the current sequence length
    T_new = T + max_new_tokens
    # Set the maximum sequence length to the new sequence length
    max_seq_length = T_new
    # Determine the maximum batch size based on the shape of the conditioning tensor
    max_batch_size = cond.shape[0]

    # Get the device where the conditioning tensor is located
    device = cond.device
    # Set the device context to ensure all subsequent operations are performed on the correct device
    with torch.device(device):
        # If CFG scale is greater than 1.0, double the maximum batch size to accommodate combined conditioning tensors
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        # Set up model caches with the appropriate maximum batch size and sequence length, and the same data type as the token embeddings
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    # Check if embedding masks are provided
    if emb_masks is not None:
        # Ensure the first dimension of embedding masks matches the maximum batch size
        assert emb_masks.shape[0] == max_batch_size
        # Ensure the last dimension of embedding masks matches the current sequence length T
        assert emb_masks.shape[-1] == T
        # If CFG scale is greater than 1.0, concatenate the embedding masks with themselves
        if cfg_scale > 1.0:
            if negative_cond_masks is not None:
                # import pdb
                # pdb.set_trace()
                model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, negative_cond_masks]).unsqueeze(1)
            # else:
            #     model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            # If CFG scale is not greater than 1.0, use the original embedding masks
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        # Create an identity matrix of the same size as the causal mask
        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        # Update the causal mask by combining it with the identity matrix
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    # Create a tensor representing the input positions from 0 to T (current sequence length)
    input_pos = torch.arange(0, T, device=device)
    # Generate the next token using the prefill function with combined conditioning, input positions, and sampling kwargs
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    # Place the generated next token into the sequence tensor at the position starting from T to T+1
    seq[:, T:T+1] = next_token

    # Create a tensor representing the next input position, which is T (current sequence length)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    
    # Use tqdm to create a progress bar
    pbar = tqdm(total=max_new_tokens-1, desc="Generating tokens")
    generated_tokens = []
    start_time = time.time()

    for i in range(max_new_tokens-1):
        # Generate the next token
        next_token, _ = decode_one_token(model, next_token, input_pos, cfg_scale, cfg_flag=(cfg_interval == -1 or i <= cfg_interval), **sampling_kwargs)
        generated_tokens.append(next_token)
        
        # Increment the input position
        input_pos += 1
        
        # Update the progress bar
        pbar.update(1)
        
        # Calculate and display the time for this step
        step_time = time.time() - start_time
        pbar.set_postfix({"Step Time": f"{step_time:.4f}s"})
        
        # Reset the start time for the next step
        start_time = time.time()

    pbar.close()

    # Place the generated tokens into the sequence tensor starting from position T+1
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    # Return the generated sequence starting from position T
    return seq[:, T:]
