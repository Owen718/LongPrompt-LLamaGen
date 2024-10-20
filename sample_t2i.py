import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import time
import argparse
from autoregressive.tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate,generate_tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from request_prompt_refiner import Complex_Human_Instruction,Generate_Negative_Prompt


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")#, strict=True) 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")
    

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            #mode="max-autotune",
            fullgraph=True,
            mode="reduce-overhead",# should not use this mode for Lora
            # fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        T5_path=args.t5_path,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    prompts = [
        
        "A cutting board topped with bread, meat and vegetables",
        "A kitchen that is in the process of having the floors done",
        "a big purple bus parked in a parking spot",
        "A furry, black bear standing in a rocky, weedy, area in the wild.",
        "a baby penguin.",
        "red apples on a tree with green leaves",
        "a corgi head depicted as an explosion of a nebula",
        "a gorilla climbing up the side of the Great Pyramid"
    ]    
    if args.using_complex_prompt:
        # model_name = "llama3.2"
        # model_name="gemma2:2b"
        model_name = "qwen2.5:3b"
        max_prompt_length = args.t5_feature_max_len
        complex_prompts = [Complex_Human_Instruction(prompt,model=model_name, max_token_num=max_prompt_length) for prompt in prompts]


        # Count words in each prompt
        for i, prompt in enumerate(complex_prompts):
            word_count = len(prompt.split())
            print(f"Prompt {i+1} word count: {word_count}")
        
        
        negative_prompts = [Generate_Negative_Prompt(prompt,model=model_name) for prompt in prompts]
        prompts = complex_prompts
    else:
        negative_prompts = None
        
    t5_model.model.to(device)
    # caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)
    caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

    if negative_prompts is not None:
        print(f"generating negative prompts...",negative_prompts)
        negative_cond,negative_cond_masks = t5_model.get_text_embeddings(negative_prompts)
    else:
        negative_prompt = "worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution"
        negative_prompts = len(prompts) * [negative_prompt]
        negative_cond,negative_cond_masks = t5_model.get_text_embeddings(negative_prompts)

    del t5_model
    if not args.no_left_padding:
        print(f"processing left-padding...")    
        # a naive way to implement left-padding
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
    else:
        new_caption_embs, new_emb_masks = caption_embs, emb_masks
    c_indices = new_caption_embs * new_emb_masks[:,:, None]
    c_emb_masks = new_emb_masks

    qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
    t1 = time.time()
    with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16):
        index_sample = generate_tqdm(
            gpt_model, c_indices, latent_size ** 2, 
            c_emb_masks, 
            negative_cond_masks=negative_cond_masks,
            cfg_scale=args.cfg_scale,
            negative_cond = negative_cond,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            cls_token_num=args.cls_token_num
            )
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
    
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape)     # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")
    file_name = args.gpt_ckpt.split("/")[-1].split(".")[0]
    save_image(samples, "sample_{}_T{}.png".format(file_name, args.temperature), nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{file_name}_T{args.temperature}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=500, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--using-complex-prompt", action='store_true', default=False)
    args = parser.parse_args()
    main(args)
