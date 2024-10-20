# 🚀 LongPrompt-LLamaGen

![./sample_0135000_model_only_T1.2_2.png](./sample_0135000_model_only_T1.2_2.png)


<p align="center">
  <a href="https://github.com/owen718/LongPrompt-LLamaGen/stargazers"><img src="https://img.shields.io/github/stars/owen718/LongPrompt-LLamaGen?style=social" alt="Stars"></a>
  <a href="https://github.com/owen718/LongPrompt-LLamaGen/network/members"><img src="https://img.shields.io/github/forks/owen718/LongPrompt-LLamaGen?style=social" alt="Forks"></a>
  <a href="https://github.com/owen718/LongPrompt-LLamaGen/issues"><img src="https://img.shields.io/github/issues/owen718/LongPrompt-LLamaGen" alt="Issues"></a>
  <a href="https://github.com/owen718/LongPrompt-LLamaGen/blob/main/LICENSE"><img src="https://img.shields.io/github/license/owen718/LongPrompt-LLamaGen" alt="License"></a>
  <a href="https://huggingface.co/Owen777/LongPrompt-LLamaGen"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue" alt="Hugging Face Model"></a>
  <a href="https://github.com/FoundationVision/LlamaGen"><img src="https://img.shields.io/badge/Original-LlamaGen-orange" alt="Original LlamaGen"></a>
</p>

## 🌟 Introduction

LongPrompt-LLamaGen is a improved LLamaGen modell that combines long-text prompts with cutting-edge AI technology, providing unprecedented image generation capabilities for creatives and developers.

### 🔥 Key Features

- **High-Quality Training Data**: Fine-tuned on 500,000 high-quality images
- **Long Text Understanding**: Each image accompanied by 300+ token prompts
- **Intelligent Prompt Optimization**: Built-in prompt refining with Complex Human Instruction for enhanced output quality
- **Continuous Updates**: Our team constantly optimizes the model to stay ahead of the curve



## 🚀 How to Use 
1. Install the required packages following the instructions in the original LlamaGen repository.
2. Download our pre-trained model from [HuggingFace Link](https://huggingface.co/Owen777/LongPrompt-LLamaGen/blob/main/0135000_model_only.pt), the model size is about 3.11G.
And install&download Language models for text-conditional image generation:
```
pip install ftfy
pip install transformers
pip install accelerate
pip install sentencepiece
pip install pandas
pip install bs4
```
Download flan-t5-xl models from [flan-t5-xl](https://huggingface.co/google/flan-t5-xl) and put into the folder of `./pretrained_models/t5-ckpt/`
Download vq-ds16-t2i models from [vq-ds16-t2i](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt) and put into the folder of `./pretrained_models/vq-ckpt/`

3. Modify `sample_t2i.py` to specify the paths of the pre-trained model, t5-ckpt, and vq-ckpt.
4. Use the model to generate images by following the example code provided in the repository.
5. For Complex Human Instruction, please install [Ollama](https://ollama.com), our Complex Human Instruction using ollama as backend, and automatically load&using model.

## ⚠️ Important Note

Please be aware that LongPrompt-LLamaGen is an ongoing development project. The model is continuously being trained and improved. We kindly ask for your patience as we work on refining the model.

A concise technical report detailing our methodology and findings will be released soon. Stay tuned for updates!

Thank you for your interest and support in this project.



### 🙏 Acknowledgements

We would like to express our gratitude to the LlamaGen team for their groundbreaking work: [Autoregressive Model Beats Diffusion: 🦙 Llama for Scalable Image Generation](https://github.com/FoundationVision/LlamaGen?tab=readme-ov-file). Their research has been instrumental in advancing the field of image generation using autoregressive models.



