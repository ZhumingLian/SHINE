<div align="center">
<h1>Does FLUX Already Know How to Perform Physically Plausible Image Composition?</h1>
Shilin Lu <sup>1*</sup> · 
Zhuming Lian <sup>1*</sup> ·
Zihan Zhou <sup>1</sup> ·
Shaocong Zhang <sup>1</sup> ·
Chen Zhao <sup>1</sup> ·
Adams Wai-Kin Kong <sup>1</sup>
<sup>1</sup>Nanyang Technological University

<!-- [Paper](https://arxiv.org/abs/2509.21278) | [Project Page](https://cjlxzh32.github.io) -->
<!-- <a href='https://cjlxzh32.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> -->
<a href='https://arxiv.org/abs/2509.21278'><img src='https://img.shields.io/badge/Paper-Page-red'></a>
</div>

<img src='assets/teaser.jpg'>


## ⬇️ Download

### 1. IP-Adapter Checkpoints

Please refer to [InstantCharacter](https://github.com/Tencent-Hunyuan/InstantCharacter) to download the IP-Adapter checkpoints and place them in `./ckpts/adapter_ckpts` directory. 
You can also download them using the following command:

```
hf download Tencent/InstantCharacter --local-dir ./ckpts/adapter_ckpts
```

### 2. LoRA Checkpoints

The LoRA weights used in our experiments are hosted on Hugging Face [Shine_lora_ckpts](https://huggingface.co/cjlxzh32/Shine_lora_ckpts). Please download with:

```
hf download cjlxzh32/Shine_lora_ckpts --local-dir ./ckpts/LoRA_ckpts
```

### 3. IRF Evaluation Checkpoints

Please refer to  
[1st-Place-Solution-in-Google-Universal-Image-Embedding](https://github.com/ShihaoShao-GH/1st-Place-Solution-in-Google-Universal-Image-Embedding?tab=readme-ov-file)

and place the downloaded files in `ckpts/IRF_ckpts/` directory.

### 4. Datasets

The datasets used in our experiments are hosted on Hugging Face:

- **[Shine-DreamEditBench](https://huggingface.co/datasets/cjlxzh32/Shine-DreamEditBench)** — a reformatted version of DreamEditBench  
- **[ComplexCompo](https://huggingface.co/datasets/cjlxzh32/ComplexCompo)** — our benchmark dataset for evaluating physically plausible image composition

Please download with:

```
hf download --repo-type dataset cjlxzh32/Shine-DreamEditBench --local-dir ./datasets/Shine-DreamEditBench
hf download --repo-type dataset cjlxzh32/ComplexCompo --local-dir ./datasets/ComplexCompo
```

## 📁 Repository Structure

```
.
├── assets/
├── ckpts/
│   ├── adapter_ckpts/                   # Pretrained IP-Adapter weights
│   │   └── instantcharacter_ip-adapter.bin
│   ├── dream_sim_ckpts/                 # dream_sim metric cache directory
│   ├── IRF_ckpts/                       # IRF metric checkpoints
│       └── arcface all vith 18 last and middle first 3 280 all 3 290 first 1 overlap last 6 middle 6 first 3 dropout.pth
│   └── LoRA_ckpts/
│       └── instance/
│           └── pytorch_lora_weights.safetensors
├── datasets/                            # Benchmark datasets
│   ├── DreamEditBench/
│   └── ComplexCompo/
│       └── instance/
│           ├── bg/
│           │   ├── 0_512_rect.png       # resized image from 0.jpg with a short side of 512
│           │   ├── 0_512_square.png     # cropped image from 0_512_rect.png
│           │   ├── 0_768_rect.png       # resized image from 0.jpg with a short side of 768
│           │   ├── 0_768_square.png     # cropped image from 0_768_rect.png
│           │   ├── 0_w_mask.png         # 0_768_rect.png with bbox
│           │   ├── 0.jpg                # original background image
│           │   ├── content_512.json     # contains prompt and bbox information, match with 0_768_square.png
│           │   ├── content_768.json     # contains prompt and bbox information, match with 0_512_square.png
│           │   └── content.json         # contains prompt and bbox information, match with 0_768_rect.png
│           └── fg/
│               ├── 00.jpg               # reference image
│               └── 00.png               # mask image
├── evaluation/                          # evaluation scripts
│   ├── evaluation_complexcompo.py
│   ├── evaluation_dreameditbench.py
│   └── evaluation.py                    # single image evaluation
├── examples/                            # Example inputs
│   ├── instance/
│   │   ├── bg/
│   │   │   ├── bg.jpg                   # background image
│   │   │   └── content.json             # contains bbox information
│   │   └── fg/
│   │       ├── 00.jpg                   # reference image
│   │       └── 00.png                   # mask image
│   └── eval_image_metrics_config.json   # evaluation content configuration
├── models/                              # Model framework
│   ├── adapter/
│   │   ├── attn_processor.py
│   │   ├── norm_layer.py
│   │   ├── pipeline.py
│   │   ├── resampler.py
│   │   └── utils.py
│   ├── lora/
│   │   ├── SHINE_attn_processor.py
│   │   └── SHINE_pipeline_flux.py
│   └── SHINE_transformer_flux.py
├── scripts/                             # Experiment scripts
│   ├── Complexcompo                     # running on ComplexCompo dataset
│   │   ├── main_adapter_complexCompo.py
│   │   └── main_lora_complexCompo.py
│   ├── Dreambooth                       # running on DreamEditBench dataset                  
│   │   ├── main_adapter_dreambooth.py
│   │   └── main_lora_dreambooth.py          
│   ├── main_adapter.py                  # IP-Adapter inference script
│   └── main_lora.py                     # LoRA inference script
├── tools/                               # evaluation tools
│   │── cladapter_score.py
│   ├── dinov2_score.py
│   └── first_score.py
├── .gitignore
├── README.md
└── requirements.txt
```


## 🚀 Quick Start

### Requirements

```
Python >= 3.10, PyTorch >= 2.0, CUDA >= 11.8
```

### Environment Setup

```
conda create -n shine python=3.13 -y
conda activate shine
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install image-reward
pip install -r requirements.txt
```

### Inference

For single image inference, the commands are as follows:
```
# IP-Adapter
CUDA_VISIBLE_DEVICES=0 python scripts/main_adapter.py \
--input_path examples/cat/bg/content.json \
--enable_model_cpu_offload True

# LoRA
CUDA_VISIBLE_DEVICES=0 python scripts/main_lora.py \
--input_path examples/cat/bg/content.json \
--enable_model_cpu_offload True
```

The result image will be saved to:

```
examples/cat/result.png
```

For running Shine-DreamEditBench and ComplexCompo datasets, the commands are as follows:
```
# IP-Adapter
CUDA_VISIBLE_DEVICES=0 python scripts/Dreambooth/main_adapter.py \
--dataset_dir datasets/Shine-DreamEditBench \
--output_dir outputs_dreameditbench/test_adapter \
--enable_model_cpu_offload True

CUDA_VISIBLE_DEVICES=0 python scripts/Complexcompo/main_adapter.py \
--dataset_dir datasets/ComplexCompo \
--output_dir outputs_complexcompo/test_adapter \
--enable_model_cpu_offload True

# LoRA
CUDA_VISIBLE_DEVICES=0 python scripts/Dreambooth/main_lora.py \
--dataset_dir datasets/Shine-DreamEditBench \
--output_dir outputs_dreameditbench/test_lora \
--enable_model_cpu_offload True

CUDA_VISIBLE_DEVICES=0 python scripts/Complexcompo/main_lora.py \
--dataset_dir datasets/ComplexCompo \
--output_dir outputs_complexcompo/test_lora \
--enable_model_cpu_offload True
```

### Evaluation

For single example evaluation, please use the command:
```
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluation.py \
--evaluation_file examples/eval_image_metrics_config.json
```

For Shine-DreamEditBench and ComplexCompo datasets evaluation, please use the following commands:
```
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluation_dreameditbench.py \
--dataset_dir datasets/Shine-DreamEditBench \
--results_dir outputs_dreameditbench

CUDA_VISIBLE_DEVICES=0 python evaluation/evaluation_complexcompo.py \
--dataset_dir datasets/ComplexCompo \
--results_dir outputs_complexcompo
```

## 🙏 Acknowledgements

This codebase is built upon:

[HuggingFace](https://huggingface.co)<br>
[Diffusers](https://github.com/huggingface/diffusers)<br>
[InstantCharacter](https://github.com/Tencent-Hunyuan/InstantCharacter)

## 📜 Citation

If you find this work useful, please cite:

```
@article{lu2025does,
  title={Does flux already know how to perform physically plausible image composition?},
  author={Lu, Shilin and Lian, Zhuming and Zhou, Zihan and Zhang, Shaocong and Zhao, Chen and Kong, Adams Wai-Kin},
  journal={arXiv preprint arXiv:2509.21278},
  year={2025}
}
```