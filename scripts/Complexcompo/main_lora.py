import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)
from diffusers import FluxFillPipeline
from models.lora.SHINE_pipeline_flux import SHINE_FluxPipeline
import torch
import json
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from models.lora.SHINE_attn_processor import FluxAttnProcessor
from models.SHINE_transformer_flux import shine_flux_transformer_2d_model_forward
from optimum.quanto import quantize, freeze, qint8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_prompt", type=str, default="<sks>", help="lora token")
    parser.add_argument("--fill_model_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev", help="fill model")
    parser.add_argument("--edit_model_path", type=str, default="black-forest-labs/FLUX.1-dev", help="base model")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_model_cpu_offload", type=bool, default=True)
    parser.add_argument("--dataset_dir", type=str, default="datasets/ComplexCompo")
    parser.add_argument("--output_dir", type=str, default="outputs_complexcompo/test_lora")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--sampling_start", type=int, default=6, help="denoising start")
    
    parser.add_argument("--msa_optim_start", type=int, default=0, help="the start step of applying MSA")
    parser.add_argument("--msa_optim_end", type=int, default=1, help="the end step of applying MSA")
    parser.add_argument("--msa_iter", type=int, default=2, help="the times of applying MSA in one step")
    parser.add_argument("--msa_scale_list", type=int, nargs=2, default=[500, 3000], help="the strength of applying MSA")

    parser.add_argument("--dsg_start", type=int, default=0, help="the start step of applying DSG")
    parser.add_argument("--dsg_scale", type=float, default=0.7, help="the strength of applying DSG")
    parser.add_argument("--dsg_blur_sigma", type=float, default=10.0, help="the strength of bluring attention map")

    parser.add_argument("--abb_steps", type=int, default=12, help="the end step of replace the background with the original background")
    parser.add_argument("--abb_bin_threshold", type=float, default=0.4, help="the threshold of binarizing the attention map")
    parser.add_argument("--abb_dilation_kernel_size", type=int, default=3, help="used to dilate the binarized attention map")

    args = parser.parse_args()
    return args


def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def main(args):
    # 1. load fill model and editing model
    fill_pipe = FluxFillPipeline.from_pretrained(args.fill_model_path, torch_dtype=torch.bfloat16)
    quantize(fill_pipe.transformer, qint8)
    freeze(fill_pipe.transformer)
    if args.enable_model_cpu_offload:
        fill_pipe.enable_model_cpu_offload()
    else:
        fill_pipe.to(args.device)
    edit_pipe = SHINE_FluxPipeline.from_pretrained(args.edit_model_path, transformer=None, torch_dtype=torch.bfloat16)
    # customized processor
    shine_processor = FluxAttnProcessor()

    for class_name in os.listdir(args.dataset_dir):
        # 2. load lora weight
        adapter_file_path = f'./ckpts/LoRA_ckpts/{class_name}/pytorch_lora_weights.safetensors'
        if not os.path.isfile(adapter_file_path):
            continue
        edit_pipe.transformer = FluxTransformer2DModel.from_pretrained(args.edit_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        edit_pipe.transformer.forward = shine_flux_transformer_2d_model_forward.__get__(edit_pipe.transformer, edit_pipe.transformer.__class__)
        edit_pipe.transformer.set_attn_processor(shine_processor)
        edit_pipe.load_lora_weights(adapter_file_path)
        quantize(edit_pipe.transformer, weights=qint8)
        freeze(edit_pipe.transformer)
        if args.enable_model_cpu_offload:
            edit_pipe.enable_model_cpu_offload()
        else:
            edit_pipe.to(args.device)
        # 3. preprocess
        class_path = os.path.join(args.dataset_dir, class_name)
        bg_path = os.path.join(class_path, "bg")
        content_file = os.path.join(bg_path, "content.json")
        if not os.path.isfile(content_file):
            print(f"content.json not found: {content_file}")
            continue
        output_dir = os.path.join(args.output_dir, f"{class_name}")
        os.makedirs(output_dir, exist_ok=True)
        with open(content_file, 'r') as f:
            data = json.load(f)
            for item in data:
                image_name = item['resized_img']
                bbox = item['bbox']
                fill_prompt = item['fill_prompt']
                source_prompt = item['source_prompt']
                target_prompt = item['target_prompt']
                fill_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + f"_fill.png")
                result_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + f".png")
                if os.path.exists(result_image_path):
                    print(f"file already exists: {result_image_path}")
                    continue
                bg_img_path = os.path.join(bg_path, image_name)
                bg_image = Image.open(bg_img_path)
                # generate user mask accroding to the bbox
                user_mask = Image.new("L", (bg_image.size[0], bg_image.size[1]), 0)
                draw = ImageDraw.Draw(user_mask)
                draw.rectangle(bbox, fill=255)
                # 4. fill
                if not os.path.exists(fill_image_path):
                    with torch.inference_mode():
                        fix_all_seed(args.seed)
                        fill_image = fill_pipe(
                            prompt=fill_prompt,
                            image=bg_image,
                            mask_image=user_mask,
                            height=bg_image.size[1],
                            width=bg_image.size[0],
                            guidance_scale=30,
                            num_inference_steps=30,
                            max_sequence_length=512,
                            generator=torch.Generator(args.device).manual_seed(args.seed),
                        ).images[0]
                    # fill_image.save(fill_image_path)
                    print(f"Finish filling, the fill image was saved to {fill_image_path}")
                else:
                    fill_image = Image.open(fill_image_path)
                # 5. use the original background
                region_area = fill_image.crop(bbox)
                fill_image_new = bg_image.copy()
                fill_image_new.paste(region_area, (bbox[0], bbox[1]))

                # 6. edit
                fix_all_seed(args.seed)
                result_image = edit_pipe(
                    seed=args.seed,
                    image=fill_image_new,
                    mask_box=bbox,
                    prompt=target_prompt,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    instance_prompt=args.instance_prompt,
                    height=fill_image_new.size[1],
                    width=fill_image_new.size[0],
                    num_inference_steps=args.num_inference_steps,
                    sampling_start=args.sampling_start,
                    guidance_scale=3.5,
                    msa_iter=args.msa_iter,
                    msa_optim_start=args.msa_optim_start,
                    msa_optim_end=args.msa_optim_end,
                    msa_scale_list=args.msa_scale_list,
                    dsg_scale=args.dsg_scale,
                    dsg_start=args.dsg_start,
                    dsg_blur_sigma=args.dsg_blur_sigma,
                    abb_steps=args.abb_steps,
                    abb_bin_threshold=args.abb_bin_threshold,
                    abb_dilation_kernel_size=args.abb_dilation_kernel_size,
                    generator=torch.Generator(args.device).manual_seed(args.seed),
                ).images[0]
                result_image.save(result_image_path)
                print(f"Finish editing, the result image was saved to {result_image_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
