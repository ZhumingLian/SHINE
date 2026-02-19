import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)
import torch
import json
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from diffusers import FluxFillPipeline
from models.adapter.pipeline import InstantCharacterFluxPipeline
from models.SHINE_transformer_flux import shine_flux_transformer_2d_model_forward
from optimum.quanto import quantize, freeze, qint8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="black-forest-labs/FLUX.1-dev", help="base model")
    parser.add_argument("--fill_model_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev", help="fill model")
    parser.add_argument("--ip_adapter_path", type=str, default="ckpts/adapter_ckpts/instantcharacter_ip-adapter.bin", help="adapter file path")
    parser.add_argument("--image_encoder_path", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--image_encoder_2_path", type=str, default="facebook/dinov2-giant")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_model_cpu_offload", type=bool, default=True)
    parser.add_argument("--dataset_dir", type=str, default="datasets/Shine-DreamEditBench")
    parser.add_argument("--output_dir", type=str, default="outputs_dreameditbench/test_adapter")
    parser.add_argument("--subject_scale", type=float, default=1.2, help="the strength of generating the reference object")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--sampling_start", type=int, default=5, help="denoising start")
    
    parser.add_argument("--msa_optim_start", type=int, default=0, help="the start step of applying MSA")
    parser.add_argument("--msa_optim_end", type=int, default=2, help="the end step of applying MSA")
    parser.add_argument("--msa_iter", type=int, default=10, help="the times of applying MSA in one step")
    parser.add_argument("--msa_scale_list", type=int, nargs=3, default=[5000, 7500, 10000], help="the strength of applying MSA")

    parser.add_argument("--dsg_start", type=int, default=0, help="the start step of applying DSG")
    parser.add_argument("--dsg_scale", type=float, default=0.5, help="the strength of applying DSG")
    parser.add_argument("--dsg_blur_sigma", type=float, default=10.0, help="the strength of bluring attention map")

    parser.add_argument("--abb_steps", type=int, default=13, help="the end step of replace the background with the original background")
    parser.add_argument("--abb_bin_threshold", type=float, default=0.2, help="the threshold of binarizing the attention map")
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


def load_editing_model(args):
    pipe = InstantCharacterFluxPipeline.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    quantize(pipe.transformer, qint8)
    freeze(pipe.transformer)
    pipe.init_adapter(
        image_encoder_path=args.image_encoder_path,
        image_encoder_2_path=args.image_encoder_2_path,
        subject_ipadapter_cfg=dict(subject_ip_adapter_path=args.ip_adapter_path, nb_token=1024),
        device=args.device,
    )
    pipe.transformer.forward = shine_flux_transformer_2d_model_forward.__get__(pipe.transformer, pipe.transformer.__class__)
    return pipe


def main(args):
    fill_pipe = FluxFillPipeline.from_pretrained(args.fill_model_path, torch_dtype=torch.bfloat16)
    quantize(fill_pipe.transformer, qint8)
    freeze(fill_pipe.transformer)
    pipe = load_editing_model(args)
    if args.enable_model_cpu_offload:
        fill_pipe.enable_model_cpu_offload()
        pipe.enable_model_cpu_offload()
    else:
        fill_pipe.to(args.device)
        pipe.to(args.device)
    class_list = ['cat', 'duck_toy', 'grey_sloth_plushie', 'monster_toy', 'rc_car', 'robot_toy', 'vase', 'wolf_plushie']
    for class_name in os.listdir(args.dataset_dir):
        if class_name not in class_list:
            continue
        class_path = os.path.join(args.dataset_dir, class_name)
        bg_path = os.path.join(class_path, "bg")
        content_file = os.path.join(bg_path, "content.json")
        if not os.path.exists(content_file):
            print(f"content.json not found: {content_file}")
            continue
        fg_img_path = os.path.join(class_path, "fg/03.jpg")
        fg_mask_path = os.path.join(class_path, "fg/03.png")
        fg_image = Image.open(fg_img_path).convert("RGB")
        fg_mask = Image.open(fg_mask_path).convert("L")
        transform = T.ToTensor()
        image_tensor = transform(fg_image)
        mask_tensor = transform(fg_mask) 
        mask_tensor = (mask_tensor > 0.5).float()
        white_bg = torch.ones_like(image_tensor)
        result = image_tensor * mask_tensor + white_bg * (1 - mask_tensor)
        to_pil = T.ToPILImage()
        ref_image = to_pil(result)
        output_dir = os.path.join(args.output_dir, f"{class_name}")
        os.makedirs(output_dir, exist_ok=True)
        with open(content_file, 'r') as f:
            data = json.load(f)
            for item in data:
                image_name = item['image']
                bbox = item['bbox']
                # prompt = item['source_prompt']
                fill_prompt = item['fill_prompt']
                target_prompt = item['target_prompt']
                instance_prompt = class_name.replace("_", " ")
                prompt = target_prompt.replace("<sks>", instance_prompt)
                fill_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + f"_fill.png")
                result_image_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + f".png")
                if os.path.exists(result_image_path):
                    print(f"file already exists: {result_image_path}")
                    continue
                bg_img_path = os.path.join(bg_path, image_name)
                bg_image = Image.open(bg_img_path)
                bg_image = bg_image.resize((768, 768), Image.Resampling.LANCZOS)
                bg_w = bg_image.size[0]
                bg_h = bg_image.size[1]
                def nearest_multiple_of_16(x):
                    lower = (x // 16) * 16
                    upper = ((x + 15) // 16) * 16
                    return lower if abs(x - lower) <= abs(x - upper) else upper
                w_new = nearest_multiple_of_16(bg_w)
                h_new = nearest_multiple_of_16(bg_h)
                # bg_image_resized = bg_image.resize((w_new, h_new), Image.Resampling.LANCZOS)
                x_scale = w_new / bg_w
                y_scale = h_new / bg_h
                resized_box = (int(x_scale * bbox[0]),
                                int(y_scale * bbox[1]),
                                int(x_scale * bbox[2]),
                                int(y_scale * bbox[3]))
                # generate user mask accroding to the bbox
                user_mask = Image.new("L", (bg_image.size[0], bg_image.size[1]), 0)
                draw = ImageDraw.Draw(user_mask)
                draw.rectangle(resized_box, fill=255)
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
                # use the original background
                region_area = fill_image.crop(resized_box)
                fill_image_new = bg_image.copy()
                fill_image_new.paste(region_area, (resized_box[0], resized_box[1]))
                fix_all_seed(args.seed)
                result_image = pipe(
                    seed=args.seed,
                    prompt=prompt, 
                    instance_prompt=instance_prompt,
                    subject_image=ref_image,
                    image=fill_image_new,
                    mask_box=resized_box,
                    width=fill_image_new.size[0],
                    height=fill_image_new.size[1],
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
                    num_inference_steps=args.num_inference_steps,
                    sampling_start=args.sampling_start,
                    guidance_scale=3.5,
                    subject_scale=args.subject_scale,
                    generator=torch.Generator(args.device).manual_seed(args.seed),
                ).images[0]
                result_image.save(result_image_path)
                print(f"Finish editing, the result image was saved to {result_image_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)