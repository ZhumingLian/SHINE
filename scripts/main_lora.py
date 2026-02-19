import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import torch
from diffusers import FluxFillPipeline
from models.lora.SHINE_pipeline_flux import SHINE_FluxPipeline
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
from models.lora.SHINE_attn_processor import FluxAttnProcessor
from models.SHINE_transformer_flux import shine_flux_transformer_2d_model_forward
from optimum.quanto import quantize, freeze, qint8
from PIL import Image, ImageDraw
import torchvision.transforms as T
import json, argparse, random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="black-forest-labs/FLUX.1-dev", help="base model")
    parser.add_argument("--fill_model_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev", help="fill model")
    parser.add_argument("--vlm_model_path", type=str, default="Salesforce/xgen-mm-phi3-mini-instruct-r-v1", help="used to generate prompt")
    parser.add_argument("--input_path", type=str, default="examples/dog/bg/content.json", help="input json file including bg_img_path, fg_img_path, fg_mask_path, bbox, instance_prompt, fill_image_path and save_image_path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_vlm", type=bool, default=False, help="whether to use vlm model to generate prompt")
    parser.add_argument("--enable_model_cpu_offload", type=bool, default=True)
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


def load_vlm_model(args):
    orig_to = torch.nn.Module.to

    def safe_to(self, *args, **kwargs):
        try:
            return orig_to(self, *args, **kwargs)
        except NotImplementedError:
            return self.to_empty(*args, **kwargs)

    # replace
    torch.nn.Module.to = safe_to

    # load models
    model = AutoModelForVision2Seq.from_pretrained(args.vlm_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # restore
    torch.nn.Module.to = orig_to
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_model_path, trust_remote_code=True, use_fast=False, legacy=False, torch_dtype=torch.bfloat16)
    image_processor = AutoImageProcessor.from_pretrained(args.vlm_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = model.update_special_tokens(tokenizer)
    model = model.to(args.device)
    return image_processor, tokenizer, model


def get_vlm_prompt(args, model, image_processor, tokenizer, image):
    # define the prompt template
    def apply_prompt_template(prompt):
        s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
            )
        return s 
    class EosListStoppingCriteria(StoppingCriteria):
        def __init__(self, eos_sequence = [32007]):
            self.eos_sequence = eos_sequence

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
            return self.eos_sequence in last_ids      

    query = 'describe this image with only one paragraph'
    inputs = image_processor([image], return_tensors="pt", image_aspect_ratio='anyres').to(torch.bfloat16)
    prompt = apply_prompt_template(query)
    language_inputs = tokenizer([prompt], return_tensors="pt")
    inputs.update(language_inputs)
    inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}
    generated_text = model.generate(**inputs, image_size=[image.size],
                                    pad_token_id=tokenizer.pad_token_id,
                                    do_sample=False, max_new_tokens=512, top_p=None, num_beams=1,
                                    stopping_criteria = [EosListStoppingCriteria()],
                                    )
    prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
    return prediction


def load_editing_model(args, lora_weight_path):
    edit_pipe = SHINE_FluxPipeline.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    edit_pipe.transformer.forward = shine_flux_transformer_2d_model_forward.__get__(edit_pipe.transformer, edit_pipe.transformer.__class__)
    shine_processor = FluxAttnProcessor()
    edit_pipe.transformer.set_attn_processor(shine_processor)
    edit_pipe.load_lora_weights(lora_weight_path)
    quantize(edit_pipe.transformer, weights=qint8)
    freeze(edit_pipe.transformer)
    return edit_pipe


def nearest_multiple_of_16(x):
    lower = (x // 16) * 16
    upper = ((x + 15) // 16) * 16
    return lower if abs(x - lower) <= abs(x - upper) else upper


def resize_shortest_edge_to_768(img):
    w, h = img.size
    scale = 768 / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img_resized


def main(args):
    with open(args.input_path, 'r') as f:
        data = json.load(f)
    bbox = data[0]['bbox']
    instance_prompt = data[0]['instance_prompt']
    fill_prompt = data[0]['fill_prompt']
    source_prompt = data[0]['source_prompt']
    target_prompt = data[0]['target_prompt']
    bg_img_path = data[0]['bg_img_path']
    lora_weight_path = data[0]['lora_weight_path']
    lora_prompt = data[0]['lora_prompt']
    if args.use_vlm:
        vlm_image_processor, vlm_tokenizer, vlm_model = load_vlm_model(args)
    fill_pipe = FluxFillPipeline.from_pretrained(args.fill_model_path, torch_dtype=torch.bfloat16)
    quantize(fill_pipe.transformer, qint8)
    freeze(fill_pipe.transformer)
    editing_pipe = load_editing_model(args, lora_weight_path)
    if args.enable_model_cpu_offload:
        fill_pipe.enable_model_cpu_offload()
        editing_pipe.enable_model_cpu_offload()
    else:
        fill_pipe.to(args.device)
        editing_pipe.to(args.device)

    bg_image = Image.open(bg_img_path)
    w = bg_image.size[0]
    h = bg_image.size[1]
    # resize so that the height and width are divisible by 16
    new_w = nearest_multiple_of_16(bg_image.size[0])
    new_h = nearest_multiple_of_16(bg_image.size[1])
    bg_image = bg_image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    bbox = [int(bbox[0]*new_w/w), int(bbox[1]*new_h/h), int(bbox[2]*new_w/w), int(bbox[3]*new_h/h)]

    # generate user mask accroding to the bbox
    user_mask = Image.new("L", (new_w, new_h), 0)
    draw = ImageDraw.Draw(user_mask)
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=255)

    fg_image = Image.open(data[0]['fg_img_path']).convert("RGB")
    fg_mask = Image.open(data[0]['fg_mask_path']).convert("L")
    # reference image (object image) should use the white background according to the code of InstantCharacter
    transform = T.ToTensor()
    image_tensor = transform(fg_image)
    mask_tensor = transform(fg_mask)
    mask_tensor = (mask_tensor > 0.5).float()
    white_bg = torch.ones_like(image_tensor)
    result = image_tensor * mask_tensor + white_bg * (1 - mask_tensor)
    to_pil = T.ToPILImage()
    ref_image = to_pil(result)

    with torch.inference_mode():
        if args.use_vlm:
            fill_prompt = get_vlm_prompt(args, vlm_model, vlm_image_processor, vlm_tokenizer, ref_image)
            print(fill_prompt)
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
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]
        # fill_image.save(data[0]['fill_image_path'])
    # use the original background
    region_area = fill_image.crop(bbox)
    fill_image_new = bg_image.copy()
    fill_image_new.paste(region_area, (bbox[0], bbox[1]))
    
    with torch.no_grad():
         # please make sure prompt contains instance_prompt
        if args.use_vlm:
            source_prompt = get_vlm_prompt(args, vlm_model, vlm_image_processor, vlm_tokenizer, fill_image_new)
            print(source_prompt)
            target_prompt = source_prompt
            replace_prompt = instance_prompt.split("_")
            for p in replace_prompt:
                target_prompt = target_prompt.replace(p, lora_prompt)
    fix_all_seed(args.seed)
    result_image = editing_pipe(
        prompt=target_prompt, 
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        instance_prompt=lora_prompt,
        num_inference_steps=args.num_inference_steps,
        sampling_start=args.sampling_start,
        guidance_scale=3.5,
        image=fill_image_new,
        mask_box=bbox,
        width=bg_image.size[0],
        height=bg_image.size[1],
        seed=args.seed,
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
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    result_image.save(data[0]['save_image_path'])
    print(f"Result image has been saved to {data[0]['save_image_path']}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

