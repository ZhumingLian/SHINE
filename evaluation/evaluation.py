import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import torch
import numpy as np
import lpips
import time
import json
import argparse
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageDraw
from torchvision import transforms
import ImageReward as RM
from tools.clip_score import SimilarityCalculator
from tools.dinov2_score import init_dinov2_model, calculate_dinov2_score
from tools.IRF_score import init_IRF_model, calculate_IRF_score
from dreamsim import dreamsim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--dinov2_model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--vit_model", type=str, default="ViT-H-14", help="used in IRF model")
    parser.add_argument("--IRF_model_path", type=str, default="ckpts/IRF_ckpts/arcface all vith 18 last and middle first 3 280 all 3 290 first 1 overlap last 6 middle 6 first 3 dropout.pth")
    parser.add_argument("--dream_sim_model_path", type=str, default="ckpts/dream_sim_ckpts", help="strore dream sim model checkpoints")
    parser.add_argument("--image_reward_model", type=str, default="ImageReward-v1.0")
    parser.add_argument("--evaluation_file", type=str, default="examples/eval_image_metrics_config.json", help="Path to JSON file containing evaluation config (background/ref/ref_mask/result images path + bbox for image metrics evaluation)")
    args = parser.parse_args()
    return args
    

def main(args):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    clip_model = SimilarityCalculator(args.clip_model, args.device)
    dinov2_model, dinov2_processor = init_dinov2_model(args.dinov2_model, args.device)
    IRF_model, IRF_preprocess = init_IRF_model(args.IRF_model_path, args.vit_model, args.device)
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=args.device, cache_dir=args.dream_sim_model_path)
    lpips_model = lpips.LPIPS(net='alex')  # alex / vgg
    image_reward_model = RM.load(args.image_reward_model).to(args.device)
    with open(args.evaluation_file, 'r') as f:
        data = json.load(f)
        for item in data:
            bg_img = Image.open(item['bg_img'])
            ref_img = Image.open(item['ref_img'])
            ref_mask = Image.open(item['ref_mask']).convert("L")
            result_img = Image.open(item['result_img'])
            bbox = item['bbox']
            prompt = item['prompt']
            # result fg
            result_fg = result_img.crop(bbox)
            # result bg
            result_bg = result_img.copy()
            draw = ImageDraw.Draw(result_bg)
            draw.rectangle(bbox, fill=(0, 0, 0))
            # original bg
            bg_masked_img = bg_img.copy()
            draw = ImageDraw.Draw(bg_masked_img)
            draw.rectangle(bbox, fill=(0, 0, 0))
            # original fg
            image_np = np.array(ref_img)
            mask_np = np.array(ref_mask) / 255.0
            mask_np = np.expand_dims(mask_np, axis=2)  # (H, W, 1)
            masked_np = (image_np * mask_np).astype(np.uint8)
            masked_fg = Image.fromarray(masked_np)

            with torch.no_grad():
                # fg comparison
                # masked_fg.save("masked_fg.png")
                # result_fg.save("result_fg.png")
                print(f"Evaluating {item['result_img']}:")
                # clip
                clip_similarity = clip_model.calculate_similarity(masked_fg, result_fg)
                print(f"Clip score: {clip_similarity:.4f}")
                
                # dinov2
                dinov2_similarity = calculate_dinov2_score(dinov2_model, dinov2_processor, masked_fg, result_fg, args.device)
                print(f"Dinov2 score: {dinov2_similarity:.4f}")
                
                # IRF
                IRF_similarity = calculate_IRF_score(IRF_model, IRF_preprocess, masked_fg, result_fg, args.device)
                print(f"IRF score: {IRF_similarity:.4f}")
                
                # dream sim
                distance = dreamsim_model(dreamsim_preprocess(masked_fg).to(args.device), dreamsim_preprocess(result_fg).to(args.device))
                print(f"dream sim score: {distance.item():.4f}")

                # bg comparison
                # bg_masked_img.save("bg_masked_img.png")
                # result_bg.save("result_bg.png")
                # load image and transform to tensor（LPIPS requires [-1, 1] range，3 channels）
                bg_masked_img_tensor = transform(bg_masked_img).unsqueeze(0)  # (1,3,H,W)
                result_bg_tensor = transform(result_bg).unsqueeze(0)
                # ---- LPIPS ----
                lpips_dist = lpips_model(bg_masked_img_tensor, result_bg_tensor).item()
                print(f"LPIPS: {lpips_dist:.4f}")
                # ---- SSIM ----
                bg_masked_img_np = np.array(bg_masked_img)
                result_bg_np = np.array(result_bg)
                ssim_val = compare_ssim(bg_masked_img_np, result_bg_np, channel_axis=-1, data_range=255)
                print(f"SSIM: {ssim_val:.4f}")

                # general comparison
                # result_img.save("result_img.png")
                # image reward
                image_reward_score = image_reward_model.score(prompt, result_img)
                print(f"image reward:{image_reward_score:.4f}")
                print(f"Finish evaluating {item['bg_img']}.")
                    

if __name__ == "__main__":
    args = parse_args()
    main(args)