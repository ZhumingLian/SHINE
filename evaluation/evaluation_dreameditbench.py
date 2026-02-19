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


# python evaluation/evaluation_dreameditbench.py
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--dinov2_model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--vit_model", type=str, default="ViT-H-14", help="used in IRF model")
    parser.add_argument("--IRF_model_path", type=str, default="ckpts/IRF_ckpts/arcface all vith 18 last and middle first 3 280 all 3 290 first 1 overlap last 6 middle 6 first 3 dropout.pth")
    parser.add_argument("--dream_sim_model_path", type=str, default="ckpts/dream_sim_ckpts", help="strore dream sim model checkpoints")
    parser.add_argument("--image_reward_model", type=str, default="ImageReward-v1.0")
    
    parser.add_argument("--dataset_dir", type=str, default="datasets/Shine-DreamEditBench")
    parser.add_argument("--results_dir", type=str, default="outputs_dreameditbench")
    args = parser.parse_args()
    return args
    

def main(args):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    clip_model = SimilarityCalculator(args.clip_model, args.device)
    dinov2_model, dinov2_processor = init_dinov2_model(args.dinov2_model, args.device)
    IRF_model, IRF_preprocess = init_IRF_model(args.IRF_model_path, args.vit_model, args.device)
    model, preprocess = dreamsim(pretrained=True, device=args.device, cache_dir=args.dream_sim_model_path)
    image_reward_model = RM.load(args.image_reward_model).to(args.device)
    lpips_model = lpips.LPIPS(net='alex')  # alex / vgg
    clip_similarity_list = []
    dinov2_similarity_list = []
    IRF_similarity_list = []
    dream_sim_list = []
    lpips_list = []
    ssim_list = []
    image_reward_list = []
    for benchmark in os.listdir(args.results_dir):
        if benchmark.endswith(".txt"):
            continue
        benchmark_dir = os.path.join(args.results_dir, benchmark)
        start_time = time.time()
        for class_name in os.listdir(benchmark_dir):
            class_path = os.path.join(benchmark_dir, class_name)
            ref_class_path = os.path.join(args.dataset_dir, class_name)
            bg_path = os.path.join(ref_class_path, "bg")
            content_file = os.path.join(bg_path, "content.json")
            fg_path = os.path.join(ref_class_path, "fg/03.jpg")
            fg_mask_path = os.path.join(ref_class_path, "fg/03.png")
            if not os.path.exists(content_file):
                continue
            # original_fg
            fg = Image.open(fg_path)
            fg_mask = Image.open(fg_mask_path).convert("L")
            with open(content_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    image_name = item['image']
                    # bbox = item['bbox']
                    bbox = [x * 512 // 768 for x in item['bbox']]
                    source_prompt = item['source_prompt']
                    index = os.path.splitext(image_name)[0]
                    result_img_path = os.path.join(class_path, f"{index}.png")
                    if not os.path.exists(result_img_path):
                        continue
                    # result image
                    result_img = Image.open(result_img_path).resize((512, 512), resample=Image.Resampling.LANCZOS)
                    bg_image_path = bg_path + f"/{index}.jpg"
                    # original background image
                    bg_img = Image.open(bg_image_path).resize((512, 512), resample=Image.Resampling.LANCZOS)
                    mask = Image.new("L", bg_img.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.rectangle(bbox, fill=255)
                    # result fg
                    result_fg = result_img.crop(bbox)
                    # result bg
                    result_bg = result_img.copy()
                    draw = ImageDraw.Draw(result_bg)
                    draw.rectangle(bbox, fill=0)
                    # bg
                    bg_masked_img = bg_img.copy()
                    draw = ImageDraw.Draw(bg_masked_img)
                    draw.rectangle(bbox, fill=0)
                    # fg
                    image_np = np.array(fg)
                    mask_np = np.array(fg_mask) / 255.0
                    mask_np = np.expand_dims(mask_np, axis=2)  # (H, W, 1)
                    masked_np = (image_np * mask_np).astype(np.uint8)
                    # masked_bg
                    masked_fg = Image.fromarray(masked_np)

                    # fg comparison
                    # masked_fg.save("masked_fg.png")
                    # result_fg.save("result_fg.png")
                    print(f"Evaluating {result_img_path}:")
                    with torch.no_grad():
                        # clip
                        clip_similarity = clip_model.calculate_similarity(masked_fg, result_fg)
                        clip_similarity_list.append(clip_similarity)
                        print(f"Clip score: {clip_similarity:.4f}")
                        
                        # dinov2
                        dinov2_similarity = calculate_dinov2_score(dinov2_model, dinov2_processor, masked_fg, result_fg, args.device)
                        dinov2_similarity_list.append(dinov2_similarity)
                        print(f"Dinov2 score: {dinov2_similarity:.4f}")
                        
                        # IRF
                        IRF_similarity = calculate_IRF_score(IRF_model, IRF_preprocess, masked_fg, result_fg, args.device)
                        IRF_similarity_list.append(IRF_similarity)
                        print(f"IRF score: {IRF_similarity:.4f}")
                        
                        # dream sim
                        distance = model(preprocess(masked_fg).to(args.device), preprocess(result_fg).to(args.device))
                        dream_sim_list.append(distance.item())
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
                        lpips_list.append(lpips_dist)
                        # ---- SSIM ----
                        bg_masked_img_np = np.array(bg_masked_img)
                        result_bg_np = np.array(result_bg)
                        
                        ssim_val = compare_ssim(bg_masked_img_np, result_bg_np, channel_axis=-1, data_range=255)
                        print(f"SSIM: {ssim_val:.4f}")
                        ssim_list.append(ssim_val)

                        # general comparison
                        # result_img.save("result_img.png")

                        # image reward
                        image_reward_score = image_reward_model.score(source_prompt, result_img_path)
                        image_reward_list.append(image_reward_score)
                        print(f"image reward:{image_reward_score}")

                    with open(f"{args.results_dir}/{benchmark}.txt", "a") as f:
                        f.write(f"{result_img_path}\n")
                        f.write(f"clip:{clip_similarity:.4f}\n")
                        f.write(f"dinov2:{dinov2_similarity:.4f}\n")
                        f.write(f"IRF:{IRF_similarity:.4f}\n")
                        f.write(f"dream sim:{distance.item():.4f}\n")
                        f.write(f"lpips:{lpips_dist:.4f}\n")
                        f.write(f"ssim:{ssim_val:.4f}\n")
                        f.write(f"image reward:{image_reward_score:.4f}\n\n")
        with open(f"{args.results_dir}/{benchmark}.txt", "a") as f:
            end_time = time.time()
            f.write("final_result:\n")
            f.write(f"clip:{sum(clip_similarity_list) / len(clip_similarity_list):.4f}\n")
            f.write(f"dinov2:{sum(dinov2_similarity_list) / len(dinov2_similarity_list):.4f}\n")
            f.write(f"IRF:{sum(IRF_similarity_list) / len(IRF_similarity_list):.4f}\n")
            f.write(f"dream sim:{sum(dream_sim_list) / len(dream_sim_list):.4f}\n")
            f.write(f"lpips:{sum(lpips_list) / len(lpips_list):.4f}\n")
            f.write(f"ssim:{sum(ssim_list) / len(ssim_list):.4f}\n")
            f.write(f"image_reward:{sum(image_reward_list) / len(image_reward_list):.4f}\n")
            f.write(f"time used:{end_time - start_time}\n\n")
            clip_similarity_list.clear()
            dinov2_similarity_list.clear()
            IRF_similarity_list.clear()
            dream_sim_list.clear()
            lpips_list.clear()
            ssim_list.clear()
            image_reward_list.clear()

if __name__ == "__main__":
    args = parse_args()
    main(args)