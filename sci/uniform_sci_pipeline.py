"""
Uniform SCI Enhancement Pipeline.

Applies static Self-Calibrated Illumination (SCI) enhancement to degraded 
datasets based on predefined rules per degradation level. This script generates 
the baseline enhanced datasets used to demonstrate the limitations of uniform 
enhancement prior to introducing luminance-adaptive routing.

Usage:
    python uniform_sci_pipeline.py \
        --input_dir /path/to/degraded_datasets \
        --output_dir /path/to/sci_lighting_datasets \
        --weights_dir /path/to/sci_weights
"""

import os
import cv2
import shutil
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm

# Import the architecture and utilities from the local sci module
from sci.sci_model import SCIEnhancer
from sci.enhance_lighting import apply_sci

def main():
    parser = argparse.ArgumentParser(description="Run uniform SCI enhancement on degraded splits.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to degraded datasets (e.g., lighting_base_dir).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the enhanced datasets.")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing medium.pt and difficult.pt.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on compute device: {device}")

    # ── 1. Load both weight variants ──────────────────────────────────────────
    print("Loading SCI models...")
    sci_medium = SCIEnhancer(os.path.join(args.weights_dir, "medium.pt")).to(device).eval()
    sci_difficult = SCIEnhancer(os.path.join(args.weights_dir, "difficult.pt")).to(device).eval()
    print("Both SCI models loaded successfully.\n")

    # ── 2. Per-level configuration ────────────────────────────────────────────
    # Dusk:   medium weights, no denoising
    # Night:  difficult weights + mild denoising
    # Severe: difficult weights + moderate denoising
    level_configs = {
        "level_1_dusk": {
            "model": sci_medium,
            "denoise": False,
            "denoise_strength": 0,
        },
        "level_2_night": {
            "model": sci_difficult,
            "denoise": True,
            "denoise_strength": 5,
        },
        "level_3_severe": {
            "model": sci_difficult,
            "denoise": True,
            "denoise_strength": 7,
        },
    }

    # ── 3. Execution Pipeline ─────────────────────────────────────────────────
    print("--- Starting Uniform SCI Enhancement ---")

    for level_name, cfg in level_configs.items():
        weight_type = 'difficult' if cfg['model'] is sci_difficult else 'medium'
        print(f"\nProcessing [{level_name}] "
              f"(weights={weight_type}, denoise={cfg['denoise']})")

        src_img_dir = os.path.join(args.input_dir, level_name, "images")
        src_lbl_dir = os.path.join(args.input_dir, level_name, "labels")
        out_img_dir = os.path.join(args.output_dir, level_name, "images")
        out_lbl_dir = os.path.join(args.output_dir, level_name, "labels")
        
        if not os.path.exists(src_img_dir):
            print(f"  -> Skipping {level_name}: Directory not found.")
            continue

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for img_name in tqdm(images, desc=level_name):
            img_bgr = cv2.imread(os.path.join(src_img_dir, img_name))
            
            # Convert to RGB tensor
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor_in = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

            # Apply enhancement using the shared utility function
            tensor_out = apply_sci(
                sci_model=cfg["model"],
                img_bgr=img_bgr,
                base_tensor=tensor_in,
                device=device,
                denoise=cfg["denoise"],
                denoise_k=cfg["denoise_strength"]
            )

            # Convert back to OpenCV BGR format and save
            out_np = tensor_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_bgr = cv2.cvtColor(np.clip(out_np * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_img_dir, img_name), out_bgr)

            # Copy bounding box labels
            lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
            src_lbl = os.path.join(src_lbl_dir, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(out_lbl_dir, lbl_name))

    print(f"\n✓ Done. Uniformly enhanced datasets saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
