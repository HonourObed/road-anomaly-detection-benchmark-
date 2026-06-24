"""
Jetson Edge Hardware Latency Benchmark.

Evaluates the throughput and latency of the luminance-adaptive object detection 
pipeline on edge hardware (Jetson Orin Nano). It measures the architectural 
delay of each routing branch (A, B, C) by isolating the time taken for luminance 
calculation, image enhancement, and bounding box detection.

Usage:
    python jetson/latency_benchmark.py \
        --mixed_images /home/jetson/samples/mixed_test_set \
        --weights_dir /home/jetson/weights \
        --output_csv /home/jetson/adaptive_latency_results.csv
"""

import os
import cv2
import time
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from ultralytics import YOLO, RTDETR
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Import the adaptive router from the local sci module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sci.adaptive_router import AdaptiveRouter

def load_faster_rcnn(weights_path: str, device: torch.device, num_classes: int = 3):
    """Loads the Faster R-CNN architecture with custom trained weights."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes + 1)
    
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device).eval()

def load_image_tensor(img_path: str, img_size: int, device: torch.device) -> tuple:
    """Loads image and converts to PyTorch tensor."""
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_bgr, tensor

def main():
    parser = argparse.ArgumentParser(description="Run latency benchmarks on Jetson hardware.")
    parser.add_argument("--mixed_images", type=str, required=True, help="Directory with mixed lighting images.")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing all .pt weight files.")
    parser.add_argument("--output_csv", type=str, default="adaptive_latency_results.csv", help="Output CSV path.")
    parser.add_argument("--img_size", type=int, default=640, help="Inference resolution.")
    parser.add_argument("--warmup_runs", type=int, default=10, help="Number of warmup runs to stabilize GPU clock.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Hardware] Compute Device: {device}")
    if device.type == 'cuda':
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}\n")

    # ── Initialize Router ─────────────────────────────────────────────────────
    print("[Init] Booting Adaptive Router...")
    router = AdaptiveRouter(
        medium_weights_path=os.path.join(args.weights_dir, "sci_medium.pt"),
        difficult_weights_path=os.path.join(args.weights_dir, "sci_difficult.pt"),
        device=device
    )

    # ── Load Detection Models ─────────────────────────────────────────────────
    print("[Init] Loading Detection Models...")
    models = [
        ("YOLOv8n", YOLO(os.path.join(args.weights_dir, "yolov8n_best.pt")), True),
        ("YOLOv12n", YOLO(os.path.join(args.weights_dir, "yolov12n_best.pt")), True),
        ("RT-DETR", RTDETR(os.path.join(args.weights_dir, "rtdetr_best.pt")), True),
        ("Faster R-CNN", load_faster_rcnn(os.path.join(args.weights_dir, "fasterrcnn_best.pt"), device), False),
    ]

    # ── Prepare Image Stream ──────────────────────────────────────────────────
    image_paths = [os.path.join(args.mixed_images, f) for f in os.listdir(args.mixed_images) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_paths:
        raise FileNotFoundError(f"No valid images found in {args.mixed_images}")

    results = []

    # ── Execution Loop ────────────────────────────────────────────────────────
    for model_name, detector, is_ultralytics in models:
        print(f"\n--- Benchmarking: {model_name} ---")
        
        # Track stats per branch
        stats = {
            "A (Clean)":  {"lum_ms": [], "enh_ms": [], "det_ms": []},
            "B (Dusk)":   {"lum_ms": [], "enh_ms": [], "det_ms": []},
            "C (Severe)": {"lum_ms": [], "enh_ms": [], "det_ms": []}
        }

        for i, img_path in enumerate(image_paths):
            img_bgr, tensor = load_image_tensor(img_path, args.img_size, device)

            # 1. Routing & Enhancement (Handled by AdaptiveRouter)
            processed_tensor, branch, routing_latencies = router.process(img_bgr, tensor)

            # 2. Detection Inference
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t_det0 = time.perf_counter()

            if is_ultralytics:
                # Ultralytics models require numpy arrays
                img_array = (processed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                _ = detector(img_array, verbose=False)
            else:
                # Torchvision models accept raw tensor lists
                _ = detector([processed_tensor.squeeze(0)])

            torch.cuda.synchronize() if device.type == 'cuda' else None
            t_det1 = time.perf_counter()
            det_ms = (t_det1 - t_det0) * 1000

            # 3. Log results (skipping warmup runs)
            if i >= args.warmup_runs:
                stats[branch]["lum_ms"].append(routing_latencies["lum_ms"])
                stats[branch]["enh_ms"].append(routing_latencies["enh_ms"])
                stats[branch]["det_ms"].append(det_ms)

        # Compile statistics
        for branch_name, branch_data in stats.items():
            count = len(branch_data["det_ms"])
            if count == 0: continue
                
            avg_lum = np.mean(branch_data["lum_ms"])
            avg_enh = np.mean(branch_data["enh_ms"])
            avg_det = np.mean(branch_data["det_ms"])
            avg_total = avg_lum + avg_enh + avg_det
            fps = 1000.0 / avg_total if avg_total > 0 else 0

            print(f"Branch {branch_name:10s} (N={count:2d}) | "
                  f"Lum: {avg_lum:4.1f}ms | Enh: {avg_enh:5.1f}ms | "
                  f"Det: {avg_det:5.1f}ms | Total: {avg_total:6.1f}ms | FPS: {fps:5.1f}")

            results.append({
                "Model": model_name,
                "Branch": branch_name,
                "Sample_Count": count,
                "Luminance_ms": round(avg_lum, 2),
                "Enhancement_ms": round(avg_enh, 2),
                "Detection_ms": round(avg_det, 2),
                "Total_Pipeline_ms": round(avg_total, 2),
                "Overall_FPS": round(fps, 2)
            })

    # ── Save Results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Benchmark complete. CSV saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
