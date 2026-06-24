"""
Faster R-CNN Evaluation Script.

Evaluates a trained Torchvision Faster R-CNN model on YOLO-formatted datasets.
Computes mAP using torchmetrics and exports results to CSV.

Usage:
    python evaluation/eval_fasterrcnn.py \
        --model_path /path/to/weights/fasterrcnn_best.pt \
        --data_dir /path/to/datasets/ \
        --num_classes 3 \
        --output_csv results_frcnn.csv
"""

import os
import cv2
import torch
import argparse
import pandas as pd
from tqdm.auto import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> tuple:
    """Parses YOLO format txt to absolute [xmin, ymin, xmax, ymax] boxes."""
    boxes, labels = [], []
    if not os.path.exists(label_path):
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.int64)

    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_c, y_c, w, h = map(float, line.strip().split())
            xmin = (x_c - w / 2) * img_w
            ymin = (y_c - h / 2) * img_h
            xmax = (x_c + w / 2) * img_w
            ymax = (y_c + h / 2) * img_h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(class_id))

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

def evaluate_condition(model, images_dir: str, labels_dir: str, device: torch.device) -> dict:
    """Runs validation for a single degradation directory."""
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    for img_name in tqdm(image_files, desc="Inferencing", leave=False):
        # Load Image
        img_path = os.path.join(images_dir, img_name)
        img_bgr = cv2.imread(img_path)
        img_h, img_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        # Load Ground Truth
        lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
        gt_boxes, gt_labels = parse_yolo_label(os.path.join(labels_dir, lbl_name), img_w, img_h)
        
        target = [{
            "boxes": gt_boxes.to(device),
            "labels": gt_labels.to(device)
        }]

        # Inference
        with torch.no_grad():
            prediction = model(tensor_img)

        # Update metric
        metric.update(prediction, target)

    # Compute final metrics for this condition
    result = metric.compute()
    return {
        "mAP_50": round(result['map_50'].item(), 4),
        "mAP_50_95": round(result['map'].item(), 4),
        "Precision": round(result['map_small'].item(), 4), # Mapping map_small as proxy if exact global P/R isn't yielded by default
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on degraded splits.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best.pt weights.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory containing split folders (e.g., clean/, dusk/).")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of foreground classes (excluding background).")
    parser.add_argument("--output_csv", type=str, default="frcnn_eval.csv", help="Output CSV filename.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Build model architecture (num_classes + 1 for background)
    print("Loading Faster R-CNN architecture...")
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=args.num_classes + 1)
    
    # Load weights
    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device).eval()

    conditions = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    results = []

    print("\nStarting Evaluation Matrix...")
    for condition in conditions:
        print(f"  -> Evaluating condition: [{condition}]")
        img_dir = os.path.join(args.data_dir, condition, "images")
        lbl_dir = os.path.join(args.data_dir, condition, "labels")
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"     Skipping {condition}: Missing images/ or labels/ subdirectories.")
            continue
            
        metrics = evaluate_condition(model, img_dir, lbl_dir, device)
        
        results.append({
            "Model": "Faster R-CNN",
            "Condition": condition,
            **metrics
        })

    # Save Results
    df = pd.DataFrame(results)
    df.sort_values(by="Condition", inplace=True)
    df.to_csv(args.output_csv, index=False)
    
    print(f"\n✓ Evaluation complete. Results saved to {args.output_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
