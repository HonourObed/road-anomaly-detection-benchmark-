"""
Ultralytics Evaluation Script (YOLO & RT-DETR).

Evaluates trained Ultralytics models across multiple degradation splits.
Aggregates mAP@50 and mAP@50-95 scores into a consolidated CSV report.

Usage:
    python evaluation/eval_yolo_rtdetr.py \
        --model_path /path/to/weights/yolov8n_best.pt \
        --model_type yolo \
        --data_configs /path/to/configs/ \
        --output_csv results_yolo.csv
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO, RTDETR

def evaluate_split(model, data_yaml: str) -> dict:
    """
    Runs validation on a specific dataset split.
    
    Args:
        model: Loaded Ultralytics model.
        data_yaml: Path to the dataset configuration YAML.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    # Run validation in quiet mode to keep terminal clean
    metrics = model.val(data=data_yaml, split='val', verbose=False)
    
    return {
        "mAP_50": round(metrics.box.map50, 4),
        "mAP_50_95": round(metrics.box.map, 4),
        "Precision": round(metrics.box.mp, 4),
        "Recall": round(metrics.box.mr, 4)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO/RT-DETR models.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt weight file.")
    parser.add_argument("--model_type", type=str, choices=["yolo", "rtdetr"], required=True, help="Architecture type.")
    parser.add_argument("--data_configs", type=str, required=True, help="Directory containing dataset .yaml files (clean.yaml, dusk.yaml, etc.).")
    parser.add_argument("--output_csv", type=str, default="ultralytics_eval.csv", help="Output CSV filename.")
    args = parser.parse_args()

    print(f"Loading {args.model_type.upper()} model from {args.model_path}...")
    if args.model_type == "yolo":
        model = YOLO(args.model_path)
    else:
        model = RTDETR(args.model_path)

    config_dir = Path(args.data_configs)
    yaml_files = list(config_dir.glob("*.yaml"))
    
    if not yaml_files:
        print(f"Error: No .yaml files found in {args.data_configs}")
        return

    results = []
    
    print("\nStarting Evaluation Matrix...")
    for yaml_path in yaml_files:
        condition_name = yaml_path.stem
        print(f"  -> Evaluating condition: [{condition_name}]")
        
        metrics = evaluate_split(model, str(yaml_path))
        
        results.append({
            "Model": Path(args.model_path).stem,
            "Condition": condition_name,
            **metrics
        })

    # Save and display results
    df = pd.DataFrame(results)
    
    # Sort logically if naming conventions allow (e.g., clean -> dusk -> night -> severe)
    df.sort_values(by="Condition", inplace=True)
    df.to_csv(args.output_csv, index=False)
    
    print(f"\n✓ Evaluation complete. Results saved to {args.output_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
