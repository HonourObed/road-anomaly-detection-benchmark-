"""
Dataset Sanity Check Utility.

Validates the integrity of the generated datasets before inference or training.
Ensures YOLO format bounding boxes are within normalized limits [0, 1] and checks 
for orphaned images or labels.

Usage:
    python sanity_check.py \
        --images /path/to/images \
        --labels /path/to/labels \
        --fix
"""

import os
import argparse
from tqdm.auto import tqdm

def check_yolo_bounds(label_path: str, fix: bool = False) -> int:
    """
    Validates that YOLO coordinates (x_center, y_center, width, height) 
    are strictly bounded between 0.0 and 1.0.
    
    Args:
        label_path: Path to the .txt label file.
        fix: If True, clips out-of-bounds coordinates to 1.0.
        
    Returns:
        Number of bounding box errors found in the file.
    """
    errors = 0
    valid_lines = []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # Skip malformed lines
            
        cls_id = parts[0]
        coords = [float(p) for p in parts[1:]]
        
        # Check if any coordinate is outside [0, 1]
        if any(c < 0.0 or c > 1.0 for c in coords):
            errors += 1
            if fix:
                coords = [max(0.0, min(1.0, c)) for c in coords]
                
        valid_lines.append(f"{cls_id} {' '.join(f'{c:.6f}' for c in coords)}\n")
        
    if fix and errors > 0:
        with open(label_path, 'w') as f:
            f.writelines(valid_lines)
            
    return errors

def main():
    parser = argparse.ArgumentParser(description="Sanity check YOLO dataset integrity.")
    parser.add_argument("--images", type=str, required=True, help="Path to images folder.")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels folder.")
    parser.add_argument("--fix", action="store_true", help="Automatically clip out-of-bounds coordinates.")
    args = parser.parse_args()

    images = {f.rsplit(".", 1)[0]: f for f in os.listdir(args.images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    labels = {f.rsplit(".", 1)[0]: f for f in os.listdir(args.labels) if f.lower().endswith('.txt')}

    missing_labels = set(images.keys()) - set(labels.keys())
    missing_images = set(labels.keys()) - set(images.keys())
    
    print("\n--- Structural Check ---")
    print(f"Total Images: {len(images)}")
    print(f"Total Labels: {len(labels)}")
    print(f"Images without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(missing_images)}")

    print("\n--- Bounding Box Validation ---")
    total_errors = 0
    files_with_errors = 0
    
    for lbl_key, lbl_file in tqdm(labels.items(), desc="Checking labels"):
        lbl_path = os.path.join(args.labels, lbl_file)
        
        # Ignore completely empty files (background images)
        if os.path.getsize(lbl_path) == 0:
            continue
            
        errors = check_yolo_bounds(lbl_path, fix=args.fix)
        if errors > 0:
            total_errors += errors
            files_with_errors += 1

    print(f"Files with coordinate out-of-bounds errors: {files_with_errors}")
    print(f"Total bad bounding boxes: {total_errors}")
    if args.fix:
        print("✓ All out-of-bounds errors were automatically clipped and fixed.")
    elif total_errors > 0:
        print("! Run with --fix to automatically resolve out-of-bounds coordinates.")
        
if __name__ == "__main__":
    main()
