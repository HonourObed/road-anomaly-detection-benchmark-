"""
Physically Grounded Weather Degradation Pipeline.

Simulates adverse weather conditions (fog, rain) by applying atmospheric 
scattering models and kinetic blur to clean images. This generates the 
synthetic evaluation splits required for the detection benchmark.

Usage:
    python weather_pipeline.py \
        --clean_images /path/to/valid/images \
        --clean_labels /path/to/valid/labels \
        --output_base /path/to/output
"""

import os
import cv2
import shutil
import numpy as np
import argparse
from tqdm.auto import tqdm

def simulate_fog(image_rgb: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    Simulates fog using a simplified atmospheric scattering model.
    
    Args:
        image_rgb: Input RGB uint8 image.
        severity:  Fog density [0.0 to 1.0]. Higher is denser.
        
    Returns:
        Fog-degraded RGB uint8 image.
    """
    img_float = image_rgb.astype(np.float32) / 255.0
    
    # Assume a uniform atmospheric light (white/grayish)
    atmospheric_light = np.array([0.85, 0.85, 0.85], dtype=np.float32)
    
    # Generate a depth map proxy (using a simple gradient for this simulation)
    h, w = img_float.shape[:2]
    depth_map = np.tile(np.linspace(0.5, 1.0, h)[:, None], (1, w))
    
    # Transmission map based on Beer-Lambert law
    transmission = np.exp(-severity * depth_map)
    transmission = np.expand_dims(transmission, axis=-1)
    
    # Blend original image with atmospheric light
    fog_img = img_float * transmission + atmospheric_light * (1 - transmission)
    fog_img = np.clip(fog_img * 255.0, 0, 255).astype(np.uint8)
    
    return fog_img

def simulate_rain(image_rgb: np.ndarray, drop_length: int = 15, drop_angle: int = -5) -> np.ndarray:
    """
    Simulates rain by generating noise, elongating it via motion blur, 
    and blending it into the image.
    
    Args:
        image_rgb:   Input RGB uint8 image.
        drop_length: Length of the motion blur kernel (rain streak length).
        drop_angle:  Angle of the falling rain.
        
    Returns:
        Rain-degraded RGB uint8 image.
    """
    h, w, c = image_rgb.shape
    rain_drops = np.random.uniform(0, 255, (h, w)).astype(np.float32)
    
    # Threshold to create sparse drops
    rain_drops[rain_drops < 240] = 0
    
    # Create motion blur kernel for streaks
    kernel = np.zeros((drop_length, drop_length), dtype=np.float32)
    center = drop_length // 2
    cv2.line(kernel, (center - int(drop_angle), 0), (center + int(drop_angle), drop_length), 1, 1)
    kernel /= kernel.sum()
    
    # Apply motion blur to the drops
    rain_streaks = cv2.filter2D(rain_drops, -1, kernel)
    rain_streaks = np.expand_dims(rain_streaks, axis=-1).repeat(3, axis=-1)
    
    # Blend using Screen blending mode
    img_float = image_rgb.astype(np.float32)
    blended = img_float + rain_streaks
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended

def main():
    parser = argparse.ArgumentParser(description="Apply weather degradation to a dataset.")
    parser.add_argument("--clean_images", type=str, required=True, help="Path to clean images.")
    parser.add_argument("--clean_labels", type=str, required=True, help="Path to clean YOLO labels.")
    parser.add_argument("--output_base", type=str, required=True, help="Base output directory.")
    args = parser.parse_args()

    configs = {
        "fog_light": {"type": "fog", "severity": 0.3},
        "fog_heavy": {"type": "fog", "severity": 0.7},
        "rain_heavy": {"type": "rain", "drop_length": 25, "drop_angle": -10},
    }

    images = [f for f in os.listdir(args.clean_images) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for level_name, cfg in configs.items():
        out_img_dir = os.path.join(args.output_base, level_name, "images")
        out_lbl_dir = os.path.join(args.output_base, level_name, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        for img_name in tqdm(images, desc=level_name):
            img_bgr = cv2.imread(os.path.join(args.clean_images, img_name))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if cfg["type"] == "fog":
                degraded_rgb = simulate_fog(img_rgb, severity=cfg["severity"])
            elif cfg["type"] == "rain":
                degraded_rgb = simulate_rain(img_rgb, drop_length=cfg["drop_length"], drop_angle=cfg["drop_angle"])

            degraded_bgr = cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_img_dir, img_name), degraded_bgr)

            # Copy corresponding labels
            lbl_name = img_name.rsplit(".", 1)[0] + ".txt"
            src_lbl = os.path.join(args.clean_labels, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(out_lbl_dir, lbl_name))

    print(f"\nDone. Saved degraded datasets to: {args.output_base}")

if __name__ == "__main__":
    main()
