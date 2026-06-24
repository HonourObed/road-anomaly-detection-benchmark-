"""
Physically grounded lighting degradation pipeline.

Simulates three levels of lighting degradation using gamma curve compression
(perceptually correct darkening) and signal-dependent Gaussian noise
(approximating Poisson shot noise from real camera sensors).

Usage:
    python degradation/lighting_pipeline.py \
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


def simulate_low_light(image_rgb: np.ndarray, gamma: float,
                       noise_sigma: float, blur_kernel: int = 0) -> np.ndarray:
    """
    Simulates low-light degradation.

    Args:
        image_rgb:    Input RGB uint8 image.
        gamma:        Gamma > 1 darkens. 1.6=dusk, 2.8=night, 4.2=severe.
        noise_sigma:  Std dev of signal-dependent Gaussian noise in float space.
                      0.025=dusk, 0.06=night, 0.11=severe.
        blur_kernel:  Horizontal motion blur kernel size (0 = no blur, must be odd).

    Returns:
        Degraded RGB uint8 image.
    """
    img_float = image_rgb.astype(np.float32) / 255.0

    # Gamma compression: perceptually correct darkening
    img_dark = np.power(img_float, gamma)

    # Signal-dependent noise: darker areas get proportionally less noise
    # approximates Poisson shot noise behavior
    signal_dependent_std = noise_sigma * (1.0 - img_dark + 0.05)
    noise = np.random.normal(0, 1, img_dark.shape).astype(np.float32) * signal_dependent_std
    img_noisy = img_dark + noise

    # Optional horizontal motion blur (simulates camera shake at low shutter speeds)
    if blur_kernel > 0 and blur_kernel % 2 == 1:
        kernel = np.zeros((blur_kernel, blur_kernel), dtype=np.float32)
        kernel[blur_kernel // 2, :] = 1.0 / blur_kernel
        img_noisy = cv2.filter2D(img_noisy, -1, kernel)

    return (np.clip(img_noisy, 0.0, 1.0) * 255).astype(np.uint8)


DEGRADATION_CONFIGS = {
    "level_1_dusk": {
        "gamma": 1.6,
        "noise_sigma": 0.025,
        "blur_kernel": 0,
    },
    "level_2_night": {
        "gamma": 2.8,
        "noise_sigma": 0.06,
        "blur_kernel": 0,
    },
    "level_3_severe": {
        "gamma": 4.2,
        "noise_sigma": 0.11,
        "blur_kernel": 5,
    },
}


def run_pipeline(clean_images: str, clean_labels: str, output_base: str):
    print("--- Starting Lighting Degradation Pipeline ---")

    for level_name, cfg in DEGRADATION_CONFIGS.items():
        print(f"\nGenerating {level_name} "
              f"(gamma={cfg['gamma']}, sigma={cfg['noise_sigma']}, "
              f"blur={cfg['blur_kernel']})...")

        out_img_dir = os.path.join(output_base, level_name, "images")
        out_lbl_dir = os.path.join(output_base, level_name, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        images = [f for f in os.listdir(clean_images)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for img_name in tqdm(images, desc=level_name):
            img_bgr = cv2.imread(os.path.join(clean_images, img_name))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            degraded_rgb = simulate_low_light(
                img_rgb,
                gamma=cfg["gamma"],
                noise_sigma=cfg["noise_sigma"],
                blur_kernel=cfg["blur_kernel"],
            )

            degraded_bgr = cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_img_dir, img_name), degraded_bgr)

            lbl_name = img_name.rsplit(".", 1)[0] + ".txt"
            src_lbl = os.path.join(clean_labels, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(out_lbl_dir, lbl_name))

    print(f"\nDone. Saved to: {output_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_images", required=True)
    parser.add_argument("--clean_labels", required=True)
    parser.add_argument("--output_base", default="degradation/degraded_datasets")
    args = parser.parse_args()
    run_pipeline(args.clean_images, args.clean_labels, args.output_base)
