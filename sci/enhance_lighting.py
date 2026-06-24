"""
Lighting Enhancement Utilities.

Provides helper functions to convert OpenCV images to PyTorch tensors and 
apply the SCI enhancement model, including optional bilateral filtering for 
severe noise conditions.
"""

import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple

def prepare_image_tensor(img_bgr: np.ndarray, img_size: int, device: torch.device) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Resizes and converts an OpenCV BGR image into a normalized RGB PyTorch tensor.
    
    Args:
        img_bgr: Raw input image from OpenCV.
        img_size: Target size for inference (e.g., 640).
        device: Target compute device (CPU/CUDA).
        
    Returns:
        Tuple containing the resized BGR image and the [1, 3, H, W] float tensor.
    """
    img_resized = cv2.resize(img_bgr, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_resized, tensor


@torch.no_grad()
def apply_sci(sci_model: nn.Module, img_bgr: np.ndarray, base_tensor: torch.Tensor, 
              device: torch.device, denoise: bool = False, denoise_k: int = 5) -> torch.Tensor:
    """
    Applies the SCI enhancement model to an image.
    
    Args:
        sci_model: The loaded SCIEnhancer PyTorch model.
        img_bgr: The resized BGR numpy image.
        base_tensor: The clean tensor representation of the image.
        device: Target compute device.
        denoise: If True, applies cv2.bilateralFilter before enhancement.
        denoise_k: Kernel size for the bilateral filter.
        
    Returns:
        Enhanced image tensor [1, 3, H, W] ready for object detection.
    """
    if denoise:
        # Apply edge-preserving bilateral filter to mitigate severe noise
        img_bgr = cv2.bilateralFilter(img_bgr, d=denoise_k, sigmaColor=35, sigmaSpace=35)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        input_tensor = base_tensor
        
    return sci_model(input_tensor)
