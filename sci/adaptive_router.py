"""
Luminance-Adaptive Inference Router.

Measures the Rec. 601 perceptual luminance of incoming frames and routes them 
to the appropriate enhancement branch (Clean, Dusk, or Severe) to optimize 
both latency and downstream object detection accuracy.
"""

import cv2
import time
import torch
import numpy as np
from typing import Dict, Tuple

# Import from local sci module
from sci.sci_model import SCIEnhancer
from sci.enhance_lighting import apply_sci

class AdaptiveRouter:
    """Routes images to varying enhancement branches based on perceptual luminance."""
    
    def __init__(self, medium_weights_path: str, difficult_weights_path: str, device: torch.device):
        """
        Initializes the router and loads the required SCI models into memory.
        """
        self.device = device
        
        print("Loading SCI Medium (Dusk) branch...")
        self.sci_medium = SCIEnhancer(medium_weights_path).to(device).eval()
        
        print("Loading SCI Difficult (Severe) branch...")
        self.sci_difficult = SCIEnhancer(difficult_weights_path).to(device).eval()
        
        # Luminance Thresholds
        self.THRESHOLD_CLEAN = 0.40
        self.THRESHOLD_SEVERE = 0.20

    def process(self, img_bgr: np.ndarray, tensor: torch.Tensor) -> Tuple[torch.Tensor, str, Dict[str, float]]:
        """
        Measures luminance and routes the tensor through the appropriate pipeline.
        
        Args:
            img_bgr: Numpy BGR image array (for luminance and denoising).
            tensor: PyTorch tensor representation of the image.
            
        Returns:
            Tuple containing:
            - Final processed tensor ready for detection.
            - Branch name string ("A (Clean)", "B (Dusk)", "C (Severe)").
            - Dictionary containing millisecond latencies for 'lum_ms' and 'enh_ms'.
        """
        latencies = {"lum_ms": 0.0, "enh_ms": 0.0}

        # --- 1. Measure Luminance ---
        t_lum0 = time.perf_counter()
        
        # cv2.cvtColor to GRAY inherently uses Rec. 601 perceptual weights
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lum = np.mean(gray) / 255.0
        
        t_lum1 = time.perf_counter()
        latencies["lum_ms"] = (t_lum1 - t_lum0) * 1000

        # --- 2. Adaptive Routing ---
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        t_enh0 = time.perf_counter()

        if lum >= self.THRESHOLD_CLEAN:
            # Branch A: Image is well-lit. Bypass enhancement.
            branch = "A (Clean)"
            output_tensor = tensor
            
        elif self.THRESHOLD_SEVERE <= lum < self.THRESHOLD_CLEAN:
            # Branch B: Moderate darkness (Dusk). Apply standard enhancement.
            branch = "B (Dusk)"
            output_tensor = apply_sci(
                self.sci_medium, img_bgr, tensor, self.device, denoise=False
            )
            
        else:
            # Branch C: Severe darkness. Apply denoising + aggressive enhancement.
            branch = "C (Severe)"
            output_tensor = apply_sci(
                self.sci_difficult, img_bgr, tensor, self.device, denoise=True, denoise_k=5
            )

        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        t_enh1 = time.perf_counter()
        
        if branch != "A (Clean)":
            latencies["enh_ms"] = (t_enh1 - t_enh0) * 1000

        return output_tensor, branch, latencies
