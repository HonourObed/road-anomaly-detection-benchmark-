"""
Self-Calibrated Illumination (SCI) Network Architecture.

This module defines the inline PyTorch architecture for the SCI model used in 
the low-light enhancement pipeline. It includes the core EnhanceNetwork and 
the SCIEnhancer wrapper that handles partial state-dict loading.
"""

import torch
import torch.nn as nn

class EnhanceNetwork(nn.Module):
    """Core SCI enhancement network consisting of convolutional blocks."""
    
    def __init__(self, layers: int, channels: int):
        super().__init__()
        kernel_size, dilation = 3, 1
        padding = int((kernel_size - 1) / 2) * dilation
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, 1, padding), 
            nn.ReLU()
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels), 
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([self.conv for _ in range(layers)])
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, 1, 1), 
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        illu = self.out_conv(fea) + x
        return torch.clamp(illu, 0.0001, 1)


class SCIEnhancer(nn.Module):
    """
    Wrapper for the SCI Network that automatically loads pre-trained weights 
    and handles the division operation to yield the enhanced image.
    """
    
    def __init__(self, weights_path: str):
        super().__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        
        # Load weights safely
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        
        # Extract only the enhancement network weights (ignoring illumination/calibration heads)
        enhance_state = {
            k.replace('enhance.', ''): v
            for k, v in state.items()
            if k.startswith('enhance.')
        }
        self.enhance.load_state_dict(enhance_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W) normalized to [0, 1].
        Returns:
            Enhanced image tensor of shape (B, 3, H, W).
        """
        illumination_map = self.enhance(x)
        enhanced_image = x / illumination_map
        return torch.clamp(enhanced_image, 0, 1)
