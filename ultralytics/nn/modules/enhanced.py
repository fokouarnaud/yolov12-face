"""
Modules Enhanced pour YOLOv12-Face
Version corrigée sans conflits avec les modules Ultralytics existants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class A2Module(nn.Module):
    """Area Attention Module pour la détection de visages"""
    
    def __init__(self, in_channels, out_channels, n=1, reduction=16, *args, **kwargs):
        """Initialize A2Module
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n: Not used, for compatibility with YOLO parser
            reduction: Channel reduction factor for attention
        """
        super(A2Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Ensure we have at least 1 channel after reduction
        mid_channels = max(1, in_channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        # Output projection
        self.conv_out = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        
        # Apply channel attention
        x = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
        
        # Apply spatial attention
        x = x * spatial_att
        
        return self.conv_out(x)


class RELAN(nn.Module):
    """Residual Efficient Layer Aggregation Network pour l'agrégation multi-échelle"""
    
    def __init__(self, in_channels, out_channels, n=1, *args, **kwargs):
        """Initialize RELAN
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n: Not used, for compatibility with YOLO parser
        """
        super(RELAN, self).__init__()
        
        # Multi-scale convolutions
        # Ensure we have at least 1 channel for each branch
        branch_channels = max(1, out_channels // 4)
        
        self.conv1x1 = nn.Conv2d(in_channels, branch_channels, 1)
        self.conv3x3 = nn.Conv2d(in_channels, branch_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, branch_channels, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, branch_channels, 7, padding=3)
        
        # Feature fusion
        # Total channels after concatenation = branch_channels * 4
        total_channels = branch_channels * 4
        self.fusion = nn.Conv2d(total_channels, out_channels, 1)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
            
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Multi-scale features
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv5x5(x)
        f4 = self.conv7x7(x)
        
        # Concatenate and fuse
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        fused = self.fusion(fused)
        
        # Residual connection
        residual = self.residual(x)
        
        return self.relu(fused + residual)


# Export seulement les modules principaux - SANS ALIAS pour éviter les conflits
__all__ = ['A2Module', 'RELAN']
