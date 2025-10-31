import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers.
    It uses InstanceNorm for normalization.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        """
        Adds the input tensor to the output of the block (skip connection).
        """
        return x + self.block(x)

print("modules_vqvae2_perceptual.py loaded.")