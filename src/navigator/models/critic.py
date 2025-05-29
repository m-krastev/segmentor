"""
Critic network for the Navigator RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ConvBlock(nn.Module):
    """
    A simple convolutional block with Conv3D, GroupNorm, and GELU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class CriticNetwork(nn.Module):
    """
    Critic network for PPO that estimates state values.

    The network processes 3D patches using convolutional layers
    and outputs value estimates.
    """

    def __init__(self, input_channels=4):
        """
        Initialize the critic network.

        Args:
            input_channels: Number of input channels (default: 4)
        """
        super().__init__()

        self.conv1 = ConvBlock(input_channels, 32, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.Conv3d(32, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=16)
        self.pool2 = nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1, num_groups=32)
        self.pool3 = nn.Conv3d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        out = self.head(x)
        return out