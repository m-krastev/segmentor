"""
Critic network for the Navigator RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..config import Config


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
        self.net = nn.Sequential(
            nn.Conv3d(input_channels, 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, 8),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out = self.net(x)
        return out