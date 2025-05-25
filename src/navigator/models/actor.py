"""
Actor network for the Navigator RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from ..config import Config


class ActorNetwork(nn.Module):
    """
    Actor network for PPO that outputs Beta distribution parameters.

    The network processes 3D patches using convolutional layers
    and outputs alpha and beta parameters for Beta distributions.
    """

    def __init__(self, input_channels=3, eps = 1.001):
        """
        Initialize the actor network.

        Args:
            input_channels: Number of input channels (default: 3)
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(input_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 8),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        # Output layer for alpha/beta parameters (6 values = 3 dimensions × 2 params)
        self.alpha = nn.Linear(64, 3)
        self.beta = nn.Linear(64, 3)
        self.eps = eps

    def forward(self, x):
        """Forward pass through the network."""
        x = self.net(x)

        # Output alpha/beta parameters
        alpha = torch.clamp(F.softplus(self.alpha(x), threshold=5) + self.eps, max=100)
        beta = torch.clamp(F.softplus(self.beta(x), threshold=5) + self.eps, max=100)
        return alpha, beta

    def get_action_dist(self, obs_actor: torch.Tensor) -> Beta:
        """
        Get Beta distribution from observation.

        Args:
            obs_actor: Observation tensor

        Returns:
            Beta distribution object
        """
        # alpha_beta = self.forward(obs_actor)
        # alpha_beta_pairs = alpha_beta.view(-1, 3, 2)
        # alphas = alpha_beta_pairs[..., 0]
        # betas = alpha_beta_pairs[..., 1]
        alphas, betas = self(obs_actor)
        dist = Beta(alphas, betas)
        return dist
