"""
Actor network for the Navigator RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

class ConvBlock(nn.Module):
    """
    A simple convolutional block with Conv3D, GroupNorm, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=8):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

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

        # self.net = nn.Sequential(
        #     ConvBlock(input_channels, 64, kernel_size=3, padding=1),
        #     # Strided convolution to downsample
        #     nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
        #     ConvBlock(64, 128, kernel_size=3, padding=1),
        #     # Strided convolution to downsample
        #     nn.Conv3d(128, 128, kernel_size=2, stride=2, padding=0, bias=False),
        #     ConvBlock(128, 256, kernel_size=3, padding=1),
        #     # Strided convolution to downsample
        #     nn.Conv3d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),
        #     nn.Flatten(),
        #     nn.LazyLinear(256),
        #     nn.GroupNorm(8, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 256),
        #     nn.GroupNorm(8, 256),
        #     nn.GELU(),
        # )

        # TODO: Add downscaled patch of the larger position 
        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.Conv3d(16, 16, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=16)
        self.pool2 = nn.Conv3d(32, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=32)
        self.pool3 = nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.GELU()
        )

        # Output layer for alpha/beta parameters (6 values = 3 dimensions × 2 params)
        self.alpha = nn.Linear(512, 3)
        self.beta = nn.Linear(512, 3)
        self.alpha.bias.data.zero_()
        self.beta.bias.data.zero_()
        self.eps = eps

    def forward(self, x):
        """Forward pass through the network."""
        # print(x.shape)
        x = self.conv1(x) # Residual connection
        # print(x.shape)

        x = self.pool1(x)
        # print(x.shape)

        x = self.conv2(x) # Residual connection
        # print(x.shape)

        x = self.pool2(x)
        # print(x.shape)

        x = self.conv3(x) # Residual connection
        # print(x.shape)

        x = self.pool3(x)
        # print(x.shape)
        x = self.head(x)

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
        return dist # dist \in [0,1] -> 2 * dist - 1 -> [-1,1] * d -> [-d, d]
    
        # dist \in [0,1]^3 -> slow down speed by gradient 
