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

    def __init__(self, input_channels=4, agent_orientation_size=4, eps=1.001):
        """
        Initialize the actor network.

        Args:
            input_channels: Number of input channels for patches (default: 4)
            agent_orientation_size: Size of the agent orientation vector (default: 4)
        """
        super().__init__()

        # Convolutional backbone for processing patches
        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.Conv3d(16, 16, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=16)
        self.pool2 = nn.Conv3d(32, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=32)
        self.pool3 = nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        
        # Head for processing the flattened features from conv backbone
        self.conv_head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),  # Reduced size to make space for agent_pose
            nn.GELU()
        )

        # Combined head for both patch features and agent pose
        self.combined_head = nn.Sequential(
            nn.LazyLinear(512),
            nn.GELU()
        )

        # Output layers for alpha/beta parameters
        self.alpha_layer = nn.Linear(512, 3)
        self.beta_layer = nn.Linear(512, 3)
        self.alpha_layer.bias.data.zero_()
        self.beta_layer.bias.data.zero_()
        self.eps = eps

    def forward(self, patches: torch.Tensor, agent_orientation: torch.Tensor, **kwargs):
        """Forward pass through the network."""
        # Process patches through convolutional backbone
        p = self.conv1(patches)
        p = self.pool1(p)
        p = self.conv2(p)
        p = self.pool2(p)
        p = self.conv3(p)
        p = self.pool3(p)
        patch_features = self.conv_head(p)

        # Concatenate patch features with agent orientation
        combined_features = torch.cat([patch_features, agent_orientation], dim=-1)
        
        # Process combined features through the final head
        x = self.combined_head(combined_features)

        # Output alpha/beta parameters
        alpha = torch.clamp(F.softplus(self.alpha_layer(x)) + self.eps, max=100)
        beta = torch.clamp(F.softplus(self.beta_layer(x)) + self.eps, max=100)
        return alpha, beta

    def get_action_dist(self, obs_actor: dict) -> Beta:
        """
        Get Beta distribution from observation.

        Args:
            obs_actor: A dictionary containing 'patches' and 'agent_orientation'.

        Returns:
            Beta distribution object
        """
        alphas, betas = self.forward(obs_actor["patches"], obs_actor["agent_orientation"])
        dist = Beta(alphas, betas)
        return dist # dist \in [0,1] -> 2 * dist - 1 -> [-1,1] * d -> [-d, d]
    
        # dist \in [0,1]^3 -> slow down speed by gradient
