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

    def __init__(self, config: Config, input_channels=3):
        """
        Initialize the actor network.

        Args:
            config: Configuration object
            input_channels: Number of input channels (default: 3)
        """
        super().__init__()
        self.config = config
        self.patch_size_vox = config.patch_size_vox

        # Convolutional layers
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 16)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(8, 64)
        self.pool4 = nn.MaxPool3d(2)

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *self.patch_size_vox)
            pooled_output = self.pool4(self.pool3(self.pool2(self.pool1(dummy_input))))
            final_channels = 64
            self.flattened_size = (
                final_channels
                * pooled_output.shape[-3]
                * pooled_output.shape[-2]
                * pooled_output.shape[-1]
            )

        if self.flattened_size <= 0:
            raise ValueError(f"Actor flat size <= 0 ({self.flattened_size}).")

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.gn_fc1 = nn.GroupNorm(8, 64)

        self.fc2 = nn.Linear(64, 64)
        self.gn_fc2 = nn.GroupNorm(8, 64)

        # Output layer for alpha/beta parameters (6 values = 3 dimensions × 2 params)
        self.fc_out = nn.Linear(64, 6)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))
        x = self.pool4(F.relu(self.gn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        if x.shape[1] != self.flattened_size:
            raise ValueError(
                f"Runtime Actor flat size mismatch. Expected {self.flattened_size}, got {x.shape[1]}."
            )

        # Fully connected layers
        x = F.relu(self.gn_fc1(self.fc1(x)))
        x = F.relu(self.gn_fc2(self.fc2(x)))

        # Output alpha/beta parameters
        ab_params = self.fc_out(x)
        alpha_beta = F.softplus(ab_params) + 1.0  # Ensure alpha, beta > 1

        return alpha_beta

    def get_action_dist(self, obs_actor: torch.Tensor) -> Beta:
        """
        Get Beta distribution from observation.

        Args:
            obs_actor: Observation tensor

        Returns:
            Beta distribution object
        """
        alpha_beta = self.forward(obs_actor)
        alpha_beta_pairs = alpha_beta.view(-1, 3, 2)
        alphas = alpha_beta_pairs[..., 0]
        betas = alpha_beta_pairs[..., 1]
        dist = Beta(alphas, betas)
        return dist
