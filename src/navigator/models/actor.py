"""
Actor network for the Navigator RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Dict


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

    The network processes 3D patches using convolutional layers,
    a sequence of past positions using an LSTM, and then combines
    these features to output alpha and beta parameters for Beta distributions.
    """

    def __init__(
        self, input_channels=4, lstm_hidden_size=128, num_lstm_layers=1, eps=1.001
    ):  # Patches input_channels=4
        super().__init__()
        self.eps = eps

        # CNN for patches
        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.Conv3d(16, 16, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=16)
        self.pool2 = nn.Conv3d(32, 32, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=32)
        self.pool3 = nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.cnn_flatten = nn.Flatten()


        self.embed = nn.Linear(3, 128)
        encoder_layer = nn.TransformerEncoderLayer(128, 8, 512, dropout=0.05, batch_first=True)
        self.attention = nn.TransformerEncoder(encoder_layer, 2)

        # Combined feature processing head
        self.head_fc = nn.LazyLinear(256)

        # Output layers for Beta distribution parameters
        self.alpha_layer = nn.Linear(256, 3)  # Renamed from self.alpha to avoid conflict
        self.beta_layer = nn.Linear(256, 3)  # Renamed from self.beta
        self.alpha_layer.bias.data.zero_()
        self.beta_layer.bias.data.zero_()

    def forward(
        self, patches: torch.Tensor, position_sequence: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process patches with CNN
        x_conv = self.conv1(patches)
        x_conv = self.pool1(x_conv)
        x_conv = self.conv2(x_conv)
        x_conv = self.pool2(x_conv)
        x_conv = self.conv3(x_conv)
        x_conv = self.pool3(x_conv)
        cnn_features = self.cnn_flatten(x_conv)

        # We take the output of the last time step
        lstm_features = self.attention(
            self.embed(position_sequence), src_key_padding_mask=mask
        ).mean(dim=1)

        # Concatenate CNN features and LSTM features
        combined_features = torch.cat((cnn_features, lstm_features), dim=1)

        # Pass combined features through the fully connected head
        x = self.head_fc(combined_features)

        # Output alpha/beta parameters
        alpha_params = torch.clamp(F.softplus(self.alpha_layer(x), threshold=5) + self.eps, max=100)
        beta_params = torch.clamp(F.softplus(self.beta_layer(x), threshold=5) + self.eps, max=100)

        return alpha_params, beta_params

    def get_action_dist(self, obs_actor: Dict[str, torch.Tensor]) -> Beta:
        alphas, betas = self.forward(obs_actor)  # Call the modified forward method
        dist = Beta(alphas, betas)
        return dist
