"""
Critic network for the Navigator RL agent.
"""

import torch
import torch.nn as nn


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

    def __init__(self, input_channels=4, context_features=5):
        """
        Initialize the critic network.

        Args:
            input_channels: Number of input channels (default: 4)
        """
        super().__init__()

        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=8)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=8)

        # Preserve coarse spatial layout while keeping the parameter count
        # independent of the input patch dimensions.
        self.spatial_pool = nn.AdaptiveAvgPool3d(2)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2**3, 256),
            nn.GELU(),
        )
        self.predict = nn.Linear(256 + context_features, 1)

    def forward(self, x, context):
        batch_shape = x.shape[:-4]
        if x.dim() > 5:
            x = x.flatten(0, -5)
            context = context.flatten(0, -2)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        x = self.spatial_pool(x)
        out = self.head(x)
        out = self.predict(torch.cat([out, context], dim=-1))

        if len(batch_shape) > 1:
            out = out.view(*batch_shape, -1)
        return out


class StateActionValueNetwork(nn.Module):
    """
    Critic network for TD3 that estimates state-action values.

    The network processes 3D patches using convolutional layers
    and outputs value estimates.
    """

    def __init__(self, input_channels=4, action_dim=3, context_features=5):
        """
        Initialize the critic network.

        Args:
            input_channels: Number of input channels (default: 4)
        """
        super().__init__()

        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=8)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=8)

        self.spatial_pool = nn.AdaptiveAvgPool3d(2)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2**3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.predict = nn.Linear(128 + context_features + action_dim, 1)

    def forward(self, x, context, action):
        batch_shape = x.shape[:-4]
        if x.dim() > 5:
            x = x.flatten(0, -5)
            context = context.flatten(0, -2)
            action = action.flatten(0, -2)  # Action is (..., 3)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        x = self.spatial_pool(x)
        out = self.head(x)
        out = self.predict(torch.cat([out, context, action], dim=-1))

        if len(batch_shape) > 1:
            out = out.view(*batch_shape, -1)
        return out
