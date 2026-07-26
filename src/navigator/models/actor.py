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
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
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

    def __init__(
        self,
        input_channels=3,
        context_features=5,
        goal_action_prior=0.0,
        eps=1.001,
    ):
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

        # Path-mask updates no longer retain full-volume dilation graphs, so the
        # policy can afford enough channels to represent local 3D topology.
        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=8)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=8)

        # Keep a coarse 2x2x2 spatial grid: full global pooling destroys the
        # directional layout that a navigation policy needs.
        self.spatial_pool = nn.AdaptiveAvgPool3d(2)

        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 2**3, 256), nn.GELU())

        # Output dimensions: 3D actions (mean/std parameters)
        self.alpha = nn.Linear(256 + context_features, 3)
        self.beta = nn.Linear(256 + context_features, 3)
        self.alpha.bias.data.zero_()
        self.beta.bias.data.zero_()
        self.goal_action_prior = goal_action_prior
        self.eps = eps

    def forward(self, x, context):
        """Forward pass through the network."""
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
        x = self.head(x)
        goal_action = ((context[..., -3:] + 1.0) * 0.5).clamp(0.0, 1.0)
        x = torch.cat([x, context], dim=-1)

        # Start exploration around the known endpoint direction. The learned
        # logits remain free to override this prior when local anatomy requires
        # a turn, but the initial policy no longer behaves like a random walk.
        alpha = (
            F.softplus(self.alpha(x))
            + self.eps
            + self.goal_action_prior * goal_action
        )
        beta = (
            F.softplus(self.beta(x))
            + self.eps
            + self.goal_action_prior * (1.0 - goal_action)
        )
        alpha = torch.clamp(alpha, max=100)
        beta = torch.clamp(beta, max=100)

        if len(batch_shape) > 1:
            alpha = alpha.view(*batch_shape, -1)
            beta = beta.view(*batch_shape, -1)

        return alpha, beta

    def get_action_dist(self, obs_actor: torch.Tensor, context: torch.Tensor) -> Beta:
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
        alphas, betas = self(obs_actor, context)
        dist = Beta(alphas, betas)
        return dist  # dist \in [0,1] -> 2 * dist - 1 -> [-1,1] * d -> [-d, d]

        # dist \in [0,1]^3 -> slow down speed by gradient
