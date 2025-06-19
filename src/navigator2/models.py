import torch.nn as nn
from torch.nn import functional as F
from skrl.models.torch import DeterministicMixin, Model
from navigator2.beta import BetaMixin


class ActorNetwork(BetaMixin, Model):
    """
    Actor network for PPO that outputs Beta distribution parameters.

    The network processes 3D patches using convolutional layers
    and outputs alpha and beta parameters for Beta distributions.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        reduction="sum",
        input_channels=3,
    ):
        """
        Initialize the actor network.

        Args:
            config: Configuration object
            input_channels: Number of input channels (default: 3)
        """
        Model.__init__(self, observation_space, action_space, device)
        BetaMixin.__init__(self, reduction=reduction, role="policy")

        self.net = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.MaxPool3d(2),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.MaxPool3d(2),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.MaxPool3d(2),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.MaxPool3d(2),
            nn.GELU(),
            # 64 x 1 x 1 x 1 after pooling
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.LazyLinear(64),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        # Output layer for alpha/beta parameters (6 values = 3 dimensions × 2 params)
        self.alpha = nn.Linear(64, 3)
        self.beta = nn.Linear(64, 3)

    def compute(self, x, role):
        """Forward pass through the network."""
        x = self.net(x["states"])

        # Output alpha/beta parameters
        alpha = F.softplus(self.alpha(x)) + 1
        beta = F.softplus(self.beta(x)) + 1
        return alpha, beta, {"mean_actions": None}


class CriticNetwork(DeterministicMixin, Model):
    """
    Critic network for PPO that estimates state values.

    The network processes 3D patches using convolutional layers
    and outputs value estimates.
    """

    def __init__(
        self, observation_space, action_space, device=None, input_channels=4, clip_actions=False
    ):
        """
        Initialize the critic network.

        Args:
            config: Configuration object
            input_channels: Number of input channels (default: 4)
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role="value")
        self.net = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.MaxPool3d(2),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.MaxPool3d(2),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.MaxPool3d(2),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.MaxPool3d(2),
            nn.GELU(),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.LazyLinear(64),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def compute(self, x, role):
        return self.net(x["states"]), {}
