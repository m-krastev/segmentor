"""Shared visual encoders and recurrent memory modules for Navigator."""

from __future__ import annotations

import math

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import Unbounded
from torchrl.envs import TensorDictPrimer
from torchrl.modules import recurrent_mode

from .actor import ConvBlock


class NavigatorVisualEncoder(nn.Module):
    """Encode local 3-D observations and non-spatial context once.

    The actor and critic share this module in recurrent configurations. Keeping
    the coarse 2x2x2 grid preserves directional information while bounding the
    activation size for sequence training.
    """

    def __init__(
        self,
        input_channels: int,
        context_features: int,
        output_features: int,
    ):
        super().__init__()
        self.conv1 = ConvBlock(input_channels, 16, kernel_size=3, padding=1, num_groups=8)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(16, 32, kernel_size=3, padding=1, num_groups=8)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, padding=1, num_groups=8)
        self.spatial_pool = nn.AdaptiveAvgPool3d(2)
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2**3 + context_features, output_features),
            nn.LayerNorm(output_features),
            nn.GELU(),
        )

    def encode_spatial(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the shared convolutional feature map before spatial pooling."""

        if observation.dim() > 5:
            observation = observation.flatten(0, -5)

        features = self.conv1(observation)
        features = self.pool1(features)
        features = self.conv2(features)
        features = self.pool2(features)
        features = self.conv3(features)
        return features

    def spatial_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only parameters trained by image-space pretraining."""

        return {
            name: value
            for name, value in self.state_dict().items()
            if name.startswith(("conv1.", "conv2.", "conv3."))
        }

    def load_spatial_checkpoint(self, checkpoint_path: str) -> None:
        """Load image-only convolutional weights with safe channel expansion."""

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        source_state = checkpoint.get("spatial_encoder_state_dict", checkpoint)
        if not isinstance(source_state, dict):
            raise ValueError(
                "Visual encoder checkpoint must contain a state dictionary"
            )

        target_state = self.state_dict()
        adapted_state = {}
        for name, source in source_state.items():
            if name not in target_state:
                raise ValueError(f"Unexpected pretrained encoder parameter: {name}")
            target = target_state[name]
            if source.shape == target.shape:
                adapted_state[name] = source
                continue
            if (
                name == "conv1.conv.weight"
                and source.dim() == target.dim() == 5
                and source.shape[0] == target.shape[0]
                and source.shape[2:] == target.shape[2:]
                and source.shape[1] <= target.shape[1]
            ):
                expanded = torch.zeros_like(target)
                expanded[:, : source.shape[1]].copy_(source)
                adapted_state[name] = expanded
                continue
            raise ValueError(
                f"Incompatible pretrained encoder parameter {name}: "
                f"{tuple(source.shape)} != {tuple(target.shape)}"
            )

        missing, unexpected = self.load_state_dict(adapted_state, strict=False)
        non_project_missing = [
            name
            for name in missing
            if not name.startswith("project.")
        ]
        if non_project_missing or unexpected:
            raise ValueError(
                "Incomplete pretrained spatial encoder state: "
                f"missing={non_project_missing}, unexpected={unexpected}"
            )

    def forward(self, observation: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_shape = observation.shape[:-4]
        if observation.dim() > 5:
            context = context.flatten(0, -2)

        features = self.encode_spatial(observation)
        features = self.spatial_pool(features)
        features = self.project(torch.cat([features.flatten(-4), context], dim=-1))

        if len(batch_shape) > 1:
            features = features.view(*batch_shape, -1)
        return features


class NavigatorDenoisingAutoencoder(nn.Module):
    """Pretrain Navigator's spatial encoder on label-free image reconstruction."""

    def __init__(self, input_channels: int = 5):
        super().__init__()
        self.encoder = NavigatorVisualEncoder(
            input_channels=input_channels,
            context_features=0,
            output_features=256,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32, kernel_size=3, padding=1, num_groups=8),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            ConvBlock(16, 16, kernel_size=3, padding=1, num_groups=8),
            nn.Conv3d(16, input_channels, kernel_size=1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder.encode_spatial(observation))


class RecurrentBetaHead(nn.Module):
    """Map recurrent features to a bounded independent Beta policy."""

    def __init__(
        self,
        input_features: int,
        context_features: int,
        goal_action_prior: float = 0.0,
        eps: float = 1.001,
    ):
        super().__init__()
        self.alpha = nn.Linear(input_features, 3)
        self.beta = nn.Linear(input_features, 3)
        nn.init.zeros_(self.alpha.bias)
        nn.init.zeros_(self.beta.bias)
        self.context_features = context_features
        self.goal_action_prior = goal_action_prior
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve checkpoint compatibility with the existing optional prior.
        # It is inactive by default and should remain zero for annotation-free
        # experiments.
        alpha = torch.nn.functional.softplus(self.alpha(features)) + self.eps
        beta = torch.nn.functional.softplus(self.beta(features)) + self.eps
        if self.goal_action_prior:
            goal_action = ((context[..., -3:] + 1.0) * 0.5).clamp(0.0, 1.0)
            alpha = alpha + self.goal_action_prior * goal_action
            beta = beta + self.goal_action_prior * (1.0 - goal_action)
        return alpha.clamp_max(100), beta.clamp_max(100)


class RecurrentCategoricalHead(nn.Module):
    """Score the exact integer displacements executed by the environment."""

    def __init__(
        self,
        input_features: int,
        action_displacements: tuple[tuple[int, int, int], ...],
    ):
        super().__init__()
        if not action_displacements:
            raise ValueError("action_displacements cannot be empty")
        self.logits = nn.Linear(input_features, len(action_displacements))

        # Give each Chebyshev step length equal initial probability mass.
        # Without this prior, the combinatorics of a cube put most uniform
        # categorical mass on the longest steps.
        step_lengths = torch.as_tensor(
            [max(abs(value) for value in action) for action in action_displacements],
            dtype=torch.long,
        )
        counts = torch.bincount(step_lengths)
        with torch.no_grad():
            self.logits.bias.copy_(-counts[step_lengths].float().log())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features)


class RecurrentFactorizedCategoricalHead(nn.Module):
    """Score each coordinate of an exact integer displacement separately."""

    def __init__(self, input_features: int, axis_action_count: int):
        super().__init__()
        if axis_action_count < 3 or axis_action_count % 2 == 0:
            raise ValueError("axis_action_count must be an odd integer of at least three")
        self.axis_action_count = axis_action_count
        self.logits = nn.Linear(input_features, 3 * axis_action_count)
        # Start close to a uniform joint policy while retaining deterministic
        # feature-dependent tie breaking for diagnostic mode rollouts.
        nn.init.normal_(self.logits.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.logits.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features).unflatten(-1, (3, self.axis_action_count))


class DiagonalS5Cell(nn.Module):
    """A compact recurrent diagonal state-space layer inspired by S5.

    This implements the defining S5-style multi-input, multi-output linear
    state-space recurrence with stable complex diagonal dynamics and bilinear
    discretization. It intentionally uses a sequential scan: rollout inference
    is O(1) in history length and short PPO subsequences remain practical on a
    16 GB GPU without a custom CUDA kernel.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_size: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_size = state_size

        # Negative real parts guarantee stable continuous dynamics. Frequencies
        # cover short and long oscillatory modes before learning.
        self.log_decay = nn.Parameter(torch.zeros(state_size))
        frequencies = math.pi * torch.arange(state_size, dtype=torch.float32)
        self.frequency = nn.Parameter(frequencies)
        self.log_dt = nn.Parameter(
            torch.empty(state_size).uniform_(math.log(dt_min), math.log(dt_max))
        )

        scale = 1.0 / math.sqrt(max(1, input_size))
        self.b_real = nn.Parameter(torch.randn(state_size, input_size) * scale)
        self.b_imag = nn.Parameter(torch.randn(state_size, input_size) * scale)
        self.c_real = nn.Parameter(torch.randn(hidden_size, state_size) * scale)
        self.c_imag = nn.Parameter(torch.randn(hidden_size, state_size) * scale)
        self.skip = nn.Linear(input_size, hidden_size, bias=False)
        self.gate = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def _discretized_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Bilinear discretization keeps stable continuous eigenvalues inside
        # the unit disk and behaves well for long recurrent rollouts.
        eigenvalues = torch.complex(
            -torch.nn.functional.softplus(self.log_decay) - 1e-4,
            self.frequency,
        )
        dt = self.log_dt.exp()
        denominator = 1.0 - 0.5 * dt * eigenvalues
        a_bar = (1.0 + 0.5 * dt * eigenvalues) / denominator
        b = torch.complex(self.b_real, self.b_imag)
        b_bar = (dt / denominator).unsqueeze(-1) * b
        c = torch.complex(self.c_real, self.c_imag)
        return a_bar, b_bar, c

    @staticmethod
    def _as_complex(state: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(state.float().contiguous())

    @staticmethod
    def _as_real(state: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(state)

    def _step(
        self,
        value: torch.Tensor,
        state: torch.Tensor,
        is_init: torch.Tensor,
        a_bar: torch.Tensor,
        b_bar: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reset = is_init.bool()
        while reset.dim() < state.dim():
            reset = reset.unsqueeze(-1)
        state = torch.where(reset, torch.zeros_like(state), state)
        state = a_bar * state + torch.einsum(
            "pi,bi->bp",
            b_bar,
            value.to(b_bar.dtype),
        )
        output = 2.0 * torch.einsum("hp,bp->bh", c, state).real
        output = output + self.skip(value)
        output = self.norm(output)
        output = torch.nn.functional.gelu(output) * torch.sigmoid(self.gate(value))
        return output, state

    def forward(
        self,
        value: torch.Tensor,
        state: torch.Tensor | None,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = value.dtype
        value = value.float()
        a_bar, b_bar, c = self._discretized_parameters()

        if not recurrent_mode():
            flat_value = value.reshape(-1, self.input_size)
            if state is None:
                flat_state = torch.zeros(
                    flat_value.shape[0],
                    self.state_size,
                    dtype=torch.complex64,
                    device=value.device,
                )
            else:
                flat_state = self._as_complex(
                    state.reshape(-1, self.state_size, 2)
                )
            flat_init = is_init.reshape(-1)
            output, next_state = self._step(
                flat_value,
                flat_state,
                flat_init,
                a_bar,
                b_bar,
                c,
            )
            output = output.reshape(*value.shape[:-1], self.hidden_size)
            next_state = self._as_real(next_state).reshape(
                *value.shape[:-1], self.state_size, 2
            )
            return output.to(original_dtype), next_state

        # Recurrent PPO supplies [batch, time, features], or [time, features]
        # for a single trajectory.
        squeezed_batch = value.dim() == 2
        if squeezed_batch:
            value = value.unsqueeze(0)
            is_init = is_init.unsqueeze(0)
            if state is not None:
                state = state.unsqueeze(0)

        batch_size, steps = value.shape[:2]
        if state is None:
            current_state = torch.zeros(
                batch_size,
                self.state_size,
                dtype=torch.complex64,
                device=value.device,
            )
        else:
            current_state = self._as_complex(state[:, 0])

        outputs = []
        states = []
        for step in range(steps):
            output, current_state = self._step(
                value[:, step],
                current_state,
                is_init[:, step].reshape(batch_size),
                a_bar,
                b_bar,
                c,
            )
            outputs.append(output)
            states.append(self._as_real(current_state))

        output_sequence = torch.stack(outputs, dim=1).to(original_dtype)
        state_sequence = torch.stack(states, dim=1)
        if squeezed_batch:
            output_sequence = output_sequence.squeeze(0)
            state_sequence = state_sequence.squeeze(0)
        return output_sequence, state_sequence


class S5TensorDictModule(TensorDictModule):
    """TensorDict wrapper that exposes the recurrent-state environment primer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_size: int,
    ):
        self.state_size = state_size
        super().__init__(
            module=DiagonalS5Cell(
                input_size=input_size,
                hidden_size=hidden_size,
                state_size=state_size,
            ),
            in_keys=["memory_input", "s5_state", "is_init"],
            out_keys=["features", ("next", "s5_state")],
        )

    def make_tensordict_primer(self) -> TensorDictPrimer:
        return TensorDictPrimer(
            {
                "s5_state": Unbounded(
                    shape=(self.state_size, 2),
                    dtype=torch.float32,
                )
            },
            expand_specs=True,
        )
