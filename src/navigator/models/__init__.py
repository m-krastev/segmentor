"""
Neural network models for the Navigator RL agent.
"""

from typing import Union
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import InteractionType
from tensordict.nn.utils import biased_softplus
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.data import Bounded
from torchrl.modules.distributions import TanhNormal
from torch.distributions import Beta, Independent
from .actor import ActorNetwork
from .critic import CriticNetwork, StateActionValueNetwork
from ..config import Config

__all__ = ["ActorNetwork", "CriticNetwork", "create_ppo_modules"]

# Define action spec constants
ACTION_LOW = 0.0
ACTION_HIGH = 1.0
ACTION_DIM = 3  # Assuming 3D action space based on previous context


class BetaParamExtractor(torch.nn.Module):
    def __init__(self, parameter_mapping=None, **kwargs):
        super().__init__()
        match parameter_mapping:
            case "biased_softplus":
                self.parameter_mapping = biased_softplus(**kwargs)

            case _:
                self.parameter_mapping = lambda x: x

        self.kwargs = kwargs

    def forward(self, *tensors: torch.Tensor):
        tensor, *others = tensors
        param0, param1 = self.parameter_mapping(tensor).chunk(2, -1)
        return (param0, param1, *others)

def ind(alpha, beta):
    return Independent(
        Beta(alpha, beta),
        reinterpreted_batch_ndims=1,
    )


class IndependentBeta(Independent):
    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        min: Union[float, torch.Tensor] = 0.0,
        max: Union[float, torch.Tensor] = 1.0,
        event_dims: int = 1,
    ):
        self.min = torch.as_tensor(min, device=alpha.device).broadcast_to(alpha.shape)
        self.max = torch.as_tensor(max, device=alpha.device).broadcast_to(alpha.shape)
        self.scale = self.max - self.min
        self.eps = torch.finfo(alpha.dtype).eps
        base_dist = Beta(alpha, beta)
        super().__init__(base_dist, event_dims)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return super().sample(sample_shape) * self.scale + self.min

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        return super().rsample(sample_shape) * self.scale + self.min

    def log_prob(self, value: torch.Tensor):
        return super().log_prob(((value - self.min) / self.scale).clamp(self.eps, 1.0 - self.eps))

# --- TorchRL Modules ---
def create_ppo_modules(config: Config, device: torch.device, qnets: bool = False, in_channels_actor = 3, in_channels_critic = 3):
    """Creates the PPO actor and critic modules compatible with TorchRL."""

    # Actor Network Base
    actor_cnn_base = ActorNetwork(
        input_channels=in_channels_actor,
    ).to(device)

    # Wrap CNN base to extract "actor" obs and output "dist_params"
    actor_cnn_module = TensorDictModule(
        module=actor_cnn_base,
        in_keys=["actor", "aux", "mask"],  # Input key from observation spec
        # out_keys=["dist_params"],  # Intermediate output key
        out_keys=["alpha", "beta"],  # Intermediate output key
    )

    action_spec = Bounded(
        low=0, high=1, shape=torch.Size([ACTION_DIM]), dtype=torch.float32, device=device
    )

    # # Define the policy module using ProbabilisticActor
    # policy_module = ProbabilisticActor(
    #     module=TensorDictSequential(
    #         actor_cnn_module,  # Outputs TD with "dist_params"
    #         TensorDictModule(
    #             module=NormalParamExtractor(),  # The nn.Module to wrap
    #             in_keys=["dist_params"],  # Key containing the raw parameters
    #             out_keys=["loc", "scale"],  # Keys for the split outputs
    #         ),
    #     ),
    #     spec=action_spec,
    #     in_keys=["loc", "scale"],  # Keys needed to create the distribution
    #     out_keys=["action"],
    #     distribution_class=TanhNormal,
    #     distribution_kwargs=dict(low=-config.max_step_vox, high=config.max_step_vox),
    #     return_log_prob=True,
    # ).to(device)

    policy_module = ProbabilisticActor(
        module=TensorDictSequential(
            actor_cnn_module,  # Outputs TD with "alpha, beta"
        ),
        spec=action_spec,
        in_keys=["alpha", "beta"],
        out_keys=["action"],
        distribution_class=IndependentBeta,
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM,
    ).to(device)

    if qnets:
        critic_base = StateActionValueNetwork(in_channels_critic, 3).to(device)
    else:
        # Critic Network Base
        critic_base = CriticNetwork(
            input_channels=in_channels_critic,
        ).to(device)

    # Wrap critic using ValueOperator
    value_module = ValueOperator(
        module=critic_base,
        in_keys=["actor", "aux", "mask", "action"] if qnets else ["actor", "aux", "mask"],  # Input key from observation spec
        out_keys=["state_action_value" if qnets else "state_value"],  # Standard output key for value estimates
    ).to(device)

    return policy_module, value_module
