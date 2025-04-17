"""
Neural network models for the Navigator RL agent.
"""

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.data import Bounded
from torchrl.modules.distributions import TanhNormal  # Import TanhNormal directly

from .actor import ActorNetwork
from .critic import CriticNetwork
from ..config import Config

__all__ = ["ActorNetwork", "CriticNetwork", "create_ppo_modules"]

# Define action spec constants
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_DIM = 3  # Assuming 3D action space based on previous context


# --- TorchRL Modules ---
def create_ppo_modules(config: Config, device: torch.device):
    """Creates the PPO actor and critic modules compatible with TorchRL."""

    # Actor Network Base
    actor_cnn_base = ActorNetwork(
        config=config,
        input_channels=3,  # From environment spec["actor"]
        # Ensure ActorNetwork outputs 2 * ACTION_DIM features for loc and scale
    ).to(device)

    # Wrap CNN base to extract "actor" obs and output "dist_params"
    actor_cnn_module = TensorDictModule(
        module=actor_cnn_base,
        in_keys=["actor"],  # Input key from observation spec
        out_keys=["dist_params"],  # Intermediate output key
    )
    normal_param_extractor_module = TensorDictModule(
        module=NormalParamExtractor(),  # The nn.Module to wrap
        in_keys=["dist_params"],  # Key containing the raw parameters
        out_keys=["loc", "scale"],  # Keys for the split outputs
    )

    # Define the policy module using ProbabilisticActor
    policy_module = ProbabilisticActor(
        module=TensorDictSequential(
            actor_cnn_module,  # Outputs TD with "dist_params"
            normal_param_extractor_module,  # Takes TD, uses "dist_params", outputs TD with "loc", "scale"
        ),
        spec=Bounded(  # Action spec for the output distribution
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=torch.Size([ACTION_DIM]),  # Use constant ACTION_DIM
            dtype=torch.float32,
            device=device,  # Add device to spec
        ),
        in_keys=["loc", "scale"],  # Keys needed to create the distribution
        out_keys=["action"],  # Standard output key for sampled action
        distribution_class=TanhNormal,  # Bounded distribution
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    ).to(device)

    # Critic Network Base
    critic_base = CriticNetwork(
        config=config,
        input_channels=4,  # From environment spec["critic"]
    ).to(device)

    # Wrap critic using ValueOperator
    value_module = ValueOperator(
        module=critic_base,
        in_keys=["critic"],  # Input key from observation spec
        out_keys=["state_value"],  # Standard output key for value estimates
    ).to(device)

    return policy_module, value_module
