"""
Neural network models for the Navigator RL agent.
"""

from typing import Union
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn import InteractionType
from tensordict.nn.utils import biased_softplus
from torchrl.modules import (
    ActorValueOperator,
    GRUModule,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.data import Bounded, Categorical as CategoricalSpec
from torch.distributions import Beta, Categorical, Independent
from .actor import ActorNetwork
from .critic import CriticNetwork, StateActionValueNetwork
from .memory import (
    NavigatorVisualEncoder,
    RecurrentBetaHead,
    RecurrentCategoricalHead,
    RecurrentFactorizedCategoricalHead,
    S5TensorDictModule,
)
from ..config import Config

__all__ = [
    "ActorNetwork",
    "CriticNetwork",
    "FeasibleCategorical",
    "NavigatorVisualEncoder",
    "create_ppo_modules",
]

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


class IndependentCategorical(Independent):
    """A three-axis categorical distribution with a joint scalar log-probability."""

    def __init__(self, logits: torch.Tensor):
        super().__init__(Categorical(logits=logits), 1)


class FeasibleCategorical(Categorical):
    """Categorical distribution normalized over a state-dependent valid set.

    The mask is part of the environment state stored in every rollout. PPO
    therefore reconstructs exactly the same normalized distribution when it
    evaluates the old action under the updated policy.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
        if action_mask.shape != logits.shape:
            action_mask = torch.broadcast_to(action_mask, logits.shape)
        if not action_mask.any(dim=-1).all():
            raise ValueError("Every state must expose at least one feasible action")
        self.action_mask = action_mask
        super().__init__(logits=logits.masked_fill(~action_mask, -torch.inf))


# --- TorchRL Modules ---
def create_ppo_modules(
    config: Config,
    device: torch.device,
    qnets: bool = False,
    in_channels_actor: int | None = None,
    in_channels_critic: int | None = None,
):
    """Creates the PPO actor and critic modules compatible with TorchRL."""
    in_channels_actor = in_channels_actor or config.observation_channels
    in_channels_critic = in_channels_critic or config.observation_channels

    if config.memory_model != "none":
        if qnets:
            raise ValueError("Recurrent memory baselines currently support PPO only")
        if in_channels_actor != in_channels_critic:
            raise ValueError(
                "A shared recurrent encoder requires matching actor/critic channels"
            )
        return _create_recurrent_ppo_modules(
            config,
            device,
            input_channels=in_channels_actor,
        )

    # Actor Network Base
    actor_cnn_base = ActorNetwork(
        input_channels=in_channels_actor,
        context_features=config.context_features,
        goal_action_prior=config.goal_action_prior,
    ).to(device)

    # Wrap CNN base to extract "actor" obs and output "dist_params"
    actor_cnn_module = TensorDictModule(
        module=actor_cnn_base,
        in_keys=["actor", "context"],
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
        critic_base = StateActionValueNetwork(
            in_channels_critic,
            action_dim=3,
            context_features=config.context_features,
        ).to(device)
    else:
        # Critic Network Base
        critic_base = CriticNetwork(
            input_channels=in_channels_critic,
            context_features=config.context_features,
        ).to(device)

    # Wrap critic using ValueOperator
    value_module = ValueOperator(
        module=critic_base,
        in_keys=["actor", "context", "action"] if qnets else ["actor", "context"],
        out_keys=["state_action_value" if qnets else "state_value"],  # Standard output key for value estimates
    ).to(device)

    return policy_module, value_module


def _create_recurrent_ppo_modules(
    config: Config,
    device: torch.device,
    input_channels: int,
):
    """Create a shared-encoder recurrent actor and critic."""

    encoder = TensorDictModule(
        NavigatorVisualEncoder(
            input_channels=input_channels,
            context_features=config.context_features,
            output_features=config.memory_hidden_size,
        ),
        in_keys=["actor", "context"],
        out_keys=["memory_input"],
    )

    if config.memory_model == "gru":
        memory = GRUModule(
            input_size=config.memory_hidden_size,
            hidden_size=config.memory_hidden_size,
            num_layers=config.memory_num_layers,
            in_keys=["memory_input", "recurrent_state", "is_init"],
            out_keys=["features", ("next", "recurrent_state")],
            recurrent_backend=config.recurrent_backend,
            recurrent_compute_dtype=torch.float32,
            default_recurrent_mode=False,
            device=device,
        )
    elif config.memory_model == "s5":
        if config.memory_num_layers != 1:
            raise ValueError(
                "The dependency-free S5 baseline currently supports one layer"
            )
        memory = S5TensorDictModule(
            input_size=config.memory_hidden_size,
            hidden_size=config.memory_hidden_size,
            state_size=config.s5_state_size,
        )
    else:
        raise ValueError(f"Unsupported memory model: {config.memory_model}")

    common = TensorDictSequential(encoder, memory)
    if config.action_distribution == "beta":
        parameter_module = TensorDictModule(
            RecurrentBetaHead(
                input_features=config.memory_hidden_size,
                context_features=config.context_features,
                goal_action_prior=config.goal_action_prior,
            ),
            in_keys=["features", "context"],
            out_keys=["alpha", "beta"],
        )
        action_spec = Bounded(
            low=0,
            high=1,
            shape=torch.Size([ACTION_DIM]),
            dtype=torch.float32,
            device=device,
        )
        distribution_class = IndependentBeta
        distribution_in_keys = ["alpha", "beta"]
    elif config.action_distribution in {"categorical", "masked_categorical"}:
        parameter_module = TensorDictModule(
            RecurrentCategoricalHead(
                input_features=config.memory_hidden_size,
                action_displacements=config.action_displacements,
            ),
            in_keys=["features"],
            out_keys=["logits"],
        )
        action_spec = CategoricalSpec(
            n=config.categorical_action_count,
            shape=torch.Size([]),
            dtype=torch.int64,
            device=device,
        )
        if config.action_distribution == "masked_categorical":
            distribution_class = FeasibleCategorical
            distribution_in_keys = ["logits", "action_mask"]
        else:
            distribution_class = Categorical
            distribution_in_keys = ["logits"]
    else:
        parameter_module = TensorDictModule(
            RecurrentFactorizedCategoricalHead(
                input_features=config.memory_hidden_size,
                axis_action_count=config.factorized_axis_action_count,
            ),
            in_keys=["features"],
            out_keys=["logits"],
        )
        action_spec = Bounded(
            low=0,
            high=config.factorized_axis_action_count - 1,
            shape=torch.Size([ACTION_DIM]),
            dtype=torch.int64,
            device=device,
        )
        distribution_class = IndependentCategorical
        distribution_in_keys = ["logits"]
    policy_head = ProbabilisticActor(
        module=parameter_module,
        spec=action_spec,
        in_keys=distribution_in_keys,
        out_keys=["action"],
        distribution_class=distribution_class,
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM,
    )
    value_head = ValueOperator(
        module=torch.nn.Linear(config.memory_hidden_size, 1),
        in_keys=["features"],
        out_keys=["state_value"],
    )

    actor_value = ActorValueOperator(common, policy_head, value_head).to(device)
    policy_module = actor_value.get_policy_operator()
    value_module = actor_value.get_value_operator()
    policy_module.memory_model = config.memory_model
    value_module.memory_model = config.memory_model
    return policy_module, value_module
