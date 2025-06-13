import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec

class DummyActor(nn.Module):
    def __init__(self, observation_spec: CompositeSpec, action_spec: DiscreteTensorSpec):
        super().__init__()
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        
        # Calculate input features based on observation_spec
        if "observation" in observation_spec.keys():
            # If it's a CompositeSpec with an "observation" key
            input_features = observation_spec["observation"].shape.numel()
        else:
            # Assume it's a simple TensorSpec (e.g., Box)
            input_features = observation_spec.shape.numel()
        
        # Determine output features based on action_spec
        # For discrete action spaces, use .space.n
        if isinstance(action_spec, DiscreteTensorSpec):
            output_features = action_spec.space.n
        # For continuous action spaces, use the last dimension of the shape
        elif isinstance(action_spec, UnboundedContinuousTensorSpec):
            output_features = action_spec.shape[-1]
        else:
            raise ValueError(f"Unsupported action_spec type: {type(action_spec)}")

        self.net = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_features),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Ensure the input is flattened if it's not already
        if observation.dim() > 1 and observation.shape[1] > 1:
            observation = observation.view(observation.shape[0], -1)
        return self.net(observation)

class DummyValue(nn.Module):
    def __init__(self, observation_spec: CompositeSpec):
        super().__init__()
        self.observation_spec = observation_spec
        
        # Calculate input features based on observation_spec
        if "observation" in observation_spec.keys():
            # If it's a CompositeSpec with an "observation" key
            input_features = observation_spec["observation"].shape.numel()
        else:
            # Assume it's a simple TensorSpec (e.g., Box)
            input_features = observation_spec.shape.numel()

        self.net = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1), # Output a single value for the value function
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Ensure the input is flattened if it's not already
        if observation.dim() > 1 and observation.shape[1] > 1:
            observation = observation.view(observation.shape[0], -1)
        return self.net(observation)

def make_dummy_actor_critic(env_specs):
    # Create actor and critic networks
    actor_net = DummyActor(env_specs.observation_spec, env_specs.action_spec)
    value_net = DummyValue(env_specs.observation_spec)

    # Wrap them in TensorDictModule for TorchRL compatibility
    # The actor takes 'observation' and outputs 'action'
    actor_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["logits"]
    )
    # The value network takes 'observation' and outputs 'state_value'
    value_module = TensorDictModule(
        value_net, in_keys=["observation"], out_keys=["state_value"]
    )
    return actor_module, value_module
