import torch
from torch.utils.data import Subset

import numpy as np
import os

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from navigator.config import Config, parse_args
from navigator.dataset import SmallBowelDataset
from navigator.utils import SubjectIterator
from navigator2.environment import SkrlSmallBowelEnv
from navigator2.models import ActorNetwork, CriticNetwork

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# load and wrap the Omniverse Isaac Gym environment
config = parse_args()
dataset = SmallBowelDataset(
    data_dir=config.data_dir,
    config=config,
)
print(f"Dataset loaded with {len(dataset)} samples.")

# --- Setup ---
os.makedirs(config.checkpoint_dir, exist_ok=True)

# --- Dataset Splitting and Iterators ---
train_size = int(len(dataset) * config.train_val_split)
indices = np.arange(len(dataset))
if config.shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[:train_size], indices[train_size:]

train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)
print(f"Val indices: {val_indices}, subjects: {[dataset.subjects[idx]['id'] for idx in val_indices]}")

iterator = SubjectIterator(
    dataset=train_set,
)

env = SkrlSmallBowelEnv(config=config, dataset_iterator=iterator)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = ActorNetwork(env.observation_space, env.action_space, device)
models["value"] = CriticNetwork(env.observation_space, env.action_space, device, input_channels=3)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 16 * 8192 / 32768
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 5e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.02}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape, "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": env.observation_space.shape, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/AllegroHand"

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()


# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# # download the trained agent's checkpoint from Hugging Face Hub and load it
# path = download_model_from_huggingface("skrl/OmniIsaacGymEnvs-AllegroHand-PPO", filename="agent.pt")
# agent.load(path)

# # start evaluation
# trainer.eval()
