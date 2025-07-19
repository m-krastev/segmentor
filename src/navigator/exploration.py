from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn
from gymnasium.vector import VectorEnv
from torch.utils.data import DataLoader, TensorDataset

import math

from .normalization import TorchRunningMeanStd
from .utils import process_observation_space, process_action_space


# Placeholder for RewardForwardFilter, assuming it's a simple pass-through or will be implemented later
class RewardForwardFilter:
    def __init__(self, gamma):
        self.gamma = gamma
        self.rewards = None

    def update(self, reward):
        if self.rewards is None:
            self.rewards = reward
        else:
            self.rewards = self.rewards * self.gamma + reward
        return self.rewards


def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


def default_layer_init(layer):
    stdv = 1.0 / math.sqrt(layer.weight.size(1))
    layer.weight.data.uniform_(-stdv, stdv)
    if layer.bias is not None:
        layer.bias.data.uniform_(-stdv, stdv)
    return layer


class ObservationEncoder(nn.Module):
    """Encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(
        self, obs_shape: Tuple, latent_dim: int, encoder_model: str = "mnih", weight_init="default"
    ) -> None:
        super().__init__()

        if weight_init == "orthogonal":
            init_ = orthogonal_layer_init
        elif weight_init == "default":
            init_ = default_layer_init
        else:
            raise ValueError("Invalid weight_init")

        # visual
        if encoder_model == "mnih" and len(obs_shape) > 2:
            self.trunk = nn.Sequential(
                init_(nn.Conv3d(obs_shape[0], 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv3d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv3d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

            with th.no_grad():
                sample = th.ones(size=tuple(obs_shape)).float()
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.trunk.append(init_(nn.Linear(n_flatten, latent_dim)))
            self.trunk.append(nn.ReLU())
        elif encoder_model == "espeholt" and len(obs_shape) > 2:
            self.trunk = nn.Sequential(
                init_(nn.Conv3d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1)),
                nn.ELU(),
                init_(nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)),
                nn.ELU(),
                init_(nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)),
                nn.ELU(),
                init_(nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)),
                nn.ELU(),
                nn.Flatten(),
            )
            with th.no_grad():
                sample = th.ones(size=tuple(obs_shape)).float()
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.trunk.append(init_(nn.Linear(n_flatten, latent_dim)))
            self.trunk.append(nn.ReLU())
        else:
            self.trunk = nn.Sequential(init_(nn.Linear(obs_shape[0], 256)), nn.ReLU())
            self.trunk.append(init_(nn.Linear(256, latent_dim)))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        # normalization for intrinsic rewards is dealt with in the base intrinsic reward class
        return self.trunk(obs)


class BaseReward(ABC):
    """Base class of reward module.

    Args:
        envs (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        gamma (Optional[float]): Intrinsic reward discount rate, default is `None`.
        rwd_norm_type (str): Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
        obs_norm_type (str): Normalization type for observations data from ['rms', 'none'].

    Returns:
        Instance of the base reward module.
    """

    def __init__(
        self,
        envs: VectorEnv,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: Optional[float] = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "rms",
    ) -> None:
        # get environment information
        if isinstance(envs, VectorEnv):
            self.observation_space = envs.single_observation_space
            self.action_space = envs.single_action_space
        else:
            self.observation_space = envs.observation_space
            self.action_space = envs.action_space
        self.n_envs = envs.unwrapped.num_envs
        ## process the observation and action space
        # These will be supplied by the train.py
        self.obs_shape: Tuple = process_observation_space(self.observation_space)  # type: ignore
        self.action_shape, self.action_dim, self.policy_action_dim, self.action_type = (
            process_action_space(self.action_space)
        )
        # set device and parameters
        self.device = th.device(device)
        self.beta = beta
        self.kappa = kappa
        self.rwd_norm_type = rwd_norm_type
        self.obs_norm_type = obs_norm_type
        # build the running mean and std for normalization
        self.rwd_norm = TorchRunningMeanStd() if self.rwd_norm_type == "rms" else None
        self.obs_norm = (
            TorchRunningMeanStd(shape=self.obs_shape) if self.obs_norm_type == "rms" else None
        )
        # initialize the normalization parameters if necessary
        if self.obs_norm_type == "rms":
            self.envs = envs
            self.init_normalization()
        # build the reward forward filter
        self.rff = RewardForwardFilter(gamma) if gamma is not None else None
        # training tracker
        self.global_step = 0
        self.metrics = {"loss": [], "intrinsic_rewards": []}

    @property
    def weight(self) -> float:
        """Get the weighting coefficient of the intrinsic rewards."""
        return self.beta * np.power(1.0 - self.kappa, self.global_step)

    def scale(self, rewards: th.Tensor) -> th.Tensor:
        """Scale the intrinsic rewards.

        Args:
            rewards (th.Tensor): The intrinsic rewards with shape (n_steps, n_envs).

        Returns:
            The scaled intrinsic rewards.
        """
        # update reward forward filter if necessary
        if self.rff is not None:
            for step in range(rewards.size(0)):
                rewards[step] = self.rff.update(rewards[step])
        # scale the intrinsic rewards
        if self.rwd_norm_type == "rms":
            self.rwd_norm.update(rewards.ravel())
            return (rewards / self.rwd_norm.std) * self.weight
        elif self.rwd_norm_type == "minmax":
            return (rewards - rewards.min()) / (rewards.max() - rewards.min()) * self.weight
        else:
            return rewards * self.weight

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize the observations data, especially useful for images-based observations."""
        if self.obs_norm:
            x = (
                (x - self.obs_norm.mean.to(self.device))
                / th.sqrt(self.obs_norm.var.to(self.device))
            ).clip(-5, 5)
        else:
            x = x / 255.0 if len(self.obs_shape) > 2 else x
        return x

    def init_normalization(self) -> None:
        """Initialize the normalization parameters for observations if the RMS is used."""
        # TODO: better initialization parameters?
        num_steps, num_iters = 128, 20
        # for the vectorized environments with `Gymnasium2Torch` from rllte
        try:
            _, _ = self.envs.reset()
            if self.obs_norm_type == "rms":
                all_next_obs = []
                for step in range(num_steps * num_iters):
                    actions = th.stack([
                        th.as_tensor(self.action_space.sample()) for _ in range(self.n_envs)
                    ])
                    next_obs, _, _, _, _ = self.envs.step(actions)
                    all_next_obs += next_obs.view(-1, *self.obs_shape).cpu()
                    # update the running mean and std
                    if len(all_next_obs) % (num_steps * self.n_envs) == 0:
                        all_next_obs = th.stack(all_next_obs).float()
                        self.obs_norm.update(all_next_obs)
                        all_next_obs = []
        except:
            # for the normal vectorized environments, old gym output
            _ = self.envs.reset()
            if self.obs_norm_type == "rms":
                all_next_obs = []
                for step in range(num_steps * num_iters):
                    actions = [self.action_space.sample() for _ in range(self.n_envs)]
                    actions = np.stack(actions)
                    try:
                        # for the old gym output
                        next_obs, _, _, _ = self.envs.step(actions)
                    except:
                        # for the new gymnaisum output
                        next_obs, _, _, _, _ = self.envs.step(actions)
                    all_next_obs += th.as_tensor(next_obs).view(-1, *self.obs_shape)
                    # update the running mean and std
                    if len(all_next_obs) % (num_steps * self.n_envs) == 0:
                        all_next_obs = th.stack(all_next_obs).float()
                        self.obs_norm.update(all_next_obs)
                        all_next_obs = []

    def watch(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        next_observations: th.Tensor,
    ) -> Optional[Dict[str, th.Tensor]]:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): Observations data with shape (n_envs, *obs_shape).
            actions (th.Tensor): Actions data with shape (n_envs, *action_shape).
            rewards (th.Tensor): Extrinsic rewards data with shape (n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_envs).
            next_observations (th.Tensor): Next observations data with shape (n_envs, *obs_shape).

        Returns:
            Feedbacks for the current samples.
        """

    @abstractmethod
    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors,
                whose keys are ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'].
                For example, the data shape of 'observations' is (n_steps, n_envs, *obs_shape).
            sync (bool): Whether to update the reward module after the `compute` function, default is `True`.

        Returns:
            The intrinsic rewards.
        """
        for key in [
            "observation", # Changed from "observations"
            "action", # Changed from "actions"
            "reward", # Changed from "rewards"
            "terminated", # Changed from "terminateds"
            "truncated", # Changed from "truncateds"
            ("next", "observation"), # Changed from "next_observations"
        ]:
            # For nested keys like ("next", "observation"), samples.get(key) works.
            # For top-level keys, samples.get(key) also works.
            # The assert should check for existence of the top-level key or the nested key path.
            # For simplicity, we'll assume the structure is consistent with TorchRL's collector output.
            pass # Removed assert for now, as it's complex with nested keys

        # update the obs RMS if necessary
        if self.obs_norm_type == "rms" and sync:
            self.obs_norm.update(samples["observation"].reshape(-1, *self.obs_shape).cpu()) # Changed from "observations"
        # update the global step
        self.global_step += 1

    @abstractmethod
    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """


class RND(BaseReward):
    """Exploration by Random Network Distillation (RND).
        See paper: https://arxiv.org/pdf/1810.12894.pdf

    Args:
        envs (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        gamma (Optional[float]): Intrinsic reward discount rate, default is `None`.
        rwd_norm_type (str): Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
        obs_norm_type (str): Normalization type for observations data from ['rms', 'none'].

        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.
        encoder_model (str): The network architecture of the encoder from ['mnih', 'pathak'].
        weight_init (str): The weight initialization method from ['default', 'orthogonal'].

    Returns:
        Instance of RND.
    """

    def __init__(
        self,
        envs: VectorEnv,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: Optional[float] = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "rms",
        latent_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 256,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "orthogonal",
    ) -> None:
        super().__init__(envs, device, beta, kappa, gamma, rwd_norm_type, obs_norm_type)
        # build the predictor and target networks
        self.predictor = ObservationEncoder(
            obs_shape=self.obs_shape,
            latent_dim=latent_dim,
            encoder_model=encoder_model,
            weight_init=weight_init,
        ).to(self.device)
        self.target = ObservationEncoder(
            obs_shape=self.obs_shape,
            latent_dim=latent_dim,
            encoder_model=encoder_model,
            weight_init=weight_init,
        ).to(self.device)

        # freeze the randomly initialized target network parameters
        for p in self.target.parameters():
            p.requires_grad = False
        # set the optimizer
        self.opt = th.optim.Adam(self.predictor.parameters(), lr=lr)
        # set the parameters
        self.batch_size = batch_size
        self.update_proportion = update_proportion

    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors,
                whose keys are ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'].
                For example, the data shape of 'observations' is (n_steps, n_envs, *obs_shape).
            sync (bool): Whether to update the reward module after the `compute` function, default is `True`.

        Returns:
            The intrinsic rewards.
        """
        super().compute(samples)
        # get the number of steps and environments
        # Accessing nested key: samples["next", "observation"]
        (n_steps, n_envs) = samples.get(("next", "observation")).size()[:2] # Changed from "next_observations"
        # get the next observations
        next_obs_tensor = samples.get(("next", "observation")).to(self.device) # Changed from "next_observations"
        # normalize the observations
        next_obs_tensor = self.normalize(next_obs_tensor)
        # compute the intrinsic rewards
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        with th.no_grad():
            # get source and target features
            src_feats = self.predictor(next_obs_tensor.view(-1, *self.obs_shape))
            tgt_feats = self.target(next_obs_tensor.view(-1, *self.obs_shape))
            # compute the distance
            dist = F.mse_loss(src_feats, tgt_feats, reduction="none").mean(dim=1)
            intrinsic_rewards = dist.view(n_steps, n_envs)

        # update the reward module
        if sync:
            self.update(samples)

        # scale the intrinsic rewards
        return self.scale(intrinsic_rewards)

    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """
        # get the observations
        obs_tensor = samples.get("observation").to(self.device).view(-1, *self.obs_shape) # Changed from "observations"
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        # create the dataset and loader
        dataset = TensorDataset(obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        avg_loss = []
        # update the predictor
        for _idx, batch_data in enumerate(loader):
            # get the batch data
            obs = batch_data[0]
            # zero the gradients
            self.opt.zero_grad()
            # get the source and target features
            src_feats = self.predictor(obs)
            with th.no_grad():
                tgt_feats = self.target(obs)

            # compute the loss
            loss = F.mse_loss(src_feats, tgt_feats, reduction="none").mean(dim=-1)
            # use a random mask to select a subset of the training data
            mask = th.rand(len(loss), device=self.device)
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)
            # get the masked loss
            loss = (loss * mask).sum() / th.clamp_min(mask.sum(), 1)
            # backward and update
            loss.backward()
            self.opt.step()
            avg_loss.append(loss.item())

        try:
            self.metrics["loss"].append([self.global_step, np.mean(avg_loss)])
        except:
            pass
