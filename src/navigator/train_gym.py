import numpy as np
import torch
import torch.optim as optim
from tensordict.nn import set_composite_lp_aggregate

# TorchRL components
from torchrl.collectors import (
    SyncDataCollector,
)
from torchrl.data import (
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)
from torchrl.envs import GymEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import (
    ClipPPOLoss,
)
from torchrl.objectives.value import GAE
from tqdm import tqdm

import wandb

# Your project components
from .config import Config
from .models.dummy_net import make_dummy_actor_critic  # Import the dummy network factory

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.capture_dynamic_output_shape_ops = True
set_composite_lp_aggregate(False).set()


def log_wandb(data: dict, **kwargs):
    """Log data to wandb."""
    if wandb is not None:
        wandb.log(data, **kwargs)
    else:
        print("WandB not initialized. Skipping logging.")


def train_gym_environment(config: Config):
    """
    Dummy function to train a simple Gym environment using TorchRL.
    This demonstrates the integration of GymEnv and a basic training loop.
    """
    print("Starting dummy Gym environment training...")

    device = torch.device(config.device)
    total_timesteps = 3_000_000  # A smaller number for a dummy run
    frames_per_batch = 2048
    batch_size = 256
    update_epochs = 5
    learning_rate = 1e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Ensure wandb is initialized if tracking is enabled
    if config.track_wandb and wandb is None:
        print("WandB not initialized. Please ensure it's set up in main.py if you want to track.")
        config.track_wandb = False  # Disable tracking for this run if not initialized

    # 1. Create GymEnv
    # Using 'CartPole-v1' as a simple example
    env = GymEnv("Acrobot-v1", device=device)
    print(f"Gym Environment created: {env.env_name}")
    print(f"Observation spec: {env.observation_spec}")
    print(f"Action spec: {env.action_spec}")

    # 2. Create dummy actor and critic networks
    print(env.specs)
    actor_module, value_module = make_dummy_actor_critic(env.observation_spec, env.action_spec)

    # Wrap actor_module with ProbabilisticActor for discrete actions
    actor_module = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        in_keys=["logits"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    # Wrap value_module with ValueOperator
    value_module = ValueOperator(module=value_module)

    actor_module.to(device)
    value_module.to(device)

    print(f"Dummy Actor parameters: {sum(p.numel() for p in actor_module.parameters())}")
    print(f"Dummy Value parameters: {sum(p.numel() for p in value_module.parameters())}")

    # 3. Setup Loss Function (PPO)
    loss_module = ClipPPOLoss(
        actor_network=actor_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_coef=ent_coef,
        entropy_bonus=bool(ent_coef),
        critic_coef=vf_coef,
        loss_critic_type="smooth_l1",
        normalize_advantage=True,
    )

    # 4. Optimizer
    optimizer = optim.AdamW(loss_module.parameters(), lr=learning_rate)

    # 5. Advantage Module (GAE)
    adv_module = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=value_module,
        average_gae=True,
    )

    # 6. Collector
    collector = SyncDataCollector(
        create_env_fn=lambda: GymEnv("Acrobot-v1", device=device),
        policy=actor_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_timesteps,
        device=device,
        storing_device=device,
        max_frames_per_traj=env.spec.max_episode_steps,
    )

    # 7. Training Loop
    pbar = tqdm(total=total_timesteps, desc="Dummy Gym Training", unit="steps")
    collected_frames = 0
    for i, batch_data in enumerate(collector):
        current_frames = batch_data.numel()
        pbar.update(current_frames)
        collected_frames += current_frames

        with torch.no_grad():
            adv_module(batch_data)

        actor_losses, critic_losses, entropy_losses = [], [], []
        actor_module.train()
        value_module.train()
        for _ in range(update_epochs):
            batch_data = batch_data.reshape(-1)  # Flatten for replay buffer
            # Create a dummy replay buffer for minibatches
            replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=frames_per_batch, device=device),
                batch_size=batch_size,
                sampler=SamplerWithoutReplacement(),
            )
            replay_buffer.extend(batch_data)

            for minibatch in replay_buffer:
                loss_dict = loss_module(minibatch)
                actor_loss = loss_dict["loss_objective"] + loss_dict["loss_entropy"]
                critic_loss = loss_dict["loss_critic"]

                # Combined loss
                loss = actor_loss + critic_loss

                optimizer.zero_grad()
                loss.backward()
                _grad = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(loss_dict["loss_entropy"].item())

        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        avg_entropy_loss = np.mean(entropy_losses)
        idxs = batch_data["next", "done"].nonzero()
        # avg_reward = batch_data["next", "reward"][idxs].mean().item()
        avg_reward = sum(1 for i in idxs)

        pbar.set_postfix({
            "R": f"{avg_reward:.2f}",
            "loss_A": f"{avg_actor_loss:.2f}",
            "loss_C": f"{avg_critic_loss:.2f}",
        })

        # Log to wandb
        if config.track_wandb and wandb is not None:
            log_data = {
                "gym_losses/policy_loss": avg_actor_loss,
                "gym_losses/value_loss": avg_critic_loss,
                "gym_losses/entropy": avg_entropy_loss,
                "gym_train/reward": avg_reward,
                "gym_charts/learning_rate": learning_rate,  # Static for this dummy example
                "gym_charts/collected_frames": collected_frames,
                "gym_charts/grad": _grad.item(),
            }
            log_wandb(log_data, step=collected_frames)

        if collected_frames >= total_timesteps:
            break

    collector.shutdown()
    env.close()
    print("Dummy Gym environment training finished.")
