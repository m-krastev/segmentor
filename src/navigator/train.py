# train.py (rewrite significantly)
import os

import numpy as np
import torch
import torch.optim as optim
from tensordict.nn import set_composite_lp_aggregate

# Use IterableDataset concept
from torch.utils.data import DataLoader

# TorchRL components
from torchrl.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.data import (
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import (
    ClipPPOLoss,
    HardUpdate,
    KLPENPPOLoss,
    SoftUpdate,
    TD3Loss,
)
from torchrl.objectives.value import GAE
from tqdm import tqdm

import wandb  # Assuming wandb is used

# Your project components
from .config import Config
from .dataset import SmallBowelDataset  # Keep for creating the iterator

# Use the TorchRL environment wrapper and factory function
from .environment import SmallBowelEnv, make_sb_env
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


# --- Validation Loop (Adaptation Needed) ---
def validation_loop_torchrl(
    actor_module,  # Pass the trained policy module
    config: Config,
    val_dataset: SmallBowelDataset,  # Pass the validation subset
    device: torch.device = None,
):
    """Validation loop adapted for TorchRL env and modules."""
    actor_module.eval()  # Set actor to evaluation mode
    val_results = {
        "val_reward_sum": [],
        "val_length": [],
        "val_coverage": [],
    }

    # Create a validation environment instance
    # Pass the validation iterator to this env instance
    val_env = make_sb_env(config, val_dataset, device, 1, check_env=False)

    num_val_subjects = len(val_dataset)

    with (
        torch.no_grad(),
        set_exploration_type(ExplorationType.MODE),
    ):  # Use deterministic actions
        for i in tqdm(range(num_val_subjects), desc="Validation"):
            # best of k
            # Store intermediate results
            paths = []
            path_masks = []
            intermediate_results = []
            reward, step_count, final_coverage = 0, 0, 0
            must_load_new_subject = True
            for _ in range(10):
                try:
                    # Reset the environment for the current subject
                    # This will load the new subject's data
                    tensordict = val_env._reset(
                        must_load_new_subject=must_load_new_subject
                    )
                    rollout = val_env.rollout(
                        config.max_episode_steps,
                        actor_module,
                        auto_reset=False,
                        tensordict=tensordict,
                    )
                    must_load_new_subject=False

                    reward = rollout["next", "reward"].mean().item()
                    step_count = rollout["action"].shape[1]
                    final_coverage = val_env._get_final_coverage().item()

                    paths.append(val_env.get_tracking_history())
                    path_masks.append(val_env.get_tracking_mask())
                    intermediate_results.append((reward, step_count, final_coverage))
                except Exception as e:
                    print(f"Error during validation rollout for subject {i}: {e}")

            # Choose the best result from the 10 rollouts
            if len(intermediate_results) == 0:
                print(
                    f"Too many errors caused no successful rollout to be generated. Skipping subject: {i}"
                )
                continue
            best_run = intermediate_results.index(
                max(intermediate_results, key=lambda x: x[0])
            )
            reward, step_count, final_coverage = intermediate_results[best_run]
            path = paths[best_run]
            path_mask = path_masks[best_run]

            # Save the best path and mask
            val_env.tracking_path_history = path
            val_env.cumulative_path_mask = path_mask
            val_env.save_path()

            val_results["val_reward_sum"].append(reward)
            val_results["val_length"].append(step_count)
            val_results["val_coverage"].append(final_coverage)

    val_env.close()  # Close the validation environment

    # Calculate mean results
    final_metrics = {
        "validation/avg_reward": np.mean(val_results["val_reward_sum"]),
        "validation/avg_length": np.mean(val_results["val_length"]),
        "validation/avg_coverage": np.mean(val_results["val_coverage"]),
    }

    print(
        f"Validation Results: Avg R/L/C: {final_metrics['validation/avg_reward']:.3f} / "
        f"{final_metrics['validation/avg_length']:.1f} / {final_metrics['validation/avg_coverage']:.3f} "
    )
    return final_metrics


# --- Main Training Function ---
def train_torchrl(
    policy_module,
    value_module,
    config: Config,
    train_set: SmallBowelDataset,
    val_set: SmallBowelDataset,
    device: torch.device = None,
    qnets: bool = False,
):
    """Main PPO training loop using TorchRL."""
    # --- Setup ---
    total_timesteps = getattr(config, "total_timesteps", 1_000_000)
    device = device or torch.device(config.device)
    batch_size = getattr(config, "batch_size", 32)

    print(
        f"Total trainable parameters: {sum(p.numel() for p in policy_module.parameters()) + sum(p.numel() for p in value_module.parameters())}"
    )

    # --- Loss Function ---
    # loss_module = KLPENPPOLoss(
    loss_module = (
        ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.ent_coef,
            entropy_bonus=bool(config.ent_coef),
            critic_coef=config.vf_coef,
            loss_critic_type="smooth_l1",  # TorchRL standard
            # loss_critic_type="l2",
            normalize_advantage=True,
        )
        if not qnets
        else TD3Loss(
            actor_network=policy_module,
            qvalue_network=value_module,
            bounds=(0, 1),
            num_qvalue_nets=2,
        )
    )

    if qnets:
        updater = SoftUpdate(loss_module, tau=0.1)
        loss_module.make_value_estimator(loss_module.value_type, gamma=config.gamma)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        # policy_module.parameters(),
        loss_module.parameters(),
        lr=config.learning_rate,
    )

    amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    scaler = torch.GradScaler(enabled=config.amp and amp_dtype==torch.float16)
    # Cosine annealing scheduler (optional)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(total_timesteps * config.update_epochs // batch_size),
        eta_min=5e-6,
    )
    # scheduler_c = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_critic,
    #     T_max=(total_timesteps * config.update_epochs // batch_size),
    #     eta_min=5e-6,
    # )
    collected_frames, num_updates = 0, 0

    # --- Checkpoint Reloading ---
    if config.reload_checkpoint_path:
        print(f"Loading checkpoint from {config.reload_checkpoint_path}")
        try:
            checkpoint = torch.load(config.reload_checkpoint_path, map_location=device)
            policy_module.load_state_dict(checkpoint["policy_module_state_dict"])
            value_module.load_state_dict(checkpoint["value_module_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            collected_frames = checkpoint.get("collected_frames", 0)
            num_updates = checkpoint.get("num_updates", 0)
            best_val_metric = checkpoint.get("best_val_metric", float("-inf"))
            print("Checkpoint loaded successfully.")
            print(
                f"Resuming training from collected_frames: {collected_frames}, num_updates: {num_updates}"
            )
        except FileNotFoundError:
            print(
                f"Error: Checkpoint file not found at {config.reload_checkpoint_path}"
            )
            # Decide how to handle this: exit, start fresh, etc.
            # For now, we'll print an error and continue without loading
            exit(1)
        except KeyError as e:
            print(f"Error loading checkpoint: Missing key {e}")
            # Handle missing keys if checkpoint structure changes
            exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while loading checkpoint: {e}")
            exit(1)

    # # Advance the scheduler to the correct state
    # for _ in range(num_updates):
    #     scheduler.step()

    # --- Collector ---
    # Collects data by interacting policy_module with environment instances
    env_maker = lambda: make_sb_env(
        config,
        train_set,
        device,
        num_episodes_per_sample=config.num_episodes_per_sample,
        check_env=False,
    )
    collector = SyncDataCollector(
        create_env_fn=env_maker,  # Function to create environments
        policy=policy_module,  # Policy module to use for action selection
        # Total frames (steps) to collect in training
        total_frames=total_timesteps - collected_frames,
        # Number of frames collected in each rollout() call
        frames_per_batch=config.frames_per_batch,
        # No initial random exploration phase needed if policy handles exploration
        init_random_frames=-1,
        split_trajs=False,  # Process rollouts as single batch
        device=device,  # Device for collector ops (usually same as models/env)
        # Device where data is stored (can be CPU if memory is tight)
        storing_device=device,
        max_frames_per_traj=config.max_episode_steps,  # Max steps per episode trajectory
        # num_threads=8
    )

    # --- Replay Buffer ---
    # replay_buffer = TensorDictReplayBuffer(
    #     storage=LazyTensorStorage(max_size=config.frames_per_batch, device=device),
    #     sampler=SamplerWithoutReplacement(),
    #     batch_size=batch_size,  # PPO minibatch size for sampling
    # )

    # --- Advantage Module (GAE) ---
    adv_module = GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=value_module,  # Pass the value module instance
        average_gae=True, # Standardize GAE
    )

    # --- Training Loop ---
    print(f"Starting training for {total_timesteps} total steps...")
    pbar = tqdm(
        total=total_timesteps, desc="Training", unit="steps", initial=collected_frames
    )
    # Use a specific metric like coverage or reward
    best_val_metric = float("-inf")
    # Use collector's iterator
    for i, batch_data in enumerate(collector, start=collected_frames):
        current_frames = batch_data.numel()  # Number of steps collected in this batch
        pbar.update(current_frames)
        collected_frames += current_frames

        # --- PPO Update Phase ---
        actor_losses, critic_losses, entropy_losses, kl_div = [], [], [], []
        for _ in range(config.update_epochs):
            batch_data = batch_data.reshape(-1)

            with (
                torch.no_grad(),
                torch.autocast(device.type, amp_dtype, enabled=config.amp),
            ):
                if not qnets:
                    adv_module(batch_data)

            for j in range(0, config.frames_per_batch, batch_size):
                minibatch = batch_data[j : j + batch_size]
                with torch.autocast(device.type, amp_dtype, enabled=config.amp):
                    loss_dict = loss_module(minibatch)

                    if qnets:
                        actor_loss = loss_dict["loss_actor"]
                        critic_loss = loss_dict["loss_qvalue"]
                    else:
                        actor_loss = loss_dict["loss_objective"] + loss_dict["loss_entropy"]
                        critic_loss = loss_dict["loss_critic"]

                optimizer.zero_grad()
                scaler.scale(actor_loss).backward()
                scaler.scale(critic_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()

                # Log losses for this minibatch update
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                entropy_losses.append(
                    loss_dict["loss_entropy"] if not qnets else torch.tensor(0.0)
                )
                kl_div.append(
                    loss_dict["kl_approx"] if not qnets else torch.tensor(0.0)
                )

            scheduler.step()
            # scheduler_c.step()
            if qnets:
                updater.step()
            num_updates += 1  # Count PPO update cycles

        # --- Logging ---
        avg_actor_loss = torch.stack(actor_losses).mean().item()
        avg_critic_loss = torch.stack(critic_losses).mean().item()
        avg_entropy_loss = torch.stack(entropy_losses).mean().item()
        avg_kldiv = torch.stack(kl_div).mean().item()
        avg_reward = batch_data["next", "reward"].mean().item()
        max_reward = batch_data["next", "reward"].max().item()
        idx = batch_data["next", "done"]
        # Log episode stats from collected batch_data
        final_coverage = batch_data["next", "info", "final_coverage"]
        final_coverage = final_coverage[idx].mean()
        step_count = batch_data["next", "info", "final_step_count"].float()
        step_count = step_count[idx].mean()
        ep_len = batch_data["next", "info", "final_length"].float()
        ep_len = ep_len[idx].mean()
        wall_gradient = batch_data["next", "info", "final_wall_gradient"].float()
        wall_gradient = wall_gradient[idx].mean()
        total_reward = batch_data["next", "info", "total_reward"]
        total_reward = total_reward[idx].mean()
        action = (
            (batch_data["action"] * 2 - 1) * config.max_step_vox
        ).round()
        max_gdt_achieved = batch_data["next", "info", "max_gdt_achieved"][idx]
        max_std, max_mean = torch.std_mean(max_gdt_achieved)
        log_data = {
            "losses/policy_loss": avg_actor_loss,
            "losses/value_loss": avg_critic_loss,
            "losses/entropy": avg_entropy_loss,
            "losses/kl_div": avg_kldiv,
            "losses/grad_norm": grad_norm,
            "losses/alpha": batch_data["alpha"].mean(),
            "losses/beta": batch_data["beta"].mean(),
            "train/reward": avg_reward,
            "train/max_reward": max_reward,
            "train/step_count": step_count,
            "train/wall_gradient": wall_gradient,
            "train/episode_len": ep_len,
            "train/final_coverage": final_coverage,
            "train/total_reward": total_reward,
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/max_gdt_achieved": max_mean,
            "charts/max_gdt_achieved_std": max_std,
            "charts/max_gdt_achieved_max": max_gdt_achieved.max(),
            "charts/num_updates": num_updates,
            "charts/action_0": action[:, 0].mean(),
            "charts/action_1": action[:, 1].mean(),
            "charts/action_2": action[:, 2].mean(),
            "charts/action_0_std": action[:, 0].std(),
            "charts/action_1_std": action[:, 1].std(),
            "charts/action_2_std": action[:, 2].std(),
            "charts/action_0_mode": action[:, 0].cpu().mode()[0],
            "charts/action_1_mode": action[:, 1].cpu().mode()[0],
            "charts/action_2_mode": action[:, 2].cpu().mode()[0],
        }

        pbar.set_postfix({
            "R": f"{avg_reward:.1f}",
            "Cov": f"{final_coverage:.1f}",
            "loss_P": f"{avg_actor_loss:.2f}",
            "loss_V": f"{avg_critic_loss:.2f}",
        })

        if config.track_wandb and wandb is not None:
            log_wandb(log_data, step=collected_frames)

        # --- Validation and Checkpointing (Periodically) ---
        # Use num_updates or collected_frames to trigger validation/saving
        if num_updates % config.eval_interval == 0:
            # Run validation using the adapted loop
            val_metrics = validation_loop_torchrl(
                actor_module=policy_module,
                config=config,
                val_dataset=val_set,
                device=device,
            )
            policy_module.train()
            if config.track_wandb and wandb is not None:
                log_wandb(val_metrics, step=collected_frames)

            # Checkpointing logic (save based on validation metric)
            current_metric = val_metrics.get(
                config.metric_to_optimize, float("-inf")
            )  # e.g., "validation/avg_coverage"
            is_best = current_metric > best_val_metric
            if is_best:
                best_val_metric = current_metric
                print(
                    f"  New best validation metric ({config.metric_to_optimize}): {best_val_metric:.4f}"
                )
                best_model_path = os.path.join(
                    config.checkpoint_dir, "best_model_torchrl.pth"
                )
                save_dict = {
                    # Save TorchRL modules' state_dicts
                    "policy_module_state_dict": policy_module.state_dict(),
                    "value_module_state_dict": value_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "collected_frames": collected_frames,
                    "num_updates": num_updates,
                    "best_val_metric": best_val_metric,
                    "config": vars(config),
                }
                torch.save(save_dict, best_model_path)
                print(f"  Best model saved to {best_model_path}")

        # Regular checkpoint saving
        if num_updates % config.save_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"checkpoint_torchrl_{collected_frames}.pth"
            )
            save_dict = {
                "policy_module_state_dict": policy_module.state_dict(),
                "value_module_state_dict": value_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "collected_frames": collected_frames,
                "num_updates": num_updates,
                "config": vars(config),
            }
            torch.save(save_dict, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # --- End of Training ---
    pbar.close()
    collector.shutdown()  # Clean up collector resources
    print("Training finished.")
    # Final save?
    final_model_path = os.path.join(config.checkpoint_dir, "final_model_torchrl.pth")
    # save_dict = {...}  # Same structure as checkpoints
    torch.save(save_dict, final_model_path)
    print(f"Final model saved to {final_model_path}")


def train_gym_environment(config: Config):
    """
    Dummy function to train a simple Gym environment using TorchRL.
    This demonstrates the integration of GymEnv and a basic training loop.
    """
    print("Starting dummy Gym environment training...")

    device = torch.device(config.device)
    total_timesteps = 3_000_000 # A smaller number for a dummy run
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
        config.track_wandb = False # Disable tracking for this run if not initialized

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
    value_module = ValueOperator(
        module=value_module
    )

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
            batch_data = batch_data.reshape(-1) # Flatten for replay buffer
            # Create a dummy replay buffer for minibatches
            replay_buffer = TensorDictReplayBuffer(
                storage=LazyTensorStorage(max_size=frames_per_batch, device=device),
                batch_size=batch_size, sampler=SamplerWithoutReplacement()
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
                "gym_charts/learning_rate": learning_rate, # Static for this dummy example
                "gym_charts/collected_frames": collected_frames,
                "gym_charts/grad": _grad.item(),
            }
            log_wandb(log_data, step=collected_frames)

        if collected_frames >= total_timesteps:
            break

    collector.shutdown()
    env.close()
    print("Dummy Gym environment training finished.")
