# train.py (rewrite significantly)
import torch
import torch.optim as optim
import numpy as np
import wandb  # Assuming wandb is used
import os
from tqdm import tqdm

# Use IterableDataset concept
from torch.utils.data import DataLoader

# TorchRL components
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Your project components
from .config import Config
from .dataset import SmallBowelDataset  # Keep for creating the iterator

# Use the TorchRL environment wrapper and factory function
from .environment import make_sb_env, SmallBowelEnv


def log_wandb(data: dict, **kwargs):
    """Log data to wandb."""
    if wandb is not None:
        wandb.log(data, **kwargs)
    else:
        print("WandB not initialized. Skipping logging.")


# --- Dataset Iterator Helper ---
class SubjectIterator:
    """Wraps DataLoader to provide an infinite iterator for env resets."""

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: x[0],  # Get single subject dict
            pin_memory=False,  # Pin memory handled by TorchRL if needed
            persistent_workers=num_workers > 0,
        )
        self.iterator = iter(self.dataloader)
        self.dataset = dataset

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)  # Restart iterator
            return next(self.iterator)

    def __iter__(self):
        return self


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

    # Create a dedicated iterator for validation data
    val_iterator = SubjectIterator(
        val_dataset, shuffle=False, num_workers=0
    )  # No shuffle for validation
    # Create a validation environment instance
    # Pass the validation iterator to this env instance
    val_env = make_sb_env(config, val_iterator, device, 1)

    num_val_subjects = len(val_dataset)

    with (
        torch.no_grad(),
        set_exploration_type(ExplorationType.DETERMINISTIC),
    ):  # Use deterministic actions
        for i in tqdm(range(num_val_subjects), desc="Validation"):
            try:
                # Reset env (this will load the next subject from val_iterator)
                tensordict = val_env._reset(must_load_new_subject=True)
                rollout = val_env.rollout(
                    config.max_episode_steps, actor_module, auto_reset=False, tensordict=tensordict
                )

                reward = rollout["next", "reward"].mean().item()
                step_count = rollout["action"].shape[1]
                final_coverage = val_env._get_final_coverage().item()

                val_env.save_path()

            except Exception as e:
                reward, step_count, final_coverage = 0, 0, 0
                raise e
            finally:
                val_results["val_reward_sum"].append(reward)
                val_results["val_length"].append(step_count)
                val_results["val_coverage"].append(final_coverage)

    val_env.close()  # Close the validation environment

    # Calculate mean results
    final_metrics = {
        "validation/avg_reward": np.mean(val_results["val_reward_sum"]),
        "validation/avg_length": np.mean(val_results["val_length"]),
        "validation/avg_coverage": np.mean(val_results["val_coverage"]),
        "validation/num_subjects": num_val_subjects,
    }

    print(
        f"Validation Results: Avg R/L/C: {final_metrics['validation/avg_reward']:.3f} / "
        f"{final_metrics['validation/avg_length']:.1f} / {final_metrics['validation/avg_coverage']:.3f} "
    )
    return final_metrics


# --- Main Training Function ---
def train_torchrl(policy_module, value_module, config: Config, train_set: SmallBowelDataset, val_set: SmallBowelDataset, device: torch.device = None):
    """Main PPO training loop using TorchRL."""
    # --- Setup ---
    total_timesteps = getattr(config, "total_timesteps", 1_000_000)
    device = device or torch.device(config.device)
    
    train_iterator = SubjectIterator(
        train_set,
        shuffle=config.shuffle_dataset,
        num_workers=getattr(config, "num_workers", 0),
    )

    print(
        f"Total trainable parameters: {sum(p.numel() for p in policy_module.parameters()) + sum(p.numel() for p in value_module.parameters())}"
    )

    # --- Collector ---
    # Collects data by interacting policy_module with environment instances
    collector = SyncDataCollector(
        create_env_fn=lambda: make_sb_env(
            config, train_iterator, device, num_episodes_per_sample=config.num_episodes_per_sample
        ),  # Function to create environments
        policy=policy_module,  # Policy module to use for action selection
        # Total frames (steps) to collect in training
        total_frames=total_timesteps,
        # Number of frames collected in each rollout() call
        frames_per_batch=config.frames_per_batch,
        # No initial random exploration phase needed if policy handles exploration
        init_random_frames=-1,
        split_trajs=False,  # Process rollouts as single batch
        device=device,  # Device for collector ops (usually same as models/env)
        # Device where data is stored (can be CPU if memory is tight)
        storing_device=device,
        max_frames_per_traj=config.max_episode_steps,  # Max steps per episode trajectory
    )

    # --- Replay Buffer ---
    # replay_buffer = TensorDictReplayBuffer(
    #     storage=LazyTensorStorage(max_size=config.frames_per_batch, device=device),
    #     sampler=SamplerWithoutReplacement(),
    #     batch_size=config.batch_size,  # PPO minibatch size for sampling
    # )

    # --- Loss Function ---
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=config.clip_epsilon,
        entropy_coef=config.ent_coef,
        critic_coef=config.vf_coef,
        # value_loss_type="huber", # Or "mse"
        loss_critic_type="smooth_l1",  # TorchRL standard
        normalize_advantage=True,  # Recommended for PPO
    )

    # --- Advantage Module (GAE) ---
    adv_module = GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=value_module,  # Pass the value module instance
        average_gae=False,
    )

    # --- Optimizer ---
    optimizer = optim.AdamW(
        policy_module.parameters(),
        lr=config.learning_rate,
        eps=1e-4,  # PPO stability
    )
    optimizer_critic = optim.AdamW(
        value_module.parameters(),
        lr=config.learning_rate / 10,
        eps=1e-4,  # PPO stability
    )
    # Cosine annealing scheduler (optional)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(
            total_timesteps * config.update_epochs // config.batch_size
        ), 
        eta_min=1e-6,
    )
    scheduler_c = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_critic,
        T_max=(total_timesteps * config.update_epochs // config.batch_size), 
        eta_min=1e-6,
    )

    # --- Training Loop ---
    print(f"Starting training for {total_timesteps} total steps...")
    pbar = tqdm(total=total_timesteps)
    collected_frames = 0
    num_updates = 0
    # Use a specific metric like coverage or reward
    best_val_metric = float("-inf")

    # Use collector's iterator
    for i, batch_data in enumerate(collector):
        current_frames = batch_data.numel()  # Number of steps collected in this batch
        pbar.update(current_frames)
        collected_frames += current_frames

        # --- PPO Update Phase ---
        actor_losses, critic_losses, entropy_losses = [], [], []
        for _ in range(config.update_epochs):
            # Computes advantages and value targets (returns) in-place
            adv_module(batch_data)

            # Add collected data to the replay buffer
            batch_data = batch_data.reshape(-1)
            # replay_buffer.extend(batch_data)
            for i in range(
                0, config.frames_per_batch, config.batch_size
            ):  # Iterate over minibatches in the collected batch
                # minibatch = replay_buffer.sample()  # Sample a minibatch
                # minibatch = minibatch.squeeze(0)  # Remove batch dim
                # loss_dict = loss_module(minibatch)  # Calculate PPO losses
                minibatch = batch_data[i : i + config.batch_size]
                loss_dict = loss_module(minibatch)

                # Sum losses
                loss = (
                    loss_dict["loss_objective"]
                    + loss_dict["loss_critic"]
                    + loss_dict["loss_entropy"]
                )

                # Optimization step
                optimizer.zero_grad()
                optimizer_critic.zero_grad()

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), config.max_grad_norm
                )
                optimizer.step()
                optimizer_critic.step()

                scheduler.step()
                scheduler_c.step()

                # Log losses for this minibatch update
                actor_losses.append(loss_dict["loss_objective"].item())
                critic_losses.append(loss_dict["loss_critic"].item())
                entropy_losses.append(loss_dict["loss_entropy"].item())

            num_updates += 1  # Count PPO update cycles

            # --- Logging ---
            avg_actor_loss = np.mean(actor_losses)
            avg_critic_loss = np.mean(critic_losses)
            avg_entropy_loss = np.mean(entropy_losses)
            avg_reward = batch_data["next", "reward"].mean().item()
            # Log episode stats from collected batch_data
            log_data = {
                # "train/epoch": i,  # Or calculate epoch based on collected_frames
                # "train/collected_frames": collected_frames,
                "losses/policy_loss": avg_actor_loss,
                "losses/value_loss": avg_critic_loss,
                "losses/entropy": avg_entropy_loss,
                "losses/grad_norm": grad_norm.item(),
                "losses/std": batch_data["scale"].mean(),
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "train/num_updates": num_updates,
                "train/reward": avg_reward,
                "train/episode_len": batch_data["done"].float().mean().reciprocal().item(),
                "train/action_0": batch_data["action"][:,0].mean().item(),
                "train/action_1": batch_data["action"][:,1].mean().item(),
                "train/action_2": batch_data["action"][:,2].mean().item(),
            }

            pbar.set_postfix({
                "R": f"{avg_reward:.1f}",
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
                best_model_path = os.path.join(config.checkpoint_dir, "best_model_torchrl.pth")
                save_dict = {
                    # Save TorchRL modules' state_dicts
                    "policy_module_state_dict": policy_module.state_dict(),
                    "value_module_state_dict": value_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
