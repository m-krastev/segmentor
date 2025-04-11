# train.py (rewrite significantly)
import torch
import torch.optim as optim
import numpy as np
import wandb  # Assuming wandb is used
import os
from tqdm import tqdm
# Use IterableDataset concept
from torch.utils.data import DataLoader, IterableDataset

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential

# TorchRL components
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Your project components
from .config import Config
from .dataset import SmallBowelDataset  # Keep for creating the iterator

# Use the TorchRL environment wrapper and factory function
from .environment import make_sb_env, SmallBowelEnv

# Use the TorchRL module creation function
from .models import create_ppo_modules


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
            print("\n[SubjectIterator] Restarting DataLoader...")
            self.iterator = iter(self.dataloader)  # Restart iterator
            return next(self.iterator)

    def __iter__(self):
        return self


# --- Validation Loop (Adaptation Needed) ---
def validation_loop_torchrl(
    env_factory,  # Pass a function to create validation env
    actor_module,  # Pass the trained policy module
    config: Config,
    device: torch.device,
    val_dataset: SmallBowelDataset,  # Pass the validation subset
):
    """Validation loop adapted for TorchRL env and modules."""
    print("\nRunning validation...")
    actor_module.eval()  # Set actor to evaluation mode
    val_results = {
        "val_reward_sum": [],
        "val_length": [],
        "val_coverage": [],
        "val_subject_id": [],
    }

    # Create a dedicated iterator for validation data
    val_iterator = SubjectIterator(
        val_dataset, shuffle=False, num_workers=0
    )  # No shuffle for validation
    # Create a validation environment instance
    # Pass the validation iterator to this env instance
    val_env = env_factory(config, val_iterator, device)

    num_val_subjects = len(val_dataset)

    with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):  # Use deterministic actions
        for i in tqdm(range(num_val_subjects), desc="Validation"):
            subject_id = "N/A"  # Need to get this from the env after reset potentially
            try:
                # Reset env (this will load the next subject from val_iterator)
                td_reset = val_env.reset()
                # Extract subject ID if stored during _load_next_subject
                if val_env._current_subject_data:
                    subject_id = val_env._current_subject_data.get("id", "N/A")

                done = td_reset.get("done").squeeze().item()
                terminated = td_reset.get("terminated").squeeze().item()

                # Check if reset failed (e.g., dataset exhausted prematurely or load error)
                if done or terminated:
                    print(
                        f"  Validation: Env reset returned done/terminated immediately for subj attempt {i + 1}. Skipping."
                    )
                    continue

                current_td = td_reset
                total_reward = 0.0
                step_count = 0
                episode_done = False
                final_coverage = 0.0  # Need to extract from info if available

                while not episode_done and step_count < config.max_episode_steps:
                    # Actor selects action based on current observation in TensorDict
                    # Get action distribution params + sample
                    current_td = actor_module(current_td)

                    # Step the environment
                    current_td = val_env.step(current_td)

                    total_reward += current_td.get(("next", "reward")).item()
                    step_count += 1
                    episode_done = current_td.get(("next", "done")).item()
                    # Retrieve coverage if info was added in _step (requires adding info keys)
                    # final_coverage = current_td.get(("next", "info_coverage"), torch.tensor(0.0)).item()

                # After episode ends (or max steps)
                # Get final coverage from the *base* environment state if possible
                final_coverage = val_env.base_env._get_final_coverage()

                val_results["val_reward_sum"].append(total_reward)
                val_results["val_length"].append(step_count)
                val_results["val_coverage"].append(final_coverage)
                val_results["val_subject_id"].append(subject_id)

            except Exception as e:
                print(
                    f"  Error during validation for subject {subject_id} (attempt {i + 1}): {e}")
                # Optionally record failure
                val_results["val_reward_sum"].append(0)
                val_results["val_length"].append(0)
                val_results["val_coverage"].append(0)
                val_results["val_subject_id"].append(f"{subject_id}_ERROR")

    val_env.close()  # Close the validation environment

    # Calculate mean results
    final_metrics = {
        "validation/avg_reward": np.mean(val_results["val_reward_sum"])
        if val_results["val_reward_sum"]
        else 0.0,
        "validation/avg_length": np.mean(val_results["val_length"])
        if val_results["val_length"]
        else 0.0,
        "validation/avg_coverage": np.mean(val_results["val_coverage"])
        if val_results["val_coverage"]
        else 0.0,
        "validation/num_success": len(val_results["val_reward_sum"]),
        "validation/num_subjects": num_val_subjects,
    }

    print(
        f"Validation Results: Avg R/L/C: {final_metrics['validation/avg_reward']:.3f} / "
        f"{final_metrics['validation/avg_length']:.1f} / {final_metrics['validation/avg_coverage']:.3f} "
        f"({final_metrics['validation/num_success']}/{final_metrics['validation/num_subjects']} successful)"
    )
    return final_metrics


# --- Main Training Function ---
def train_torchrl(config: Config, dataset: SmallBowelDataset):
    """Main PPO training loop using TorchRL."""
    # --- Setup ---
    device = torch.device(config.device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    total_timesteps = getattr(
        config, "total_timesteps", 1_000_000)  # Define total steps
    # Steps collected per rollout
    frames_per_batch = getattr(config, "frames_per_batch", 1024)
    # steps_to_collect in original code maps roughly to frames_per_batch
    # Size of replay buffer
    buffer_size = getattr(config, "buffer_size", frames_per_batch * 10)
    # PPO update minibatch size
    mini_batch_size = getattr(config, "batch_size", 64)
    ppo_update_epochs = getattr(
        config, "update_epochs", 10)  # PPO inner update loops

    # --- Dataset Splitting and Iterators ---
    train_size = int(len(dataset) * config.train_val_split)
    val_size = len(dataset) - train_size
    # Note: random_split might load all data, consider indices if dataset is huge
    indices = np.arange(len(dataset))
    if config.shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Use Subset if your dataset supports it, otherwise filter manually
    from torch.utils.data import Subset

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(
        f"Training: {len(train_set)} subjects, Validation: {len(val_set)} subjects.")
    train_iterator = SubjectIterator(
        train_set,
        shuffle=config.shuffle_dataset,
        num_workers=getattr(config, "num_workers", 0),
    )

    # --- Environment ---
    # Create an environment factory function needed for the collector
    def env_factory(): return make_sb_env(config, train_iterator, device)
    # Create a single environment instance for initialization checks (optional)
    # test_env = env_factory()
    # print("Env specs:", test_env.specs)
    # test_env.close()

    # --- Models ---
    policy_module, value_module = create_ppo_modules(config, device)
    print("Policy Module:", policy_module)
    print("Value Module:", value_module)
    print(
        f"Total parameters: {sum(p.numel() for p in policy_module.parameters()) + sum(p.numel() for p in value_module.parameters())}"
    )

    # --- Collector ---
    # Collects data by interacting policy_module with env_factory instances
    collector = SyncDataCollector(
        create_env_fn=env_factory,  # Function to create environments
        policy=policy_module,  # Policy module to use for action selection
        # Total frames (steps) to collect in training
        total_frames=total_timesteps,
        # Number of frames collected in each rollout() call
        frames_per_batch=frames_per_batch,
        # No initial random exploration phase needed if policy handles exploration
        init_random_frames=-1,
        split_trajs=False,  # Process rollouts as single batch
        device=device,  # Device for collector ops (usually same as models/env)
        # Device where data is stored (can be CPU if memory is tight)
        storing_device=device,
        max_frames_per_traj=config.max_episode_steps,  # Max steps per episode trajectory
    )

    # --- Replay Buffer ---
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size, device=device),
        batch_size=mini_batch_size,  # PPO minibatch size for sampling
    )

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
        # average_gae=False, # Keep as False usually
    )

    # --- Optimizer ---
    optimizer = optim.Adam(
        loss_module.parameters(),  # Get all parameters from the loss module (actor + critic)
        lr=config.learning_rate,
        eps=1e-5,  # PPO stability
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

        # --- Compute Advantages ---
        # GAE expects ["next"]["state_value"] and ["state_value"]
        # Ensure value_module runs on the 'next' observations if not already done by collector
        with torch.no_grad():
            # Run value module on the 'next' observations stored by the collector
            # Collector usually stores obs in batch_data["next"]["observation"] etc.
            # Need to ensure our "critic" obs is there.
            # Let's manually add the 'next' state value if collector didn't
            if ("next", "state_value") not in batch_data.keys(True, True):
                # Get observations for next step
                next_obs_td = batch_data.get("next").clone()
                value_module(next_obs_td)  # Calculate value for next step
                batch_data.set(("next", "state_value"),
                               next_obs_td.get("state_value"))

        # Computes advantages and value targets (returns) in-place
        adv_module(batch_data)

        # Reshape data for buffer (Time x Batch dims -> single Batch dim)
        batch_data_view = batch_data.reshape(-1)
        replay_buffer.extend(batch_data_view)  # Add collected data to buffer

        # --- PPO Update Phase ---
        if (
            collected_frames >= config.min_buffer_fill
        ):  # Start updates only after buffer has enough data
            actor_losses, critic_losses, entropy_losses = [], [], []
            for _ in range(ppo_update_epochs):
                for _ in range(
                    frames_per_batch // mini_batch_size
                ):  # Iterate over minibatches in the collected batch
                    minibatch = replay_buffer.sample()  # Sample a minibatch
                    loss_dict = loss_module(minibatch)  # Calculate PPO losses

                    # Sum losses
                    loss = (
                        loss_dict["loss_objective"]
                        + loss_dict["loss_critic"]
                        + loss_dict["loss_entropy"]
                    )

                    # Optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), config.max_grad_norm
                    )
                    optimizer.step()

                    # Log losses for this minibatch update
                    actor_losses.append(loss_dict["loss_objective"].item())
                    critic_losses.append(loss_dict["loss_critic"].item())
                    entropy_losses.append(loss_dict["loss_entropy"].item())

            num_updates += 1  # Count PPO update cycles

            # --- Logging ---
            avg_actor_loss = np.mean(actor_losses)
            avg_critic_loss = np.mean(critic_losses)
            avg_entropy_loss = np.mean(entropy_losses)

            # Log episode stats from collected batch_data
            ep_rewards = []
            ep_lengths = []
            ep_coverages = []  # Need to extract from info if stored
            # TorchRL collectors store episode stats under "next" -> "episode_reward" etc. when done
            # Check the batch_data structure after collection to confirm keys
            done_indices = batch_data.get(
                ("next", "done")).squeeze(-1).nonzero().squeeze(-1)
            if len(done_indices) > 0:
                ep_rewards = (
                    batch_data.get(("next", "episode_reward"))[
                        done_indices].cpu().numpy().tolist()
                )
                ep_lengths = (
                    batch_data.get(("next", "episode_length"))[
                        done_indices].cpu().numpy().tolist()
                )
                # ep_coverages = batch_data.get(("next", "info_coverage"))[done_indices].cpu().numpy().tolist() # If stored

            avg_ep_reward = np.mean(ep_rewards) if ep_rewards else np.nan
            avg_ep_length = np.mean(ep_lengths) if ep_lengths else np.nan
            avg_ep_coverage = np.mean(ep_coverages) if ep_coverages else np.nan

            log_data = {
                "train/epoch": i,  # Or calculate epoch based on collected_frames
                "train/collected_frames": collected_frames,
                "train/num_updates": num_updates,
                "rollout/avg_reward": avg_ep_reward,
                "rollout/avg_length": avg_ep_length,
                "rollout/avg_coverage": avg_ep_coverage,  # If available
                "losses/policy_loss": avg_actor_loss,
                "losses/value_loss": avg_critic_loss,
                "losses/entropy": avg_entropy_loss,
                "losses/grad_norm": grad_norm.item(),
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
            }

            pbar.set_postfix(
                {
                    "R": f"{avg_ep_reward:.1f}",
                    "L": f"{avg_ep_length:.0f}",
                    "loss_P": f"{avg_actor_loss:.2f}",
                    "loss_V": f"{avg_critic_loss:.2f}",
                }
            )

            if config.track_wandb and wandb is not None:
                log_wandb(log_data, step=collected_frames)

            # --- Validation and Checkpointing (Periodically) ---
            # Use num_updates or collected_frames to trigger validation/saving
            if num_updates % config.eval_interval == 0:
                # Run validation using the adapted loop
                val_metrics = validation_loop_torchrl(
                    env_factory=make_sb_env,  # Pass factory, not instance
                    actor_module=policy_module,
                    config=config,
                    device=device,
                    val_dataset=val_set,  # Pass the validation dataset view
                )
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
                        config.checkpoint_dir, "best_model_torchrl.pth")
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
    final_model_path = os.path.join(
        config.checkpoint_dir, "final_model_torchrl.pth")
    # save_dict = {...}  # Same structure as checkpoints
    torch.save(save_dict, final_model_path)
    print(f"Final model saved to {final_model_path}")
