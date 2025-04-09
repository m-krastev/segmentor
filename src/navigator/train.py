"""
Training logic for the Navigator PPO agent.
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
import os
from typing import Optional
from torch.utils.data import random_split, DataLoader, Subset
from websockets import Data

# Optional wandb import
try:
    import wandb
except ImportError:
    pass

from .config import Config
from .environment import SmallBowelEnv
from .models.actor import ActorNetwork
from .models.critic import CriticNetwork
from .dataset import SmallBowelDataset, load_subject_data


def train(
    config: Config,
    env: SmallBowelEnv,
    actor: ActorNetwork,
    critic: CriticNetwork,
    dataset: Optional[SmallBowelDataset] = None,
):
    """
    Main PPO training loop with epoch-based training and optional wandb logging.

    Args:
        config: Configuration object with hyperparameters
        env: Environment instance
        actor: Actor network
        critic: Critic network
        dataset: Optional dataset for multi-subject training
    """

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Initialize optimizer
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=config.learning_rate, eps=1e-5
    )
    print(f"Total number of parameters: {sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())}")

    # Determine max steps per episode
    steps_per_episode = min(config.steps_per_episode, config.max_episode_steps)

    # Prepare training subjects if using dataset
    train_set, val_set = random_split(
        dataset,
        [int(len(dataset) * config.train_val_split), len(dataset) - int(len(dataset) * config.train_val_split)],
    )

    val_set = Subset(val_set, [0])  # Use only the first subject for validation
    print(f"Training on {len(train_set)} subjects, validating on {len(val_set)} subjects.")

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=config.shuffle_dataset,
        num_workers=4,
    )

    # --- Storage setup ---
    obs_actor_shape = (steps_per_episode, 3, *config.patch_size_vox)
    obs_critic_shape = (steps_per_episode, 4, *config.patch_size_vox)
    action_shape = (steps_per_episode, 3)
    log_prob_shape = (steps_per_episode, 3)

    # Buffers for storing trajectory data
    obs_actor_buf = torch.zeros(obs_actor_shape, dtype=torch.float32).to(config.device)
    obs_critic_buf = torch.zeros(obs_critic_shape, dtype=torch.float32).to(config.device)
    actions_buf = torch.zeros(action_shape, dtype=torch.float32).to(config.device)
    log_probs_buf = torch.zeros(log_prob_shape, dtype=torch.float32).to(config.device)
    rewards_buf = torch.zeros(steps_per_episode, dtype=torch.float32).to(config.device)
    dones_buf = torch.zeros(steps_per_episode, dtype=torch.float32).to(config.device)
    values_buf = torch.zeros(steps_per_episode, dtype=torch.float32).to(config.device)

    # --- Epoch-based Training Loop ---
    print(f"Starting training for {config.num_epochs} epochs...")
    global_step = 0
    best_val_reward = float("-inf")

    # Store episode stats for logging
    ep_info_buffer = []  # Store episode info dictionaries

    for epoch in range(1, config.num_epochs + 1):
        # Reset the environment with a different subject if available
        for subject_data in train_loader:
            # Load subject data
            subject = load_subject_data(subject_data)

            # Update environment with new subject data
            env.update_data(
                image=subject["image"],
                seg=subject["seg"],
                duodenum=subject["duodenum"],
                colon=subject["colon"],
                gt_path=subject.get("gt_path", None),
            )

            epoch_rewards = []
            epoch_lengths = []
            epoch_coverages = []

            # Run multiple episodes per epoch
            for episode in range(1, config.episodes_per_epoch + 1):
                actor.eval()
                critic.eval()  # Set to eval mode for collection

                current_episode_reward = 0
                current_episode_length = 0

                # Reset environment at the start of each episode
                obs_dict = env.reset()
                next_obs_actor = obs_dict["actor"]
                next_obs_critic = obs_dict["critic"]
                next_done = torch.zeros(1, dtype=torch.float32).to(config.device)

                # Buffer index for this episode
                buffer_idx = 0

                # --- Collect trajectory for this episode ---
                pbar = tqdm(
                    range(steps_per_episode),
                    desc=f"Epoch {epoch}/{config.num_epochs}, Episode {episode}/{config.episodes_per_epoch}",
                    leave=False,
                )

                for step in pbar:
                    global_step += 1

                    if buffer_idx < steps_per_episode:
                        obs_actor_buf[buffer_idx] = next_obs_actor
                        obs_critic_buf[buffer_idx] = next_obs_critic
                        dones_buf[buffer_idx] = next_done.item()

                        # Sample actions
                        with torch.no_grad():
                            action_dist = actor.get_action_dist(next_obs_actor.unsqueeze(0))
                            normalized_action = action_dist.sample()
                            log_prob = action_dist.log_prob(normalized_action).sum(dim=-1)
                            value = critic(next_obs_critic.unsqueeze(0))

                        # Store values
                        actions_buf[buffer_idx] = normalized_action.squeeze(0)
                        log_probs_buf[buffer_idx] = action_dist.log_prob(normalized_action).squeeze(0)
                        values_buf[buffer_idx] = value.squeeze()

                        # Convert action to environment format
                        action_mapped = (2.0 * normalized_action.squeeze(0) - 1.0) * config.max_step_vox
                        action_vox_delta = tuple(torch.round(action_mapped).int().tolist())

                        # Step environment
                        obs_dict, reward, done, info = env.step(action_vox_delta)
                        rewards_buf[buffer_idx] = torch.tensor(reward, dtype=torch.float32).to(
                            config.device
                        )
                        next_obs_actor = obs_dict["actor"]
                        next_obs_critic = obs_dict["critic"]
                        next_done = torch.tensor([done], dtype=torch.float32).to(config.device)

                        buffer_idx += 1

                        # Track episode stats
                        current_episode_reward += reward
                        current_episode_length += 1

                    # If episode ended or max steps reached
                    if done or step == steps_per_episode - 1:
                        ep_coverage = info.get("episode_coverage", 0.0)  # Get coverage if available

                        # Record episode statistics
                        epoch_rewards.append(current_episode_reward)
                        epoch_lengths.append(current_episode_length)
                        epoch_coverages.append(ep_coverage)

                        # Update progress bar
                        pbar.set_postfix({
                            "Reward": f"{current_episode_reward:.2f}",
                            "Length": current_episode_length,
                            "Coverage": f"{ep_coverage:.3f}",
                        })

                        # Add to episode info buffer for logging
                        ep_info_buffer.append({
                            "reward": current_episode_reward,
                            "length": current_episode_length,
                            "coverage": ep_coverage,
                            "epoch": epoch,
                            "episode": episode,
                        })

                        # Keep buffer size manageable
                        if len(ep_info_buffer) > 100:
                            ep_info_buffer = ep_info_buffer[-100:]

                        # End this episode's collection
                        break

                # --- PPO Update once per episode ---
                # Calculate advantages (GAE)
                valid_indices = range(buffer_idx)
                if buffer_idx > 0:  # Only update if we collected some steps
                    actor.train()
                    critic.train()

                    # Valid buffer data
                    valid_obs_actor = obs_actor_buf[:buffer_idx]
                    valid_obs_critic = obs_critic_buf[:buffer_idx]
                    valid_actions = actions_buf[:buffer_idx]
                    valid_log_probs = log_probs_buf[:buffer_idx]
                    valid_rewards = rewards_buf[:buffer_idx]
                    valid_dones = dones_buf[:buffer_idx]
                    valid_values = values_buf[:buffer_idx]

                    # Calculate returns using GAE
                    advantages = torch.zeros_like(valid_rewards).to(config.device)
                    last_gae_lam = 0

                    with torch.no_grad():
                        if done:
                            next_value = 0.0  # Episode ended, value is 0
                        else:
                            next_value = critic(next_obs_critic.unsqueeze(0)).reshape(1, -1).item()

                    # Compute advantages and returns
                    for t in reversed(range(buffer_idx)):
                        if t == buffer_idx - 1:
                            nextnonterminal = 1.0 - done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - valid_dones[t + 1]
                            nextvalues = valid_values[t + 1]

                        delta = (
                            valid_rewards[t]
                            + config.gamma * nextvalues * nextnonterminal
                            - valid_values[t]
                        )
                        advantages[t] = last_gae_lam = (
                            delta + config.gamma * config.gae_lambda * nextnonterminal * last_gae_lam
                        )

                    returns = advantages + valid_values

                    # Store losses for logging
                    all_pg_loss, all_v_loss, all_ent_loss, all_total_loss = [], [], [], []

                    # Perform multiple PPO update epochs
                    for update_epoch in range(config.update_epochs):
                        # Shuffle indices
                        inds = np.arange(buffer_idx)
                        np.random.shuffle(inds)

                        # Mini-batch updates
                        for start in range(0, buffer_idx, config.batch_size):
                            end = min(start + config.batch_size, buffer_idx)
                            if end <= start:
                                continue

                            mb_inds = inds[start:end]

                            # Get mini-batch data
                            mb_obs_actor = valid_obs_actor[mb_inds]
                            mb_obs_critic = valid_obs_critic[mb_inds]
                            mb_actions = valid_actions[mb_inds]
                            mb_log_probs_old = valid_log_probs[mb_inds]
                            mb_advantages = advantages[mb_inds]
                            mb_returns = returns[mb_inds]

                            # Forward pass
                            new_action_dist = actor.get_action_dist(mb_obs_actor)
                            new_log_probs = new_action_dist.log_prob(mb_actions)
                            entropy = new_action_dist.entropy()
                            new_values = critic(mb_obs_critic)

                            # Calculate probability ratio
                            new_log_probs_sum = new_log_probs.sum(dim=-1)
                            mb_log_probs_old_sum = mb_log_probs_old.sum(dim=-1)
                            entropy_sum = entropy.sum(dim=-1)
                            logratio = new_log_probs_sum - mb_log_probs_old_sum
                            ratio = torch.exp(logratio)

                            # Normalize advantages
                            mb_advantages_norm = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8
                            )

                            # PPO clipped objective
                            pg_loss1 = -mb_advantages_norm * ratio
                            pg_loss2 = -mb_advantages_norm * torch.clamp(
                                ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon
                            )
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            new_values = new_values.view(-1)
                            value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

                            # Entropy loss
                            entropy_loss = entropy_sum.mean()

                            # Total loss
                            loss = (
                                pg_loss - config.ent_coef * entropy_loss + config.vf_coef * value_loss
                            )

                            # Gradient step
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                list(actor.parameters()) + list(critic.parameters()),
                                config.max_grad_norm,
                            )
                            optimizer.step()

                            # Store losses
                            all_pg_loss.append(pg_loss.item())
                            all_v_loss.append(value_loss.item())
                            all_ent_loss.append(entropy_loss.item())
                            all_total_loss.append(loss.item())

        # --- End of epoch: log metrics and save models ---
        if epoch_rewards:  # If we have completed episodes in this epoch
            avg_epoch_reward = np.mean(epoch_rewards)
            avg_epoch_length = np.mean(epoch_lengths)
            avg_epoch_coverage = np.mean(epoch_coverages)

            # Logging info
            log_data = {
                "epoch": epoch,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch/avg_reward": avg_epoch_reward,
                "epoch/avg_length": avg_epoch_length,
                "epoch/avg_coverage": avg_epoch_coverage,
                "global_step": global_step,
            }

            # Add loss metrics if available
            if all_pg_loss:
                log_data.update({
                    "losses/policy_loss": np.mean(all_pg_loss),
                    "losses/value_loss": np.mean(all_v_loss),
                    "losses/entropy": np.mean(all_ent_loss),
                    "losses/total_loss": np.mean(all_total_loss),
                })

            print(
                f"Epoch {epoch}/{config.num_epochs}: Avg Reward: {avg_epoch_reward:.3f}, "
                f"Avg Length: {avg_epoch_length:.1f}, Avg Coverage: {avg_epoch_coverage:.3f}"
            )

            if all_pg_loss:
                print(
                    f"  Losses(P/V/E): {np.mean(all_pg_loss):.3f}/{np.mean(all_v_loss):.3f}/"
                    f"{np.mean(all_ent_loss):.3f}"
                )

            # Log to wandb if enabled
            if config.track_wandb:
                log_wandb(log_data, step=global_step)

            # Validation loop
            if dataset:
                val_retdict = validation_loop(
                    env,
                    actor,
                    critic,
                    val_set,
                    config,
                )
                
                val_reward_sum = np.mean(val_retdict["val_reward_sum"])
                val_length = np.mean(val_retdict["val_length"])
                val_coverage = np.mean(val_retdict["val_coverage"])


                # Log validation metrics
                if config.track_wandb:
                    log_wandb(
                        {
                            "validation/avg_reward": val_reward_sum,
                            "validation/avg_length": val_length,
                            "validation/avg_coverage": val_coverage,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
    
                # Check if this is the best validation performance
                if val_reward_sum > best_val_reward:
                    best_val_reward = val_reward_sum

                    # Save best model
                    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
                    best_save_dict = {
                        "actor_state_dict": actor.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_reward": val_reward_sum,
                        "val_coverage": val_coverage,
                        "config": config,
                    }
                    torch.save(best_save_dict, best_model_path)
                    print(f"  New best model saved to {best_model_path}")

            # Save checkpoint every N epochs
            if epoch % 10 == 0 or epoch == config.num_epochs:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
                )
                checkpoint_dict = {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": config,
                }
                torch.save(checkpoint_dict, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                # Also save to the configured save path
                torch.save(checkpoint_dict, config.save_path)
                print(f"Model also saved to {config.save_path}")

    print("Training finished.")


def validation_loop(
    env: SmallBowelEnv,
    actor: ActorNetwork,
    critic: CriticNetwork,
    dataset: SmallBowelDataset,
    config: Config,
):
    """
    Validation loop for evaluating the trained model on a separate validation set.

    Args:
        env: Environment instance
        actor: Actor network
        critic: Critic network
        dataset: Dataset for validation
        config: Configuration object with hyperparameters
    """
    # Initialize validation subjects

    val_retdict = {
        "val_reward_sum": [],
        "val_length": [],
        "val_coverage": [],
    }

    val_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # Run validation episodes
    for subject in val_loader:
        # Load subject data
        subject_data = load_subject_data(subject)

        # Update environment with new subject data
        env.update_data(
            image=subject_data["image"],
            seg=subject_data["seg"],
            duodenum=subject_data["duodenum"],
            colon=subject_data["colon"],
            gt_path=subject_data.get("gt_path", None),
        )

        # Reset environment for validation
        obs_dict = env.reset()
        obs_actor = obs_dict["actor"]
        obs_critic = obs_dict["critic"]
        done = False

        total_reward = 0.0
        step_count = 0

        while not done and step_count < config.max_episode_steps:
            with torch.no_grad():
                action_dist = actor.get_action_dist(obs_actor.unsqueeze(0))
                normalized_action = action_dist.sample()
                action_mapped = (2.0 * normalized_action.squeeze(0) - 1.0) * config.max_step_vox
                action_vox_delta = tuple(torch.round(action_mapped).int().tolist())

                obs_dict, reward, done, info = env.step(action_vox_delta)
                obs_actor = obs_dict["actor"]
                total_reward += reward
                step_count += 1

        # Store validation results
        val_retdict["val_reward_sum"].append(total_reward)
        val_retdict["val_length"].append(step_count)
        val_retdict["val_coverage"].append(info.get("episode_coverage", 0.0))
        print(f"Validation Subject {subject['id']}: Reward: {total_reward:.2f}, Steps: {step_count}")

    return val_retdict 


def log_wandb(data: dict, **kwargs):
    """
    Log data to wandb.

    Args:
        data: Dictionary of data to log
    """
    try:
        if "wandb" in globals():
            wandb.log(data, **kwargs)
        else:
            print("Wandb not initialized. Skipping logging.")
    except Exception as e:
        print(f"Error logging to wandb: {e}")