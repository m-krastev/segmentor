import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Optional
from torch.utils.data import DataLoader, random_split, Subset
from .environment import SmallBowelEnv
from .models import ActorNetwork, CriticNetwork
from .config import Config
from .dataset import SmallBowelDataset
from torch import optim
from torch.nn import functional as F


# Validation loop and log_wandb remain the same as in the previous standard PPO version
# (Make sure validation_loop uses Subset and DataLoader correctly)
# ... (paste validation_loop and log_wandb functions here) ...
def validation_loop(
    env: SmallBowelEnv,
    actor: ActorNetwork,
    dataset: Subset,  # Use Subset type hint
    config: Config,
    device: torch.device,
):
    """Validation loop (no changes needed structurally, but relies on env methods)."""
    actor.eval()
    val_retdict = {
        "val_reward_sum": [],
        "val_length": [],
        "val_coverage": [],
        "val_subject_id": [],
    }
    # Create a DataLoader for the validation Subset
    val_loader = DataLoader(
        dataset,
        batch_size=1,  # Process one subject at a time
        shuffle=False,
        num_workers=getattr(config, "num_workers", 0),
        collate_fn=lambda x: x[0],  # Assumes dataset returns dicts
        pin_memory=True,  # Add pin_memory
        persistent_workers=getattr(config, "num_workers", 0) > 0,  # Add persistent_workers
    )

    # Store original env state if necessary (optional, depends on env)
    # original_env_state = env.get_state() # If env has such methods

    for subject_data in tqdm(val_loader, desc="Validation", leave=False):
        subject_id = subject_data.get("id", "N/A")
        try:
            # Update env with validation subject data
            env.update_data(
                image=subject_data["image"],
                seg=subject_data["seg"],
                duodenum=subject_data["duodenum"],
                colon=subject_data["colon"],
                gt_path=subject_data.get("gt_path", None),
                spacing=subject_data.get("spacing"),
                image_affine=subject_data.get("image_affine"),
                # Add any other necessary fields from subject_data
            )
            # Reset env for this specific subject
            obs_dict = env.reset()
            if not isinstance(obs_dict, dict) or "actor" not in obs_dict:
                raise TypeError(f"Invalid obs_dict after reset for subject {subject_id}")
            obs_actor = obs_dict["actor"].to(device)
            done = False
        except Exception as e:
            print(f"Error setting up validation subject {subject_id}: {e}")
            continue  # Skip this subject

        total_reward, step_count = 0.0, 0
        final_info = {}

        # Run one episode per validation subject
        while not done and step_count < config.max_episode_steps:
            with torch.no_grad():
                action_dist = actor.get_action_dist(obs_actor.unsqueeze(0))
                # Use mean for deterministic validation typically
                normalized_action = action_dist.mean
                # normalized_action = action_dist.sample() # Or sample if needed

                action_mapped = (2.0 * normalized_action.squeeze(0) - 1.0) * config.max_step_vox
                action_vox_delta = tuple(torch.round(action_mapped).int().cpu().tolist())

                try:
                    obs_dict, reward, done, info = env.step(action_vox_delta)
                    if not done and (not isinstance(obs_dict, dict) or "actor" not in obs_dict):
                        raise TypeError(f"Invalid obs_dict during step for subject {subject_id}")
                    if not done:  # Only update obs if not done
                        obs_actor = obs_dict["actor"].to(device)
                    total_reward += reward
                    step_count += 1
                    final_info = info  # Store last info dict
                except Exception as e:
                    print(
                        f"Error during validation step for subject {subject_id} at step {step_count}: {e}"
                    )
                    final_info = {"error": str(e)}
                    done = True  # End episode on error

        # Extract final info after the loop
        ep_info = final_info.get("episode_info", final_info)  # Check for nested or flat info
        val_retdict["val_reward_sum"].append(total_reward)
        val_retdict["val_length"].append(step_count)
        val_retdict["val_coverage"].append(
            ep_info.get("coverage", ep_info.get("episode_coverage", 0.0))
        )
        val_retdict["val_subject_id"].append(subject_id)

    # Restore original env state if necessary (optional)
    # env.set_state(original_env_state)

    return val_retdict


def log_wandb(data: Dict[str, Any], step: Optional[int] = None, **kwargs):
    """Log data to wandb (no changes needed)."""
    if wandb is not None and wandb.run is not None:
        # Filter out NaN values before logging
        loggable_data = {
            k: v for k, v in data.items() if not (isinstance(v, float) and np.isnan(v))
        }
        try:
            wandb.log(loggable_data, step=step, **kwargs)
        except Exception as e:
            print(f"Error logging to wandb: {e}")


def train(
    config: Config,
    env: SmallBowelEnv,
    actor: ActorNetwork,
    critic: CriticNetwork,
    dataset: SmallBowelDataset,
):
    """
    Main PPO training loop with epoch structure and manual subject switching.
    Collects a large buffer across subjects and updates when full.
    """
    # --- Setup ---
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = config.device
    actor.to(device)
    critic.to(device)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate,
        eps=1e-5,
    )
    print(
        f"Total parameters: {sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())}"
    )

    # --- Dataset Splitting ---
    train_size = int(len(dataset) * config.train_val_split)
    val_size = len(dataset) - train_size
    all_indices = list(range(len(dataset)))
    if config.shuffle_dataset:  # Shuffle indices once initially if needed for split
        np.random.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f"Training subjects: {len(train_set)}, Validation subjects: {len(val_set)}")

    # --- Rollout Buffer Storage (Size = Update Batch Size) ---
    buffer_size = config.steps_to_collect  # e.g., 2048
    obs_actor_shape = (buffer_size, 3, *config.patch_size_vox)
    obs_critic_shape = (buffer_size, 4, *config.patch_size_vox)
    action_shape = (buffer_size, 3)
    obs_actor_buf = torch.zeros(obs_actor_shape, dtype=torch.float32, device=device)
    obs_critic_buf = torch.zeros(obs_critic_shape, dtype=torch.float32, device=device)
    actions_buf = torch.zeros(action_shape, dtype=torch.float32, device=device)
    log_probs_buf = torch.zeros(buffer_size, dtype=torch.float32, device=device)
    rewards_buf = torch.zeros(buffer_size, dtype=torch.float32, device=device)
    dones_buf = torch.zeros(
        buffer_size, dtype=torch.float32, device=device
    )  # Stores done *before* step
    values_buf = torch.zeros(buffer_size, dtype=torch.float32, device=device)

    # --- Training Loop ---
    total_timesteps = getattr(
        config, "total_timesteps", float("inf")
    )  # Use total_timesteps or run for num_epochs
    num_epochs = (
        config.num_epochs if total_timesteps == float("inf") else 10000
    )  # Set a high epoch number if using total_timesteps
    print(f"Starting training for {num_epochs} epochs or ~{total_timesteps} steps...")

    global_step = 0
    num_updates = 0
    best_val_reward = float("-inf")

    # State variables - track the state *persistently* across updates
    current_obs_dict = None
    current_done = True  # Start as True to force initial subject load and reset
    buffer_idx = 0  # Current position in the buffer

    ep_info_buffer = []  # Stores recent episode infos across updates

    # --- Epoch Loop ---
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Starting Epoch {epoch} ---")
        epoch_ep_rewards, epoch_ep_lengths, epoch_ep_coverages = [], [], []
        epoch_start_step = global_step

        # Create DataLoader for this epoch (shuffles if configured)
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=config.shuffle_dataset,  # Shuffle subjects each epoch
            num_workers=getattr(config, "num_workers", 0),
            collate_fn=lambda x: x[0],
            pin_memory=True,
            persistent_workers=getattr(config, "num_workers", 0) > 0,
        )
        subject_iterator = iter(train_loader)
        subjects_processed_in_epoch = 0
        epoch_finished_naturally = False

        # --- Inner Loop (Processes subjects until epoch ends or buffer fills) ---
        # This loop continues as long as there are subjects left in the epoch iterator
        # OR until the total timestep limit is reached.
        while True:  # Loop handles subject loading and stepping
            # --- Load Next Subject If Necessary ---
            if current_done:
                try:
                    subject_data = next(subject_iterator)
                    subjects_processed_in_epoch += 1
                    subject_id = subject_data.get("id", "N/A")
                    print(
                        f"  Epoch {epoch} ({subjects_processed_in_epoch}/{len(train_loader)}): Loading subject {subject_id}..."
                    )

                    # Update environment with new subject data
                    env.update_data(
                        image=subject_data["image"],
                        seg=subject_data["seg"],
                        duodenum=subject_data["duodenum"],
                        colon=subject_data["colon"],
                        gt_path=subject_data.get("gt_path"),
                        spacing=subject_data.get("spacing"),
                        image_affine=subject_data.get("image_affine"),
                    )
                    current_obs_dict = env.reset()  # Reset env for the new subject
                    current_done = False  # Reset done flag
                    print(f"  Subject {subject_id} reset.")

                    if (
                        not isinstance(current_obs_dict, dict)
                        or "actor" not in current_obs_dict
                        or "critic" not in current_obs_dict
                    ):
                        print(
                            f"  Warning: Invalid observation dict after reset for subject {subject_id}. Skipping subject."
                        )
                        current_done = True  # Mark as done to trigger next subject load attempt
                        continue  # Skip to next iteration to reload

                except StopIteration:
                    # No more subjects left in this epoch's DataLoader
                    print(f"  Epoch {epoch}: All subjects processed.")
                    epoch_finished_naturally = True
                    break  # Exit the inner 'while True' loop for this epoch

                except Exception as e:
                    print(f"  Error loading/resetting subject {subject_id}: {e}. Skipping subject.")
                    # Try to recover by marking as done to load the *next* subject
                    current_done = True
                    continue  # Go to next iteration of 'while True' to attempt loading next subject

            # --- Check if environment state is valid ---
            if current_obs_dict is None:
                print(
                    "  Error: current_obs_dict is None but current_done is False. Forcing reload."
                )
                current_done = True  # Force reload attempt
                continue  # Go to next iteration of 'while True'

            # --- Collect ONE step ---
            actor.eval()  # Ensure eval mode for collection
            critic.eval()

            current_obs_actor = current_obs_dict["actor"].to(device)
            current_obs_critic = current_obs_dict["critic"].to(device)

            with torch.no_grad():
                action_dist = actor.get_action_dist(current_obs_actor.unsqueeze(0))
                normalized_action = action_dist.sample()
                log_prob = action_dist.log_prob(normalized_action).sum(dim=-1)
                value = critic(current_obs_critic.unsqueeze(0))

            # Store data in buffer at current index `buffer_idx`
            obs_actor_buf[buffer_idx] = current_obs_actor
            obs_critic_buf[buffer_idx] = current_obs_critic
            actions_buf[buffer_idx] = normalized_action.squeeze(0)
            log_probs_buf[buffer_idx] = log_prob.squeeze(0)
            values_buf[buffer_idx] = value.squeeze()
            dones_buf[buffer_idx] = float(current_done)  # Store done state *before* this step

            # --- Environment Step ---
            action_mapped = (2.0 * normalized_action.squeeze(0) - 1.0) * config.max_step_vox
            action_vox_delta = tuple(torch.round(action_mapped).int().cpu().tolist())

            try:
                next_obs_dict, reward, next_done, info = env.step(action_vox_delta)
                rewards_buf[buffer_idx] = reward  # Store reward for the step taken
            except Exception as e:
                print(
                    f"\n  Error during env.step for subject {subject_id} at step {global_step}: {e}. Treating as done."
                )
                rewards_buf[buffer_idx] = 0.0  # Assign default reward
                next_done = True
                next_obs_dict = None  # No valid next state
                info = {"error": str(e)}

            # --- Update State for Next Iteration ---
            # Important: Keep track of the state *after* the step for the *next* iteration's input
            # and for the potential GAE bootstrap value if the buffer fills now.
            obs_after_step = next_obs_dict
            done_after_step = next_done

            # Log episode info if episode finished *after* this step
            if done_after_step:
                ep_info = info.get("episode_info", {})
                ep_rew = ep_info.get("reward", info.get("episode_reward"))
                ep_len = ep_info.get("length", info.get("episode_length"))
                ep_cov = ep_info.get("coverage", info.get("episode_coverage", 0.0))

                # Simple fallback if info is missing
                if ep_rew is None:
                    ep_rew = reward  # Use last reward as proxy? Or track sum? Needs env support.
                if ep_len is None:
                    ep_len = info.get("step_count", "N/A")  # Needs env support

                completed_ep_info = {
                    "reward": ep_rew if ep_rew is not None else np.nan,
                    "length": ep_len if ep_len is not None else np.nan,
                    "coverage": ep_cov,
                    "epoch": epoch,
                    "global_step": global_step + 1,
                    "subject_id": subject_id,
                }
                # Only add if reward/length are valid
                if ep_rew is not None and ep_len is not None:
                    ep_info_buffer.append(completed_ep_info)
                    epoch_ep_rewards.append(ep_rew)
                    epoch_ep_lengths.append(ep_len)
                    epoch_ep_coverages.append(ep_cov)
                    if len(ep_info_buffer) > 100:
                        ep_info_buffer = ep_info_buffer[-100:]
                    print(
                        f"    Episode End (Subj {subject_id}): R={ep_rew:.2f}, L={ep_len}, C={ep_cov:.3f}"
                    )

            # --- Increment Pointers ---
            buffer_idx += 1
            global_step += 1

            # --- Check if Buffer is Full ---
            if buffer_idx == buffer_size:
                print(f"\n  Buffer full at step {global_step}. Performing PPO update...")
                num_updates += 1
                actor.train()
                critic.train()

                # Bootstrap value using the state *after* the last step collected
                with torch.no_grad():
                    if not done_after_step and obs_after_step is not None:
                        if "critic" in obs_after_step and isinstance(
                            obs_after_step["critic"], torch.Tensor
                        ):
                            next_value = (
                                critic(obs_after_step["critic"].to(device).unsqueeze(0))
                                .reshape(1, -1)
                                .item()
                            )
                        else:
                            print(
                                "  Warning: Critic observation missing/invalid after final buffer step. Using 0."
                            )
                            next_value = 0.0
                    else:  # Episode ended or obs is None/invalid after last step
                        next_value = 0.0

                # --- Calculate Advantages (GAE) ---
                advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)
                last_gae_lam = 0
                # Use dones_buf which stores done state *before* the step
                for t in reversed(range(buffer_size)):
                    if t == buffer_size - 1:
                        nextnonterminal = 1.0 - float(done_after_step)  # Use final done state
                        nextvalues = next_value
                    else:
                        # done state recorded *after* step t is dones_buf[t + 1]
                        nextnonterminal = 1.0 - dones_buf[t + 1]
                        nextvalues = values_buf[t + 1]
                    delta = (
                        rewards_buf[t] + config.gamma * nextvalues * nextnonterminal - values_buf[t]
                    )
                    advantages[t] = last_gae_lam = (
                        delta + config.gamma * config.gae_lambda * nextnonterminal * last_gae_lam
                    )
                returns = advantages + values_buf  # values_buf already has size buffer_size

                # --- Perform PPO Updates ---
                b_obs_actor = obs_actor_buf.reshape(-1, 3, *config.patch_size_vox)
                b_obs_critic = obs_critic_buf.reshape(-1, 4, *config.patch_size_vox)
                b_actions = actions_buf.reshape(-1, 3)
                b_log_probs_old = log_probs_buf.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)

                all_pg_loss, all_v_loss, all_ent_loss, all_approx_kl, all_clip_frac = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                inds = np.arange(buffer_size)
                for _ in range(config.update_epochs):
                    np.random.shuffle(inds)
                    for start in range(0, buffer_size, config.batch_size):
                        end = start + config.batch_size
                        mb_inds = inds[start:end]

                        # Slice data using mb_inds
                        mb_obs_actor = b_obs_actor[mb_inds]
                        mb_obs_critic = b_obs_critic[mb_inds]
                        mb_actions = b_actions[mb_inds]
                        mb_log_probs_old = b_log_probs_old[mb_inds]
                        mb_advantages = b_advantages[mb_inds]
                        mb_returns = b_returns[mb_inds]

                        # --- PPO Loss Calculation --- (Same as before)
                        new_action_dist = actor.get_action_dist(mb_obs_actor)
                        new_log_probs = new_action_dist.log_prob(mb_actions).sum(dim=-1)
                        entropy = new_action_dist.entropy().sum(dim=-1)
                        new_values = critic(mb_obs_critic).view(-1)

                        with torch.no_grad():
                            logratio = new_log_probs - mb_log_probs_old
                            ratio = torch.exp(logratio)
                            approx_kl = ((ratio - 1) - logratio).mean().item()
                            clip_frac = (
                                (torch.abs(ratio - 1.0) > config.clip_epsilon).float().mean().item()
                            )

                        mb_advantages_norm = mb_advantages
                        if mb_advantages.numel() > 1:
                            mb_advantages_norm = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8
                            )

                        pg_loss1 = -mb_advantages_norm * ratio
                        pg_loss2 = -mb_advantages_norm * torch.clamp(
                            ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        value_loss = 0.5 * F.mse_loss(new_values, mb_returns)
                        entropy_loss = entropy.mean()
                        loss = (
                            pg_loss - config.ent_coef * entropy_loss + config.vf_coef * value_loss
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(actor.parameters()) + list(critic.parameters()),
                            config.max_grad_norm,
                        )
                        optimizer.step()

                        all_pg_loss.append(pg_loss.item())
                        all_v_loss.append(value_loss.item())
                        all_ent_loss.append(entropy_loss.item())
                        all_approx_kl.append(approx_kl)
                        all_clip_frac.append(clip_frac)
                # --- End PPO Update Sub-Epochs ---

                # --- Logging after Update Cycle ---
                avg_pg_loss = np.mean(all_pg_loss) if all_pg_loss else 0
                avg_v_loss = np.mean(all_v_loss) if all_v_loss else 0
                avg_ent_loss = np.mean(all_ent_loss) if all_ent_loss else 0
                avg_approx_kl = np.mean(all_approx_kl) if all_approx_kl else 0
                avg_clip_frac = np.mean(all_clip_frac) if all_clip_frac else 0

                # Log recent episode stats (smoothed)
                if ep_info_buffer:
                    avg_recent_reward = np.mean([e["reward"] for e in ep_info_buffer])
                    avg_recent_length = np.mean([e["length"] for e in ep_info_buffer])
                    avg_recent_coverage = np.mean([e["coverage"] for e in ep_info_buffer])
                else:
                    avg_recent_reward, avg_recent_length, avg_recent_coverage = (
                        np.nan,
                        np.nan,
                        np.nan,
                    )

                log_data = {
                    "epoch": epoch,
                    "update": num_updates,
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/ep_reward_mean_recent": avg_recent_reward,
                    "charts/ep_length_mean_recent": avg_recent_length,
                    "charts/ep_coverage_mean_recent": avg_recent_coverage,
                    "losses/policy_loss": avg_pg_loss,
                    "losses/value_loss": avg_v_loss,
                    "losses/entropy": avg_ent_loss,
                    "losses/approx_kl": avg_approx_kl,
                    "losses/clip_fraction": avg_clip_frac,
                    "global_step": global_step,
                }
                print(f"  Update {num_updates} (Step {global_step}):")
                print(
                    f"    Recent Episodic Avg R/L/C: {avg_recent_reward:.2f} / {avg_recent_length:.1f} / {avg_recent_coverage:.3f}"
                )
                print(
                    f"    Losses(P/V/E): {avg_pg_loss:.3f} / {avg_v_loss:.3f} / {avg_ent_loss:.3f}"
                )
                print(f"    KL: {avg_approx_kl:.3f}, ClipFrac: {avg_clip_frac:.3f}")

                if config.track_wandb and wandb is not None:
                    log_wandb(log_data, step=global_step)

                # --- Reset Buffer Index ---
                buffer_idx = 0
                # --- Crucially, update the persistent state for the *next* collection step ---
                current_obs_dict = obs_after_step
                current_done = done_after_step

            # --- Check Timestep Limit ---
            if global_step >= total_timesteps:
                print(f"\nReached total_timesteps limit ({total_timesteps}) during epoch {epoch}.")
                break  # Exit the inner 'while True' loop

        # --- End of Inner 'while True' Loop (Epoch finished or total steps reached) ---

        # --- End of Epoch Actions ---
        print(f"\n--- Epoch {epoch} Finished ---")
        if epoch_ep_rewards:  # Log epoch summary if any episodes finished *within this epoch*
            avg_epoch_reward = np.mean(epoch_ep_rewards)
            avg_epoch_length = np.mean(epoch_ep_lengths)
            avg_epoch_coverage = np.mean(epoch_ep_coverages)
            print(f"  Avg Episodic Reward (Epoch {epoch}): {avg_epoch_reward:.3f}")
            print(f"  Avg Episodic Length (Epoch {epoch}): {avg_epoch_length:.1f}")
            print(f"  Avg Episodic Coverage (Epoch {epoch}): {avg_epoch_coverage:.3f}")
            if config.track_wandb and wandb is not None:
                log_wandb(
                    {
                        "epoch/avg_reward": avg_epoch_reward,
                        "epoch/avg_length": avg_epoch_length,
                        "epoch/avg_coverage": avg_epoch_coverage,
                        "epoch": epoch,
                    },
                    step=global_step,
                )
        else:
            print(f"  No complete episodes finished during Epoch {epoch}.")

        # --- Validation ---
        print("\nRunning validation...")
        val_retdict = validation_loop(
            env=env, actor=actor, dataset=val_set, config=config, device=device
        )
        val_reward_mean = np.mean(val_retdict["val_reward_sum"])
        val_length_mean = np.mean(val_retdict["val_length"])
        val_coverage_mean = np.mean(val_retdict["val_coverage"])
        print(
            f"Validation Results (Epoch {epoch}): Avg R/L/C: {val_reward_mean:.3f} / {val_length_mean:.1f} / {val_coverage_mean:.3f}"
        )
        if config.track_wandb and wandb is not None:
            log_wandb(
                {
                    "validation/avg_reward": val_reward_mean,
                    "validation/avg_length": val_length_mean,
                    "validation/avg_coverage": val_coverage_mean,
                    "epoch": epoch,
                },
                step=global_step,
            )

        # --- Checkpointing ---
        save_freq_epochs = getattr(config, "save_freq", 10)  # Save every N epochs
        is_best = val_reward_mean > best_val_reward
        if is_best:
            best_val_reward = val_reward_mean
            print(f"  New best validation reward: {best_val_reward:.3f}")
            best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            save_dict = {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "num_updates": num_updates,
                "best_val_reward": best_val_reward,
                "config": vars(config),
            }
            torch.save(save_dict, best_model_path)
            print(f"  Best model saved to {best_model_path}")

        if epoch % save_freq_epochs == 0 or epoch == num_epochs or global_step >= total_timesteps:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            save_dict = {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "num_updates": num_updates,
                "config": vars(config),
            }
            torch.save(save_dict, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            if (
                hasattr(config, "save_path")
                and config.save_path
                and config.save_path != checkpoint_path
            ):
                torch.save(save_dict, config.save_path)
                print(f"Checkpoint also saved to {config.save_path}")

        # --- Check Timestep Limit again after epoch actions ---
        if global_step >= total_timesteps:
            break  # Exit the outer epoch loop

    # --- End of Training ---
    print(f"\nTraining finished after {epoch} epochs, {global_step} steps, {num_updates} updates.")
