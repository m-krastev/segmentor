import os
import wandb
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Optional
from torch.utils.data import DataLoader, random_split
from .environment import SmallBowelEnv
from .models import ActorNetwork, CriticNetwork
from .config import Config
from .dataset import SmallBowelDataset
from torch import optim
from torch.nn import functional as F


def collect_rollout(
    # Core components
    env: SmallBowelEnv,
    actor: ActorNetwork,
    critic: CriticNetwork,
    config: Config,
    device: torch.device,
    # Buffers to fill
    obs_actor_buf: torch.Tensor,
    obs_critic_buf: torch.Tensor,
    actions_buf: torch.Tensor,
    log_probs_buf: torch.Tensor,
    rewards_buf: torch.Tensor,
    dones_buf: torch.Tensor,
    values_buf: torch.Tensor,
    # State carry-over (input) - state *within* the current subject's episode
    current_obs_dict: Dict[str, torch.Tensor],  # Must be provided, not optional
    current_done: bool,  # Done state *before* starting this rollout segment
    # Global state to update
    global_step: int,
    pbar: tqdm,
    current_epoch: int,
) -> Tuple[int, Optional[Dict[str, torch.Tensor]], bool, List[Dict[str, Any]], int]:
    """
    Collects a rollout of experience within the environment's current data context.
    Assumes env is already updated with the desired subject data by the caller.
    Handles resets internally if an episode finishes during the rollout.

    Args:
        env: The environment instance (already configured with subject data).
        actor: The actor network (should be in eval mode).
        critic: The critic network (should be in eval mode).
        config: Configuration object.
        device: Torch device.
        obs_actor_buf, ...: Data buffers to store the trajectory.
        current_obs_dict: Observation from the previous step/reset.
        current_done: Done flag from the previous step/reset.
        global_step: Current global step counter.
        pbar: tqdm progress bar instance.
        current_epoch: The current training epoch number.

    Returns:
        Tuple containing:
        - steps_collected: Number of steps actually collected in this rollout.
        - obs_after_rollout: The observation after the last step.
        - done_after_rollout: The done flag after the last step.
        - ep_infos: List of dictionaries containing info about completed episodes during this rollout.
        - updated_global_step: The incremented global step counter.
    """
    ep_infos = []
    steps_collected = 0
    obs_dict = current_obs_dict  # Use local variable
    done = current_done  # Use local variable

    actor.eval()  # Ensure networks are in eval mode for collection
    critic.eval()

    for step in range(config.num_steps):
        # --- Handle Episode Termination (Reset within the *same* subject) ---
        if done:
            # Log info from the completed episode if available
            if obs_dict is not None and "episode_info" in obs_dict:
                ep_infos.append(obs_dict["episode_info"])
                # Clear the info once logged to prevent re-logging if reset fails
                del obs_dict["episode_info"]

            # Reset environment (uses the currently loaded subject data)
            try:
                obs_dict = env.reset()
                if (
                    not isinstance(obs_dict, dict)
                    or "actor" not in obs_dict
                    or "critic" not in obs_dict
                ):
                    raise TypeError(
                        f"env.reset() must return a dict with 'actor' and 'critic' keys. Got: {obs_dict}"
                    )
                done = False  # Reset done flag
            except Exception as e:
                print(f"\nError resetting environment during rollout: {e}. Stopping rollout.")
                # Return the state *before* the failed reset
                # Pass back done=True to signal the issue to the main loop
                return steps_collected, obs_dict, True, ep_infos, global_step

        # --- Agent Interaction ---
        # Ensure obs_dict is valid before proceeding
        if obs_dict is None:
            print("\nError: obs_dict is None before agent interaction. Stopping rollout.")
            return (
                steps_collected,
                obs_dict,
                True,
                ep_infos,
                global_step,
            )  # Signal error state

        pbar.update(1)
        global_step += 1
        steps_collected += 1

        current_obs_actor = obs_dict["actor"].to(device)
        current_obs_critic = obs_dict["critic"].to(device)

        with torch.no_grad():
            action_dist = actor.get_action_dist(current_obs_actor.unsqueeze(0))
            normalized_action = action_dist.sample()
            log_prob = action_dist.log_prob(normalized_action).sum(dim=-1)
            value = critic(current_obs_critic.unsqueeze(0))

        # Store data in buffer (use steps_collected-1 as index)
        buffer_idx = steps_collected - 1
        obs_actor_buf[buffer_idx] = current_obs_actor
        obs_critic_buf[buffer_idx] = current_obs_critic
        actions_buf[buffer_idx] = normalized_action.squeeze(0)
        log_probs_buf[buffer_idx] = log_prob.squeeze(0)
        values_buf[buffer_idx] = value.squeeze()
        dones_buf[buffer_idx] = done

        # --- Environment Step ---
        action_mapped = (2.0 * normalized_action.squeeze(0) - 1.0) * config.max_step_vox
        action_vox_delta = tuple(torch.round(action_mapped).int().cpu().tolist())

        next_obs_dict, reward, current_done, info = env.step(action_vox_delta)
        rewards_buf[buffer_idx] = reward

        # Prepare for next iteration
        obs_dict = next_obs_dict  # Update observation
        done = current_done  # Update done flag

        # Store episode info if episode finished *after* this step
        if done:
            info = info if isinstance(info, dict) else {}
            # Store in obs_dict temporarily to be picked up at start of next loop iteration or end of rollout
            obs_dict["episode_info"] = {
                "reward": info.get("episode_reward", reward),
                "length": info.get(
                    "episode_length", locals().get("step_count", 0) + 1
                ),  # Track step_count if needed
                "coverage": info.get("episode_coverage", 0.0),
                "epoch": current_epoch,
                "global_step": global_step,
                # Add subject ID here if needed, passed from train loop? No, env should know.
                # "subject_id": env.current_subject_id # If env tracks this
            }
            # NOTE: We don't signal subject exhaustion here anymore. The train loop manages subjects.

    # --- After loop ---
    # Check if the last step resulted in 'done' and add info if not already added
    if done and obs_dict is not None and "episode_info" in obs_dict:
        ep_infos.append(obs_dict["episode_info"])
        # Don't delete here, the main loop needs the final obs_dict state

    # Return the state *after* the loop finishes
    return steps_collected, obs_dict, done, ep_infos, global_step


def train(
    config: Config,
    env: SmallBowelEnv,
    actor: ActorNetwork,
    critic: CriticNetwork,
    dataset: SmallBowelDataset,
):
    """
    Main PPO training loop. Manages subject iteration and calls collect_rollout.
    """
    # --- Setup ---
    # (Same as before: optimizer, device, checkpoint dir, network setup)
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
    # (Same as before: random_split, DataLoader setup)
    if not hasattr(config, "train_val_split"):
        config.train_val_split = 0.8
    train_size = int(len(dataset) * config.train_val_split)
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError("Dataset too small for split")
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Training: {len(train_set)} subjects, Validation: {len(val_set)} subjects.")
    if not hasattr(config, "shuffle_dataset"):
        config.shuffle_dataset = True
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=config.shuffle_dataset,
        num_workers=getattr(config, "num_workers", 0),
        collate_fn=lambda x: x[0],
        pin_memory=True,
        persistent_workers=getattr(config, "num_workers", 0) > 0,
    )

    # --- Rollout Buffer Storage ---
    # (Same as before: buffer initialization)
    if not hasattr(config, "num_steps"):
        config.num_steps = 2048
    obs_actor_shape = (config.num_steps, 3, *config.patch_size_vox)
    obs_critic_shape = (config.num_steps, 4, *config.patch_size_vox)
    action_shape = (config.num_steps, 3)
    obs_actor_buf = torch.zeros(obs_actor_shape, dtype=torch.float32, device=device)
    obs_critic_buf = torch.zeros(obs_critic_shape, dtype=torch.float32, device=device)
    actions_buf = torch.zeros(action_shape, dtype=torch.float32, device=device)
    log_probs_buf = torch.zeros(config.num_steps, dtype=torch.float32, device=device)
    rewards_buf = torch.zeros(config.num_steps, dtype=torch.float32, device=device)
    dones_buf = torch.zeros(config.num_steps, dtype=torch.float32, device=device)
    values_buf = torch.zeros(config.num_steps, dtype=torch.float32, device=device)

    # --- Training Loop ---
    print(f"Starting training for {config.num_epochs} epochs...")
    global_step = 0
    num_updates = 0
    best_val_reward = float("-inf")

    # State variables - track the state *between* rollouts
    # These are reset when a *new subject* is loaded.
    current_obs_dict = None
    current_done = True  # Start as True to force initial reset/load

    ep_info_buffer = []  # Stores recent episode infos across updates

    total_timesteps = getattr(config, "total_timesteps", config.num_epochs * len(train_set) * 1000)
    pbar = tqdm(total=total_timesteps, desc="Total Steps")

    # --- Epoch Loop ---
    for epoch in range(1, config.num_epochs + 1):
        epoch_ep_rewards, epoch_ep_lengths, epoch_ep_coverages = [], [], []

        # --- Subject Loop (Iterate through DataLoader) ---
        for subject_data in tqdm(train_loader, desc="Training on patient data"):
            subject_id = subject_data.get("id")
            print(f"\nEpoch {epoch}, Subject {subject_id}: Loading data...")

            # 1. Update Environment with Subject Data
            try:
                env.update_data(
                    image=subject_data["image"],
                    seg=subject_data["seg"],
                    duodenum=subject_data["duodenum"],
                    colon=subject_data["colon"],
                    gt_path=subject_data.get("gt_path"),
                    spacing=subject_data.get("spacing"),
                    image_affine=subject_data.get("image_affine"),
                )
            except Exception as e:
                print(f"  Error updating env data for subject {subject_id}: {e}. Skipping subject.")
                continue  # Skip to the next subject

            current_obs_dict = env.reset()  # Get initial observation
            current_done = False  # Reset done flag for the start of the subject
            print(f"  Subject {subject_id}: Environment reset.")

            # --- Rollout Collection and Update Loop (for this subject) ---
            # Decide how many updates/rollouts per subject.
            # Simplest: Collect one rollout (config.num_steps) per subject load.
            # More complex: Loop until subject is 'done' or step limit reached.
            # Let's do ONE rollout per subject load for simplicity now.
            # If global_step exceeds total_timesteps, we'll break later.

            # 3. Collect Rollout
            (
                steps_collected,
                obs_after_rollout,
                done_after_rollout,
                current_rollout_ep_infos,
                global_step,
            ) = collect_rollout(
                env=env,
                actor=actor,
                critic=critic,
                config=config,
                device=device,
                obs_actor_buf=obs_actor_buf,
                obs_critic_buf=obs_critic_buf,
                actions_buf=actions_buf,
                log_probs_buf=log_probs_buf,
                rewards_buf=rewards_buf,
                dones_buf=dones_buf,
                values_buf=values_buf,
                current_obs_dict=current_obs_dict,  # Pass current state
                current_done=current_done,  # Pass current done status
                global_step=global_step,
                pbar=pbar,
                current_epoch=epoch,
            )

            # Update persistent state for the *next* potential rollout (if we were looping within subject)
            # Since we are doing one rollout per subject load now, these might seem less critical,
            # but they are essential for the GAE calculation's next_value.
            current_obs_dict = obs_after_rollout
            current_done = done_after_rollout

            # Log collected episode info
            ep_info_buffer.extend(current_rollout_ep_infos)
            if len(ep_info_buffer) > 100:
                ep_info_buffer = ep_info_buffer[-100:]
            epoch_ep_rewards.extend([e["reward"] for e in current_rollout_ep_infos])
            epoch_ep_lengths.extend([e["length"] for e in current_rollout_ep_infos])
            epoch_ep_coverages.extend([e["coverage"] for e in current_rollout_ep_infos])

            # 4. PPO Update Phase (if steps were collected)
            if steps_collected == 0:
                print(
                    f"  Warning: collect_rollout returned 0 steps for subject {subject_id}. Skipping update."
                )
                # Reset done flag to True to force potential reload/reset on next iteration if needed
                current_done = True
                continue  # Skip update, move to next subject

            num_updates += 1
            actor.train()
            critic.train()

            # Bootstrap value using the state *after* the rollout
            with torch.no_grad():
                if not current_done and current_obs_dict is not None:
                    if "critic" in current_obs_dict and isinstance(
                        current_obs_dict["critic"], torch.Tensor
                    ):
                        next_value = (
                            critic(current_obs_dict["critic"].to(device).unsqueeze(0))
                            .reshape(1, -1)
                            .item()
                        )
                    else:
                        print(
                            "\nWarning: Critic observation missing/invalid after rollout for bootstrapping. Using 0."
                        )
                        next_value = 0.0
                else:  # Episode ended or obs is None
                    next_value = 0.0

            # --- Calculate Advantages (GAE) ---
            # (Calculation logic is the same, uses buffers filled by collect_rollout
            # and the next_value calculated above)
            advantages = torch.zeros(steps_collected, dtype=torch.float32, device=device)
            last_gae_lam = 0
            for t in reversed(range(steps_collected)):
                if t == steps_collected - 1:
                    nextnonterminal = 1.0 - float(current_done)  # Use done state *after* last step
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]  # done state after step t
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + config.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = last_gae_lam = (
                    delta + config.gamma * config.gae_lambda * nextnonterminal * last_gae_lam
                )
            returns = advantages + values_buf[:steps_collected]

            # --- Perform PPO Updates ---
            # (Update logic using mini-batches is the same)
            # ... (copy the PPO update loop here) ...
            b_obs_actor = obs_actor_buf[:steps_collected].reshape(-1, 3, *config.patch_size_vox)
            b_obs_critic = obs_critic_buf[:steps_collected].reshape(-1, 4, *config.patch_size_vox)
            b_actions = actions_buf[:steps_collected].reshape(-1, 3)
            b_log_probs_old = log_probs_buf[:steps_collected].reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            all_pg_loss, all_v_loss, all_ent_loss, all_approx_kl, all_clip_frac = (
                [],
                [],
                [],
                [],
                [],
            )

            inds = np.arange(steps_collected)
            for _ in range(config.update_epochs):  # Renamed update_epoch to _
                np.random.shuffle(inds)
                for start in range(0, steps_collected, config.batch_size):
                    end = start + config.batch_size
                    mb_inds = inds[start:end]

                    mb_obs_actor = b_obs_actor[mb_inds]
                    mb_obs_critic = b_obs_critic[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_log_probs_old = b_log_probs_old[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns = b_returns[mb_inds]

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

                    loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * value_loss

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

            # --- Logging after Update Cycle ---
            # (Logging logic is the same)
            # ... (copy logging logic here) ...
            avg_pg_loss = np.mean(all_pg_loss)
            avg_v_loss = np.mean(all_v_loss)
            avg_ent_loss = np.mean(all_ent_loss)
            avg_approx_kl = np.mean(all_approx_kl)
            avg_clip_frac = np.mean(all_clip_frac)

            if current_rollout_ep_infos:
                avg_rollout_reward = np.mean([e["reward"] for e in current_rollout_ep_infos])
                avg_rollout_length = np.mean([e["length"] for e in current_rollout_ep_infos])
                avg_rollout_coverage = np.mean([e["coverage"] for e in current_rollout_ep_infos])
            else:
                avg_rollout_reward, avg_rollout_length, avg_rollout_coverage = (
                    np.nan,
                    np.nan,
                    np.nan,
                )

            log_data = {
                "epoch": epoch,
                "update": num_updates,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "rollout/avg_reward": avg_rollout_reward,
                "rollout/avg_length": avg_rollout_length,
                "rollout/avg_coverage": avg_rollout_coverage,
                "losses/policy_loss": avg_pg_loss,
                "losses/value_loss": avg_v_loss,
                "losses/entropy": avg_ent_loss,
                "losses/approx_kl": avg_approx_kl,
                "losses/clip_fraction": avg_clip_frac,
                "global_step": global_step,
            }

            print(
                f"\nUpdate {num_updates} (Epoch {epoch}, Step {global_step}, Subject {subject_id}):"
            )
            print(
                f"  Rollout Episodic Avg R/L/C: {avg_rollout_reward:.2f} / {avg_rollout_length:.1f} / {avg_rollout_coverage:.3f}"
            )
            print(f"  Losses(P/V/E): {avg_pg_loss:.3f} / {avg_v_loss:.3f} / {avg_ent_loss:.3f}")
            print(f"  KL: {avg_approx_kl:.3f}, ClipFrac: {avg_clip_frac:.3f}")

            if config.track_wandb and wandb is not None:
                log_wandb(log_data, step=global_step)

            # Check for total timesteps completion *after* update and logging
            if global_step >= total_timesteps:
                print("\nReached total_timesteps limit.")
                break  # Break subject loop

        # --- End of Subject Loop ---
        if global_step >= total_timesteps:
            print(f"Stopping epoch {epoch} early due to reaching total_timesteps.")
            # Perform end-of-epoch actions like validation even if stopped early
            # break # Break epoch loop (optional, could let validation run)

        # --- End of Epoch: Validation and Checkpointing ---
        # (Validation and Checkpointing logic remains the same)
        # ... (validation_loop call, logging, checkpointing) ...
        if epoch_ep_rewards:  # Log epoch summary if any episodes finished
            avg_epoch_reward = np.mean(epoch_ep_rewards)
            avg_epoch_length = np.mean(epoch_ep_lengths)
            avg_epoch_coverage = np.mean(epoch_ep_coverages)
            print(f"\n--- Epoch {epoch} Summary ---")
            print(f"  Avg Episodic Reward: {avg_epoch_reward:.3f}")
            print(f"  Avg Episodic Length: {avg_epoch_length:.1f}")
            print(f"  Avg Episodic Coverage: {avg_epoch_coverage:.3f}")
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

        if not hasattr(config, "save_freq"):
            config.save_freq = 10
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
                "best_val_reward": best_val_reward,
                "config": vars(config),
            }
            torch.save(save_dict, best_model_path)
            print(f"  Best model saved to {best_model_path}")

        if epoch % config.save_freq == 0 or epoch == config.num_epochs:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            save_dict = {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
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

        # Break epoch loop if total steps reached
        if global_step >= total_timesteps:
            break

    # --- End of Training ---
    pbar.close()
    print("Training finished.")


# --- Validation Loop and Wandb Logging (Keep as separate functions) ---
def validation_loop(
    env: SmallBowelEnv,
    actor: ActorNetwork,
    dataset: SmallBowelDataset,
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
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=getattr(config, "num_workers", 0),
        collate_fn=lambda x: x[0],
    )

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
            )
            # Reset env for this subject
            obs_dict = env.reset()
            if not isinstance(obs_dict, dict) or "actor" not in obs_dict:
                raise TypeError("Invalid obs_dict")
            obs_actor = obs_dict["actor"].to(device)
        except Exception as e:
            print(f"Error setting up validation subject {subject_id}: {e}")
            continue  # Skip this subject

        total_reward, step_count = 0.0, 0
        final_info = {}
        done = False

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
                    if not isinstance(obs_dict, dict) or "actor" not in obs_dict:
                        raise TypeError("Invalid obs_dict")
                    obs_actor = obs_dict["actor"].to(device)
                    total_reward += reward
                    step_count += 1
                    final_info = info
                except Exception as e:
                    print(
                        f"Error during validation step for subject {subject_id} at step {step_count}: {e}"
                    )
                    final_info = {"error": str(e)}

        val_retdict["val_reward_sum"].append(total_reward)
        val_retdict["val_length"].append(step_count)
        val_retdict["val_coverage"].append(
            final_info.get("episode_coverage", 0.0) if isinstance(final_info, dict) else 0.0
        )
        val_retdict["val_subject_id"].append(subject_id)

    return val_retdict


def log_wandb(data: Dict[str, Any], step: Optional[int] = None, **kwargs):
    """Log data to wandb (no changes needed)."""
    if wandb is not None and wandb.run is not None:
        try:
            wandb.log(data, step=step, **kwargs)
        except Exception as e:
            print(f"Error logging to wandb: {e}")
