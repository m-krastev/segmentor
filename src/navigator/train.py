import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from tensordict.nn import set_composite_lp_aggregate

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

import wandb

# Your project components
from .config import Config
from .dataset import SmallBowelDataset  # Keep for creating the iterator
from .exploration import RND # Import RND

# Use the TorchRL environment wrapper and factory function
from .environment import SmallBowelEnv, make_sb_env

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
    val_results = defaultdict(list)

    # Create a validation environment instance
    # Pass the validation iterator to this env instance
    save_path = Path("results").joinpath(
        config.data_dir.split("/")[-1],
        f"ps{config.patch_size_vox}_y{config.gamma:.03f}_rv1{config.r_val1:g}_rv2{config.r_val2:g}",
    )
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.json", "w") as f:
        json.dump(vars(config), f, indent=4)
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
                    tensordict = val_env._reset(must_load_new_subject=must_load_new_subject)
                    rollout = val_env.rollout(
                        config.max_episode_steps,
                        actor_module,
                        auto_reset=False,
                        tensordict=tensordict,
                    )
                    must_load_new_subject = False

                    reward = rollout["next", "reward"].mean().item()
                    total_reward = rollout["next", "info", "total_reward"].sum().item()
                    step_count = rollout["action"].shape[1]
                    final_coverage = val_env._get_final_coverage().item()

                    paths.append(val_env.get_tracking_history())
                    path_masks.append(val_env.get_tracking_mask())
                    intermediate_results.append((reward, step_count, final_coverage, total_reward))
                except Exception as e:
                    print(f"Error during validation rollout for subject {i}: {e}")

            # Choose the best result from the 10 rollouts
            if len(intermediate_results) == 0:
                print(
                    f"Too many errors caused no successful rollout to be generated. Skipping subject: {i}"
                )
                continue
            best_run = intermediate_results.index(max(intermediate_results, key=lambda x: x[-1]))
            reward, step_count, final_coverage, total_reward = intermediate_results[best_run]
            path = paths[best_run]
            path_mask = path_masks[best_run]

            # Save the best path and mask
            val_env.tracking_path_history = path
            val_env.cumulative_path_mask = path_mask
            val_env.save_path(save_path)

            val_results["reward"].append(reward)
            val_results["length"].append(step_count)
            val_results["coverage"].append(final_coverage)
            val_results["total_reward"].append(total_reward)

    val_env.close()  # Close the validation environment

    # Calculate mean results
    final_metrics = {
        "validation/avg_reward": np.mean(val_results["reward"]),
        "validation/avg_length": np.mean(val_results["length"]),
        "validation/avg_coverage": np.mean(val_results["coverage"]),
        "validation/total_reward": np.mean(val_results["total_reward"]),
    }

    with open(save_path / "metrics.json", "w") as f:
        json.dump(final_metrics | val_results, f, indent=4)

    print(
        f"Validation Results: Avg R/L/C: {final_metrics['validation/avg_reward']:.2f} / "
        f"{final_metrics['validation/avg_length']:.1f} / {final_metrics['validation/avg_coverage']:.2f} "
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

    # Initialize RND module if enabled
    rnd_module = None
    if config.use_intrinsic_reward:
        # The environment will be created by the collector, so we pass a dummy env for RND init
        # The actual envs will be passed during the collector's init
        dummy_env = make_sb_env(config, train_set, device, num_episodes_per_sample=1, check_env=False)
        rnd_module = RND(
            envs=dummy_env,
            device=device,
            beta=config.intrinsic_reward_beta,
            kappa=config.intrinsic_reward_kappa,
            gamma=config.intrinsic_reward_gamma,
            rwd_norm_type=config.intrinsic_reward_rwd_norm_type,
            obs_norm_type=config.intrinsic_reward_obs_norm_type,
            latent_dim=config.rnd_latent_dim,
            lr=config.rnd_lr,
            batch_size=config.rnd_batch_size,
            update_proportion=config.rnd_update_proportion,
            encoder_model=config.rnd_encoder_model,
            weight_init=config.rnd_weight_init,
        )
        dummy_env.close() # Close the dummy env

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
    scaler = torch.GradScaler(enabled=config.amp and amp_dtype == torch.float16)
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
            print(f"Error: Checkpoint file not found at {config.reload_checkpoint_path}")
        except KeyError as e:
            print(f"Error loading checkpoint: Missing key {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading checkpoint: {e}")

    # Advance the scheduler to the correct state
    for _ in range(num_updates):
        scheduler.step()

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
        # cudagraph_policy=True # <- This screws with the distribution, don't use.
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
        average_gae=True,  # Standardize GAE
    )

    # --- Training Loop ---
    pbar = tqdm(total=total_timesteps, desc="Training", unit="steps", initial=collected_frames)
    # Use a specific metric like coverage or reward
    best_val_metric = float("-inf")
    # Use collector's iterator
    for i, batch_data in enumerate(collector, start=collected_frames):
        current_frames = batch_data.numel()  # Number of steps collected in this batch
        pbar.update(current_frames)
        collected_frames += current_frames

        # Compute intrinsic rewards and add to batch_data if enabled
        intrinsic_rewards_mean = torch.tensor(0.0, device=device)
        if config.use_intrinsic_reward and rnd_module is not None:
            # RND module expects (N_steps, N_envs, ...) for observations
            # batch_data is (N_steps * N_envs, ...) after reshape(-1)
            # We need to reshape it back to (N_steps, N_envs, ...) for RND
            n_steps = config.frames_per_batch // collector.env.num_envs
            n_envs = collector.env.num_envs
            
            # Create a temporary tensordict with the correct shape for RND
            # This assumes 'observations' and 'next_observations' are present
            # and have the correct structure for RND's compute method.
            # The collector should already provide these.
            rnd_input_td = batch_data.select("observations", "next_observations").reshape(n_steps, n_envs)
            
            intrinsic_rewards = rnd_module.compute(rnd_input_td, sync=True)
            
            # Add intrinsic rewards to the "next", "reward" key
            # Ensure shapes match: intrinsic_rewards is (n_steps, n_envs), batch_data["next", "reward"] is (n_steps, n_envs, 1)
            # We need to reshape intrinsic_rewards to (n_steps * n_envs, 1) to match batch_data's flattened structure
            batch_data["next", "reward"] = batch_data["next", "reward"] + intrinsic_rewards.view(-1, 1)
            intrinsic_rewards_mean = intrinsic_rewards.mean().item()

        # --- PPO Update Phase ---
        actor_losses, critic_losses, entropy_losses, kl_div = [], [], [], []
        for _ in range(config.update_epochs):
            batch_data = batch_data.reshape(-1) # Flatten for minibatch sampling

            with (
                torch.no_grad(),
                torch.autocast(device.type, amp_dtype, enabled=config.amp),
            ):
                if not qnets:
                    adv_module(batch_data) # GAE calculation happens here

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
                entropy_losses.append(loss_dict["loss_entropy"] if not qnets else torch.tensor(0.0))
                kl_div.append(loss_dict["kl_approx"] if not qnets else torch.tensor(0.0))

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
        action = ((batch_data["action"] * 2 - 1) * config.max_step_vox).round()
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
        if config.use_intrinsic_reward:
            log_data["train/intrinsic_reward"] = intrinsic_rewards_mean

        pbar.set_postfix({
            "R": f"{avg_reward:.1f}",
            "Cov": f"{final_coverage:.1f}",
            "loss_P": f"{avg_actor_loss:.2f}",
            "loss_V": f"{avg_critic_loss:.2f}",
        })

        if config.track_wandb and wandb is not None:
            log_wandb(log_data, step=collected_frames)

        # --- Validation and Checkpointing ---
        if num_updates % config.eval_interval == 0:
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
            current_metric = val_metrics.get(config.metric_to_optimize, float("-inf"))
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                print(
                    f"  New best validation metric ({config.metric_to_optimize}): {best_val_metric:.4f}"
                )
                save_checkpoint(
                    policy_module,
                    value_module,
                    optimizer,
                    scheduler,
                    collected_frames,
                    num_updates,
                    config,
                    True,
                    best_val_metric,
                )

        # Regular checkpoint saving
        if num_updates % config.save_freq == 0:
            save_checkpoint(
                policy_module,
                value_module,
                optimizer,
                scheduler,
                collected_frames,
                num_updates,
                config,
                False,
                best_val_metric,
            )

    # --- End of Training ---
    pbar.close()
    collector.shutdown()
    print("Training finished.")

    final_model_path = os.path.join(config.checkpoint_dir, "final_model_torchrl.pth")
    save_checkpoint(
        policy_module,
        value_module,
        optimizer,
        scheduler,
        collected_frames,
        num_updates,
        config,
        False,
        best_val_metric,
        final_model_path
    )
    print(f"Final model saved to {final_model_path}")


def save_checkpoint(
    policy_module,
    value_module,
    optimizer,
    scheduler,
    collected_frames,
    num_updates,
    config: Config,
    best=False,
    best_val_metric: float = float("-inf"),
    checkpoint_path: str =None,
):
    checkpoint_path = checkpoint_path or os.path.join(
        config.checkpoint_dir, f"checkpoint_{collected_frames}{'best' if best else ''}.pth"
    )

    save_dict = {
        "policy_module_state_dict": policy_module.state_dict(),
        "value_module_state_dict": value_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "collected_frames": collected_frames,
        "num_updates": num_updates,
        "best_val_metric": best_val_metric,
        "config": vars(config),
    }
    torch.save(save_dict, checkpoint_path)
