import os
import json
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from tensordict.nn import set_composite_lp_aggregate

# TorchRL renamed SyncDataCollector to Collector in 0.13.
try:
    from torchrl.collectors import SyncDataCollector
except ImportError:
    from torchrl.collectors import Collector as SyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import (
    ClipPPOLoss,
    SoftUpdate,
    TD3Loss,
)
from torchrl.objectives.value import GAE
from tqdm import tqdm

import wandb

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Your project components
from .config import Config
from .dataset import SmallBowelDataset  # Keep for creating the iterator

# Use the TorchRL environment wrapper and factory function
from .environment import make_sb_env
from .metrics import compute_path_metrics

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.capture_dynamic_output_shape_ops = True
set_composite_lp_aggregate(False).set()


def log_wandb(data: dict, **kwargs):
    """Log data to wandb."""
    if wandb is not None:
        wandb.log(data, **kwargs)
    else:
        print("WandB not initialized. Skipping logging.")


def create_tensorboard_writer(config: Config):
    """Create a local TensorBoard writer when requested."""
    if not config.track_tensorboard:
        return None
    if SummaryWriter is None:
        print(
            "TensorBoard tracking requested, but tensorboard is not installed. "
            "Install project dependencies with `uv sync`."
        )
        return None

    if config.tensorboard_log_dir:
        log_dir = config.tensorboard_log_dir
    else:
        run_name = config.wandb_run_name or (f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}")
        log_dir = str(Path(config.checkpoint_dir) / "tensorboard" / run_name)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
    writer.add_text(
        "configuration",
        f"```json\n{json.dumps(vars(config), indent=2)}\n```",
        global_step=0,
    )
    print(f"TensorBoard logs: {Path(log_dir).resolve()}")
    return writer


def log_tensorboard(writer, data: dict, step: int):
    """Write scalar entries from an existing metrics dictionary."""
    if writer is None:
        return
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, global_step=step)


def validation_rank(metrics: dict) -> tuple[float, float, float, float]:
    """Rank checkpoints by the preregistered traversal objective.

    This is lexicographic: completed traversals dominate endpoint-only runs,
    endpoint reach breaks success-rate ties, and Dice/distance only break the
    remaining ties. It avoids selecting a high-Dice path that never reaches
    the anatomical endpoint while still making progress visible before the
    first complete traversal.
    """

    return (
        float(metrics["validation/traversal_success_rate"]),
        float(metrics["validation/endpoint_reach_rate"]),
        float(metrics["validation/avg_dice"]),
        -float(metrics["validation/avg_endpoint_distance_mm"]),
    )


# --- Validation Loop (Adaptation Needed) ---
def validation_loop_torchrl(
    actor_module,  # Pass the trained policy module
    config: Config,
    val_dataset: SmallBowelDataset,  # Pass the validation subset
    device: torch.device = None,
    global_step: int | None = None,
):
    """Validation loop adapted for TorchRL env and modules."""
    actor_module.eval()  # Set actor to evaluation mode
    val_results = defaultdict(list)

    # Create a validation environment instance
    # Pass the validation iterator to this env instance
    save_path = (
        Path(config.validation_output_dir)
        if config.validation_output_dir
        else Path("results").joinpath(
            config.data_dir.split("/")[-1],
            (
                f"ps{config.patch_size_vox}_y{config.gamma:.03f}_"
                f"rv1{config.r_val1:g}_rv2{config.r_val2:g}"
            ),
        )
    )
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.json", "w") as f:
        json.dump(vars(config), f, indent=4)
    val_env = make_sb_env(
        config,
        val_dataset,
        device,
        1,
        check_env=False,
        shuffle=False,
    )

    num_val_subjects = len(val_dataset)

    with (
        torch.no_grad(),
        # Behavioral cloning supervises and rolls out the Beta mean. Using the
        # mode here can point elsewhere when concentration parameters are near
        # one, so deterministic validation must use the same policy statistic.
        set_exploration_type(ExplorationType.MEAN),
    ):
        for i in tqdm(range(num_val_subjects), desc="Validation"):
            # Deterministic mode produces one reproducible rollout per subject.
            paths = []
            path_masks = []
            intermediate_results = []
            reward, step_count, final_coverage, success = 0, 0, 0, 0
            endpoint_reached, endpoint_distance_mm = 0, float("inf")
            must_load_new_subject = True
            for _ in range(1):
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
                    path = val_env.get_tracking_history()
                    independent_metrics = compute_path_metrics(
                        val_env.seg.numpy(force=True),
                        path,
                        val_env.goal,
                        tuple(float(value) for value in val_env.spacing),
                        config.cumulative_path_radius_mm,
                        config.endpoint_tolerance_mm,
                        config.success_coverage_threshold,
                    )
                    final_coverage = independent_metrics.dice
                    success = float(independent_metrics.traversal_success)
                    endpoint_distance_mm = independent_metrics.endpoint_distance_mm
                    endpoint_reached = float(independent_metrics.endpoint_reached)

                    paths.append(path)
                    path_masks.append(val_env.get_tracking_mask())
                    intermediate_results.append(
                        (
                            reward,
                            step_count,
                            final_coverage,
                            total_reward,
                            success,
                            endpoint_reached,
                            endpoint_distance_mm,
                        )
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Validation failed for subject {i}; refusing to report "
                        "a metric on a silently reduced cohort."
                    ) from e

            # Keep the common selection path so stochastic validation can be
            # reintroduced explicitly later without silently cherry-picking.
            if len(intermediate_results) == 0:
                print(
                    f"Too many errors caused no successful rollout to be generated. Skipping subject: {i}"
                )
                continue
            best_run = intermediate_results.index(max(intermediate_results, key=lambda x: x[3]))
            (
                reward,
                step_count,
                final_coverage,
                total_reward,
                success,
                endpoint_reached,
                endpoint_distance_mm,
            ) = intermediate_results[best_run]
            path = paths[best_run]
            path_mask = path_masks[best_run]

            # Save the best path and mask
            case_id = val_env._current_subject_data["id"]
            if config.validation_save_paths:
                val_env.tracking_path_history = path
                val_env.cumulative_path_mask = path_mask
                val_env.save_path(save_path / case_id)

            val_results["case"].append(case_id)
            val_results["reward"].append(reward)
            val_results["length"].append(step_count)
            val_results["coverage"].append(final_coverage)
            val_results["total_reward"].append(total_reward)
            val_results["success"].append(success)
            val_results["endpoint_reached"].append(endpoint_reached)
            val_results["endpoint_distance_mm"].append(endpoint_distance_mm)

    val_env.close()  # Close the validation environment

    if len(val_results["coverage"]) != num_val_subjects:
        raise RuntimeError(
            f"Validation produced {len(val_results['coverage'])}/{num_val_subjects} results."
        )

    # Calculate mean results
    final_metrics = {
        "validation/avg_reward": np.mean(val_results["reward"]),
        "validation/avg_length": np.mean(val_results["length"]),
        "validation/avg_coverage": np.mean(val_results["coverage"]),
        "validation/avg_dice": np.mean(val_results["coverage"]),
        "validation/total_reward": np.mean(val_results["total_reward"]),
        "validation/success_rate": np.mean(val_results["success"]),
        "validation/traversal_success_rate": np.mean(val_results["success"]),
        "validation/endpoint_reach_rate": np.mean(val_results["endpoint_reached"]),
        "validation/avg_endpoint_distance_mm": np.mean(val_results["endpoint_distance_mm"]),
        "validation/num_cases": len(val_results["coverage"]),
    }

    metric_payload = final_metrics | val_results
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metric_payload, f, indent=4)
    if global_step is not None:
        with open(save_path / f"metrics_{global_step}.json", "w") as f:
            json.dump(metric_payload, f, indent=4)

    print(
        f"Validation Results: Avg R/L/C/S: {final_metrics['validation/avg_reward']:.2f} / "
        f"{final_metrics['validation/avg_length']:.1f} / "
        f"{final_metrics['validation/avg_coverage']:.2f} / "
        f"{final_metrics['validation/success_rate']:.2f}"
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
    """Run training and reliably flush local TensorBoard events."""
    tensorboard_writer = create_tensorboard_writer(config)
    try:
        return _train_torchrl(
            policy_module,
            value_module,
            config,
            train_set,
            val_set,
            device=device,
            qnets=qnets,
            tensorboard_writer=tensorboard_writer,
        )
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.close()


def _train_torchrl(
    policy_module,
    value_module,
    config: Config,
    train_set: SmallBowelDataset,
    val_set: SmallBowelDataset,
    device: torch.device = None,
    qnets: bool = False,
    tensorboard_writer=None,
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
            entropy_coeff=config.ent_coef,
            entropy_bonus=bool(config.ent_coef),
            critic_coeff=config.vf_coef,
            loss_critic_type="smooth_l1",  # TorchRL standard
            # loss_critic_type="l2",
            # GAE standardizes once over the full rollout. Renormalizing each
            # temporal minibatch changes advantage signs and destabilizes PPO.
            normalize_advantage=False,
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
    scheduler_steps = math.ceil(total_timesteps / config.frames_per_batch) * config.update_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, scheduler_steps),
        eta_min=5e-6,
    )
    # scheduler_c = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_critic,
    #     T_max=(total_timesteps * config.update_epochs // batch_size),
    #     eta_min=5e-6,
    # )
    collected_frames, num_updates = 0, 0
    best_val_metric = float("-inf")
    best_val_rank = (float("-inf"),) * 4

    # --- Checkpoint Reloading ---
    if config.reload_checkpoint_path:
        try:
            # Checkpoints are created locally by this trainer and include optimizer
            # state with NumPy scalar values, which PyTorch's weights-only loader
            # rejects by default in recent releases.
            checkpoint = torch.load(
                config.reload_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            policy_module.load_state_dict(checkpoint["policy_module_state_dict"])
            value_module.load_state_dict(checkpoint["value_module_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            collected_frames = checkpoint.get("collected_frames", 0)
            num_updates = checkpoint.get("num_updates", 0)
            best_val_metric = checkpoint.get("best_val_metric", float("-inf"))
            saved_rank = checkpoint.get("best_val_rank")
            best_val_rank = tuple(
                saved_rank
                if saved_rank is not None
                else (best_val_metric, float("-inf"), float("-inf"), float("-inf"))
            )
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

    # --- Collector ---
    # Collects data by interacting policy_module with environment instances
    def env_maker():
        return make_sb_env(
            config,
            train_set,
            device,
            num_episodes_per_sample=config.num_episodes_per_sample,
            num_steps_per_sample=config.num_steps_per_sample,
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
    # Use collector's iterator
    for i, batch_data in enumerate(collector, start=collected_frames):
        current_frames = batch_data.numel()  # Number of steps collected in this batch
        pbar.update(current_frames)
        collected_frames += current_frames

        # --- GAE Computation ---
        # 1. Compute advantages BEFORE flattening the batch
        with (
            torch.no_grad(),
            torch.autocast(device.type, amp_dtype, enabled=config.amp),
        ):
            if not qnets:
                adv_module(batch_data)

        # 2. Flatten for PPO minibatches
        batch_data = batch_data.reshape(-1)
        current_frames_flat = batch_data.numel()

        # --- PPO Update Phase ---
        actor_losses, critic_losses, entropy_losses, kl_div = [], [], [], []
        for _ in range(config.update_epochs):
            # 3. Shuffle data for i.i.d. minibatches
            perm = torch.randperm(current_frames_flat, device=device)
            batch_data_shuffled = batch_data[perm]

            for j in range(0, current_frames_flat, batch_size):
                minibatch = batch_data_shuffled[j : j + batch_size]
                with torch.autocast(device.type, amp_dtype, enabled=config.amp):
                    loss_dict = loss_module(minibatch)

                    if qnets:
                        actor_loss = loss_dict["loss_actor"]
                        critic_loss = loss_dict["loss_qvalue"]
                    else:
                        actor_loss = loss_dict["loss_objective"]
                        if "loss_entropy" in loss_dict.keys():
                            actor_loss = actor_loss + loss_dict["loss_entropy"]
                        critic_loss = loss_dict["loss_critic"]

                total_loss = actor_loss + critic_loss
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()

                # Log losses for this minibatch update
                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())
                entropy_losses.append(
                    loss_dict["loss_entropy"].detach()
                    if not qnets and "loss_entropy" in loss_dict.keys()
                    else torch.tensor(0.0, device=device)
                )
                kl_div.append(
                    loss_dict["kl_approx"].detach()
                    if not qnets
                    else torch.tensor(0.0, device=device)
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
        # A 2,048-step episode legitimately spans multiple 1,024-frame
        # collector batches. Keep optimizing on those batches while logging NaN
        # for episode-only statistics until a terminal transition is present.
        completed_episode = bool(idx.any().item())
        missing_episode_stat = torch.full((), float("nan"), device=device)

        def completed_mean(key: str, *, as_float: bool = False):
            if not completed_episode:
                return missing_episode_stat
            values = batch_data["next", "info", key]
            if as_float:
                values = values.float()
            return values[idx].mean()

        final_coverage = completed_mean("final_coverage")
        success_rate = completed_mean("final_success")
        step_count = completed_mean("final_step_count", as_float=True)
        ep_len = completed_mean("final_length", as_float=True)
        wall_gradient = completed_mean("final_wall_gradient", as_float=True)
        total_reward = completed_mean("total_reward")
        action = ((batch_data["action"] * 2 - 1) * config.max_step_vox).round()
        if completed_episode:
            max_gdt_achieved = batch_data["next", "info", "max_gdt_achieved"][idx]
            max_std, max_mean = torch.std_mean(max_gdt_achieved, unbiased=False)
            max_gdt_value = max_gdt_achieved.max()
        else:
            max_std = max_mean = max_gdt_value = missing_episode_stat
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
            "train/final_dice": final_coverage,
            "train/success_rate": success_rate,
            "train/traversal_success_rate": success_rate,
            "train/total_reward": total_reward,
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/max_gdt_achieved": max_mean,
            "charts/max_gdt_achieved_std": max_std,
            "charts/max_gdt_achieved_max": max_gdt_value,
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

        pbar.set_postfix(
            {
                "R": f"{avg_reward:.1f}",
                "Cov": f"{final_coverage:.1f}",
                "loss_P": f"{avg_actor_loss:.2f}",
                "loss_V": f"{avg_critic_loss:.2f}",
            }
        )

        if config.track_wandb and wandb is not None:
            log_wandb(log_data, step=collected_frames)
        log_tensorboard(tensorboard_writer, log_data, step=collected_frames)

        # --- Validation and Checkpointing ---
        if num_updates % config.eval_interval == 0:
            val_metrics = validation_loop_torchrl(
                actor_module=policy_module,
                config=config,
                val_dataset=val_set,
                device=device,
                global_step=collected_frames,
            )
            policy_module.train()
            if config.track_wandb and wandb is not None:
                log_wandb(val_metrics, step=collected_frames)
            log_tensorboard(tensorboard_writer, val_metrics, step=collected_frames)

            # Checkpointing logic (save based on validation metric)
            current_metric = val_metrics.get(config.metric_to_optimize, float("-inf"))
            current_rank = validation_rank(val_metrics)
            if current_rank > best_val_rank:
                best_val_metric = current_metric
                best_val_rank = current_rank
                print(
                    f"  New best validation rank: {best_val_rank} "
                    f"({config.metric_to_optimize}={best_val_metric:.4f})"
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
                    best_val_rank=best_val_rank,
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
                best_val_rank=best_val_rank,
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
        final_model_path,
        best_val_rank=best_val_rank,
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
    checkpoint_path: str = None,
    best_val_rank: tuple[float, float, float, float] | None = None,
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
        "best_val_rank": best_val_rank,
        "config": vars(config),
    }
    torch.save(save_dict, checkpoint_path)
