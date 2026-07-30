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
from torchrl.modules import set_recurrent_mode
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
from .dataset import (
    SmallBowelDataset,
    load_nnunet_evaluation_target,
)

# Use the TorchRL environment wrapper and factory function
from .environment import REWARD_COMPONENT_INFO_KEYS, make_sb_env
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


def advance_periodic_threshold(
    current_value: int,
    next_threshold: int,
    interval: int,
) -> tuple[bool, int]:
    """Return whether a threshold was crossed and advance past the current value."""
    if interval < 1:
        raise ValueError("interval must be positive")
    due = current_value >= next_threshold
    while next_threshold <= current_value:
        next_threshold += interval
    return due, next_threshold


def should_run_final_validation(
    collected_frames: int,
    last_validation_frame: int | None,
) -> bool:
    """Require a final metric unless this exact policy state was validated."""
    return collected_frames > 0 and last_validation_frame != collected_frames


def deterministic_exploration_type(config: Config) -> ExplorationType:
    """Return the configured deterministic statistic for Beta rollouts."""

    if config.deterministic_action_statistic == "mean":
        return ExplorationType.MEAN
    if config.deterministic_action_statistic == "mode":
        return ExplorationType.MODE
    raise ValueError(
        "deterministic_action_statistic must be either 'mean' or 'mode'"
    )


def recurrent_minibatches(batch_data, sequence_length: int, batch_size: int):
    """Yield shuffled contiguous sequence minibatches.

    Recurrent state stored at the first transition of each chunk initializes
    the memory model. Transitions within a chunk are never shuffled.
    """

    total_steps = batch_data.numel()
    full_sequences = total_steps // sequence_length
    sequences_per_batch = max(1, batch_size // sequence_length)

    if full_sequences:
        sequence_data = batch_data[: full_sequences * sequence_length].reshape(
            full_sequences,
            sequence_length,
        )
        order = torch.randperm(full_sequences, device=batch_data.device)
        for start in range(0, full_sequences, sequences_per_batch):
            yield sequence_data[order[start : start + sequences_per_batch]]

    tail_start = full_sequences * sequence_length
    if tail_start < total_steps:
        yield batch_data[tail_start:].unsqueeze(0)


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
        policy=actor_module,
    )
    tracking_env = val_env.base_env if config.memory_model != "none" else val_env

    num_val_subjects = len(val_dataset)

    with (
        torch.no_grad(),
        set_exploration_type(deterministic_exploration_type(config)),
    ):
        for i in tqdm(range(num_val_subjects), desc="Validation"):
            # One deterministic statistic produces one reproducible rollout
            # per subject.
            paths = []
            path_masks = []
            intermediate_results = []
            reward, step_count, final_coverage, success = 0, 0, 0, 0
            endpoint_reached, endpoint_distance_mm = 0, float("inf")
            action_executed_fraction = positive_gdt_fraction = 0.0
            recent_unique_position_fraction = boundary_state_fraction = 0.0
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
                    action_executed_fraction = rollout[
                        "next", "info", "action_executed"
                    ].float().mean().item()
                    positive_gdt_fraction = (
                        rollout["next", "info", "reward_gdt"] > 0
                    ).float().mean().item()
                    recent_unique_position_fraction = rollout[
                        "next", "info", "recent_unique_position_fraction"
                    ].reshape(-1)[-1].item()
                    if config.action_distribution == "masked_categorical":
                        boundary_state_fraction = (
                            rollout["action_mask"].sum(dim=-1)
                            < config.categorical_action_count
                        ).float().mean().item()
                    path = tracking_env.get_tracking_history()
                    if config.annotation_free:
                        evaluation_target = load_nnunet_evaluation_target(
                            config.nnunet_raw_dir,
                            config.nnunet_cache_dir,
                            tracking_env._current_subject_data["id"],
                        )
                        target_mask = evaluation_target["segmentation"]
                        evaluation_goal = evaluation_target["goal"]
                        evaluation_spacing = evaluation_target["spacing"]
                    else:
                        target_mask = tracking_env.seg.numpy(force=True)
                        evaluation_goal = tracking_env.goal
                        evaluation_spacing = tuple(
                            float(value) for value in tracking_env.spacing
                        )
                    independent_metrics = compute_path_metrics(
                        target_mask,
                        path,
                        evaluation_goal,
                        evaluation_spacing,
                        config.cumulative_path_radius_mm,
                        config.endpoint_tolerance_mm,
                        config.success_coverage_threshold,
                    )
                    final_coverage = independent_metrics.dice
                    success = float(independent_metrics.traversal_success)
                    endpoint_distance_mm = independent_metrics.endpoint_distance_mm
                    endpoint_reached = float(independent_metrics.endpoint_reached)

                    paths.append(path)
                    # Keep only the small report artifact, not a second GPU
                    # copy of the complete patient mask.
                    path_masks.append(tracking_env.get_tracking_mask().cpu())
                    intermediate_results.append(
                        (
                            reward,
                            step_count,
                            final_coverage,
                            total_reward,
                            success,
                            endpoint_reached,
                            endpoint_distance_mm,
                            action_executed_fraction,
                            positive_gdt_fraction,
                            recent_unique_position_fraction,
                            boundary_state_fraction,
                        )
                    )
                    # A 2,048-step rollout contains every 3-D observation.
                    # Release it before resetting onto the next validation
                    # subject instead of retaining both subjects on the GPU.
                    del rollout, tensordict
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
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
                action_executed_fraction,
                positive_gdt_fraction,
                recent_unique_position_fraction,
                boundary_state_fraction,
            ) = intermediate_results[best_run]
            path = paths[best_run]
            path_mask = path_masks[best_run]

            # Save the best path and mask
            case_id = tracking_env._current_subject_data["id"]
            if config.validation_save_paths:
                tracking_env.tracking_path_history = path
                tracking_env.cumulative_path_mask = path_mask
                tracking_env.save_path(save_path / case_id)

            val_results["case"].append(case_id)
            val_results["reward"].append(reward)
            val_results["length"].append(step_count)
            val_results["coverage"].append(final_coverage)
            val_results["total_reward"].append(total_reward)
            val_results["success"].append(success)
            val_results["endpoint_reached"].append(endpoint_reached)
            val_results["endpoint_distance_mm"].append(endpoint_distance_mm)
            val_results["action_executed_fraction"].append(
                action_executed_fraction
            )
            val_results["positive_gdt_fraction"].append(positive_gdt_fraction)
            val_results["recent_unique_position_fraction"].append(
                recent_unique_position_fraction
            )
            val_results["boundary_state_fraction"].append(
                boundary_state_fraction
            )

    val_env.close()  # Close the validation environment
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
        "validation/action_executed_fraction": np.mean(
            val_results["action_executed_fraction"]
        ),
        "validation/positive_gdt_fraction": np.mean(
            val_results["positive_gdt_fraction"]
        ),
        "validation/recent_unique_position_fraction": np.mean(
            val_results["recent_unique_position_fraction"]
        ),
        "validation/boundary_state_fraction": np.mean(
            val_results["boundary_state_fraction"]
        ),
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
        "Total unique trainable parameters: "
        f"{sum(p.numel() for p in {id(p): p for p in [*policy_module.parameters(), *value_module.parameters()]}.values())}"
    )
    recurrent_policy = config.memory_model != "none"

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
            # Optionally prevent a much larger critic gradient from dominating
            # the shared visual encoder and recurrent memory.
            separate_losses=config.separate_actor_critic_losses,
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
        # Recurrent dynamics are sensitive to decay on transition and time
        # constants. Keep the GRU/S5 comparison matched and decay-free.
        weight_decay=0.0 if recurrent_policy else 0.01,
    )
    policy_parameters = list(policy_module.parameters())
    policy_parameter_ids = {id(parameter) for parameter in policy_parameters}
    value_only_parameters = [
        parameter
        for parameter in value_module.parameters()
        if id(parameter) not in policy_parameter_ids
    ]

    amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    scaler = torch.GradScaler(enabled=config.amp and amp_dtype == torch.float16)
    # A validated short annealing horizon can be retained in a longer job. The
    # scheduler is explicitly frozen at eta_min after that horizon; stepping a
    # CosineAnnealingLR beyond T_max would otherwise increase the LR again.
    anneal_timesteps = config.lr_anneal_timesteps or total_timesteps
    anneal_timesteps = min(anneal_timesteps, total_timesteps)
    scheduler_steps = (
        math.ceil(anneal_timesteps / config.frames_per_batch)
        * config.update_epochs
    )
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

    next_validation_update = (
        num_updates // config.eval_interval + 1
    ) * config.eval_interval
    next_checkpoint_update = (
        num_updates // config.save_freq + 1
    ) * config.save_freq

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
            policy=policy_module,
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
        # Native recurrent modules reset on data-dependent ``is_init`` flags,
        # which cannot be evaluated inside functorch vmap.
        deactivate_vmap=recurrent_policy,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- Training Loop ---
    pbar = tqdm(total=total_timesteps, desc="Training", unit="steps", initial=collected_frames)
    last_validation_frame: int | None = None

    def validate_current_policy() -> dict:
        """Validate, log, and update the best checkpoint at the current frame."""
        nonlocal best_val_metric, best_val_rank, last_validation_frame
        val_metrics = validation_loop_torchrl(
            actor_module=policy_module,
            config=config,
            val_dataset=val_set,
            device=device,
            global_step=collected_frames,
        )
        last_validation_frame = collected_frames
        policy_module.train()
        if config.track_wandb and wandb is not None:
            log_wandb(val_metrics, step=collected_frames)
        log_tensorboard(tensorboard_writer, val_metrics, step=collected_frames)

        if config.annotation_free:
            print(
                "  Annotation-free protocol: validation labels are "
                "report-only and cannot select a checkpoint."
            )
            return val_metrics

        current_metric = val_metrics.get(
            config.metric_to_optimize,
            float("-inf"),
        )
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
        return val_metrics

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
            set_recurrent_mode(recurrent_policy),
        ):
            if not qnets:
                adv_module(batch_data)

        # Keep the collector order for recurrent PPO. A separate flattened
        # view is sufficient for scalar logging after the update.
        batch_data = batch_data.reshape(-1)
        current_frames_flat = batch_data.numel()

        # --- PPO Update Phase ---
        actor_losses, critic_losses, entropy_losses, kl_div = [], [], [], []
        ppo_epochs_completed = 0
        kl_early_stop = False
        for _ in range(config.update_epochs):
            epoch_kl_start = len(kl_div)
            if recurrent_policy:
                minibatches = recurrent_minibatches(
                    batch_data,
                    config.recurrent_sequence_length,
                    batch_size,
                )
            else:
                perm = torch.randperm(current_frames_flat, device=device)
                batch_data_shuffled = batch_data[perm]
                minibatches = (
                    batch_data_shuffled[j : j + batch_size]
                    for j in range(0, current_frames_flat, batch_size)
                )

            for minibatch in minibatches:
                with (
                    torch.autocast(device.type, amp_dtype, enabled=config.amp),
                    set_recurrent_mode(recurrent_policy),
                ):
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
                if config.separate_actor_critic_losses:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy_parameters,
                        config.max_grad_norm,
                    )
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        value_only_parameters,
                        config.max_grad_norm,
                    )
                    grad_norm = actor_grad_norm
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), config.max_grad_norm
                    )
                    actor_grad_norm = critic_grad_norm = grad_norm
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

            if scheduler.last_epoch < scheduler_steps:
                scheduler.step()
            # scheduler_c.step()
            if qnets:
                updater.step()
            num_updates += 1  # Count PPO update cycles
            ppo_epochs_completed += 1
            epoch_kl_values = kl_div[epoch_kl_start:]
            if (
                not qnets
                and config.target_kl > 0
                and epoch_kl_values
                and torch.stack(epoch_kl_values).mean().item() > config.target_kl
            ):
                kl_early_stop = True
                break

        # --- Logging ---
        avg_actor_loss = torch.stack(actor_losses).mean().item()
        avg_critic_loss = torch.stack(critic_losses).mean().item()
        avg_entropy_loss = torch.stack(entropy_losses).mean().item()
        avg_kldiv = torch.stack(kl_div).mean().item()
        avg_reward = batch_data["next", "reward"].mean().item()
        max_reward = batch_data["next", "reward"].max().item()
        avg_episodic_cell_reward = batch_data[
            "next", "info", "episodic_cell_reward"
        ].mean().item()
        reward_component_means = {
            key: batch_data["next", "info", key].mean().item()
            for key in REWARD_COMPONENT_INFO_KEYS
        }
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
        if config.annotation_free:
            # GT metrics do not exist inside the training environment. They
            # are computed only by the post-rollout validation evaluator.
            final_coverage = missing_episode_stat
            success_rate = missing_episode_stat
        if config.action_distribution in {
            "categorical",
            "masked_categorical",
        }:
            displacement_table = torch.as_tensor(
                config.action_displacements,
                dtype=torch.float32,
                device=device,
            )
            action = displacement_table[batch_data["action"].long().reshape(-1)]
        elif config.action_distribution == "factorized_categorical":
            action = (
                batch_data["action"].long().reshape(-1, 3)
                - config.max_step_vox
            ).float()
        else:
            action = ((batch_data["action"] * 2 - 1) * config.max_step_vox).round()
        action_executed = batch_data[
            "next", "info", "action_executed"
        ].reshape(-1).bool()
        executed_action_fraction = action_executed.float().mean()
        positive_gdt_fraction = (
            batch_data["next", "info", "reward_gdt"].reshape(-1) > 0
        ).float().mean()
        off_target_fraction = (
            batch_data["next", "info", "reward_off_target"].reshape(-1) < 0
        ).float().mean()
        recent_unique_position_fraction = batch_data[
            "next", "info", "recent_unique_position_fraction"
        ].reshape(-1).mean()
        immediate_reversal_fraction = torch.tensor(0.0, device=device)
        if action.shape[0] > 1:
            valid_pairs = action_executed[1:] & action_executed[:-1]
            if "is_init" in batch_data.keys():
                valid_pairs &= ~batch_data["is_init"].reshape(-1)[1:].bool()
            if valid_pairs.any():
                immediate_reversal_fraction = (
                    (action[1:] == -action[:-1]).all(dim=-1)[valid_pairs]
                    .float()
                    .mean()
                )
        available_action_fraction = torch.tensor(1.0, device=device)
        boundary_state_fraction = torch.tensor(0.0, device=device)
        if config.action_distribution == "masked_categorical":
            action_mask = batch_data["action_mask"].bool()
            available_action_fraction = action_mask.float().mean()
            boundary_state_fraction = (
                action_mask.sum(dim=-1) < config.categorical_action_count
            ).float().mean()
        if completed_episode:
            max_gdt_achieved = batch_data["next", "info", "max_gdt_achieved"][idx]
            max_std, max_mean = torch.std_mean(max_gdt_achieved, unbiased=False)
            max_gdt_value = max_gdt_achieved.max()
        else:
            max_std = max_mean = max_gdt_value = missing_episode_stat
        if config.annotation_free:
            max_std = max_mean = max_gdt_value = missing_episode_stat
        log_data = {
            "losses/policy_loss": avg_actor_loss,
            "losses/value_loss": avg_critic_loss,
            "losses/entropy": avg_entropy_loss,
            "losses/kl_div": avg_kldiv,
            "losses/grad_norm": grad_norm,
            "losses/actor_grad_norm": actor_grad_norm,
            "losses/critic_grad_norm": critic_grad_norm,
            "train/reward": avg_reward,
            "train/max_reward": max_reward,
            "train/episodic_cell_reward": avg_episodic_cell_reward,
            "train/step_count": step_count,
            "train/wall_gradient": wall_gradient,
            "train/episode_len": ep_len,
            "train/final_coverage": final_coverage,
            "train/final_dice": final_coverage,
            "train/success_rate": success_rate,
            "train/traversal_success_rate": success_rate,
            "train/total_reward": total_reward,
            "train/executed_action_fraction": executed_action_fraction,
            "train/invalid_action_fraction": 1.0 - executed_action_fraction,
            "train/positive_gdt_fraction": positive_gdt_fraction,
            "train/off_target_fraction": off_target_fraction,
            "train/recent_unique_position_fraction": (
                recent_unique_position_fraction
            ),
            "train/immediate_reversal_fraction": immediate_reversal_fraction,
            "train/boundary_state_fraction": boundary_state_fraction,
            "train/available_action_fraction": available_action_fraction,
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/max_gdt_achieved": max_mean,
            "charts/max_gdt_achieved_std": max_std,
            "charts/max_gdt_achieved_max": max_gdt_value,
            "charts/num_updates": num_updates,
            "charts/ppo_epochs_completed": ppo_epochs_completed,
            "charts/kl_early_stop": float(kl_early_stop),
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
        log_data.update(
            {
                f"train/{key}": value
                for key, value in reward_component_means.items()
            }
        )
        if config.action_distribution == "beta":
            log_data.update(
                {
                    "losses/alpha": batch_data["alpha"].mean(),
                    "losses/beta": batch_data["beta"].mean(),
                }
            )
        elif config.action_distribution in {
            "categorical",
            "masked_categorical",
        }:
            action_logits = batch_data["logits"]
            if config.action_distribution == "masked_categorical":
                action_logits = action_logits.masked_fill(
                    ~batch_data["action_mask"].bool(),
                    -torch.inf,
                )
            action_probabilities = action_logits.softmax(dim=-1)
            log_data.update(
                {
                    "policy/max_action_probability": (
                        action_probabilities.max(dim=-1).values.mean()
                    ),
                    "policy/logit_std": batch_data["logits"].std(dim=-1).mean(),
                }
            )
        else:
            axis_probabilities = batch_data["logits"].softmax(dim=-1)
            log_data.update(
                {
                    "policy/max_action_probability": (
                        axis_probabilities.max(dim=-1).values.prod(dim=-1).mean()
                    ),
                    "policy/logit_std": batch_data["logits"].std(dim=-1).mean(),
                }
            )
        if device.type == "cuda":
            log_data.update(
                {
                    "system/cuda_peak_allocated_mb": (
                        torch.cuda.max_memory_allocated(device) / 2**20
                    ),
                    "system/cuda_peak_reserved_mb": (
                        torch.cuda.max_memory_reserved(device) / 2**20
                    ),
                }
            )

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
        validation_due, next_validation_update = advance_periodic_threshold(
            num_updates,
            next_validation_update,
            config.eval_interval,
        )
        if validation_due:
            validate_current_policy()

        # Regular checkpoint saving
        checkpoint_due, next_checkpoint_update = advance_periodic_threshold(
            num_updates,
            next_checkpoint_update,
            config.save_freq,
        )
        if checkpoint_due:
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
    # PPO KL early stopping makes the update counter intentionally variable.
    # Threshold-crossing scheduling cannot guarantee that a finite run ends
    # exactly on a validation threshold, so always score the final policy.
    if should_run_final_validation(collected_frames, last_validation_frame):
        print(f"Running mandatory final validation at frame {collected_frames}.")
        validate_current_policy()

    pbar.close()
    collector.shutdown()
    print("Training finished.")
    if device.type == "cuda":
        print(
            "Peak CUDA memory: "
            f"{torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB allocated, "
            f"{torch.cuda.max_memory_reserved(device) / 2**20:.1f} MiB reserved"
        )

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
