"""Behavior-cloning warm start for the Navigator policy."""

from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tqdm import tqdm

from .environment import make_sb_env


def _geodesic_expert_action(env) -> torch.Tensor:
    """Choose the realizable action with the lowest mask-constrained goal distance."""
    current = np.asarray(env.current_pos_vox, dtype=int)
    candidates = []
    seen_displacements = set()
    for direction in product((-1, 0, 1), repeat=3):
        if not any(direction):
            continue
        action = torch.as_tensor(
            (np.asarray(direction, dtype=np.float32) + 1.0) / 2.0,
            dtype=env.dtype,
            device=env.device,
        )
        displacement = env._project_action_to_allowed_displacement(action)
        if not any(displacement) or displacement in seen_displacements:
            continue
        seen_displacements.add(displacement)
        next_position = tuple((current + np.asarray(displacement, dtype=int)).tolist())
        goal_distance = float(env.goal_distance_map[next_position])
        if np.isfinite(goal_distance):
            candidates.append(
                (
                    goal_distance,
                    -float(np.linalg.norm(displacement)),
                    action,
                )
            )
    if not candidates:
        raise RuntimeError(
            f"No realizable geodesic expert action from {env.current_pos_vox} toward {env.goal}."
        )
    return min(candidates, key=lambda candidate: candidate[:2])[2]


def _monotonic_expert_action(env, path_index: int) -> tuple[torch.Tensor, int]:
    """Return an action toward the next centerline waypoint.

    ``path_index`` is a monotonic route cursor, not a nearest-point lookup.
    Only a bounded number of contiguous future waypoints may be skipped, and
    only when the exact straight displacement is traversable. This prevents
    the cursor from jumping to a spatially close but topologically later bowel
    loop. Sparse phantom waypoints remain supported by repeatedly moving
    toward the next waypoint without advancing the cursor.
    """
    path = env.gt_path_voxels
    if path is None or len(path) < 2:
        raise ValueError("Behavior cloning requires a non-empty ground-truth path.")

    current = np.asarray(env.current_pos_vox)
    if path_index >= len(path) - 1:
        return torch.full((3,), 0.5, dtype=env.dtype, device=env.device), path_index

    target_index = None
    # A dense 26-connected route can advance by at most max_step_vox ordered
    # waypoints per action. Restricting lookahead by route order, rather than
    # Euclidean proximity, is what prevents cross-loop cursor jumps.
    final_candidate = min(
        len(path) - 1,
        path_index + env.config.max_step_vox,
    )
    for candidate_index in range(final_candidate, path_index, -1):
        candidate_displacement = np.asarray(path[candidate_index]) - current
        if not np.any(candidate_displacement):
            target_index = candidate_index
            break
        if float(np.max(np.abs(candidate_displacement))) > env.config.max_step_vox:
            continue
        is_traversable = getattr(env, "_is_allowed_displacement", None)
        if is_traversable is None or is_traversable(
            tuple(candidate_displacement.astype(int).tolist())
        ):
            target_index = candidate_index
            break

    if target_index is None:
        # Sparse paths may place the next waypoint beyond one action. Move
        # toward it, but retain the cursor until a later call can execute the
        # exact final displacement.
        target_index = path_index + 1
        next_path_index = path_index
    else:
        next_path_index = target_index

    displacement = path[target_index] - current
    # Encode both direction and requested length. Long waypoint deltas are
    # capped to the action radius while the final short displacement remains
    # short, preventing deterministic endpoint oscillation.
    largest_component = max(float(np.max(np.abs(displacement))), 1.0)
    if largest_component > env.config.max_step_vox:
        displacement = displacement * (env.config.max_step_vox / largest_component)
    action = torch.as_tensor(
        (displacement / env.config.max_step_vox + 1.0) / 2.0,
        dtype=env.dtype,
        device=env.device,
    ).clamp(0.0, 1.0)
    return action, next_path_index


def _resynchronize_path_index(env, path_index: int) -> int:
    """Advance a route cursor locally after a learned-policy rollout step."""

    path = env.gt_path_voxels
    if path is None or len(path) < 2:
        return path_index
    current = np.asarray(env.current_pos_vox)
    first = min(max(int(path_index), 0), len(path) - 1)
    # One action spans at most max_step_vox in each axis. A modestly larger
    # ordered window lets an off-route policy step rejoin the demonstration
    # without matching a distant, spatially touching bowel loop.
    final = min(len(path) - 1, first + 4 * env.config.max_step_vox)
    candidates = np.arange(first, final + 1)
    distances = np.linalg.norm(np.asarray(path)[candidates] - current, axis=1)
    best_offset = min(
        range(len(candidates)),
        key=lambda offset: (float(distances[offset]), -int(candidates[offset])),
    )
    best_index = int(candidates[best_offset])
    maximum_rejoin_distance = env.config.max_step_vox * np.sqrt(3.0)
    if float(distances[best_offset]) <= maximum_rejoin_distance:
        return best_index
    return first


def _behavior_cloning_action(distribution, statistic: str) -> torch.Tensor:
    """Return the configured differentiable Beta-policy action statistic."""

    if statistic == "mean":
        return distribution.mean
    if statistic == "mode":
        return distribution.mode
    raise ValueError(
        "behavior_cloning_action_statistic must be either 'mean' or 'mode'"
    )


def pretrain_behavior_cloning(policy_module, config, train_set, device) -> None:
    """Fit the stochastic actor to training-set route actions.

    Ground-truth or segmentation-derived skeleton routes are used only for this
    training warm start. Validation subjects remain held out, and subsequent
    PPO training uses the unchanged environment reward and strict success
    criterion.
    """
    device = torch.device(device)
    optimizer = torch.optim.AdamW(
        policy_module.parameters(),
        lr=config.behavior_cloning_learning_rate,
    )
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    scaler = torch.GradScaler(enabled=config.amp and amp_dtype == torch.float16)
    env = make_sb_env(
        config,
        train_set,
        device,
        num_episodes_per_sample=1,
        num_steps_per_sample=config.max_episode_steps,
        check_env=False,
    )

    def update(batch) -> float:
        actor = torch.stack([sample[0] for sample in batch])
        context = torch.stack([sample[1] for sample in batch])
        # The Beta policy has open interval support. The geometric expert can
        # emit exact 0/1 components, so keep targets just inside the support.
        expert_action = torch.stack([sample[2] for sample in batch]).clamp(1e-4, 1 - 1e-4)
        tensordict = TensorDict(
            {"actor": actor, "context": context},
            batch_size=torch.Size([len(batch)]),
            device=device,
        )
        with torch.autocast(device.type, amp_dtype, enabled=config.amp):
            distribution = policy_module.get_dist(tensordict)
            predicted_action = _behavior_cloning_action(
                distribution,
                config.behavior_cloning_action_statistic,
            )
            predicted_direction = 2 * predicted_action - 1
            expert_direction = 2 * expert_action - 1
            direction_loss = (
                1
                - F.cosine_similarity(
                    predicted_direction,
                    expert_direction,
                    dim=-1,
                    eps=1e-6,
                )
            ).mean()
            action_loss = F.mse_loss(predicted_action, expert_action)
            likelihood_loss = -distribution.log_prob(expert_action).mean()
            loss = 2 * direction_loss + 5 * action_loss + 0.05 * likelihood_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy_module.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    policy_module.train()
    try:
        for epoch in range(config.behavior_cloning_epochs):
            batch = []
            losses = []
            solved = 0
            steps = 0
            policy_probability = config.behavior_cloning_max_policy_probability * (
                epoch / max(config.behavior_cloning_epochs - 1, 1)
            )
            progress = tqdm(
                range(len(train_set)),
                desc=f"Behavior cloning {epoch + 1}/{config.behavior_cloning_epochs}",
            )
            for _ in progress:
                observation = env._reset(must_load_new_subject=True)
                path_index = 0
                for _ in range(config.max_episode_steps):
                    current_path_index = path_index
                    if env.gt_path_voxels is None:
                        expert_action = _geodesic_expert_action(env)
                    else:
                        expert_action, expert_path_index = _monotonic_expert_action(
                            env,
                            current_path_index,
                        )
                    batch.append(
                        (
                            observation["actor"][0].detach().clone(),
                            observation["context"][0].detach().clone(),
                            expert_action.detach().clone(),
                        )
                    )
                    steps += 1
                    if len(batch) >= config.behavior_cloning_batch_size:
                        losses.append(update(batch))
                        batch.clear()

                    rollout_action = expert_action
                    used_policy = False
                    if torch.rand(()) < policy_probability:
                        with torch.no_grad():
                            rollout_action = _behavior_cloning_action(
                                policy_module.get_dist(observation),
                                config.behavior_cloning_action_statistic,
                            ).squeeze(0)
                        used_policy = True
                    transition = env._step(
                        TensorDict(
                            {"action": rollout_action.unsqueeze(0)},
                            batch_size=torch.Size([1]),
                            device=device,
                        )
                    )
                    if env.gt_path_voxels is not None:
                        path_index = (
                            _resynchronize_path_index(env, current_path_index)
                            if used_policy
                            else expert_path_index
                        )
                    done = bool(transition["done"].item())
                    observation = transition
                    if done:
                        solved += int(transition["info", "final_success"].item())
                        break

                progress.set_postfix(
                    loss=f"{sum(losses) / max(len(losses), 1):.3f}",
                    solved=f"{solved}/{progress.n + 1}",
                    steps=steps,
                )

            if batch:
                losses.append(update(batch))
            print(
                f"Behavior cloning epoch {epoch + 1}: "
                f"loss={sum(losses) / max(len(losses), 1):.4f}, "
                f"rollout_success={solved}/{len(train_set)}, "
                f"policy_probability={policy_probability:.2f}, steps={steps}"
            )
    finally:
        env.close()
