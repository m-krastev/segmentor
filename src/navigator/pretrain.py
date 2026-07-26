"""Behavior-cloning warm start for the Navigator policy."""

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tqdm import tqdm

from .environment import make_sb_env


def _monotonic_expert_action(env, path_index: int) -> tuple[torch.Tensor, int]:
    """Return an action toward the next centerline waypoint.

    ``path_index`` is a monotonic segment cursor, not a nearest-point lookup.
    The phantom paths are sparse (adjacent waypoints can be more than one
    action apart), so the expert keeps aiming at the next waypoint until it is
    within one action radius, then advances. This also prevents jumps between
    spatially adjacent bowel loops.
    """
    path = env.gt_path_voxels
    if path is None or len(path) < 2:
        raise ValueError("Behavior cloning requires a non-empty ground-truth path.")

    current = np.asarray(env.current_pos_vox)
    while path_index < len(path) - 1:
        next_distance = float(np.linalg.norm(path[path_index + 1] - current))
        if next_distance > env.config.max_step_vox:
            break
        path_index += 1
    if path_index >= len(path) - 1:
        return torch.full((3,), 0.5, dtype=env.dtype, device=env.device), path_index

    target_index = path_index + 1
    displacement = path[target_index] - current
    # The environment treats action magnitude as irrelevant and normalizes the
    # largest component before projection. Encode a canonical full-range
    # direction so behavioral cloning does not waste capacity fitting arbitrary
    # waypoint distances.
    displacement = displacement / max(float(np.max(np.abs(displacement))), 1.0)
    action = torch.as_tensor(
        (displacement + 1.0) / 2.0,
        dtype=env.dtype,
        device=env.device,
    ).clamp(0.0, 1.0)
    return action, path_index


def pretrain_behavior_cloning(policy_module, config, train_set, device) -> None:
    """Fit the stochastic actor to training-set centerline actions.

    Ground-truth paths are used only for this training warm start. Validation
    subjects remain held out, and subsequent PPO training uses the unchanged
    environment reward and strict success criterion.
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
            predicted_action = distribution.mean
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
                    expert_action, path_index = _monotonic_expert_action(env, path_index)
                    if path_index >= len(env.gt_path_voxels) - 1:
                        break
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
                    if torch.rand(()) < policy_probability:
                        with torch.no_grad():
                            rollout_action = (
                                policy_module.get_dist(observation).mean.squeeze(0)
                            )
                    transition = env._step(
                        TensorDict(
                            {"action": rollout_action.unsqueeze(0)},
                            batch_size=torch.Size([1]),
                            device=device,
                        )
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
