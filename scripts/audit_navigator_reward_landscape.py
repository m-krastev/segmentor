#!/usr/bin/env python3
"""Audit exact one-step rewards for every categorical displacement."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import cycle
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from navigator.config import Config
from navigator.dataset import NNUNetActualDataset
from navigator.environment import SmallBowelEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gdt-reward-scale", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output")
    return parser.parse_args()


def audit_case(
    config: Config,
    dataset: NNUNetActualDataset,
    case_index: int,
    device: torch.device,
    top: int,
) -> dict:
    subject = dataset[case_index]
    env = SmallBowelEnv(
        config=config,
        dataset_iterator=cycle([subject]),
        num_episodes_per_sample=1_000_000,
        num_steps_per_sample=1_000_000_000,
        device=device,
    )
    rows = []
    try:
        env._reset()
        start = tuple(int(value) for value in env.current_pos_vox)
        initial_goal_distance = float(env.initial_goal_distance)
        for action_index, requested in enumerate(config.action_displacements):
            env._reset(must_load_new_subject=False)
            before_distance = float(env.current_goal_distance)
            before_dice = float(env.current_coverage)
            executed = env._project_desired_displacement(requested)
            next_position = tuple(
                position + delta
                for position, delta in zip(env.current_pos_vox, executed)
            )
            transition = env._step(
                TensorDict(
                    {
                        "action": torch.tensor(
                            [action_index],
                            dtype=torch.long,
                            device=device,
                        )
                    },
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            after_distance = float(env.current_goal_distance)
            after_dice = float(env.current_coverage)
            distance_delta = (
                before_distance - after_distance
                if isfinite(before_distance) and isfinite(after_distance)
                else float("nan")
            )
            rows.append(
                {
                    "action_index": action_index,
                    "requested": tuple(int(value) for value in requested),
                    "executed": tuple(int(value) for value in executed),
                    "next_position": next_position,
                    "reward": float(transition["reward"].item()),
                    "gdt_delta": distance_delta,
                    "normalized_gdt_delta": (
                        distance_delta / initial_goal_distance
                        if initial_goal_distance > 0 and isfinite(distance_delta)
                        else float("nan")
                    ),
                    "dice_delta": after_dice - before_dice,
                    "next_inside_target": bool(env.seg[next_position].item())
                    if all(executed)
                    or any(executed)
                    else bool(env.seg[start].item()),
                    "step_length_vox": float(np.linalg.norm(executed)),
                    "chebyshev_step": int(max(abs(value) for value in requested)),
                }
            )
    finally:
        env.close()

    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    gdt_deltas = np.asarray([row["gdt_delta"] for row in rows], dtype=np.float64)
    shell_counts = Counter(row["chebyshev_step"] for row in rows)
    shell_balanced_weights = np.asarray(
        [
            1.0 / config.max_step_vox / shell_counts[row["chebyshev_step"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    ranking = sorted(rows, key=lambda row: row["reward"], reverse=True)
    return {
        "case": subject["id"],
        "start": start,
        "goal": tuple(int(value) for value in env.end_coord),
        "initial_goal_distance": initial_goal_distance,
        "num_actions": len(rows),
        "positive_reward_actions": int((rewards > 0).sum()),
        "forward_gdt_actions": int((gdt_deltas > 0).sum()),
        "inside_target_actions": sum(row["next_inside_target"] for row in rows),
        "reward_min": float(rewards.min()),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "initial_shell_balanced_expected_reward": float(
            np.dot(shell_balanced_weights, rewards)
        ),
        "top_actions": ranking[:top],
        "bottom_actions": ranking[-top:],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = Config(
        device=str(device),
        nnunet_raw_dir=args.nnunet_raw,
        nnunet_cache_dir=args.cache_dir,
        nnunet_seed_dir=args.seed_dir,
        annotation_free=False,
        reward_supervised=True,
        voxel_size_mm=1.5,
        patch_size_mm=24,
        max_step_displacement_mm=6,
        cumulative_path_radius_mm=9,
        endpoint_tolerance_mm=3,
        allowed_area_radius_mm=0,
        max_episode_steps=2048,
        coverage_reward_scale=50,
        gdt_reward_scale=args.gdt_reward_scale,
        r_final=50,
        r_val1=0.25,
        use_immediate_gdt_reward=True,
        terminate_on_success=True,
        action_distribution="categorical",
        deterministic_action_statistic="mode",
        memory_model="gru",
        behavior_cloning_epochs=0,
        num_workers=0,
        log_episode_ends=False,
    )
    dataset = NNUNetActualDataset(
        nnunet_raw=args.nnunet_raw,
        cache_dir=args.cache_dir,
        config=config,
        case_ids=args.case_id,
    )
    results = {
        "gdt_reward_scale": args.gdt_reward_scale,
        "cases": [
            audit_case(config, dataset, case_index, device, args.top)
            for case_index in range(len(dataset))
        ],
    }
    output = json.dumps(results, indent=2)
    print(output)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n")


if __name__ == "__main__":
    main()
