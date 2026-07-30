#!/usr/bin/env python3
"""Audit task feasibility and immediate reward alignment on BOMOPI routes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import cycle
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from skimage.draw import line_nd
from tensordict import TensorDict

from navigator.config import Config
from navigator.dataset import SmallBowelDataset
from navigator.environment import REWARD_COMPONENT_INFO_KEYS, SmallBowelEnv
from navigator.metrics import compute_path_metrics, rasterize_path
from navigator.oracle import (
    _mask_path,
    compress_route_with_action_support,
    skeleton_covering_route,
)
from navigator.rewards import coverage_potential_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/bomopi_resampled2_unique-v1"),
    )
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--samples-per-case", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument(
        "--reward-contract",
        choices=("shin_normalized_repaired", "potential"),
        default="shin_normalized_repaired",
    )
    parser.add_argument("--potential-coverage-scale", type=float, default=50.0)
    parser.add_argument("--potential-gdt-scale", type=float, default=1.0)
    parser.add_argument("--potential-revisit-scale", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def json_default(value):
    """Convert NumPy scalar results without silently coercing other objects."""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def exact_config(args: argparse.Namespace) -> Config:
    reward_options = {}
    if args.reward_contract == "potential":
        reward_options = {
            "use_immediate_gdt_reward": True,
            "gate_positive_shaping_on_target_segment": True,
            "coverage_reward_scale": args.potential_coverage_scale,
            "gdt_reward_scale": args.potential_gdt_scale,
            "gdt_progress_normalization": "max_step",
            "target_recovery_reward_scale": 0.2,
            "target_distance_penalty_scale": 0.1,
            "target_distance_penalty_radius_mm": 60.0,
            "step_penalty": 0.01,
            "revisit_penalty_scale": args.potential_revisit_scale,
            "wall_penalty_scale": 0.0,
            "r_val1": 0.0,
            "r_val2": 1.0,
            "r_zero_mov": 1.0,
            "terminal_success_bonus": 50.0,
            "terminal_failure_penalty": 0.0,
            "episodic_cell_reward_scale": 0.0,
        }
    return Config(
        data_dir=str(args.data_dir),
        device=args.device,
        voxel_size_mm=1.5,
        patch_size_mm=60,
        max_step_displacement_mm=9,
        cumulative_path_radius_mm=6,
        endpoint_tolerance_mm=3,
        allowed_area_radius_mm=0,
        max_episode_steps=args.horizon,
        reward_supervised=True,
        reward_contract=args.reward_contract,
        policy_observation_contract="shin_068_repaired",
        action_distribution="masked_categorical",
        categorical_action_support="direction_length",
        deterministic_action_statistic="mode",
        memory_model="gru",
        gamma=0.99,
        success_coverage_threshold=0.40,
        num_workers=0,
        log_episode_ends=False,
        **reward_options,
    )


def route_metrics(
    subject: dict,
    history: np.ndarray,
    config: Config,
) -> dict[str, float | bool | int]:
    metrics = compute_path_metrics(
        subject["seg"],
        history,
        subject["end_coord"],
        tuple(float(value) for value in subject["spacing"]),
        config.cumulative_path_radius_mm,
        config.endpoint_tolerance_mm,
        config.success_coverage_threshold,
    )
    return {
        "dice": metrics.dice,
        "endpoint_distance_mm": metrics.endpoint_distance_mm,
        "endpoint_reached": metrics.endpoint_reached,
        "traversal_success": metrics.traversal_success,
        "path_voxels": metrics.path_voxels,
        "target_intersection": metrics.target_intersection,
    }


def sampled_action_indices(num_actions: int, requested: int) -> np.ndarray:
    if num_actions < 1:
        return np.empty(0, dtype=np.int64)
    count = min(num_actions, max(1, requested))
    return np.unique(
        np.linspace(0, num_actions - 1, num=count, dtype=np.int64)
    )


def initialize_prefix_path_state(
    env: SmallBowelEnv,
    history: np.ndarray,
) -> np.ndarray:
    """Reconstruct the environment's exact dilated and undilated path state."""

    env.cumulative_path_mask.zero_()
    env.cumulative_path_mask_pen[:] = 0
    env.path_voxels = 0
    env.path_target_intersection = 0
    env._add_path_segment(history[0])
    if len(history) > 1:
        prefix_centerline = rasterize_path(env.image.shape, history)
        env.cumulative_path_mask_pen[prefix_centerline] = 1
        for start, end in zip(history[:-1], history[1:]):
            env._add_path_segment(
                line_nd(tuple(start), tuple(end), endpoint=True)
            )
    else:
        prefix_centerline = np.zeros(env.image.shape, dtype=bool)
    env.current_coverage = float(env._get_final_coverage())
    return prefix_centerline


def coverage_after_candidate_segment(
    env: SmallBowelEnv,
    segment: tuple[np.ndarray, ...],
) -> float:
    """Return post-action Dice without mutating the current path mask."""

    points = np.asarray(segment, dtype=np.int64).T
    relative_points = points - points[0]
    cache_key = tuple(
        tuple(int(value) for value in point) for point in relative_points
    )
    relative_coordinates = env._dilated_line_offsets.get(cache_key)
    if relative_coordinates is None:
        points_tensor = torch.as_tensor(
            relative_points,
            dtype=torch.long,
            device=env.device,
        )
        relative_coordinates = torch.unique(
            (
                points_tensor[:, None, :]
                + env.path_dilation_offsets[None, :, :]
            ).reshape(-1, 3),
            dim=0,
        )
        env._dilated_line_offsets[cache_key] = relative_coordinates
    coordinates = relative_coordinates + torch.as_tensor(
        points[0],
        dtype=torch.long,
        device=env.device,
    )
    valid = (
        (coordinates >= 0) & (coordinates < env._volume_shape_tensor)
    ).all(dim=1)
    coordinates = coordinates[valid]
    indices = tuple(coordinates[:, axis] for axis in range(3))
    new_coordinates = coordinates[env.cumulative_path_mask[indices] == 0]
    if not new_coordinates.numel():
        return float(env.current_coverage)
    new_indices = tuple(new_coordinates[:, axis] for axis in range(3))
    next_path_voxels = env.path_voxels + int(new_coordinates.shape[0])
    next_intersection = env.path_target_intersection + int(
        env.current_target_mask[new_indices].sum().item()
    )
    denominator = env.target_voxels + next_path_voxels
    return float(2 * next_intersection / denominator) if denominator else 0.0


def evaluate_actions_at_state(
    env: SmallBowelEnv,
    config: Config,
    history: np.ndarray,
    oracle_action: tuple[int, int, int],
    route_action_number: int,
) -> dict:
    current = tuple(int(value) for value in history[-1])
    env.current_pos_vox = current
    env.tracking_path_history = [
        tuple(int(value) for value in position) for position in history
    ]
    env.current_step_count = route_action_number
    prefix_centerline = initialize_prefix_path_state(env, history)

    finite_prefix = np.asarray(env.gdt[prefix_centerline])
    finite_prefix = finite_prefix[np.isfinite(finite_prefix)]
    env.max_gdt_achieved = (
        float(finite_prefix.max())
        if finite_prefix.size
        else float(env.gdt[current])
    )
    base_maximum_gdt = float(env.max_gdt_achieved)
    base_wall_gradient = env.wall_gradient
    current_goal_distance = float(env.goal_distance_map[current])
    current_target_distance = float(env.target_distance_map[current])

    rows = []
    with torch.no_grad():
        for action_index, displacement in enumerate(config.action_displacements):
            next_position = tuple(
                coordinate + delta
                for coordinate, delta in zip(current, displacement)
            )
            if not env._is_valid_pos(next_position):
                continue
            env.max_gdt_achieved = base_maximum_gdt
            env.wall_gradient = base_wall_gradient
            env.current_goal_distance = current_goal_distance
            env.current_target_distance = current_target_distance
            env._reset_step_reward_components()
            reward, segment = env._calculate_reward(
                displacement,
                next_position,
            )
            if (
                config.reward_contract == "potential"
                and segment
            ):
                next_coverage = coverage_after_candidate_segment(env, segment)
                coverage_reward = coverage_potential_reward(
                    env.current_coverage,
                    next_coverage,
                    config.coverage_reward_scale,
                )
                if (
                    coverage_reward <= 0
                    or not config.gate_positive_shaping_on_target_segment
                    or env._segment_is_on_target(segment)
                ):
                    reward += env._reward_term(
                        "reward_coverage",
                        coverage_reward,
                    )
            next_goal_distance = float(env.goal_distance_map[next_position])
            rows.append(
                {
                    "action_index": action_index,
                    "displacement": tuple(int(value) for value in displacement),
                    "reward": float(reward.item()),
                    "on_target": env._segment_is_on_target(segment),
                    "forward_goal": (
                        isfinite(next_goal_distance)
                        and next_goal_distance < current_goal_distance
                        and env._segment_is_on_target(segment)
                    ),
                    "components": {
                        key: float(env.last_reward_components[key].item())
                        for key in REWARD_COMPONENT_INFO_KEYS
                    },
                }
            )

    oracle_row = next(
        row for row in rows if row["displacement"] == oracle_action
    )
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    oracle_reward = float(oracle_row["reward"])
    tolerance = 1e-9
    oracle_rank = 1 + int((rewards > oracle_reward + tolerance).sum())
    best_index = int(np.argmax(rewards))
    best = rows[best_index]
    oracle_segment = line_nd(
        current,
        tuple(
            coordinate + delta
            for coordinate, delta in zip(current, oracle_action)
        ),
        endpoint=True,
    )
    oracle_tail = tuple(axis[1:] for axis in oracle_segment)
    oracle_revisit = bool(
        np.asarray(env.cumulative_path_mask_pen[oracle_tail]).any()
    )
    return {
        "route_action_number": route_action_number,
        "position": current,
        "oracle_action": oracle_action,
        "oracle_action_reward": oracle_reward,
        "oracle_action_rank": oracle_rank,
        "oracle_action_normalized_rank": (
            (oracle_rank - 1) / max(len(rows) - 1, 1)
        ),
        "oracle_action_is_reward_maximum": oracle_rank == 1,
        "oracle_action_is_top5": oracle_rank <= 5,
        "oracle_action_positive": oracle_reward > 0,
        "oracle_action_on_target": oracle_row["on_target"],
        "oracle_action_forward_goal": oracle_row["forward_goal"],
        "oracle_action_revisits_prefix": oracle_revisit,
        "oracle_action_components": oracle_row["components"],
        "valid_action_count": len(rows),
        "positive_reward_action_count": int((rewards > 0).sum()),
        "mean_action_reward": float(rewards.mean()),
        "maximum_action_reward": float(rewards.max()),
        "oracle_reward_regret": float(rewards.max() - oracle_reward),
        "reward_maximum_action": best["displacement"],
        "reward_maximum_on_target": best["on_target"],
        "reward_maximum_forward_goal": best["forward_goal"],
        "reward_maximum_components": best["components"],
    }


def rollout_compact_route(
    subject: dict,
    config: Config,
    compact_route: np.ndarray,
) -> dict:
    """Execute the exact compact covering route under the registered horizon."""

    env = SmallBowelEnv(
        config=config,
        dataset_iterator=cycle([subject]),
        num_episodes_per_sample=1_000_000,
        num_steps_per_sample=1_000_000_000,
        device=torch.device(config.device),
    )
    component_totals = {
        key: 0.0 for key in REWARD_COMPONENT_INFO_KEYS
    }
    rewards = []
    action_lookup = {
        displacement: index
        for index, displacement in enumerate(config.action_displacements)
    }
    try:
        env._reset()
        done = False
        for current, following in zip(compact_route[:-1], compact_route[1:]):
            displacement = tuple(
                int(value) for value in (following - current)
            )
            action_index = action_lookup[displacement]
            transition = env._step(
                TensorDict(
                    {
                        "action": torch.tensor(
                            [action_index],
                            dtype=torch.long,
                            device=env.device,
                        )
                    },
                    batch_size=torch.Size([1]),
                    device=env.device,
                )
            )
            rewards.append(float(transition["reward"].item()))
            for key in REWARD_COMPONENT_INFO_KEYS:
                component_totals[key] += float(
                    transition["info", key].item()
                )
            done = bool(transition["done"].item())
            if done:
                break
        history = env.get_tracking_history()
        metrics = route_metrics(subject, history, config)
        return {
            "actions_executed": len(rewards),
            "route_actions_available": int(len(compact_route) - 1),
            "route_completed": len(rewards) == len(compact_route) - 1,
            "environment_done": done,
            "total_reward": float(sum(rewards)),
            "discounted_return": float(
                sum(
                    (config.gamma**step) * reward
                    for step, reward in enumerate(rewards)
                )
            ),
            "positive_reward_step_fraction": float(
                np.mean(np.asarray(rewards) > 0)
            ),
            "reward_component_totals": component_totals,
            "metrics": metrics,
            "final_position": tuple(
                int(value) for value in env.current_pos_vox
            ),
        }
    finally:
        env.close()


def summarize_states(states: list[dict]) -> dict:
    if not states:
        return {"num_states": 0}
    components = {
        key: float(
            np.mean(
                [state["oracle_action_components"][key] for state in states]
            )
        )
        for key in REWARD_COMPONENT_INFO_KEYS
    }
    return {
        "num_states": len(states),
        "oracle_action_reward_mean": float(
            np.mean([state["oracle_action_reward"] for state in states])
        ),
        "oracle_action_reward_min": float(
            np.min([state["oracle_action_reward"] for state in states])
        ),
        "oracle_action_positive_rate": float(
            np.mean([state["oracle_action_positive"] for state in states])
        ),
        "oracle_action_reward_maximum_rate": float(
            np.mean(
                [state["oracle_action_is_reward_maximum"] for state in states]
            )
        ),
        "oracle_action_top5_rate": float(
            np.mean([state["oracle_action_is_top5"] for state in states])
        ),
        "oracle_action_mean_normalized_rank": float(
            np.mean(
                [state["oracle_action_normalized_rank"] for state in states]
            )
        ),
        "oracle_action_mean_regret": float(
            np.mean([state["oracle_reward_regret"] for state in states])
        ),
        "oracle_action_revisit_rate": float(
            np.mean(
                [state["oracle_action_revisits_prefix"] for state in states]
            )
        ),
        "reward_maximum_on_target_rate": float(
            np.mean([state["reward_maximum_on_target"] for state in states])
        ),
        "reward_maximum_forward_goal_rate": float(
            np.mean(
                [state["reward_maximum_forward_goal"] for state in states]
            )
        ),
        "mean_positive_reward_action_fraction": float(
            np.mean(
                [
                    state["positive_reward_action_count"]
                    / state["valid_action_count"]
                    for state in states
                ]
            )
        ),
        "oracle_action_component_means": components,
    }


def audit_case(
    subject: dict,
    config: Config,
    args: argparse.Namespace,
) -> dict:
    dense_covering_route = skeleton_covering_route(
        subject["seg"],
        subject["start_coord"],
        subject["end_coord"],
    )
    compact_route, dense_indices = compress_route_with_action_support(
        subject["seg"],
        dense_covering_route,
        config.action_displacements,
        max_route_lookahead=config.max_step_vox,
    )
    direct_route = _mask_path(
        np.asarray(subject["seg"], dtype=bool),
        tuple(subject["start_coord"]),
        tuple(subject["end_coord"]),
    )
    compact_direct_route, _ = compress_route_with_action_support(
        subject["seg"],
        direct_route,
        config.action_displacements,
        max_route_lookahead=config.max_step_vox,
    )
    prefix_waypoints = compact_route[: min(args.horizon, len(compact_route) - 1) + 1]

    env = SmallBowelEnv(
        config=config,
        dataset_iterator=cycle([subject]),
        num_episodes_per_sample=1_000_000,
        num_steps_per_sample=1_000_000_000,
        device=torch.device(args.device),
    )
    states = []
    try:
        env._reset()
        action_lookup = {
            displacement: index
            for index, displacement in enumerate(config.action_displacements)
        }
        auditable_actions = min(args.horizon, len(compact_route) - 1)
        for action_number in sampled_action_indices(
            auditable_actions,
            args.samples_per_case,
        ):
            current = compact_route[action_number]
            following = compact_route[action_number + 1]
            oracle_action = tuple(
                int(value) for value in (following - current)
            )
            if oracle_action not in action_lookup:
                raise AssertionError(
                    f"Compressed route emitted unsupported action {oracle_action}"
                )
            states.append(
                evaluate_actions_at_state(
                    env,
                    config,
                    compact_route[: action_number + 1],
                    oracle_action,
                    int(action_number),
                )
            )
    finally:
        env.close()

    novel_states = [
        state for state in states
        if not state["oracle_action_revisits_prefix"]
    ]
    revisit_states = [
        state for state in states
        if state["oracle_action_revisits_prefix"]
    ]
    return {
        "case": subject["id"],
        "shape": tuple(int(value) for value in subject["seg"].shape),
        "target_voxels": int(np.asarray(subject["seg"]).sum()),
        "dense_covering_route_voxels": int(len(dense_covering_route)),
        "compact_covering_route_actions": int(len(compact_route) - 1),
        "compact_route_dense_indices_final": int(dense_indices[-1]),
        "compact_covering_route_fits_horizon": (
            len(compact_route) - 1 <= args.horizon
        ),
        "direct_endpoint_route": route_metrics(subject, direct_route, config),
        "compact_direct_endpoint_route_actions": int(
            len(compact_direct_route) - 1
        ),
        "exact_compact_direct_rollout": rollout_compact_route(
            subject,
            config,
            compact_direct_route,
        ),
        "full_compact_covering_route": route_metrics(
            subject,
            compact_route,
            config,
        ),
        "horizon_compact_covering_prefix": route_metrics(
            subject,
            prefix_waypoints,
            config,
        ),
        "exact_compact_oracle_rollout": rollout_compact_route(
            subject,
            config,
            compact_route,
        ),
        "sampled_reward_states": summarize_states(states),
        "sampled_novel_oracle_steps": summarize_states(novel_states),
        "sampled_revisit_oracle_steps": summarize_states(revisit_states),
        "states": states,
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_case < 1:
        raise ValueError("samples-per-case must be positive")
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    config = exact_config(args)
    if len(config.action_displacements) != 156:
        raise AssertionError(
            "Expected exact 26-direction x six-length compact support, got "
            f"{len(config.action_displacements)} actions"
        )
    dataset = SmallBowelDataset(args.data_dir, config)
    requested = set(args.case_id)
    available = {subject["id"] for subject in dataset.subjects}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Requested BOMOPI cases are unavailable: {missing}")

    cases = []
    for index, metadata in enumerate(dataset.subjects):
        if metadata["id"] not in requested:
            continue
        cases.append(audit_case(dataset[index], config, args))

    aggregate_groups: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        for key in (
            "sampled_reward_states",
            "sampled_novel_oracle_steps",
            "sampled_revisit_oracle_steps",
        ):
            aggregate_groups[key].extend(
                state
                for state in case["states"]
                if (
                    key == "sampled_reward_states"
                    or (
                        key == "sampled_novel_oracle_steps"
                        and not state["oracle_action_revisits_prefix"]
                    )
                    or (
                        key == "sampled_revisit_oracle_steps"
                        and state["oracle_action_revisits_prefix"]
                    )
                )
            )
    result = {
        "protocol": {
            "reward_contract": config.reward_contract,
            "policy_observation_contract": config.policy_observation_contract,
            "categorical_action_support": config.categorical_action_support,
            "num_actions": len(config.action_displacements),
            "max_step_vox": config.max_step_vox,
            "voxel_size_mm": config.voxel_size_mm,
            "path_radius_mm": config.cumulative_path_radius_mm,
            "horizon": args.horizon,
            "samples_per_case": args.samples_per_case,
            "coverage_reward_scale": config.coverage_reward_scale,
            "gdt_reward_scale": config.gdt_reward_scale,
            "revisit_penalty_scale": config.revisit_penalty_scale,
        },
        "aggregate": {
            key: summarize_states(states)
            for key, states in aggregate_groups.items()
        },
        "cases": cases,
    }
    encoded = json.dumps(result, indent=2, default=json_default)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
        console_result = {
            "protocol": result["protocol"],
            "aggregate": result["aggregate"],
            "cases": [
                {
                    key: value
                    for key, value in case.items()
                    if key != "states"
                }
                for case in cases
            ],
            "output": str(args.output),
        }
        print(json.dumps(console_result, indent=2, default=json_default))
    else:
        print(encoded)


if __name__ == "__main__":
    main()
