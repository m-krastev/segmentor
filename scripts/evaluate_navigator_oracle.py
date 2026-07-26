#!/usr/bin/env python
"""Evaluate the mask-constrained geodesic oracle on nnU-Net subjects."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from navigator.config import Config
from navigator.dataset import NNUNetActualDataset
from navigator.environment import make_sb_env
from navigator.metrics import compute_path_metrics
from navigator.oracle import skeleton_covering_route
from navigator.pretrain import _geodesic_expert_action, _monotonic_expert_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=Path("data/nnunet/nnUNet_raw"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/navigator_nnunet/cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/navigator_nnunet/oracle"),
    )
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--case-ids-file", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--voxel-size-mm", type=float, default=1.5)
    parser.add_argument("--patch-size-mm", type=int, default=24)
    parser.add_argument("--max-episode-steps", type=int, default=2048)
    parser.add_argument("--success-dice", type=float, default=0.40)
    parser.add_argument(
        "--oracle",
        choices=("geodesic", "skeleton"),
        default="geodesic",
    )
    parser.add_argument("--path-radius-mm", type=float, default=9.0)
    parser.add_argument("--endpoint-tolerance-mm", type=float, default=3.0)
    return parser.parse_args()


def read_case_ids(args: argparse.Namespace) -> list[str] | None:
    case_ids = list(args.cases or [])
    if args.case_ids_file:
        case_ids.extend(
            line.strip()
            for line in args.case_ids_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return sorted(set(case_ids)) or None


def main() -> None:
    args = parse_args()
    case_ids = read_case_ids(args)
    config = Config(
        data_dir="nnunet-actual-oracle",
        checkpoint_dir=str(args.output_dir / "checkpoints"),
        device=args.device,
        voxel_size_mm=args.voxel_size_mm,
        patch_size_mm=args.patch_size_mm,
        max_step_displacement_mm=6,
        cumulative_path_radius_mm=args.path_radius_mm,
        endpoint_tolerance_mm=args.endpoint_tolerance_mm,
        terminate_on_success=args.oracle != "skeleton",
        allowed_area_radius_mm=0,
        success_coverage_threshold=args.success_dice,
        coverage_reward_scale=50,
        r_val2=1,
        r_zero_mov=1,
        max_episode_steps=args.max_episode_steps,
        num_workers=0,
        shuffle_dataset=False,
        track_wandb=False,
    )
    dataset = NNUNetActualDataset(
        nnunet_raw=args.nnunet_raw,
        case_ids=case_ids,
        cache_dir=args.cache_dir,
        config=config,
    )
    env = make_sb_env(
        config,
        dataset,
        torch.device(args.device),
        num_episodes_per_sample=1,
        num_steps_per_sample=config.max_episode_steps,
        check_env=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for _ in range(len(dataset)):
            observation = env._reset(must_load_new_subject=True)
            case_id = env._current_subject_data["id"]
            transition = None
            endpoint_reached = False
            dense_route = None
            path_index = 0
            if args.oracle == "skeleton":
                dense_route = skeleton_covering_route(
                    env.seg.numpy(force=True),
                    env.start_coord,
                    env.end_coord,
                )
                env.gt_path_voxels = dense_route
            for _ in range(config.max_episode_steps):
                if args.oracle == "geodesic":
                    action = _geodesic_expert_action(env)
                else:
                    action, path_index = _monotonic_expert_action(env, path_index)
                transition = env._step(
                    TensorDict(
                        {"action": action.unsqueeze(0)},
                        batch_size=torch.Size([1]),
                        device=env.device,
                    )
                )
                observation = transition
                endpoint_distance_vox = math.dist(env.current_pos_vox, env.goal)
                endpoint_reached = (
                    endpoint_distance_vox <= config.endpoint_tolerance_vox
                )
                route_complete = dense_route is not None and path_index >= len(dense_route) - 2
                if (
                    bool(transition["done"].item())
                    or (args.oracle == "geodesic" and endpoint_reached)
                    or (route_complete and endpoint_reached)
                ):
                    break

            history = env.get_tracking_history()
            case_output = args.output_dir / case_id
            case_output.mkdir(parents=True, exist_ok=True)
            path_output = case_output / "path.txt"
            np.savetxt(path_output, history, fmt="%d")
            metrics = compute_path_metrics(
                env.seg.numpy(force=True),
                history,
                env.goal,
                tuple(float(value) for value in env.spacing),
                config.cumulative_path_radius_mm,
                config.endpoint_tolerance_mm,
                config.success_coverage_threshold,
            )
            endpoint_distance_mm = metrics.endpoint_distance_mm
            dice = metrics.dice
            result = {
                "case": case_id,
                "steps": int(env.current_step_count),
                "valid_moves": int(history.shape[0] - 1),
                "dense_route_voxels": (int(len(dense_route)) if dense_route is not None else None),
                "dice": dice,
                "endpoint_reached": int(metrics.endpoint_reached),
                "traversal_success": int(metrics.traversal_success),
                "endpoint_distance_mm": endpoint_distance_mm,
                "start": [int(value) for value in env.start_coord],
                "goal": [int(value) for value in env.goal],
                "final": [int(value) for value in env.current_pos_vox],
                "path": str(path_output),
            }
            results.append(result)
            print("ORACLE_RESULT", json.dumps(result, sort_keys=True), flush=True)
    finally:
        env.close()

    summary = {
        "oracle": args.oracle,
        "path_radius_mm": config.cumulative_path_radius_mm,
        "path_radius_geometry": "euclidean_physical",
        "endpoint_tolerance_mm": config.endpoint_tolerance_mm,
        "success_dice": config.success_coverage_threshold,
        "num_cases": len(results),
        "average_dice": float(np.mean([result["dice"] for result in results])),
        "minimum_dice": float(np.min([result["dice"] for result in results])),
        "endpoint_reach_rate": float(np.mean([result["endpoint_reached"] for result in results])),
        "traversal_success_rate": float(
            np.mean([result["traversal_success"] for result in results])
        ),
        "average_endpoint_distance_mm": float(
            np.mean([result["endpoint_distance_mm"] for result in results])
        ),
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("ORACLE_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
