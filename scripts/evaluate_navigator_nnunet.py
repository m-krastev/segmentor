#!/usr/bin/env python
"""Evaluate a frozen Navigator checkpoint on nnU-Net raw subjects."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from navigator.config import Config
from navigator.dataset import NNUNetActualDataset
from navigator.environment import make_sb_env
from navigator.models import create_ppo_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nnunet-raw",
        type=Path,
        default=Path("data/nnunet/nnUNet_raw"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/dagger-ppo-million-v1/data/phantoms/checkpoint_102400best.pth"),
    )
    parser.add_argument("--cases", nargs="+", default=["s0001"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--interaction-type",
        choices=("mean", "mode"),
        default="mean",
        help="Mean matches the DAgger-trained checkpoint; mode reproduces the paper.",
    )
    parser.add_argument("--voxel-size-mm", type=float, default=1.5)
    parser.add_argument("--patch-size-mm", type=int, default=24)
    parser.add_argument("--max-episode-steps", type=int, default=1024)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/navigator_nnunet/cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/navigator_nnunet/evaluation"),
    )
    return parser.parse_args()


def create_policy(config: Config, checkpoint_path: Path) -> torch.nn.Module:
    device = torch.device(config.device)
    policy, _ = create_ppo_modules(
        config,
        device,
        in_channels_actor=config.observation_channels,
        in_channels_critic=config.observation_channels,
    )
    with torch.no_grad():
        policy(
            TensorDict(
                {
                    "actor": torch.zeros(
                        1,
                        config.observation_channels,
                        *config.patch_size_vox,
                        device=device,
                    ),
                    "context": torch.zeros(
                        1,
                        config.context_features,
                        device=device,
                    ),
                },
                batch_size=torch.Size([1]),
                device=device,
            )
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_module_state_dict"])
    policy.eval()
    return policy


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    config = Config(
        data_dir="nnunet-actual",
        checkpoint_dir=str(args.output_dir / "checkpoints"),
        device=args.device,
        voxel_size_mm=args.voxel_size_mm,
        patch_size_mm=args.patch_size_mm,
        max_step_displacement_mm=6,
        cumulative_path_radius_mm=6,
        allowed_area_radius_mm=0,
        goal_action_prior=0,
        success_coverage_threshold=0.55,
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
        case_ids=args.cases,
        cache_dir=args.cache_dir,
        config=config,
    )
    policy = create_policy(config, args.checkpoint)
    device = torch.device(args.device)
    env = make_sb_env(
        config,
        dataset,
        device,
        num_episodes_per_sample=1,
        num_steps_per_sample=config.max_episode_steps,
    )
    interaction_type = (
        ExplorationType.MEAN if args.interaction_type == "mean" else ExplorationType.MODE
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for _ in range(len(dataset)):
            initial = env._reset(must_load_new_subject=True)
            case_id = env._current_subject_data["id"]
            with torch.no_grad(), set_exploration_type(interaction_type):
                rollout = env.rollout(
                    config.max_episode_steps,
                    policy,
                    auto_reset=False,
                    tensordict=initial,
                )

            history = env.get_tracking_history()
            case_output_dir = args.output_dir / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)
            path_output = case_output_dir / "path.txt"
            np.savetxt(path_output, history, fmt="%d")
            endpoint_distance_vox = float(
                np.linalg.norm(np.asarray(env.current_pos_vox) - np.asarray(env.goal))
            )
            result = {
                "case": case_id,
                "steps": int(history.shape[0] - 1),
                "coverage": float(env.current_coverage),
                "success": int(rollout["next", "info", "final_success"].sum().item()),
                "endpoint_distance_mm": (endpoint_distance_vox * config.voxel_size_mm),
                "start": [int(value) for value in env.start_coord],
                "goal": [int(value) for value in env.goal],
                "final": [int(value) for value in env.current_pos_vox],
                "path": str(path_output),
            }
            results.append(result)
            print("ACTUAL_RESULT", json.dumps(result, sort_keys=True), flush=True)
    finally:
        env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "interaction_type": args.interaction_type,
        "voxel_size_mm": args.voxel_size_mm,
        "success_rate": float(np.mean([result["success"] for result in results])),
        "average_coverage": float(np.mean([result["coverage"] for result in results])),
        "average_endpoint_distance_mm": float(
            np.mean([result["endpoint_distance_mm"] for result in results])
        ),
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
