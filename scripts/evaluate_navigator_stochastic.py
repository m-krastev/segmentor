#!/usr/bin/env python3
"""Evaluate independent stochastic rollouts from a saved Navigator policy."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import Subset
from torchrl.envs.utils import ExplorationType

# A worktree can intentionally share a uv environment whose editable install
# points at another checkout. Prefer the source tree adjacent to this script so
# an evaluation always reconstructs the checkpoint with the code being audited.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from navigator.config import Config
from navigator.dataset import SmallBowelDataset
from navigator.models import create_ppo_modules
import navigator.train as train_module
from navigator.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-seed", type=int, action="append", required=True)
    parser.add_argument("--expected-case-id", action="append")
    parser.add_argument(
        "--selection",
        choices=("stochastic", "mode"),
        default="stochastic",
        help="Sample from the policy or use its deterministic categorical mode.",
    )
    return parser.parse_args()


def config_from_checkpoint(saved: dict, output_dir: Path) -> Config:
    initializable = {field.name for field in fields(Config) if field.init}
    values = {
        key: value
        for key, value in saved.items()
        if key in initializable
    }
    values.update(
        {
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "validation_output_dir": str(output_dir),
            "validation_save_paths": False,
            "track_wandb": False,
            "track_tensorboard": False,
            "eval_only": True,
        }
    )
    return Config(**values)


def initialize_policy(config: Config):
    policy_module, _ = create_ppo_modules(
        config,
        config.device,
        qnets=config.td3,
        in_channels_actor=config.observation_channels,
        in_channels_critic=config.observation_channels,
    )
    dummy = {
        "actor": torch.zeros(
            1,
            config.observation_channels,
            *config.patch_size_vox,
            device=config.device,
        ),
        "context": torch.zeros(
            1,
            config.context_features,
            device=config.device,
        ),
        "is_init": torch.ones(1, 1, dtype=torch.bool, device=config.device),
        "action_mask": torch.ones(
            1,
            config.categorical_action_count,
            dtype=torch.bool,
            device=config.device,
        ),
    }
    if config.memory_model == "gru":
        dummy["recurrent_state"] = torch.zeros(
            1,
            config.memory_num_layers,
            config.memory_hidden_size,
            device=config.device,
        )
    elif config.memory_model == "s5":
        dummy["s5_state"] = torch.zeros(
            1,
            config.s5_state_size,
            2,
            device=config.device,
        )
    with torch.no_grad():
        policy_module(
            TensorDict(
                dummy,
                batch_size=torch.Size([1]),
                device=config.device,
            )
        )
    return policy_module


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = config_from_checkpoint(checkpoint["config"], args.output_dir)
    seed_everything(config.seed)
    dataset = SmallBowelDataset(config.data_dir, config)
    indices = np.arange(len(dataset))
    if config.shuffle_dataset:
        np.random.shuffle(indices)
    train_size = int(len(dataset) * config.train_val_split)
    val_indices = indices[train_size:]
    val_case_ids = [dataset.subjects[index]["id"] for index in val_indices]
    if args.expected_case_id:
        expected = sorted(args.expected_case_id)
        if sorted(val_case_ids) != expected:
            raise ValueError(
                f"Validation split is {val_case_ids}, expected {expected}"
            )
    val_set = Subset(dataset, val_indices)

    policy_module = initialize_policy(config)
    policy_module.load_state_dict(checkpoint["policy_module_state_dict"])
    policy_module.to(config.device)
    policy_module.eval()

    # This process-local override cannot affect training or checkpoint
    # selection. Each call below executes exactly one independently seeded
    # rollout per held-out subject.
    if args.selection == "stochastic":
        train_module.deterministic_exploration_type = (
            lambda _: ExplorationType.RANDOM
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for rollout_seed in args.rollout_seed:
        seed_everything(rollout_seed)
        train_module.validation_loop_torchrl(
            policy_module,
            config,
            val_set,
            config.device,
            global_step=rollout_seed,
        )
        with open(args.output_dir / f"metrics_{rollout_seed}.json") as metric_file:
            payloads.append(json.load(metric_file))

    summary = {
        "checkpoint": str(args.checkpoint),
        "selection": (
            "stochastic_sample"
            if args.selection == "stochastic"
            else "deterministic_mode"
        ),
        "rollout_seeds": args.rollout_seed,
        "validation_cases": val_case_ids,
        "num_rollouts": len(payloads),
        "num_case_episodes": len(payloads) * len(val_case_ids),
        "mean_dice": float(
            np.mean(
                [
                    dice
                    for payload in payloads
                    for dice in payload["coverage"]
                ]
            )
        ),
        "mean_endpoint_distance_mm": float(
            np.mean(
                [
                    distance
                    for payload in payloads
                    for distance in payload["endpoint_distance_mm"]
                ]
            )
        ),
        "endpoint_reach_rate": float(
            np.mean(
                [
                    reached
                    for payload in payloads
                    for reached in payload["endpoint_reached"]
                ]
            )
        ),
        "traversal_success_rate": float(
            np.mean(
                [
                    success
                    for payload in payloads
                    for success in payload["success"]
                ]
            )
        ),
        "mean_recent_unique_position_fraction": float(
            np.mean(
                [
                    diversity
                    for payload in payloads
                    for diversity in payload[
                        "recent_unique_position_fraction"
                    ]
                ]
            )
        ),
        "mean_boundary_state_fraction": float(
            np.mean(
                [
                    boundary
                    for payload in payloads
                    for boundary in payload["boundary_state_fraction"]
                ]
            )
        ),
        "rollouts": payloads,
    }
    summary_path = args.output_dir / (
        "stochastic_summary.json"
        if args.selection == "stochastic"
        else "deterministic_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rollouts"}, indent=2))


if __name__ == "__main__":
    main()
