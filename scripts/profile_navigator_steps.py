#!/usr/bin/env python3
"""Profile Navigator environment steps on one cached nnU-Net case."""

from __future__ import annotations

import argparse
import cProfile
from itertools import cycle
import pstats
from time import perf_counter

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
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
        max_episode_steps=max(args.steps + args.warmup_steps + 1, 2048),
        coverage_reward_scale=50,
        gdt_reward_scale=1,
        r_final=50,
        r_val1=0.25,
        use_immediate_gdt_reward=True,
        terminate_on_success=True,
        num_workers=0,
        log_episode_ends=False,
    )
    dataset = NNUNetActualDataset(
        nnunet_raw=args.nnunet_raw,
        cache_dir=args.cache_dir,
        config=config,
        case_ids=[args.case_id],
    )
    subject = dataset[0]
    env = SmallBowelEnv(
        config=config,
        dataset_iterator=cycle([subject]),
        num_episodes_per_sample=1_000_000,
        num_steps_per_sample=1_000_000_000,
        device=device,
    )
    env._reset()

    generator = torch.Generator(device=device).manual_seed(12345)
    actions = torch.rand(
        args.steps + args.warmup_steps,
        1,
        3,
        generator=generator,
        device=device,
    )

    def step(index: int) -> None:
        transition = env._step(
            TensorDict(
                {"action": actions[index]},
                batch_size=torch.Size([1]),
                device=device,
            )
        )
        if bool(transition["done"].item()):
            env._reset()

    try:
        for index in range(args.warmup_steps):
            step(index)
        synchronize(device)

        profiler = cProfile.Profile()
        profiler.enable()
        started = perf_counter()
        for index in range(args.warmup_steps, len(actions)):
            step(index)
        synchronize(device)
        elapsed = perf_counter() - started
        profiler.disable()

        print(
            f"\nPROFILE case={args.case_id} device={device} steps={args.steps} "
            f"elapsed={elapsed:.3f}s steps_per_second={args.steps / elapsed:.2f} "
            f"milliseconds_per_step={elapsed * 1000 / args.steps:.3f}"
        )
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(args.top)
    finally:
        env.close()


if __name__ == "__main__":
    main()
