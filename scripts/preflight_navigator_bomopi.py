#!/usr/bin/env python3
"""Validate and precompute the BOMOPI caches used by Navigator."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from navigator.config import Config
from navigator.dataset import SmallBowelDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/bomopi_resampled2"),
    )
    args = parser.parse_args()

    config = Config(
        data_dir=str(args.data_dir),
        device="cuda",
        voxel_size_mm=1.5,
        patch_size_mm=24,
        max_step_displacement_mm=6,
        reward_supervised=True,
        gdt_progress_normalization="max_step",
        gdt_reward_scale=0.1,
        target_recovery_reward_scale=0.05,
        target_distance_penalty_scale=0.1,
        target_distance_penalty_radius_mm=600,
        gate_positive_shaping_on_target_segment=True,
        wall_penalty_scale=0,
        intrinsic_novelty_reward_scale=0,
        curvature_penalty_scale=0,
        episodic_cell_reward_scale=0.01,
        memory_model="gru",
        action_distribution="factorized_categorical",
        deterministic_action_statistic="mode",
    )
    dataset = SmallBowelDataset(args.data_dir, config)
    failures: list[str] = []

    for index, subject in enumerate(dataset.subjects):
        case_id = subject["id"]
        try:
            data = dataset[index]
            shape = tuple(data["image"].shape)
            if tuple(data["seg"].shape) != shape:
                raise ValueError("CT and small-bowel shapes differ")
            if tuple(data["image_features"].shape) != (4, *shape):
                raise ValueError(
                    f"navigation filters have shape {data['image_features'].shape}"
                )
            if tuple(data["target_distance"].shape) != shape:
                raise ValueError("target-distance and CT shapes differ")
            if not data["seg"][data["start_coord"]]:
                raise ValueError("start is outside the small-bowel mask")
            if not data["seg"][data["end_coord"]]:
                raise ValueError("end is outside the small-bowel mask")
            if not np.isfinite(data["gdt_end"][data["start_coord"]]):
                raise ValueError("start and end are disconnected")
            print(
                f"PASS {case_id}: shape={shape}, start={data['start_coord']}, "
                f"end={data['end_coord']}",
                flush=True,
            )
            del data
            gc.collect()
        except Exception as error:  # continue to report all unusable subjects
            failures.append(f"{case_id}: {error}")
            print(f"FAIL {failures[-1]}", flush=True)

    if failures:
        raise SystemExit(
            "BOMOPI preflight failed:\n" + "\n".join(f"- {failure}" for failure in failures)
        )
    print(f"BOMOPI preflight passed for {len(dataset)} subjects.", flush=True)


if __name__ == "__main__":
    main()
