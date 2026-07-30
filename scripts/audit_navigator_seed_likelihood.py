#!/usr/bin/env python3
"""Audit label-free, seed-conditioned bowel-likelihood features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score


CHANNEL_NAMES = (
    "ct_clipped",
    "dark_tubularity",
    "bright_tubularity",
    "band_pass",
    "gradient",
)
CHANNEL_SUBSETS = {
    "ct": ("ct_clipped",),
    "dark": ("dark_tubularity",),
    "ct_dark": ("ct_clipped", "dark_tubularity"),
    "all": CHANNEL_NAMES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-case-id", action="append", required=True)
    parser.add_argument("--test-case-id", action="append", required=True)
    parser.add_argument("--voxel-size-mm", type=float, default=1.5)
    parser.add_argument("--seed-radius-mm", type=float, default=3.0)
    parser.add_argument("--shell-radius-mm", type=float, default=30.0)
    parser.add_argument("--max-samples-per-class", type=int, default=25_000)
    parser.add_argument("--global-scale-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def ball_coordinates(
    center: np.ndarray,
    shape: tuple[int, int, int],
    radius_vox: float,
) -> tuple[np.ndarray, ...]:
    lower = np.maximum(np.floor(center - radius_vox).astype(int), 0)
    upper = np.minimum(np.ceil(center + radius_vox).astype(int), np.asarray(shape) - 1)
    grid = np.meshgrid(
        *[
            np.arange(start, stop + 1, dtype=int)
            for start, stop in zip(lower, upper)
        ],
        indexing="ij",
    )
    coordinates = np.stack(grid, axis=-1)
    keep = np.linalg.norm(coordinates - center, axis=-1) <= radius_vox
    selected = coordinates[keep]
    return tuple(selected[:, axis] for axis in range(3))


def load_case(
    data_dir: Path,
    case_id: str,
    *,
    voxel_size_mm: float,
    seed_radius_mm: float,
    shell_radius_mm: float,
    max_samples_per_class: int,
    global_scale_samples: int,
    rng: np.random.Generator,
) -> dict:
    case_dir = data_dir / case_id
    ct = nib.load(case_dir / "ct.nii.gz").get_fdata(dtype=np.float32)
    segmentation = (
        nib.load(case_dir / "segmentations" / "small_bowel.nii.gz").get_fdata(
            dtype=np.float32
        )
        > 0
    )
    filters = nib.load(
        case_dir / "cache" / "navigation_filters-v1-mm-3-6-9.nii"
    ).get_fdata(dtype=np.float32)
    if filters.shape != (*ct.shape, 4):
        raise ValueError(f"{case_id}: unexpected filter shape {filters.shape}")
    start_end = np.loadtxt(
        case_dir / "cache" / "start_end.npy",
        dtype=int,
    ).reshape(2, 3)
    start = start_end[0]
    if np.any(start < 0) or np.any(start >= np.asarray(ct.shape)):
        raise ValueError(f"{case_id}: start seed is outside the image")
    if not segmentation[tuple(start)]:
        raise ValueError(f"{case_id}: start seed is outside the audit target")

    clipped_ct = np.clip(ct, -120.0, 180.0)
    clipped_ct = (clipped_ct + 120.0) / 300.0
    flat_channels = np.column_stack(
        [
            clipped_ct.reshape(-1),
            *(filters[..., index].reshape(-1) for index in range(4)),
        ]
    )

    outside_distance_mm = distance_transform_edt(
        ~segmentation,
        sampling=voxel_size_mm,
    )
    local_shell = (~segmentation) & (outside_distance_mm <= shell_radius_mm)
    positive_indices = np.flatnonzero(segmentation.reshape(-1))
    negative_indices = np.flatnonzero(local_shell.reshape(-1))
    sample_count = min(
        len(positive_indices),
        len(negative_indices),
        max_samples_per_class,
    )
    positive_indices = rng.choice(positive_indices, sample_count, replace=False)
    negative_indices = rng.choice(negative_indices, sample_count, replace=False)
    sample_indices = np.concatenate([positive_indices, negative_indices])
    samples = flat_channels[sample_indices]
    labels = np.concatenate(
        [
            np.ones(sample_count, dtype=np.uint8),
            np.zeros(sample_count, dtype=np.uint8),
        ]
    )

    volume_sample_count = min(len(flat_channels), global_scale_samples)
    volume_indices = rng.choice(
        len(flat_channels),
        volume_sample_count,
        replace=False,
    )
    volume_samples = flat_channels[volume_indices]
    global_low, global_high = np.percentile(volume_samples, [10.0, 90.0], axis=0)
    global_scale = np.maximum(global_high - global_low, 1e-3)

    seed_coordinates = ball_coordinates(
        start,
        ct.shape,
        seed_radius_mm / voxel_size_mm,
    )
    seed_features = np.column_stack(
        [
            clipped_ct[seed_coordinates],
            *(filters[..., index][seed_coordinates] for index in range(4)),
        ]
    )
    prototype = np.median(seed_features, axis=0)
    seed_low, seed_high = np.percentile(seed_features, [10.0, 90.0], axis=0)

    metrics = {}
    for subset_name, subset_channels in CHANNEL_SUBSETS.items():
        indices = [CHANNEL_NAMES.index(name) for name in subset_channels]
        point_distance = np.mean(
            np.abs(
                (samples[:, indices] - prototype[indices])
                / global_scale[indices]
            ),
            axis=1,
        )
        interval_distance = np.mean(
            (
                np.maximum(seed_low[indices] - samples[:, indices], 0.0)
                + np.maximum(samples[:, indices] - seed_high[indices], 0.0)
            )
            / global_scale[indices],
            axis=1,
        )
        metrics[f"{subset_name}_prototype"] = float(
            roc_auc_score(labels, -point_distance)
        )
        metrics[f"{subset_name}_interval"] = float(
            roc_auc_score(labels, -interval_distance)
        )

    return {
        "case_id": case_id,
        "sample_count_per_class": sample_count,
        "start_seed": start.tolist(),
        "seed_sample_count": int(len(seed_features)),
        "seed_target_fraction_for_audit": float(
            np.mean(segmentation[seed_coordinates])
        ),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    requested = args.train_case_id + args.test_case_id
    if len(set(requested)) != len(requested):
        raise ValueError("Train and test case IDs must be unique")
    if min(
        args.voxel_size_mm,
        args.seed_radius_mm,
        args.shell_radius_mm,
        args.max_samples_per_class,
        args.global_scale_samples,
    ) <= 0:
        raise ValueError("All scales and sample counts must be positive")

    rng = np.random.default_rng(args.seed)
    cases = [
        load_case(
            args.data_dir,
            case_id,
            voxel_size_mm=args.voxel_size_mm,
            seed_radius_mm=args.seed_radius_mm,
            shell_radius_mm=args.shell_radius_mm,
            max_samples_per_class=args.max_samples_per_class,
            global_scale_samples=args.global_scale_samples,
            rng=rng,
        )
        for case_id in requested
    ]
    train_ids = set(args.train_case_id)
    metric_names = sorted(cases[0]["metrics"])
    aggregate = {}
    for metric_name in metric_names:
        train_values = [
            case["metrics"][metric_name]
            for case in cases
            if case["case_id"] in train_ids
        ]
        test_values = [
            case["metrics"][metric_name]
            for case in cases
            if case["case_id"] not in train_ids
        ]
        aggregate[metric_name] = {
            "train_mean_auc": float(np.mean(train_values)),
            "train_min_auc": float(np.min(train_values)),
            "test_mean_auc": float(np.mean(test_values)),
            "test_min_auc": float(np.min(test_values)),
        }

    payload = {
        "data_dir": str(args.data_dir),
        "train_case_ids": args.train_case_id,
        "test_case_ids": args.test_case_id,
        "voxel_size_mm": args.voxel_size_mm,
        "seed_radius_mm": args.seed_radius_mm,
        "shell_radius_mm": args.shell_radius_mm,
        "channel_names": CHANNEL_NAMES,
        "cases": cases,
        "aggregate": aggregate,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
