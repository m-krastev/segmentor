#!/usr/bin/env python3
"""Audit whether image-only Navigator channels separate bowel from its local shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CHANNEL_NAMES = (
    "ct_clipped",
    "dark_tubularity",
    "bright_tubularity",
    "band_pass",
    "gradient",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--voxel-size-mm", type=float, default=1.5)
    parser.add_argument("--shell-radius-mm", type=float, default=30.0)
    parser.add_argument("--max-samples-per-class", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_case(
    data_dir: Path,
    case_id: str,
    *,
    voxel_size_mm: float,
    shell_radius_mm: float,
    max_samples_per_class: int,
    rng: np.random.Generator,
) -> dict:
    case_dir = data_dir / case_id
    ct_path = case_dir / "ct.nii.gz"
    seg_path = case_dir / "segmentations" / "small_bowel.nii.gz"
    filter_path = case_dir / "cache" / "navigation_filters-v1-mm-3-6-9.nii"
    ct = nib.load(ct_path).get_fdata(dtype=np.float32)
    seg = nib.load(seg_path).get_fdata(dtype=np.float32) > 0
    filters = nib.load(filter_path).get_fdata(dtype=np.float32)
    if filters.shape != (*ct.shape, 4):
        raise ValueError(
            f"{case_id}: filter shape {filters.shape} does not match CT {ct.shape}"
        )
    if seg.shape != ct.shape:
        raise ValueError(
            f"{case_id}: segmentation shape {seg.shape} does not match CT {ct.shape}"
        )

    ct_clipped = np.clip(ct, -120.0, 180.0)
    ct_clipped = (ct_clipped + 120.0) / 300.0
    channels = np.concatenate([ct_clipped[..., None], filters], axis=-1)
    outside_distance_mm = distance_transform_edt(
        ~seg,
        sampling=voxel_size_mm,
    )
    local_shell = (~seg) & (outside_distance_mm <= shell_radius_mm)
    positive_indices = np.flatnonzero(seg.reshape(-1))
    negative_indices = np.flatnonzero(local_shell.reshape(-1))
    sample_count = min(
        len(positive_indices),
        len(negative_indices),
        max_samples_per_class,
    )
    positive_indices = rng.choice(
        positive_indices,
        size=sample_count,
        replace=False,
    )
    negative_indices = rng.choice(
        negative_indices,
        size=sample_count,
        replace=False,
    )
    flat_channels = channels.reshape(-1, len(CHANNEL_NAMES))
    positive = flat_channels[positive_indices]
    negative = flat_channels[negative_indices]
    features = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [
            np.ones(sample_count, dtype=np.uint8),
            np.zeros(sample_count, dtype=np.uint8),
        ]
    )

    channel_metrics = {}
    for channel_index, name in enumerate(CHANNEL_NAMES):
        values = features[:, channel_index]
        auc = float(roc_auc_score(labels, values))
        positive_values = positive[:, channel_index]
        negative_values = negative[:, channel_index]
        channel_metrics[name] = {
            "auc": auc,
            "direction_free_auc": max(auc, 1.0 - auc),
            "positive_zero_fraction": float(np.mean(positive_values == 0)),
            "negative_zero_fraction": float(np.mean(negative_values == 0)),
            "positive_mean": float(np.mean(positive_values)),
            "negative_mean": float(np.mean(negative_values)),
            "positive_p95": float(np.percentile(positive_values, 95)),
            "negative_p95": float(np.percentile(negative_values, 95)),
        }

    return {
        "case_id": case_id,
        "sample_count_per_class": sample_count,
        "segmentation_voxels": int(seg.sum()),
        "local_shell_voxels": int(local_shell.sum()),
        "channel_metrics": channel_metrics,
        "features": features,
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    if len(args.case_id) != 2:
        raise ValueError("Specify exactly two cases for the cross-case audit")
    if args.voxel_size_mm <= 0 or args.shell_radius_mm <= 0:
        raise ValueError("voxel and shell scales must be positive")
    if args.max_samples_per_class < 1:
        raise ValueError("max_samples_per_class must be positive")

    rng = np.random.default_rng(args.seed)
    cases = [
        load_case(
            args.data_dir,
            case_id,
            voxel_size_mm=args.voxel_size_mm,
            shell_radius_mm=args.shell_radius_mm,
            max_samples_per_class=args.max_samples_per_class,
            rng=rng,
        )
        for case_id in args.case_id
    ]

    cross_case = []
    for train_case, test_case in (cases, cases[::-1]):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                random_state=args.seed,
            ),
        )
        model.fit(train_case["features"], train_case["labels"])
        probabilities = model.predict_proba(test_case["features"])[:, 1]
        cross_case.append(
            {
                "train_case": train_case["case_id"],
                "test_case": test_case["case_id"],
                "auc": float(
                    roc_auc_score(test_case["labels"], probabilities)
                ),
            }
        )

    payload = {
        "data_dir": str(args.data_dir),
        "case_ids": args.case_id,
        "voxel_size_mm": args.voxel_size_mm,
        "shell_radius_mm": args.shell_radius_mm,
        "channel_names": CHANNEL_NAMES,
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key not in {"features", "labels"}
            }
            for case in cases
        ],
        "cross_case_logistic_regression": cross_case,
        "mean_cross_case_auc": float(
            np.mean([result["auc"] for result in cross_case])
        ),
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
