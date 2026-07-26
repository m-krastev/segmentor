#!/usr/bin/env python
"""Preflight every nnU-Net case and freeze train/validation/test manifests."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from navigator.config import Config
from navigator.dataset import NNUNetActualDataset


def deterministic_split(
    case_ids: list[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[str]]:
    """Return a deterministic, disjoint split of an eligible cohort."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below one")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_ids contains duplicates")
    if len(case_ids) < 3:
        raise ValueError("At least three eligible cases are required")

    shuffled = np.asarray(sorted(case_ids), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    if train_end == 0 or validation_end == train_end or validation_end == len(shuffled):
        raise ValueError("Split fractions produced an empty partition")
    return {
        "train": sorted(str(case_id) for case_id in shuffled[:train_end]),
        "validation": sorted(
            str(case_id) for case_id in shuffled[train_end:validation_end]
        ),
        "test": sorted(str(case_id) for case_id in shuffled[validation_end:]),
    }


def manifest_text(case_ids: list[str], metadata: str) -> str:
    return f"# {metadata}\n" + "".join(f"{case_id}\n" for case_id in case_ids)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def release_optional_gpu_cache() -> None:
    """Return CuCIM/CuPy scratch allocations between independent cases."""

    try:
        import cupy
    except ImportError:
        return
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel-size-mm", type=float, default=1.5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--generate-expert-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require construction of a continuous coverage-then-end expert route.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace immutable preflight output: {output_dir}. "
            "Choose a new versioned directory."
        )

    config = Config(
        device="cpu",
        voxel_size_mm=args.voxel_size_mm,
        patch_size_mm=24,
        max_step_displacement_mm=6,
        cumulative_path_radius_mm=9,
        endpoint_tolerance_mm=3,
        nnunet_generate_expert_path=args.generate_expert_path,
        track_wandb=False,
    )
    dataset = NNUNetActualDataset(
        nnunet_raw=args.nnunet_raw,
        config=config,
        cache_dir=args.cache_dir,
    )

    eligible: list[str] = []
    rejected: list[dict[str, str]] = []
    case_metadata: dict[str, dict[str, int | float]] = {}
    for index, case_id in enumerate(dataset.case_ids, start=1):
        try:
            subject = dataset[index - 1]
            eligible.append(case_id)
            segmentation_voxels = int(np.asarray(subject["seg"], dtype=bool).sum())
            traversable_voxels = int(np.isfinite(subject["gdt_start"]).sum())
            case_metadata[case_id] = {
                "segmentation_voxels": segmentation_voxels,
                "start_component_voxels": traversable_voxels,
                "start_component_fraction": (
                    traversable_voxels / segmentation_voxels
                    if segmentation_voxels
                    else 0.0
                ),
                "expert_route_voxels": (
                    int(len(subject["gt_path"]))
                    if subject.get("gt_path") is not None
                    else 0
                ),
            }
            print(f"[{index}/{len(dataset)}] eligible {case_id}", flush=True)
        except Exception as error:
            rejected.append(
                {
                    "case": case_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                f"[{index}/{len(dataset)}] rejected {case_id}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
        finally:
            release_optional_gpu_cache()

    splits = deterministic_split(
        eligible,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = (
        f"Navigator nnU-Net preflight v1; seed={args.seed}; "
        f"created_at={created_at}"
    )
    manifest_contents = {
        name: manifest_text(case_ids, metadata)
        for name, case_ids in splits.items()
    }

    output_dir.mkdir(parents=True)
    for name, contents in manifest_contents.items():
        (output_dir / f"{name}.txt").write_text(contents)
    (output_dir / "eligible.txt").write_text(manifest_text(sorted(eligible), metadata))
    (output_dir / "rejected.json").write_text(
        json.dumps(rejected, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "case_metadata.json").write_text(
        json.dumps(case_metadata, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": 1,
        "created_at": created_at,
        "nnunet_raw": str(Path(args.nnunet_raw).resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "seed": args.seed,
        "voxel_size_mm": args.voxel_size_mm,
        "generate_expert_path": args.generate_expert_path,
        "discovered_cases": len(dataset),
        "eligible_cases": len(eligible),
        "rejected_cases": len(rejected),
        "split_counts": {name: len(case_ids) for name, case_ids in splits.items()},
        "manifest_sha256": {
            f"{name}.txt": sha256_text(contents)
            for name, contents in manifest_contents.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
