#!/usr/bin/env python3
"""Freeze label-derived start seeds for a reward-supervised experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_case_ids(paths: list[Path]) -> list[str]:
    case_ids: list[str] = []
    for path in paths:
        case_ids.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Input manifests overlap or contain duplicate case IDs")
    return sorted(case_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace immutable seed output: {args.output_dir}"
        )

    case_ids = read_case_ids(args.case_manifest)
    seeds: dict[str, list[int]] = {}
    for case_id in case_ids:
        start_end_path = (
            args.cache_dir / case_id / "cache" / "start_end.npy"
        )
        start_end = np.loadtxt(start_end_path, dtype=int).reshape(-1, 3)
        if start_end.shape != (2, 3):
            raise ValueError(
                f"Expected two native-XYZ endpoints in {start_end_path}, "
                f"got {start_end.shape}"
            )
        seeds[case_id] = [int(value) for value in start_end[0]]

    args.output_dir.mkdir(parents=True)
    for case_id, seed in seeds.items():
        (args.output_dir / f"{case_id}.txt").write_text(
            " ".join(str(value) for value in seed) + "\n"
        )

    digest_input = "".join(
        f"{case_id} {' '.join(str(value) for value in seeds[case_id])}\n"
        for case_id in case_ids
    )
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "reward-supervised/inference-clean",
        "provenance": (
            "First anatomical endpoint from each preflight start_end.npy cache; "
            "derived from training/validation labels, not operator supplied."
        ),
        "coordinate_order": "native NIfTI XYZ",
        "cache_dir": str(args.cache_dir.resolve()),
        "case_manifests": [str(path.resolve()) for path in args.case_manifest],
        "case_count": len(case_ids),
        "seed_manifest_sha256": hashlib.sha256(
            digest_input.encode("utf-8")
        ).hexdigest(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
