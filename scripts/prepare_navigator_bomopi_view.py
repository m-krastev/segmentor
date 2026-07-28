#!/usr/bin/env python3
"""Create a non-destructive manifest-backed view of usable BOMOPI subjects."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Manifest must contain unique subject IDs: {path}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/bomopi_resampled2"),
    )
    parser.add_argument(
        "--view",
        type=Path,
        default=Path("data/bomopi_resampled2_unique-v1"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/navigator-bomopi-v1/eligible.txt"),
    )
    args = parser.parse_args()

    ids = read_ids(args.manifest)
    args.view.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name for path in args.view.iterdir() if path.name not in set(ids)
    }
    if unexpected:
        raise ValueError(
            f"Refusing a view containing subjects outside the manifest: {sorted(unexpected)}"
        )

    for case_id in ids:
        source = (args.source / case_id).resolve()
        destination = args.view / case_id
        if not source.is_dir():
            raise FileNotFoundError(f"BOMOPI source subject missing: {source}")
        if destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError(
                    f"Existing view link has the wrong target: {destination}"
                )
        elif destination.exists():
            raise ValueError(f"Existing view entry is not a symlink: {destination}")
        else:
            destination.symlink_to(source, target_is_directory=True)

    print(
        f"Prepared {len(ids)}-subject BOMOPI view at {args.view}",
        flush=True,
    )


if __name__ == "__main__":
    main()
