#!/usr/bin/env python3
import argparse
from pathlib import Path
import nibabel as nib
import shutil
import numpy as np
from typing import List


def filter_patients(data_dir: Path, organs: List[str]) -> List[Path]:
    files = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    patients = []
    for file in files:
        if all(
            np.any(
                np.asarray(nib.load(file / "segmentations" / f"{organ}.nii.gz").dataobj)
            )
            for organ in organs
        ):
            patients.append(file)
    return patients


def copy_patients(patients: List[Path], output_dir: str) -> None:
    for patient in patients:
        new = Path(output_dir) / patient.name
        new.mkdir(parents=True, exist_ok=True)
        shutil.copytree(patient, new)
        print(f"Saved {patient} to {new}")
    print(f"Saved {len(patients)} patients to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter out files without bowel segmentation."
    )
    parser.add_argument(
        "--data_dir", type=Path, help="Path to the data directory", required=True
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Path to the output directory", required=True
    )
    parser.add_argument(
        "--organs",
        nargs="+",
        help="List of organs",
        default=["small_bowel", "colon", "duodenum"],
    )
    args = parser.parse_args()
    patients = filter_patients(Path(args.data_dir), args.organs)
    copy_patients(patients, args.output_dir)


if __name__ == "__main__":
    main()
