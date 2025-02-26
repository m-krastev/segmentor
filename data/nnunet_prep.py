#!/usr/bin/env python3

import argparse
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm

DATASET_LABELS = {
    "small_bowel": 42,
}

# It's a good idea to train on multiple labels at once, so that the model can learn to differentiate between them. In doing that, it can also learn to better segment the label of interest. We can also try to identify the centerline of the whole gastrointestinal tract, although this seems like at best might be a very slight improvement, and at worst might be a distraction for the auxilliary loss (for centerline prediction).


def create_dataset(
    data_dir: Path,
    output_dir: Path,
    labels: list[str],
    ratio: float,
    seed: int = 42,
    file_ending: str = ".nii.gz",
):
    modality = 0  # CT
    for label in labels:
        output_dir = output_dir / "nnUNet_raw" / f"Dataset{DATASET_LABELS[label]:03d}_{label}"
        train_image_dir = output_dir / "imagesTr"
        train_label_dir = output_dir / "labelsTr"
        test_image_dir = output_dir / "imagesTs"
        test_label_dir = output_dir / "labelsTs"
        for _dir in [
            train_image_dir,
            train_label_dir,
            test_image_dir,
            test_label_dir,
        ]:
            _dir.mkdir(parents=True, exist_ok=True)

        patients = sorted(filter(lambda file: file.is_dir(), data_dir.iterdir()))

        # Names must follow the format LABEL_XXXX.ext
        # where XXXX is a 4 digit number indicating the channel/modality, I suppose in this case it works for cineMRI, but here we only deal with CT
        new_names_images = [f"{patient.stem}_{modality:04d}{file_ending}" for patient in patients]

        new_names_labels = [f"{patient.stem}{file_ending}" for patient in patients]

        # Copy the files to a new directory
        for i, patient in tqdm(enumerate(patients), desc="Copying files: ", total=len(patients)):
            shutil.copy(patient / "ct.nii.gz", train_image_dir / new_names_images[i])

            shutil.copy(
                patient / "segmentations" / f"{label}.nii.gz",
                train_label_dir / new_names_labels[i],
            )

        patients = sorted(train_image_dir.iterdir())

        random.seed(seed)
        random.shuffle(patients)
        num_train = int(len(patients) * ratio)
        train_patients = patients[:num_train]
        test_patients = patients[num_train:]

        for patient in tqdm(test_patients, desc="Moving test files: ", total=len(test_patients)):
            patient.rename(patient.as_posix().replace("imagesTr", "imagesTs"))
            patient_label = Path(
                patient.with_name(patient.stem.partition("_")[0] + file_ending)
                .as_posix()
                .replace("imagesTr", "labelsTr")
            )
            patient_label.rename(patient_label.as_posix().replace("labelsTr", "labelsTs"))

        dataset_metadata = {
            "channel_names": {  # formerly modalities
                str(modality): "CT",
            },
            "labels": {
                "background": 0,
                label: 1,
            },
            "numTraining": len(train_patients),
            "file_ending": file_ending,
            # "overwrite_image_reader_writer": "SimpleITKIO"
        }

        with (output_dir / "dataset.json").open("w") as f:
            json.dump(dataset_metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create nnUNet dataset from TotalSegmentator-like file structure. "
        "Currently only supports CT and small_bowel.",
    )
    parser.add_argument("--data_dir", type=Path, help="Path to data directory", required=True)
    parser.add_argument("--output_dir", type=Path, help="Path to output directory", required=True)
    parser.add_argument("--ratio", "-r", type=float, help="Ratio of training data", default=0.8)
    parser.add_argument("--seed", type=int, help="Seed for random shuffling", default=42)
    parser.add_argument("--labels", nargs="+", help="List of labels", default=["small_bowel"])
    args = parser.parse_args()
    create_dataset(args.data_dir, args.output_dir, args.labels, args.ratio, args.seed)
