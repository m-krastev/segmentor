#!/usr/bin/env python3
# Script for organizing the TotalSegmentator-like dataset into the nnUNet format for multiple labels.
# /// script
# dependencies = ["tqdm", "nibabel"]
# ///

import argparse
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import numpy as np

DATASET_LABELS = {
    "small_bowel": 18,
    "duodenum" : 19,
    "colon": 20
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
    # Use a fixed dataset ID for the combined multilabel dataset
    dataset_id = 42 # This can be any unused ID
    dataset_name = "multilabel_bowel"
    output_dataset_dir = output_dir / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}"

    train_image_dir = output_dataset_dir / "imagesTr"
    train_label_dir = output_dataset_dir / "labelsTr"
    test_image_dir = output_dataset_dir / "imagesTs"
    test_label_dir = output_dataset_dir / "labelsTs"

    for _dir in [
        train_image_dir,
        train_label_dir,
        test_image_dir,
        test_label_dir,
    ]:
        _dir.mkdir(parents=True, exist_ok=True)

    patients = sorted(filter(lambda file: file.is_dir(), data_dir.iterdir()))

    # Names must follow the format LABEL_XXXX.ext
    # where XXXX is a 4 digit number indicating the channel/modality
    new_names_images = [f"{patient.stem}_{modality:04d}{file_ending}" for patient in patients]
    new_names_labels = [f"{patient.stem}{file_ending}" for patient in patients]

    # Map original labels to new incrementing labels for the combined image
    label_mapping = {"background": 0}
    for i, label_name in enumerate(labels):
        label_mapping[label_name] = i + 1 # Start from 1 for actual labels

    # Copy and combine segmentations
    for i, patient_path in tqdm(enumerate(patients), desc="Processing patients: ", total=len(patients)):
        # Copy CT image
        shutil.copy(patient_path / "ct.nii.gz", train_image_dir / new_names_images[i])

        # Load CT image to get affine and header for combined label image
        ct_img = nib.load(patient_path / "ct.nii.gz")
        ct_data = ct_img.get_fdata()
        combined_segmentation_data = np.zeros(ct_data.shape, dtype=np.uint8)

        for label_name in labels:
            segmentation_path = patient_path / "segmentations" / f"{label_name}.nii.gz"
            if segmentation_path.exists():
                seg_img = nib.load(segmentation_path)
                seg_data = seg_img.get_fdata()
                # Assign the new incrementing label value to the combined image
                combined_segmentation_data[seg_data > 0] = label_mapping[label_name]
            else:
                print(f"Warning: Segmentation file not found for {label_name} in {patient_path}")

        # Save the combined segmentation image
        combined_seg_img = nib.Nifti1Image(combined_segmentation_data, ct_img.affine, ct_img.header)
        nib.save(combined_seg_img, train_label_dir / new_names_labels[i])

    patients_in_train_dir = sorted(train_image_dir.iterdir())

    random.seed(seed)
    random.shuffle(patients_in_train_dir)
    num_train = int(len(patients_in_train_dir) * ratio)
    train_patients = patients_in_train_dir[:num_train]
    test_patients = patients_in_train_dir[num_train:]

    for patient in tqdm(test_patients, desc="Moving test files: ", total=len(test_patients)):
        patient.rename(patient.as_posix().replace("imagesTr", "imagesTs"))
        patient_label = Path(
            patient.with_name(patient.stem.partition("_")[0] + file_ending)
            .as_posix()
            .replace("imagesTr", "labelsTr")
        )
        patient_label.rename(patient_label.as_posix().replace("labelsTr", "labelsTs"))

    # Prepare labels for dataset.json
    nnunet_labels = {"background": 0}
    for label_name, value in label_mapping.items():
        if label_name != "background":
            nnunet_labels[label_name] = value

    dataset_metadata = {
        "channel_names": {  # formerly modalities
            str(modality): "CT",
        },
        "labels": nnunet_labels,
        "numTraining": len(train_patients),
        "file_ending": file_ending,
        # "overwrite_image_reader_writer": "SimpleITKIO"
    }

    with (output_dataset_dir / "dataset.json").open("w") as f:
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
    parser.add_argument("--labels", nargs="+", help="List of labels", default=list(DATASET_LABELS.keys()), choices=DATASET_LABELS.keys())
    args = parser.parse_args()
    create_dataset(args.data_dir, args.output_dir, args.labels, args.ratio, args.seed)
