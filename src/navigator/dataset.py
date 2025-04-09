"""
Data loading utilities for Navigator's small bowel tracking.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any

from segmentor.utils.medutils import load_and_normalize_nifti

FILE_PATTERNS = {
    "seg": "small_bowel.nii.gz",
    "duodenum": "duodenum.nii.gz",
    "colon": "colon.nii.gz",
    "path": "path.npy",
}


class SmallBowelDataset(Dataset):
    """
    Dataset class for small bowel path tracking that scans a directory for matching files.
    """

    def __init__(
        self,
        data_dir: str | Path,
        preload: bool = False,
        transform=None,
    ):
        """
        Initialize the dataset by scanning a directory for data files.

        Args:
            data_dir: Directory containing the data files
            preload: If True, load all data into memory at initialization (default: False)
            transform: Optional transform to be applied to the data
        """
        self.transform = transform
        self.preload = preload
        self.cached_data = {}  # Cache for preloaded data
        self.data_dir = Path(data_dir)

        self.subjects = self._find_subjects()
        print(f"Found {len(self.subjects)} complete subject(s) in {data_dir}")

    def _find_subjects(self) -> List[Dict[str, Path]]:
        """
        Find all subjects with the required files in the data directory.

        Returns:
            List of dictionaries containing paths to each subject's files
        """
        # First, find all patient directories
        patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        subjects = []

        # For each patient directory, find the required files
        for patient_dir in patient_dirs:
            subject_id = patient_dir.name  # Use directory name as subject ID

            image_file = patient_dir / "ct.nii.gz"
            segmentation_dir = patient_dir / "segmentations"

            # Check if required files exist
            if image_file.exists() and segmentation_dir.exists():
                subject = {
                    "id": subject_id,
                    "image": image_file,
                    "seg": patient_dir / "small_bowel.nii.gz",  # Updated to use specific file name
                }

                # Check for duodenum segmentation files
                duodenum_file = segmentation_dir / "duodenum.nii.gz"
                subject["duodenum"] = duodenum_file if duodenum_file.exists() else None
                # Check for colon segmentation file
                colon_file = segmentation_dir / "colon.nii.gz"
                subject["colon"] = colon_file if colon_file.exists() else None
                # Check for ground truth path file
                path_file = patient_dir / "path.npy"
                subject["path"] = path_file if path_file.exists() else None

                subjects.append(subject)

        # If we're preloading data, do that now
        if self.preload and subjects:
            print(f"Preloading {len(subjects)} subjects into memory...")
            for i, subject in enumerate(subjects):
                subject_id = subject["id"]
                self.cached_data[subject_id] = load_subject_data(subject)
            print("Preloading complete")

        return subjects

    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a subject's data by index.

        Args:
            idx: Index of the subject

        Returns:
            Dictionary with the loaded subject data (images as tensors)
        """
        if isinstance(idx, int):
            # Handle index out of bounds
            if idx < 0 or idx >= len(self.subjects):
                raise IndexError(
                    f"Index {idx} out of range for dataset with {len(self.subjects)} subjects"
                )

            # Get the subject entry
            subject = self.subjects[idx]
            subject_id = subject["id"]

            # Check if we have this cached already
            if self.preload and subject_id in self.cached_data:
                data = self.cached_data[subject_id]
            else:
                # Load the data
                data = load_subject_data(subject)

                # Cache it if we're preloading
                if self.preload:
                    self.cached_data[subject_id] = data

            # Convert NumPy arrays to PyTorch tensors
            tensor_data = {
                "id": data["id"],
                "image": torch.from_numpy(data["image"]),
                "seg": torch.from_numpy(data["seg"]),
                "duodenum": torch.from_numpy(data["duodenum"]),
                "colon": torch.from_numpy(data["colon"]),
                "image_affine": data["image_affine"],  # Keep as numpy array
                "image_header": data["image_header"],  # Keep nibabel header
            }

            # Add ground truth path if available
            if "gt_path" in data:
                tensor_data["gt_path"] = torch.from_numpy(data["gt_path"])

            # Apply any transformations
            if self.transform:
                tensor_data = self.transform(tensor_data)

            return tensor_data
        else:
            # Handle slice indexing
            return [self[i] for i in range(*idx.indices(len(self)))]


def load_subject_data(subject_data: Dict[str, str]) -> Dict[str, Any]:
    """
    Load data for a single subject.

    Args:
        subject_data: Dictionary with paths to the subject's files

    Returns:
        Dictionary with loaded data arrays
    """
    result = {"id": subject_data["id"]}

    # Load image data
    image_nii = nib.load(subject_data["image"])
    image_np = load_and_normalize_nifti(subject_data["image"])

    result["image"] = image_np
    result["image_affine"] = image_nii.affine
    result["image_header"] = image_nii.header

    # Load segmentations
    sb_seg_nii = nib.load(subject_data["seg"])
    duodenum_seg_nii = nib.load(subject_data["duodenum"])
    colon_seg_nii = nib.load(subject_data["colon"])

    result["seg"] = (sb_seg_nii.get_fdata() > 0.5).astype(np.uint8)
    result["duodenum"] = (duodenum_seg_nii.get_fdata() > 0.5).astype(np.uint8)
    result["colon"] = (colon_seg_nii.get_fdata() > 0.5).astype(np.uint8)

    # Load ground truth path if available
    if "path" in subject_data:
        try:
            result["gt_path"] = np.load(subject_data["path"])
        except Exception as e:
            print(f"Warning: Could not load GT path {subject_data['path']}: {e}")

    return result
