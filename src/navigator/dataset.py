"""
Data loading utilities for Navigator's small bowel tracking.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Tuple

from segmentor.utils.medutils import load_and_normalize_nifti

# Import necessary calculation functions
from .utils import find_start_end, compute_wall_map, compute_gdt
from .config import Config  # Example: Assuming Config holds wall_map_sigmas

FILE_PATTERNS = {
    "seg": "small_bowel.nii.gz",
    "duodenum": "duodenum.nii.gz",
    "colon": "colon.nii.gz",
    "path": "path.npy",
}

# Define cache filenames
CACHE_FILES = {
    "start_end": "start_end.npy",
    "wall_map": "wall_map.nii.gz",
    "gdt_start": "gdt_start.nii.gz",
    "gdt_end": "gdt_end.nii.gz",
}


class SmallBowelDataset(Dataset):
    """
    Dataset class for small bowel path tracking that scans a directory for matching files.
    Handles calculation and caching of start/end points, wall map, and GDT.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: Config,  # Pass config for parameters like wall_map_sigmas
    ):
        """
        Initialize the dataset by scanning a directory for data files.

        Args:
            data_dir: Directory containing the data files
            config: Configuration object (needed for calculation parameters)
            transform: Optional transform to be applied to the data
        """
        self.config = config  # Store config
        self.data_dir = Path(data_dir)

        self.subjects = self._find_subjects()
        print(f"Found {len(self.subjects)} complete subject(s) in {data_dir}")

    # ... existing _find_subjects method ...
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
                    "seg": segmentation_dir
                    / "small_bowel.nii.gz",  # Updated to use specific file name
                    "patient_dir": patient_dir,  # Store patient dir for caching
                }

                # Check for duodenum segmentation files
                duodenum_file = segmentation_dir / "duodenum.nii.gz"
                if not duodenum_file.exists():
                    print(f"Warning: Duodenum file missing for subject {subject_id}. Skipping.")
                    continue  # Skip subject if essential files are missing
                subject["duodenum"] = duodenum_file

                # Check for colon segmentation file
                colon_file = segmentation_dir / "colon.nii.gz"
                if not colon_file.exists():
                    print(f"Warning: Colon file missing for subject {subject_id}. Skipping.")
                    continue  # Skip subject if essential files are missing
                subject["colon"] = colon_file

                # Check for ground truth path file (optional)
                path_file = patient_dir / "path.npy"
                subject["path"] = path_file if path_file.exists() else None

                # Check base segmentation file exists
                if not subject["seg"].exists():
                    print(
                        f"Warning: Small bowel seg file missing for subject {subject_id}. Skipping."
                    )
                    continue

                subjects.append(subject)

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
            Dictionary with the loaded subject data (images/maps as tensors)
        """
        if isinstance(idx, int) or np.issubdtype(type(idx), np.integer):
            # Handle index out of bounds
            if idx < 0 or idx >= len(self.subjects):
                raise IndexError(
                    f"Index {idx} out of range for dataset with {len(self.subjects)} subjects"
                )

            # Get the subject entry
            subject = self.subjects[idx]
            data = load_subject_data(subject, self.config)

            data = {
                "id": data["id"],
                "image": data["image"],
                "seg": data["seg"],
                "duodenum": data["duodenum"],
                "colon": data["colon"],
                "wall_map": data["wall_map"],  # Add wall_map tensor
                "gdt_start": data["gdt_start"],  # Add gdt_start tensor
                "gdt_end": data["gdt_end"],  # Add gdt_end tensor
                "image_affine": data["image_affine"],  # Keep as numpy array
                "spacing": data["spacing"],  # Keep spacing in (Z, Y, X) order
                "start_coord": data["start_coord"],  # Keep as tuple/numpy
                "end_coord": data["end_coord"],  # Keep as tuple/numpy
            }

            return data
        else:
            # Handle slice indexing
            return [self[i] for i in range(*idx.indices(len(self)))]


def load_subject_data(subject_data: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """
    Load data for a single subject, calculating and caching derived data if needed.

    Args:
        subject_data: Dictionary with paths to the subject's files and patient_dir
        config: Configuration object

    Returns:
        Dictionary with loaded data arrays (numpy format)
    """
    result = {"id": subject_data["id"]}
    patient_dir = subject_data["patient_dir"]
    cache_dir = patient_dir / "cache"
    cache_dir.mkdir(exist_ok=True)  # Ensure cache directory exists

    # --- Load Base Data ---
    image_nii = nib.load(subject_data["image"])
    image_np = load_and_normalize_nifti(subject_data["image"]).astype(np.float32)
    image_np = np.transpose(image_np, (2, 1, 0))  # Change to (Z, Y, X) format

    result["image"] = image_np
    result["image_affine"] = image_nii.affine
    result["spacing"] = image_nii.header.get_zooms()[::-1]  # Get spacing in (Z, Y, X) order

    # Load segmentations (ensure they exist before loading)
    sb_seg_nii = nib.load(subject_data["seg"])
    duodenum_seg_nii = nib.load(subject_data["duodenum"])
    colon_seg_nii = nib.load(subject_data["colon"])

    seg_np = np.transpose((sb_seg_nii.get_fdata() > 0.5).astype(np.uint8), (2, 1, 0))
    duodenum_np = np.transpose((duodenum_seg_nii.get_fdata() > 0.5).astype(np.uint8), (2, 1, 0))
    colon_np = np.transpose((colon_seg_nii.get_fdata() > 0.5).astype(np.uint8), (2, 1, 0))

    result["seg"] = seg_np
    result["duodenum"] = duodenum_np
    result["colon"] = colon_np

    # Load ground truth path if available
    if "path" in subject_data and subject_data["path"] is not None:
        try:
            result["gt_path"] = np.load(subject_data["path"])
        except Exception as e:
            print(f"Warning: Could not load GT path {subject_data['path']}: {e}")

    # --- Load/Calculate Start/End Coordinates ---
    start_end_cache_path = cache_dir / CACHE_FILES["start_end"]
    if start_end_cache_path.exists():
        start_coord_np, end_coord_np = np.loadtxt(start_end_cache_path).astype(int)
        start_coord = tuple(start_coord_np)
        end_coord = tuple(end_coord_np)
    else:
        start_coord, end_coord = find_start_end(
            duodenum_volume=duodenum_np, colon_volume=colon_np, small_bowel_volume=seg_np
        )
        np.savetxt(start_end_cache_path, np.stack((start_coord, end_coord)), fmt="%d")
    result["start_coord"] = start_coord
    result["end_coord"] = end_coord

    # --- Load/Calculate Wall Map ---
    wall_map_cache_path = cache_dir / CACHE_FILES["wall_map"]
    if wall_map_cache_path.exists():
        wall_map_nii = nib.load(wall_map_cache_path)
        wall_map_np = np.transpose(wall_map_nii.get_fdata(), (2, 1, 0)).astype(np.float32)
    else:
        wall_map_np = compute_wall_map(result["image"], sigmas=config.wall_map_sigmas).astype(
            np.float32
        )
        # Save in Nifti format (preserving original orientation for saving)
        wall_map_to_save = np.transpose(wall_map_np, (2, 1, 0))
        wall_map_nii_save = nib.Nifti1Image(wall_map_to_save, result["image_affine"])
        nib.save(wall_map_nii_save, wall_map_cache_path)
    result["wall_map"] = wall_map_np

    # --- Load/Calculate GDT (Start) ---
    gdt_start_cache_path = cache_dir / CACHE_FILES["gdt_start"]
    if gdt_start_cache_path.exists():
        gdt_start_nii = nib.load(gdt_start_cache_path)
        gdt_start_np = np.transpose(gdt_start_nii.get_fdata(), (2, 1, 0)).astype(np.float32)
    else:
        # Use numpy seg and start_coord for calculation
        gdt_start_tensor = compute_gdt(result["seg"], result["start_coord"], result["spacing"])
        gdt_start_np = (
            gdt_start_tensor.astype(np.float32)  # Save in Nifti format
        )
        gdt_start_to_save = np.transpose(gdt_start_np, (2, 1, 0))
        gdt_start_nii_save = nib.Nifti1Image(gdt_start_to_save, result["image_affine"])
        nib.save(gdt_start_nii_save, gdt_start_cache_path)
    result["gdt_start"] = gdt_start_np

    # --- Load/Calculate GDT (End) ---
    gdt_end_cache_path = cache_dir / CACHE_FILES["gdt_end"]
    if gdt_end_cache_path.exists():
        gdt_end_nii = nib.load(gdt_end_cache_path)
        gdt_end_np = np.transpose(gdt_end_nii.get_fdata(), (2, 1, 0)).astype(np.float32)
    else:
        # Use numpy seg and end_coord for calculation
        gdt_end_tensor = compute_gdt(result["seg"], result["end_coord"], result["spacing"])
        gdt_end_np = gdt_end_tensor.astype(np.float32)
        # Save in Nifti format
        gdt_end_to_save = np.transpose(gdt_end_np, (2, 1, 0))
        gdt_end_nii_save = nib.Nifti1Image(gdt_end_to_save, result["image_affine"])
        nib.save(gdt_end_nii_save, gdt_end_cache_path)
    result["gdt_end"] = gdt_end_np

    return result
