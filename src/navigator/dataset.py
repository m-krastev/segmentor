"""
Data loading utilities for Navigator's small bowel tracking.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Any

from segmentor.utils.medutils import load_and_normalize_nifti

# Import necessary calculation functions
from .utils import find_start_end, compute_wall_map, compute_gdt
from .config import Config
from skimage.feature import peak_local_max

FILE_PATTERNS = {
    "ct": "ct.nii",
    "small_bowel": "segmentations/small_bowel.nii",
    "duodenum": "segmentations/duodenum.nii",
    "colon": "segmentations/colon.nii",
    "path": "path.npy",
}

# Define cache filenames
CACHE_FILES = {
    "start_end": "start_end.npy",
    "wall_map": "wall_map.nii",
    "gdt_start": "gdt_start.nii",
    "gdt_end": "gdt_end.nii",
    "local_peaks": "local_peaks.npy"
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

            image_file = patient_dir / FILE_PATTERNS["ct"]
            image_file = image_file if image_file.exists() else image_file.with_suffix(".nii.gz")

            # Check if required files exist
            if image_file.exists():
                subject = {
                    "id": subject_id,
                    "image": image_file,
                    "patient_dir": patient_dir,  # Store patient dir for caching
                }
                for organ in ["small_bowel", "duodenum", "colon"]:
                    # Check for small bowel segmentation file
                    seg_file = patient_dir / FILE_PATTERNS[organ]
                    seg_file = seg_file if seg_file.exists() else seg_file.with_suffix(".nii.gz")
                    if not seg_file.exists():
                        print(f"Warning: {organ.capitalize()} file missing for subject {subject_id}. Skipping.")
                        # continue
                        seg_file = None
                    subject[organ] = seg_file

                # Check for ground truth path file (optional)
                path_file = patient_dir / FILE_PATTERNS["path"]
                subject["path"] = path_file if path_file.exists() else None
                
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
                "local_peaks": data["local_peaks"],
            }

            return data
        else:
            # Handle slice indexing
            return [self[i] for i in range(*idx.indices(len(self)))]


def load_subject_data(subject_data: Dict[str, Any], config: Config, **cache) -> Dict[str, Any]:
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
    image_nii: nib.nifti1.Nifti1Image = nib.load(subject_data["image"])
    image_np = image_nii.get_fdata(dtype=np.float32)
    image_np = np.transpose(image_np, (2, 1, 0))  # Change to (Z, Y, X) format

    result["image"] = image_np
    result["image_affine"] = image_nii.affine
    result["spacing"] = image_nii.header.get_zooms()[::-1]  # Get spacing in (Z, Y, X) order

    # Load segmentations (ensure they exist before loading)
    sb_seg_nii = np.asanyarray(nib.load(subject_data["small_bowel"]).dataobj)
    seg_np = np.transpose(sb_seg_nii, (2, 1, 0))
    result["seg"] = seg_np

    if subject_data.get("duodenum"):
        duodenum_seg_nii = np.asanyarray(nib.load(subject_data["duodenum"]).dataobj)
        duodenum_np = np.transpose(duodenum_seg_nii, (2, 1, 0))
        result["duodenum"] = duodenum_np
    else:
        result["duodenum"] = None
    if subject_data.get("colon"):
        colon_seg_nii = np.asanyarray(nib.load(subject_data["colon"]).dataobj)
        colon_np = np.transpose(colon_seg_nii, (2, 1, 0))
        result["colon"] = colon_np
    else:
        result["colon"] = None

    # Load ground truth path if available
    if subject_data.get("path") is not None:
        try:
            result["gt_path"] = np.loadtxt(subject_data["path"], dtype=int)
        except Exception as e:
            print(f"Warning: Could not load GT path {subject_data['path']}: {e}")

    # --- Load/Calculate Start/End Coordinates ---
    start_end_cache_path = cache_dir / CACHE_FILES["start_end"]
    if start_end_cache_path.exists():
        start_coord_np, end_coord_np = np.loadtxt(start_end_cache_path, dtype=int)
        start_coord = tuple(start_coord_np)
        end_coord = tuple(end_coord_np)
    else:
        assert result["duodenum"] is not None, "Duodenum segmentation is required to find start/end coordinates."
        assert result["colon"] is not None, "Colon segmentation is required to find start/end coordinates."
        start_coord, end_coord = find_start_end(
            duodenum_volume=result["duodenum"], colon_volume=result["colon"], small_bowel_volume=seg_np
        )
        np.savetxt(start_end_cache_path, np.stack((start_coord, end_coord)), fmt="%d")
    result["start_coord"] = start_coord
    result["end_coord"] = end_coord

    # --- Load/Calculate Wall Map ---
    wall_map_cache_path = cache_dir / CACHE_FILES["wall_map"]
    if wall_map_cache_path.exists():
        wall_map_nii = nib.load(wall_map_cache_path)
        wall_map_np = np.transpose(wall_map_nii.get_fdata(dtype=np.float32), (2, 1, 0))
    else:
        wall_map_np = compute_wall_map(result["image"], sigmas=config.wall_map_sigmas)
        # Save in Nifti format (preserving original orientation for saving)
        wall_map_to_save = np.transpose(wall_map_np, (2, 1, 0))
        wall_map_nii_save = nib.Nifti1Image(wall_map_to_save, result["image_affine"])
        nib.save(wall_map_nii_save, wall_map_cache_path)
    result["wall_map"] = wall_map_np

    # --- Load/Calculate GDT (Start) ---
    gdt_start_cache_path = cache_dir / CACHE_FILES["gdt_start"]
    if gdt_start_cache_path.exists():
        gdt_start_nii = nib.load(gdt_start_cache_path)
        gdt_start_np = np.transpose(gdt_start_nii.get_fdata(dtype=np.float32), (2, 1, 0))
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
        gdt_end_np = np.transpose(gdt_end_nii.get_fdata(dtype=np.float32), (2, 1, 0))
    else:
        # Use numpy seg and end_coord for calculation
        gdt_end_tensor = compute_gdt(result["seg"], result["end_coord"], result["spacing"])
        gdt_end_np = gdt_end_tensor.astype(np.float32)
        # Save in Nifti format
        gdt_end_to_save = np.transpose(gdt_end_np, (2, 1, 0))
        gdt_end_nii_save = nib.Nifti1Image(gdt_end_to_save, result["image_affine"])
        nib.save(gdt_end_nii_save, gdt_end_cache_path)
    result["gdt_end"] = gdt_end_np

    local_peaks_cache_path = cache_dir / CACHE_FILES["local_peaks"]
    if local_peaks_cache_path.exists():
        local_peaks_np = np.loadtxt(local_peaks_cache_path, dtype=int)
    else:
        from .utils import distance_transform_edt, binary_dilation
        distances = distance_transform_edt(
            binary_dilation(result["seg"], iterations=3)
        )
        local_peaks_np = peak_local_max(distances, min_distance=6, threshold_abs=4, num_peaks=1024)
        np.savetxt(local_peaks_cache_path, local_peaks_np, fmt="%d")
    result["local_peaks"] = local_peaks_np

    return result
