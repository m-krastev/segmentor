"""
Data loading utilities for Navigator's small bowel tracking.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Any


# Import necessary calculation functions
from .utils import (
    find_start_end,
    compute_wall_map,
    compute_navigation_filter_bank,
    compute_gdt,
    distance_transform_edt,
    release_gpu_preprocessing_memory,
)

from .config import Config
from .oracle import skeleton_covering_route
from skimage.feature import peak_local_max

FILE_PATTERNS = {
    "ct": "ct.nii",
    "small_bowel": "segmentations/small_bowel.nii",
    "duodenum": "segmentations/duodenum.nii",
    "colon": "segmentations/colon.nii",
    "path": "path.npy",
}

MRI_PATH_FILE_PATTERNS = {
    "mri": "mri.nii",
    "small_bowel_seg": "segmentations/small_bowel.nii",
    "path_prefix": "path_",
}

# Define cache filenames
CACHE_FILES = {
    "start_end": "start_end.npy",
    "wall_map": "wall_map.nii",
    # Version the distance caches so pre-fix fast-marching fields, which could
    # cross forbidden background, are never silently reused.
    "gdt_start": "gdt_start_mcp26_v1.nii",
    "gdt_end": "gdt_end_mcp26_v1.nii",
    # Distance to the endpoint-connected GDT support, not to the union of all
    # labels: disconnected annotation islands must remain off-target.
    "target_distance": "target_distance_connected_edt_v2.nii",
    "local_peaks": "local_peaks.npy",
    "expert_path": "expert_path_skeleton_tree_v2.npy",
}

NNUNET_CASE_FILES = (
    ("Dataset018_small_bowel", "imagesTr", "_0000.nii.gz"),
    ("Dataset018_small_bowel", "labelsTr", ".nii.gz"),
    ("Dataset019_duodenum", "labelsTr", ".nii.gz"),
    ("Dataset020_colon", "labelsTr", ".nii.gz"),
)


def normalize_coordinate_rows(
    coordinates: np.ndarray,
    fallback: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """Return voxel coordinates with a stable ``[N, 3]`` shape."""
    coordinates = np.asarray(coordinates, dtype=int)
    if coordinates.size == 0:
        coordinates = np.asarray(fallback, dtype=int)
    if coordinates.size % 3:
        raise ValueError(
            f"Coordinate array contains {coordinates.size} values, expected a multiple of 3"
        )
    return coordinates.reshape(-1, 3)


def load_annotation_free_subject_data(
    *,
    case_id: str,
    image_path: Path,
    patient_dir: Path,
    seed_path: Path,
    config: Config,
) -> dict[str, Any]:
    """Load only image-derived policy data plus one external start seed.

    This function deliberately has no segmentation, endpoint, GDT, or expert
    path arguments. Keeping this loader structurally separate makes accidental
    label access in annotation-free training auditable.
    """

    print(f"Loading annotation-free subject {case_id}", flush=True)
    image_nii: nib.nifti1.Nifti1Image = nib.load(image_path)
    image = image_nii.get_fdata(dtype=np.float32)
    spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    expected_spacing = (config.voxel_size_mm,) * 3
    if not np.allclose(spacing, expected_spacing, atol=1e-3):
        raise ValueError(
            f"NIfTI spacing {spacing} for {case_id} does not match configured "
            f"isotropic spacing {expected_spacing}."
        )

    seed_xyz = np.loadtxt(seed_path, dtype=int)
    if np.asarray(seed_xyz).size != 3:
        raise ValueError(
            f"External seed for {case_id} must contain exactly three native-XYZ "
            f"voxel coordinates: {seed_path}"
        )
    seed_xyz = np.asarray(seed_xyz, dtype=int).reshape(3)
    if np.any(seed_xyz < 0) or np.any(seed_xyz >= np.asarray(image.shape)):
        raise ValueError(f"External seed for {case_id} is outside the image: {seed_xyz}")

    cache_dir = patient_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_features = load_or_compute_navigation_filter_bank(
        image,
        image_nii.affine,
        cache_dir,
        config,
    )
    release_gpu_preprocessing_memory()

    # The environment uses ZYX tensor order.
    return {
        "id": case_id,
        "image": np.transpose(image, (2, 1, 0)),
        "image_features": np.transpose(image_features, (0, 3, 2, 1)),
        # The first filter remains the intrinsic image-energy response.
        "wall_map": np.transpose(image_features[0], (2, 1, 0)),
        "image_affine": image_nii.affine,
        "spacing": spacing[::-1],
        "start_coord": tuple(int(value) for value in seed_xyz[::-1]),
    }


def load_or_compute_navigation_filter_bank(
    image: np.ndarray,
    affine: np.ndarray,
    cache_dir: Path,
    config: Config,
) -> np.ndarray:
    """Return the versioned four-channel image-only navigation filter bank."""

    scales_key = "-".join(f"{scale:g}" for scale in config.navigation_filter_scales_mm)
    cache_path = cache_dir / f"navigation_filters-v1-mm-{scales_key}.nii"
    if cache_path.exists():
        cached = nib.load(cache_path).get_fdata(dtype=np.float32)
        if cached.ndim != 4 or cached.shape[-1] != 4:
            raise ValueError(f"Invalid navigation-filter cache shape: {cache_path}")
        return np.moveaxis(cached, -1, 0)

    features = compute_navigation_filter_bank(
        image,
        spacing_mm=config.voxel_size_mm,
        scales_mm=config.navigation_filter_scales_mm,
    )
    nib.save(
        nib.Nifti1Image(np.moveaxis(features, 0, -1), affine),
        cache_path,
    )
    return features


def load_nnunet_evaluation_target(
    nnunet_raw: str | Path,
    cache_dir: str | Path,
    case_id: str,
) -> dict[str, Any]:
    """Load held-out labels only after an annotation-free rollout finishes."""

    nnunet_raw = Path(nnunet_raw)
    segmentation_path = (
        nnunet_raw / "Dataset018_small_bowel" / "labelsTr" / f"{case_id}.nii.gz"
    )
    start_end_path = Path(cache_dir) / case_id / "cache" / CACHE_FILES["start_end"]
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Evaluation target missing: {segmentation_path}")
    if not start_end_path.is_file():
        raise FileNotFoundError(f"Evaluation endpoint cache missing: {start_end_path}")

    segmentation_nii = nib.load(segmentation_path)
    segmentation = np.asanyarray(segmentation_nii.dataobj) > 0
    start_end = np.loadtxt(start_end_path, dtype=int).reshape(-1, 3)
    if len(start_end) != 2:
        raise ValueError(f"Expected one start/end pair in {start_end_path}")
    return {
        "segmentation": np.transpose(segmentation, (2, 1, 0)),
        "goal": tuple(int(value) for value in start_end[1][::-1]),
        "spacing": tuple(
            float(value) for value in segmentation_nii.header.get_zooms()[:3][::-1]
        ),
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
                        print(
                            f"Warning: {organ.capitalize()} file missing for subject {subject_id}. Skipping."
                        )
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
            Dictionary with the loaded subject data
        """
        if isinstance(idx, int) or np.issubdtype(type(idx), np.integer):
            # Get the subject entry
            subject = self.subjects[idx]
            data = load_subject_data(subject, self.config)
            data = {
                "id": data["id"],
                "image": data["image"],
                "seg": data["seg"],
                "duodenum": data["duodenum"],
                "colon": data["colon"],
                "wall_map": data["wall_map"],
                "gdt_start": data["gdt_start"],
                "gdt_end": data["gdt_end"],
                "image_affine": data["image_affine"],
                "spacing": data["spacing"],
                "start_coord": data["start_coord"],
                "end_coord": data["end_coord"],
                "local_peaks": data["local_peaks"],
                "gt_path": data.get("gt_path"),
            }

            return data
        else:
            # Handle slice indexing
            return [self[i] for i in range(*idx.indices(len(self)))]


class NNUNetActualDataset(Dataset):
    """Join matching small-bowel, duodenum, and colon nnU-Net cases."""

    def __init__(
        self,
        nnunet_raw: str | Path,
        config: Config,
        cache_dir: str | Path,
        case_ids: list[str] | None = None,
    ):
        self.nnunet_raw = Path(nnunet_raw)
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.seed_dir = (
            Path(config.nnunet_seed_dir)
            if config.clean_policy_inputs and config.nnunet_seed_dir
            else None
        )
        if config.clean_policy_inputs and self.seed_dir is None:
            raise ValueError(
                "Image-only nnU-Net policy loading requires an external seed directory"
            )

        if not self.nnunet_raw.is_dir():
            raise FileNotFoundError(f"nnU-Net raw directory not found: {self.nnunet_raw}")

        if case_ids is None:
            case_ids = self._discover_complete_case_ids()
        self.case_ids = sorted(case_ids)
        if not self.case_ids:
            raise ValueError(f"No complete nnU-Net cases found in {self.nnunet_raw}")
        self._validate_cases()

        # Match SmallBowelDataset's public metadata used by the main entrypoint.
        self.subjects = [{"id": case_id} for case_id in self.case_ids]
        print(f"Found {len(self.case_ids)} complete nnU-Net subject(s) in {self.nnunet_raw}")

    def _case_path(self, dataset: str, folder: str, case_id: str) -> Path:
        suffix = "_0000.nii.gz" if folder == "imagesTr" else ".nii.gz"
        return self.nnunet_raw / dataset / folder / f"{case_id}{suffix}"

    def _discover_complete_case_ids(self) -> list[str]:
        if self.config.annotation_free:
            directory = self.nnunet_raw / "Dataset018_small_bowel" / "imagesTr"
            suffix = "_0000.nii.gz"
            if not directory.is_dir():
                raise FileNotFoundError(f"nnU-Net image directory not found: {directory}")
            return sorted(
                path.name[: -len(suffix)]
                for path in directory.glob(f"*{suffix}")
                if path.name.endswith(suffix)
            )
        case_sets = []
        for dataset, folder, suffix in NNUNET_CASE_FILES:
            directory = self.nnunet_raw / dataset / folder
            if not directory.is_dir():
                raise FileNotFoundError(f"nnU-Net dataset directory not found: {directory}")
            case_sets.append(
                {
                    path.name[: -len(suffix)]
                    for path in directory.glob(f"*{suffix}")
                    if path.name.endswith(suffix)
                }
            )
        return sorted(set.intersection(*case_sets))

    def _validate_cases(self) -> None:
        missing = []
        for case_id in self.case_ids:
            if self.config.annotation_free:
                image_path = self._case_path(
                    "Dataset018_small_bowel", "imagesTr", case_id
                )
                seed_path = self.seed_dir / f"{case_id}.txt"
                for path in (image_path, seed_path):
                    if not path.is_file():
                        missing.append(str(path))
            else:
                for dataset, folder, _ in NNUNET_CASE_FILES:
                    path = self._case_path(dataset, folder, case_id)
                    if not path.is_file():
                        missing.append(str(path))
                if self.config.reward_supervised:
                    seed_path = self.seed_dir / f"{case_id}.txt"
                    if not seed_path.is_file():
                        missing.append(str(seed_path))
        if missing:
            raise FileNotFoundError(
                "Missing nnU-Net files:\n" + "\n".join(f"- {path}" for path in missing)
            )

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict:
        case_id = self.case_ids[index]
        patient_dir = self.cache_dir / case_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        if self.config.annotation_free:
            return load_annotation_free_subject_data(
                case_id=case_id,
                image_path=self._case_path(
                    "Dataset018_small_bowel", "imagesTr", case_id
                ),
                patient_dir=patient_dir,
                seed_path=self.seed_dir / f"{case_id}.txt",
                config=self.config,
            )
        data = load_subject_data(
            {
                "id": case_id,
                "image": self._case_path("Dataset018_small_bowel", "imagesTr", case_id),
                "small_bowel": self._case_path("Dataset018_small_bowel", "labelsTr", case_id),
                "duodenum": self._case_path("Dataset019_duodenum", "labelsTr", case_id),
                "colon": self._case_path("Dataset020_colon", "labelsTr", case_id),
                "path": None,
                "patient_dir": patient_dir,
            },
            self.config,
        )
        result = {
            "id": data["id"],
            "image": data["image"],
            "seg": data["seg"],
            "duodenum": data["duodenum"],
            "colon": data["colon"],
            "wall_map": data["wall_map"],
            "gdt_start": data["gdt_start"],
            "gdt_end": data["gdt_end"],
            "target_distance": data.get("target_distance"),
            "image_affine": data["image_affine"],
            "spacing": data["spacing"],
            "start_coord": data["start_coord"],
            "end_coord": data["end_coord"],
            "local_peaks": data["local_peaks"],
            "gt_path": data.get("gt_path"),
        }
        if self.config.reward_supervised:
            native_image = np.transpose(data["image"], (2, 1, 0))
            native_features = load_or_compute_navigation_filter_bank(
                native_image,
                data["image_affine"],
                patient_dir / "cache",
                self.config,
            )
            seed_xyz = np.asarray(
                np.loadtxt(self.seed_dir / f"{case_id}.txt", dtype=int),
                dtype=int,
            )
            if seed_xyz.size != 3:
                raise ValueError(
                    f"External seed for {case_id} must contain exactly three values"
                )
            seed_zyx = tuple(int(value) for value in seed_xyz.reshape(3)[::-1])
            if not result["seg"][seed_zyx]:
                raise ValueError(
                    f"External seed for {case_id} is outside the reward-supervision mask"
                )
            result["start_coord"] = seed_zyx
            result["image_features"] = np.transpose(
                native_features,
                (0, 3, 2, 1),
            )
        return result


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
    print(f"Loading subject {subject_data['id']}", flush=True)
    patient_dir = subject_data["patient_dir"]
    cache_dir = patient_dir / "cache"
    cache_dir.mkdir(exist_ok=True)  # Ensure cache directory exists

    # --- Load Base Data ---
    image_nii: nib.nifti1.Nifti1Image = nib.load(subject_data["image"])
    image_np = image_nii.get_fdata(dtype=np.float32)

    result["image"] = image_np
    result["image_affine"] = image_nii.affine
    result["spacing"] = image_nii.header.get_zooms()
    expected_spacing = (config.voxel_size_mm,) * 3
    if not np.allclose(result["spacing"][:3], expected_spacing, atol=1e-3):
        raise ValueError(
            f"NIfTI spacing {result['spacing'][:3]} for {result['id']} does not match "
            f"configured isotropic spacing {expected_spacing}. Pass the correct "
            "--voxel-size-mm value or resample the data before training."
        )

    # Load segmentations (ensure they exist before loading)
    seg_np = np.asanyarray(nib.load(subject_data["small_bowel"]).dataobj)
    result["seg"] = (seg_np > 0).astype(np.uint8)

    if subject_data.get("duodenum"):
        duodenum_np = np.asanyarray(nib.load(subject_data["duodenum"]).dataobj)
        result["duodenum"] = (duodenum_np > 0).astype(np.uint8)
    else:
        result["duodenum"] = None
    if subject_data.get("colon"):
        colon_np = np.asanyarray(nib.load(subject_data["colon"]).dataobj)
        result["colon"] = (colon_np > 0).astype(np.uint8)
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
    start_coord, end_coord = None, None
    cache_valid = False

    if start_end_cache_path.exists():
        try:
            start_coord_np, end_coord_np = np.loadtxt(start_end_cache_path, dtype=int)
            start_coord = tuple(start_coord_np)
            end_coord = tuple(end_coord_np)
            # Validate coordinates are within mask
            if result["seg"][start_coord] > 0 and result["seg"][end_coord] > 0:
                cache_valid = True
            else:
                print(
                    f"Warning: Cached coordinates for {result['id']} are outside mask. Recalculating."
                )
        except Exception as e:
            print(f"Warning: Could not load or validate cache for {result['id']}: {e}")

    if not cache_valid:
        if result["duodenum"] is not None and result["colon"] is not None:
            # Get them in XYZ order
            start_coord, end_coord = find_start_end(
                duodenum_volume=result["duodenum"],
                colon_volume=result["colon"],
                small_bowel_volume=result["seg"],
            )
        elif result.get("gt_path") is not None:
            # Fallback for phantoms: use GT path.
            # Nibabel arrays and path.npy both use native XYZ voxel order.
            start_coord = tuple(result["gt_path"][0].astype(int))
            end_coord = tuple(result["gt_path"][-1].astype(int))
            print(f"Using GT path as start/end fallback for {result['id']}")
        else:
            raise ValueError(
                f"Neither anatomical segmentations nor GT path available to find start/end for {result['id']}."
            )

        # Save validated/recalculated coordinates
        np.savetxt(start_end_cache_path, (start_coord, end_coord), fmt="%d")

        # Invalidate dependent caches (GDTs)
        for gdt_file in [CACHE_FILES["gdt_start"], CACHE_FILES["gdt_end"]]:
            p = cache_dir / gdt_file
            if p.exists():
                p.unlink()
            if p.with_suffix(".nii.gz").exists():
                p.with_suffix(".nii.gz").unlink()

    result["start_coord"] = start_coord
    result["end_coord"] = end_coord

    # --- Load/Calculate Wall Map ---
    sigma_key = "-".join(str(sigma) for sigma in config.wall_map_sigmas)
    wall_map_cache_path = cache_dir / f"wall_map_sigmas-{sigma_key}.nii"
    if wall_map_cache_path.exists():
        wall_map_np = nib.load(wall_map_cache_path).get_fdata(dtype=np.float32)
    else:
        wall_map_np = compute_wall_map(result["image"], sigmas=config.wall_map_sigmas)
        # Save in Nifti format (preserving original orientation for saving)
        wall_map_nii_save = nib.Nifti1Image(wall_map_np, result["image_affine"])
        nib.save(wall_map_nii_save, wall_map_cache_path)
    result["wall_map"] = wall_map_np

    # --- Load/Calculate GDT (Start) ---
    gdt_start_cache_path = cache_dir / CACHE_FILES["gdt_start"]
    if gdt_start_cache_path.exists():
        gdt_start_np = nib.load(gdt_start_cache_path).get_fdata(dtype=np.float32)
    else:
        # Use numpy seg and start_coord for calculation
        gdt_start_np = compute_gdt(result["seg"], result["start_coord"], result["spacing"])
        # Save in Nifti format
        gdt_start_np = gdt_start_np.astype(np.float32)
        gdt_start_nii_save = nib.Nifti1Image(gdt_start_np, result["image_affine"])
        nib.save(gdt_start_nii_save, gdt_start_cache_path)
    result["gdt_start"] = gdt_start_np

    # --- Load/Calculate GDT (End) ---
    gdt_end_cache_path = cache_dir / CACHE_FILES["gdt_end"]
    if gdt_end_cache_path.exists():
        gdt_end_np = nib.load(gdt_end_cache_path).get_fdata(dtype=np.float32)
    else:
        # Use numpy seg and end_coord for calculation
        gdt_end_np = compute_gdt(result["seg"], result["end_coord"], result["spacing"])
        gdt_end_np = gdt_end_np.astype(np.float32)
        # Save in Nifti format
        gdt_end_nii_save = nib.Nifti1Image(gdt_end_np, result["image_affine"])
        nib.save(gdt_end_nii_save, gdt_end_cache_path)
    # Handles weird edge case related to the local peaks
    # Disconnected segmentation components lead to wildly inconsistent result, in particular when using the GDT which turns any unreachable point into -inf.
    result["gdt_end"] = gdt_end_np
    if not np.isfinite(gdt_end_np[result["start_coord"]]):
        raise ValueError(
            f"Anatomical endpoints for {result['id']} are disconnected in the "
            "small-bowel segmentation; full end-to-end traversal is impossible."
        )

    # ``getattr`` keeps long-running workers compatible if a newer checkout
    # adds the derived flag after the parent process instantiated Config.
    needs_target_distance = getattr(
        config,
        "needs_target_distance",
        bool(
            config.target_recovery_reward_scale
            or getattr(config, "target_distance_penalty_scale", 0.0)
            or getattr(
                config,
                "gate_positive_shaping_on_target_segment",
                False,
            )
        ),
    )
    if needs_target_distance:
        target_distance_cache_path = cache_dir / CACHE_FILES["target_distance"]
        if target_distance_cache_path.exists():
            target_distance_np = nib.load(target_distance_cache_path).get_fdata(
                dtype=np.float32
            )
        else:
            connected_target = (
                (result["seg"] != 0)
                & np.isfinite(result["gdt_start"])
                & np.isfinite(result["gdt_end"])
            )
            target_distance_np = distance_transform_edt(
                ~connected_target,
                sampling=result["spacing"],
            ).astype(np.float32)
            nib.save(
                nib.Nifti1Image(target_distance_np, result["image_affine"]),
                target_distance_cache_path,
            )
        result["target_distance"] = target_distance_np

    local_peaks_cache_path = cache_dir / CACHE_FILES["local_peaks"]
    if local_peaks_cache_path.exists():
        local_peaks_np = np.loadtxt(local_peaks_cache_path, dtype=int)
    else:
        distances = distance_transform_edt(result["gdt_start"] > 0)
        local_peaks_np = peak_local_max(distances, min_distance=4, threshold_abs=3, num_peaks=1024)
        # Choose only peaks for which gdt > 0
        # peaks = [peak for peak in local_peaks_np if gdt_start_np[tuple(peak)] > 0]
        # local_peaks_np = np.array(peaks)
    local_peaks_np = normalize_coordinate_rows(
        local_peaks_np,
        fallback=(result["start_coord"], result["end_coord"]),
    )
    np.savetxt(local_peaks_cache_path, local_peaks_np, fmt="%d")
    result["local_peaks"] = local_peaks_np

    if result.get("gt_path") is None and config.nnunet_generate_expert_path:
        expert_path_cache = cache_dir / CACHE_FILES["expert_path"]
        if expert_path_cache.exists():
            expert_path = np.load(expert_path_cache)
        else:
            expert_path = skeleton_covering_route(
                result["seg"],
                result["start_coord"],
                result["end_coord"],
            )
            np.save(expert_path_cache, expert_path)
        result["gt_path"] = normalize_coordinate_rows(
            expert_path,
            fallback=(result["start_coord"], result["end_coord"]),
        )

    if True:
        # Transpose everything
        result["image"] = np.transpose(result["image"], (2, 1, 0))
        result["seg"] = np.transpose(result["seg"], (2, 1, 0))
        result["duodenum"] = (
            np.transpose(result["duodenum"], (2, 1, 0)) if result["duodenum"] is not None else None
        )
        result["colon"] = (
            np.transpose(result["colon"], (2, 1, 0)) if result["colon"] is not None else None
        )
        result["spacing"] = result["spacing"][::-1]
        result["start_coord"] = result["start_coord"][::-1]
        result["end_coord"] = result["end_coord"][::-1]
        result["wall_map"] = np.transpose(result["wall_map"], (2, 1, 0))
        result["gdt_start"] = np.transpose(result["gdt_start"], (2, 1, 0))
        result["gdt_end"] = np.transpose(result["gdt_end"], (2, 1, 0))
        if result.get("target_distance") is not None:
            result["target_distance"] = np.transpose(
                result["target_distance"],
                (2, 1, 0),
            )
        result["local_peaks"] = np.fliplr(result["local_peaks"])
        # path.npy is native XYZ on disk. Keep the conversion next to the
        # volume transposes so the path and every spatial tensor enter the
        # environment in the same ZYX convention.
        if result.get("gt_path") is not None:
            result["gt_path"] = np.fliplr(result["gt_path"])

    release_gpu_preprocessing_memory()
    return result


class MRIPathDataset(Dataset):
    """
    Dataset class for MRI path tracking that scans a directory for matching files.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: Config,
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        self.subjects = self._find_subjects()
        print(f"Found {len(self.subjects)} complete subject(s) in {data_dir}")

    def _find_subjects(self) -> List[Dict[str, Any]]:
        patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        subjects = []

        for patient_dir in patient_dirs:
            subject_id = patient_dir.name

            mri_file = patient_dir / MRI_PATH_FILE_PATTERNS["mri"]
            mri_file = mri_file if mri_file.exists() else mri_file.with_suffix(".nii.gz")

            small_bowel_seg_file = patient_dir / MRI_PATH_FILE_PATTERNS["small_bowel_seg"]
            small_bowel_seg_file = (
                small_bowel_seg_file
                if small_bowel_seg_file.exists()
                else small_bowel_seg_file.with_suffix(".nii.gz")
            )

            if mri_file.exists() and small_bowel_seg_file.exists():
                # Find all path_i.npy files
                path_files = sorted(
                    sorted(patient_dir.glob(f"{MRI_PATH_FILE_PATTERNS['path_prefix']}*.npy"))
                )

                if path_files:
                    subject = {
                        "id": subject_id,
                        "mri": mri_file,
                        "small_bowel_seg": small_bowel_seg_file,
                        "paths": path_files,
                        "patient_dir": patient_dir,
                    }
                    subjects.append(subject)
                else:
                    print(f"Warning: No path_i.npy files found for subject {subject_id}. Skipping.")
            else:
                print(
                    f"Warning: MRI or small bowel segmentation missing for subject {subject_id}. Skipping."
                )
        return subjects

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, int) or np.issubdtype(type(idx), np.integer):
            subject = self.subjects[idx]
            data = load_mri_path_data(subject, self.config)

            # Return data in a similar structure to SmallBowelDataset for compatibility
            # with existing training loops, but with MRI-specific fields.
            # gdt_start, gdt_end, local_peaks are zeros as requested.
            # duodenum and colon are None.
            return {
                "id": data["id"],
                "image": data["mri"],  # Renamed from 'image' to 'mri' in load_mri_path_data
                "seg": data["small_bowel_seg"],  # Renamed from 'seg' to 'small_bowel_seg'
                "duodenum": None,  # Not present in this dataset
                "colon": None,  # Not present in this dataset
                "wall_map": data["wall_map"],
                "gdt_start": data["gdt_start"],  # Zeros
                "gdt_end": data["gdt_end"],  # Zeros
                "image_affine": data["image_affine"],
                "spacing": data["spacing"],
                "start_coord": data["start_coord"],  # First point of each path
                "end_coord": data["end_coord"],  # Last point of each path
                "local_peaks": data["local_peaks"],  # Zeros
                "paths": data["paths"],  # All loaded paths
            }
        else:
            return [self[i] for i in range(*idx.indices(len(self)))]


def load_mri_path_data(subject_data: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """
    Load data for a single subject in the MRI path dataset.

    Args:
        subject_data: Dictionary with paths to the subject's files and patient_dir
        config: Configuration object

    Returns:
        Dictionary with loaded data arrays (numpy format)
    """
    result = {"id": subject_data["id"]}
    patient_dir = subject_data["patient_dir"]
    cache_dir = patient_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # --- Load Base Data ---
    mri_nii: nib.nifti1.Nifti1Image = nib.load(subject_data["mri"])
    mri_np = mri_nii.get_fdata(dtype=np.float32)

    result["mri"] = mri_np
    result["image_affine"] = mri_nii.affine
    result["spacing"] = mri_nii.header.get_zooms()

    small_bowel_seg_np = np.asanyarray(nib.load(subject_data["small_bowel_seg"]).dataobj)
    result["small_bowel_seg"] = (small_bowel_seg_np).astype(np.uint8)

    # Load all path_i.npy files
    loaded_paths = []
    start_coords = []
    end_coords = []
    for path_file in subject_data["paths"]:
        try:
            path_np = np.loadtxt(path_file, dtype=int)
            loaded_paths.append(path_np)
            # Start and end will be the first point and the last point of each path
            if path_np.shape[0] > 0:
                start_coords.append(tuple(path_np[0].astype(int)))
                end_coords.append(tuple(path_np[-1].astype(int)))
            else:
                start_coords.append(None)
                end_coords.append(None)
        except Exception as e:
            print(f"Warning: Could not load path {path_file}: {e}")
            loaded_paths.append(None)
            start_coords.append(None)
            end_coords.append(None)
    result["paths"] = loaded_paths
    result["start_coord"] = start_coords  # List of start coords for each path
    result["end_coord"] = end_coords  # List of end coords for each path

    # --- Calculate Wall Map (still relevant for MRI) ---
    wall_map_cache_path = cache_dir / CACHE_FILES["wall_map"]
    if wall_map_cache_path.exists():
        wall_map_np = nib.load(wall_map_cache_path).get_fdata(dtype=np.float32)
    else:
        wall_map_np = compute_wall_map(result["mri"], sigmas=config.wall_map_sigmas)
        wall_map_nii_save = nib.Nifti1Image(wall_map_np, result["image_affine"])
        nib.save(wall_map_nii_save, wall_map_cache_path)
    result["wall_map"] = wall_map_np

    # --- Return zeros for gdt_start, gdt_end and local_peaks ---
    # These are typically derived from segmentations and start/end points,
    # but the request specifies returning zeros.
    image_shape = result["mri"].shape
    result["gdt_start"] = np.zeros(image_shape, dtype=np.float32)
    result["gdt_end"] = np.zeros(image_shape, dtype=np.float32)
    result["local_peaks"] = np.zeros((0, 3), dtype=int)  # Kx3, K can be 0

    # Transpose everything if needed (assuming the same logic as SmallBowelDataset)
    if True:  # This condition is always true in the original, so keeping it.
        result["mri"] = np.transpose(result["mri"], (2, 1, 0))
        result["small_bowel_seg"] = np.transpose(result["small_bowel_seg"], (2, 1, 0))
        result["spacing"] = result["spacing"][::-1]
        # Start and end coords are lists of tuples, need to transpose each tuple
        result["start_coord"] = [
            tuple(c[::-1]) if c is not None else None for c in result["start_coord"]
        ]
        result["end_coord"] = [
            tuple(c[::-1]) if c is not None else None for c in result["end_coord"]
        ]
        result["wall_map"] = np.transpose(result["wall_map"], (2, 1, 0))
        result["gdt_start"] = np.transpose(result["gdt_start"], (2, 1, 0))
        result["gdt_end"] = np.transpose(result["gdt_end"], (2, 1, 0))
        result["paths"] = [np.fliplr(p) for p in result["paths"]]
        # local_peaks is already (0,3), so no need to flip. If it were to contain data, it would need flipping.
        # result["local_peaks"] = np.fliplr(result["local_peaks"]) # This would error on (0,3) array

    return result
