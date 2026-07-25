import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


def fix_phantom_cache(data_dir: Path):
    subjects = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Checking {len(subjects)} subjects in {data_dir}")

    for subject_dir in tqdm(subjects):
        cache_dir = subject_dir / "cache"
        start_end_path = cache_dir / "start_end.npy"
        path_file = subject_dir / "path.npy"
        seg_path = subject_dir / "segmentations" / "small_bowel.nii"
        if not seg_path.exists():
            seg_path = seg_path.with_suffix(".nii.gz")

        if not path_file.exists():
            continue

        # Load ground truth path
        try:
            gt_path = np.loadtxt(path_file, dtype=int)
            if gt_path.shape[0] < 2:
                continue

            # The path in phantoms seems to be in XYZ, but let's be careful.
            # In dataset.py:197, it does np.fliplr(result["gt_path"]) if True.
            # However, start_end.npy is saved before the fliplr in the original code logic?
            # No, find_start_end (which phantoms don't have) returns them.

            new_start = gt_path[0]
            new_end = gt_path[-1]

            # Save it
            cache_dir.mkdir(exist_ok=True)
            np.savetxt(start_end_path, (new_start, new_end), fmt="%d")

            # Also delete GDT caches if they exist, as they depend on start/end
            gdt_start = cache_dir / "gdt_start.nii"
            gdt_end = cache_dir / "gdt_end.nii"
            for p in [
                gdt_start,
                gdt_end,
                gdt_start.with_suffix(".nii.gz"),
                gdt_end.with_suffix(".nii.gz"),
            ]:
                if p.exists():
                    p.unlink()

        except Exception as e:
            print(f"Error processing {subject_dir.name}: {e}")


if __name__ == "__main__":
    phantom_dir = Path("/home/matey/project/segmentor/data/phantoms")
    fix_phantom_cache(phantom_dir)
