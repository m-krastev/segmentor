import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from navigator.config import Config
from navigator.dataset import NNUNetActualDataset


class AnnotationFreeDatasetTest(unittest.TestCase):
    def test_nnunet_policy_loader_requires_no_label_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nnunet_raw = root / "nnUNet_raw"
            image_dir = nnunet_raw / "Dataset018_small_bowel" / "imagesTr"
            seed_dir = root / "seeds"
            cache_dir = root / "cache"
            image_dir.mkdir(parents=True)
            seed_dir.mkdir()

            case_id = "case001"
            image = np.random.default_rng(3).normal(size=(12, 13, 14)).astype(np.float32)
            nib.save(
                nib.Nifti1Image(image, np.eye(4)),
                image_dir / f"{case_id}_0000.nii.gz",
            )
            # Native NIfTI XYZ; the environment receives reversed ZYX.
            (seed_dir / f"{case_id}.txt").write_text("3 5 7\n")

            config = Config(
                nnunet_raw_dir=str(nnunet_raw),
                nnunet_cache_dir=str(cache_dir),
                nnunet_seed_dir=str(seed_dir),
                annotation_free=True,
                use_immediate_gdt_reward=False,
                terminate_on_success=False,
                coverage_reward_scale=0,
                gdt_reward_scale=0,
                r_final=0,
                r_val1=0,
                voxel_size_mm=1.0,
                patch_size_mm=8,
            )
            dataset = NNUNetActualDataset(
                nnunet_raw=nnunet_raw,
                config=config,
                cache_dir=cache_dir,
                case_ids=[case_id],
            )
            subject = dataset[0]

            self.assertEqual(subject["start_coord"], (7, 5, 3))
            self.assertEqual(subject["image"].shape, (14, 13, 12))
            self.assertEqual(subject["wall_map"].shape, subject["image"].shape)
            self.assertEqual(
                subject["image_features"].shape,
                (4, *subject["image"].shape),
            )
            for forbidden_key in (
                "seg",
                "duodenum",
                "colon",
                "gdt_start",
                "gdt_end",
                "end_coord",
                "local_peaks",
                "gt_path",
            ):
                self.assertNotIn(forbidden_key, subject)


if __name__ == "__main__":
    unittest.main()
