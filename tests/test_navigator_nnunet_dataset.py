import tempfile
import unittest
from pathlib import Path

from navigator.config import Config
from navigator.dataset import NNUNET_CASE_FILES, NNUNetActualDataset


class NNUNetActualDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "nnUNet_raw"
        self.cache_dir = Path(self.temp_dir.name) / "cache"
        self.config = Config(
            device="cpu",
            patch_size_mm=8,
            voxel_size_mm=1.0,
        )
        for dataset, folder, _ in NNUNET_CASE_FILES:
            (self.root / dataset / folder).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_case_file(self, case_id, dataset, folder, suffix):
        (self.root / dataset / folder / f"{case_id}{suffix}").touch()

    def add_complete_case(self, case_id):
        for dataset, folder, suffix in NNUNET_CASE_FILES:
            self.add_case_file(case_id, dataset, folder, suffix)

    def test_discovers_only_complete_case_intersection(self):
        self.add_complete_case("s0002")
        self.add_complete_case("s0001")
        dataset, folder, suffix = NNUNET_CASE_FILES[0]
        self.add_case_file("incomplete", dataset, folder, suffix)

        actual = NNUNetActualDataset(
            nnunet_raw=self.root,
            cache_dir=self.cache_dir,
            config=self.config,
        )

        self.assertEqual(actual.case_ids, ["s0001", "s0002"])
        self.assertEqual(actual.subjects, [{"id": "s0001"}, {"id": "s0002"}])

    def test_explicit_missing_case_reports_required_paths(self):
        with self.assertRaisesRegex(FileNotFoundError, "s9999_0000.nii.gz"):
            NNUNetActualDataset(
                nnunet_raw=self.root,
                case_ids=["s9999"],
                cache_dir=self.cache_dir,
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
