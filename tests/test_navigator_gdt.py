import unittest

import numpy as np

from navigator.utils import compute_gdt


class NavigatorGeodesicDistanceTests(unittest.TestCase):
    def test_background_is_an_impassable_barrier(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask[3, 3, 3] = 1

        distance = compute_gdt(mask, (1, 1, 1), voxel_size=1.0)

        self.assertEqual(distance[1, 1, 1], 0.0)
        self.assertEqual(distance[3, 3, 3], -np.inf)
        self.assertEqual(distance[2, 2, 2], -np.inf)

    def test_diagonal_connectivity_matches_environment_movement(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[0, 0, 0] = 1
        mask[1, 1, 1] = 1
        mask[2, 2, 2] = 1

        distance = compute_gdt(mask, (0, 0, 0), voxel_size=1.5)

        self.assertAlmostEqual(
            float(distance[2, 2, 2]),
            2 * np.sqrt(3) * 1.5,
            places=5,
        )

    def test_start_must_be_inside_mask(self):
        mask = np.ones((3, 3, 3), dtype=np.uint8)
        mask[1, 1, 1] = 0

        with self.assertRaisesRegex(ValueError, "outside the segmentation"):
            compute_gdt(mask, (1, 1, 1), voxel_size=1.0)


if __name__ == "__main__":
    unittest.main()
