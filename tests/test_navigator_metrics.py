import unittest

import numpy as np

from navigator.metrics import compute_path_metrics, physical_path_tube, rasterize_path


class NavigatorMetricsTest(unittest.TestCase):
    def test_rasterize_path_fills_sparse_segments(self):
        history = np.asarray([(2, 2, 2), (2, 2, 6)])
        centerline = rasterize_path((9, 9, 9), history)

        self.assertEqual(int(centerline.sum()), 5)
        self.assertTrue(centerline[2, 2, 4])

    def test_physical_tube_uses_euclidean_spacing(self):
        history = np.asarray([(8, 8, 8)])
        tube = physical_path_tube(
            (17, 17, 17),
            history,
            spacing_mm=(1.5, 1.5, 1.5),
            radius_mm=9,
        )

        self.assertTrue(tube[14, 8, 8])  # 6 voxels = 9 mm
        self.assertFalse(tube[15, 8, 8])
        self.assertTrue(tube[12, 12, 8])  # sqrt(4^2 + 4^2) * 1.5 < 9
        self.assertFalse(tube[13, 12, 8])  # sqrt(5^2 + 4^2) * 1.5 > 9

    def test_endpoint_tolerance_is_independent_from_path_radius(self):
        shape = (17, 17, 17)
        history = np.asarray([(8, 8, 2), (8, 8, 8)])
        target = physical_path_tube(
            shape,
            history,
            spacing_mm=(1.5, 1.5, 1.5),
            radius_mm=9,
        )
        metrics = compute_path_metrics(
            target,
            history,
            goal=(8, 8, 11),
            spacing_mm=(1.5, 1.5, 1.5),
            path_radius_mm=9,
            endpoint_tolerance_mm=3,
            success_dice=0.4,
        )

        self.assertGreaterEqual(metrics.dice, 0.4)
        self.assertAlmostEqual(metrics.endpoint_distance_mm, 4.5)
        self.assertFalse(metrics.endpoint_reached)
        self.assertFalse(metrics.traversal_success)


if __name__ == "__main__":
    unittest.main()
