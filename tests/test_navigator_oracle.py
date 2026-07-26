import unittest

import numpy as np
from skimage.morphology import skeletonize

from navigator.oracle import path_dice, skeleton_covering_route


class NavigatorSkeletonOracleTests(unittest.TestCase):
    def test_route_covers_branches_and_preserves_endpoints(self):
        mask = np.zeros((11, 11, 11), dtype=np.uint8)
        mask[5, 5, 1:10] = 1
        mask[5, 5:10, 5] = 1
        start = (5, 5, 1)
        end = (5, 5, 9)

        route = skeleton_covering_route(mask, start, end)

        self.assertEqual(tuple(route[0]), start)
        self.assertEqual(tuple(route[-1]), end)
        self.assertTrue(np.all(np.max(np.abs(np.diff(route, axis=0)), axis=1) <= 1))
        skeleton_coordinates = {tuple(coordinate) for coordinate in np.argwhere(skeletonize(mask))}
        self.assertTrue(skeleton_coordinates.issubset({tuple(point) for point in route}))
        self.assertGreater(path_dice(mask, route, radius_vox=0), 0.9)

    def test_disconnected_endpoint_is_rejected(self):
        mask = np.zeros((7, 7, 7), dtype=np.uint8)
        mask[1, 1, 1:4] = 1
        mask[5, 5, 3:6] = 1

        with self.assertRaisesRegex(ValueError, "No mask-constrained path"):
            skeleton_covering_route(mask, (1, 1, 1), (5, 5, 5))

    def test_route_ignores_nearby_disconnected_skeleton(self):
        mask = np.zeros((22, 22, 22), dtype=np.uint8)
        mask[5:16, 5:16, 5:16] = 1
        mask[3, 10, 10] = 1
        start = (5, 10, 10)
        end = (15, 10, 10)

        route = skeleton_covering_route(mask, start, end)

        self.assertEqual(tuple(route[0]), start)
        self.assertEqual(tuple(route[-1]), end)
        self.assertNotIn((3, 10, 10), {tuple(point) for point in route})
        self.assertTrue(np.asarray(mask[tuple(route.T)]).all())

    def test_empty_medial_skeleton_falls_back_to_mask_path(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        start = (5, 10, 10)
        end = (14, 10, 10)

        route = skeleton_covering_route(mask, start, end)

        self.assertEqual(tuple(route[0]), start)
        self.assertEqual(tuple(route[-1]), end)
        self.assertTrue(np.asarray(mask[tuple(route.T)]).all())


if __name__ == "__main__":
    unittest.main()
