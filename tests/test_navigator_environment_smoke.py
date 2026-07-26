import unittest

import numpy as np
import torch
from tensordict import TensorDict

from navigator.config import Config
from navigator.environment import SmallBowelEnv
from navigator.utils import BinaryDilation3D


class NavigatorEnvironmentSmokeTest(unittest.TestCase):
    @staticmethod
    def _make_subject(shape, start, end, segmentation):
        coordinates = np.indices(shape)
        gdt_start = np.sqrt(
            sum((coordinates[axis] - start[axis]) ** 2 for axis in range(3))
        ).astype(np.float32)
        gdt_end = np.sqrt(sum((coordinates[axis] - end[axis]) ** 2 for axis in range(3))).astype(
            np.float32
        )
        return {
            "id": "synthetic",
            "image": np.zeros(shape, dtype=np.float32),
            "seg": segmentation,
            "wall_map": np.zeros(shape, dtype=np.float32),
            "gdt_start": gdt_start,
            "gdt_end": gdt_end,
            "start_coord": start,
            "end_coord": end,
            "gt_path": np.asarray([start, end]),
            "spacing": (1.0, 1.0, 1.0),
            "image_affine": np.eye(4),
            "local_peaks": np.empty((0, 3), dtype=int),
        }

    def test_local_path_dilation_matches_repeated_star_kernel(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=2,
            allowed_area_radius_mm=0,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        end = (8, 8, 12)
        segmentation = np.ones(shape, dtype=np.uint8)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            expected = torch.zeros(shape, dtype=torch.uint8, device=device)
            expected[start] = 1
            dilation = torch.nn.Sequential(BinaryDilation3D(), BinaryDilation3D()).to(device)
            expected = dilation(expected[None, None]).squeeze()
            self.assertTrue(torch.equal(environment.cumulative_path_mask, expected))
            self.assertEqual(environment.path_voxels, int(expected.sum().item()))
        finally:
            environment.close()

    def test_timeout_is_terminal_and_preserves_ground_truth_path(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=1,
            max_episode_steps=1,
            r_final=10,
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 10)
        coordinates = np.indices(shape)
        gdt_start = np.sqrt(
            sum((coordinates[axis] - start[axis]) ** 2 for axis in range(3))
        ).astype(np.float32)
        gdt_end = np.sqrt(sum((coordinates[axis] - end[axis]) ** 2 for axis in range(3))).astype(
            np.float32
        )
        gt_path = np.asarray([(4, 4, x) for x in range(start[2], end[2] + 1)])
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[tuple(gt_path.T)] = 1
        subject = {
            "id": "synthetic",
            "image": np.zeros(shape, dtype=np.float32),
            "seg": segmentation,
            "wall_map": np.zeros(shape, dtype=np.float32),
            "gdt_start": gdt_start,
            "gdt_end": gdt_end,
            "start_coord": start,
            "end_coord": end,
            "gt_path": gt_path,
            "spacing": (1.0, 1.0, 1.0),
            "image_affine": np.eye(4),
            "local_peaks": np.empty((0, 3), dtype=int),
        }
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            initial = environment._reset()
            ground_truth_voxels = int(environment.gt_path_vol.sum().item())
            self.assertGreater(environment.cumulative_path_mask.sum().item(), 1)

            action = TensorDict(
                {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                batch_size=torch.Size([1]),
                device=device,
            )
            transition = environment._step(action)

            self.assertEqual(
                initial["actor"].shape,
                torch.Size([1, config.observation_channels, *config.patch_size_vox]),
            )
            self.assertEqual(
                initial["context"].shape,
                torch.Size([1, config.context_features]),
            )
            self.assertTrue(transition["done"].item())
            self.assertTrue(transition["terminated"].item())
            self.assertFalse(transition["truncated"].item())
            self.assertLess(transition["reward"].item(), 0.0)
            self.assertEqual(int(environment.gt_path_vol.sum().item()), ground_truth_voxels)
            self.assertEqual(environment.current_pos_vox, (4, 4, 6))
            self.assertLess(environment.current_goal_distance, environment.initial_goal_distance)
        finally:
            environment.close()

    def test_action_segment_cannot_cut_through_unsegmented_space(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=4,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 8)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[start] = 1
        segmentation[end] = 1
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            shortcut = TensorDict(
                {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                batch_size=torch.Size([1]),
                device=device,
            )
            transition = environment._step(shortcut)
            self.assertEqual(environment.current_pos_vox, start)
            self.assertFalse(transition["done"].item())
            self.assertLess(transition["reward"].item(), 0.0)
        finally:
            environment.close()

    def test_outward_and_zero_actions_project_to_valid_tangent(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=4,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 8)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[4, 4, 4:9] = 1

        for raw_action in (
            torch.tensor([[1.0, 0.5, 0.5]], device=device),
            torch.tensor([[0.5, 0.5, 0.5]], device=device),
        ):
            environment = SmallBowelEnv(
                config=config,
                dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
                num_episodes_per_sample=1,
                device=device,
            )
            try:
                environment._reset()
                transition = environment._step(
                    TensorDict(
                        {"action": raw_action},
                        batch_size=torch.Size([1]),
                        device=device,
                    )
                )
                self.assertNotEqual(environment.current_pos_vox, start)
                self.assertEqual(environment.current_pos_vox[:2], start[:2])
                self.assertFalse(transition["done"].item())
            finally:
                environment.close()

    def test_endpoint_alone_does_not_end_episode(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
            success_coverage_threshold=0.55,
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 6)
        segmentation = np.ones(shape, dtype=np.uint8)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            direct_to_endpoint = TensorDict(
                {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                batch_size=torch.Size([1]),
                device=device,
            )
            transition = environment._step(direct_to_endpoint)
            self.assertEqual(environment.current_pos_vox, end)
            self.assertFalse(transition["done"].item())
            self.assertEqual(transition["info", "final_success"].item(), 0.0)
        finally:
            environment.close()

    def test_traced_path_can_succeed(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
            success_coverage_threshold=0.55,
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 8)
        segmentation = np.zeros(shape, dtype=np.uint8)
        for z_coord in range(start[2], end[2] + 1):
            for delta in (
                (0, 0, 0),
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ):
                position = (
                    start[0] + delta[0],
                    start[1] + delta[1],
                    z_coord + delta[2],
                )
                segmentation[position] = 1
        subject = self._make_subject(shape, start, end, segmentation)
        subject["gt_path"] = np.asarray(
            [(start[0], start[1], z) for z in range(start[2], end[2] + 1)]
        )
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            forward = TensorDict(
                {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                batch_size=torch.Size([1]),
                device=device,
            )
            first = environment._step(forward)
            self.assertFalse(first["done"].item())
            second = environment._step(forward)
            self.assertTrue(second["done"].item())
            self.assertEqual(second["info", "final_success"].item(), 1.0)
            self.assertGreaterEqual(
                second["info", "final_coverage"].item(),
                config.success_coverage_threshold,
            )
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
