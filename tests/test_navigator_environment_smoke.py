import unittest
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from navigator.config import Config
from navigator.environment import SmallBowelEnv, repeat_loader
from navigator.metrics import physical_path_tube
from navigator.pretrain import (
    _geodesic_expert_action,
    _monotonic_expert_action,
    _resynchronize_path_index,
)
from navigator.utils import compute_gdt


class NavigatorEnvironmentSmokeTest(unittest.TestCase):
    def test_repeat_loader_restarts_without_itertools_cycle(self):
        iterator = repeat_loader([1, 2, 3])
        self.assertEqual([next(iterator) for _ in range(8)], [1, 2, 3, 1, 2, 3, 1, 2])

    def test_gdt_reward_scale_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "gdt_reward_scale"):
            Config(gdt_reward_scale=-1)

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

    def test_local_path_dilation_is_euclidean(self):
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
            initial = environment._reset()
            expected = torch.zeros(shape, dtype=torch.uint8, device=device)
            coordinates = torch.stack(
                torch.meshgrid(
                    *(torch.arange(size, device=device) for size in shape),
                    indexing="ij",
                ),
                dim=-1,
            )
            center = torch.tensor(start, device=device)
            expected[(coordinates - center).float().square().sum(dim=-1).sqrt() <= 2.0] = 1
            self.assertTrue(torch.equal(environment.cumulative_path_mask, expected))
            self.assertEqual(environment.path_voxels, int(expected.sum().item()))
            torch.testing.assert_close(
                initial["actor"][0, 2],
                torch.ones(config.patch_size_vox, device=device),
            )
            environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            independent = physical_path_tube(
                shape,
                environment.get_tracking_history(),
                spacing_mm=(1.0, 1.0, 1.0),
                radius_mm=2,
            )
            self.assertTrue(
                torch.equal(
                    environment.cumulative_path_mask.bool().cpu(),
                    torch.from_numpy(independent),
                )
            )
        finally:
            environment.close()

    def test_goal_distance_channel_is_endpoint_directed_and_opt_in(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            observe_goal_distance=True,
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
            initial = environment._reset()
            goal_distance = initial["actor"][0, -1]
            center = tuple(size // 2 for size in config.patch_size_vox)
            self.assertEqual(config.observation_channels, 6)
            self.assertGreater(goal_distance[center[0], center[1], center[2] + 1].item(), 0)
            self.assertLess(goal_distance[center[0], center[1], center[2] - 1].item(), 0)
        finally:
            environment.close()

        self.assertEqual(Config().observation_channels, 5)

    def test_coverage_gate_latches_mask_constrained_goal_planner(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=4,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            success_coverage_threshold=0.4,
            coverage_gated_goal_planner=True,
        )
        shape = (16, 16, 16)
        start = (8, 8, 4)
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
            environment.current_coverage = config.success_coverage_threshold
            transition = environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.0, 0.5, 0.5]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertTrue(environment._goal_planner_active)
            self.assertEqual(environment.current_pos_vox, (8, 8, 8))
            self.assertLess(
                transition["info", "max_gdt_achieved"].item(),
                float("inf"),
            )
        finally:
            environment.close()

        self.assertFalse(Config().coverage_gated_goal_planner)

    def test_nine_mm_radius_and_three_mm_endpoint_are_independent(self):
        config = Config(
            device="cpu",
            patch_size_mm=24,
            voxel_size_mm=1.5,
            max_step_displacement_mm=6,
            cumulative_path_radius_mm=9,
            endpoint_tolerance_mm=3,
        )

        self.assertEqual(config.cumulative_path_radius_vox, 6)
        self.assertEqual(config.endpoint_tolerance_vox, 2)

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

    def test_action_magnitude_controls_step_length(self):
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
        end = (4, 4, 12)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[4, 4, 4:13] = 1
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            one_voxel_forward = torch.tensor([0.5, 0.5, 0.625], device=device)
            self.assertEqual(
                environment._project_action_to_allowed_displacement(one_voxel_forward),
                (0, 0, 1),
            )
            full_forward = torch.tensor([0.5, 0.5, 1.0], device=device)
            self.assertEqual(
                environment._project_action_to_allowed_displacement(full_forward),
                (0, 0, 4),
            )
        finally:
            environment.close()

    def test_geodesic_expert_reduces_mask_constrained_goal_distance(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            endpoint_tolerance_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 10)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[4, 4, 4:11] = 1
        subject = self._make_subject(shape, start, end, segmentation)
        subject["gdt_start"] = compute_gdt(segmentation, start)
        subject["gdt_end"] = compute_gdt(segmentation, end)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            before = environment.current_goal_distance
            expert_action = _geodesic_expert_action(environment)
            displacement = environment._project_action_to_allowed_displacement(expert_action)
            next_position = tuple(
                np.asarray(environment.current_pos_vox) + np.asarray(displacement)
            )
            self.assertLess(environment.goal_distance_map[next_position], before)
        finally:
            environment.close()

    def test_monotonic_expert_targets_final_waypoint(self):
        environment = SimpleNamespace(
            gt_path_voxels=np.asarray(
                [(4, 4, 4), (4, 4, 5), (4, 4, 6)],
                dtype=int,
            ),
            current_pos_vox=(4, 4, 5),
            config=SimpleNamespace(max_step_vox=4),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        action, path_index = _monotonic_expert_action(environment, path_index=1)

        self.assertEqual(path_index, 2)
        torch.testing.assert_close(action, torch.tensor([0.5, 0.5, 0.625]))

    def test_policy_rollout_cursor_only_rejoins_local_future_route(self):
        route = np.asarray([(4, 4, z) for z in range(24)], dtype=int)
        route[20] = route[3]
        environment = SimpleNamespace(
            gt_path_voxels=route,
            current_pos_vox=(4, 4, 3),
            config=SimpleNamespace(max_step_vox=4),
        )

        self.assertEqual(_resynchronize_path_index(environment, path_index=0), 3)

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
            endpoint_tolerance_mm=1,
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
