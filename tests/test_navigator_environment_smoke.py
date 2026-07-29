import unittest
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from navigator.config import Config
from navigator.environment import (
    REWARD_COMPONENT_INFO_KEYS,
    SmallBowelEnv,
    repeat_loader,
)
from navigator.metrics import physical_path_tube
from navigator.pretrain import (
    _geodesic_expert_action,
    _monotonic_expert_action,
    _resynchronize_path_index,
)
from navigator.utils import compute_gdt, get_patch


class NavigatorEnvironmentSmokeTest(unittest.TestCase):
    @staticmethod
    def _annotation_free_config(device: torch.device, **overrides):
        values = {
            "device": str(device),
            "patch_size_mm": 8,
            "voxel_size_mm": 1.0,
            "max_step_displacement_mm": 2,
            "cumulative_path_radius_mm": 1,
            "max_episode_steps": 4,
            "annotation_free": True,
            "use_immediate_gdt_reward": False,
            "terminate_on_success": False,
            "coverage_reward_scale": 0,
            "gdt_reward_scale": 0,
            "r_final": 0,
            "r_val1": 0,
        }
        values.update(overrides)
        return Config(**values)

    def test_annotation_free_configuration_rejects_privileged_defaults(self):
        with self.assertRaisesRegex(ValueError, "privileged options"):
            Config(annotation_free=True)

    def test_multichannel_patch_matches_independent_channel_extraction(self):
        volume = torch.arange(3 * 7 * 8 * 9, dtype=torch.float32).reshape(3, 7, 8, 9)
        for center in ((3, 4, 4), (0, 0, 0), (6, 7, 8)):
            actual = get_patch(volume, center, (6, 5, 4), pad_value=-7)
            expected = torch.stack(
                [
                    get_patch(channel, center, (6, 5, 4), pad_value=-7)
                    for channel in volume
                ]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_annotation_free_transitions_are_invariant_to_labels_and_endpoint(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = self._annotation_free_config(device)
        shape = (16, 16, 16)
        start = (8, 8, 8)
        image = np.random.default_rng(42).normal(size=shape).astype(np.float32)
        wall_map = np.random.default_rng(7).uniform(size=shape).astype(np.float32)

        first = self._make_subject(
            shape,
            start,
            (8, 8, 12),
            np.zeros(shape, dtype=np.uint8),
        )
        second = self._make_subject(
            shape,
            start,
            (2, 13, 4),
            np.random.default_rng(19).integers(0, 2, size=shape, dtype=np.uint8),
        )
        for subject in (first, second):
            subject["image"] = image.copy()
            subject["wall_map"] = wall_map.copy()
        second["gdt_start"] = np.full(shape, -1234, dtype=np.float32)
        second["gdt_end"] = np.full(shape, 9876, dtype=np.float32)
        second["gt_path"] = np.asarray([(1, 1, 1), (14, 14, 14)])

        environments = [
            SmallBowelEnv(
                config=config,
                dataset_iterator=iter([subject]),
                num_episodes_per_sample=1,
                device=device,
            )
            for subject in (first, second)
        ]
        try:
            resets = [environment._reset() for environment in environments]
            torch.testing.assert_close(resets[0]["actor"], resets[1]["actor"])
            torch.testing.assert_close(resets[0]["context"], resets[1]["context"])
            self.assertEqual(
                resets[0]["actor"].shape,
                torch.Size([1, 6, *config.patch_size_vox]),
            )
            self.assertEqual(resets[0]["context"].shape, torch.Size([1, 7]))
            self.assertIsNone(environments[0].seg)
            self.assertIsNone(environments[1].seg)

            actions = [
                torch.tensor([[0.5, 0.5, 1.0]], device=device),
                torch.tensor([[0.5, 1.0, 0.5]], device=device),
                torch.tensor([[0.5, 0.5, 0.0]], device=device),
            ]
            for action in actions:
                transitions = [
                    environment._step(
                        TensorDict(
                            {"action": action.clone()},
                            batch_size=torch.Size([1]),
                            device=device,
                        )
                    )
                    for environment in environments
                ]
                for key in ("actor", "context", "reward", "done", "terminated"):
                    torch.testing.assert_close(transitions[0][key], transitions[1][key])
                self.assertEqual(
                    environments[0].current_pos_vox,
                    environments[1].current_pos_vox,
                )
        finally:
            for environment in environments:
                environment.close()

    def test_annotation_free_action_projection_does_not_read_segmentation(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = self._annotation_free_config(device)
        shape = (16, 16, 16)
        start = (8, 8, 8)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[start] = 1
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter(
                [self._make_subject(shape, start, (8, 8, 9), segmentation)]
            ),
            num_episodes_per_sample=1,
            device=device,
        )
        try:
            environment._reset()
            transition = environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(environment.current_pos_vox, (8, 8, 10))
            self.assertFalse(transition["done"].item())
        finally:
            environment.close()

    def test_reward_supervision_changes_reward_but_not_policy_state_or_dynamics(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=4,
            reward_supervised=True,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        image = np.random.default_rng(23).normal(size=shape).astype(np.float32)
        wall_map = np.zeros(shape, dtype=np.float32)

        first = self._make_subject(
            shape,
            start,
            (8, 8, 12),
            np.ones(shape, dtype=np.uint8),
        )
        second_segmentation = np.zeros(shape, dtype=np.uint8)
        second_segmentation[start] = 1
        second = self._make_subject(
            shape,
            start,
            (2, 2, 2),
            second_segmentation,
        )
        # Match real mask-constrained GDTs: the requested next position is
        # finite in the first subject and infinite in the second. An infinite
        # supervised potential may change reward, but must not reject movement.
        second["gdt_end"].fill(np.inf)
        second["gdt_end"][start] = 12.0
        second["gdt_end"][second["end_coord"]] = 0.0
        for subject in (first, second):
            subject["image"] = image.copy()
            subject["wall_map"] = wall_map.copy()

        environments = [
            SmallBowelEnv(
                config=config,
                dataset_iterator=iter([subject]),
                num_episodes_per_sample=1,
                device=device,
            )
            for subject in (first, second)
        ]
        try:
            resets = [environment._reset() for environment in environments]
            torch.testing.assert_close(resets[0]["actor"], resets[1]["actor"])
            torch.testing.assert_close(resets[0]["context"], resets[1]["context"])
            action = torch.tensor([[0.5, 0.5, 1.0]], device=device)
            transitions = [
                environment._step(
                    TensorDict(
                        {"action": action.clone()},
                        batch_size=torch.Size([1]),
                        device=device,
                    )
                )
                for environment in environments
            ]
            torch.testing.assert_close(
                transitions[0]["actor"],
                transitions[1]["actor"],
            )
            torch.testing.assert_close(
                transitions[0]["context"],
                transitions[1]["context"],
            )
            self.assertEqual(
                environments[0].current_pos_vox,
                environments[1].current_pos_vox,
            )
            self.assertEqual(environments[0].current_pos_vox, (8, 8, 10))
            self.assertNotEqual(
                transitions[0]["reward"].item(),
                transitions[1]["reward"].item(),
            )
        finally:
            for environment in environments:
                environment.close()

    def test_recovery_reward_guides_return_without_profitable_round_trip(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=4,
            reward_supervised=True,
            target_recovery_reward_scale=1.0,
            coverage_reward_scale=0.0,
            r_final=0.0,
            terminate_on_success=False,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[start] = 1
        subject = self._make_subject(shape, start, start, segmentation)
        subject["gdt_start"].fill(np.inf)
        subject["gdt_end"].fill(np.inf)
        subject["gdt_start"][start] = 0.0
        subject["gdt_end"][start] = 0.0
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            leave = environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(environment.current_pos_vox, (8, 8, 10))
            recover = environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.5, 0.5, 0.0]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(environment.current_pos_vox, start)
            self.assertLess(leave["reward"].item(), 0.0)
            self.assertGreater(recover["reward"].item(), 0.0)
            self.assertLess(
                leave["reward"].item() + recover["reward"].item(),
                0.0,
            )
        finally:
            environment.close()

    def test_episodic_cell_bonus_is_first_visit_only_and_diminishes(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=8,
            reward_supervised=True,
            coverage_reward_scale=0.0,
            gdt_reward_scale=0.0,
            use_immediate_gdt_reward=False,
            r_final=0.0,
            terminate_on_success=False,
            episodic_cell_reward_scale=0.05,
            episodic_cell_size_mm=2.0,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        segmentation = np.ones(shape, dtype=np.uint8)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter(
                [self._make_subject(shape, start, (8, 8, 14), segmentation)]
            ),
            num_episodes_per_sample=1,
            device=device,
        )

        def step(z_action):
            return environment._step(
                TensorDict(
                    {
                        "action": torch.tensor(
                            [[0.5, 0.5, z_action]],
                            device=device,
                        )
                    },
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )

        try:
            environment._reset()
            first = step(1.0)
            second = step(1.0)
            revisit = step(0.0)
            self.assertAlmostEqual(
                first["info", "episodic_cell_reward"].item(),
                0.05,
                places=6,
            )
            self.assertAlmostEqual(
                second["info", "episodic_cell_reward"].item(),
                0.05 / np.sqrt(2.0),
                places=6,
            )
            self.assertEqual(
                revisit["info", "episodic_cell_reward"].item(),
                0.0,
            )
        finally:
            environment.close()

    def test_episodic_bonus_cannot_make_new_off_target_cell_profitable(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=4,
            reward_supervised=True,
            target_recovery_reward_scale=0.0,
            coverage_reward_scale=0.0,
            gdt_reward_scale=0.0,
            use_immediate_gdt_reward=False,
            r_final=0.0,
            terminate_on_success=False,
            episodic_cell_reward_scale=0.05,
            episodic_cell_size_mm=2.0,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[start] = 1
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter(
                [self._make_subject(shape, start, start, segmentation)]
            ),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            transition = environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertAlmostEqual(
                transition["info", "episodic_cell_reward"].item(),
                0.05,
                places=6,
            )
            self.assertLess(transition["reward"].item(), 0.0)
        finally:
            environment.close()

    def test_outside_penalty_covers_entire_action_segment(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=4,
            reward_supervised=True,
            target_recovery_reward_scale=1.0,
            coverage_reward_scale=0.0,
            terminate_on_success=False,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        end = (8, 8, 12)
        intact = np.ones(shape, dtype=np.uint8)
        gap = intact.copy()
        gap[8, 8, 9] = 0
        environments = [
            SmallBowelEnv(
                config=config,
                dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
                num_episodes_per_sample=1,
                device=device,
            )
            for segmentation in (intact, gap)
        ]

        try:
            for environment in environments:
                environment._reset()
            action = TensorDict(
                {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                batch_size=torch.Size([1]),
                device=device,
            )
            intact_transition = environments[0]._step(action.clone())
            gap_transition = environments[1]._step(action.clone())
            self.assertEqual(environments[0].current_pos_vox, (8, 8, 10))
            self.assertEqual(environments[1].current_pos_vox, (8, 8, 10))
            self.assertAlmostEqual(
                intact_transition["reward"].item()
                - gap_transition["reward"].item(),
                config.r_val1,
                places=5,
            )
        finally:
            for environment in environments:
                environment.close()

    def test_calibrated_reward_contract_rejects_cycles_and_shortcuts(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=8,
            reward_supervised=True,
            coverage_reward_scale=0.0,
            gdt_reward_scale=0.1,
            gdt_progress_normalization="max_step",
            target_recovery_reward_scale=0.05,
            target_distance_penalty_scale=0.1,
            target_distance_penalty_radius_mm=600.0,
            gate_positive_shaping_on_target_segment=True,
            r_val1=0.0,
            wall_penalty_scale=0.0,
            r_final=0.0,
            terminate_on_success=False,
            terminal_success_bonus=50.0,
            terminal_failure_penalty=0.0,
            episodic_cell_reward_scale=0.01,
            episodic_cell_size_mm=2.0,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        end = (8, 8, 12)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[8, 8, 4:13] = 1

        def action(x, y, z):
            return TensorDict(
                {"action": torch.tensor([[x, y, z]], device=device)},
                batch_size=torch.Size([1]),
                device=device,
            )

        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter(
                [self._make_subject(shape, start, end, segmentation)]
            ),
            num_episodes_per_sample=1,
            device=device,
        )
        try:
            environment._reset()
            forward = environment._step(action(0.5, 0.5, 1.0))
            backward = environment._step(action(0.5, 0.5, 0.0))
            self.assertGreater(forward["reward"].item(), 0.0)
            self.assertLess(
                forward["reward"].item() + backward["reward"].item(),
                0.0,
            )

            leave = environment._step(action(1.0, 0.5, 0.5))
            recover = environment._step(action(0.0, 0.5, 0.5))
            self.assertLess(leave["reward"].item(), 0.0)
            self.assertLess(
                leave["reward"].item() + recover["reward"].item(),
                0.0,
            )
            self.assertEqual(
                leave["info", "episodic_cell_reward"].item(),
                0.0,
            )
            self.assertAlmostEqual(
                sum(
                    leave["info", key].item()
                    for key in REWARD_COMPONENT_INFO_KEYS
                ),
                leave["reward"].item(),
                places=6,
            )
        finally:
            environment.close()

        shortcut_segmentation = np.zeros(shape, dtype=np.uint8)
        shortcut_segmentation[start] = 1
        shortcut_end = (8, 8, 10)
        shortcut_segmentation[shortcut_end] = 1
        shortcut_environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter(
                [
                    self._make_subject(
                        shape,
                        start,
                        shortcut_end,
                        shortcut_segmentation,
                    )
                ]
            ),
            num_episodes_per_sample=1,
            device=device,
        )
        try:
            shortcut_environment._reset()
            shortcut = shortcut_environment._step(action(0.5, 0.5, 1.0))
            self.assertEqual(shortcut_environment.current_pos_vox, shortcut_end)
            self.assertLess(shortcut["reward"].item(), 0.0)
            self.assertEqual(
                shortcut["info", "episodic_cell_reward"].item(),
                0.0,
            )
        finally:
            shortcut_environment.close()

    def test_revisit_penalty_uses_undilated_tail_and_is_length_normalized(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        revisit_scale = 0.2
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=4,
            cumulative_path_radius_mm=3,
            max_episode_steps=8,
            reward_supervised=True,
            use_immediate_gdt_reward=False,
            coverage_reward_scale=0.0,
            gdt_reward_scale=0.0,
            target_recovery_reward_scale=0.0,
            target_distance_penalty_scale=0.0,
            revisit_penalty_scale=revisit_scale,
            r_val1=0.0,
            wall_penalty_scale=0.0,
            step_penalty=0.0,
            episodic_cell_reward_scale=0.0,
            r_final=0.0,
            terminate_on_success=False,
        )
        shape = (24, 24, 24)
        start = (12, 12, 8)
        end = (12, 12, 20)
        segmentation = np.ones(shape, dtype=np.uint8)

        def action(z: float):
            return TensorDict(
                {
                    "action": torch.tensor(
                        [[0.5, 0.5, z]],
                        device=device,
                    )
                },
                batch_size=torch.Size([1]),
                device=device,
            )

        # A normalized Beta action of 0.625/0.375 requests +/-1 voxel when
        # max_step_vox is four; 1.0/0.0 requests +/-4 voxels.
        for forward_z, backward_z in ((0.625, 0.375), (1.0, 0.0)):
            with self.subTest(step=(forward_z, backward_z)):
                environment = SmallBowelEnv(
                    config=config,
                    dataset_iterator=iter(
                        [self._make_subject(shape, start, end, segmentation)]
                    ),
                    num_episodes_per_sample=1,
                    device=device,
                )
                try:
                    reset = environment._reset()
                    # The observation mask is the 3-voxel-radius Dice tube,
                    # whereas revisit bookkeeping begins as an empty thin
                    # centerline and is updated only after a valid action.
                    self.assertGreater(
                        int(reset["actor"][0, -1].sum().item()),
                        1,
                    )
                    self.assertEqual(
                        int(environment.cumulative_path_mask_pen.sum()),
                        0,
                    )

                    forward = environment._step(action(forward_z))
                    backward = environment._step(action(backward_z))
                    self.assertAlmostEqual(
                        forward["info", "reward_revisit"].item(),
                        0.0,
                        places=6,
                    )
                    self.assertAlmostEqual(
                        backward["info", "reward_revisit"].item(),
                        -revisit_scale,
                        places=6,
                    )
                    self.assertAlmostEqual(
                        backward["reward"].item(),
                        -revisit_scale,
                        places=6,
                    )
                finally:
                    environment.close()

    def test_revisit_penalty_scale_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "revisit_penalty_scale"):
            Config(revisit_penalty_scale=-0.01)

    def test_shin_normalized_contract_rejects_cycles_and_logs_exact_terms(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=8,
            reward_supervised=True,
            reward_contract="shin_normalized",
            observe_segmentation=True,
            coverage_reward_scale=50.0,
            step_penalty=10.0,
            terminate_on_success=False,
        )
        shape = (20, 20, 20)
        start = (10, 10, 6)
        end = (10, 10, 16)
        segmentation = np.ones(shape, dtype=np.uint8)
        subject = self._make_subject(shape, start, end, segmentation)
        subject["wall_map"][:] = 0.25
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )

        def action(z: float):
            return TensorDict(
                {"action": torch.tensor([[0.5, 0.5, z]], device=device)},
                batch_size=torch.Size([1]),
                device=device,
            )

        try:
            environment._reset()
            # Subject loading normalizes a constant nonzero wall volume to
            # one. Set the post-load response explicitly so this test checks
            # reward arithmetic rather than preprocessing normalization.
            environment.wall_map.fill_(0.25)
            forward = environment._step(action(1.0))
            backward = environment._step(action(0.0))
            expected_forward = 2.0 / np.sqrt(12.0) - 0.25
            self.assertAlmostEqual(
                forward["reward"].item(),
                expected_forward,
                places=5,
            )
            self.assertAlmostEqual(
                backward["reward"].item(),
                -0.25 - 2.0 / 3.0,
                places=5,
            )
            self.assertLess(
                forward["reward"].item() + backward["reward"].item(),
                0.0,
            )
            self.assertEqual(
                forward["info", "reward_coverage"].item(),
                0.0,
            )
            self.assertEqual(
                forward["info", "reward_step"].item(),
                0.0,
            )
            for transition in (forward, backward):
                self.assertAlmostEqual(
                    sum(
                        transition["info", key].item()
                        for key in REWARD_COMPONENT_INFO_KEYS
                    ),
                    transition["reward"].item(),
                    places=6,
                )
        finally:
            environment.close()

    def test_shin_outside_segmentation_overwrites_other_terms(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=8,
            reward_supervised=True,
            reward_contract="shin_normalized",
            terminate_on_success=False,
        )
        shape = (20, 20, 20)
        start = (10, 10, 6)
        end = (10, 10, 16)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[10, 10, 4:18] = 1
        subject = self._make_subject(shape, start, end, segmentation)
        subject["wall_map"][:] = 0.75
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )
        try:
            environment._reset()
            transition = environment._step(
                TensorDict(
                    {"action": torch.tensor([[1.0, 0.5, 0.5]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertAlmostEqual(transition["reward"].item(), -2.0 / 3.0)
            self.assertAlmostEqual(
                transition["info", "reward_off_target"].item(),
                -2.0 / 3.0,
            )
            self.assertAlmostEqual(
                sum(
                    transition["info", key].item()
                    for key in REWARD_COMPONENT_INFO_KEYS
                ),
                transition["reward"].item(),
                places=6,
            )
        finally:
            environment.close()

    def test_guarded_shin_rejects_background_crossing_segment(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=8,
            reward_supervised=True,
            reward_contract="shin_normalized_guarded",
            terminate_on_success=False,
        )
        shape = (20, 20, 20)
        start = (10, 10, 6)
        end = (10, 10, 16)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[start] = 1
        segmentation[10, 10, 8] = 1
        segmentation[end] = 1
        subject = self._make_subject(shape, start, end, segmentation)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )
        try:
            environment._reset()
            transition = environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(environment.current_pos_vox, (10, 10, 8))
            self.assertAlmostEqual(transition["reward"].item(), -2.0 / 3.0)
            self.assertEqual(transition["info", "reward_gdt"].item(), 0.0)
            self.assertEqual(transition["info", "reward_wall"].item(), 0.0)
            self.assertEqual(transition["info", "reward_step"].item(), 0.0)
            self.assertAlmostEqual(
                transition["info", "reward_off_target"].item(),
                -2.0 / 3.0,
            )
        finally:
            environment.close()

    def test_repaired_shin_gates_shortcut_credit_and_scales_off_target_distance(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=8,
            reward_supervised=True,
            reward_contract="shin_normalized_repaired",
            policy_observation_contract="shin_068_repaired",
            gamma=0.99,
            terminate_on_success=False,
        )
        shape = (20, 20, 20)
        start = (10, 10, 6)
        end = (10, 10, 16)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[start] = 1
        segmentation[10, 10, 8] = 1
        segmentation[end] = 1
        subject = self._make_subject(shape, start, end, segmentation)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )
        try:
            reset = environment._reset()
            self.assertEqual(
                reset["actor"].shape,
                torch.Size([1, 3, *config.patch_size_vox]),
            )
            self.assertEqual(reset["context"].shape, torch.Size([1, 3]))
            self.assertEqual(int(reset["actor"][0, 2].sum().item()), 1)
            self.assertEqual(int(environment.cumulative_path_mask_pen.sum()), 0)

            shortcut = environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(environment.current_pos_vox, (10, 10, 8))
            self.assertEqual(shortcut["info", "reward_gdt"].item(), 0.0)
            self.assertLess(shortcut["info", "reward_off_target"].item(), 0.0)
            self.assertGreater(
                shortcut["info", "reward_off_target"].item(),
                -2.0 / 3.0,
            )
            self.assertAlmostEqual(
                shortcut["info", "reward_step"].item(),
                -(1.0 - config.gamma) * (100.0 / 6.0),
                places=6,
            )
            self.assertEqual(int(shortcut["actor"][0, 2].sum().item()), 3)
            self.assertLess(shortcut["reward"].item(), 0.0)
            self.assertAlmostEqual(
                sum(
                    shortcut["info", key].item()
                    for key in REWARD_COMPONENT_INFO_KEYS
                ),
                shortcut["reward"].item(),
                places=6,
            )
        finally:
            environment.close()

    def test_repaired_observation_contract_is_strictly_image_and_agent_owned(self):
        config = Config(
            reward_supervised=True,
            reward_contract="shin_normalized_repaired",
            policy_observation_contract="shin_068_repaired",
        )
        self.assertEqual(config.observation_channels, 3)
        self.assertEqual(config.context_features, 3)
        self.assertTrue(config.needs_target_distance)
        with self.assertRaisesRegex(ValueError, "cannot include"):
            Config(
                reward_supervised=True,
                reward_contract="shin_normalized_repaired",
                policy_observation_contract="shin_068_repaired",
                observe_segmentation=True,
            )
        with self.assertRaisesRegex(ValueError, "policy_observation_contract"):
            Config(policy_observation_contract="mystery_state")
        with self.assertRaisesRegex(ValueError, "clean policy inputs"):
            Config(policy_observation_contract="shin_068_repaired")

    def test_guarded_shin_makes_low_coverage_endpoint_a_failure(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shape = (20, 20, 20)
        start = (10, 10, 6)
        end = (10, 10, 8)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[10, 10, 4:17] = 1

        def run(contract: str):
            environment = SmallBowelEnv(
                config=Config(
                    device=str(device),
                    patch_size_mm=8,
                    voxel_size_mm=1.0,
                    max_step_displacement_mm=2,
                    cumulative_path_radius_mm=0,
                    max_episode_steps=8,
                    reward_supervised=True,
                    reward_contract=contract,
                    success_coverage_threshold=0.9,
                    terminate_on_success=True,
                ),
                dataset_iterator=iter(
                    [self._make_subject(shape, start, end, segmentation)]
                ),
                num_episodes_per_sample=1,
                device=device,
            )
            try:
                environment._reset()
                return environment._step(
                    TensorDict(
                        {
                            "action": torch.tensor(
                                [[0.5, 0.5, 1.0]],
                                device=device,
                            )
                        },
                        batch_size=torch.Size([1]),
                        device=device,
                    )
                )
            finally:
                environment.close()

        literal = run("shin_normalized")
        guarded = run("shin_normalized_guarded")
        self.assertTrue(literal["terminated"].item())
        self.assertTrue(guarded["terminated"].item())
        self.assertEqual(literal["info", "final_success"].item(), 0.0)
        self.assertEqual(guarded["info", "final_success"].item(), 0.0)
        self.assertGreater(literal["info", "reward_terminal"].item(), 0.0)
        self.assertLess(guarded["info", "reward_terminal"].item(), 0.0)
        self.assertAlmostEqual(
            guarded["info", "reward_step"].item(),
            -(1.0 - 0.999) * (100.0 / 6.0),
            places=6,
        )
        self.assertGreater(literal["reward"].item(), 0.0)
        self.assertLess(guarded["reward"].item(), 0.0)

    def test_shin_contract_configuration_is_explicitly_supervised(self):
        with self.assertRaisesRegex(ValueError, "reward_contract"):
            Config(reward_contract="paperish")
        with self.assertRaisesRegex(ValueError, "GT segmentation"):
            Config(
                annotation_free=True,
                reward_contract="shin_normalized",
                use_immediate_gdt_reward=False,
                terminate_on_success=False,
                coverage_reward_scale=0,
                gdt_reward_scale=0,
                r_final=0,
                r_val1=0,
            )

    def test_max_step_gdt_normalization_is_subject_length_invariant(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=8,
            reward_supervised=True,
            coverage_reward_scale=0.0,
            gdt_reward_scale=0.1,
            gdt_progress_normalization="max_step",
            target_recovery_reward_scale=0.0,
            target_distance_penalty_scale=0.0,
            gate_positive_shaping_on_target_segment=True,
            r_val1=0.0,
            wall_penalty_scale=0.0,
            step_penalty=0.0,
            episodic_cell_reward_scale=0.0,
            r_final=0.0,
            terminate_on_success=False,
        )
        shape = (20, 20, 20)
        start = (8, 8, 8)
        segmentation = np.ones(shape, dtype=np.uint8)
        environments = [
            SmallBowelEnv(
                config=config,
                dataset_iterator=iter(
                    [self._make_subject(shape, start, end, segmentation)]
                ),
                num_episodes_per_sample=1,
                device=device,
            )
            for end in ((8, 8, 12), (8, 8, 16))
        ]
        try:
            rewards = []
            for environment in environments:
                environment._reset()
                transition = environment._step(
                    TensorDict(
                        {
                            "action": torch.tensor(
                                [[0.5, 0.5, 1.0]],
                                device=device,
                            )
                        },
                        batch_size=torch.Size([1]),
                        device=device,
                    )
                )
                rewards.append(transition["reward"])
            torch.testing.assert_close(rewards[0], rewards[1])
            self.assertAlmostEqual(
                rewards[0].item(),
                0.1 * 2.0 / np.sqrt(12.0),
                places=6,
            )
        finally:
            for environment in environments:
                environment.close()

    def test_recovery_target_excludes_disconnected_label_islands(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            max_episode_steps=4,
            reward_supervised=True,
            target_recovery_reward_scale=1.0,
            coverage_reward_scale=0.0,
            terminate_on_success=False,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        disconnected_island = (8, 8, 10)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[start] = 1
        segmentation[disconnected_island] = 1
        subject = self._make_subject(shape, start, start, segmentation)
        subject["gdt_start"].fill(np.inf)
        subject["gdt_end"].fill(np.inf)
        subject["gdt_start"][start] = 0.0
        subject["gdt_end"][start] = 0.0
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([subject]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            self.assertGreater(
                environment.target_distance_map[disconnected_island],
                0.0,
            )
            transition = environment._step(
                TensorDict(
                    {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(environment.current_pos_vox, disconnected_island)
            self.assertLess(transition["reward"].item(), 0.0)
        finally:
            environment.close()

    def test_repeat_loader_restarts_without_itertools_cycle(self):
        iterator = repeat_loader([1, 2, 3])
        self.assertEqual([next(iterator) for _ in range(8)], [1, 2, 3, 1, 2, 3, 1, 2])

    def test_gdt_reward_scale_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "gdt_reward_scale"):
            Config(gdt_reward_scale=-1)

    def test_target_recovery_reward_scale_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "target_recovery_reward_scale"):
            Config(target_recovery_reward_scale=-1)

    def test_calibrated_distance_reward_configuration_must_be_valid(self):
        with self.assertRaisesRegex(ValueError, "gdt_progress_normalization"):
            Config(gdt_progress_normalization="subject_magic")
        with self.assertRaisesRegex(ValueError, "target_distance_penalty_scale"):
            Config(target_distance_penalty_scale=-1)
        with self.assertRaisesRegex(ValueError, "target_distance_penalty_radius_mm"):
            Config(target_distance_penalty_radius_mm=0)
        with self.assertRaisesRegex(ValueError, "terminal_success_bonus"):
            Config(terminal_success_bonus=-1)
        with self.assertRaisesRegex(ValueError, "terminal_failure_penalty"):
            Config(terminal_failure_penalty=-2)
        with self.assertRaisesRegex(ValueError, "lr_anneal_timesteps"):
            Config(lr_anneal_timesteps=-1)
        with self.assertRaisesRegex(ValueError, "target_kl"):
            Config(target_kl=-0.01)
        with self.assertRaisesRegex(ValueError, "gamma"):
            Config(gamma=0)
        with self.assertRaisesRegex(ValueError, "gamma"):
            Config(gamma=1.01)

    def test_episodic_cell_reward_configuration_must_be_valid(self):
        with self.assertRaisesRegex(ValueError, "episodic_cell_reward_scale"):
            Config(episodic_cell_reward_scale=-1)
        with self.assertRaisesRegex(ValueError, "episodic_cell_size_mm"):
            Config(episodic_cell_size_mm=0)

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

    def test_cached_path_geometry_matches_independent_physical_tube(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=2,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
        )
        shape = (20, 20, 20)
        start = (5, 5, 5)
        end = (5, 5, 15)
        segmentation = np.ones(shape, dtype=np.uint8)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            action = TensorDict(
                {"action": torch.tensor([[0.5, 0.5, 1.0]], device=device)},
                batch_size=torch.Size([1]),
                device=device,
            )
            environment._step(action)
            cache_size = len(environment._dilated_line_offsets)
            environment._step(action)
            self.assertEqual(len(environment._dilated_line_offsets), cache_size)
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
            self.assertEqual(
                environment.path_voxels,
                int(environment.cumulative_path_mask.sum().item()),
            )
            self.assertEqual(
                environment.path_target_intersection,
                environment.path_voxels,
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

    def test_reward_supervised_policy_can_opt_in_to_segmentation_channel(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=2,
            cumulative_path_radius_mm=1,
            reward_supervised=True,
            observe_segmentation=True,
        )
        shape = (16, 16, 16)
        start = (8, 8, 8)
        end = (8, 8, 12)
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[8, 8, 4:13] = 1
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter(
                [self._make_subject(shape, start, end, segmentation)]
            ),
            num_episodes_per_sample=1,
            device=device,
        )
        try:
            initial = environment._reset()
            self.assertEqual(config.observation_channels, 7)
            expected = get_patch(
                environment.seg,
                environment.current_pos_vox,
                config.patch_size_vox,
            ).to(environment.dtype)
            torch.testing.assert_close(initial["actor"][0, -1], expected)
        finally:
            environment.close()

        with self.assertRaisesRegex(ValueError, "observe_segmentation"):
            Config(
                annotation_free=True,
                observe_segmentation=True,
                use_immediate_gdt_reward=False,
                terminate_on_success=False,
                coverage_reward_scale=0,
                gdt_reward_scale=0,
                r_final=0,
                r_val1=0,
            )

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

    def test_outward_action_projects_to_valid_tangent(self):
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
                    {"action": torch.tensor([[1.0, 0.5, 0.5]], device=device)},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertNotEqual(environment.current_pos_vox, start)
            self.assertEqual(environment.current_pos_vox[:2], start[:2])
            self.assertFalse(transition["done"].item())
        finally:
            environment.close()

    def test_factorized_zero_action_remains_zero_and_is_penalized(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=4,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
            reward_supervised=True,
            memory_model="gru",
            action_distribution="factorized_categorical",
            deterministic_action_statistic="mode",
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 12)
        segmentation = np.ones(shape, dtype=np.uint8)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            center = config.max_step_vox
            transition = environment._step(
                TensorDict(
                    {
                        "action": torch.full(
                            (1, 3),
                            center,
                            dtype=torch.long,
                            device=device,
                        )
                    },
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(environment.current_pos_vox, start)
            self.assertAlmostEqual(
                transition["reward"].item(),
                -config.r_zero_mov,
                places=6,
            )
            self.assertAlmostEqual(
                transition["info", "reward_invalid"].item(),
                -config.r_zero_mov,
                places=6,
            )
            self.assertFalse(transition["done"].item())
        finally:
            environment.close()

    def test_clean_policy_outward_boundary_action_is_not_reflected(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=4,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
            reward_supervised=True,
            memory_model="gru",
            action_distribution="factorized_categorical",
            deterministic_action_statistic="mode",
        )
        shape = (16, 16, 16)
        start = (15, 15, 15)
        end = (4, 4, 4)
        segmentation = np.ones(shape, dtype=np.uint8)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            outward = torch.full(
                (1, 3),
                2 * config.max_step_vox,
                dtype=torch.long,
                device=device,
            )
            transition = environment._step(
                TensorDict(
                    {"action": outward},
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(environment.current_pos_vox, start)
            self.assertAlmostEqual(
                transition["info", "reward_invalid"].item(),
                -config.r_zero_mov,
                places=6,
            )
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

    def test_categorical_action_executes_the_selected_integer_displacement(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=4,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
            reward_supervised=True,
            memory_model="gru",
            action_distribution="categorical",
            deterministic_action_statistic="mode",
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 12)
        segmentation = np.ones(shape, dtype=np.uint8)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            selected = (1, -2, 4)
            action_index = config.action_displacements.index(selected)
            transition = environment._step(
                TensorDict(
                    {
                        "action": torch.tensor(
                            [action_index],
                            dtype=torch.long,
                            device=device,
                        )
                    },
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(
                environment.current_pos_vox,
                tuple(
                    coordinate + delta
                    for coordinate, delta in zip(start, selected)
                ),
            )
            self.assertFalse(transition["done"].item())
        finally:
            environment.close()

    def test_factorized_categorical_action_executes_exact_integer_displacement(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = Config(
            device=str(device),
            patch_size_mm=8,
            voxel_size_mm=1.0,
            max_step_displacement_mm=4,
            cumulative_path_radius_mm=1,
            allowed_area_radius_mm=0,
            max_episode_steps=8,
            reward_supervised=True,
            memory_model="gru",
            action_distribution="factorized_categorical",
            deterministic_action_statistic="mode",
        )
        shape = (16, 16, 16)
        start = (4, 4, 4)
        end = (4, 4, 12)
        segmentation = np.ones(shape, dtype=np.uint8)
        environment = SmallBowelEnv(
            config=config,
            dataset_iterator=iter([self._make_subject(shape, start, end, segmentation)]),
            num_episodes_per_sample=1,
            device=device,
        )

        try:
            environment._reset()
            selected = (1, -2, 4)
            axis_indices = [
                delta + config.max_step_vox for delta in selected
            ]
            transition = environment._step(
                TensorDict(
                    {
                        "action": torch.tensor(
                            [axis_indices],
                            dtype=torch.long,
                            device=device,
                        )
                    },
                    batch_size=torch.Size([1]),
                    device=device,
                )
            )
            self.assertEqual(
                environment.current_pos_vox,
                tuple(
                    coordinate + delta
                    for coordinate, delta in zip(start, selected)
                ),
            )
            self.assertFalse(transition["done"].item())
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
