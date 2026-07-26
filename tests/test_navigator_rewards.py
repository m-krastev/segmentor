import unittest

from navigator.rewards import (
    coverage_potential_reward,
    gdt_progress_reward,
    is_path_success,
    terminal_path_reward,
)


class TerminalPathRewardTests(unittest.TestCase):
    def test_failed_episode_is_strictly_penalized(self):
        for coverage in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertEqual(
                terminal_path_reward(coverage, False, 0.55, 20.0, 21.0),
                -21.0,
            )

    def test_endpoint_without_coverage_is_not_success(self):
        self.assertFalse(is_path_success(True, 0.05, 0.55))
        self.assertLess(terminal_path_reward(0.05, True, 0.55, 20.0, 21.0), 0.0)

    def test_success_requires_endpoint_and_coverage(self):
        self.assertTrue(is_path_success(True, 0.55, 0.55))
        self.assertFalse(is_path_success(False, 1.0, 0.55))
        self.assertEqual(terminal_path_reward(0.75, True, 0.55, 20.0, 21.0), 15.0)


class CoveragePotentialRewardTests(unittest.TestCase):
    def test_revisits_cannot_create_coverage_reward(self):
        self.assertEqual(coverage_potential_reward(0.4, 0.4, 20.0), 0.0)

    def test_rewards_telescope_to_final_dice_change(self):
        coverages = (0.1, 0.2, 0.15, 0.8)
        total = sum(
            coverage_potential_reward(before, after, 20.0)
            for before, after in zip(coverages, coverages[1:])
        )
        self.assertAlmostEqual(total, 20.0 * (coverages[-1] - coverages[0]))

    def test_damaging_dice_is_penalized(self):
        self.assertLess(coverage_potential_reward(0.4, 0.3, 20.0), 0.0)


class GdtProgressRewardTests(unittest.TestCase):
    def test_reward_is_proportional_without_step_bonus(self):
        self.assertEqual(gdt_progress_reward(2.0, 10.0, 6.0), 1.2)
        self.assertEqual(gdt_progress_reward(0.0, 10.0, 6.0), 0.0)

    def test_moving_away_from_goal_is_penalized(self):
        self.assertEqual(gdt_progress_reward(-2.0, 10.0, 6.0), -1.2)

    def test_round_trip_cannot_create_progress_reward(self):
        forward = gdt_progress_reward(2.0, 10.0, 6.0)
        backward = gdt_progress_reward(-2.0, 10.0, 6.0)
        self.assertAlmostEqual(forward + backward, 0.0)

    def test_implausible_jump_is_penalized(self):
        self.assertEqual(gdt_progress_reward(11.0, 10.0, 6.0), -6.0)


if __name__ == "__main__":
    unittest.main()
