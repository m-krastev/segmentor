#!/usr/bin/env python3
"""Numerically audit canonical Navigator reward trajectories.

This is deliberately independent of PPO. It answers whether the scalar reward
itself makes obvious failure modes profitable before GPU time is spent.
"""

from __future__ import annotations

import argparse
from math import sqrt

from navigator.rewards import (
    gdt_progress_reward,
    target_distance_state_penalty,
    target_recovery_potential_reward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-step-mm", type=float, default=6.0 * sqrt(3.0))
    parser.add_argument("--gdt-scale", type=float, default=0.1)
    parser.add_argument("--recovery-scale", type=float, default=0.05)
    parser.add_argument("--distance-scale", type=float, default=0.1)
    parser.add_argument("--distance-radius-mm", type=float, default=600.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--episodic-scale", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def gdt(delta_mm: float) -> float:
        return gdt_progress_reward(
            delta_mm,
            args.max_step_mm,
            args.gdt_scale,
        )

    def recovery(previous_mm: float, next_mm: float) -> float:
        return target_recovery_potential_reward(
            previous_mm,
            next_mm,
            args.max_step_mm,
            args.recovery_scale,
        )

    def distance(max_segment_mm: float) -> float:
        return target_distance_state_penalty(
            max_segment_mm,
            args.distance_radius_mm,
            args.distance_scale,
        )

    step = -args.step_penalty
    first_novel = args.episodic_scale
    scenarios = {
        "inside_forward_6mm_new": gdt(6.0) + step + first_novel,
        "inside_backward_6mm_revisit": gdt(-6.0) + step,
        "leave_1.5mm_gated": recovery(0.0, 1.5) + distance(1.5) + step,
        "return_1.5mm_revisit": recovery(1.5, 0.0) + distance(1.5) + step,
        "leave_6mm_gated": recovery(0.0, 6.0) + distance(6.0) + step,
        "return_6mm_revisit": recovery(6.0, 0.0) + distance(6.0) + step,
        "outside_tangent_6mm": distance(6.0) + step,
        "outside_tangent_30mm": distance(30.0) + step,
        "outside_tangent_600mm": distance(600.0) + step,
        # A background-crossing shortcut receives no positive GDT, coverage,
        # or episodic reward. The gap is represented by its segment maximum.
        "cross_loop_1.5mm_gap": distance(1.5) + step,
    }
    cycles = {
        "inside_forward_backward": (
            scenarios["inside_forward_6mm_new"]
            + scenarios["inside_backward_6mm_revisit"]
        ),
        "outside_1.5mm_leave_return": (
            scenarios["leave_1.5mm_gated"]
            + scenarios["return_1.5mm_revisit"]
        ),
        "outside_6mm_leave_return": (
            scenarios["leave_6mm_gated"]
            + scenarios["return_6mm_revisit"]
        ),
    }

    print("scenario\treturn")
    for name, value in scenarios.items():
        print(f"{name}\t{value:+.6f}")
    for name, value in cycles.items():
        print(f"{name}\t{value:+.6f}")

    if scenarios["inside_forward_6mm_new"] <= 0:
        raise SystemExit("Contract failed: valid forward progress is not positive")
    if scenarios["cross_loop_1.5mm_gap"] >= 0:
        raise SystemExit("Contract failed: cross-loop shortcut is profitable")
    profitable_cycles = {
        name: value for name, value in cycles.items() if value >= 0
    }
    if profitable_cycles:
        raise SystemExit(f"Contract failed: profitable cycles {profitable_cycles}")

    maximum_intrinsic_return = args.episodic_scale * sum(
        1.0 / sqrt(discovery)
        for discovery in range(1, args.horizon + 1)
    )
    print(
        "episode_bound\tmaximum_intrinsic_return\t"
        f"{maximum_intrinsic_return:+.6f}"
    )
    print(
        "episode_bound\tfixed_step_cost\t"
        f"{-args.step_penalty * args.horizon:+.6f}"
    )


if __name__ == "__main__":
    main()
