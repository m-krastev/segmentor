from argparse import Namespace
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_navigator_hessian_energy_beam import plan_energy_beam  # noqa: E402


def planner_args(**overrides) -> Namespace:
    values = {
        "voxel_size_mm": 1.5,
        "hessian_scale_mm": 6.0,
        "hessian_strength_percentile": 95.0,
        "hessian_polarity": "unsigned",
        "step_mm": 4.5,
        "maximum_steps": 20,
        "beam_width": 32,
        "maximum_turn_deg": 60.0,
        "axis_weight": 2.0,
        "curvature_weight": 0.5,
        "strength_weight": 0.25,
        "polarity_weight": 0.0,
        "polarity_gate_power": 0.0,
        "ct_weight": 0.0,
        "dark_weight": 2.0,
        "seed_ct_weight": 0.0,
        "boundary_weight": 0.0,
        "tube_balance_weight": 0.25,
        "revisit_penalty": 1.0,
        "step_cost": 1.75,
        "recent_revisit_window": 0,
        "seed_ct_bandwidth": 0.2,
    }
    values.update(overrides)
    return Namespace(**values)


def test_energy_beam_traverses_and_stops_at_finite_bright_tube() -> None:
    shape = (36, 32, 32)
    _, y, z = np.indices(shape)
    image = np.full(shape, -100.0, dtype=np.float32)
    image[(y - 16) ** 2 + (z - 16) ** 2 <= 16] = 180.0
    dark_filter = np.zeros(shape, dtype=np.float32)
    start = np.asarray([4, 16, 16])
    args = planner_args(hessian_polarity="bright", polarity_weight=0.25)

    branches = [
        plan_energy_beam(
            image,
            dark_filter,
            start,
            args,
            initial_axis_sign=axis_sign,
        )
        for axis_sign in (-1, 1)
    ]
    selected = max(branches, key=lambda branch: branch["final_score"])
    endpoint = selected["rounded_history"][-1]

    assert selected["steps"] >= 8
    assert endpoint[0] >= 28
    assert abs(endpoint[1] - 16) <= 1
    assert abs(endpoint[2] - 16) <= 1
    assert selected["unique_voxels"] == selected["steps"] + 1
    assert selected["stop_reason"] == "energy_stop"
