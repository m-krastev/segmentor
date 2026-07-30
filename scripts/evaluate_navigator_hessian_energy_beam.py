#!/usr/bin/env python3
"""Evaluate a label-free Hessian/filter beam tracker from one start seed."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from navigator.metrics import compute_path_metrics
from evaluate_navigator_hessian_streamline import compute_hessian_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--voxel-size-mm", type=float, default=1.5)
    parser.add_argument("--hessian-scale-mm", type=float, default=6.0)
    parser.add_argument("--hessian-strength-percentile", type=float, default=95.0)
    parser.add_argument(
        "--hessian-polarity",
        choices=("bright", "dark", "unsigned"),
        default="unsigned",
    )
    parser.add_argument("--step-mm", type=float, default=4.5)
    parser.add_argument("--maximum-steps", type=int, default=2048)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--maximum-turn-deg", type=float, default=60.0)
    parser.add_argument("--axis-weight", type=float, default=2.0)
    parser.add_argument("--curvature-weight", type=float, default=0.5)
    parser.add_argument("--strength-weight", type=float, default=0.25)
    parser.add_argument("--polarity-weight", type=float, default=0.0)
    parser.add_argument("--polarity-gate-power", type=float, default=0.0)
    parser.add_argument("--ct-weight", type=float, default=0.0)
    parser.add_argument("--dark-weight", type=float, default=2.0)
    parser.add_argument("--seed-ct-weight", type=float, default=0.0)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--tube-balance-weight", type=float, default=0.25)
    parser.add_argument("--revisit-penalty", type=float, default=1.0)
    parser.add_argument("--step-cost", type=float, default=1.75)
    parser.add_argument(
        "--recent-revisit-window",
        type=int,
        default=0,
        help="Visited-path memory in voxels; 0 keeps the full episode",
    )
    parser.add_argument(
        "--initial-axis-sign",
        type=int,
        choices=(-1, 0, 1),
        default=0,
        help="Constrain the seed-axis branch; 0 searches both orientations",
    )
    parser.add_argument("--seed-ct-bandwidth", type=float, default=0.2)
    parser.add_argument("--path-radius-mm", type=float, default=6.0)
    parser.add_argument("--endpoint-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--success-dice", type=float, default=0.4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--path-output-dir", type=Path)
    return parser.parse_args()


def action_directions() -> np.ndarray:
    directions = np.asarray(
        [
            direction
            for direction in product((-1.0, 0.0, 1.0), repeat=3)
            if any(direction)
        ],
        dtype=np.float64,
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def sample_volume(volume: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return map_coordinates(
        volume,
        np.asarray(positions, dtype=np.float64).T,
        order=1,
        mode="nearest",
        prefilter=False,
    )


def tube_axes_batch(
    components: dict[str, np.ndarray],
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = {
        name: sample_volume(volume, positions)
        for name, volume in components.items()
    }
    hessian = np.zeros((len(positions), 3, 3), dtype=np.float64)
    hessian[:, 0, 0] = values["xx"]
    hessian[:, 1, 1] = values["yy"]
    hessian[:, 2, 2] = values["zz"]
    hessian[:, 0, 1] = hessian[:, 1, 0] = values["xy"]
    hessian[:, 0, 2] = hessian[:, 2, 0] = values["xz"]
    hessian[:, 1, 2] = hessian[:, 2, 1] = values["yz"]

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    absolute_order = np.argsort(np.abs(eigenvalues), axis=1)
    batch_indices = np.arange(len(positions))
    axes = eigenvectors[batch_indices, :, absolute_order[:, 0]]
    absolute_values = np.take_along_axis(
        np.abs(eigenvalues),
        absolute_order,
        axis=1,
    )
    confidence = (
        (absolute_values[:, 1] - absolute_values[:, 0])
        / np.maximum(absolute_values[:, 1], np.finfo(np.float64).eps)
    )
    transverse_balance = (
        absolute_values[:, 1]
        / np.maximum(absolute_values[:, 2], np.finfo(np.float64).eps)
    )
    transverse_strength = absolute_values[:, 1]
    sorted_eigenvalues = np.take_along_axis(
        eigenvalues,
        absolute_order,
        axis=1,
    )
    transverse_eigenvalues = sorted_eigenvalues[:, 1:]
    bright_support = np.mean(transverse_eigenvalues < 0.0, axis=1)
    return (
        axes,
        confidence,
        transverse_balance,
        transverse_strength,
        bright_support,
    )


def robust_hessian_strength_scale(
    components: dict[str, np.ndarray],
    percentile: float,
    maximum_samples: int = 200_000,
) -> float:
    """Estimate a subject-specific transverse-curvature scale without labels."""

    voxel_count = components["xx"].size
    sample_count = min(voxel_count, maximum_samples)
    flat_indices = np.linspace(
        0,
        voxel_count - 1,
        sample_count,
        dtype=np.int64,
    )
    hessian = np.zeros((sample_count, 3, 3), dtype=np.float64)
    hessian[:, 0, 0] = components["xx"].reshape(-1)[flat_indices]
    hessian[:, 1, 1] = components["yy"].reshape(-1)[flat_indices]
    hessian[:, 2, 2] = components["zz"].reshape(-1)[flat_indices]
    hessian[:, 0, 1] = hessian[:, 1, 0] = components["xy"].reshape(-1)[
        flat_indices
    ]
    hessian[:, 0, 2] = hessian[:, 2, 0] = components["xz"].reshape(-1)[
        flat_indices
    ]
    hessian[:, 1, 2] = hessian[:, 2, 1] = components["yz"].reshape(-1)[
        flat_indices
    ]
    absolute_eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(hessian)), axis=1)
    scale = float(np.percentile(absolute_eigenvalues[:, 1], percentile))
    return max(scale, np.finfo(np.float32).eps)


def boundary_clearance(positions: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    upper = np.asarray(shape, dtype=np.float64) - 1.0 - positions
    return np.minimum(positions, upper).min(axis=1)


def plan_energy_beam(
    image: np.ndarray,
    dark_filter: np.ndarray,
    start: np.ndarray,
    args: argparse.Namespace,
    *,
    initial_axis_sign: int | None = None,
) -> dict:
    clipped_ct = np.clip(image, -120.0, 180.0)
    clipped_ct = ((clipped_ct + 120.0) / 300.0).astype(np.float32)
    components = compute_hessian_components(
        clipped_ct,
        args.hessian_scale_mm / args.voxel_size_mm,
    )
    hessian_strength_scale = robust_hessian_strength_scale(
        components,
        args.hessian_strength_percentile,
    )
    directions = action_directions()
    step_vox = args.step_mm / args.voxel_size_mm
    minimum_turn_cosine = float(np.cos(np.deg2rad(args.maximum_turn_deg)))
    seed_ct = float(clipped_ct[tuple(np.asarray(start, dtype=int))])

    positions = np.asarray(start, dtype=np.float64).reshape(1, 3)
    previous_directions = np.zeros((1, 3), dtype=np.float64)
    scores = np.zeros(1, dtype=np.float64)
    start_voxel = tuple(int(value) for value in np.rint(positions[0]))
    path_histories: list[frozenset[tuple[int, int, int]] | tuple] = [
        frozenset((start_voxel,))
    ]
    if args.recent_revisit_window:
        path_histories = [(start_voxel,)]
    position_layers = [positions.copy()]
    parent_layers: list[np.ndarray] = []
    component_names = (
        "axis",
        "curvature",
        "strength",
        "polarity",
        "ct",
        "dark",
        "seed_ct",
        "boundary",
        "tube_balance",
        "revisit",
        "step_cost",
    )
    component_layers: list[dict[str, np.ndarray]] = []
    best_depth = 0
    best_index = 0
    best_score = 0.0
    stop_reason = "maximum_steps"

    for depth in range(args.maximum_steps):
        (
            axes,
            confidence,
            transverse_balance,
            transverse_strength,
            bright_support,
        ) = tube_axes_batch(components, positions)
        normalized_strength = np.clip(
            transverse_strength / hessian_strength_scale,
            0.0,
            1.0,
        )
        if args.hessian_polarity == "bright":
            polarity_support = bright_support
        elif args.hessian_polarity == "dark":
            polarity_support = 1.0 - bright_support
        else:
            polarity_support = np.full_like(bright_support, 0.5)
        if args.hessian_polarity == "unsigned" or args.polarity_gate_power == 0:
            structural_strength = normalized_strength
        else:
            structural_strength = (
                normalized_strength
                * polarity_support**args.polarity_gate_power
            )
        expanded = []
        for state_index in range(len(positions)):
            previous = previous_directions[state_index]
            first_step = depth == 0
            axis = axes[state_index]
            if not first_step and np.dot(axis, previous) < 0:
                axis = -axis
            candidate_turn_cosine = directions @ previous
            if first_step:
                axis_alignment = directions @ axis
                if initial_axis_sign is None:
                    allowed = np.ones(len(directions), dtype=bool)
                    alignment = np.abs(axis_alignment)
                else:
                    alignment = initial_axis_sign * axis_alignment
                    allowed = alignment >= minimum_turn_cosine
                curvature = np.zeros(len(directions), dtype=np.float64)
            else:
                allowed = candidate_turn_cosine >= minimum_turn_cosine
                alignment = directions @ axis
                curvature = candidate_turn_cosine
            candidate_positions = (
                positions[state_index] + step_vox * directions[allowed]
            )
            candidate_directions = directions[allowed]
            candidate_alignment = alignment[allowed]
            candidate_curvature = curvature[allowed]
            clearance = boundary_clearance(
                candidate_positions,
                clipped_ct.shape,
            )
            inside = clearance >= 1.0
            for local_index in np.flatnonzero(inside):
                expanded.append(
                    (
                        state_index,
                        candidate_positions[local_index],
                        candidate_directions[local_index],
                        candidate_alignment[local_index],
                        candidate_curvature[local_index],
                        clearance[local_index],
                        confidence[state_index],
                        transverse_balance[state_index],
                        structural_strength[state_index],
                        polarity_support[state_index],
                    )
                )
        if not expanded:
            stop_reason = "no_feasible_expansion"
            break

        parent_indices = np.asarray([item[0] for item in expanded], dtype=int)
        candidate_positions = np.asarray([item[1] for item in expanded])
        candidate_directions = np.asarray([item[2] for item in expanded])
        alignment = np.asarray([item[3] for item in expanded])
        curvature = np.asarray([item[4] for item in expanded])
        clearance = np.asarray([item[5] for item in expanded])
        confidence = np.asarray([item[6] for item in expanded])
        transverse_balance = np.asarray([item[7] for item in expanded])
        structural_strength = np.asarray([item[8] for item in expanded])
        polarity_support = np.asarray([item[9] for item in expanded])

        dark = np.clip(sample_volume(dark_filter, candidate_positions), 0.0, 1.0)
        candidate_ct = np.clip(
            sample_volume(clipped_ct, candidate_positions),
            0.0,
            1.0,
        )
        seed_similarity = np.exp(
            -np.abs(candidate_ct - seed_ct) / args.seed_ct_bandwidth
        )
        rounded = np.rint(candidate_positions).astype(int)
        segment_sample_count = max(int(np.ceil(step_vox * 2.0)), 1)
        segment_fractions = np.linspace(
            1.0 / segment_sample_count,
            1.0,
            segment_sample_count,
        )
        parent_positions = positions[parent_indices]
        segment_samples = (
            parent_positions[:, None, :]
            + segment_fractions[None, :, None]
            * (candidate_positions - parent_positions)[:, None, :]
        )
        segment_voxels = np.rint(segment_samples).astype(int)
        candidate_path_voxels = []
        for parent_position, samples in zip(parent_positions, segment_voxels):
            parent_voxel = tuple(int(value) for value in np.rint(parent_position))
            path_voxels = tuple(
                dict.fromkeys(
                    tuple(int(value) for value in sample)
                    for sample in samples
                    if tuple(int(value) for value in sample) != parent_voxel
                )
            )
            candidate_path_voxels.append(path_voxels)
        revisited = np.asarray(
            [
                any(
                    path_voxel in path_histories[parent_index]
                    for path_voxel in path_voxels
                )
                for path_voxels, parent_index in zip(
                    candidate_path_voxels,
                    parent_indices,
                )
            ],
            dtype=np.float64,
        )

        components_now = {
            "axis": (
                args.axis_weight
                * confidence
                * structural_strength
                * alignment
            ),
            "curvature": args.curvature_weight * curvature,
            "strength": args.strength_weight * structural_strength,
            "polarity": args.polarity_weight * (2.0 * polarity_support - 1.0),
            "ct": args.ct_weight * candidate_ct,
            "dark": -args.dark_weight * dark,
            "seed_ct": args.seed_ct_weight * seed_similarity,
            "boundary": args.boundary_weight * np.clip(clearance / 10.0, 0.0, 1.0),
            "tube_balance": (
                args.tube_balance_weight
                * transverse_balance
                * structural_strength
            ),
            "revisit": -args.revisit_penalty * revisited,
            "step_cost": np.full(
                len(candidate_positions),
                -args.step_cost,
                dtype=np.float64,
            ),
        }
        increments = sum(components_now.values())
        total_scores = scores[parent_indices] + increments

        # Keep only the strongest state per rounded voxel before global beam
        # pruning. This avoids wasting the beam on numerically equivalent moves.
        best_by_voxel: dict[tuple[int, int, int], int] = {}
        for index, voxel in enumerate(rounded):
            key = tuple(int(value) for value in voxel)
            previous_best = best_by_voxel.get(key)
            if (
                previous_best is None
                or total_scores[index] > total_scores[previous_best]
            ):
                best_by_voxel[key] = index
        unique_indices = np.fromiter(best_by_voxel.values(), dtype=int)
        order = unique_indices[
            np.argsort(total_scores[unique_indices])[::-1][: args.beam_width]
        ]

        positions = candidate_positions[order]
        previous_directions = candidate_directions[order]
        scores = total_scores[order]
        selected_parents = parent_indices[order]
        if args.recent_revisit_window:
            path_histories = [
                (
                    path_histories[parent_index]
                    + candidate_path_voxels[candidate_index]
                )[-args.recent_revisit_window :]
                for candidate_index, parent_index in zip(order, selected_parents)
            ]
        else:
            path_histories = [
                path_histories[parent_index].union(
                    candidate_path_voxels[candidate_index]
                )
                for candidate_index, parent_index in zip(order, selected_parents)
            ]
        parent_layers.append(selected_parents)
        component_layers.append(
            {
                name: np.asarray(values[order], dtype=np.float64)
                for name, values in components_now.items()
            }
        )
        position_layers.append(positions.copy())
        layer_best_index = int(np.argmax(scores))
        if scores[layer_best_index] > best_score:
            best_depth = len(parent_layers)
            best_index = layer_best_index
            best_score = float(scores[layer_best_index])

    if best_depth < len(parent_layers):
        stop_reason = "energy_stop"
    final_index = best_index
    reverse_path = [position_layers[best_depth][final_index]]
    reverse_components: list[dict[str, float]] = []
    for layer_index in range(best_depth - 1, -1, -1):
        reverse_components.append(
            {
                name: float(values[final_index])
                for name, values in component_layers[layer_index].items()
            }
        )
        final_index = int(parent_layers[layer_index][final_index])
        reverse_path.append(position_layers[layer_index][final_index])
    points = np.asarray(reverse_path[::-1], dtype=np.float64)
    selected_components = reverse_components[::-1]
    selected_step_rewards = np.asarray(
        [sum(components.values()) for components in selected_components],
        dtype=np.float64,
    )
    rounded_history = np.rint(points).astype(np.int64)
    if len(selected_step_rewards):
        quarter_length = max(len(selected_step_rewards) // 4, 1)
        component_means = {
            name: float(
                np.mean([components[name] for components in selected_components])
            )
            for name in component_names
        }
        reward_summary = {
            "minimum": float(np.min(selected_step_rewards)),
            "q25": float(np.quantile(selected_step_rewards, 0.25)),
            "median": float(np.median(selected_step_rewards)),
            "q75": float(np.quantile(selected_step_rewards, 0.75)),
            "maximum": float(np.max(selected_step_rewards)),
            "first_quarter_mean": float(
                np.mean(selected_step_rewards[:quarter_length])
            ),
            "last_quarter_mean": float(
                np.mean(selected_step_rewards[-quarter_length:])
            ),
        }
        cumulative_rewards = np.cumsum(selected_step_rewards)
        prefix_steps = list(range(256, len(selected_step_rewards) + 1, 256))
        if not prefix_steps or prefix_steps[-1] != len(selected_step_rewards):
            prefix_steps.append(len(selected_step_rewards))
        energy_prefixes = [
            {
                "steps": step_count,
                "cumulative_score": float(cumulative_rewards[step_count - 1]),
                "mean_score": float(
                    cumulative_rewards[step_count - 1] / step_count
                ),
                "last_64_mean": float(
                    np.mean(
                        selected_step_rewards[max(0, step_count - 64) : step_count]
                    )
                ),
            }
            for step_count in prefix_steps
        ]
    else:
        component_means = {name: 0.0 for name in component_names}
        reward_summary = {
            "minimum": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "maximum": 0.0,
            "first_quarter_mean": 0.0,
            "last_quarter_mean": 0.0,
        }
        energy_prefixes = []
    return {
        "points": points,
        "rounded_history": rounded_history,
        "steps": len(points) - 1,
        "search_depth": len(parent_layers),
        "hessian_strength_scale": hessian_strength_scale,
        "unique_voxels": len({tuple(point) for point in rounded_history}),
        "final_score": best_score,
        "mean_score_per_step": float(best_score / max(len(points) - 1, 1)),
        "mean_selected_component_rewards": component_means,
        "selected_step_reward_summary": reward_summary,
        "selected_energy_prefixes": energy_prefixes,
        "beam_width_final": int(len(positions)),
        "stop_reason": stop_reason,
    }


def evaluate_case(args: argparse.Namespace, case_id: str) -> dict:
    case_dir = args.data_dir / case_id
    image = nib.load(case_dir / "ct.nii.gz").get_fdata(dtype=np.float32)
    dark_filter = nib.load(
        case_dir / "cache" / "navigation_filters-v1-mm-3-6-9.nii"
    ).get_fdata(dtype=np.float32)[..., 0]
    start_end = np.loadtxt(
        case_dir / "cache" / "start_end.npy",
        dtype=int,
    ).reshape(2, 3)
    start, endpoint = start_end
    if args.initial_axis_sign:
        plan = plan_energy_beam(
            image,
            dark_filter,
            start,
            args,
            initial_axis_sign=args.initial_axis_sign,
        )
        plan["selected_axis_branch"] = args.initial_axis_sign
        signed_branch_plans = [(args.initial_axis_sign, plan)]
    else:
        signed_branch_plans = [
            (
                axis_sign,
                plan_energy_beam(
                    image,
                    dark_filter,
                    start,
                    args,
                    initial_axis_sign=axis_sign,
                ),
            )
            for axis_sign in (-1, 1)
        ]
        selected_branch_index = int(
            np.argmax(
                [
                    branch["final_score"]
                    for _, branch in signed_branch_plans
                ]
            )
        )
        plan = signed_branch_plans[selected_branch_index][1]
        plan["selected_axis_branch"] = (-1, 1)[selected_branch_index]
        plan["axis_branch_summaries"] = [
            {
                "axis_sign": axis_sign,
                "steps": branch["steps"],
                "final_score": branch["final_score"],
                "mean_score_per_step": branch["mean_score_per_step"],
                "stop_reason": branch["stop_reason"],
            }
            for axis_sign, branch in signed_branch_plans
        ]

    target = (
        nib.load(case_dir / "segmentations" / "small_bowel.nii.gz").get_fdata(
            dtype=np.float32
        )
        > 0
    )
    metrics = compute_path_metrics(
        target,
        plan["rounded_history"],
        tuple(int(value) for value in endpoint),
        (args.voxel_size_mm,) * 3,
        args.path_radius_mm,
        args.endpoint_tolerance_mm,
        args.success_dice,
    )
    goal_array = np.asarray(endpoint, dtype=np.float64)
    spacing_array = np.full(3, args.voxel_size_mm, dtype=np.float64)
    endpoint_distances_mm = np.linalg.norm(
        (plan["points"] - goal_array) * spacing_array,
        axis=1,
    )
    closest_endpoint_step = int(np.argmin(endpoint_distances_mm))
    minimum_endpoint_distance_mm = float(
        endpoint_distances_mm[closest_endpoint_step]
    )
    branch_audit = []
    for axis_sign, branch in signed_branch_plans:
        branch_metrics = compute_path_metrics(
            target,
            branch["rounded_history"],
            tuple(int(value) for value in endpoint),
            (args.voxel_size_mm,) * 3,
            args.path_radius_mm,
            args.endpoint_tolerance_mm,
            args.success_dice,
        )
        branch_endpoint_distances_mm = np.linalg.norm(
            (branch["points"] - goal_array) * spacing_array,
            axis=1,
        )
        branch_audit.append(
            {
                "axis_sign": axis_sign,
                "selected_by_image_energy": (
                    axis_sign == plan["selected_axis_branch"]
                ),
                "dice": float(branch_metrics.dice),
                "endpoint_distance_mm": float(
                    branch_metrics.endpoint_distance_mm
                ),
                "minimum_endpoint_distance_mm": float(
                    np.min(branch_endpoint_distances_mm)
                ),
                "closest_endpoint_step": int(
                    np.argmin(branch_endpoint_distances_mm)
                ),
                "endpoint_reached": bool(branch_metrics.endpoint_reached),
                "traversal_success": bool(branch_metrics.traversal_success),
            }
        )
    if args.path_output_dir:
        args.path_output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            args.path_output_dir / f"{case_id}_history_float.npy",
            plan["points"],
        )
        np.save(
            args.path_output_dir / f"{case_id}_history_vox.npy",
            plan["rounded_history"],
        )
        for axis_sign, branch in signed_branch_plans:
            branch_name = "neg" if axis_sign < 0 else "pos"
            np.save(
                args.path_output_dir
                / f"{case_id}_history_float_{branch_name}.npy",
                branch["points"],
            )
            np.save(
                args.path_output_dir
                / f"{case_id}_history_vox_{branch_name}.npy",
                branch["rounded_history"],
            )
    return {
        "case_id": case_id,
        "start": start.tolist(),
        "endpoint_for_audit_only": endpoint.tolist(),
        **{
            key: value
            for key, value in plan.items()
            if key not in {"points", "rounded_history"}
        },
        "axis_branch_audit_after_planning": branch_audit,
        "dice": float(metrics.dice),
        "endpoint_distance_mm": float(metrics.endpoint_distance_mm),
        "minimum_endpoint_distance_mm": minimum_endpoint_distance_mm,
        "closest_endpoint_step": closest_endpoint_step,
        "endpoint_ever_reached": bool(
            minimum_endpoint_distance_mm <= args.endpoint_tolerance_mm
        ),
        "endpoint_reached": bool(metrics.endpoint_reached),
        "traversal_success": bool(metrics.traversal_success),
        "path_voxels": int(metrics.path_voxels),
        "target_intersection": int(metrics.target_intersection),
    }


def main() -> None:
    args = parse_args()
    if len(set(args.case_id)) != len(args.case_id):
        raise ValueError("Case IDs must be unique")
    positive_values = (
        args.voxel_size_mm,
        args.hessian_scale_mm,
        args.step_mm,
        args.maximum_steps,
        args.beam_width,
        args.maximum_turn_deg,
        args.seed_ct_bandwidth,
        args.path_radius_mm,
        args.endpoint_tolerance_mm,
    )
    if min(positive_values) <= 0:
        raise ValueError("All scales, counts, and tolerances must be positive")
    if args.recent_revisit_window < 0:
        raise ValueError("recent_revisit_window must be non-negative")
    if not 0 < args.hessian_strength_percentile < 100:
        raise ValueError("hessian_strength_percentile must be in (0, 100)")
    if not 0 < args.maximum_turn_deg <= 180:
        raise ValueError("maximum_turn_deg must be in (0, 180]")
    if min(
        args.axis_weight,
        args.curvature_weight,
        args.strength_weight,
        args.polarity_weight,
        args.polarity_gate_power,
        args.ct_weight,
        args.dark_weight,
        args.seed_ct_weight,
        args.boundary_weight,
        args.tube_balance_weight,
        args.revisit_penalty,
        args.step_cost,
    ) < 0:
        raise ValueError("Energy weights and penalties must be non-negative")
    if not 0 <= args.success_dice <= 1:
        raise ValueError("success_dice must be in [0, 1]")

    cases = [evaluate_case(args, case_id) for case_id in args.case_id]
    payload = {
        "data_dir": str(args.data_dir),
        "case_ids": args.case_id,
        "planner_inputs": [
            "clipped_ct",
            "dark_tubularity",
            "start_seed",
        ],
        "endpoint_and_segmentation_used_only_after_planning": True,
        "config": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "data_dir",
                "case_id",
                "output",
                "path_output_dir",
            }
        },
        "mean_dice": float(np.mean([case["dice"] for case in cases])),
        "mean_endpoint_distance_mm": float(
            np.mean([case["endpoint_distance_mm"] for case in cases])
        ),
        "mean_minimum_endpoint_distance_mm": float(
            np.mean(
                [case["minimum_endpoint_distance_mm"] for case in cases]
            )
        ),
        "endpoint_reach_rate": float(
            np.mean([case["endpoint_reached"] for case in cases])
        ),
        "endpoint_ever_reach_rate": float(
            np.mean([case["endpoint_ever_reached"] for case in cases])
        ),
        "traversal_success_rate": float(
            np.mean([case["traversal_success"] for case in cases])
        ),
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
