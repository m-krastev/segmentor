#!/usr/bin/env python3
"""Evaluate a label-free continuous Hessian streamline from one start seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from navigator.metrics import compute_path_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--voxel-size-mm", type=float, default=1.5)
    parser.add_argument("--hessian-scale-mm", type=float, default=6.0)
    parser.add_argument("--step-mm", type=float, default=4.5)
    parser.add_argument("--maximum-steps", type=int, default=2048)
    parser.add_argument("--direction-momentum", type=float, default=0.5)
    parser.add_argument("--recent-revisit-window", type=int, default=24)
    parser.add_argument("--path-radius-mm", type=float, default=6.0)
    parser.add_argument("--endpoint-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--success-dice", type=float, default=0.4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--path-output-dir", type=Path)
    return parser.parse_args()


def compute_hessian_components(
    image: np.ndarray,
    sigma_vox: float,
) -> dict[str, np.ndarray]:
    """Compute scale-normalized Hessian fields without anatomical labels."""

    result = {}
    for name, order in (
        ("xx", (2, 0, 0)),
        ("yy", (0, 2, 0)),
        ("zz", (0, 0, 2)),
        ("xy", (1, 1, 0)),
        ("xz", (1, 0, 1)),
        ("yz", (0, 1, 1)),
    ):
        result[name] = (
            gaussian_filter(
                image,
                sigma=sigma_vox,
                order=order,
                mode="nearest",
            )
            * sigma_vox**2
        ).astype(np.float32)
    return result


def sample_hessian(
    components: dict[str, np.ndarray],
    position: np.ndarray,
) -> np.ndarray:
    coordinates = np.asarray(position, dtype=np.float64).reshape(3, 1)

    def sample(name: str) -> float:
        return float(
            map_coordinates(
                components[name],
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )[0]
        )

    xx, yy, zz = sample("xx"), sample("yy"), sample("zz")
    xy, xz, yz = sample("xy"), sample("xz"), sample("yz")
    return np.asarray(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ],
        dtype=np.float64,
    )


def tube_axis(
    components: dict[str, np.ndarray],
    position: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    hessian = sample_hessian(components, position)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    absolute_order = np.argsort(np.abs(eigenvalues))
    absolute_values = np.abs(eigenvalues[absolute_order])
    axis = eigenvectors[:, absolute_order[0]]
    axis_gap_confidence = float(
        (absolute_values[1] - absolute_values[0])
        / max(absolute_values[1], np.finfo(np.float64).eps)
    )
    transverse_balance = float(
        absolute_values[1]
        / max(absolute_values[2], np.finfo(np.float64).eps)
    )
    return axis, axis_gap_confidence, transverse_balance


def inside_margin(position: np.ndarray, shape: tuple[int, ...]) -> bool:
    return bool(
        np.all(position >= 1.0)
        and np.all(position <= np.asarray(shape, dtype=np.float64) - 2.0)
    )


def follow_streamline(
    components: dict[str, np.ndarray],
    start: np.ndarray,
    *,
    initial_sign: float,
    step_vox: float,
    maximum_steps: int,
    direction_momentum: float,
    recent_revisit_window: int,
) -> dict:
    """Integrate one unoriented Hessian axis branch in continuous voxel space."""

    position = np.asarray(start, dtype=np.float64)
    initial_axis, confidence, balance = tube_axis(components, position)
    previous_direction = initial_axis * initial_sign
    points = [position.copy()]
    confidences = [confidence]
    balances = [balance]
    rounded_history = [tuple(np.rint(position).astype(int))]
    stop_reason = "maximum_steps"

    for _ in range(maximum_steps):
        axis, confidence, balance = tube_axis(components, position)
        if np.dot(axis, previous_direction) < 0:
            axis = -axis
        direction = (
            direction_momentum * previous_direction
            + (1.0 - direction_momentum) * axis
        )
        norm = float(np.linalg.norm(direction))
        if norm <= np.finfo(np.float64).eps:
            stop_reason = "degenerate_direction"
            break
        direction /= norm

        midpoint = position + 0.5 * step_vox * direction
        if not inside_margin(midpoint, components["xx"].shape):
            stop_reason = "boundary"
            break
        midpoint_axis, midpoint_confidence, midpoint_balance = tube_axis(
            components,
            midpoint,
        )
        if np.dot(midpoint_axis, direction) < 0:
            midpoint_axis = -midpoint_axis
        direction = (
            direction_momentum * direction
            + (1.0 - direction_momentum) * midpoint_axis
        )
        direction /= max(
            float(np.linalg.norm(direction)),
            np.finfo(np.float64).eps,
        )
        next_position = position + step_vox * direction
        if not inside_margin(next_position, components["xx"].shape):
            stop_reason = "boundary"
            break

        rounded = tuple(np.rint(next_position).astype(int))
        older_history = rounded_history[:-recent_revisit_window]
        if rounded in older_history:
            stop_reason = "revisit"
            break

        position = next_position
        previous_direction = direction
        points.append(position.copy())
        confidences.append(midpoint_confidence)
        balances.append(midpoint_balance)
        rounded_history.append(rounded)

    unique_voxels = len(set(rounded_history))
    mean_confidence = float(np.mean(confidences))
    mean_balance = float(np.mean(balances))
    # Select the seed direction without target information. Long, confident,
    # transversely tube-like tracks dominate short exits and flat-field drift.
    selection_score = (
        unique_voxels
        * max(mean_confidence, 0.0)
        * max(mean_balance, 0.0)
    )
    return {
        "points": np.asarray(points, dtype=np.float64),
        "rounded_history": np.asarray(rounded_history, dtype=np.int64),
        "steps": len(points) - 1,
        "unique_voxels": unique_voxels,
        "mean_axis_confidence": mean_confidence,
        "mean_transverse_balance": mean_balance,
        "selection_score": float(selection_score),
        "stop_reason": stop_reason,
    }


def plan_streamline(
    image: np.ndarray,
    start: np.ndarray,
    *,
    sigma_vox: float,
    step_vox: float,
    maximum_steps: int,
    direction_momentum: float,
    recent_revisit_window: int,
) -> tuple[dict, list[dict]]:
    components = compute_hessian_components(image, sigma_vox)
    candidates = [
        follow_streamline(
            components,
            start,
            initial_sign=sign,
            step_vox=step_vox,
            maximum_steps=maximum_steps,
            direction_momentum=direction_momentum,
            recent_revisit_window=recent_revisit_window,
        )
        for sign in (-1.0, 1.0)
    ]
    selected = max(candidates, key=lambda candidate: candidate["selection_score"])
    return selected, candidates


def evaluate_case(args: argparse.Namespace, case_id: str) -> dict:
    case_dir = args.data_dir / case_id
    image = nib.load(case_dir / "ct.nii.gz").get_fdata(dtype=np.float32)
    image = np.clip(image, -120.0, 180.0)
    start_end = np.loadtxt(
        case_dir / "cache" / "start_end.npy",
        dtype=int,
    ).reshape(2, 3)
    start, endpoint = start_end

    selected, candidates = plan_streamline(
        image,
        start,
        sigma_vox=args.hessian_scale_mm / args.voxel_size_mm,
        step_vox=args.step_mm / args.voxel_size_mm,
        maximum_steps=args.maximum_steps,
        direction_momentum=args.direction_momentum,
        recent_revisit_window=args.recent_revisit_window,
    )

    # The target and endpoint are opened only after label-free planning.
    target = (
        nib.load(case_dir / "segmentations" / "small_bowel.nii.gz").get_fdata(
            dtype=np.float32
        )
        > 0
    )
    metrics = compute_path_metrics(
        target,
        selected["rounded_history"],
        tuple(int(value) for value in endpoint),
        (args.voxel_size_mm,) * 3,
        args.path_radius_mm,
        args.endpoint_tolerance_mm,
        args.success_dice,
    )
    if args.path_output_dir:
        args.path_output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            args.path_output_dir / f"{case_id}_history_float.npy",
            selected["points"],
        )
        np.save(
            args.path_output_dir / f"{case_id}_history_vox.npy",
            selected["rounded_history"],
        )

    candidate_summaries = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"points", "rounded_history"}
        }
        for candidate in candidates
    ]
    return {
        "case_id": case_id,
        "start": start.tolist(),
        "endpoint_for_audit_only": endpoint.tolist(),
        "selected_candidate": int(
            np.argmax([candidate["selection_score"] for candidate in candidates])
        ),
        "candidate_summaries": candidate_summaries,
        "dice": float(metrics.dice),
        "endpoint_distance_mm": float(metrics.endpoint_distance_mm),
        "endpoint_reached": bool(metrics.endpoint_reached),
        "traversal_success": bool(metrics.traversal_success),
        "path_voxels": int(metrics.path_voxels),
        "target_intersection": int(metrics.target_intersection),
    }


def main() -> None:
    args = parse_args()
    if len(set(args.case_id)) != len(args.case_id):
        raise ValueError("Case IDs must be unique")
    if min(
        args.voxel_size_mm,
        args.hessian_scale_mm,
        args.step_mm,
        args.maximum_steps,
        args.recent_revisit_window,
        args.path_radius_mm,
        args.endpoint_tolerance_mm,
    ) <= 0:
        raise ValueError("All scales, counts, and tolerances must be positive")
    if not 0 <= args.direction_momentum < 1:
        raise ValueError("direction_momentum must be in [0, 1)")
    if not 0 <= args.success_dice <= 1:
        raise ValueError("success_dice must be in [0, 1]")

    cases = [evaluate_case(args, case_id) for case_id in args.case_id]
    payload = {
        "data_dir": str(args.data_dir),
        "case_ids": args.case_id,
        "planner_inputs": ["clipped_ct", "start_seed"],
        "endpoint_and_segmentation_used_only_after_planning": True,
        "hessian_scale_mm": args.hessian_scale_mm,
        "step_mm": args.step_mm,
        "maximum_steps": args.maximum_steps,
        "direction_momentum": args.direction_momentum,
        "recent_revisit_window": args.recent_revisit_window,
        "path_radius_mm": args.path_radius_mm,
        "endpoint_tolerance_mm": args.endpoint_tolerance_mm,
        "success_dice": args.success_dice,
        "mean_dice": float(np.mean([case["dice"] for case in cases])),
        "mean_endpoint_distance_mm": float(
            np.mean([case["endpoint_distance_mm"] for case in cases])
        ),
        "endpoint_reach_rate": float(
            np.mean([case["endpoint_reached"] for case in cases])
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
