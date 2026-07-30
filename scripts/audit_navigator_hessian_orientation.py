#!/usr/bin/env python3
"""Audit whether label-free CT Hessian axes align with bowel tangents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


RANDOM_AXIS_BASELINE = {
    "mean_abs_cosine": 0.5,
    "median_angle_deg": 60.0,
    "fraction_within_30_deg": 1.0 - np.cos(np.deg2rad(30.0)),
    "fraction_within_45_deg": 1.0 - np.cos(np.deg2rad(45.0)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--voxel-size-mm", type=float, default=1.5)
    parser.add_argument(
        "--scale-mm",
        type=float,
        action="append",
        default=[],
        help="Gaussian Hessian sigma in physical millimetres",
    )
    parser.add_argument("--tangent-radius-mm", type=float, default=6.0)
    parser.add_argument("--minimum-linearity", type=float, default=3.0)
    parser.add_argument("--max-points-per-case", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def connected_skeleton(
    segmentation: np.ndarray,
    start: np.ndarray,
) -> np.ndarray:
    components, _ = label(
        segmentation,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    component_id = int(components[tuple(start)])
    if component_id == 0:
        raise ValueError("Start seed is outside the segmentation")
    component = components == component_id
    skeleton = skeletonize(component, method="lee")
    coordinates = np.argwhere(skeleton)
    if not len(coordinates):
        raise ValueError("Endpoint-connected component has an empty skeleton")
    return coordinates


def sample_tangents(
    skeleton_coordinates: np.ndarray,
    *,
    tangent_radius_vox: float,
    minimum_linearity: float,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(skeleton_coordinates)
    order = rng.permutation(len(skeleton_coordinates))
    selected_points = []
    selected_tangents = []
    selected_linearity = []
    for coordinate_index in order:
        coordinate = skeleton_coordinates[coordinate_index]
        neighbor_indices = tree.query_ball_point(
            coordinate,
            tangent_radius_vox,
        )
        if len(neighbor_indices) < 5:
            continue
        neighborhood = skeleton_coordinates[neighbor_indices].astype(np.float64)
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order_descending = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order_descending]
        linearity = float(
            eigenvalues[0] / max(eigenvalues[1], np.finfo(np.float64).eps)
        )
        if linearity < minimum_linearity:
            continue
        selected_points.append(coordinate)
        selected_tangents.append(eigenvectors[:, order_descending[0]])
        selected_linearity.append(linearity)
        if len(selected_points) >= max_points:
            break
    if not selected_points:
        raise ValueError("No locally linear skeleton points passed the tangent gate")
    return (
        np.asarray(selected_points, dtype=np.int64),
        np.asarray(selected_tangents, dtype=np.float64),
        np.asarray(selected_linearity, dtype=np.float64),
    )


def sampled_hessian(
    image: np.ndarray,
    points: np.ndarray,
    sigma_vox: float,
) -> np.ndarray:
    point_indices = tuple(points[:, axis] for axis in range(3))
    components = {}
    for name, order in (
        ("xx", (2, 0, 0)),
        ("yy", (0, 2, 0)),
        ("zz", (0, 0, 2)),
        ("xy", (1, 1, 0)),
        ("xz", (1, 0, 1)),
        ("yz", (0, 1, 1)),
    ):
        derivative = gaussian_filter(
            image,
            sigma=sigma_vox,
            order=order,
            mode="nearest",
        )
        components[name] = np.asarray(
            derivative[point_indices],
            dtype=np.float64,
        )
        del derivative

    hessian = np.zeros((len(points), 3, 3), dtype=np.float64)
    hessian[:, 0, 0] = components["xx"]
    hessian[:, 1, 1] = components["yy"]
    hessian[:, 2, 2] = components["zz"]
    hessian[:, 0, 1] = hessian[:, 1, 0] = components["xy"]
    hessian[:, 0, 2] = hessian[:, 2, 0] = components["xz"]
    hessian[:, 1, 2] = hessian[:, 2, 1] = components["yz"]
    return hessian


def alignment_metrics(
    predicted_axes: np.ndarray,
    target_tangents: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
) -> dict:
    abs_cosine = np.abs(
        np.einsum("ij,ij->i", predicted_axes, target_tangents)
    )
    abs_cosine = np.clip(abs_cosine, 0.0, 1.0)
    angles = np.rad2deg(np.arccos(abs_cosine))

    def summarize(indices: np.ndarray) -> dict:
        selected_cosine = abs_cosine[indices]
        selected_angles = angles[indices]
        return {
            "point_count": int(len(indices)),
            "mean_abs_cosine": float(np.mean(selected_cosine)),
            "median_angle_deg": float(np.median(selected_angles)),
            "fraction_within_30_deg": float(
                np.mean(selected_angles <= 30.0)
            ),
            "fraction_within_45_deg": float(
                np.mean(selected_angles <= 45.0)
            ),
        }

    result = summarize(np.arange(len(predicted_axes)))
    if confidence is not None and len(confidence) >= 2:
        threshold = float(np.median(confidence))
        confident_indices = np.flatnonzero(confidence >= threshold)
        result["top_half_confidence_threshold"] = threshold
        result["top_half"] = summarize(confident_indices)
    return result


def audit_case(
    data_dir: Path,
    case_id: str,
    *,
    voxel_size_mm: float,
    scales_mm: tuple[float, ...],
    tangent_radius_mm: float,
    minimum_linearity: float,
    max_points: int,
    rng: np.random.Generator,
) -> dict:
    case_dir = data_dir / case_id
    image = nib.load(case_dir / "ct.nii.gz").get_fdata(dtype=np.float32)
    image = np.clip(image, -120.0, 180.0)
    segmentation = (
        nib.load(case_dir / "segmentations" / "small_bowel.nii.gz").get_fdata(
            dtype=np.float32
        )
        > 0
    )
    start = np.loadtxt(
        case_dir / "cache" / "start_end.npy",
        dtype=int,
    ).reshape(2, 3)[0]
    skeleton_coordinates = connected_skeleton(segmentation, start)
    points, tangents, linearity = sample_tangents(
        skeleton_coordinates,
        tangent_radius_vox=tangent_radius_mm / voxel_size_mm,
        minimum_linearity=minimum_linearity,
        max_points=max_points,
        rng=rng,
    )

    scale_results = {}
    for scale_mm in scales_mm:
        hessian = sampled_hessian(
            image,
            points,
            sigma_vox=scale_mm / voxel_size_mm,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        absolute_order = np.argsort(np.abs(eigenvalues), axis=1)
        point_indices = np.arange(len(points))
        min_abs_indices = absolute_order[:, 0]
        min_abs_axes = eigenvectors[point_indices, :, min_abs_indices]
        sorted_absolute = np.take_along_axis(
            np.abs(eigenvalues),
            absolute_order,
            axis=1,
        )
        confidence = (
            (sorted_absolute[:, 1] - sorted_absolute[:, 0])
            / np.maximum(sorted_absolute[:, 1], np.finfo(np.float64).eps)
        )
        scale_results[f"{scale_mm:g}"] = {
            "sigma_vox": scale_mm / voxel_size_mm,
            "min_abs_eigenvector": alignment_metrics(
                min_abs_axes,
                tangents,
                confidence=confidence,
            ),
            "most_negative_eigenvector": alignment_metrics(
                eigenvectors[:, :, 0],
                tangents,
            ),
            "most_positive_eigenvector": alignment_metrics(
                eigenvectors[:, :, 2],
                tangents,
            ),
            "median_axis_confidence": float(np.median(confidence)),
        }

    return {
        "case_id": case_id,
        "connected_skeleton_voxels": int(len(skeleton_coordinates)),
        "sampled_tangent_points": int(len(points)),
        "median_gt_linearity": float(np.median(linearity)),
        "scales": scale_results,
    }


def main() -> None:
    args = parse_args()
    scales_mm = tuple(args.scale_mm or (3.0, 6.0, 9.0))
    if len(set(args.case_id)) != len(args.case_id):
        raise ValueError("Case IDs must be unique")
    if min(
        args.voxel_size_mm,
        *scales_mm,
        args.tangent_radius_mm,
        args.minimum_linearity,
        args.max_points_per_case,
    ) <= 0:
        raise ValueError("All physical scales, gates, and counts must be positive")

    rng = np.random.default_rng(args.seed)
    cases = [
        audit_case(
            args.data_dir,
            case_id,
            voxel_size_mm=args.voxel_size_mm,
            scales_mm=scales_mm,
            tangent_radius_mm=args.tangent_radius_mm,
            minimum_linearity=args.minimum_linearity,
            max_points=args.max_points_per_case,
            rng=rng,
        )
        for case_id in args.case_id
    ]

    aggregate = {}
    for scale_mm in scales_mm:
        scale_key = f"{scale_mm:g}"
        for axis_name in (
            "min_abs_eigenvector",
            "most_negative_eigenvector",
            "most_positive_eigenvector",
        ):
            metrics = [
                case["scales"][scale_key][axis_name]
                for case in cases
            ]
            aggregate[f"{scale_key}mm/{axis_name}"] = {
                key: float(np.mean([metric[key] for metric in metrics]))
                for key in (
                    "mean_abs_cosine",
                    "median_angle_deg",
                    "fraction_within_30_deg",
                    "fraction_within_45_deg",
                )
            }
            if axis_name == "min_abs_eigenvector":
                aggregate[f"{scale_key}mm/{axis_name}"]["top_half"] = {
                    key: float(
                        np.mean(
                            [metric["top_half"][key] for metric in metrics]
                        )
                    )
                    for key in (
                        "mean_abs_cosine",
                        "median_angle_deg",
                        "fraction_within_30_deg",
                        "fraction_within_45_deg",
                    )
                }

    payload = {
        "data_dir": str(args.data_dir),
        "case_ids": args.case_id,
        "voxel_size_mm": args.voxel_size_mm,
        "scales_mm": scales_mm,
        "tangent_radius_mm": args.tangent_radius_mm,
        "minimum_linearity": args.minimum_linearity,
        "max_points_per_case": args.max_points_per_case,
        "random_unoriented_axis_baseline": RANDOM_AXIS_BASELINE,
        "cases": cases,
        "aggregate": aggregate,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
