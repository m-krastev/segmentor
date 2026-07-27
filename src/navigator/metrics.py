"""Independent physical-space metrics for Navigator trajectories."""

from dataclasses import dataclass
from math import dist

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import line_nd


@dataclass(frozen=True)
class PathMetrics:
    dice: float
    endpoint_distance_mm: float
    endpoint_reached: bool
    traversal_success: bool
    path_voxels: int
    target_intersection: int


def _physical_boundary_tolerance(value: float) -> float:
    """Return a float32-scale tolerance for physical threshold comparisons."""

    return 8 * np.finfo(np.float32).eps * max(1.0, abs(float(value)))


def rasterize_path(
    shape: tuple[int, int, int],
    history: np.ndarray,
) -> np.ndarray:
    """Rasterize every segment of an ordered voxel trajectory."""

    points = np.asarray(history, dtype=np.int64).reshape(-1, 3)
    if not len(points):
        raise ValueError("A trajectory must contain at least one point.")
    shape_array = np.asarray(shape, dtype=np.int64)
    if np.any(points < 0) or np.any(points >= shape_array):
        raise ValueError("Trajectory contains points outside the target volume.")

    centerline = np.zeros(shape, dtype=bool)
    centerline[tuple(points[0])] = True
    for start, end in zip(points[:-1], points[1:]):
        centerline[line_nd(tuple(start), tuple(end), endpoint=True)] = True
    return centerline


def physical_path_tube(
    shape: tuple[int, int, int],
    history: np.ndarray,
    spacing_mm: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    """Return an isotropic physical-radius tube around a voxel trajectory."""

    if radius_mm < 0:
        raise ValueError("radius_mm must be non-negative")
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing_mm must contain three positive values")

    centerline = rasterize_path(shape, history)
    if radius_mm == 0:
        return centerline
    distance_to_path = distance_transform_edt(~centerline, sampling=spacing)
    return distance_to_path <= float(radius_mm) + _physical_boundary_tolerance(radius_mm)


def compute_path_metrics(
    target_mask: np.ndarray,
    history: np.ndarray,
    goal: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    path_radius_mm: float,
    endpoint_tolerance_mm: float,
    success_dice: float,
) -> PathMetrics:
    """Compute Dice and endpoint success independently from environment state."""

    if endpoint_tolerance_mm < 0:
        raise ValueError("endpoint_tolerance_mm must be non-negative")
    if not 0 <= success_dice <= 1:
        raise ValueError("success_dice must be between zero and one")

    target = np.asarray(target_mask, dtype=bool)
    path = physical_path_tube(
        target.shape,
        history,
        spacing_mm,
        path_radius_mm,
    )
    intersection = int(np.logical_and(path, target).sum())
    denominator = int(path.sum()) + int(target.sum())
    dice = float(2 * intersection / denominator) if denominator else 0.0

    final = np.asarray(history, dtype=np.float64).reshape(-1, 3)[-1]
    goal_array = np.asarray(goal, dtype=np.float64)
    spacing_array = np.asarray(spacing_mm, dtype=np.float64)
    endpoint_distance_mm = dist(final * spacing_array, goal_array * spacing_array)
    endpoint_reached = (
        endpoint_distance_mm
        <= endpoint_tolerance_mm
        + _physical_boundary_tolerance(endpoint_tolerance_mm)
    )

    return PathMetrics(
        dice=dice,
        endpoint_distance_mm=endpoint_distance_mm,
        endpoint_reached=endpoint_reached,
        traversal_success=endpoint_reached and dice >= success_dice,
        path_voxels=int(path.sum()),
        target_intersection=intersection,
    )
