import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import argparse
import nibabel as nib
import os

# --- Default Parameters (can be overridden by argparse) ---
DEFAULT_VOLUME_SHAPE = (128, 128, 128)  # Z, Y, X
DEFAULT_N_CONTROL_POINTS = 15
DEFAULT_TUBE_RADIUS = 3
DEFAULT_DILATION_ITERATIONS = 2
DEFAULT_DILATION_STRUCTURE_CONNECTIVITY = 1  # 1 for 6-connectivity, 2 for 18, 3 for 26

# Intensity values (Hounsfield Units - like, for example)
BG_INTENSITY = -1000  # Air
BODY_INTENSITY = 50  # Soft tissue
BOWEL_INTENSITY = 300  # Contrast-enhanced bowel

# Ellipsoid parameters for the 'body' (as fractions of volume shape)
ELLIPSOID_A_RATIO = 0.45  # x-axis radius
ELLIPSOID_B_RATIO = 0.35  # y-axis radius
ELLIPSOID_C_RATIO = 0.40  # z-axis radius

# --- Helper Functions ---


def create_ellipsoid_mask(shape, center_ratios, radii_ratios):
    """Creates a binary mask for an ellipsoid."""
    center_z, center_y, center_x = [s * r for s, r in zip(shape, center_ratios)]
    radius_z, radius_y, radius_x = [s * r for s, r in zip(shape, radii_ratios)]
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = (
        (x - center_x) ** 2 / radius_x**2
        + (y - center_y) ** 2 / radius_y**2
        + (z - center_z) ** 2 / radius_z**2
    ) <= 1
    return mask


def generate_spline_path(
    num_points_requested,
    num_control_points,
    bounds_min,
    bounds_max,
    tube_radius_for_margin,
    body_mask_for_filtering,  # FIX 1: Pass body_mask
    verbose=False,
):
    """Generates a smooth 3D spline path, ensuring its points are within the body_mask."""
    margin = tube_radius_for_margin * 3.0  # Margin for control points

    control_points_coords = []
    for d in range(3):  # For Z, Y, X dimensions
        dim_min, dim_max = bounds_min[d], bounds_max[d]
        eff_min, eff_max = dim_min + margin, dim_max - margin
        if eff_min >= eff_max:
            if verbose:
                print(
                    f"Warning: Dimension {d} for spline control points is too small due to margin. Using center of bound."
                )
            pts_dim = np.full(num_control_points, (dim_min + dim_max) / 2.0)
        else:
            pts_dim = np.random.uniform(eff_min, eff_max, num_control_points)
        control_points_coords.append(np.clip(pts_dim, dim_min, dim_max))

    ctrl_pts_z, ctrl_pts_y, ctrl_pts_x = (
        control_points_coords[0],
        control_points_coords[1],
        control_points_coords[2],
    )

    sorted_indices = np.argsort(ctrl_pts_z)
    ctrl_pts_z, ctrl_pts_y, ctrl_pts_x = (
        ctrl_pts_z[sorted_indices],
        ctrl_pts_y[sorted_indices],
        ctrl_pts_x[sorted_indices],
    )

    min_unique_pts = min(
        len(np.unique(ctrl_pts_x)), len(np.unique(ctrl_pts_y)), len(np.unique(ctrl_pts_z))
    )
    spline_degree = min(3, max(1, min_unique_pts - 1))

    if len(ctrl_pts_x) <= spline_degree:
        if verbose:
            print(
                f"Error: Not enough control points ({len(ctrl_pts_x)}) for spline of degree {spline_degree}. Returning empty path."
            )
        return np.array([])

    try:
        tck, u = splprep([ctrl_pts_x, ctrl_pts_y, ctrl_pts_z], s=2, k=spline_degree, quiet=2)
    except Exception as e:
        if verbose:
            print(f"Error during splprep: {e}. Returning empty path.")
        return np.array([])

    u_fine = np.linspace(0, 1, num_points_requested)
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    path_candidates = np.vstack((z_fine, y_fine, x_fine)).T.astype(int)

    # Clip path points to be strictly within the volume dimensions before checking body_mask
    path_candidates[:, 0] = np.clip(path_candidates[:, 0], 0, body_mask_for_filtering.shape[0] - 1)
    path_candidates[:, 1] = np.clip(path_candidates[:, 1], 0, body_mask_for_filtering.shape[1] - 1)
    path_candidates[:, 2] = np.clip(path_candidates[:, 2], 0, body_mask_for_filtering.shape[2] - 1)

    # FIX 1: Filter points to be within the body_mask
    # Accessing body_mask_for_filtering with (z, y, x) arrays
    valid_mask_indices = body_mask_for_filtering[
        path_candidates[:, 0], path_candidates[:, 1], path_candidates[:, 2]
    ]
    filtered_path_points = path_candidates[valid_mask_indices]

    if verbose:
        print(
            f"   Spline: {path_candidates.shape[0]} candidates, {filtered_path_points.shape[0]} after body mask filtering."
        )
    return filtered_path_points


def _draw_sphere_at_point(volume, center_z, center_y, center_x, radius, value=1):
    """Helper to draw a single sphere. Modifies volume in-place."""
    z_coords, y_coords, x_coords = volume.shape
    z_min, z_max = max(0, center_z - radius), min(z_coords, center_z + radius + 1)
    y_min, y_max = max(0, center_y - radius), min(y_coords, center_y + radius + 1)
    x_min, x_max = max(0, center_x - radius), min(x_coords, center_x + radius + 1)

    for z_idx in range(int(np.floor(z_min)), int(np.ceil(z_max))):
        for y_idx in range(int(np.floor(y_min)), int(np.ceil(y_max))):
            for x_idx in range(int(np.floor(x_min)), int(np.ceil(x_max))):
                if 0 <= z_idx < z_coords and 0 <= y_idx < y_coords and 0 <= x_idx < x_coords:
                    if (x_idx - center_x) ** 2 + (y_idx - center_y) ** 2 + (
                        z_idx - center_z
                    ) ** 2 <= radius**2:
                        volume[z_idx, y_idx, x_idx] = value


def create_tube_from_path(shape, path_points, radius):
    """Creates a uint8 mask of a tube along a given path, ensuring continuity."""
    tube_mask = np.zeros(shape, dtype=np.uint8)
    if path_points.shape[0] < 2:  # Need at least two points to form a segment for interpolation
        if path_points.shape[0] == 1:  # Single point, draw a sphere
            _draw_sphere_at_point(
                tube_mask, path_points[0, 0], path_points[0, 1], path_points[0, 2], radius, value=1
            )
        return tube_mask

    dense_path_points_for_drawing = []
    dense_path_points_for_drawing.append(path_points[0])  # Start with the first point

    for i in range(len(path_points) - 1):
        p1 = path_points[i].astype(float)
        p2 = path_points[i + 1].astype(float)
        dist = np.linalg.norm(p2 - p1)
        step_size = max(1.0, radius * 0.75)  # Ensure overlap, step at most 0.75 * radius

        if dist > step_size:
            num_steps = int(np.ceil(dist / step_size))
            if num_steps > 1:
                for step_idx in range(1, num_steps):
                    interp_ratio = step_idx / num_steps
                    interp_point = p1 * (1 - interp_ratio) + p2 * interp_ratio
                    dense_path_points_for_drawing.append(interp_point.astype(int))
        dense_path_points_for_drawing.append(p2.astype(int))

    # Remove duplicates that might arise from int casting or dense interpolation
    # Using a set of tuples for uniqueness
    unique_dense_points_tuples = sorted(list(set(map(tuple, dense_path_points_for_drawing))))
    final_points_to_draw = [np.array(pt) for pt in unique_dense_points_tuples]

    for point in final_points_to_draw:
        _draw_sphere_at_point(tube_mask, point[0], point[1], point[2], radius, value=1)

    return tube_mask


# --- Main Generation Logic ---
def generate_synthetic_ct_with_bowel(
    volume_shape,
    n_control_points,
    tube_radius,
    dilation_iterations,
    dilation_structure_connectivity,
    verbose=False,
):
    print("1. Initializing volumes...")
    ct_volume = np.full(volume_shape, BG_INTENSITY, dtype=np.int16)
    small_bowel_gt_mask = np.zeros(volume_shape, dtype=np.uint8)

    print("2. Creating 'Body' region (ellipsoid)...")
    body_center_ratios = (0.5, 0.5, 0.5)
    body_radii_ratios = (ELLIPSOID_C_RATIO, ELLIPSOID_B_RATIO, ELLIPSOID_A_RATIO)
    body_mask = create_ellipsoid_mask(volume_shape, body_center_ratios, body_radii_ratios)
    ct_volume[body_mask] = BODY_INTENSITY
    if verbose:
        print(f"   Body mask voxels: {np.sum(body_mask)}")

    print("3. Defining bounds for spline within the body...")
    where_body = np.where(body_mask)
    if not all(len(coords) > 0 for coords in where_body):
        print("Error: Body mask is empty. Check ellipsoid parameters and volume shape.")
        return None, None, None, None  # CT, GT, start, end
    body_bounds_min = [np.min(coords) for coords in where_body]
    body_bounds_max = [np.max(coords) for coords in where_body]
    if verbose:
        print(f"   Body bounds (min): {body_bounds_min}, (max): {body_bounds_max}")

    print("4. Generating small bowel path (spline)...")
    num_spline_points_requested = int(np.sum(volume_shape) * 0.75)
    bowel_path_points = generate_spline_path(
        num_spline_points_requested,
        n_control_points,
        body_bounds_min,
        body_bounds_max,
        tube_radius,
        body_mask,  # Pass body_mask for filtering
        verbose=verbose,
    )

    start_coord_zyx, end_coord_zyx = None, None
    if bowel_path_points.shape[0] == 0:
        print(
            "Warning: Bowel path is empty after spline generation and filtering. No bowel will be drawn."
        )
    elif bowel_path_points.shape[0] < 2:
        print(
            "Warning: Bowel path has fewer than 2 points. Only a sphere (or nothing) will be drawn."
        )
        start_coord_zyx = bowel_path_points[0]
        end_coord_zyx = bowel_path_points[0]  # Start and end are the same
    else:
        start_coord_zyx = bowel_path_points[0]
        end_coord_zyx = bowel_path_points[-1]
        if verbose:
            print(
                f"   Bowel path: {bowel_path_points.shape[0]} points. Start: {start_coord_zyx}, End: {end_coord_zyx}"
            )

    print("5. Creating precise small bowel ground truth mask...")
    # Pass the filtered bowel_path_points to create_tube_from_path
    small_bowel_gt_mask_raw = create_tube_from_path(volume_shape, bowel_path_points, tube_radius)
    if verbose:
        print(
            f"   Raw GT bowel voxels (before final body masking): {np.sum(small_bowel_gt_mask_raw)}"
        )

    small_bowel_gt_mask = small_bowel_gt_mask_raw.copy()  # Work on a copy
    small_bowel_gt_mask[~body_mask] = 0  # Ensure GT is only within the body region
    if verbose:
        print(f"   GT bowel voxels (after final body masking): {np.sum(small_bowel_gt_mask)}")

    if np.sum(small_bowel_gt_mask) == 0 and bowel_path_points.shape[0] > 0:
        print(
            "Warning: Small bowel ground truth mask is empty even though path points existed. This might indicate all path points were at the very edge and tube voxels fell outside body_mask."
        )

    print("6. Dilating bowel mask for CT appearance...")
    dilation_structure = ndi.generate_binary_structure(3, dilation_structure_connectivity)
    dilated_bowel_for_ct_bool = ndi.binary_dilation(
        small_bowel_gt_mask.astype(bool),
        structure=dilation_structure,
        iterations=dilation_iterations,
    )
    dilated_bowel_for_ct = dilated_bowel_for_ct_bool.astype(np.uint8)
    dilated_bowel_for_ct[~body_mask] = 0  # Ensure dilated bowel also respects body boundary

    if verbose:
        print(
            f"   Dilated bowel voxels for CT (after body masking): {np.sum(dilated_bowel_for_ct)}"
        )

    print("7. Embedding bowel into CT volume...")
    ct_volume[dilated_bowel_for_ct == 1] = BOWEL_INTENSITY
    ct_volume[~body_mask] = BG_INTENSITY

    print("Generation complete.")
    return ct_volume, bowel_path_points, small_bowel_gt_mask, start_coord_zyx, end_coord_zyx


def save_nifti(data_array, filepath, affine=None):
    if affine is None:
        affine = np.eye(4)
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Determine dtype for saving
    if data_array.dtype == np.uint8:  # Typically masks
        save_dtype = np.uint8
    elif data_array.dtype.kind in ["i", "u"]:  # Other integers
        save_dtype = np.int16  # Common for CT, adjust if necessary
    else:  # Floats, etc.
        save_dtype = data_array.dtype  # Or convert to a standard float type

    nifti_img = nib.Nifti1Image(data_array.astype(save_dtype), affine)
    nib.save(nifti_img, filepath)
    print(f"Saved NIfTI file to: {filepath}")


def save_coordinates_to_txt(start_coord_zyx, end_coord_zyx, filepath):
    """Saves start and end coordinates (X, Y, Z) to a text file."""
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(filepath, "w") as f:
        if start_coord_zyx is not None:
            # Convert ZYX to XYZ for output
            f.write(f"{start_coord_zyx[0]} {start_coord_zyx[0]} {start_coord_zyx[2]}\n")
        else:
            f.write("N/A N/A N/A\n")  # Placeholder if no start coordinate

        if end_coord_zyx is not None:
            f.write(f"{end_coord_zyx[0]} {end_coord_zyx[1]} {end_coord_zyx[2]}\n")
        else:
            f.write("N/A N/A N/A\n")  # Placeholder if no end coordinate
    print(f"Saved start/end coordinates to: {filepath}")


def parse_volume_shape_arg(shape_str):
    # (Implementation from previous version)
    try:
        dims = tuple(map(int, shape_str.split(",")))
        if len(dims) != 3:
            raise argparse.ArgumentTypeError(
                "Volume shape must be three comma-separated integers (e.g., 128,128,128)"
            )
        for d in dims:
            if d <= 0:
                raise argparse.ArgumentTypeError("Volume dimensions must be positive integers.")
        return dims
    except ValueError:
        raise argparse.ArgumentTypeError("Volume shape components must be integers.")


# --- Main Execution (with argparse) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic CT scan with a small bowel ground truth."
    )
    parser.add_argument(
        "--output_ct_path", type=str, required=True, help="Path to save CT NIfTI (e.g., ct.nii.gz)."
    )
    parser.add_argument(
        "--output_gt_path",
        type=str,
        required=True,
        help="Path to save GT mask NIfTI (e.g., gt.nii.gz).",
    )
    parser.add_argument(
        "--output_coords_path",
        type=str,
        required=True,
        help="Path to save start/end coordinates TXT (e.g., coords.txt).",
    )

    parser.add_argument(
        "--output_path_path",
        type=str,
        required=True,
        help="Path to save the path points.",
    )

    parser.add_argument(
        "--volume_shape",
        type=parse_volume_shape_arg,
        default=",".join(map(str, DEFAULT_VOLUME_SHAPE)),
        help=f"Z,Y,X shape. Default: {DEFAULT_VOLUME_SHAPE}",
    )
    parser.add_argument(
        "--n_control_points",
        type=int,
        default=DEFAULT_N_CONTROL_POINTS,
        help=f"Spline control points. Default: {DEFAULT_N_CONTROL_POINTS}",
    )
    parser.add_argument(
        "--tube_radius",
        type=int,
        default=DEFAULT_TUBE_RADIUS,
        help=f"Bowel tube radius. Default: {DEFAULT_TUBE_RADIUS}",
    )
    parser.add_argument(
        "--dilation_iterations",
        type=int,
        default=DEFAULT_DILATION_ITERATIONS,
        help=f"Bowel dilation iterations for CT. Default: {DEFAULT_DILATION_ITERATIONS}",
    )
    parser.add_argument(
        "--dilation_connectivity",
        type=int,
        default=DEFAULT_DILATION_STRUCTURE_CONNECTIVITY,
        choices=[1, 2, 3],
        help=f"Dilation connectivity. Default: {DEFAULT_DILATION_STRUCTURE_CONNECTIVITY}",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--visualize", action="store_true", help="Show 2D slice visualization.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output.")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    synthetic_ct, bowel_path_points, ground_truth_bowel, start_coord, end_coord = generate_synthetic_ct_with_bowel(
        volume_shape=args.volume_shape,
        n_control_points=args.n_control_points,
        tube_radius=args.tube_radius,
        dilation_iterations=args.dilation_iterations,
        dilation_structure_connectivity=args.dilation_connectivity,
        verbose=args.verbose,
    )

    if synthetic_ct is not None and ground_truth_bowel is not None:
        # (Verbose print statements from previous version can be kept here)
        if args.verbose:
            print(
                f"\nSynthetic CT shape: {synthetic_ct.shape}, dtype: {synthetic_ct.dtype}, min: {np.min(synthetic_ct)}, max: {np.max(synthetic_ct)}"
            )
            print(
                f"Ground Truth Bowel Mask shape: {ground_truth_bowel.shape}, dtype: {ground_truth_bowel.dtype}, min: {np.min(ground_truth_bowel)}, max: {np.max(ground_truth_bowel)}"
            )
            print(f"Unique values in CT: {np.unique(synthetic_ct)}")
            print(f"Unique values in GT: {np.unique(ground_truth_bowel)}")
            print(f"Number of voxels in final GT mask: {np.sum(ground_truth_bowel)}")
            if start_coord is not None:
                print(f"Start coord (ZYX): {start_coord}")
            if end_coord is not None:
                print(f"End coord (ZYX): {end_coord}")

        affine_identity = np.eye(4)
        save_nifti(synthetic_ct, args.output_ct_path, affine=affine_identity)
        save_nifti(ground_truth_bowel, args.output_gt_path, affine=affine_identity)
        # Save path points to a text file
        np.savetxt(args.output_path_path, bowel_path_points, fmt="%d", delimiter=" ")
        save_coordinates_to_txt(start_coord, end_coord, args.output_coords_path)

        if args.visualize:
            # (Visualization code from previous version)
            slice_idx_z, slice_idx_y, slice_idx_x = [d // 2 for d in args.volume_shape]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Synthetic CT and Ground Truth (Central Slices)", fontsize=16)
            vmin_ct, vmax_ct = np.min(synthetic_ct), np.max(synthetic_ct)
            if vmin_ct == vmax_ct:
                vmin_ct, vmax_ct = BG_INTENSITY, BOWEL_INTENSITY + 50
            axes[0, 0].imshow(
                synthetic_ct[slice_idx_z, :, :],
                cmap="gray",
                origin="lower",
                vmin=vmin_ct,
                vmax=vmax_ct,
            )
            axes[0, 0].set_title(f"CT - Axial (Z={slice_idx_z})")
            axes[0, 0].axis("off")
            axes[1, 0].imshow(
                ground_truth_bowel[slice_idx_z, :, :], cmap="hot", origin="lower", vmin=0, vmax=1
            )
            axes[1, 0].set_title(f"GT - Axial (Z={slice_idx_z})")
            axes[1, 0].axis("off")
            axes[0, 1].imshow(
                synthetic_ct[:, slice_idx_y, :],
                cmap="gray",
                origin="lower",
                vmin=vmin_ct,
                vmax=vmax_ct,
            )
            axes[0, 1].set_title(f"CT - Coronal (Y={slice_idx_y})")
            axes[0, 1].axis("off")
            axes[1, 1].imshow(
                ground_truth_bowel[:, slice_idx_y, :], cmap="hot", origin="lower", vmin=0, vmax=1
            )
            axes[1, 1].set_title(f"GT - Coronal (Y={slice_idx_y})")
            axes[1, 1].axis("off")
            axes[0, 2].imshow(
                synthetic_ct[:, :, slice_idx_x],
                cmap="gray",
                origin="lower",
                vmin=vmin_ct,
                vmax=vmax_ct,
            )
            axes[0, 2].set_title(f"CT - Sagittal (X={slice_idx_x})")
            axes[0, 2].axis("off")
            axes[1, 2].imshow(
                ground_truth_bowel[:, :, slice_idx_x], cmap="hot", origin="lower", vmin=0, vmax=1
            )
            axes[1, 2].set_title(f"GT - Sagittal (X={slice_idx_x})")
            axes[1, 2].axis("off")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
    else:
        print("Failed to generate synthetic data.")
