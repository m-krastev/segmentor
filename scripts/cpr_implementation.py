import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates, gaussian_filter


# --- Vector Math Helper Functions ---
def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    return v / norm


# ===================================================================
# STEP 0: GENERATE SYNTHETIC DATA
# ===================================================================
def create_synthetic_volume_and_centerline(
    vol_shape=(100, 100, 100), voxel_spacing=(1.0, 1.0, 1.0)
):
    """
    Creates a 3D NumPy array with a bright, curved tube inside it.
    This version is memory-efficient and avoids creating large intermediate arrays.
    """
    print("1. Creating synthetic 3D volume (memory-efficiently)...")

    # 1. Create the centerline (same as before)
    t = np.linspace(5 * np.pi, 9 * np.pi, 200)
    centerline = np.vstack([25 * np.cos(t) + 50, 25 * np.sin(t) + 50, np.linspace(10, 90, 200)]).T

    # 2. Create a grid of voxel coordinates in physical units (mm)
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(vol_shape[0]) * voxel_spacing[0],
        np.arange(vol_shape[1]) * voxel_spacing[1],
        np.arange(vol_shape[2]) * voxel_spacing[2],
        indexing="ij",
    )

    # 3. Initialize an array to store the minimum squared distance for each voxel
    #    This array is the same size as the volume. We start with infinity.
    min_dist_sq = np.full(vol_shape, np.inf, dtype=np.float32)

    # 4. Iterate through each point on the centerline
    for point in centerline:
        # Calculate the squared distance from every voxel to the current point
        dist_sq = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2 + (grid_z - point[2]) ** 2

        # Update our minimum distance array. This is a fast, vectorized operation.
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)

    # 5. Create the final volume based on the minimum distance
    tube_radius = 5.0
    volume = (min_dist_sq < tube_radius**2).astype(np.float32) * 255

    # 6. Smooth the result
    volume = gaussian_filter(volume, sigma=1.5)

    return volume, centerline

# ===================================================================
# STEP 1: PREPARE THE CENTERLINE
# ===================================================================
def resample_centerline(centerline, point_spacing=0.5):
    print("2. Resampling centerline to have equidistant points...")
    distances = np.sqrt(np.sum(np.diff(centerline, axis=0) ** 2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    fx, fy, fz = [interp1d(cumulative_dist, centerline[:, i]) for i in range(3)]
    num_new_points = int(cumulative_dist[-1] / point_spacing)
    new_distances = np.linspace(0, cumulative_dist[-1], num_new_points)
    resampled = np.vstack([fx(new_distances), fy(new_distances), fz(new_distances)]).T
    return resampled


# ===================================================================
# STEP 2: COMPUTE THE LOCAL COORDINATE SYSTEM
# ===================================================================
def compute_rotation_minimizing_frames(centerline):
    print("3. Computing rotation-minimizing frames (T, N, B vectors)...")
    tangents = normalize(np.gradient(centerline, axis=0))
    v_up = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(tangents[0], v_up)) > 0.999:
        v_up = np.array([0.0, 1.0, 0.0])

    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)
    binormals[0] = normalize(np.cross(tangents[0], v_up))
    normals[0] = normalize(np.cross(binormals[0], tangents[0]))

    for i in range(1, len(tangents)):
        v1 = centerline[i] - centerline[i - 1]
        c1 = np.dot(v1, v1)
        n_prev = normals[i - 1]
        n_transported = n_prev - (2 / c1) * np.dot(n_prev, v1) * v1
        normals[i] = normalize(n_transported - np.dot(n_transported, tangents[i]) * tangents[i])
        binormals[i] = normalize(np.cross(tangents[i], normals[i]))
    return tangents, normals, binormals


# ===================================================================
# STEP 3 & 4: RESAMPLE THE VOLUME AND CREATE THE CPR IMAGE
# ===================================================================
def create_cpr_image(
    volume, voxel_spacing, centerline, normals, cpr_height_mm=40, pixel_spacing_mm=0.5
):
    print("4. Generating CPR image by sampling the 3D volume...")

    cpr_width_pixels = len(centerline)
    cpr_height_pixels = int(cpr_height_mm / pixel_spacing_mm)

    u_coords = np.arange(cpr_width_pixels)
    v_coords = np.arange(cpr_height_pixels)
    uu, vv = np.meshgrid(u_coords, v_coords)

    dist_from_center = (vv - cpr_height_pixels / 2) * pixel_spacing_mm

    centerline_points = centerline[uu]
    normal_vectors = normals[uu]

    sample_coords_mm = centerline_points + dist_from_center[..., np.newaxis] * normal_vectors

    voxel_coords = sample_coords_mm / np.array(voxel_spacing)

    # --- THIS IS THE CORRECTED PART ---
    # The order must match the volume's axes: (x, y, z) for volume[x,y,z]
    # which corresponds to (axis 0, axis 1, axis 2)
    coords_for_scipy = np.array([
        voxel_coords[:, :, 0],  # x coordinates -> axis 0
        voxel_coords[:, :, 1],  # y coordinates -> axis 1
        voxel_coords[:, :, 2],  # z coordinates -> axis 2
    ])

    cpr_image_flat = map_coordinates(volume, coords_for_scipy, order=1, mode="constant", cval=0.0)

    cpr_image = cpr_image_flat.reshape(cpr_height_pixels, cpr_width_pixels)
    return cpr_image


# ===================================================================
# MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    VOXEL_SPACING = (1.0, 1.0, 1.0)
    VOLUME_SHAPE = (100, 100, 100)
    CENTERLINE_POINT_SPACING = 0.5
    CPR_HEIGHT_MM = 10
    CPR_PIXEL_SPACING = 0.5

    # volume, raw_centerline = create_synthetic_volume_and_centerline(VOLUME_SHAPE, VOXEL_SPACING)
    import nibabel as nib
    
    volume = nib.load("/Users/matey/project/segmentor/data/phantoms/patient_007/ct.nii.gz").get_fdata()
    raw_centerline = np.loadtxt("/Users/matey/project/segmentor/data/phantoms/patient_007/path.npy", dtype=int)
    resampled_centerline = resample_centerline(raw_centerline, CENTERLINE_POINT_SPACING)
    tangents, normals, binormals = compute_rotation_minimizing_frames(resampled_centerline)
    cpr_image = create_cpr_image(
        volume, VOXEL_SPACING, resampled_centerline, normals, CPR_HEIGHT_MM, CPR_PIXEL_SPACING
    )

    print("5. Displaying results...")
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(volume[:, :, VOLUME_SHAPE[2] // 2].T, cmap="gray", origin="lower")
    ax1.set_title("Original 3D Volume (Z-slice)")
    ax1.set_xlabel("X (voxels)")
    ax1.set_ylabel("Y (voxels)")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot(
        raw_centerline[:, 0],
        raw_centerline[:, 1],
        raw_centerline[:, 2],
        "r-",
        label="Original Centerline",
    )
    ax2.plot(
        resampled_centerline[:, 0],
        resampled_centerline[:, 1],
        resampled_centerline[:, 2],
        "b.",
        markersize=1,
        label="Resampled Centerline",
    )
    ax2.set_title("3D Centerline Path")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel("Z (mm)")
    ax2.legend()
    ax2.view_init(elev=20.0, azim=-35)

    ax3 = fig.add_subplot(1, 3, 3)
    # The CPR image's y-axis needs to be flipped to match the visual intuition
    ax3.imshow(cpr_image, cmap="gray", aspect="auto", origin="lower")
    ax3.set_title("Final Curved Planar Reformat (CPR)")
    ax3.set_xlabel("Distance along Centerline (pixels)")
    ax3.set_ylabel("Distance from Centerline (pixels)")

    plt.tight_layout()
    plt.show()
