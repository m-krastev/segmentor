"""
Utility functions for the Navigator system.
"""

import logging
import math
from math import copysign
from typing import List, Tuple, Union

import numpy as np
import skfmm
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.ndimage import binary_dilation
from skimage.draw import disk
from skimage.filters import meijering
from torch import nn

# import FastGeodis as fg

type Coords = Tuple[int, ...]
type Spacing = Tuple[float, ...]

try:
    import cupy
    from cucim.core.operations.morphology import (
        distance_transform_edt as _distance_transform_edt,
    )
    from cucim.skimage.color import label2rgb as _label2rgb
    from cucim.skimage.filters import (
        gaussian as _gaussian,
    )
    from cucim.skimage.filters import (
        median as _median,
    )
    from cucim.skimage.filters import (
        meijering as _meijering,
    )
    from cucim.skimage.morphology import binary_dilation as _binary_dilation

    def distance_transform_edt(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _distance_transform_edt(image, **kwargs).get()

    def binary_dilation(image, iterations=None, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        if iterations is None:
            image = _binary_dilation(image, **kwargs)
        else:
            for _ in range(iterations):
                image = _binary_dilation(image, **kwargs)
        return image.get()

    def meijering(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _meijering(image, **kwargs).get()

    def gaussian(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _gaussian(image, **kwargs).get()

    def median(image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(image, cupy.ndarray):
            image = cupy.array(image, dtype=cupy.float32)
        return _median(image, **kwargs).get()

    def label2rgb(labels, image, **kwargs):
        # Check if image is already a CuPy array
        if not isinstance(labels, cupy.ndarray):
            labels = cupy.asarray(labels, dtype=cupy.int32)
        if not isinstance(image, cupy.ndarray):
            image = cupy.asarray(image, dtype=cupy.float32)
        return _label2rgb(labels, image, **kwargs).get()

    # Significantly slower than skimage's implementation
    # from cucim.skimage.feature import peak_local_max as _peak_local_max
    # def peak_local_max(image, **kwargs):
    #     # Check if image is already a CuPy array
    #     if not isinstance(image, cupy.ndarray):
    #         image = cupy.array(image, dtype=cupy.float32)

    #     labels = kwargs.pop("labels", None)
    #     if labels is not None:
    #         labels = cupy.array(labels, dtype=cupy.int32)
    #     return _peak_local_max(image, labels=labels, **kwargs).get()

    logging.info("CuCIM/CuPy installed. Using GPU for graphics heavy operations.")

except ImportError:
    logging.warning("CuCIM/CuPy not installed. Please install it to enable GPU acceleration.")
    try:
        from edt import edt as distance_transform_edt

        logging.info("EDT installed. Using CPU for distance transform.")
    except ImportError:
        logging.warning(
            "EDT not installed. Please install it to enable fast distance transform: `pip install edt`."
        )


def seed_everything(seed: int = 42):
    """
    Set the seed for random number generation for reproducibility.

    Args:
        seed: Seed value to set
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def mm_to_vox(dist_mm: float, voxel_dim_mm: float) -> int:
    """Convert millimeter distance to voxel units."""
    return int(math.floor(dist_mm / voxel_dim_mm))


def vox_to_mm(dist_vox: int, voxel_dim_mm: float) -> float:
    """Convert voxel distance to millimeter units."""
    return dist_vox * voxel_dim_mm


def get_patch(
    volume: torch.Tensor,
    center_vox: Tuple[int, int, int],
    patch_size_vox: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Extract a 3D patch from a volume centered at a specific voxel.

    Args:
        volume: The source 3D volume
        center_vox: Center coordinates (z, y, x)
        patch_size_vox: Size of the patch (z, y, x)
        pad_value: Value used for padding if patch extends beyond volume bounds

    Returns:
        A patch of the specified size centered at center_vox
    """
    center_z, center_y, center_x = center_vox
    pz, py, px = patch_size_vox
    h_pz, h_py, h_px = pz // 2, py // 2, px // 2
    pad_z = max(0, h_pz - center_z) + max(0, center_z + (pz - h_pz) - volume.shape[0])
    pad_y = max(0, h_py - center_y) + max(0, center_y + (py - h_py) - volume.shape[1])
    pad_x = max(0, h_px - center_x) + max(0, center_x + (px - h_px) - volume.shape[2])
    padded_volume = volume
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        padding = (
            max(0, h_px - center_x),
            max(0, center_x + (px - h_px) - volume.shape[2]),
            max(0, h_py - center_y),
            max(0, center_y + (py - h_py) - volume.shape[1]),
            max(0, h_pz - center_z),
            max(0, center_z + (pz - h_pz) - volume.shape[0]),
        )
        padded_volume = (
            F.pad(volume.unsqueeze(0).unsqueeze(0), padding, mode="constant", value=pad_value)
            .squeeze(0)
            .squeeze(0)
        )
        center_z += max(0, h_pz - center_z)
        center_y += max(0, h_py - center_y)
        center_x += max(0, h_px - center_x)
    start_z, start_y, start_x = center_z - h_pz, center_y - h_py, center_x - h_px
    end_z, end_y, end_x = start_z + pz, start_y + py, start_x + px
    patch = padded_volume[start_z:end_z, start_y:end_y, start_x:end_x]
    if patch.shape != tuple(patch_size_vox):
        target_shape = tuple(patch_size_vox)
        new_patch = torch.full(target_shape, pad_value, dtype=patch.dtype, device=patch.device)
        src_shape = patch.shape
        copy_z = min(src_shape[0], target_shape[0])
        copy_y = min(src_shape[1], target_shape[1])
        copy_x = min(src_shape[2], target_shape[2])
        new_patch[:copy_z, :copy_y, :copy_x] = patch[:copy_z, :copy_y, :copy_x]
        patch = new_patch
    return patch


def compute_gdt(
    segmentation_mask: np.ndarray,
    start_voxel: Tuple[int, int, int],
    voxel_size: float | list[float] = 1.0,
) -> np.ndarray:
    """
    Compute a Geodesic Distance Transform from a start voxel through a segmentation mask.

    Args:
        segmentation_mask: Binary segmentation mask
        start_voxel: Starting point coordinates (z, y, x)
        voxel_size: Size of voxel in mm

    Returns:
        Distance map with geodesic distances from the start point
    """
    speed = np.ones_like(segmentation_mask, dtype=float)
    speed[segmentation_mask == 0] = 1e-5
    phi = np.ones_like(segmentation_mask, dtype=float) * np.inf
    if (
        0 <= start_voxel[0] < phi.shape[0]
        and 0 <= start_voxel[1] < phi.shape[1]
        and 0 <= start_voxel[2] < phi.shape[2]
    ):
        phi[start_voxel] = 0.0
    else:
        raise IndexError(f"Start voxel {start_voxel} outside mask bounds {phi.shape} for GDT.")
    masked_phi = np.ma.MaskedArray(phi, np.logical_not(segmentation_mask > 0))
    gdt = skfmm.travel_time(masked_phi, speed, dx=voxel_size)
    gdt = gdt.filled(-np.inf)
    return gdt


# def compute_gdt(
#     image: np.ndarray, start: Tuple[int, int, int], spacing: Tuple[float, ...] = None, device="cuda"
# ):
#     if spacing is None:
#         spacing = (1.0, 1.0, 1.0)
#     image = torch.from_numpy(image).to(device)
#     image = image.unsqueeze(0).unsqueeze(0).to(device)
#     mask = torch.ones_like(image, dtype=torch.uint8)
#     mask[0, 0, start[0], start[1], start[2]] = 0
#     v, lambd = 1e10, 1
#     geodesic_dist = fg.signed_generalised_geodesic3d(image.float(), mask.float(), spacing, v, lambd)
#     return geodesic_dist.squeeze().cpu().numpy()


def compute_wall_map(
    image: np.ndarray, sigmas: list[int] = (1, 3), black_ridges=True, **kwargs
) -> np.ndarray:
    """
    Compute a wall map highlighting vessel-like structures in the image.

    Args:
        image: Input image
        sigmas: Scale parameter for Meijering filter

    Returns:
        Wall map highlighting vessel-like structures
    """
    wall_map = meijering(image, sigmas=sigmas, black_ridges=black_ridges, mode="constant", **kwargs)
    return wall_map


def draw_path_sphere(
    cumulative_path_mask: torch.Tensor, center_vox: Tuple[int, int, int], radius_vox: int = 1
):
    """
    Draw a sphere in a mask at the specified location.

    Args:
        cumulative_path_mask: Tensor to modify
        center_vox: Center coordinates (z, y, x)
        radius_vox: Radius in voxels
    """
    z, y, x = center_vox
    shape = cumulative_path_mask.shape
    for dz in range(-radius_vox, radius_vox + 1):
        current_z = z + dz
        if 0 <= current_z < shape[0]:
            slice_radius_sq = radius_vox**2 - dz**2
            if slice_radius_sq >= 0:
                slice_radius = math.sqrt(slice_radius_sq)
                rr, cc = disk((y, x), radius=slice_radius + 0.5, shape=(shape[1], shape[2]))
                rr = np.clip(rr, 0, shape[1] - 1)
                cc = np.clip(cc, 0, shape[2] - 1)
                if rr.size > 0 and cc.size > 0:
                    cumulative_path_mask[current_z, rr, cc] = 1.0


def draw_path_sphere_2(
    cumulative_path_mask: torch.Tensor,
    voxels: tuple[tuple],
    dilation_module: nn.Module,
    zero_buffer: torch.Tensor = None,
):
    """
    Draw and dilate a sphere in a mask at the specified location.

    Args:
        cumulative_path_mask: Tensor to modify
        voxels: List of voxel coordinates to draw (should be indexable)
        dilation_module: Dilation module to use for dilation
        zero_buffer: Optional buffer for dilation
    """
    zero_buffer = (
        zero_buffer.zero_() if zero_buffer is not None else torch.zeros_like(cumulative_path_mask)
    )

    zero_buffer[voxels] = 1.0
    zero_buffer = zero_buffer.unsqueeze(0).unsqueeze(0)
    zero_buffer = dilation_module(zero_buffer)
    zero_buffer = zero_buffer.squeeze(0).squeeze(0)
    cumulative_path_mask[:] = torch.maximum(cumulative_path_mask, zero_buffer)
    return cumulative_path_mask


def find_start_end(
    duodenum_volume: np.ndarray,
    colon_volume: np.ndarray,
    small_bowel_volume: np.ndarray,
    affine: np.ndarray = None,
) -> Tuple[Coords, Coords]:
    """
    Find start and end points for small bowel navigation based on anatomical structures.

    NOTE: Built with the assumption that the duodenum is on the left side of the body (affine: -1 x -1 x 1) and that the input will be XYZ.

    Args:
        duodenum_volume: Duodenum segmentation
        colon_volume: Colon segmentation
        small_bowel_volume: Small bowel segmentation

    Returns:
        Tuple containing start and end coordinates.
    """
    import kimimaro
    import networkx as nx

    def find_duodenojejunal_flexure(
        duodenum_volume: np.ndarray, affine: np.ndarray = None
    ) -> np.ndarray:
        """Find the duodenojejunal flexure as a start point."""
        duodenum_skeleton = kimimaro.skeletonize(
            binary_dilation(duodenum_volume, iterations=5),
            teasar_params={
                "scale": 3,
                "const": 5,
                "pdrf_scale": 10000,
                "pdrf_exponent": 4,
                "soma_acceptance_threshold": 3500,
            },
            anisotropy=(1, 1, 1),
            dust_threshold=5,
            fix_branching=True,
            progress=False,
            parallel_chunk_size=100,
        )
        if 1 not in duodenum_skeleton:
            raise RuntimeError("Kimimaro failed on duodenum.")
        duodenum_skeleton = duodenum_skeleton[1]
        duodenum_graph = nx.Graph()
        vertices_xyz = {idx: loc for idx, loc in enumerate(duodenum_skeleton.vertices)}
        duodenum_graph.add_nodes_from((idx, {"location": loc}) for idx, loc in vertices_xyz.items())
        duodenum_graph.add_edges_from(duodenum_skeleton.edges)
        if duodenum_graph.number_of_nodes() == 0:
            raise RuntimeError("Duodenum skeleton graph empty.")
        ends = [node for node, degree in duodenum_graph.degree() if degree == 1]
        # NOTE: The DJF is the leftmost point (aka min in X axis). This is dependent on the affine transform used in the segmentation. So, sometimes it might correspond to the max in the X axis.
        # TODO: Fix this to be more robust to affine transforms
        nonzero = np.nonzero(duodenum_volume)
        marker = (
            nonzero[0].max(),
            nonzero[1].mean(),
            nonzero[2].mean(),
        )
        start_node = min(
            ends,
            key=lambda n: math.dist(duodenum_graph.nodes[n]["location"], marker),
        )
        return duodenum_graph.nodes[start_node]["location"]

    def find_ileocecal_junction(colon_volume: np.ndarray, affine: np.ndarray = None) -> np.ndarray:
        """Find the ileocecal junction as an end point."""
        colon_skeleton = kimimaro.skeletonize(
            binary_dilation(colon_volume, iterations=5),
            teasar_params={
                "scale": 3,
                "const": 5,
                "pdrf_scale": 10000,
                "pdrf_exponent": 4,
                "soma_acceptance_threshold": 3500,
            },
            # NOTE: Assuming isotropic spacing
            anisotropy=(1, 1, 1),
            dust_threshold=5,
            fix_branching=True,
            progress=False,
            parallel_chunk_size=100,
        )
        if 1 not in colon_skeleton:
            raise RuntimeError("Kimimaro failed on colon.")
        colon_skeleton = colon_skeleton[1]
        colon_graph = nx.Graph()
        vertices_xyz = {idx: loc for idx, loc in enumerate(colon_skeleton.vertices)}
        colon_graph.add_nodes_from((idx, {"location": loc}) for idx, loc in vertices_xyz.items())
        colon_graph.add_edges_from(colon_skeleton.edges)
        if colon_graph.number_of_nodes() == 0:
            raise RuntimeError("Colon skeleton graph empty.")
        ends = [node for node, degree in colon_graph.degree() if degree == 1]

        # NOTE: The ICJ is the rightmost point (aka max in X axis) and positioned relatively to the middle. This is dependent on the affine transform used in the segmentation. So, sometimes it might correspond to the min in the X axis (e.g. when the affine for the X axis is negative).
        # TODO: Fix this to be more robust to affine transforms
        colon_bounding_box = colon_volume.nonzero()
        marker = (
            colon_bounding_box[0].min(),
            colon_bounding_box[1].mean(),
            np.percentile(colon_bounding_box[2], 25),
        )
        ileocecal_end = min(ends, key=lambda n: math.dist(colon_graph.nodes[n]["location"], marker))
        return colon_graph.nodes[ileocecal_end]["location"]

    # Find approximate landmark points
    from scipy.spatial.distance import cdist

    raw_start_coord = find_duodenojejunal_flexure(duodenum_volume, affine=affine)
    raw_end_coord = find_ileocecal_junction(colon_volume, affine=affine)
    sb_coords_xyz = np.argwhere(small_bowel_volume > 0)
    if sb_coords_xyz.shape[0] == 0:
        raise ValueError("Small bowel segmentation empty.")

    # Map to nearest small bowel voxels
    start_distances = cdist(raw_start_coord.reshape(1, 3), sb_coords_xyz)
    nearest_start_idx = np.argmin(start_distances)
    final_start_coord = tuple(sb_coords_xyz[nearest_start_idx].astype(int))

    end_distances = cdist(raw_end_coord.reshape(1, 3), sb_coords_xyz)
    nearest_end_idx = np.argmin(end_distances)
    final_end_coord = tuple(sb_coords_xyz[nearest_end_idx].astype(int))

    return final_start_coord, final_end_coord


class ClipTransform(torch.nn.Module):
    """
    Clip the input tensor to a specified range.
    The output is then normalized to the range [0, 1].
    """
    def __init__(self, _min=-1, _max=1):
        super().__init__()
        self.min = _min
        self.max = _max

    def forward(self, x):
        return (torch.clamp(x, self.min, self.max) - self.min) / (self.max - self.min)


class BinaryDilation3D(nn.Module):
    """
    Performs 3D binary morphological dilation using convolution with a
    pre-defined structuring element kernel, leveraging broadcasting.

    The structuring element shape is chosen during initialization.

    Args:
        kernel_shape (str): The shape of the structuring element.
                            Currently supports:
                            - "star" (default): A 3x3x3 kernel with 1s at the center and
                                      face-connected neighbors (3D cross).
                            - "cube": A cubic kernel of ones. Requires kernel_size.
        kernel_size (Union[int, Tuple[int, int, int]], optional): The size of the
                      cubic or cuboid structuring element if kernel_shape is "cube".
                      If int, uses a cubic kernel (k, k, k). If tuple (kD, kH, kW),
                      uses a cuboid kernel. Required if kernel_shape is "cube".
                      Defaults to None.
    """

    def __init__(
        self, kernel_shape: str = "star", kernel_size: Union[int, Tuple[int, int, int], None] = None
    ):
        super().__init__()
        self.kernel_shape = kernel_shape.lower()
        self.kernel_size = kernel_size

        # Define the kernel based on the chosen shape.
        # Kernel shape for broadcasting with groups=C should be (1, 1, kD, kH, kW)
        # The first '1' is for the output channel dimension (will be broadcast to C)
        # The second '1' is for the in_channels/groups dimension when groups=C

        if self.kernel_shape == "star":
            # Base 3x3x3 kernel for the spatial dimensions
            kernel = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float)
            kernel[0, 0, 1, 1, 1] = 1.0  # Center
            kernel[0, 0, 0, 1, 1] = 1.0  # Down (Z-axis)
            kernel[0, 0, 2, 1, 1] = 1.0  # Up (Z-axis)
            kernel[0, 0, 1, 0, 1] = 1.0  # Left (Y-axis)
            kernel[0, 0, 1, 2, 1] = 1.0  # Right (Y-axis)
            kernel[0, 0, 1, 1, 0] = 1.0  # Backward (X-axis)
            kernel[0, 0, 1, 1, 2] = 1.0  # Forward (X-axis)
            self.padding = (1, 1, 1)  # Padding for a 3x3x3 kernel

        elif self.kernel_shape == "cube":
            if self.kernel_size is None:
                raise ValueError("kernel_size must be provided for 'cube' kernel_shape")

            if isinstance(self.kernel_size, int):
                k_d = k_h = k_w = self.kernel_size
                self.padding = (k_d - 1) // 2
            elif isinstance(self.kernel_size, tuple) and len(self.kernel_size) == 3:
                k_d, k_h, k_w = self.kernel_size
                self.padding = ((k_d - 1) // 2, (k_h - 1) // 2, (k_w - 1) // 2)
            else:
                raise ValueError(
                    "kernel_size must be an int or a tuple of three ints for 'cube' kernel_shape"
                )

            # Base kernel for the spatial dimensions
            kernel = torch.ones(1, 1, k_d, k_h, k_w, dtype=torch.float32)

        else:
            raise ValueError(
                f"Unsupported kernel_shape: {kernel_shape}. Supported shapes: 'star', 'cube'"
            )

        # Register the kernel as a buffer. It has shape (1, 1, kD, kH, kW)
        # and will be broadcast to (C, 1, kD, kH, kW) by F.conv3d when groups=C.
        self.register_buffer("dilation_kernel", kernel)

    def forward(self, binary_volume: torch.Tensor) -> torch.Tensor:
        """
        Applies 3D binary dilation to the input volume.

        Args:
            binary_volume (torch.Tensor): A binary tensor (0s and 1s, or booleans)
                                          of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: The dilated binary tensor (float 0.0 or 1.0).
        """
        # Ensure input is float for convolution operation
        if not torch.is_floating_point(binary_volume):
            binary_volume = binary_volume.float()

        # The dilation_kernel (shape 1, 1, kD, kH, kW) will be broadcast by F.conv3d
        dilated_volume_sum = F.conv3d(
            binary_volume, self.dilation_kernel, padding=self.padding, groups=binary_volume.size(1)
        )

        # Threshold the result: any sum > 0 means there was at least one '1' under the kernel
        # Convert the boolean result back to float (0.0 or 1.0)
        dilated_volume = (dilated_volume_sum > 0).float()

        return dilated_volume


def determine_do_sep_z_and_axis(
    force_separate_z: bool,
    current_spacing,
    new_spacing,
    separate_z_anisotropy_threshold: float = 3,
) -> Tuple[bool, Union[int, None]]:
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = np.argmax(current_spacing)
        else:
            axis = None
    else:
        if (max(current_spacing) / min(current_spacing)) > separate_z_anisotropy_threshold:
            do_separate_z = True
            axis = np.argmax(current_spacing)
        elif (max(new_spacing) / min(new_spacing)) > separate_z_anisotropy_threshold:
            do_separate_z = True
            axis = np.argmax(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
            axis = None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_torch_simple(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    is_seg: bool = False,
    device: torch.device = torch.device("cpu"),
    memefficient_seg_resampling: bool = False,
    mode="linear",
):
    if mode == "linear":
        if data.ndim == 4:
            torch_mode = "trilinear"
        elif data.ndim == 3:
            torch_mode = "bilinear"
        else:
            raise RuntimeError
    else:
        torch_mode = mode

    if isinstance(new_shape, np.ndarray):
        new_shape = new_shape.tolist()

    if all([i == j for i, j in zip(new_shape, data.shape[1:])]):
        return data

    new_shape = tuple(new_shape)
    with torch.no_grad():
        input_was_numpy = isinstance(data, np.ndarray)
        if input_was_numpy:
            data = torch.from_numpy(data).to(device)
        else:
            orig_device = data.device
            data = data.to(device)

        if is_seg:
            unique_values = torch.unique(data)
            result_dtype = torch.uint8 if max(unique_values) < 255 else torch.uint16
            result = torch.zeros((data.shape[0], *new_shape), dtype=result_dtype, device=device)
            if not memefficient_seg_resampling:
                # believe it or not, the implementation below is 3x as fast (at least on Liver CT and on CPU)
                # Why? Because argmax is slow. The implementation below immediately sets most locations and only lets the
                # uncertain ones be determined by argmax

                # unique_values = torch.unique(data)
                # result = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16)
                # for i, u in enumerate(unique_values):
                #     result[i] = F.interpolate((data[None] == u).float() * 1000, new_shape, mode='trilinear', antialias=False)[0]
                # result = unique_values[result.argmax(0)]

                result_tmp = torch.zeros(
                    (len(unique_values), data.shape[0], *new_shape),
                    dtype=torch.float16,
                    device=device,
                )
                scale_factor = 1000
                done_mask = torch.zeros_like(result, dtype=torch.bool, device=device)
                for i, u in enumerate(unique_values):
                    result_tmp[i] = F.interpolate(
                        (data[None] == u).float() * scale_factor,
                        new_shape,
                        mode=torch_mode,
                        antialias=False,
                    )[0]
                    mask = result_tmp[i] > (0.7 * scale_factor)
                    result[mask] = u.item()
                    done_mask |= mask
                if not torch.all(done_mask):
                    # print('resolving argmax', torch.sum(~done_mask), "voxels to go")
                    result[~done_mask] = unique_values[result_tmp[:, ~done_mask].argmax(0)].to(
                        result_dtype
                    )
            else:
                for i, u in enumerate(unique_values):
                    if u == 0:
                        pass
                    result[
                        F.interpolate(
                            (data[None] == u).float(),
                            new_shape,
                            mode=torch_mode,
                            antialias=False,
                        )[0]
                        > 0.5
                    ] = u
        else:
            result = F.interpolate(data[None].float(), new_shape, mode=torch_mode, antialias=False)[
                0
            ]
        if input_was_numpy:
            result = result.cpu().numpy()
        else:
            result = result.to(orig_device)
    return result


def resample_torch_fornnunet(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    device: torch.device = torch.device("cpu"),
    memefficient_seg_resampling: bool = False,
    force_separate_z: Union[bool, None] = None,
    separate_z_anisotropy_threshold: float = 3,
    mode="linear",
    aniso_axis_mode="nearest-exact",
):
    """
    Resample a 3D image to a new shape using PyTorch.
    Args:
        data (torch.Tensor or np.ndarray): Input data to be resampled.
        new_shape (tuple): New shape for the resampled data.
        current_spacing (tuple): Current spacing of the input data.
        new_spacing (tuple): New spacing for the resampled data.
        is_seg (bool): If True, treat the input as a segmentation map.
        device (torch.device): Device to perform the computation on.
        memefficient_seg_resampling (bool): If True, use memory-efficient segmentation resampling.
        force_separate_z (bool or None): If True, force separate z-axis resampling.
        separate_z_anisotropy_threshold (float): Threshold for anisotropy separation.
        mode (str): Interpolation mode ('linear', 'nearest-exact', etc.).
        aniso_axis_mode (str): Interpolation mode for anisotropic axis.
    """
    assert data.ndim == 4, "data must be c, x, y, z"
    new_shape = [int(i) for i in new_shape]
    orig_shape = data.shape

    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold
    )

    if not do_separate_z:
        return resample_torch_simple(data, new_shape, is_seg, device, memefficient_seg_resampling)

    was_numpy = isinstance(data, np.ndarray)
    if was_numpy:
        data = torch.from_numpy(data)

    axis_letter = "xyz"[axis]
    others_int = [i for i in range(3) if i != axis]
    others = ["xyz"[i] for i in others_int]

    # reshape by overloading c channel
    data = rearrange(data, f"c x y z -> (c {axis_letter}) {others[0]} {others[1]}")

    # reshape in-plane
    tmp_new_shape = [new_shape[i] for i in others_int]
    data = resample_torch_simple(
        data,
        tmp_new_shape,
        is_seg=is_seg,
        device=device,
        memefficient_seg_resampling=memefficient_seg_resampling,
        mode=mode,
    )
    data = rearrange(
        data,
        f"(c {axis_letter}) {others[0]} {others[1]} -> c x y z",
        **{
            axis_letter: orig_shape[axis + 1],
            others[0]: tmp_new_shape[0],
            others[1]: tmp_new_shape[1],
        },
    )
    # reshape out of plane w/ nearest
    data = resample_torch_simple(
        data,
        new_shape,
        is_seg=is_seg,
        device=device,
        memefficient_seg_resampling=memefficient_seg_resampling,
        mode=aniso_axis_mode,
    )
    return data.numpy(force=True) if was_numpy else data


def resample_uniform_res(scan, affine, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Resample the input scan to a uniform resolution.

    Parameters
    ----------
    scan : np.ndarray
        The input scan to be resampled.
    affine : np.ndarray
        The affine transformation matrix.
    kwargs : dict
        Additional arguments to be passed to the resampling function.
        This can include parameters like `force_separate_z`, `memefficient_seg_resampling`, etc.
        See the `resample_torch_fornnunet` function for more details.
    Returns
    -------
    np.ndarray
        The resampled scan.
    np.ndarray
        The updated affine transformation matrix.
    """
    # Get the spacing of the image
    spacing = np.abs(np.diag(affine)[:3])

    new_spacing = (min(spacing),) * 3

    # Let new shape be uniform spacing
    new_shape = np.asarray(scan.shape) * (spacing / min(spacing))

    new_img = resample_torch_fornnunet(
        scan[None],  # add batch dimension
        new_shape=new_shape,
        current_spacing=spacing,
        new_spacing=new_spacing,
        **kwargs,
    )

    affine = affine.copy()

    # Fix affine matrix
    affine[0, 0] = copysign(new_spacing[0], affine[0, 0])
    affine[1, 1] = copysign(new_spacing[1], affine[1, 1])
    affine[2, 2] = copysign(new_spacing[2], affine[2, 2])
    affine[0, 3] = affine[0, 3] + (new_spacing[0] - spacing[0]) * scan.shape[0] / 2
    affine[1, 3] = affine[1, 3] + (new_spacing[1] - spacing[1]) * scan.shape[1] / 2
    affine[2, 3] = affine[2, 3] + (new_spacing[2] - spacing[2]) * scan.shape[2] / 2

    return new_img[0], affine


def resample_to_spacing(
    scan: np.ndarray,
    affine: np.ndarray,
    target_spacing: tuple[float, float, float],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the input scan to a specified target spacing.

    Parameters
    ----------
    scan : np.ndarray
        The input scan to be resampled.
    affine : np.ndarray
        The affine transformation matrix.
    target_spacing : tuple[float, float, float]
        The target spacing for the resampling.
    kwargs : dict
        Additional arguments to be passed to the resampling function.
        This can include parameters like `force_separate_z`, `memefficient_seg_resampling`, etc.
        See the `resample_torch_fornnunet` function for more details.
    Returns
    -------
    np.ndarray
        The resampled scan.
    np.ndarray
        The updated affine transformation matrix.
    """
    # Get the spacing of the image
    spacing = np.abs(np.diag(affine)[:3])

    # Let new shape be uniform spacing
    new_shape = np.asarray(scan.shape) * (spacing / target_spacing)

    new_img = resample_torch_fornnunet(
        scan[None],  # add batch dimension
        new_shape=new_shape,
        current_spacing=spacing,
        new_spacing=target_spacing,
        **kwargs,
    )

    affine = affine.copy()
    # Fix affine matrix
    affine[0, 0] = copysign(target_spacing[0], affine[0, 0])
    affine[1, 1] = copysign(target_spacing[1], affine[1, 1])
    affine[2, 2] = copysign(target_spacing[2], affine[2, 2])
    affine[0, 3] = affine[0, 3] + (target_spacing[0] - spacing[0]) * scan.shape[0] / 2
    affine[1, 3] = affine[1, 3] + (target_spacing[1] - spacing[1]) * scan.shape[1] / 2
    affine[2, 3] = affine[2, 3] + (target_spacing[2] - spacing[2]) * scan.shape[2] / 2

    return new_img[0], affine


def resample_to_spacing_and_crop(
    scan: np.ndarray,
    affine: np.ndarray,
    colon: np.ndarray,
    duodenum: np.ndarray,
    small_bowel: np.ndarray,
    target_spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
    margin: int = 30,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the input scan to a specified target spacing and crop to the region of interest.

    Parameters
    ----------
    scan : np.ndarray
        The input scan to be resampled.
    affine : np.ndarray
        The affine transformation matrix.
    target_spacing : tuple[float, float, float]
        The target spacing for the resampling.
    colon : np.ndarray
        The colon segmentation mask.
    duodenum : np.ndarray
        The duodenum segmentation mask.
    small_bowel : np.ndarray
        The small bowel segmentation mask.
    Returns
    -------
    np.ndarray
        The resampled and cropped scan.
    np.ndarray
        The updated affine transformation matrix.
    """

    # Full resampling + cropping to the region of interest (bowel, colon, duodenum)
    new_img, aff = resample_to_spacing(scan, affine=affine, target_spacing=target_spacing, **kwargs)

    # Do the same for the segmentations
    colon, _ = resample_to_spacing(
        colon, affine=affine, target_spacing=target_spacing, is_seg=True, **kwargs
    )
    duodenum, _ = resample_to_spacing(
        duodenum, affine=affine, target_spacing=target_spacing, is_seg=True, **kwargs
    )
    small_bowel, _ = resample_to_spacing(
        small_bowel, affine=affine, target_spacing=target_spacing, is_seg=True, **kwargs
    )

    # Get the bounding box of the bowel
    overlayed = colon + duodenum + small_bowel
    nonzero = np.nonzero(overlayed)
    min_x, max_x = np.min(nonzero[0]), np.max(nonzero[0])
    min_y, max_y = np.min(nonzero[1]), np.max(nonzero[1])
    min_z, max_z = np.min(nonzero[2]), np.max(nonzero[2])

    # Add a margin
    min_x = max(0, min_x - margin)
    max_x = min(new_img.shape[0], max_x + margin)
    min_y = max(0, min_y - margin)
    max_y = min(new_img.shape[1], max_y + margin)
    min_z = max(0, min_z - margin)
    max_z = min(new_img.shape[2], max_z + margin)

    # Crop the image to the bounding box
    new_img = new_img[min_x:max_x, min_y:max_y, min_z:max_z]
    colon = colon[min_x:max_x, min_y:max_y, min_z:max_z]
    duodenum = duodenum[min_x:max_x, min_y:max_y, min_z:max_z]
    small_bowel = small_bowel[min_x:max_x, min_y:max_y, min_z:max_z]
    # Update the affine matrix
    aff[0, 3] = aff[0, 3] - (min_x * target_spacing[0])
    aff[1, 3] = aff[1, 3] - (min_y * target_spacing[1])
    aff[2, 3] = aff[2, 3] - (min_z * target_spacing[2])
    return new_img, aff, colon, duodenum, small_bowel
