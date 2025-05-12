"""
Utility functions for the Navigator system.
"""

import logging
import math
from typing import Tuple, Union

import numpy as np
import skfmm
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from skimage.draw import disk
from skimage.filters import meijering
from torch import nn
# import FastGeodis as fg

type Coords = Tuple[int, int, int]
type Spacing = Tuple[float, float, float]

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
    duodenum_volume: np.ndarray, colon_volume: np.ndarray, small_bowel_volume: np.ndarray
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Find start and end points for small bowel navigation based on anatomical structures.

    Args:
        duodenum_volume: Duodenum segmentation
        colon_volume: Colon segmentation
        small_bowel_volume: Small bowel segmentation

    Returns:
        Tuple containing start and end coordinates (each as z,y,x coordinates)
    """

    def find_duodenojejunal_flexure(start_volume: np.ndarray) -> np.ndarray:
        """Find the duodenojejunal flexure as a start point."""
        import kimimaro
        import networkx as nx

        start_volume_xyz = np.transpose(start_volume, (2, 1, 0))
        duodenum_skeleton = kimimaro.skeletonize(
            binary_dilation(start_volume_xyz, iterations=5),
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
        vertices_zyx = {idx: loc[::-1] for idx, loc in enumerate(duodenum_skeleton.vertices)}
        duodenum_graph.add_nodes_from((idx, {"location": loc}) for idx, loc in vertices_zyx.items())
        duodenum_graph.add_edges_from(duodenum_skeleton.edges)
        if duodenum_graph.number_of_nodes() == 0:
            raise RuntimeError("Duodenum skeleton graph empty.")
        ends = [node for node, degree in duodenum_graph.degree() if degree == 1]
        if not ends:
            start_node = max(
                duodenum_graph.nodes, key=lambda n: duodenum_graph.nodes[n]["location"][0]
            )
        else:
            start_node = max(ends, key=lambda n: duodenum_graph.nodes[n]["location"][0])
        return duodenum_graph.nodes[start_node]["location"]

    def find_ileocecal_junction(end_volume: np.ndarray) -> np.ndarray:
        """Find the ileocecal junction as an end point."""
        import kimimaro
        import networkx as nx

        end_volume_xyz = np.transpose(end_volume, (2, 1, 0))
        colon_skeleton = kimimaro.skeletonize(
            binary_dilation(end_volume_xyz, iterations=5),
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
        if 1 not in colon_skeleton:
            raise RuntimeError("Kimimaro failed on colon.")
        colon_skeleton = colon_skeleton[1]
        colon_graph = nx.Graph()
        vertices_zyx = {idx: loc[::-1] for idx, loc in enumerate(colon_skeleton.vertices)}
        colon_graph.add_nodes_from((idx, {"location": loc}) for idx, loc in vertices_zyx.items())
        colon_graph.add_edges_from(colon_skeleton.edges)
        if colon_graph.number_of_nodes() == 0:
            raise RuntimeError("Colon skeleton graph empty.")
        largest_cc = max(nx.connected_components(colon_graph), key=len, default=None)
        if largest_cc is None:
            raise RuntimeError("Colon skeleton graph no CCs.")
        colon_subgraph = colon_graph.subgraph(largest_cc).copy()
        ends = [node for node, degree in colon_subgraph.degree() if degree == 1]
        if not ends:
            print("Warning: Colon skeleton no ends. Using lowest/farthest.")
            rectal_end = max(
                colon_subgraph.nodes, key=lambda n: colon_subgraph.nodes[n]["location"][0]
            )
            ileocecal_end = max(
                colon_subgraph.nodes,
                key=lambda n: np.linalg.norm(
                    np.array(colon_subgraph.nodes[n]["location"])
                    - np.array(colon_subgraph.nodes[rectal_end]["location"])
                ),
            )
        else:
            rectal_end = max(ends, key=lambda n: colon_subgraph.nodes[n]["location"][0])
            ends.remove(rectal_end)
            if not ends:
                print("Warning: Only rectal end found. Using farthest node.")
                ileocecal_end = max(
                    colon_subgraph.nodes,
                    key=lambda n: nx.shortest_path_length(colon_subgraph, rectal_end, n)
                    if nx.has_path(colon_subgraph, rectal_end, n)
                    else -1,
                )
            elif len(ends) > 1:
                ileocecal_end = max(
                    ends, key=lambda n: nx.shortest_path_length(colon_subgraph, rectal_end, n)
                )
            else:
                ileocecal_end = ends[0]
        return colon_subgraph.nodes[ileocecal_end]["location"]

    # Find approximate landmark points
    from scipy.spatial.distance import cdist

    raw_start_coord_zyx = find_duodenojejunal_flexure(duodenum_volume)
    raw_end_coord_zyx = find_ileocecal_junction(colon_volume)
    sb_coords_zyx = np.argwhere(small_bowel_volume > 0)
    if sb_coords_zyx.shape[0] == 0:
        raise ValueError("Small bowel segmentation empty.")

    # Map to nearest small bowel voxels
    start_distances = cdist(raw_start_coord_zyx.reshape(1, 3), sb_coords_zyx)
    nearest_start_idx = np.argmin(start_distances)
    final_start_coord = tuple(sb_coords_zyx[nearest_start_idx].astype(int))

    end_distances = cdist(raw_end_coord_zyx.reshape(1, 3), sb_coords_zyx)
    nearest_end_idx = np.argmin(end_distances)
    final_end_coord = tuple(sb_coords_zyx[nearest_end_idx].astype(int))

    return final_start_coord, final_end_coord


class ClipTransform(torch.nn.Module):
    def __init__(self, _min=-1, _max=1):
        super().__init__()
        self.min = _min
        self.max = _max

    def forward(self, x):
        return (torch.clamp(x, self.min, self.max) - self.min) / (self.max - self.min)

# class ClipTransform(torch.nn.Module):
#     def __init__(self, _min=-1, _max=1):
#         super().__init__()
#         self.min = _min
#         self.max = _max

#     def forward(self, x):
#         x = x.where(x < self.max, self.min).where(x > self.min, self.min)
#         return (x - self.min) / (self.max - self.min)


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
