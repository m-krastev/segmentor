"""
Utility functions for the Navigator system.
"""

import numpy as np
import torch
import torch.nn.functional as F
import math
from skimage.filters import meijering
from skimage.draw import disk
import skfmm
from typing import Tuple


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
    segmentation_mask: np.ndarray, start_voxel: Tuple[int, int, int], voxel_size: float
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
    gdt = gdt.filled(np.inf)
    return gdt


def compute_wall_map(image: np.ndarray, sigmas: list[int] = (1, 3)) -> np.ndarray:
    """
    Compute a wall map highlighting vessel-like structures in the image.
    
    Args:
        image: Input image
        sigmas: Scale parameter for Meijering filter
        
    Returns:
        Wall map highlighting vessel-like structures
    """
    wall_map = meijering(
        image, sigmas=sigmas, black_ridges=False, mode="constant", cval=0
    )
    # max_val = np.max(wall_map)
    # if max_val > 1e-6:
    #     wall_map = wall_map / max_val
    return wall_map


def draw_path_sphere(
    cumulative_path_mask: torch.Tensor, center_vox: Tuple[int, int, int], radius_vox: int
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
    radius_vox = max(0, radius_vox)
    for dz in range(-radius_vox, radius_vox + 1):
        current_z = z + dz
        if 0 <= current_z < shape[0]:
            slice_radius_sq = radius_vox**2 - dz**2
            if slice_radius_sq >= 0:
                slice_radius = int(math.sqrt(slice_radius_sq))
                try:
                    rr, cc = disk((y, x), radius=slice_radius + 0.5, shape=(shape[1], shape[2]))
                    rr = np.clip(rr, 0, shape[1] - 1)
                    cc = np.clip(cc, 0, shape[2] - 1)
                    if rr.size > 0 and cc.size > 0:
                        cumulative_path_mask[current_z, rr, cc] = 1.0
                except Exception:
                    # Less verbose error handling
                    if 0 <= y < shape[1] and 0 <= x < shape[2]:
                        cumulative_path_mask[current_z, y, x] = 1.0


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
    print("Finding start/end points using skeletonization...")

    def find_duodenojejunal_flexure(start_volume: np.ndarray) -> np.ndarray:
        """Find the duodenojejunal flexure as a start point."""
        import kimimaro
        import networkx as nx
        from scipy.ndimage import binary_dilation

        print("  Skeletonizing duodenum...")
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
            print("Warning: Duodenum skeleton no ends. Using lowest node.")
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
        from scipy.ndimage import binary_dilation

        print("  Skeletonizing colon...")
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
    print("  Mapping skeleton points to nearest small bowel voxel...")
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
    
    print(f"  Mapped DJ flexure to SB voxel: {final_start_coord}")
    print(f"  Mapped IC junction to SB voxel: {final_end_coord}")
    print("Start/end point finding finished.")
    return final_start_coord, final_end_coord
