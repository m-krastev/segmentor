import numpy as np
from math import dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import torch.optim as optim
import nibabel as nib
from skimage.filters import meijering
from skimage.draw import line_nd, disk
import skfmm
import math
import os
import warnings
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List, Union
from tqdm import tqdm

try:
    import kimimaro
    import networkx as nx
    # from scipy.ndimage import binary_dilation, grey_dilation
    from navigator.utils import binary_dilation
    from scipy.spatial.distance import cdist
except ImportError as e:
    print(f"Error importing dependencies for start/end finding: {e}")
    print("Please install them: pip install kimimaro networkx scipy")
    exit()
# --------------------

import wandb  # Import wandb


# --- Configuration Dataclass (Added Wandb & Evaluation Options) ---
@dataclass
class Config:
    # --- Input/Output ---
    nifti_path: str = "dummy_data/dummy_image.nii.gz"
    seg_path: str = "dummy_data/dummy_seg.nii.gz"
    duodenum_seg_path: str = "dummy_data/duodenum_seg.nii.gz"
    colon_seg_path: str = "dummy_data/colon_seg.nii.gz"
    gt_path_path: Optional[str] = None
    save_path: str = "ppo_small_bowel_tracker.pth"

    # --- Wandb Logging ---
    track_wandb: bool = True  # Flag to enable/disable wandb
    wandb_project_name: str = "SmallBowelPathTracking"
    wandb_entity: Optional[str] = (
        None  # Your wandb username or team name (optional)
    )
    wandb_run_name: Optional[str] = (
        None  # Optional run name, defaults to auto-generated
    )

    # --- Environment Hyperparameters ---
    voxel_size_mm: float = 1.5
    patch_size_mm: int = 60
    max_step_displacement_mm: float = 9.0
    max_episode_steps: int = 1024
    cumulative_path_radius_mm: float = 4
    wall_map_sigma: float = 1.0

    # --- Reward Hyperparameters ---
    r_wall: float = 4.0
    r_val2: float = 6.0
    r_final: float = 100.0
    use_immediate_gdt_reward: bool = False
    wall_penalty_scale: float = 1.5

    # --- PPO Hyperparameters ---
    learning_rate: float = 1e-5
    total_timesteps: int = 10_000_000
    n_steps: int = 2048
    batch_size: int = 128
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- Training/Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Evaluation Parameters (NEW) ---
    eval_mode: bool = False  # If True, run evaluation instead of training
    load_path: Optional[str] = (
        None  # Path to the .pth file for evaluation (required if eval_mode=True)
    )
    eval_output_path: str = (
        "evaluated_path.npy"  # Path to save the resulting trajectory NPY file
    )
    eval_deterministic: bool = (
        True  # Use deterministic actions (mean) during evaluation
    )

    # --- Derived Parameters ---
    gdt_cell_length: float = field(init=False)
    max_step_vox: int = field(init=False)
    patch_size_vox: Tuple[int, int, int] = field(init=False)
    cumulative_path_radius_vox: int = field(init=False)
    gdt_max_increase_theta: float = field(init=False)

    def __post_init__(self):
        self.gdt_cell_length = self.voxel_size_mm
        self.max_step_vox = mm_to_vox(
            self.max_step_displacement_mm, self.voxel_size_mm
        )
        patch_vox_dim = mm_to_vox(self.patch_size_mm, self.voxel_size_mm)
        self.patch_size_vox = (patch_vox_dim,) * 3
        self.cumulative_path_radius_vox = mm_to_vox(
            self.cumulative_path_radius_mm, self.voxel_size_mm
        )
        self.gdt_max_increase_theta = max(0.0, np.sqrt(3 * self.max_step_vox **2))


# --- Argument Parsing (Updated for Wandb & Evaluation) ---
def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Deep Reinforcement Learning for Small Bowel Path Tracking"
    )
    default_config = Config()

    # Add arguments dynamically from Config fields
    for field_name, field_info in Config.__dataclass_fields__.items():
        if field_info.init is False:  # Skip derived fields
            continue

        field_type = field_info.type
        default_val = getattr(default_config, field_name)
        arg_type = field_type
        required = False
        arg_name = f"--{field_name.replace('_', '-')}"

        # Make duodenum/colon paths required if not using dummy defaults
        if (
            field_name in ["duodenum_seg_path", "colon_seg_path"]
            and "dummy_data" in default_val
        ):
            pass  # Keep dummy defaults optional
        elif field_name in ["duodenum_seg_path", "colon_seg_path"]:
            required = True  # Make required if default isn't dummy

        # Handle Optional type hint for argparse type
        if (
            hasattr(field_type, "__origin__")
            and field_type.__origin__ is Union
            and type(None) in field_type.__args__
        ):
            # Special case for wandb_entity, wandb_run_name, load_path which can be None
            if field_name in ["wandb_entity", "wandb_run_name", "load_path"]:
                arg_type = str  # Expect string or nothing
            else:
                arg_type = field_type.__args__[0]  # Get the non-None type

        if arg_type == bool:
            # Use BooleanOptionalAction for flags like --eval-mode / --no-eval-mode
            parser.add_argument(
                arg_name,
                action=argparse.BooleanOptionalAction,
                default=default_val,
                help=f"{field_name} (default: {default_val})",
            )
        else:
            # Handle load_path being conditionally required later
            is_conditionally_required = field_name == "load_path"
            parser.add_argument(
                arg_name,
                type=arg_type,
                default=default_val,
                required=required
                and not is_conditionally_required,  # Standard requirement check
                help=f"{field_name} (default: {default_val})",
            )

    args = parser.parse_args()

    # --- Conditional Requirement Check ---
    if args.eval_mode and args.load_path is None:
        parser.error("--load-path is required when --eval-mode is set.")

    config_dict = {
        k: v for k, v in vars(args).items() if k in Config.__annotations__
    }
    config = Config(**config_dict)
    return config


# --- Helper Functions (Unchanged) ---
def mm_to_vox(dist_mm: float, voxel_dim_mm: float) -> int:
    return int(math.floor(dist_mm / voxel_dim_mm))


def vox_to_mm(dist_vox: int, voxel_dim_mm: float) -> float:
    return dist_vox * voxel_dim_mm


def get_patch(
    volume: torch.Tensor,
    center_vox: Tuple[int, int, int],
    patch_size_vox: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> torch.Tensor:
    center_z, center_y, center_x = center_vox
    pz, py, px = patch_size_vox
    h_pz, h_py, h_px = pz // 2, py // 2, px // 2
    pad_z = max(0, h_pz - center_z) + max(
        0, center_z + (pz - h_pz) - volume.shape[0]
    )
    pad_y = max(0, h_py - center_y) + max(
        0, center_y + (py - h_py) - volume.shape[1]
    )
    pad_x = max(0, h_px - center_x) + max(
        0, center_x + (px - h_px) - volume.shape[2]
    )
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
            F.pad(
                volume.unsqueeze(0).unsqueeze(0),
                padding,
                mode="constant",
                value=pad_value,
            )
            .squeeze(0)
            .squeeze(0)
        )
        center_z += max(0, h_pz - center_z)
        center_y += max(0, h_py - center_y)
        center_x += max(0, h_px - center_x)
    start_z, start_y, start_x = (
        center_z - h_pz,
        center_y - h_py,
        center_x - h_px,
    )
    end_z, end_y, end_x = start_z + pz, start_y + py, start_x + px
    patch = padded_volume[start_z:end_z, start_y:end_y, start_x:end_x]
    if patch.shape != tuple(patch_size_vox):
        target_shape = tuple(patch_size_vox)
        new_patch = torch.full(
            target_shape, pad_value, dtype=patch.dtype, device=patch.device
        )
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
    voxel_size: float,
) -> np.ndarray:
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
        raise IndexError(
            f"Start voxel {start_voxel} outside mask bounds {phi.shape} for GDT."
        )
    masked_phi = np.ma.MaskedArray(phi, np.logical_not(segmentation_mask > 0))
    gdt = skfmm.travel_time(masked_phi, speed, dx=voxel_size)
    gdt = gdt.filled(-np.inf)
    return gdt


def compute_wall_map(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    sigma_int = max(1, int(sigma))
    wall_map = meijering(
        image,
        sigmas=[1, 3],  # Adjusted sigmas based on previous context if needed
        black_ridges=True,
        mode="constant",
        cval=0,
    )
    max_val = np.max(wall_map)
    if max_val > 1e-6:
        wall_map = wall_map / max_val
    return wall_map


def draw_path_sphere(
    cumulative_path_mask: torch.Tensor,
    center_vox: Tuple[int, int, int],
    radius_vox: int,
):
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
                    rr, cc = disk(
                        (y, x),
                        radius=slice_radius + 0.5,
                        shape=(shape[1], shape[2]),
                    )
                    rr = np.clip(rr, 0, shape[1] - 1)
                    cc = np.clip(cc, 0, shape[2] - 1)
                    if rr.size > 0 and cc.size > 0:
                        cumulative_path_mask[current_z, rr, cc] = 1.0
                except Exception as e:
                    if 0 <= y < shape[1] and 0 <= x < shape[2]:
                        cumulative_path_mask[current_z, y, x] = 1.0

# def draw_path_sphere(
#         cumulative_path_mask,
#         start,
#         end,
#         n_iter = 1,
# ):    
#     line = line_nd(start, end, endpoint=True)
#     cumulative_path_mask[line] = 1
#     cumulative_path_mask = binary_dilation(cumulative_path_mask, n_iter)
#     return cumulative_path_mask



# --- Start/End Finding Logic (Unchanged) ---
def find_start_end(
    duodenum_volume: np.ndarray,
    colon_volume: np.ndarray,
    small_bowel_volume: np.ndarray,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    print("Finding start/end points using skeletonization...")

    def find_duodenojejunal_flexure(start_volume: np.ndarray) -> np.ndarray:
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
        vertices_zyx = {
            idx: loc[::-1] for idx, loc in enumerate(duodenum_skeleton.vertices)
        }
        duodenum_graph.add_nodes_from(
            (idx, {"location": loc}) for idx, loc in vertices_zyx.items()
        )
        duodenum_graph.add_edges_from(duodenum_skeleton.edges)
        if duodenum_graph.number_of_nodes() == 0:
            raise RuntimeError("Duodenum skeleton graph empty.")
        ends = [node for node, degree in duodenum_graph.degree() if degree == 1]
        if not ends:
            print("Warning: Duodenum skeleton no ends. Using lowest node.")
            start_node = max(
                duodenum_graph.nodes,
                key=lambda n: duodenum_graph.nodes[n]["location"][0],
            )
        else:
            start_node = max(
                ends, key=lambda n: duodenum_graph.nodes[n]["location"][0]
            )
        return duodenum_graph.nodes[start_node]["location"]

    def find_ileocecal_junction(end_volume: np.ndarray) -> np.ndarray:
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
        vertices_zyx = {
            idx: loc[::-1] for idx, loc in enumerate(colon_skeleton.vertices)
        }
        colon_graph.add_nodes_from(
            (idx, {"location": loc}) for idx, loc in vertices_zyx.items()
        )
        colon_graph.add_edges_from(colon_skeleton.edges)
        if colon_graph.number_of_nodes() == 0:
            raise RuntimeError("Colon skeleton graph empty.")
        largest_cc = max(
            nx.connected_components(colon_graph), key=len, default=None
        )
        if largest_cc is None:
            raise RuntimeError("Colon skeleton graph no CCs.")
        colon_subgraph = colon_graph.subgraph(largest_cc).copy()
        ends = [node for node, degree in colon_subgraph.degree() if degree == 1]
        if not ends:
            print("Warning: Colon skeleton no ends. Using lowest/farthest.")
            rectal_end = max(
                colon_subgraph.nodes,
                key=lambda n: colon_subgraph.nodes[n]["location"][0],
            )
            ileocecal_end = max(
                colon_subgraph.nodes,
                key=lambda n: np.linalg.norm(
                    np.array(colon_subgraph.nodes[n]["location"])
                    - np.array(colon_subgraph.nodes[rectal_end]["location"])
                ),
            )
        else:
            rectal_end = max(
                ends, key=lambda n: colon_subgraph.nodes[n]["location"][0]
            )
            ends.remove(rectal_end)
            if not ends:
                print("Warning: Only rectal end found. Using farthest node.")
                ileocecal_end = max(
                    colon_subgraph.nodes,
                    key=lambda n: nx.shortest_path_length(
                        colon_subgraph, rectal_end, n
                    )
                    if nx.has_path(colon_subgraph, rectal_end, n)
                    else -1,
                )
            elif len(ends) > 1:
                ileocecal_end = max(
                    ends,
                    key=lambda n: nx.shortest_path_length(
                        colon_subgraph, rectal_end, n
                    ),
                )
            else:
                ileocecal_end = ends[0]
        return colon_subgraph.nodes[ileocecal_end]["location"]

    raw_start_coord_zyx = find_duodenojejunal_flexure(duodenum_volume)
    raw_end_coord_zyx = find_ileocecal_junction(colon_volume)
    print("  Mapping skeleton points to nearest small bowel voxel...")
    sb_coords_zyx = np.argwhere(small_bowel_volume > 0)
    if sb_coords_zyx.shape[0] == 0:
        raise ValueError("Small bowel segmentation empty.")
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


# --- Environment Definition (Unchanged) ---
class SmallBowelEnv:
    def __init__(
        self, config: Config, start_end_coords: Dict[str, Tuple[int, int, int]]
    ):
        self.config = config
        self.start_coord = start_end_coords["start"]
        self.end_coord = start_end_coords["end"]
        self.device = torch.device(config.device)
        self.image_nii = nib.load(config.nifti_path)
        self.seg_nii = nib.load(config.seg_path)
        nii_voxel_size = self.image_nii.header.get_zooms()[:3]
        if not np.allclose(nii_voxel_size, (config.voxel_size_mm,) * 3):
            warnings.warn(
                f"NIfTI voxel size {nii_voxel_size} != configured {config.voxel_size_mm}.",
                UserWarning,
            )
        image_np = self.image_nii.get_fdata().astype(np.float32)
        self.seg_np = (self.seg_nii.get_fdata() > 0.5).astype(np.uint8)
        img_min, img_max = np.min(image_np), np.max(image_np)
        if img_max > img_min:
            self.image_np = (image_np - img_min) / (img_max - img_min)
        else:
            self.image_np = np.zeros_like(image_np)
        self.wall_map_np = compute_wall_map(
            self.image_np, sigma=config.wall_map_sigma
        )
        self.gt_path_available = False
        self.gt_path_voxels = None
        if config.gt_path_path is not None and os.path.exists(
            config.gt_path_path
        ):
            try:
                self.gt_path_voxels = np.load(config.gt_path_path).astype(int)
                self.gt_path_available = True
                self.gt_path_vol_np = np.zeros_like(
                    self.image_np, dtype=np.float32
                )
                valid_indices = (
                    (self.gt_path_voxels[:, 0] >= 0)
                    & (self.gt_path_voxels[:, 0] < self.image_np.shape[0])
                    & (self.gt_path_voxels[:, 1] >= 0)
                    & (self.gt_path_voxels[:, 1] < self.image_np.shape[1])
                    & (self.gt_path_voxels[:, 2] >= 0)
                    & (self.gt_path_voxels[:, 2] < self.image_np.shape[2])
                )
                valid_gt_path = self.gt_path_voxels[valid_indices]
                if valid_gt_path.shape[0] > 0:
                    self.gt_path_vol_np[
                        valid_gt_path[:, 0],
                        valid_gt_path[:, 1],
                        valid_gt_path[:, 2],
                    ] = 1.0
                self.gt_path_vol = torch.from_numpy(self.gt_path_vol_np).to(
                    self.device
                )
            except Exception as e:
                print(
                    f"Warning: Could not load GT path {config.gt_path_path}: {e}"
                )
                self.gt_path_available = False
                self.gt_path_vol = torch.zeros_like(
                    torch.from_numpy(self.image_np), device=self.device
                )
        else:
            self.gt_path_vol = torch.zeros_like(
                torch.from_numpy(self.image_np), device=self.device
            )
        self.image = torch.from_numpy(self.image_np).to(self.device)
        self.seg = torch.from_numpy(self.seg_np).to(self.device)
        self.wall_map = torch.from_numpy(self.wall_map_np).to(self.device)
        self.current_pos_vox: Tuple[int, int, int] = self.start_coord

        self.current_step: int = 0
        self.cumulative_path_mask: Optional[torch.Tensor] = None
        self.max_gdt_achieved: float = 0.0
        self.tracking_path_history: List[Tuple[int, int, int]] = []
        self.current_gdt_val: float = 0.0
        self.gdt = torch.from_numpy(
            compute_gdt(
                self.seg_np, self.start_coord, self.config.gdt_cell_length
            )
        ).to(self.device)

    def _get_state_patches(self) -> Dict[str, torch.Tensor]:
        center = self.current_pos_vox
        ps = self.config.patch_size_vox
        img_patch = get_patch(self.image, center, ps)
        wall_patch = get_patch(self.wall_map, center, ps)
        cum_path_mask = (
            self.cumulative_path_mask
            if self.cumulative_path_mask is not None
            else torch.zeros_like(self.image)
        )
        cum_path_patch = get_patch(cum_path_mask, center, ps)
        gt_path_patch = get_patch(self.gt_path_vol, center, ps)
        actor_state = torch.stack(
            [img_patch, wall_patch, cum_path_patch], dim=0
        )
        critic_state = torch.stack(
            [img_patch, wall_patch, cum_path_patch, gt_path_patch], dim=0
        )
        return {"actor": actor_state, "critic": critic_state}

    def _is_valid_pos(self, pos_vox: Tuple[int, int, int]) -> bool:
        s = self.image.shape
        return (
            0 <= pos_vox[0] < s[0]
            and 0 <= pos_vox[1] < s[1]
            and 0 <= pos_vox[2] < s[2]
        )

    def _get_final_coverage(self) -> float:
        if self.cumulative_path_mask is None or self.seg is None:
            return 0.0
        seg_volume = torch.sum(self.seg).item()
        if seg_volume < 1e-6:
            return 0.0
        intersection = torch.sum(self.cumulative_path_mask * self.seg).item()
        return intersection / seg_volume

    def reset(self, start_choice: str = "start") -> Dict[str, torch.Tensor]:
        self.current_step = 0
        if start_choice == "end":
            self.current_pos_vox = self.end_coord
        else:
            self.current_pos_vox = self.start_coord

        if (
            not self._is_valid_pos(self.current_pos_vox)
            or self.seg[self.current_pos_vox].item() == 0
        ):
            print(
                f"Warning: Start pos {self.current_pos_vox} invalid. Searching..."
            )
            found_valid_start = False
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        new_pos = (
                            self.current_pos_vox[0] + dz,
                            self.current_pos_vox[1] + dy,
                            self.current_pos_vox[2] + dx,
                        )
                        if (
                            self._is_valid_pos(new_pos)
                            and self.seg[new_pos].item() > 0
                        ):
                            self.current_pos_vox = new_pos
                            found_valid_start = True
                            break
                    if found_valid_start:
                        break
                if found_valid_start:
                    break
            if not found_valid_start:
                valid_voxels = torch.nonzero(self.seg > 0)
                if len(valid_voxels) > 0:
                    rand_idx = torch.randint(0, len(valid_voxels), (1,)).item()
                    self.current_pos_vox = tuple(
                        valid_voxels[rand_idx].tolist()
                    )
                    print(f"Fallback start: {self.current_pos_vox}")
                else:
                    raise ValueError("Cannot start: Seg mask empty.")

        self.cumulative_path_mask = torch.zeros_like(
            self.image, dtype=torch.float32, device=self.device
        )
        draw_path_sphere(
            self.cumulative_path_mask,
            self.current_pos_vox,
            self.config.cumulative_path_radius_vox,
        )
        self.tracking_path_history = [self.current_pos_vox]

        if self.gdt is not None and self._is_valid_pos(self.current_pos_vox):
            gdt_tensor = self.gdt[self.current_pos_vox]
            self.current_gdt_val = (
                gdt_tensor.item() if torch.isfinite(gdt_tensor) else 0.0
            )
        else:
            self.current_gdt_val = 0.0
        self.max_gdt_achieved = self.current_gdt_val
        return self._get_state_patches()

    def get_tracking_history(self) -> np.ndarray:
        return np.array(self.tracking_path_history)

    def _calculate_reward(
        self,
        action_vox: Tuple[int, int, int],
        next_pos_vox: Tuple[int, int, int],
    ) -> float:
        """Calculate the reward for the current step. Assumes tensors are valid."""

        rt = torch.tensor(0.0, device=self.device)
        # Set of voxels S on the line segment
        S = line_nd(self.current_pos_vox, next_pos_vox, endpoint=True)

        # --- 1. Zero movement or goes out of the image penalty ---
        if not any(action_vox) or not self._is_valid_pos(next_pos_vox) or not self.gt_path_vol[next_pos_vox]:
            rt -= self.config.r_wall
            return rt, S

        # --- 2. GDT-based reward ---
        next_gdt_val = self.gdt[next_pos_vox]
        # If there is some progress
        if self.config.use_immediate_gdt_reward:
            rt += max(
                0.0,
                self.config.r_val2
                * (next_gdt_val - self.current_gdt_val)
                / self.config.gdt_max_increase_theta,
            )
        elif next_gdt_val > self.max_gdt_achieved:
            delta = next_gdt_val - self.max_gdt_achieved
            # Penalty if too large
            if delta > self.config.gdt_max_increase_theta:
                rt -= self.config.r_val2
            else:
                rt += (
                    self.config.r_val2
                    * delta
                    / self.config.gdt_max_increase_theta
                )
            self.max_gdt_achieved = next_gdt_val

        # 2.5 Peaks-based reward
        # rt += self.reward_map[S].sum() * self.config.r_peaks
        # Discard the reward for visited nodes
        # self.reward_map[S] = 0

        # --- 3. Wall-based penalty ---
        rt -= self.config.r_val2 * self.wall_map[S].mean()

        # --- 4. Revisiting penalty ---
        rt -= self.config.r_wall * self.cumulative_path_mask[S].sum().bool()

        # --- 5. Out-of-segmentation penalty ---
        # rt = -self.config.r_wall

        return rt, S

    def step(
        self, action_vox_delta: Tuple[int, int, int]
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        if self.cumulative_path_mask is None:
            raise RuntimeError("Env not reset.")
        self.current_step += 1
        current_z, current_y, current_x = self.current_pos_vox
        dz, dy, dx = action_vox_delta
        next_pos_vox = (current_z + dz, current_y + dy, current_x + dx)
        reward, S = self._calculate_reward(action_vox_delta, next_pos_vox)
        next_gdt_val = np.inf
        if self.gdt is not None and self._is_valid_pos(next_pos_vox):
            next_gdt_tensor = self.gdt[next_pos_vox]
            if torch.isfinite(next_gdt_tensor):
                next_gdt_val = next_gdt_tensor.item()
                if not self.config.use_immediate_gdt_reward:
                    self.max_gdt_achieved = max(
                        self.max_gdt_achieved, next_gdt_val
                    )
                self.current_gdt_val = next_gdt_val
            else:
                self.current_gdt_val = 0.0
        else:
            self.current_gdt_val = 0.0
        is_next_pos_valid_seg = (
            self._is_valid_pos(next_pos_vox)
            # and self.seg[next_pos_vox].item() > 0
        )
        if is_next_pos_valid_seg:
            for pos in zip(*S):
                draw_path_sphere(
                self.cumulative_path_mask,
                pos,
                self.config.cumulative_path_radius_vox,
            )
            self.tracking_path_history.append(next_pos_vox)
        self.current_pos_vox = next_pos_vox
        done = False
        termination_reason = ""
        if self.current_step >= self.config.max_episode_steps:
            done = True
            termination_reason = "max_steps"
        elif not is_next_pos_valid_seg:
            done = True
            termination_reason = "out_of_bounds_or_segmentation"
        elif (
            dist(self.current_pos_vox, self.end_coord) < self.config.cumulative_path_radius_vox
        ):
            done = True
            termination_reason = "reached_end"
        info = {
            "current_pos": self.current_pos_vox,
            "termination": termination_reason,
            "steps": self.current_step,
        }
        if done:
            coverage = self._get_final_coverage()
            target_reached = termination_reason == "reached_end"
            final_reward_adjustment = (
                (coverage * self.config.r_final)
                if target_reached
                else ((coverage - 1.0) * self.config.r_final)
            )
            reward += final_reward_adjustment
            info["episode_coverage"] = coverage
        next_state_patches = self._get_state_patches()
        return next_state_patches, reward, done, info


# --- Network Definitions (Unchanged) ---
class ActorNetwork(nn.Module):
    def __init__(self, config: Config, input_channels=3):
        super().__init__()
        self.config = config
        self.patch_size_vox = config.patch_size_vox
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(8, 64)
        self.pool4 = nn.MaxPool3d(2)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *self.patch_size_vox)
            pooled_output = self.pool4(
                self.pool3(self.pool2(self.pool1(dummy_input)))
            )
            final_channels = 64
            self.flattened_size = final_channels * np.prod(
                pooled_output.shape[-3:]
            )
        if self.flattened_size <= 0:
            raise ValueError(f"Actor flat size <= 0 ({self.flattened_size}).")
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.gn_fc1 = nn.GroupNorm(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.gn_fc2 = nn.GroupNorm(8, 64)
        self.fc_out = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))
        x = self.pool4(F.relu(self.gn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        if x.shape[1] != self.flattened_size:
            raise ValueError(
                f"Runtime Actor flat size mismatch. Expected {self.flattened_size}, got {x.shape[1]}."
            )
        x = F.relu(self.gn_fc1(self.fc1(x)))
        x = F.relu(self.gn_fc2(self.fc2(x)))
        ab_params = self.fc_out(x)
        alpha_beta = F.softplus(ab_params) + 1.0
        return alpha_beta

    def get_action_dist(self, obs_actor: torch.Tensor) -> Beta:
        alpha_beta = self.forward(obs_actor)
        alpha_beta_pairs = alpha_beta.view(-1, 3, 2)
        alphas = alpha_beta_pairs[..., 0]
        betas = alpha_beta_pairs[..., 1]
        dist = Beta(alphas, betas)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, config: Config, input_channels=4):
        super().__init__()
        self.config = config
        self.patch_size_vox = config.patch_size_vox
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 64)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(8, 64)
        self.pool4 = nn.MaxPool3d(2)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *self.patch_size_vox)
            pooled_output = self.pool4(
                self.pool3(self.pool2(self.pool1(dummy_input)))
            )
            final_channels = 64
            self.flattened_size = final_channels * np.prod(
                pooled_output.shape[-3:]
            )
        if self.flattened_size <= 0:
            raise ValueError(f"Critic flat size <= 0 ({self.flattened_size}).")
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.gn_fc1 = nn.GroupNorm(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.gn_fc2 = nn.GroupNorm(8, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))
        x = self.pool4(F.relu(self.gn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        if x.shape[1] != self.flattened_size:
            raise ValueError(
                f"Runtime Critic flat size mismatch. Expected {self.flattened_size}, got {x.shape[1]}."
            )
        x = F.relu(self.gn_fc1(self.fc1(x)))
        x = F.relu(self.gn_fc2(self.fc2(x)))
        value = self.fc_out(x)
        return value


# --- PPO Training Logic (Unchanged) ---
def train(
    config: Config,
    env: SmallBowelEnv,
    actor: ActorNetwork,
    critic: CriticNetwork,
):
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate,
        eps=1e-5,
    )
    obs_actor_shape = (config.n_steps, 3, *config.patch_size_vox)
    obs_critic_shape = (config.n_steps, 4, *config.patch_size_vox)
    action_shape = (config.n_steps, 3)
    log_prob_shape = (config.n_steps, 3)
    obs_actor_buf = torch.zeros(obs_actor_shape, dtype=torch.float32).to(
        config.device
    )
    obs_critic_buf = torch.zeros(obs_critic_shape, dtype=torch.float32).to(
        config.device
    )
    actions_buf = torch.zeros(action_shape, dtype=torch.float32).to(
        config.device
    )
    log_probs_buf = torch.zeros(log_prob_shape, dtype=torch.float32).to(
        config.device
    )
    rewards_buf = torch.zeros(config.n_steps, dtype=torch.float32).to(
        config.device
    )
    dones_buf = torch.zeros(config.n_steps, dtype=torch.float32).to(
        config.device
    )
    values_buf = torch.zeros(config.n_steps, dtype=torch.float32).to(
        config.device
    )
    print(f"Starting training for {config.total_timesteps} timesteps...")
    obs_dict = env.reset()
    next_obs_actor = obs_dict["actor"]
    next_obs_critic = obs_dict["critic"]
    next_done = torch.zeros(1, dtype=torch.float32).to(config.device)
    global_step = 0
    num_updates = config.total_timesteps // config.n_steps
    ep_info_buffer = []
    for update in range(1, num_updates + 1):
        actor.eval()
        critic.eval()
        current_rollout_ep_infos = []
        current_episode_reward = 0
        current_episode_length = 0
        pbar = tqdm(
            range(config.n_steps),
            desc=f"Update {update}/{num_updates} - Collecting",
            leave=False,
        )
        for step in pbar:
            global_step += 1
            obs_actor_buf[step] = next_obs_actor
            obs_critic_buf[step] = next_obs_critic
            dones_buf[step] = next_done.item()
            with torch.no_grad():
                action_dist = actor.get_action_dist(next_obs_actor.unsqueeze(0))
                normalized_action = action_dist.sample()
                log_prob = action_dist.log_prob(normalized_action).sum(dim=-1)
                value = critic(next_obs_critic.unsqueeze(0))
            actions_buf[step] = normalized_action.squeeze(0)
            log_probs_buf[step] = action_dist.log_prob(
                normalized_action
            ).squeeze(0)
            values_buf[step] = value.squeeze()
            action_mapped = (
                2.0 * normalized_action.squeeze(0) - 1.0
            ) * config.max_step_vox
            action_vox_delta = tuple(torch.round(action_mapped).int().tolist())
            obs_dict, reward, done, info = env.step(action_vox_delta)
            rewards_buf[step] = reward
            next_obs_actor = obs_dict["actor"]
            next_obs_critic = obs_dict["critic"]
            next_done.fill_(done)
            current_episode_reward += reward.item()
            current_episode_length += 1
            if done:
                ep_coverage = info.get("episode_coverage", 0.0)
                current_rollout_ep_infos.append({
                    "reward": current_episode_reward,
                    "length": current_episode_length,
                    "coverage": ep_coverage,
                })
                pbar.set_postfix({
                    "Last Ep Reward": f"{current_episode_reward:.2f}",
                    "Ep Length": current_episode_length,
                    "Coverage": f"{ep_coverage:.3f}",
                })
                current_episode_reward = 0
                current_episode_length = 0
                obs_dict = env.reset()
                next_obs_actor = obs_dict["actor"]
                next_obs_critic = obs_dict["critic"]
                next_done = torch.zeros(1, dtype=torch.float32).to(
                    config.device
                )
        ep_info_buffer.extend(current_rollout_ep_infos)
        ep_info_buffer = ep_info_buffer[-100:]
        advantages = torch.zeros_like(rewards_buf).to(config.device)
        last_gae_lam = 0
        with torch.no_grad():
            next_value = critic(next_obs_critic.unsqueeze(0)).reshape(1, -1)
        for t in reversed(range(config.n_steps)):
            nextnonterminal = 1.0 - (
                next_done.item()
                if t == config.n_steps - 1
                else dones_buf[t + 1]
            )
            nextvalues = (
                next_value.squeeze()
                if t == config.n_steps - 1
                else values_buf[t + 1]
            )
            delta = (
                rewards_buf[t]
                + config.gamma * nextvalues * nextnonterminal
                - values_buf[t]
            )
            advantages[t] = last_gae_lam = (
                delta
                + config.gamma
                * config.gae_lambda
                * nextnonterminal
                * last_gae_lam
            )
        returns = advantages + values_buf
        actor.train()
        critic.train()
        b_obs_actor = obs_actor_buf.reshape((-1, 3, *config.patch_size_vox))
        b_obs_critic = obs_critic_buf.reshape((-1, 4, *config.patch_size_vox))
        b_actions = actions_buf.reshape((-1, 3))
        b_log_probs_old = log_probs_buf.reshape((-1, 3))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        inds = np.arange(config.n_steps)
        all_pg_loss, all_v_loss, all_ent_loss, all_total_loss = [], [], [], []
        for epoch in range(config.n_epochs):
            np.random.shuffle(inds)
            for start in range(0, config.n_steps, config.batch_size):
                end = start + config.batch_size
                mb_inds = inds[start:end]
                mb_obs_actor = b_obs_actor[mb_inds]
                mb_obs_critic = b_obs_critic[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_log_probs_old = b_log_probs_old[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                new_action_dist = actor.get_action_dist(mb_obs_actor)
                new_log_probs = new_action_dist.log_prob(mb_actions)
                entropy = new_action_dist.entropy()
                new_values = critic(mb_obs_critic)
                new_log_probs_sum = new_log_probs.sum(dim=-1)
                mb_log_probs_old_sum = mb_log_probs_old.sum(dim=-1)
                entropy_sum = entropy.sum(dim=-1)
                logratio = new_log_probs_sum - mb_log_probs_old_sum
                ratio = torch.exp(logratio)
                mb_advantages_norm = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )
                pg_loss1 = -mb_advantages_norm * ratio
                pg_loss2 = -mb_advantages_norm * torch.clamp(
                    ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                new_values = new_values.view(-1)
                value_loss = 0.5 * F.mse_loss(new_values, mb_returns)
                entropy_loss = entropy_sum.mean()
                loss = (
                    pg_loss
                    - config.ent_coef * entropy_loss
                    + config.vf_coef * value_loss
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    config.max_grad_norm,
                )
                optimizer.step()
                all_pg_loss.append(pg_loss.item())
                all_v_loss.append(value_loss.item())
                all_ent_loss.append(entropy_loss.item())
                all_total_loss.append(loss.item())
        log_data = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/policy_loss": np.mean(all_pg_loss),
            "losses/value_loss": np.mean(all_v_loss),
            "losses/entropy": np.mean(all_ent_loss),
            "losses/total_loss": np.mean(all_total_loss),
            "charts/SPS": int(
                config.n_steps
                / (
                    pbar.format_dict["elapsed"]
                    if pbar.format_dict["elapsed"] > 0
                    else 1e-6
                )
            ),
        }
        if len(ep_info_buffer) > 0:
            avg_reward = np.mean([ep["reward"] for ep in ep_info_buffer])
            avg_length = np.mean([ep["length"] for ep in ep_info_buffer])
            avg_coverage = np.mean([ep["coverage"] for ep in ep_info_buffer])
            log_data["rollout/ep_rew_mean"] = avg_reward
            log_data["rollout/ep_len_mean"] = avg_length
            log_data["rollout/ep_coverage_mean"] = avg_coverage
            print(
                f"Update: {update}, Step: {global_step}, AvgRew: {avg_reward:.2f}, AvgLen: {avg_length:.1f}, AvgCov: {avg_coverage:.3f}"
            )
        else:
            print(
                f"Update: {update}, Step: {global_step}, No episodes finished in buffer yet."
            )
        print(
            f"  Losses(P/V/E): {log_data['losses/policy_loss']:.3f}/{log_data['losses/value_loss']:.3f}/{log_data['losses/entropy']:.3f}"
        )
        if config.track_wandb:
            wandb.log(log_data, step=global_step)
        if update % 50 == 0 or update == num_updates:
            save_dict = {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update": update,
                "config": config,
            }
            torch.save(save_dict, config.save_path)
            print(f"Model saved to {config.save_path} at update {update}")
    print("Training finished.")


# --- NEW: Evaluation Function ---
def evaluate(config: Config, env: SmallBowelEnv, actor: ActorNetwork):
    """Runs the agent in evaluation mode using a loaded model."""
    print(f"\n--- Starting Evaluation ---")
    if not config.load_path or not os.path.exists(config.load_path):
        raise FileNotFoundError(
            f"Evaluation model not found at: {config.load_path}"
        )

    print(f"Loading model from: {config.load_path}")
    checkpoint = torch.load(config.load_path, map_location=config.device, weights_only=False)

    # Load actor state dict
    try:
        actor.load_state_dict(checkpoint["actor_state_dict"])
    except KeyError:
        # Handle potential older save formats or different keys
        print(
            "Warning: 'actor_state_dict' key not found in checkpoint. Trying common alternatives."
        )
        if "model_state_dict" in checkpoint:
            actor.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded state dict from 'model_state_dict'.")
        elif (
            isinstance(checkpoint, dict) and len(checkpoint) == 1
        ):  # Check if it might be just the state_dict
            key = list(checkpoint.keys())[0]
            if isinstance(
                checkpoint[key], dict
            ):  # Check if it looks like a state_dict
                try:
                    actor.load_state_dict(checkpoint[key])
                    print(f"Loaded state dict directly from key '{key}'.")
                except RuntimeError as e:
                    print(f"Failed to load state dict directly: {e}")
                    raise ValueError(
                        "Could not load actor state dict from checkpoint."
                    )
            else:
                raise ValueError(
                    "Checkpoint format not recognized for actor state dict."
                )
        else:
            raise ValueError(
                "Could not find a valid actor state dict in the checkpoint."
            )

    actor.to(config.device)
    actor.eval()  # Set actor to evaluation mode
    print("Actor model loaded successfully.")

    # Critic is not needed for evaluation, but could be loaded if desired
    # if 'critic_state_dict' in checkpoint:
    #     critic = CriticNetwork(config=config, input_channels=4).to(config.device)
    #     critic.load_state_dict(checkpoint['critic_state_dict'])
    #     critic.eval()
    #     print("Critic model loaded.")

    obs_dict = env.reset()
    state_actor = obs_dict["actor"]
    done = False
    total_reward = 0.0
    step_count = 0

    print(f"Starting evaluation run (max steps: {config.max_episode_steps})...")
    pbar = tqdm(total=config.max_episode_steps, desc="Evaluating", leave=True)

    with torch.no_grad():
        while not done and step_count < config.max_episode_steps:
            step_count += 1
            # Get action from the actor network
            action_dist = actor.get_action_dist(state_actor.unsqueeze(0))

            if config.eval_deterministic:
                # Use the mode of the distribution for deterministic action
                normalized_action = action_dist.mode
            else:
                # Sample from the distribution for stochastic action
                normalized_action = action_dist.sample()

            # Map normalized action (0 to 1) to voxel delta [-max_step, +max_step]
            action_mapped = (
                2.0 * normalized_action.squeeze(0) - 1.0
            ) * config.max_step_vox
            action_vox_delta = tuple(torch.round(action_mapped).int().tolist())

            # Step the environment
            obs_dict, reward, done, info = env.step(action_vox_delta)
            state_actor = obs_dict["actor"]
            total_reward += reward

            pbar.update(1)
            pbar.set_postfix({
                "Pos": env.current_pos_vox,
                "Reward": f"{reward:.2f}",
                "Done": done,
            })

    pbar.close()

    # --- Process and Save Results ---
    final_path_voxels = env.get_tracking_history()
    final_pos = env.current_pos_vox
    dist_to_target = np.linalg.norm(
        np.array(final_pos) - np.array(env.end_coord)
    )
    # Recalculate coverage just in case info dict didn't capture it correctly
    coverage = env._get_final_coverage()
    termination_reason = info.get(
        "termination", "max_steps_eval"
    )  # Provide default if info key missing

    print("\n--- Evaluation Results ---")
    print(f"Termination Reason: {termination_reason}")
    print(f"Total Steps Taken: {step_count}")
    print(f"Final Position: {final_pos}")
    print(f"Target Position: {env.end_coord}")
    print(f"Final Distance to Target (voxels): {dist_to_target:.2f}")
    print(f"Final Cumulative Reward: {total_reward:.2f}")
    print(f"Final Segmentation Coverage: {coverage:.4f}")
    print(f"Path Length (voxels tracked): {len(final_path_voxels)}")

    # Save the tracked path
    try:
        np.save(config.eval_output_path, final_path_voxels)
        print(
            f"Evaluation path saved successfully to: {config.eval_output_path}"
        )

        import nibabel as nib
        example = nib.load(config.nifti_path)
        final_path_map = np.zeros_like(env.image_np)
        final_path_map[tuple(final_path_voxels.T)] = 1
        for point_a, point_b in zip(final_path_voxels, final_path_voxels[1:]):
            line = line_nd(point_a, point_b, endpoint=True)
            final_path_map[line] = 1
        
        final_path_map = binary_dilation(final_path_map, iterations=3)
        example._dataobj = final_path_map
        nib.save(example, config.eval_output_path[:-4] + ".nii.gz")
    except Exception as e:
        print(f"Error saving evaluation path to {config.eval_output_path}: {e}")

    print("--- Evaluation Finished ---\n")


# --- Main Execution Block (Updated with Eval Mode) ---
if __name__ == "__main__":
    config = parse_args()
    print("Parsed configuration:")
    config_dict = {
        f.name: getattr(config, f.name)
        for f in config.__dataclass_fields__.values()
    }
    print(config_dict)

    # --- Load segmentations and find start/end points (Common for both modes) ---
    try:
        print(f"Loading main SB segmentation from: {config.seg_path}")
        sb_seg_nii = nib.load(config.seg_path)
        sb_seg_np = (sb_seg_nii.get_fdata() > 0.5).astype(np.uint8)
        print(f"Loading duodenum segmentation from: {config.duodenum_seg_path}")
        duodenum_seg_nii = nib.load(config.duodenum_seg_path)
        duodenum_seg_np = (duodenum_seg_nii.get_fdata() > 0.5).astype(np.uint8)
        print(f"Loading colon segmentation from: {config.colon_seg_path}")
        colon_seg_nii = nib.load(config.colon_seg_path)
        colon_seg_np = (colon_seg_nii.get_fdata() > 0.5).astype(np.uint8)
        if not (sb_seg_np.shape == duodenum_seg_np.shape == colon_seg_np.shape):
            raise ValueError(
                f"Seg shapes mismatch! SB:{sb_seg_np.shape}, Duo:{duodenum_seg_np.shape}, Col:{colon_seg_np.shape}"
            )
        final_start_coord, final_end_coord = find_start_end(
            duodenum_volume=duodenum_seg_np,
            colon_volume=colon_seg_np,
            small_bowel_volume=sb_seg_np,
        )
        start_end = {"start": final_start_coord, "end": final_end_coord}
    except FileNotFoundError as e:
        print(f"Error: Seg file not found: {e}")
        exit(1)
    except Exception as e:
        print(f"Error during start/end finding: {e}")
        exit(1)

    # --- Initialize Environment and Networks (Common) ---
    print("Initializing environment and networks...")
    env = SmallBowelEnv(config=config, start_end_coords=start_end)
    actor = ActorNetwork(config=config, input_channels=3).to(config.device)
    # Critic needed for training, initialize here but only used in train()
    critic = CriticNetwork(config=config, input_channels=4).to(config.device)
    print("Initialization complete.")

    # --- Mode Selection: Train or Evaluate ---
    if config.eval_mode:
        # --- Run Evaluation ---
        # No Wandb needed for evaluation usually, unless you want to log eval metrics
        if config.track_wandb:
            print(
                "Wandb tracking is enabled but typically not used during evaluation."
            )
            # Optionally, you could start a short wandb run here to log final eval metrics if desired.
        evaluate(config, env, actor)  # Pass actor only, critic not needed

    else:
        # --- Run Training ---
        run = None
        if config.track_wandb:
            try:
                run = wandb.init(
                    project=config.wandb_project_name,
                    entity=config.wandb_entity,
                    name=config.wandb_run_name,
                    sync_tensorboard=False,
                    config=config_dict,
                    monitor_gym=False,
                    save_code=True,
                )
                print(f"Wandb run initialized: {run.url}")
                # Watch model gradients with Wandb
                wandb.watch(
                    actor, log="gradients", log_freq=300, idx=0, log_graph=False
                )
                wandb.watch(
                    critic,
                    log="gradients",
                    log_freq=300,
                    idx=1,
                    log_graph=False,
                )
            except Exception as e:
                print(
                    f"Error initializing wandb: {e}. Wandb tracking disabled."
                )
                config.track_wandb = False

        try:
            train(config, env, actor, critic)
        except Exception as e:
            print(f"\nAn error occurred during training: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if config.track_wandb and run:
                # Save final model before finishing
                final_save_path = config.save_path.replace(".pth", "_final.pth")
                save_dict = {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "config": config,  # Save config used for this training run
                }
                torch.save(save_dict, final_save_path)
                print(f"Final model saved to {final_save_path}")
                # Optionally log final model as artifact
                # artifact = wandb.Artifact(f'model-{run.id}', type='model')
                # artifact.add_file(final_save_path)
                # run.log_artifact(artifact, aliases=["final"])
                run.finish()
                print("Wandb run finished.")
