"""
Environment implementation for Navigator's small bowel path tracking.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List
from skimage.draw import line_nd

from .config import Config
from .utils import compute_gdt, compute_wall_map, get_patch, draw_path_sphere, find_start_end

class SmallBowelEnv:
    """
    Environment for RL-based small bowel path tracking.

    Handles the 3D navigation environment, rewards, and state preparation.
    """

    def __init__(self, config: Config):
        """Initialize the small bowel environment."""
        self.config = config
        self.device = torch.device(config.device)

        # Initialize state variables
        self.current_pos_vox: Tuple[int, int, int] = (
            0,
            0,
            0,
        )  # Will be set in reset or update_data
        self.start_coord: Tuple[int, int, int] = (0, 0, 0)
        self.end_coord: Tuple[int, int, int] = (0, 0, 0)

        # These will be set in update_data or when loading files
        self.image_np = None
        self.seg_np = None
        self.wall_map_np = None
        self.gt_path_voxels = None
        self.gt_path_available = False
        self.gt_path_vol_np = None

        # Initialize device tensors to None
        self.image = None
        self.seg = None
        self.wall_map = None
        self.gt_path_vol = None
        self.spacing = None
        self.image_affine = None

        # Initialize path tracking variables
        self.current_step: int = 0
        self.cumulative_path_mask: Optional[torch.Tensor] = None
        self.gdt: Optional[torch.Tensor] = None
        self.max_gdt_achieved: float = 0.0
        self.tracking_path_history: List[Tuple[int, int, int]] = []
        self.current_gdt_val: float = (
            0.0  # Store GDT value of current position for immediate reward
        )

        # Flag to avoid recomputing the GDT too often
        self.gdt_computed = False
        self.start_choice = "start"  # Default start choice

    def _process_gt_path(self):
        """Process ground truth path data."""
        if self.gt_path_voxels is None or self.image_np is None:
            self._create_empty_gt_path()
            return

        self.gt_path_available = True
        self.gt_path_vol_np = np.zeros_like(self.image_np, dtype=np.float32)
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
            self.gt_path_vol_np[valid_gt_path[:, 0], valid_gt_path[:, 1], valid_gt_path[:, 2]] = 1.0
        self.gt_path_vol = torch.from_numpy(self.gt_path_vol_np).to(self.device)

    def _create_empty_gt_path(self):
        """Create an empty ground truth path tensor."""
        if self.image_np is not None:
            self.gt_path_available = False
            self.gt_path_vol_np = np.zeros_like(self.image_np, dtype=np.float32)
            self.gt_path_vol = torch.from_numpy(self.gt_path_vol_np).to(self.device)

    def update_data(
        self,
        image: torch.Tensor,
        seg: torch.Tensor,
        duodenum: torch.Tensor,
        colon: torch.Tensor,
        gt_path: Optional[torch.Tensor] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
        image_affine: Optional[np.ndarray] = None,
    ):
        """
        Update the environment with new data.

        Args:
            image: 3D image array
            seg: Small bowel segmentation array
            duodenum: Duodenum segmentation array
            colon: Colon segmentation array
            gt_path: Optional ground truth path points array (Nx3)
        """
        # Store the raw data
        self.image_np = image.numpy()
        self.seg_np = (seg.numpy() > 0.5).astype(np.uint8)
        duodenum_np = (duodenum.numpy() > 0.5).astype(np.uint8)
        colon_np = (colon.numpy() > 0.5).astype(np.uint8)
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.image_affine = image_affine if image_affine is not None else np.eye(4)
    

        # Generate wall map for the new image
        self.wall_map_np = compute_wall_map(self.image_np, sigmas=self.config.wall_map_sigmas).astype(np.float32)

        # Transfer data to device
        self.image = image.to(self.device)
        self.seg = seg.to(self.device)
        self.wall_map = torch.from_numpy(self.wall_map_np).to(self.device)

        # Update ground truth path if provided
        self.gt_path_voxels = gt_path
        if gt_path is not None:
            self._process_gt_path()
        else:
            self._create_empty_gt_path()

        # Find new start and end coordinates
        try:
            self.start_coord, self.end_coord = find_start_end(
                duodenum_volume=duodenum_np, colon_volume=colon_np, small_bowel_volume=self.seg_np
            )
            print(f"Found start point at {self.start_coord} and end point at {self.end_coord}")
        except Exception as e:
            print(f"Error finding start/end points: {e}. Using random points.")
            # Fallback to random points within the segmentation
            valid_points = np.argwhere(self.seg_np > 0)
            if len(valid_points) >= 2:
                idx1, idx2 = np.random.choice(len(valid_points), size=2, replace=False)
                self.start_coord = tuple(valid_points[idx1])
                self.end_coord = tuple(valid_points[idx2])
            else:
                raise ValueError("Segmentation is empty, cannot set start/end points")

    def _get_state_patches(self) -> Dict[str, torch.Tensor]:
        """Get state patches centered at current position for actor and critic."""
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
        actor_state = torch.stack([img_patch, wall_patch, cum_path_patch], dim=0)
        critic_state = torch.stack([img_patch, wall_patch, cum_path_patch, gt_path_patch], dim=0)
        return {"actor": actor_state, "critic": critic_state}

    def _is_valid_pos(self, pos_vox: Tuple[int, int, int]) -> bool:
        """Check if a position is within the volume bounds."""
        s = self.image.shape
        return 0 <= pos_vox[0] < s[0] and 0 <= pos_vox[1] < s[1] and 0 <= pos_vox[2] < s[2]

    def _get_final_coverage(self) -> float:
        """Calculate the coverage of the segmentation mask by the cumulative path."""
        if self.cumulative_path_mask is None or self.seg is None:
            return 0.0
        seg_volume = torch.sum(self.seg).item()
        if seg_volume < 1e-6:
            return 0.0
        intersection = torch.sum(self.cumulative_path_mask * self.seg).item()
        return intersection / seg_volume

    def reset(self, start_choice: str = "start") -> Dict[str, torch.Tensor]:
        """Reset the environment to start a new episode."""
        self.current_step = 0
        self.gdt_computed = self.start_choice == start_choice and self.gdt_computed
        self.start_choice = start_choice
        if start_choice == "end":
            self.current_pos_vox = self.end_coord
        else:
            self.current_pos_vox = self.start_coord

        # Validate start position and find alternative if invalid
        if (
            not self._is_valid_pos(self.current_pos_vox)
            or self.seg[self.current_pos_vox].item() == 0
        ):
            print(f"Warning: Start pos {self.current_pos_vox} invalid. Searching...")
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
                        if self._is_valid_pos(new_pos) and self.seg[new_pos].item() > 0:
                            self.current_pos_vox = new_pos
                            found_valid_start = True
                            break
                    if found_valid_start:
                        break
                if found_valid_start:
                    break

            # Fall back to a random valid segmentation voxel if needed
            if not found_valid_start:
                valid_voxels = torch.nonzero(self.seg > 0)
                if len(valid_voxels) > 0:
                    rand_idx = torch.randint(0, len(valid_voxels), (1,)).item()
                    self.current_pos_vox = tuple(valid_voxels[rand_idx].tolist())
                    print(f"Fallback start: {self.current_pos_vox}")
                else:
                    raise ValueError("Cannot start: Seg mask empty.")

        # Initialize path tracking
        self.cumulative_path_mask = torch.zeros_like(
            self.image, dtype=torch.float32, device=self.device
        )
        draw_path_sphere(
            self.cumulative_path_mask, self.current_pos_vox, self.config.cumulative_path_radius_vox
        )
        self.tracking_path_history = [self.current_pos_vox]

        if not self.gdt_computed:
            # Compute geodesic distance transform
            self.gdt = compute_gdt(self.seg, self.current_pos_vox, self.spacing, device=self.device)
            self.gdt_computed = True

        # Initialize current GDT value
        if self.gdt is not None and self._is_valid_pos(self.current_pos_vox):
            gdt_tensor = self.gdt[self.current_pos_vox]
            self.current_gdt_val = gdt_tensor.item() if torch.isfinite(gdt_tensor) else 0.0
        else:
            self.current_gdt_val = 0.0

        self.max_gdt_achieved = self.current_gdt_val  # Initialize max achieved with start value

        return self._get_state_patches()

    def get_tracking_history(self) -> np.ndarray:
        """Get the history of tracked positions."""
        return np.array(self.tracking_path_history)

    def _calculate_reward(
        self, action_vox: Tuple[int, int, int], next_pos_vox: Tuple[int, int, int]
    ) -> float:
        """Calculate the reward for the current step."""
        rt = 0.0
        current_pos_np = np.array(self.current_pos_vox)
        next_pos_np = np.array(next_pos_vox)
        action_np = np.array(action_vox)

        # --- 1. Zero movement penalty ---
        if np.all(action_np == 0):
            return -self.config.r_wall  # Just return reward, GDT update handled in step

        # --- Line segment voxels ---
        line_voxels_idx = line_nd(current_pos_np, next_pos_np, endpoint=True)
        S = list(zip(line_voxels_idx[0], line_voxels_idx[1], line_voxels_idx[2]))

        # --- 2. GDT-based reward (Immediate Increase) ---
        next_gdt_val = np.inf
        if self.gdt is not None and self._is_valid_pos(next_pos_vox):
            next_gdt_tensor = self.gdt[next_pos_vox]
            if torch.isfinite(next_gdt_tensor):
                next_gdt_val = next_gdt_tensor.item()

                if self.config.use_immediate_gdt_reward:
                    # Reward immediate positive change from previous step's GDT value
                    immediate_delta_gdt = next_gdt_val - self.current_gdt_val
                    if immediate_delta_gdt > 0:
                        # Simple reward scaled by r_val2
                        rt += self.config.r_val2 * (
                            immediate_delta_gdt / max(1.0, self.config.max_step_vox)
                        )  # Normalize by step size
                else:
                    # Original logic: Reward increase beyond max achieved
                    delta_gdt = next_gdt_val - self.max_gdt_achieved
                    if self.config.gdt_max_increase_theta > 1e-9:
                        if delta_gdt > self.config.gdt_max_increase_theta:
                            rt -= self.config.r_val2  # Penalty for too large step
                        elif delta_gdt > 0:
                            rt += (
                                delta_gdt / self.config.gdt_max_increase_theta
                            ) * self.config.r_val2
                    elif delta_gdt > 0:  # If theta is ~0, reward any positive change
                        rt += self.config.r_val2

        # --- 3. Wall-based penalty (Reduced Impact) ---
        wall_sum = 0.0
        num_valid_s = 0
        for s_vox in S:
            if self._is_valid_pos(s_vox):
                wall_sum += self.wall_map[s_vox].item()
                num_valid_s += 1
        if num_valid_s > 0:
            avg_wall = wall_sum / num_valid_s
            # Apply reduced penalty using wall_penalty_scale from config
            rt -= avg_wall * self.config.wall_penalty_scale * self.config.r_val2

        # --- 4. Revisiting penalty ---
        revisit_sum = 0
        if self.cumulative_path_mask is not None:
            for s_vox in S:
                if self._is_valid_pos(s_vox):
                    revisit_sum += self.cumulative_path_mask[s_vox].item()
        if revisit_sum > 0:  # Penalize if any part overlaps
            rt -= self.config.r_wall

        # --- 5. Out-of-segmentation penalty ---
        # Check the *intended* next position
        if not self._is_valid_pos(next_pos_vox) or self.seg[next_pos_vox].item() == 0:
            rt -= self.config.r_wall

        return rt

    def step(
        self, action_vox_delta: Tuple[int, int, int]
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action_vox_delta: Movement vector in voxel coordinates

        Returns:
            Tuple of (state, reward, done, info)
        """
        if self.cumulative_path_mask is None:
            raise RuntimeError("Env not reset.")

        self.current_step += 1
        current_z, current_y, current_x = self.current_pos_vox
        dz, dy, dx = action_vox_delta
        next_pos_vox = (current_z + dz, current_y + dy, current_x + dx)

        # Calculate reward based on *intended* next step
        reward = self._calculate_reward(action_vox_delta, next_pos_vox)

        # Update GDT tracking *before* updating position
        next_gdt_val = np.inf
        if self.gdt is not None and self._is_valid_pos(next_pos_vox):
            next_gdt_tensor = self.gdt[next_pos_vox]
            if torch.isfinite(next_gdt_tensor):
                next_gdt_val = next_gdt_tensor.item()
                # Update max achieved if using original reward logic
                if not self.config.use_immediate_gdt_reward:
                    self.max_gdt_achieved = max(self.max_gdt_achieved, next_gdt_val)
                # Update current GDT value for next step's immediate reward calculation
                self.current_gdt_val = next_gdt_val
            else:  # Landed outside reachable GDT area
                self.current_gdt_val = 0.0  # Or keep previous? Resetting seems safer.
        else:  # Landed outside image bounds
            self.current_gdt_val = 0.0

        # Update position
        self.current_pos_vox = next_pos_vox

        # Check validity and update cumulative path
        is_next_pos_valid_seg = (
            self._is_valid_pos(next_pos_vox) and self.seg[next_pos_vox].item() > 0
        )
        if is_next_pos_valid_seg:
            draw_path_sphere(
                self.cumulative_path_mask,
                self.current_pos_vox,
                self.config.cumulative_path_radius_vox,
            )
            self.tracking_path_history.append(self.current_pos_vox)

        # Check termination conditions
        done = False
        termination_reason = ""
        if self.current_step >= self.config.max_episode_steps:
            done = True
            termination_reason = "max_steps"
        elif not is_next_pos_valid_seg:
            done = True
            termination_reason = "out_of_bounds_or_segmentation"
        elif (
            np.linalg.norm(np.array(self.current_pos_vox) - np.array(self.end_coord))
            < self.config.cumulative_path_radius_vox
        ):
            done = True
            termination_reason = "reached_end"

        info = {
            "current_pos": self.current_pos_vox,
            "termination": termination_reason,
            "steps": self.current_step,
        }

        # Add final reward adjustment if episode is done
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
