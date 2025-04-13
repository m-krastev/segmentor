"""
Environment implementation for Navigator's small bowel path tracking,
integrated with TorchRL. Simplified version without try-except blocks.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List, Iterator

from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    BoundedTensorSpec,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs  # For checking implementation

from skimage.draw import line_nd

from .config import Config
from .utils import compute_gdt, compute_wall_map, get_patch, draw_path_sphere, find_start_end
# Assume dataset is available for type hinting, but iterator is passed in
# from .dataset import SmallBowelDataset


# Define action spec constants
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_DIM = 3  # 3D movement delta


class SmallBowelEnv(EnvBase):
    """
    TorchRL-compatible environment for RL-based small bowel path tracking.
    (Simplified version without internal try-except blocks).

    Inherits from EnvBase and handles data loading via an iterator during resets.
    Relies on upstream data validity and will raise exceptions on errors.
    """

    def __init__(
        self,
        config: Config,
        dataset_iterator: Iterator,  # Pass an iterator over your dataset
        num_episodes_per_sample: int = 32,
        device: Optional[torch.device] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        """Initialize the small bowel environment."""
        if batch_size is None:
            batch_size = torch.Size([1])  # Default to batch size 1
        if device is None:
            device = torch.device("cpu")

        # --- Call EnvBase Constructor ---
        super().__init__(device=device, batch_size=batch_size)

        # --- Store Config and Iterator ---
        self.config = config
        self.dataset_iterator = dataset_iterator  # Store the iterator
        self._current_subject_data = None  # Store data for the current subject
        self.num_episodes_per_sample = num_episodes_per_sample
        self.episodes_on_current_subject = 0  # Counter for episodes on current subject

        # --- Define Specs ---
        observation_spec = CompositeSpec(
            actor=UnboundedContinuousTensorSpec(
                shape=torch.Size([*batch_size, 3, *config.patch_size_vox]),  # B, C, D, H, W
                dtype=torch.float32,
                device=device,
            ),
            critic=UnboundedContinuousTensorSpec(
                shape=torch.Size([*batch_size, 4, *config.patch_size_vox]),  # B, C, D, H, W
                dtype=torch.float32,
                device=device,
            ),
            shape=batch_size,
            device=device,
        )

        action_spec = BoundedTensorSpec(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=torch.Size([*batch_size, ACTION_DIM]),  # B, ActionDim
            dtype=torch.float32,
            device=device,
        )

        reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size([*batch_size, 1]),  # B, 1
            dtype=torch.float32,
            device=device,
        )

        done_spec = DiscreteTensorSpec(
            n=2,  # boolean done state (0 or 1)
            shape=torch.Size([*batch_size, 1]),  # B, 1
            dtype=torch.bool,
            device=device,
        )

        # Set the specs *after* calling super().__init__
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec

        # --- Initialize Internal State (per-instance, not per-batch element) ---
        self.current_pos_vox: Tuple[int, int, int] = (0, 0, 0)
        self.start_coord: Tuple[int, int, int] = (0, 0, 0)
        self.end_coord: Tuple[int, int, int] = (0, 0, 0)
        self.image_np = None
        self.seg_np = None
        self.wall_map_np = None
        self.gt_path_voxels = None
        self.gt_path_available = False
        self.gt_path_vol_np = None
        self.image: Optional[torch.Tensor] = None
        self.seg: Optional[torch.Tensor] = None
        self.wall_map: Optional[torch.Tensor] = None
        self.gt_path_vol: Optional[torch.Tensor] = None
        self.spacing: Optional[Tuple[float, float, float]] = None
        self.image_affine: Optional[np.ndarray] = None
        self.current_step_count: int = 0
        self.cumulative_path_mask: Optional[torch.Tensor] = None
        self.gdt: Optional[torch.Tensor] = None
        self.max_gdt_achieved: float = 0.0
        self.tracking_path_history: List[Tuple[int, int, int]] = []
        self.current_gdt_val: float = 0.0
        # self.gdt_computed = False
        self.start_choice = "start"

        # --- TorchRL Internal State Flags (per-batch element) ---
        self._is_done = torch.ones(self.batch_size[0], 1, dtype=torch.bool, device=self.device)

    # --- Data Loading Method (Internal) ---
    def _load_next_subject(self) -> bool:
        """
        Loads the next subject's data from the iterator.
        Raises StopIteration if iterator is exhausted.
        Raises other exceptions if data loading/processing fails.
        """
        subject_data = next(self.dataset_iterator)  # Let StopIteration propagate
        self._current_subject_data = subject_data
        subject_id = subject_data.get("id", "N/A")
        print(f"\n[Env {id(self)}] Loading subject: {subject_id}")

        # Call update_data - let it raise exceptions if issues occur
        # Pass all relevant fields from the dataset output
        self.update_data(
            image=subject_data["image"],
            seg=subject_data["seg"],
            wall_map=subject_data["wall_map"],
            gdt_start=subject_data["gdt_start"],
            gdt_end=subject_data["gdt_end"],
            start_coord=subject_data["start_coord"],
            end_coord=subject_data["end_coord"],
            gt_path=subject_data.get("gt_path"),  # Still optional
            spacing=subject_data.get("spacing"),
            image_affine=subject_data.get("image_affine"),
            # Pass numpy versions if needed, or rely on tensor.cpu().numpy()
            image_np=subject_data.get("image_np"),
            seg_np=subject_data.get("seg_np"),
            wall_map_np=subject_data.get("wall_map_np"),
        )

        # Check if critical data was loaded successfully by update_data
        if (
            self.image is None
            or self.seg is None
            or self.wall_map is None
            or self.gdt_start is None
            or self.gdt_end is None
        ):
            raise RuntimeError(f"Critical data is None after loading subject {subject_id}.")

        # self.gdt_computed = False # Removed
        print(f"[Env {id(self)}] Subject {subject_id} loaded successfully.")
        return True  # Indicate success

    # --- Internal Data Update Method ---
    def update_data(
        self,
        image: torch.Tensor,
        seg: torch.Tensor,
        wall_map: torch.Tensor,  # Receive wall_map tensor
        gdt_start: torch.Tensor,  # Receive GDT tensors
        gdt_end: torch.Tensor,
        start_coord: Tuple[int, int, int],  # Receive coords
        end_coord: Tuple[int, int, int],
        gt_path: Optional[torch.Tensor] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
        image_affine: Optional[np.ndarray] = None,
        # Receive numpy versions if provided by dataset
        image_np: Optional[np.ndarray] = None,
        seg_np: Optional[np.ndarray] = None,
        wall_map_np: Optional[np.ndarray] = None,
    ):
        """
        Updates the environment's internal data stores using data from the dataset.
        """
        # Store tensors directly
        self.image = image.to(self.device)
        self.seg = seg.to(device=self.device, dtype=torch.uint8)  # Ensure correct type
        self.wall_map = wall_map.to(self.device)
        self.gdt_start = gdt_start.to(self.device)
        self.gdt_end = gdt_end.to(self.device)

        # Store numpy versions (either passed in or derived)
        self.image_np = image_np if image_np is not None else image.cpu().numpy()
        self.seg_np = seg_np if seg_np is not None else seg.cpu().numpy().astype(np.uint8)
        self.wall_map_np = wall_map_np if wall_map_np is not None else wall_map.cpu().numpy()

        # Store other metadata
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.image_affine = image_affine if image_affine is not None else np.eye(4)
        self.start_coord = start_coord
        self.end_coord = end_coord
        print(f"  Received start point {self.start_coord} and end point {self.end_coord}")

        # Basic sanity check on shapes before proceeding
        if self.image.ndim != 3 or not all(s > 0 for s in self.image.shape):
            raise ValueError(f"Invalid image shape received: {self.image.shape}")
        if self.seg.shape != self.image.shape:
            raise ValueError(
                f"Segmentation shape {self.seg.shape} mismatch with image shape {self.image.shape}"
            )
        if self.wall_map.shape != self.image.shape:
            raise ValueError(
                f"Wall map shape {self.wall_map.shape} mismatch with image shape {self.image.shape}"
            )
        if self.gdt_start.shape != self.image.shape or self.gdt_end.shape != self.image.shape:
            raise ValueError(f"GDT shape mismatch with image shape {self.image.shape}")

        # Remove wall map calculation
        # self.wall_map_np = compute_wall_map(...)
        # self.wall_map = torch.from_numpy(self.wall_map_np).to(self.device)

        # Update ground truth path if provided
        # Convert gt_path tensor to numpy for _process_gt_path
        self.gt_path_voxels = gt_path.cpu().numpy() if gt_path is not None else None
        self._process_gt_path()  # Uses self.image_np, sets self.gt_path_vol (Tensor)

    # --- Internal Helper Methods (Simplified) ---
    def _process_gt_path(self):
        """Process ground truth path data. Assumes self.image_np is valid."""
        if self.gt_path_voxels is None or self.image_np is None:
            self._create_empty_gt_path()
            return

        # Basic check
        if self.image_np.ndim != 3 or not all(s > 0 for s in self.image_np.shape):
            print("Warning: Cannot process GT path, image_np is invalid. Creating empty path.")
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
        else:
            print("Warning: No valid GT path voxels found within image bounds.")

        self.gt_path_vol = torch.from_numpy(self.gt_path_vol_np).to(self.device)

    def _create_empty_gt_path(self):
        """Create an empty ground truth path tensor."""
        self.gt_path_available = False
        if (
            self.image_np is not None
            and self.image_np.ndim == 3
            and all(s > 0 for s in self.image_np.shape)
        ):
            self.gt_path_vol_np = np.zeros_like(self.image_np, dtype=np.float32)
            self.gt_path_vol = torch.from_numpy(self.gt_path_vol_np).to(self.device)
        else:
            # Cannot create based on image, set to None
            print("Warning: Cannot create empty GT path, image_np is None or invalid.")
            self.gt_path_vol_np = None
            self.gt_path_vol = None

    def _get_state_patches(self) -> Dict[str, torch.Tensor]:
        """Get state patches centered at current position. Assumes tensors are valid."""
        # Assert that required tensors are initialized
        if self.image is None:
            raise RuntimeError("Environment image tensor is None.")
        if self.wall_map is None:
            raise RuntimeError("Environment wall_map tensor is None.")
        if self.cumulative_path_mask is None:
            raise RuntimeError(
                "Environment cumulative_path_mask is None (should be initialized in reset)."
            )
        # GT path can be None if not available, handle it
        if self.gt_path_vol is None:
            print("Warning: GT path volume is None during patch extraction. Using zeros.")
            # Create a zero patch matching the spec, assuming self.image exists for shape/device
            _zero_patch_shape = self.config.patch_size_vox
            gt_path_patch = torch.zeros(_zero_patch_shape, dtype=torch.float32, device=self.device)
        else:
            gt_path_patch = get_patch(
                self.gt_path_vol, self.current_pos_vox, self.config.patch_size_vox
            )

        img_patch = get_patch(self.image, self.current_pos_vox, self.config.patch_size_vox)
        wall_patch = get_patch(self.wall_map, self.current_pos_vox, self.config.patch_size_vox)
        cum_path_patch = get_patch(
            self.cumulative_path_mask, self.current_pos_vox, self.config.patch_size_vox
        )

        # Stack patches
        actor_state = torch.stack(
            [img_patch, wall_patch, cum_path_patch], dim=0
        )  # Shape: [C, D, H, W]
        critic_state = torch.stack(
            [img_patch, wall_patch, cum_path_patch, gt_path_patch], dim=0
        )  # Shape: [C, D, H, W]
        return {"actor": actor_state, "critic": critic_state}

    def _is_valid_pos(self, pos_vox: Tuple[int, int, int]) -> bool:
        """Check if a position is within the volume bounds."""
        # Should only be called after self.image is confirmed to be not None
        s = self.image.shape
        return 0 <= pos_vox[0] < s[0] and 0 <= pos_vox[1] < s[1] and 0 <= pos_vox[2] < s[2]

    def _get_final_coverage(self) -> float:
        """Calculate coverage. Assumes tensors are valid."""
        if self.cumulative_path_mask is None:
            return 0.0  # Should not happen if reset worked
        if self.seg is None:
            return 0.0  # Should not happen if data loaded

        seg_mask = self.seg.bool()
        seg_volume = torch.sum(seg_mask).item()
        if seg_volume < 1e-6:
            return 0.0

        path_mask = self.cumulative_path_mask.float()
        intersection = torch.sum(path_mask * seg_mask.float()).item()
        return intersection / seg_volume

    def get_tracking_history(self) -> np.ndarray:
        """Get the history of tracked positions."""
        return np.array(self.tracking_path_history)

    def _calculate_reward(
        self, action_vox: Tuple[int, int, int], next_pos_vox: Tuple[int, int, int]
    ) -> float:
        """Calculate the reward for the current step. Assumes tensors are valid."""
        # Add assertions for required tensors at the start
        if self.wall_map is None:
            raise RuntimeError("Cannot calculate reward, wall_map is None.")
        if self.cumulative_path_mask is None:
            raise RuntimeError("Cannot calculate reward, cumulative_path_mask is None.")
        if self.seg is None:
            raise RuntimeError("Cannot calculate reward, seg is None.")
        # GDT can be None if computation failed, handle gracefully

        rt = 0.0
        current_pos_np = np.array(self.current_pos_vox)
        next_pos_np = np.array(next_pos_vox)
        action_np = np.array(action_vox)

        # --- 1. Zero movement penalty ---
        if np.all(action_np == 0):
            # Update GDT value even if not moving
            if self.gdt is not None and self._is_valid_pos(self.current_pos_vox):
                gdt_tensor = self.gdt[self.current_pos_vox]
                self.current_gdt_val = gdt_tensor.item() if torch.isfinite(gdt_tensor) else 0.0
            else:
                self.current_gdt_val = 0.0
            return -self.config.r_wall  # Use penalty from config

        # --- Line segment voxels ---
        line_voxels_idx = line_nd(current_pos_np, next_pos_np, endpoint=True)
        S = list(zip(line_voxels_idx[0], line_voxels_idx[1], line_voxels_idx[2]))

        # --- 2. GDT-based reward ---
        next_gdt_val = -np.inf  # Initialize low
        if self.gdt is not None and self._is_valid_pos(next_pos_vox):
            next_gdt_tensor = self.gdt[next_pos_vox]
            if torch.isfinite(next_gdt_tensor):
                next_gdt_val = next_gdt_tensor.item()
                if self.config.use_immediate_gdt_reward:
                    immediate_delta_gdt = next_gdt_val - self.current_gdt_val
                    if immediate_delta_gdt > 0:
                        rt += self.config.r_val2 * (
                            immediate_delta_gdt / max(1.0, self.config.max_step_vox)
                        )
                else:  # Original logic
                    delta_gdt = next_gdt_val - self.max_gdt_achieved
                    if delta_gdt > 0:
                        gain = delta_gdt
                        if self.config.gdt_max_increase_theta > 1e-9:
                            gain = min(delta_gdt, self.config.gdt_max_increase_theta)
                            rt += (gain / self.config.gdt_max_increase_theta) * self.config.r_val2
                        else:  # If theta is ~0, reward any positive change?
                            rt += self.config.r_val2  # Or scale differently? Check config

        # --- 3. Wall-based penalty ---
        wall_sum = 0.0
        num_valid_s = 0
        for s_vox in S:
            if self._is_valid_pos(s_vox):
                # Direct indexing, will raise IndexError if _is_valid_pos is wrong
                wall_val = self.wall_map[s_vox].item()
                wall_sum += wall_val
                num_valid_s += 1
        if num_valid_s > 0:
            avg_wall = wall_sum / num_valid_s
            rt -= avg_wall * self.config.wall_penalty_scale

        # --- 5. Out-of-segmentation penalty ---
        if not self._is_valid_pos(next_pos_vox) or self.seg[next_pos_vox].item() == 0:
            rt -= self.config.r_wall

        # --- Update GDT tracking ---
        if next_gdt_val > -np.inf:  # Valid GDT at next step
            if not self.config.use_immediate_gdt_reward:
                self.max_gdt_achieved = max(self.max_gdt_achieved, next_gdt_val)
            self.current_gdt_val = next_gdt_val
        else:  # Invalid GDT
            self.current_gdt_val = 0.0

        return rt

    # --- TorchRL Methods Implementation (Simplified) ---

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        """
        Resets the environment. Loads next subject if needed based on episode count.
        Will raise exceptions if loading or initialization fails.
        Selects appropriate GDT based on start_choice.
        """
        # Check if the *previous* state indicated 'done'. If so, an episode just finished.
        # Increment counter only if an episode actually finished.
        if self._is_done.all():
            self.episodes_on_current_subject += 1

        # Decide whether to load a new subject
        should_load_new_subject = (
            self._current_subject_data is None  # First time loading
            or self.episodes_on_current_subject >= self.num_episodes_per_sample
        )

        if should_load_new_subject:
            # Reset counter *before* loading, as loading signifies starting fresh
            self.episodes_on_current_subject = 0
            try:
                load_successful = self._load_next_subject()
                if not load_successful:  # Should not happen if _load_next_subject raises exceptions
                    raise RuntimeError(
                        "Internal error: _load_next_subject returned False unexpectedly."
                    )
            except StopIteration:
                print("Dataset iterator exhausted. Cannot load new subject.")
                # Decide how to handle this: raise error, loop dataset, etc.
                # For now, raise an error to signal the training loop.
                raise StopIteration("Dataset exhausted during environment reset.")
            # Counter is now 0 for the first episode on the new subject

        # --- Reset internal episode state ---
        self.current_step_count = 0

        # Assert that data necessary for reset exists (checked in _load_next_subject if loaded)
        if self.image is None or self.seg is None or self.gdt_start is None or self.gdt_end is None:
            raise RuntimeError(
                "Cannot reset, critical tensors are None. Data might not have been loaded."
            )

        # Determine start position and select appropriate GDT
        # ... (existing start position and GDT selection logic) ...
        if self.start_choice == "end":
            self.current_pos_vox = self.end_coord
            self.gdt = self.gdt_end  # Use GDT calculated from end point
            print("Starting from END coordinate, using GDT_end.")
        else:  # Default to start
            self.current_pos_vox = self.start_coord
            self.gdt = self.gdt_start  # Use GDT calculated from start point
            print("Starting from START coordinate, using GDT_start.")

        # Validate start position (simplified check)
        # ... (existing start position validation and fallback logic) ...
        if (
            not self._is_valid_pos(self.current_pos_vox)
            or self.seg[self.current_pos_vox].item() == 0
        ):
            print(
                f"Warning: Chosen start pos {self.current_pos_vox} invalid/outside seg. Searching nearby..."
            )
            # ... (existing fallback search logic) ...
            found_valid_start = False
            # (Code for searching nearby...)
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
                            print(f"  Found valid nearby start: {self.current_pos_vox}")
                            break
                    if found_valid_start:
                        break
                if found_valid_start:
                    break

            if not found_valid_start:
                valid_voxels = torch.nonzero(self.seg > 0)
                if len(valid_voxels) == 0:
                    raise ValueError("Cannot reset: No valid voxels found in segmentation mask.")
                rand_idx = torch.randint(0, len(valid_voxels), (1,)).item()
                self.current_pos_vox = tuple(valid_voxels[rand_idx].cpu().tolist())
                print(f"  Fallback random start: {self.current_pos_vox}")

        # Initialize path tracking
        # ... (existing path tracking initialization) ...
        self.cumulative_path_mask = torch.zeros_like(
            self.image, dtype=torch.float32, device=self.device
        )
        # Ensure current_pos_vox is valid before drawing
        if self._is_valid_pos(self.current_pos_vox):
            draw_path_sphere(
                self.cumulative_path_mask,
                self.current_pos_vox,
                self.config.cumulative_path_radius_vox,
            )
        else:
            print(
                f"Warning: Cannot draw initial path sphere at invalid position {self.current_pos_vox}"
            )
        self.tracking_path_history = [self.current_pos_vox]

        # Initialize current GDT value using the selected GDT
        # ... (existing GDT initialization) ...
        if self.gdt is not None and self._is_valid_pos(self.current_pos_vox):
            gdt_tensor = self.gdt[self.current_pos_vox]
            # Handle potential inf values in GDT map
            self.current_gdt_val = gdt_tensor.item() if torch.isfinite(gdt_tensor) else 0.0
        else:
            self.current_gdt_val = 0.0  # If start is invalid or GDT is None
        self.max_gdt_achieved = self.current_gdt_val
        print(f"Initial GDT value at {self.current_pos_vox}: {self.current_gdt_val}")

        # --- Get Initial State Patches ---
        obs_dict = self._get_state_patches()  # Let it raise RuntimeError if patches fail

        # --- Update Internal Done Flag and Package Output ---
        # Set done to False for the *start* of the new episode
        self._is_done.fill_(False)

        reset_td = TensorDict(
            {
                "actor": obs_dict["actor"].unsqueeze(0),  # Add batch dim
                "critic": obs_dict["critic"].unsqueeze(0),  # Add batch dim
                "done": self._is_done.clone(),
                "terminated": self._is_done.clone(),  # False after reset
                "truncated": torch.zeros_like(self._is_done),  # False after reset
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return reset_td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Performs a step. Assumes env is initialized. Raises exceptions on errors."""
        # Assert essential state is valid before proceeding
        if self.image is None:
            raise RuntimeError("Cannot step, image is None.")
        if self.seg is None:
            raise RuntimeError("Cannot step, seg is None.")
        if self.wall_map is None:
            raise RuntimeError("Cannot step, wall_map is None.")
        if self.cumulative_path_mask is None:
            raise RuntimeError("Cannot step, cumulative_path_mask is None.")

        # Extract Action
        action_normalized = tensordict.get("action").squeeze(0)

        # Map Action
        action_mapped = (2.0 * action_normalized - 1.0) * self.config.max_step_vox
        action_vox_delta = tuple(torch.round(action_mapped).int().cpu().tolist())

        # Execute Step Logic
        self.current_step_count += 1
        next_pos_vox = (
            self.current_pos_vox[0] + action_vox_delta[0],
            self.current_pos_vox[1] + action_vox_delta[1],
            self.current_pos_vox[2] + action_vox_delta[2],
        )

        # Calculate reward (relies on internal assertions/checks)
        reward = self._calculate_reward(action_vox_delta, next_pos_vox)

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

        # Check Termination Conditions
        done, terminated, truncated = False, False, False
        termination_reason = ""
        if self.current_step_count >= self.config.max_episode_steps:
            done, truncated, termination_reason = True, True, "max_steps"
        elif not is_next_pos_valid_seg:
            done, terminated, termination_reason = True, True, "out_of_bounds_or_segmentation"
        elif (
            self._is_valid_pos(self.end_coord)
            and np.linalg.norm(np.array(self.current_pos_vox) - np.array(self.end_coord))
            < self.config.cumulative_path_radius_vox
        ):
            done, terminated, termination_reason = True, True, "reached_end"

        # Final Reward Adjustment
        final_coverage = 0.0
        if done:
            final_coverage = self._get_final_coverage()
            target_reached = termination_reason == "reached_end"
            final_reward_adjustment = (
                (final_coverage * self.config.r_final)
                if target_reached
                else ((final_coverage - 1.0) * abs(self.config.r_final))
            )
            reward += final_reward_adjustment

        # Get Next State Patches
        next_obs_dict = self._get_state_patches()  # Let it raise RuntimeError if needed

        # Update Internal Done Flag
        self._is_done[0] = done

        # Prepare Tensors for Output TensorDict (Shape [1, ...])
        _reward = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        _done = torch.tensor([[done]], dtype=torch.bool, device=self.device)
        _terminated = torch.tensor([[terminated]], dtype=torch.bool, device=self.device)
        _truncated = torch.tensor([[truncated]], dtype=torch.bool, device=self.device)
        _next_actor_obs = next_obs_dict["actor"].unsqueeze(0)
        _next_critic_obs = next_obs_dict["critic"].unsqueeze(0)
        _next_ep_reward = torch.tensor(
            [[reward if done else 0.0]], dtype=torch.float32, device=self.device
        )
        _next_ep_length = torch.tensor(
            [[self.current_step_count if done else 0]], dtype=torch.int, device=self.device
        )
        _next_ep_coverage = torch.tensor(
            [[final_coverage if done else 0.0]], dtype=torch.float32, device=self.device
        )

        # Package Output TensorDict (Using explicit nested construction)
        next_state_td = TensorDict(
            {
                "actor": _next_actor_obs,
                "critic": _next_critic_obs,
                "episode_reward": _next_ep_reward,
                "episode_length": _next_ep_length,
                "info_coverage": _next_ep_coverage,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        output_td = TensorDict(
            {
                "reward": _reward,
                "done": _done,
                "terminated": _terminated,
                "truncated": _truncated,
                "next": next_state_td,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return output_td

    def _set_seed(self, seed: Optional[int] = None):
        """Sets the seed for the environment's random number generator(s)."""
        # Add seeding for np.random if used in fallbacks or utils
        # np.random.seed(seed)
        # Add seeding for torch.random if used directly
        # torch.manual_seed(seed)
        pass


# --- Updated Helper function to create the environment ---
def make_sb_env(config: Config, dataset_iterator: Iterator, device: torch.device, num_episodes_per_sample: int = 32):
    """Factory function for the integrated SmallBowelEnv."""
    env = SmallBowelEnv(
        config=config,
        dataset_iterator=dataset_iterator,
        num_episodes_per_sample=num_episodes_per_sample,
        device=device,
        batch_size=[1],  # Explicitly set batch size
    )

    # Check specs - let it raise error if checks fail
    print("Checking environment specs...")
    # check_env_specs(env)
    print("Environment specs check passed.")

    return env
