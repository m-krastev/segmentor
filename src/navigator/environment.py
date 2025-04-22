"""
Environment implementation for Navigator's small bowel path tracking,
integrated with TorchRL. Simplified version without try-except blocks.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List, Iterator

from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    BoundedContinuous,
    UnboundedContinuous,
    Binary,
    Composite,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs  # For checking implementation

from skimage.draw import line_nd

from .config import Config
from .utils import compute_gdt, get_patch, draw_path_sphere

# from scipy.ndimage import grey_dilation, generate_binary_structure
import cupy as cp
from cupyx.scipy.ndimage import grey_dilation, generate_binary_structure
# Assume dataset is available for type hinting, but iterator is passed in
# from .dataset import SmallBowelDataset


# Define action spec constants
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_DIM = 3  # 3D movement delta

class ClipTransform(torch.nn.Module):
    def __init__(self, _min, _max):
        super().__init__()
        self.min = _min
        self.max = _max

    def forward(self, x):
        return (torch.clamp(x, self.min, self.max) - self.min) / (self.max - self.min)
    

class SmallBowelEnv(EnvBase):
    """
    TorchRL-compatible environment for RL-based small bowel path tracking.

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
        # Set the specs *after* calling super().__init__
        self.observation_spec = Composite(
            actor=UnboundedContinuous(
                shape=torch.Size([
                    *batch_size,
                    3,
                    *config.patch_size_vox,
                ]),  # B, C, D, H, W, Z
                dtype=torch.float32,
                device=device,
            ),
            critic=UnboundedContinuous(
                shape=torch.Size([
                    *batch_size,
                    4,
                    *config.patch_size_vox,
                ]),  # B, C, D, H, W, Z
                dtype=torch.float32,
                device=device,
            ),
            shape=batch_size,
            device=device,
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedContinuous(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=torch.Size([*batch_size, ACTION_DIM]),  # B, ActionDim
            dtype=torch.float32,
            device=device,
        )
        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([*batch_size, 1]),  # B, 1
            dtype=torch.float32,
            device=device,
        )
        self.done_spec = Composite(
            done=Binary(
                shape=torch.Size([*batch_size, 1]),  # boolean done state (0 or 1)
                dtype=torch.bool,
                device=device,
            ),
            terminated=Binary(
                shape=torch.Size([*batch_size, 1]),  # boolean terminated state (0 or 1)
                dtype=torch.bool,
                device=device,
            ),
            truncated=Binary(
                shape=torch.Size([*batch_size, 1]),  # boolean truncated state (0 or 1)
                dtype=torch.bool,
                device=device,
            ),
            shape=batch_size,
            device=device,
        )
        # self.info_spec = Composite()

        # --- Initialize Internal State (per-instance, not per-batch element) ---
        self.current_pos_vox: Tuple[int, int, int] = (0, 0, 0)
        self.start_coord: Tuple[int, int, int] = (0, 0, 0)
        self.end_coord: Tuple[int, int, int] = (0, 0, 0)
        self.gt_path_voxels = None
        self.gt_path_available = False
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
        self._is_done = torch.ones(
            self.batch_size[0], 1, dtype=torch.bool, device=self.device
        )

        self.transform = torch.jit.script(ClipTransform(-150, 250))
        self.zeros_patch = torch.zeros(
            self.config.patch_size_vox, dtype=torch.float32, device=self.device
        )

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
        )

        # Check if critical data was loaded successfully by update_data
        if (
            self.image is None
            or self.seg is None
            or self.wall_map is None
            or self.gdt_start is None
            or self.gdt_end is None
        ):
            raise RuntimeError(
                f"Critical data is None after loading subject {subject_id}."
            )

        return True

    # --- Internal Data Update Method ---
    def update_data(
        self,
        image: np.ndarray,
        seg: np.ndarray,
        wall_map: np.ndarray,  # Receive wall_map tensor
        gdt_start: np.ndarray,  # Receive GDT tensors
        gdt_end: np.ndarray,
        start_coord: Tuple[int, int, int],  # Receive coords
        end_coord: Tuple[int, int, int],
        gt_path: Optional[np.ndarray] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
        image_affine: Optional[np.ndarray] = None,
        # Receive numpy versions if provided by dataset
    ):
        """
        Updates the environment's internal data stores using data from the dataset.
        """
        # Store tensors directly
        self.image = self.transform(torch.from_numpy(image).to(self.device))
        
        self.seg = torch.from_numpy(seg).to(
            device=self.device, dtype=torch.uint8
        )  # Ensure correct type
        self.seg_volume = torch.sum(self.seg).item()
        self.seg_np = seg  # Keep numpy version for reference
        self.wall_map = torch.from_numpy(wall_map).to(self.device)
        self.gdt_start = torch.from_numpy(gdt_start).to(self.device)
        self.gdt_end = torch.from_numpy(gdt_end).to(self.device)

        # Store other metadata
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.image_affine = image_affine if image_affine is not None else np.eye(4)
        self.start_coord = start_coord
        self.end_coord = end_coord

        # Update ground truth path if provided
        self.gt_path_voxels = gt_path if gt_path is not None else None
        self._process_gt_path()

    def _process_gt_path(self):
        """Process ground truth path data."""
        if self.gt_path_voxels is None:
            self._create_empty_gt_path()
            return

        self.gt_path_available = True
        self.gt_path_vol = torch.zeros_like(self.image)
        valid_indices = (
            (self.gt_path_voxels[:, 0] >= 0)
            & (self.gt_path_voxels[:, 0] < self.image.shape[0])
            & (self.gt_path_voxels[:, 1] >= 0)
            & (self.gt_path_voxels[:, 1] < self.image.shape[1])
            & (self.gt_path_voxels[:, 2] >= 0)
            & (self.gt_path_voxels[:, 2] < self.image.shape[2])
        )
        valid_gt_path = self.gt_path_voxels[valid_indices]
        if valid_gt_path.shape[0] > 0:
            self.gt_path_vol[
                valid_gt_path[:, 0], valid_gt_path[:, 1], valid_gt_path[:, 2]
            ] = 1.0
        else:
            print("Warning: No valid GT path voxels found within image bounds.")

    def _create_empty_gt_path(self):
        """Create an empty ground truth path tensor."""
        self.gt_path_available = False
        self.gt_path_vol = torch.zeros_like(self.image)

    def _get_state_patches(self) -> Dict[str, torch.Tensor]:
        """Get state patches centered at current position. Assumes tensors are valid."""
        # GT path can be None if not available, handle it
        if self.gt_path_vol is None:
            # Create a zero patch matching the spec, assuming self.image exists for shape/device
            gt_path_patch = self.zeros_patch
        else:
            gt_path_patch = get_patch(
                self.gt_path_vol, self.current_pos_vox, self.config.patch_size_vox
            )

        img_patch = get_patch(
            self.image, self.current_pos_vox, self.config.patch_size_vox
        )
        wall_patch = get_patch(
            self.wall_map, self.current_pos_vox, self.config.patch_size_vox
        )
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
        s = self.image.shape
        return (
            0 <= pos_vox[0] < s[0] and 0 <= pos_vox[1] < s[1] and 0 <= pos_vox[2] < s[2]
        )

    def _get_final_coverage(self) -> float:
        """Calculate coverage. Assumes tensors are valid."""


        path_mask = torch.as_tensor(
            grey_dilation(
                cp.asarray(self.cumulative_path_mask),
                footprint=generate_binary_structure(3, 2),
            ),
            device=self.device,
        )
        intersection = torch.sum(self.cumulative_path_mask * self.seg).item()
        return intersection / self.seg_volume

    def get_tracking_history(self) -> np.ndarray:
        """Get the history of tracked positions."""
        return np.array(self.tracking_path_history)

    def _calculate_reward(
        self, action_vox: Tuple[int, int, int], next_pos_vox: Tuple[int, int, int]
    ) -> float:
        """Calculate the reward for the current step. Assumes tensors are valid."""

        rt = 0.0
        # Set of voxels S on the line segment
        S = line_nd(self.current_pos_vox, next_pos_vox, endpoint=True)
        # from segmentor.utils.medutils import plotLine3d
        # a = self.current_pos_vox
        # b = next_pos_vox
        # S = plotLine3d(a[0], a[1], a[2], b[0], b[1], b[2])

        # --- 1. Zero movement or goes out of the image penalty ---
        if np.all(action_vox == 0) or not self._is_valid_pos(next_pos_vox):
            return -self.config.r_val1

        # --- 2. GDT-based reward ---
        next_gdt_val = self.gdt[next_pos_vox]
        # If there is some progress
        if next_gdt_val > self.max_gdt_achieved:
            delta = next_gdt_val - self.max_gdt_achieved
            # Penalty if too large
            if delta > self.config.gdt_max_increase_theta:
                rt = -self.config.r_val2
            else:
                rt = self.config.r_val2 * delta / self.config.gdt_max_increase_theta
            self.max_gdt_achieved = next_gdt_val
        # --- 3. Wall-based penalty ---
        rt -= self.config.r_val2 * self.wall_map[S].mean().item()

        # --- 4. Revisiting penalty ---
        if self.cumulative_path_mask[S].sum() > 0:
            rt -= self.config.r_val1

        # --- 5. Out-of-segmentation penalty ---
        if not self.seg[next_pos_vox]:
            rt -= self.config.r_val1

        return rt

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
            self._load_next_subject()

        # --- Reset internal episode state ---
        self.current_step_count = 0

        # Determine start position and select appropriate GDT
        if not self.episodes_on_current_subject % 2:
            self.current_pos_vox = self.start_coord
            self.gdt = self.gdt_start
        else:  # Default to start
            self.current_pos_vox = self.end_coord
            self.gdt = self.gdt_end

        # Validate start position (simplified check)
        if (
            not self._is_valid_pos(self.current_pos_vox)
            or self.seg[self.current_pos_vox] == 0
        ):
            print(
                f"Warning: Chosen start pos {self.current_pos_vox} invalid/outside seg. Searching nearby..."
            )
            valid_voxels = torch.nonzero(self.seg > 0)
            if len(valid_voxels) == 0:
                raise ValueError(
                    "Cannot reset: No valid voxels found in segmentation mask."
                )
            rand_idx = torch.randint(0, len(valid_voxels), (1,)).item()
            self.current_pos_vox = tuple(valid_voxels[rand_idx].tolist())
            self.gdt = compute_gdt(self.seg.numpy(), self.current_pos_vox, self.spacing)

        # Initialize path tracking
        self.cumulative_path_mask = torch.zeros_like(
            self.image, dtype=torch.float32, device=self.device
        )

        # Draw initial path sphere
        draw_path_sphere(
            self.cumulative_path_mask,
            self.current_pos_vox,
            self.config.cumulative_path_radius_vox,
        )
        self.tracking_path_history = [self.current_pos_vox]

        # Initialize current GDT value using the selected GDT
        self.current_gdt_val = self.gdt[self.current_pos_vox].item()
        self.max_gdt_achieved = self.current_gdt_val

        # --- Get Initial State Patches ---
        obs_dict = (
            self._get_state_patches()
        )  # Let it raise RuntimeError if patches fail

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

        # Extract Action
        action_normalized = tensordict.get("action").squeeze(0)

        # Map Action
        action_mapped = (2* action_normalized - 1)* self.config.max_step_vox
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
            self._is_valid_pos(next_pos_vox) and self.seg[next_pos_vox] > 0
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
            done, terminated, termination_reason = (
                True,
                True,
                "out_of_bounds_or_segmentation",
            )
        elif (
            self._is_valid_pos(self.end_coord)
            and np.linalg.norm(
                np.array(self.current_pos_vox) - np.array(self.end_coord)
            )
            < self.config.cumulative_path_radius_vox
        ):
            done, terminated, termination_reason = True, True, "reached_end"

        # Final Reward Adjustment
        if done:
            final_coverage = self._get_final_coverage()
            final_reward_adjustment = (
                (final_coverage * self.config.r_final)
                if termination_reason == "reached_end"
                else ((1 - final_coverage) * abs(self.config.r_final))
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
        # _next_ep_reward = torch.tensor(
        #     [[reward if done else 0.0]], dtype=torch.float32, device=self.device
        # )
        # _next_ep_length = torch.tensor(
        #     [[self.current_step_count if done else 0]], dtype=torch.int, device=self.device
        # )
        # _next_ep_coverage = torch.tensor(
        #     [[final_coverage if done else 0.0]], dtype=torch.float32, device=self.device
        # )
        output_td = TensorDict(
            {
                "actor": _next_actor_obs,
                "critic": _next_critic_obs,
                "reward": _reward,
                "done": _done,
                "terminated": _terminated,
                "truncated": _truncated,
                # "next": next_state_td,
                # "info_coverage": _next_ep_coverage,
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
def make_sb_env(
    config: Config,
    dataset_iterator: Iterator,
    device: torch.device,
    num_episodes_per_sample: int = 32,
):
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
    check_env_specs(env)
    print("Environment specs check passed.")

    return env
