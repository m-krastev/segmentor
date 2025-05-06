"""
Environment implementation for Navigator's small bowel path tracking,
integrated with TorchRL. Simplified version without try-except blocks.
"""

from math import dist
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pyvista as pv
import torch
from skimage.draw import line_nd
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    Binary,
    BoundedContinuous,
    Composite,
    UnboundedContinuous,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs

from .config import Config
from .utils import (
    BinaryDilation3D,
    ClipTransform,
    compute_gdt,
    draw_path_sphere,
    get_patch,
)


# Define action spec constants
ACTION_DIM = 3  # 3D movement delta


class SmallBowelEnv(EnvBase):
    """
    TorchRL-compatible environment for RL-based small bowel path tracking.

    Relies on upstream data validity and will raise exceptions on errors.
    """

    goal: Tuple[int, int, int]
    current_pos_vox: Tuple[int, int, int]
    start_coord: Tuple[int, int, int]
    end_coord: Tuple[int, int, int]
    gt_path_available: bool
    image: torch.Tensor
    seg: torch.Tensor
    wall_map: torch.Tensor
    gt_path_voxels: np.ndarray
    gt_path_vol: torch.Tensor
    cumulative_path_mask: torch.Tensor
    gdt: torch.Tensor
    reward_map: torch.Tensor
    spacing: Optional[Tuple[float, float, float]]
    image_affine: Optional[np.ndarray]
    current_step_count: int
    tracking_path_history: List[Tuple[int, int, int]]

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
                shape=torch.Size([*batch_size, 3, *config.patch_size_vox]),
                dtype=torch.float32,
                device=device,
            ),
            critic=UnboundedContinuous(
                shape=torch.Size([*batch_size, 3, *config.patch_size_vox]),
                dtype=torch.float32,
                device=device,
            ),
            final_coverage=BoundedContinuous(
                low=0,
                high=1,
                shape=torch.Size([*batch_size, 1]),
                dtype=torch.float32,
                device=device,
            ),
            final_step_count=UnboundedContinuous(
                shape=torch.Size([*batch_size, 1]), dtype=torch.int64, device=device
            ),
            total_reward=UnboundedContinuous(
                shape=torch.Size([*batch_size, 1]), dtype=torch.float32, device=device
            ),
            max_gdt_achieved=UnboundedContinuous(
                shape=torch.Size([*batch_size, 1]), dtype=torch.float32, device=device
            ),
            shape=batch_size,
            device=device,
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedContinuous(
            low=0,
            high=1,
            shape=torch.Size([*batch_size, 3]),
            dtype=torch.float32,
            device=device,
        )
        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([*batch_size, 1]),
            dtype=torch.float32,
            device=device,
        )
        self.done_spec = Composite(
            done=Binary(shape=torch.Size([*batch_size, 1]), dtype=torch.bool, device=device),
            terminated=Binary(shape=torch.Size([*batch_size, 1]), dtype=torch.bool, device=device),
            truncated=Binary(shape=torch.Size([*batch_size, 1]), dtype=torch.bool, device=device),
            shape=batch_size,
            device=device,
        )

        self.max_gdt_achieved = 0.0
        self.cum_reward = torch.tensor(0.0, device=self.device)

        # --- TorchRL Internal State Flags (per-batch element) ---
        self._is_done = torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)

        self.transform = ClipTransform(-150, 250)
        self.transform.compile()

        # Placeholder tensor
        self.zeros_patch = torch.zeros(self.config.patch_size_vox, device=self.device)
        self.placeholder_zeros = torch.zeros_like(self._is_done, dtype=torch.float32)
        self.dilation = (
            torch.nn.Sequential(*[
                BinaryDilation3D() for _ in range(config.cumulative_path_radius_vox // 3)
            ])
            .to(self.device)
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
            local_peaks=subject_data.get("local_peaks"),
        )

        # Check if critical data was loaded successfully by update_data
        assert self.image is not None and self.seg_np is not None and self.wall_map is not None, (
            f"Critical data is None after loading subject {subject_data.get('id', 'N/A')}."
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
        local_peaks: Optional[np.ndarray] = None,
        # Receive numpy versions if provided by dataset
    ):
        """
        Updates the environment's internal data stores using data from the dataset.
        """
        # Store tensors directly
        self.image = self.transform(torch.from_numpy(image).to(self.device))
        self.seg_np = seg  # Keep numpy version for reference

        self.seg = torch.from_numpy(seg).to(
            device=self.device, dtype=torch.uint8
        )  # Ensure correct type
        self.seg_volume = torch.sum(self.seg).item()
        self.wall_map = torch.from_numpy(wall_map).to(self.device)
        self.gdt_start = gdt_start
        self.gdt_end = gdt_end
        self.cumulative_path_mask = torch.zeros_like(self.image, device=self.device)
        self.local_peaks = local_peaks

        # self.zeros_cache = torch.zeros_like(self.seg)

        # Store other metadata
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.image_affine = image_affine if image_affine is not None else np.eye(4)
        self.start_coord = start_coord
        self.end_coord = end_coord

        # Update ground truth path if provided
        self.gt_path_voxels = gt_path
        # self._process_gt_path()

    def _process_gt_path(self):
        """Process ground truth path data."""
        self.gt_path_vol = torch.zeros_like(self.image)
        if self.gt_path_voxels is None:
            return

        valid_indices = (
            (self.gt_path_voxels[:, 0] >= 0)
            & (self.gt_path_voxels[:, 0] < self.image.shape[0])
            & (self.gt_path_voxels[:, 1] >= 0)
            & (self.gt_path_voxels[:, 1] < self.image.shape[1])
            & (self.gt_path_voxels[:, 2] >= 0)
            & (self.gt_path_voxels[:, 2] < self.image.shape[2])
        )
        valid_gt_path = self.gt_path_voxels[valid_indices]
        assert valid_gt_path.shape[0] > 0, "No valid GT path voxels found within image bounds."
        self.gt_path_vol[valid_gt_path[:, 0], valid_gt_path[:, 1], valid_gt_path[:, 2]] = 1.0

    def _get_state_patches(self) -> Dict[str, torch.Tensor]:
        """Get state patches centered at current position. Assumes tensors are valid."""
        # GT path can be None if not available, handle it
        # if self.gt_path_vol is None:
        #     # Create a zero patch matching the spec, assuming self.image exists for shape/device
        #     gt_path_patch = self.zeros_patch
        # else:
        #     gt_path_patch = get_patch(
        #         self.gt_path_vol, self.current_pos_vox, self.config.patch_size_vox
        #     )

        img_patch = get_patch(self.image, self.current_pos_vox, self.config.patch_size_vox)
        wall_patch = get_patch(self.wall_map, self.current_pos_vox, self.config.patch_size_vox)
        cum_path_patch = get_patch(
            self.cumulative_path_mask, self.current_pos_vox, self.config.patch_size_vox
        )

        # Stack patches
        actor_state = torch.stack([img_patch, wall_patch, cum_path_patch], dim=0)
        critic_state = torch.stack([img_patch, wall_patch, cum_path_patch], dim=0)
        return {"actor": actor_state, "critic": critic_state}

    def _is_valid_pos(self, pos_vox: Tuple[int, int, int]) -> bool:
        """Check if a position is within the volume bounds."""
        s = self.image.shape
        return (0 <= pos_vox[0] < s[0]) and (0 <= pos_vox[1] < s[1]) and (0 <= pos_vox[2] < s[2])

    def _get_final_coverage(self) -> float:
        """Calculate coverage. Assumes tensors are valid."""
        mask = self.dilation(self.cumulative_path_mask.unsqueeze(0).unsqueeze(0)).squeeze()
        intersection = torch.sum(mask * self.seg)
        return intersection / (self.seg_volume + mask.sum())

    def get_tracking_history(self) -> np.ndarray:
        """Get the history of tracked positions."""
        return np.array(self.tracking_path_history)

    def save_path(self):
        cache_dir = Path("results") / self._current_subject_data["id"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        nif = nib.nifti1.Nifti1Image(
            self.cumulative_path_mask.numpy(force=True),
            affine=self.image_affine,
            header=nib.Nifti1Header(),
        )
        nif.header.set_zooms(self.spacing)
        nib.save(nif, cache_dir / "cumulative_path_mask.nii.gz")

        # Save the tracking history
        tracking_history_path = cache_dir / "tracking_history.npy"
        np.save(tracking_history_path, self.tracking_path_history)

        # PyVista visualization
        plotter = pv.Plotter()
        plotter.add_volume(self.seg_np * 20, cmap="viridis", opacity="linear")
        plotter.add_volume(
            self._current_subject_data["colon"] * 80, cmap="viridis", opacity="linear"
        )
        plotter.add_volume(self.reward_map, opacity="linear")
        plotter.add_volume(
            self._current_subject_data["duodenum"] * 140,
            cmap="viridis",
            opacity="linear",
        )
        lines: pv.PolyData = pv.lines_from_points(self.tracking_path_history)
        plotter.add_mesh(lines, line_width=10, cmap="viridis")
        plotter.add_points(np.array(self.tracking_path_history), color="blue", point_size=10)
        plotter.show_axes()
        plotter.export_html(cache_dir / "path.html")

    def _calculate_reward(
        self, action_vox: Tuple[int, int, int], next_pos_vox: Tuple[int, int, int]
    ) -> Tuple[float, Tuple]:
        """Calculate the reward for the current step."""

        rt = torch.tensor(0.0, device=self.device)
        # --- 1. Zero movement or goes out of the segmentation penalty ---
        if not any(action_vox) or not self._is_valid_pos(next_pos_vox):
            return rt.fill_(-self.config.r_zero_mov), ()

        # Set of voxels S on the line segment
        S = line_nd(self.current_pos_vox, next_pos_vox, endpoint=True)

        # --- 2. GDT-based reward ---
        next_gdt_val = self.gdt[next_pos_vox]
        # If there is some progress
        if next_gdt_val > self.max_gdt_achieved:
            delta = next_gdt_val - self.max_gdt_achieved
            # Penalty if too large, reward if within margins
            rt += (
                -self.config.r_val2
                if delta > self.config.gdt_max_increase_theta
                else self.config.r_val2 * delta / self.config.gdt_max_increase_theta
            )
            self.max_gdt_achieved = next_gdt_val

        # 2.5 Peaks-based reward
        rt += self.reward_map[S].sum() * self.config.r_peaks
        # Discard the reward for visited nodes
        self.reward_map[S] = 0

        # --- 3. Wall-based penalty ---
        rt -= self.config.r_val2 * 2 * self.wall_map[S].mean()

        # --- 4. Revisiting penalty ---
        rt -= self.config.r_val1 * self.cumulative_path_mask[S].sum().bool()

        if not self.seg_np[next_pos_vox]:
            rt -= self.config.r_val1

        return rt, S

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, must_load_new_subject=False
    ) -> TensorDictBase:
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

        if should_load_new_subject or must_load_new_subject:
            # Reset counter *before* loading, as loading signifies starting fresh
            self.episodes_on_current_subject = 0
            self._load_next_subject()

        # --- Reset internal episode state ---
        self.current_step_count = 0

        # Determine start position and select appropriate GDT
        if self.episodes_on_current_subject % 2 == 0:
            self.current_pos_vox = self.start_coord
            self.goal = self.end_coord
            self.gdt = self.gdt_start
        else:
            self.current_pos_vox = self.end_coord
            self.goal = self.start_coord
            self.gdt = self.gdt_end

        # Validate start position (simplified check)
        if not self._is_valid_pos(self.current_pos_vox) or not self.seg_np[self.current_pos_vox]:
            print(
                f"Warning: Chosen start pos {self.current_pos_vox} invalid/outside seg. Searching nearby..."
            )
            valid_voxels = np.nonzero(self.seg_np > 0)
            if len(valid_voxels) == 0:
                raise ValueError("Cannot reset: No valid voxels found in segmentation mask.")
            self.current_pos_vox = tuple(
                valid_voxels[random.randint(0, len(valid_voxels) - 1)].tolist()
            )
            self.goal = self.end_coord
            self.gdt = compute_gdt(self.seg_np, self.current_pos_vox, self.spacing)

        # Initialize path tracking
        self.cumulative_path_mask.zero_()

        # Draw initial path sphere
        draw_path_sphere(
            self.cumulative_path_mask,
            self.current_pos_vox,
            self.config.cumulative_path_radius_vox,
        )

        # self.cumulative_path_mask[self.current_pos_vox] = 1
        # self.cumulative_path_mask = self.dilation(
        #     self.cumulative_path_mask.unsqueeze(0).unsqueeze(0)
        # ).squeeze()

        self.tracking_path_history = [self.current_pos_vox]

        # Initialize current GDT value using the selected GDT
        self.cum_reward.fill_(0)
        self.max_gdt_achieved = self.gdt[self.current_pos_vox]
        self.reward_map = np.zeros_like(self.seg_np)
        self.reward_map[tuple(self.local_peaks.T)] = 1

        # --- Get Initial State Patches ---
        obs_dict = self._get_state_patches()  # Let it raise RuntimeError if patches fail

        # --- Update Internal Done Flag and Package Output ---
        self._is_done.fill_(False)

        reset_td = TensorDict(
            {
                "actor": obs_dict["actor"].unsqueeze(0),  # Add batch dim
                "critic": obs_dict["critic"].unsqueeze(0),  # Add batch dim
                "done": self._is_done.clone(),
                "terminated": self._is_done.clone(),
                "truncated": self._is_done.clone(),
                "final_step_count": torch.zeros_like(self._is_done, dtype=torch.int64),
                "final_coverage": self.placeholder_zeros,
                "total_reward": self.placeholder_zeros,
                "max_gdt_achieved": torch.as_tensor(self.max_gdt_achieved, dtype=torch.float32, device=self.device).view_as(self._is_done)
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
        action_mapped = (2 * action_normalized - 1) * self.config.max_step_vox
        action_vox_delta = tuple(torch.round(action_mapped).int().cpu().tolist())

        # Execute Step Logic
        self.current_step_count += 1
        next_pos_vox = (
            self.current_pos_vox[0] + action_vox_delta[0],
            self.current_pos_vox[1] + action_vox_delta[1],
            self.current_pos_vox[2] + action_vox_delta[2],
        )

        # Calculate reward (needs to be before moving to the next pos)
        reward, S = self._calculate_reward(action_vox_delta, next_pos_vox)

        # Check validity and update cumulative path
        is_next_pos_valid_seg = (
            self._is_valid_pos(next_pos_vox)  # and self.seg[next_pos_vox] > 0
        )
        if is_next_pos_valid_seg:
            # self.zeros_cache[self.current_pos_vox] = 1
            # self.zeros_cache[S] = 1
            # self.zeros_cache = self.dilation(
            #     self.zeros_cache.unsqueeze(0).unsqueeze(0)
            # ).squeeze()
            # self.cumulative_path_mask += self.zeros_cache
            # self.zeros_cache.zero_()
            for vox in zip(*S):
                draw_path_sphere(
                    self.cumulative_path_mask,
                    vox,
                    self.config.cumulative_path_radius_vox,
                )

            self.tracking_path_history.append(next_pos_vox)
            self.current_pos_vox = next_pos_vox

        # Check Termination Conditions
        done, terminated, truncated = False, False, False
        termination_reason = ""
        if self.current_step_count >= self.config.max_episode_steps:
            done, truncated, termination_reason = True, True, "max_steps"
        elif not is_next_pos_valid_seg:
            done, terminated, termination_reason = (
                True,
                True,
                "out_of_bounds",
            )
        elif dist(next_pos_vox, self.goal) < self.config.cumulative_path_radius_vox:
            done, terminated, termination_reason = True, True, "reached_goal"

        # Final Reward Adjustment
        final_coverage = 0
        if done:
            final_coverage = self._get_final_coverage()
            reward += (
                (final_coverage * self.config.r_final)
                if termination_reason == "reached_goal"
                else (final_coverage - 1) * self.config.r_final
            )

        # Get Next State Patches
        next_obs_dict = self._get_state_patches()

        # Update Internal Done Flag
        self._is_done[:] = done

        # Prepare Tensors for Output TensorDict (Shape [1, ...])
        self.cum_reward += reward
        _reward = reward.view_as(self._is_done)  # B, T, 1
        output_td = TensorDict(
            {
                "actor": next_obs_dict["actor"].unsqueeze(0),
                "critic": next_obs_dict["critic"].unsqueeze(0),
                "reward": _reward,
                "done": torch.as_tensor(done, device=self.device).view_as(_reward),
                "terminated": torch.as_tensor(terminated, device=self.device).view_as(_reward),
                "truncated": torch.as_tensor(truncated, device=self.device).view_as(_reward),
                "final_coverage": torch.as_tensor(final_coverage, device=self.device).view_as(
                    _reward
                )
                if done
                else self.placeholder_zeros,
                "final_step_count": torch.as_tensor(
                    self.current_step_count, device=self.device
                ).view_as(_reward)
                if done
                else torch.as_tensor(0, device=self.device).view_as(_reward),
                "total_reward": self.cum_reward.view_as(_reward)
                if done
                else self.placeholder_zeros,
                "max_gdt_achieved": torch.as_tensor(
                    self.max_gdt_achieved, dtype=torch.float32, device=self.device
                ).view_as(_reward),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return output_td

    def _set_seed(self, seed: Optional[int] = None):
        """Sets the seed for the environment's random number generator(s)."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.manual_seed_all(seed)


# --- Updated Helper function to create the environment ---
def make_sb_env(
    config: Config,
    dataset_iterator: Iterator,
    device: torch.device = None,
    num_episodes_per_sample: int = 32,
):
    """Factory function for the integrated SmallBowelEnv."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SmallBowelEnv(
        config=config,
        dataset_iterator=dataset_iterator,
        num_episodes_per_sample=num_episodes_per_sample,
        device=device,
        batch_size=[1],  # Explicitly set batch size
    )

    # Check specs - let it raise error if checks fail
    # check_env_specs(env)

    return env
