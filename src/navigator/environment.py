"""
Environment implementation for Navigator's small bowel path tracking,
integrated with TorchRL. Simplified version without try-except blocks.
"""

from rich import print
from itertools import cycle
from math import dist, isfinite
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pyvista as pv
import torch
from torch import nn
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
from torch.utils.data import Dataset, DataLoader
from segmentor.utils.medutils import save_nifti

from .config import Config
from .utils import (
    BinaryDilation3D,
    ClipTransform,
    compute_gdt,
    draw_path_sphere,
    draw_path_sphere_2,
    get_patch,
    Coords,
    Spacing,
)

from enum import IntEnum


class TReason(IntEnum):
    NOT_DONE = 0
    GOAL_REACHED = 1
    MAX_STEPS = 2
    OOB = 3


# Define action spec constants
ACTION_DIM = 3  # 3D movement delta


class SmallBowelEnv(EnvBase):
    """
    TorchRL-compatible environment for RL-based small bowel path tracking.

    Relies on upstream data validity and will raise exceptions on errors.
    """

    goal: Coords
    current_pos_vox: Coords
    start_coord: Coords
    end_coord: Coords
    image: torch.Tensor
    seg: torch.Tensor
    wall_map: torch.Tensor
    gt_path_voxels: np.ndarray
    gt_path_vol: torch.Tensor
    cumulative_path_mask: torch.Tensor
    gdt: np.ndarray
    reward_map: np.ndarray
    spacing: Optional[Spacing]
    image_affine: Optional[np.ndarray]
    current_step_count: int
    tracking_path_history: List[Coords]
    goal: Coords = (0, 0, 0)

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
        self.dataset_iterator = dataset_iterator
        self._current_subject_data = None
        self.num_episodes_per_sample = num_episodes_per_sample // batch_size.numel()
        # Counter for episodes on current subject, set so that it refreshes at next step
        self.episodes_on_current_subject = num_episodes_per_sample // batch_size.numel()

        # --- Define Specs ---
        # Set the specs *after* calling super().__init__
        self.dtype = torch.float32
        self._set_specs()

        # --- TorchRL Internal State Flags (per-batch element) ---
        self._is_done = torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)

        # Transforms
        self.ct_transform = torch.compile(ClipTransform(30 - 150, 30 + 150))  # -150, 250
        self.wall_transform = torch.compile(ClipTransform(0.0, 0.1))
        self.dilation = torch.compile(
            torch.nn.Sequential(*[BinaryDilation3D()] * config.cumulative_path_radius_vox)
        )
        self.dilation.to(self.device)
        self.maxarea_dilation = torch.compile(torch.nn.Sequential(*[BinaryDilation3D()] * 10))
        self.maxarea_dilation.to(self.device)

        # Placeholder tensor
        self.zeros_patch = torch.zeros(self.config.patch_size_vox, device=self.device)
        self.placeholder_zeros = torch.zeros_like(self._is_done, dtype=self.dtype)

    def _set_specs(self):
        self.observation_spec = Composite(
            actor=UnboundedContinuous(
                shape=torch.Size([*self.batch_size, 4, *self.config.patch_size_vox]),
                dtype=self.dtype,
            ),
            info=Composite(
                final_coverage=BoundedContinuous(
                    low=0, high=1, shape=torch.Size([*self.batch_size, 1]), dtype=self.dtype
                ),
                final_step_count=UnboundedContinuous(
                    shape=torch.Size([*self.batch_size, 1]), dtype=self.dtype
                ),
                final_length=UnboundedContinuous(
                    shape=torch.Size([*self.batch_size, 1]), dtype=self.dtype
                ),
                final_wall_gradient=UnboundedContinuous(
                    shape=torch.Size([*self.batch_size, 1]), dtype=self.dtype
                ),
                total_reward=UnboundedContinuous(
                    shape=torch.Size([*self.batch_size, 1]), dtype=self.dtype
                ),
                max_gdt_achieved=UnboundedContinuous(
                    shape=torch.Size([*self.batch_size, 1]), dtype=self.dtype
                ),
            ),
            shape=self.batch_size,
        )
        self.action_spec = BoundedContinuous(
            low=0, high=1, shape=torch.Size([*self.batch_size, 3]), dtype=self.dtype
        )
        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([*self.batch_size, 1]), dtype=self.dtype
        )
        self.done_spec = Composite(
            done=Binary(shape=torch.Size([*self.batch_size, 1]), dtype=torch.bool),
            terminated=Binary(shape=torch.Size([*self.batch_size, 1]), dtype=torch.bool),
            truncated=Binary(shape=torch.Size([*self.batch_size, 1]), dtype=torch.bool),
            shape=self.batch_size,
        )

    # --- Data Loading Method (Internal) ---
    def _load_next_subject(self) -> bool:
        """
        Loads the next subject's data from the iterator.
        Raises StopIteration if iterator is exhausted.
        Raises other exceptions if data loading/processing fails.
        """
        subject_data: dict = next(self.dataset_iterator)  # Let StopIteration propagate
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
        assert self.image is not None and self.seg is not None and self.wall_map is not None, (
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
        start_coord: Coords,  # Receive coords
        end_coord: Coords,
        gt_path: Optional[np.ndarray] = None,
        spacing: Optional[Spacing] = None,
        image_affine: Optional[np.ndarray] = None,
        local_peaks: Optional[np.ndarray] = None,
        # Receive numpy versions if provided by dataset
    ):
        """
        Updates the environment's internal data stores using data from the dataset.
        """
        # Store tensors directly
        self.image = self.ct_transform(torch.from_numpy(image).to(self.device)).to(self.dtype)
        # save_nifti(np.transpose(image, (2,1,0)), f"image_{self._current_subject_data['id']}.nii.gz", affine=image_affine, spacing=spacing[::-1])
        # save_nifti(np.transpose(self.image.numpy(force=True), (2,1,0)), f"image_transformed_{self._current_subject_data['id']}.nii.gz", affine=image_affine, spacing=spacing[::-1])
        self.seg = torch.from_numpy(seg).to(device=self.device, dtype=torch.uint8)
        self.seg = self.dilation(self.seg.unsqueeze(0).unsqueeze(0)).squeeze()
        # self.seg[tuple(start_coord)] = 3
        # self.seg[tuple(end_coord)] = 3
        # save_nifti(np.transpose(self.seg.numpy(force=True), (2,1,0)), f"seg_transformed_{self._current_subject_data['id']}.nii.gz", affine=image_affine, spacing=spacing)
        self.seg_volume = torch.sum(self.seg).item()
        self.wall_map = self.wall_transform(
            torch.from_numpy(wall_map).to(device=self.device, dtype=self.dtype)
        )
        self.gdt_start = gdt_start
        self.gdt_end = gdt_end
        self.cumulative_path_mask = torch.zeros_like(self.seg)
        self.local_peaks = local_peaks[30:-30]  # Restrict to center positions TODO: Remove sometime
        # self.image[tuple(local_peaks.T)] = 0.5
        self.reward_map = np.zeros_like(self.gdt_start, dtype=np.uint8)
        self.allowed_area = (
            self.maxarea_dilation(self.seg.unsqueeze(0).unsqueeze(0)).squeeze().numpy(force=True)
        )

        # Store other metadata
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.image_affine = image_affine if image_affine is not None else np.eye(4)
        self.start_coord = tuple(start_coord)
        self.end_coord = tuple(end_coord)

        # Update ground truth path if provided
        self.gt_path_voxels = gt_path
        # Process GT path data
        self.gt_path_vol = torch.zeros_like(self.seg)
        if self.gt_path_voxels is not None:
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
        _ = len(self.tracking_path_history)
        img_patch = get_patch(
            self.image, self.tracking_path_history[-1 % _], self.config.patch_size_vox
        )
        wall_patch = get_patch(self.wall_map, self.current_pos_vox, self.config.patch_size_vox)
        img_patch_1 = get_patch(
            self.image, self.tracking_path_history[-2 % _], self.config.patch_size_vox
        )
        # img_patch_2 = get_patch(
        #     self.image, self.tracking_path_history[-3 % _], self.config.patch_size_vox
        # )
        cum_path_patch = get_patch(
            self.cumulative_path_mask, self.current_pos_vox, self.config.patch_size_vox
        )
        # gt_path_patch = get_patch(
        #     self.gt_path_vol, self.current_pos_vox, self.config.patch_size_vox
        # )
        # save_nifti(np.transpose(img_patch.numpy(force=True), (2,1,0)), "img_patch.nii.gz", affine=self.image_affine, spacing=self.spacing[::-1])
        # save_nifti(np.transpose(wall_patch.numpy(force=True), (2,1,0)), "wall_patch.nii.gz", affine=self.image_affine, spacing=self.spacing[::-1])
        # save_nifti(np.transpose(cum_path_patch.numpy(force=True), (2,1,0)), "cum_path_patch.nii.gz", affine=self.image_affine, spacing=self.spacing[::-1])
        # exit()
        # Stack patches (for critic you can add another dimension and just index into it)
        # actor_state = torch.stack([img_patch, wall_patch, cum_path_patch], dim=0)
        actor_state = torch.stack([img_patch_1, img_patch, wall_patch, cum_path_patch], axis=0)
        return actor_state

    def _is_valid_pos(self, pos_vox: Coords) -> bool:
        """Check if a position is within the volume bounds."""
        s = self.image.shape
        return (0 <= pos_vox[0] < s[0]) and (0 <= pos_vox[1] < s[1]) and (0 <= pos_vox[2] < s[2])

    def _get_final_coverage(self) -> float:
        """Calculate coverage. Assumes tensors are valid."""
        intersection = torch.sum(self.cumulative_path_mask * self.seg)
        union = self.seg_volume + self.cumulative_path_mask.sum()
        return (2 * intersection / union) if union != 0 else 0

    def get_tracking_history(self) -> np.ndarray:
        """Get the history of tracked positions."""
        return np.array(self.tracking_path_history)

    def get_tracking_mask(self) -> torch.Tensor:
        """Get the cumulative path mask."""
        return self.cumulative_path_mask.clone()

    def save_path(self, save_dir: Optional[Path] = None):
        # Environment works with ZYX, so we have to do transposes here
        cache_dir = save_dir or (Path("results") / self._current_subject_data["id"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        nif = nib.nifti1.Nifti1Image(
            np.transpose(self.cumulative_path_mask.numpy(force=True), (2, 1, 0)),
            affine=self.image_affine,
        )
        nib.save(nif, cache_dir / "cumulative_path_mask.nii.gz")

        # Save the tracking history
        tracking_history_path = cache_dir / "tracking_history.npy"
        np.savetxt(tracking_history_path, np.fliplr(self.tracking_path_history), fmt="%d")

        # PyVista visualization
        plotter = pv.Plotter()
        plotter.add_volume(self.seg.numpy(force=True) * 10, cmap="viridis", opacity="linear")
        if self._current_subject_data.get("colon") is not None:
            plotter.add_volume(
                self._current_subject_data["colon"] * 80,
                cmap="viridis",
                opacity="linear",
            )
        if self._current_subject_data.get("duodenum") is not None:
            plotter.add_volume(
                self._current_subject_data["duodenum"] * 120,
                cmap="viridis",
                opacity="linear",
            )
        plotter.add_volume(self.reward_map, opacity="linear")
        lines: pv.PolyData = pv.lines_from_points(self.tracking_path_history)
        plotter.add_mesh(lines, line_width=10, cmap="viridis")
        plotter.add_points(np.array(self.tracking_path_history), color="blue", point_size=10)
        plotter.show_axes()
        plotter.export_html(cache_dir / "path.html")

    def _calculate_reward(self, action_vox: Coords, next_pos_vox: Coords) -> Tuple[float, Tuple]:
        """Calculate the reward for the current step."""

        rt = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        # --- 1. Zero movement or goes out of the segmentation penalty ---
        if not any(action_vox):
            rt -= self.config.r_val2  # self.config.r_val1
            return rt, ()

        if not self._is_valid_pos(next_pos_vox) or not self.allowed_area[next_pos_vox]:
            rt -= self.config.r_val2
            return rt, ()

        # Set of voxels S on the line segment
        S = line_nd(self.current_pos_vox, next_pos_vox)

        # --- 2. GDT-based reward ---
        next_gdt_val = self.gdt[next_pos_vox]
        # # If there is some progress
        if next_gdt_val > self.max_gdt_achieved:
            delta = next_gdt_val - self.max_gdt_achieved
            # Penalty if too large, reward if within margins
            if delta < self.config.gdt_max_increase_theta:
                # Base reward
                rt += self.config.r_val2 * delta / self.config.gdt_max_increase_theta
                # 3.3 Add shaping
                dist_before = abs(self.goal_gdt - self.max_gdt_achieved)
                dist_after = abs(self.goal_gdt - next_gdt_val)

                # Compute potentials
                phi_before = -dist_before
                phi_after = -dist_after
                shaping_bonus = self.config.gamma * phi_after - phi_before
                rt += 1 + shaping_bonus / self.config.gdt_max_increase_theta
            else:
                rt -= self.config.r_val2

            # Additional coverage reward... rt+=
            self.max_gdt_achieved = next_gdt_val

        # Survival reward
        phi_t = self.current_step_count - 1
        phi_tp1 = self.current_step_count
        rt += self.config.gamma * phi_tp1 - phi_t

        # 2.5 Peaks-based reward
        # rt += self.reward_map[S].sum() * self.config.r_peaks
        # self.reward_map[S] = 0
        # Reward for coverage (based on intersection within the segmentation on the path): ...
        rt += self.config.r_val3 * (self.seg[S] * (1-self.cumulative_path_mask[S])).float().mean()

        # --- 3. Wall-based penalty ---
        wall_map = self.wall_map[S].max()
        rt -= self.config.r_val2 * wall_map

        self.wall_gradient += wall_map


        # S always includes the start pixels, (and due to the dilation the pixels surrounding the start will always be white, therefore invoking this reward consistently)
        # --- 4. Revisiting penalty ---
        coverage = self.cumulative_path_mask[S][self.config.cumulative_path_radius_vox+1:]
        rt -= self.config.r_val3 * coverage.any()

        # --- 5. Out of seg penalty
        rt -= self.config.r_val3 * self.seg[next_pos_vox].logical_not()
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
        if (
            self.episodes_on_current_subject >= self.num_episodes_per_sample
            or must_load_new_subject
        ):
            # Reset counter *before* loading, as loading signifies starting fresh
            self.episodes_on_current_subject = 0
            self._load_next_subject()

        # --- Reset internal episode state ---
        self.current_step_count = 0
        self.current_distance_traveled = 0
        self.wall_gradient = 0

        # Determine start position and select appropriate GDT
        rand = random.randint(0, 9)  # 40-40-20
        if rand < 4:
            # Start at the beginning
            self.current_pos_vox = self.start_coord
            self.goal = self.end_coord
            self.gdt = self.gdt_start
        elif rand < 7:
            # Start at a random local peak
            self.current_pos_vox = tuple(random.choice(self.local_peaks))
            # Randomly go in either direction
            if self.episodes_on_current_subject % 2:
                self.goal = self.end_coord
                self.gdt = self.gdt_start
            else:
                self.goal = self.start_coord
                self.gdt = self.gdt_end
        else:
            # Start at the end
            self.current_pos_vox = self.end_coord
            self.goal = self.start_coord
            self.gdt = self.gdt_end

        self._start = self.current_pos_vox

        # Initialize path tracking
        self.cumulative_path_mask.zero_()
        self.cumulative_path_mask[self.current_pos_vox] = 1

        # Initialize various tracking variables
        self.tracking_path_history = [self.current_pos_vox]
        self.cum_reward = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.max_gdt_achieved = self.gdt[self.current_pos_vox]
        self.goal_gdt = self.gdt.max()
        self.reward_map[tuple(self.local_peaks.T)] = 1
        err_text = f"{self.max_gdt_achieved} at {self.current_pos_vox}, {self.start_coord}, {self.end_coord}, {self.local_peaks}, ID: {self._current_subject_data['id']}, {self.gdt.shape}"
        assert self.max_gdt_achieved >= 0 and np.isfinite(self.max_gdt_achieved), (
            f"Expected GDT>=0, got {err_text}"
        )

        # --- Get Initial State Patches ---
        obs_dict = self._get_state_patches()

        # --- Update Internal Done Flag and Package Output ---
        self._is_done.fill_(False)
        reset_td = TensorDict(
            {
                "actor": obs_dict.unsqueeze(0),  # Add batch dim
                "done": self._is_done.clone(),
                "terminated": self._is_done.clone(),
                "truncated": self._is_done.clone(),
                "info": {
                    "final_step_count": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_length": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_wall_gradient": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_coverage": self.placeholder_zeros.clone(),
                    "total_reward": self.placeholder_zeros.clone(),
                    "max_gdt_achieved": torch.as_tensor(
                        self.max_gdt_achieved, dtype=self.dtype, device=self.device
                    ).view_as(self._is_done),
                },
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return reset_td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Performs a step. Assumes env is initialized. Raises exceptions on errors."""
        self.current_step_count += 1
        # Extract Action
        action_normalized = tensordict.get("action").squeeze(0)

        # Map Action
        action_mapped = (2 * action_normalized - 1) * self.config.max_step_vox
        action_vox_delta = action_mapped.cpu().round().int().tolist()

        # Execute Step Logic
        next_pos_vox = (
            self.current_pos_vox[0] + action_vox_delta[0],
            self.current_pos_vox[1] + action_vox_delta[1],
            self.current_pos_vox[2] + action_vox_delta[2],
        )

        # Check validity and update cumulative path
        is_next_pos_valid_seg = self._is_valid_pos(next_pos_vox)

        # if not is_next_pos_valid_seg or not self.seg_np[next_pos_vox]:
        #     # Bounce in the opposite direction in the X axis
        #     next_pos_vox = (
        #         self.current_pos_vox[0] + action_vox_delta[0],
        #         self.current_pos_vox[1] + action_vox_delta[1],
        #         self.current_pos_vox[2] - action_vox_delta[2],
        #     )

        #     # If the next position is invalid, bounce back to the current position
        #     is_next_pos_valid_seg = self._is_valid_pos(next_pos_vox)
        #     if not is_next_pos_valid_seg or not self.seg_np[next_pos_vox]:
        #         next_pos_vox = self.current_pos_vox
        #         action_vox_delta = (0, 0, 0)
        #         is_next_pos_valid_seg = True

        # Calculate reward (needs to be before moving to the next pos)
        reward, S = self._calculate_reward(action_vox_delta, next_pos_vox)
        # Check Termination Conditions
        terminated, truncated = False, False
        termination_reason = TReason.NOT_DONE
        if self.current_step_count >= self.config.max_episode_steps:
            truncated, termination_reason = True, TReason.MAX_STEPS
        elif not is_next_pos_valid_seg or not self.allowed_area[next_pos_vox]:
            terminated, termination_reason = True, TReason.OOB
        elif dist(next_pos_vox, self.goal) < self.config.cumulative_path_radius_vox:
            terminated, termination_reason = True, TReason.GOAL_REACHED
        else:
            self.current_distance_traveled += dist(next_pos_vox, self.current_pos_vox)
            self.tracking_path_history.append(next_pos_vox)
            self.current_pos_vox = next_pos_vox
            self.cumulative_path_mask = draw_path_sphere_2(
                self.cumulative_path_mask,
                S,
                self.dilation,
                self.gt_path_vol,
            )
        done = terminated | truncated

        # Final Reward Adjustment
        final_coverage = 0
        if terminated:
            # TODO: Fix the shaping with coverage
            final_coverage = self._get_final_coverage()
            # Tbh that should only be added as a fine-tuning stage or something,
            # it must be messing with the value function
            # reward += self.config.r_final * (
            #     final_coverage
            #     if termination_reason == "reached_goal"
            #     else (final_coverage - 1)
            # )
            if final_coverage < 0.05:
                multiplier = 0.0
            elif final_coverage < 0.2:
                multiplier = 0.2
            elif final_coverage < 0.4:
                multiplier = 0.4
            elif final_coverage < 0.5:
                multiplier = 0.6
            elif final_coverage < 0.7:
                multiplier = 0.8
            else:
                multiplier = 1.0
            if termination_reason == TReason.GOAL_REACHED:
                # else:
                #     reward -= self.config.r_final * 0.5
                reward += self.config.r_final * multiplier
            else:
                reward -= self.config.r_final * (1-multiplier)
            # else:
            #     reward += self.config.r_final * (multiplier-1)
                # reward -= self.config.r_final * (1 - multiplier)
            rew = self.cum_reward + reward
            print(
                "[DEBUG] Episode ended; "
                f"steps={self.current_step_count:04}; "
                f"cumulative_reward={'[bold green]' if rew > 0 else '[bold red]'}{rew:>10.1f}{'[/bold green]' if rew > 0 else '[/bold red]'}; "
                f"reason={'[bold green]' if termination_reason is TReason.GOAL_REACHED else '[bold red]'}{termination_reason}{'[/bold green]' if termination_reason is TReason.GOAL_REACHED else '[/bold red]'}; "
                f"final_coverage={final_coverage:.3f}; {id(self.start_coord)} {id(self.end_coord)} {id(self.goal)} {dist(next_pos_vox, self.goal):.0f}/{dist(self._start, self.goal):.0f}"
            )

        # Get Next State Patches
        next_obs_dict = self._get_state_patches()
        # print(reward, S)

        # Update Internal Done Flag
        self._is_done[:] = done
        self.cum_reward += reward
        _reward = reward.view_as(self._is_done)  # B, T, 1

        output_td = TensorDict(
            {
                "actor": next_obs_dict.unsqueeze(0),
                "reward": _reward,
                "done": torch.as_tensor(done, device=self.device).view_as(_reward),
                "terminated": torch.as_tensor(terminated, device=self.device).view_as(_reward),
                "truncated": torch.as_tensor(truncated, device=self.device).view_as(_reward),
                "info": {
                    "final_coverage": torch.as_tensor(
                        final_coverage, device=self.device, dtype=self.dtype
                    ).view_as(_reward)
                    if done
                    else self.placeholder_zeros.clone(),
                    "final_step_count": torch.as_tensor(
                        self.current_step_count, device=self.device, dtype=self.dtype
                    ).view_as(_reward)
                    if done
                    else torch.as_tensor(0, device=self.device, dtype=self.dtype).view_as(_reward),
                    "final_length": torch.as_tensor(
                        self.current_distance_traveled if done else 0,
                        device=self.device,
                        dtype=self.dtype,
                    ).view_as(_reward),
                    "final_wall_gradient": torch.as_tensor(
                        self.wall_gradient if done else 0,
                        device=self.device,
                        dtype=self.dtype,
                    ).view_as(_reward),
                    "total_reward": self.cum_reward.view_as(_reward).clone()
                    if done
                    else self.placeholder_zeros.clone(),
                    "max_gdt_achieved": torch.as_tensor(
                        self.max_gdt_achieved, dtype=self.dtype, device=self.device
                    ).view_as(_reward),
                },
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return output_td

    def _set_seed(self, seed: Optional[int] = None):
        """Sets the seed for the environment's random number generator(s)."""
        random.seed(seed)
        np.random.seed(seed)
        self.rng = torch.manual_seed(seed)


def get_first(x):
    return x[0]


def make_sb_env(
    config: Config,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_episodes_per_sample: int = 32,
    check_env: bool = False,
):
    """Factory function for the integrated SmallBowelEnv."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_workers = getattr(config, "num_workers", 0)
    dataset_iterator = cycle(
        DataLoader(
            dataset,
            batch_size=1,
            shuffle=config.shuffle_dataset,
            num_workers=num_workers,
            collate_fn=get_first,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    )

    env = SmallBowelEnv(
        config=config,
        dataset_iterator=dataset_iterator,
        num_episodes_per_sample=num_episodes_per_sample,
        device=device,
    )

    if check_env:
        check_env_specs(env)

    return env
