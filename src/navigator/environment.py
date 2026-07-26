"""
Environment implementation for Navigator's small bowel path tracking,
integrated with TorchRL. Simplified version without try-except blocks.
"""

from rich import print
from itertools import cycle, product
from math import dist, isfinite
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
from torch.utils.data import DataLoader

from .config import Config
from .rewards import (
    coverage_potential_reward,
    gdt_progress_reward,
    is_path_success,
    terminal_path_reward,
)
from .utils import (
    BinaryDilation3D,
    ClipTransform,
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
        num_steps_per_sample: Optional[int] = None,
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
        self.num_steps_per_sample = (
            num_steps_per_sample
            if num_steps_per_sample is not None
            else config.num_steps_per_sample
        )
        self.steps_on_current_subject = 0
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
        radius = config.cumulative_path_radius_vox
        dilation_offsets = [
            offset
            for offset in product(range(-radius, radius + 1), repeat=3)
            if sum(abs(component) for component in offset) <= radius
        ]
        self.path_dilation_offsets = torch.as_tensor(
            dilation_offsets, dtype=torch.long, device=self.device
        )
        self.maxarea_dilation = torch.compile(
            torch.nn.Sequential(*[BinaryDilation3D()] * config.allowed_area_radius_vox)
        )
        self.maxarea_dilation.to(self.device)

        # Placeholder tensor
        self.zeros_patch = torch.zeros(self.config.patch_size_vox, device=self.device)
        self.placeholder_zeros = torch.zeros_like(self._is_done, dtype=self.dtype)

    def _set_specs(self):
        self.observation_spec = Composite(
            actor=UnboundedContinuous(
                shape=torch.Size(
                    [
                        *self.batch_size,
                        self.config.observation_channels,
                        *self.config.patch_size_vox,
                    ]
                ),
                dtype=self.dtype,
            ),
            context=UnboundedContinuous(
                shape=torch.Size([*self.batch_size, self.config.context_features]),
                dtype=self.dtype,
            ),
            info=Composite(
                final_coverage=BoundedContinuous(
                    low=0, high=1, shape=torch.Size([*self.batch_size, 1]), dtype=self.dtype
                ),
                final_success=BoundedContinuous(
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
        self.update_data(
            image=subject_data["image"],
            seg=subject_data["seg"],
            wall_map=subject_data["wall_map"],
            gdt_start=subject_data["gdt_start"],
            gdt_end=subject_data["gdt_end"],
            start_coord=subject_data["start_coord"],
            end_coord=subject_data["end_coord"],
            gt_path=subject_data.get("gt_path"),
            spacing=subject_data.get("spacing"),
            image_affine=subject_data.get("image_affine"),
            local_peaks=subject_data.get("local_peaks"),
        )

        # Check if critical data was loaded successfully by update_data
        if self.image is None or self.seg is None or self.wall_map is None:
            raise RuntimeError(
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
        print(f"Changed subject to {self._current_subject_data['id']}")
        # Store tensors directly
        self.image = self.ct_transform(torch.from_numpy(image).to(self.device)).to(self.dtype)
        # save_nifti(np.transpose(image, (2,1,0)), f"image_{self._current_subject_data['id']}.nii.gz", affine=image_affine, spacing=spacing[::-1])
        # save_nifti(np.transpose(self.image.numpy(force=True), (2,1,0)), f"image_transformed_{self._current_subject_data['id']}.nii.gz", affine=image_affine, spacing=spacing[::-1])
        self.seg = torch.from_numpy(seg).to(device=self.device, dtype=torch.uint8)
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
        if local_peaks is None or len(local_peaks) == 0:
            self.local_peaks = np.asarray([start_coord, end_coord], dtype=int)
        else:
            self.local_peaks = np.asarray(local_peaks, dtype=int)
        # self.image[tuple(local_peaks.T)] = 0.5
        self.reward_map = np.zeros_like(self.gdt_start, dtype=np.uint8)
        self.cumulative_path_mask_pen = np.zeros_like(self.reward_map)
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
            if valid_gt_path.shape[0] == 0:
                raise ValueError("No valid GT path voxels found within image bounds.")
            self.gt_path_vol[valid_gt_path[:, 0], valid_gt_path[:, 1], valid_gt_path[:, 2]] = 1.0

    def _get_state_patches(self) -> Dict[str, torch.Tensor]:
        """Get state patches centered at current position. Assumes tensors are valid."""
        img_patch = get_patch(self.image, self.current_pos_vox, self.config.patch_size_vox)
        wall_patch = get_patch(self.wall_map, self.current_pos_vox, self.config.patch_size_vox)
        _ = len(self.tracking_path_history)
        img_patch_1 = get_patch(
            self.image, self.tracking_path_history[-2 % _], self.config.patch_size_vox
        )
        cum_path_patch = get_patch(
            self.cumulative_path_mask, self.current_pos_vox, self.config.patch_size_vox
        )
        segmentation_patch = get_patch(
            self.seg, self.current_pos_vox, self.config.patch_size_vox
        ).to(self.dtype)
        # gt_path_patch = get_patch(
        #     self.gt_path_vol, self.current_pos_vox, self.config.patch_size_vox
        # )
        # save_nifti(np.transpose(img_patch.numpy(force=True), (2,1,0)), "img_patch.nii.gz", affine=self.image_affine, spacing=self.spacing[::-1])
        # save_nifti(np.transpose(wall_patch.numpy(force=True), (2,1,0)), "wall_patch.nii.gz", affine=self.image_affine, spacing=self.spacing[::-1])
        # save_nifti(np.transpose(cum_path_patch.numpy(force=True), (2,1,0)), "cum_path_patch.nii.gz", affine=self.image_affine, spacing=self.spacing[::-1])
        # exit()
        time_fraction = min(self.current_step_count / self.config.max_episode_steps, 1.0)
        goal_delta = torch.as_tensor(
            self.goal, dtype=self.dtype, device=self.device
        ) - torch.as_tensor(self.current_pos_vox, dtype=self.dtype, device=self.device)
        # Keep the physical direction ratios intact. Per-axis image-size
        # normalization bends the direction whenever the volume is anisotropic.
        goal_direction = goal_delta / goal_delta.abs().max().clamp_min(1)

        initial_goal_distance = float(self.initial_goal_distance)
        if initial_goal_distance <= torch.finfo(self.dtype).eps:
            progress_fraction = 0.0
        else:
            # Report progress in the same geodesic potential used by the
            # objective. Euclidean endpoint progress advertises the forbidden
            # chord across a curved bowel loop.
            progress_fraction = 1.0 - float(self.current_goal_distance) / initial_goal_distance
            progress_fraction = min(max(progress_fraction, 0.0), 1.0)
        actor_state = torch.stack(
            [
                img_patch_1,
                img_patch,
                segmentation_patch,
                wall_patch,
                cum_path_patch,
            ],
            dim=0,
        )
        context = torch.cat(
            [
                torch.as_tensor(
                    [time_fraction, progress_fraction],
                    dtype=self.dtype,
                    device=self.device,
                ),
                goal_direction,
            ]
        )
        return {"actor": actor_state, "context": context}

    def _is_valid_pos(self, pos_vox: Coords) -> bool:
        """Check if a position is within the volume bounds."""
        s = self.image.shape
        return (0 <= pos_vox[0] < s[0]) and (0 <= pos_vox[1] < s[1]) and (0 <= pos_vox[2] < s[2])

    def _is_allowed_displacement(self, displacement: Coords) -> bool:
        """Return whether the entire proposed segment is traversable."""
        if not any(displacement):
            return False
        next_pos_vox = tuple(
            current + delta for current, delta in zip(self.current_pos_vox, displacement)
        )
        if not self._is_valid_pos(next_pos_vox):
            return False
        segment = line_nd(self.current_pos_vox, next_pos_vox, endpoint=True)
        return bool(np.asarray(self.allowed_area[segment]).all())

    def _project_action_to_allowed_displacement(self, action_normalized: torch.Tensor) -> Coords:
        """Project a requested direction onto a traversable local segment.

        This projection uses neither reward nor endpoint information. It first
        shortens the requested ray, then selects the best-aligned valid
        one-voxel direction. Thus the continuous policy always controls
        direction without being dominated by invalid endpoints in a thin mask.
        """
        desired = (2.0 * action_normalized.detach().float().cpu().numpy() - 1.0).reshape(3)
        desired_norm = float(np.linalg.norm(desired))
        if desired_norm <= np.finfo(np.float32).eps:
            desired = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
            desired_norm = 1.0
        desired_unit = desired / desired_norm
        desired_chebyshev = desired / max(float(np.max(np.abs(desired))), 1e-8)

        seen: set[Coords] = set()
        for step_size in range(self.config.max_step_vox, 0, -1):
            displacement = tuple(np.rint(desired_chebyshev * step_size).astype(np.int64).tolist())
            if displacement in seen:
                continue
            seen.add(displacement)
            if self._is_allowed_displacement(displacement):
                return displacement

        candidates = []
        for displacement in product((-1, 0, 1), repeat=3):
            if not self._is_allowed_displacement(displacement):
                continue
            candidate = np.asarray(displacement, dtype=np.float32)
            alignment = float(np.dot(candidate / np.linalg.norm(candidate), desired_unit))
            candidates.append((alignment, float(np.linalg.norm(candidate)), displacement))
        if candidates:
            return max(candidates, key=lambda candidate: candidate[:2])[2]
        return (0, 0, 0)

    def _get_final_coverage(self) -> float:
        """Calculate coverage. Assumes tensors are valid."""
        union = self.target_voxels + self.path_voxels
        coverage = (2 * self.path_target_intersection / union) if union else 0.0
        return torch.as_tensor(coverage, dtype=self.dtype, device=self.device)

    def _get_target_mask(self) -> torch.Tensor:
        """Return the binary structure that the current episode must trace."""
        return self.seg.bool()

    def _add_path_segment(self, voxels: Coords | Tuple[np.ndarray, ...]) -> None:
        """Add an exactly dilated local segment and update Dice counters.

        Repeated star-kernel dilation is an L1 ball. Generating that ball around
        the handful of segment voxels is equivalent to six full-volume 3D
        convolutions, but touches only a few thousand coordinates.
        """
        points = np.asarray(voxels, dtype=np.int64)
        if points.ndim == 1:
            points = points[None, :]
        elif points.shape[0] == 3:
            points = points.T
        points_tensor = torch.as_tensor(points, dtype=torch.long, device=self.device)
        coordinates = (points_tensor[:, None, :] + self.path_dilation_offsets[None, :, :]).reshape(
            -1, 3
        )
        shape = torch.as_tensor(self.cumulative_path_mask.shape, device=self.device)
        valid = ((coordinates >= 0) & (coordinates < shape)).all(dim=1)
        coordinates = torch.unique(coordinates[valid], dim=0)
        indices = tuple(coordinates[:, axis] for axis in range(3))
        is_new = self.cumulative_path_mask[indices] == 0
        new_coordinates = coordinates[is_new]
        if new_coordinates.numel() == 0:
            return
        new_indices = tuple(new_coordinates[:, axis] for axis in range(3))
        self.cumulative_path_mask[new_indices] = 1
        self.path_voxels += int(new_coordinates.shape[0])
        self.path_target_intersection += int(self.current_target_mask[new_indices].sum().item())

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
        plotter = pv.Plotter(off_screen=True)
        plotter.add_volume(self.seg.bool().numpy(force=True) * 10, cmap=["blue"], opacity="linear")
        if self._current_subject_data.get("colon") is not None:
            plotter.add_volume(
                self._current_subject_data["colon"] * 10,
                cmap=["green"],
                opacity="linear",
            )
        if self._current_subject_data.get("duodenum") is not None:
            plotter.add_volume(
                self._current_subject_data["duodenum"] * 10,
                cmap=["yellow"],
                opacity="linear",
            )
        plotter.add_points(self.local_peaks, opacity="linear")
        lines: pv.PolyData = pv.lines_from_points(self.tracking_path_history)
        plotter.add_mesh(lines, line_width=10, cmap="viridis")
        plotter.add_points(np.array(self.tracking_path_history), color="blue", point_size=10)
        plotter.show_axes()
        plotter.view_xz()
        plotter.export_html(cache_dir / "path.html")

    def _calculate_reward(self, action_vox: Coords, next_pos_vox: Coords) -> Tuple[float, Tuple]:
        """Calculate the reward for the current step."""

        rt = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        # --- 1. Invalid movement penalty ---
        if not any(action_vox):
            rt -= self.config.r_zero_mov
            return rt, ()

        if not self._is_valid_pos(next_pos_vox):
            rt -= self.config.r_val2
            return rt, ()

        # Set of voxels S on the line segment
        S = line_nd(self.current_pos_vox, next_pos_vox, endpoint=True)
        # Checking only the endpoint permits a single action to cut through a
        # wall or jump between adjacent bowel loops.
        if not bool(np.asarray(self.allowed_area[S]).all()):
            rt -= self.config.r_val2
            return rt, ()

        # --- 2. GDT-based reward ---
        # This is a difference of the geodesic-distance state potential. Forward
        # then backward motion sums to zero before the fixed step costs, so the
        # agent cannot manufacture return by oscillating.
        next_gdt_val = self.gdt[next_pos_vox]
        if np.isfinite(next_gdt_val):
            self.max_gdt_achieved = max(self.max_gdt_achieved, next_gdt_val)

        if self.config.use_immediate_gdt_reward:
            next_goal_distance = float(self.goal_distance_map[next_pos_vox])
            if not isfinite(next_goal_distance):
                rt -= self.config.r_val2
                return rt, ()
            delta = float(self.current_goal_distance) - next_goal_distance
            rt += gdt_progress_reward(
                delta,
                max(float(self.initial_goal_distance), torch.finfo(self.dtype).eps),
                self.config.gdt_reward_scale,
            )
            self.current_goal_distance = next_goal_distance

        # A revisit has no new-coverage reward, and every action still pays this
        # cost. No separate overlap penalty is needed; every line segment
        # necessarily contains its starting voxel.
        rt -= self.config.step_penalty

        # 2.5 Peaks-based reward
        # rt += self.reward_map[S].sum() * self.config.r_peaks
        # self.reward_map[S] = 0
        # Reward for coverage (based on intersection within the segmentation on the path): ...
        # rt += self.config.r_val3 * (self.seg[S] * (1-self.cumulative_path_mask[S])).float().mean()

        # --- 3. Wall-based penalty ---
        wall_map = self.wall_map[S].max()
        rt -= self.config.wall_penalty_scale * wall_map

        self.wall_gradient += wall_map

        # --- 4. Out-of-segmentation penalty (only relevant when an allowed-area
        # dilation was explicitly requested).
        rt -= self.config.r_val1 * self.seg[next_pos_vox].logical_not()
        return rt, S

    def _reset(
        self,
        tensordict: Optional[TensorDictBase] = None,
        must_load_new_subject: Optional[bool] = None,
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
            or self.steps_on_current_subject >= self.num_steps_per_sample
            or must_load_new_subject
        ):
            # Reset counter *before* loading, as loading signifies starting fresh
            self.episodes_on_current_subject = 0
            self.steps_on_current_subject = 0
            self._load_next_subject()
        elif must_load_new_subject is False:
            self.episodes_on_current_subject = 0

        # --- Reset internal episode state ---
        self.current_step_count = 0
        self.current_distance_traveled = 0
        self.wall_gradient = 0

        # Determine start position and select appropriate GDT
        rand = self.episodes_on_current_subject % 10  # 40-30-30
        if rand < 4:
            # Start at the beginning
            self.current_pos_vox = self.start_coord
            self.goal = self.end_coord
            self.gdt = self.gdt_start
            self.goal_distance_map = self.gdt_end
        elif rand < 7:
            # Start at a random local peak
            self.current_pos_vox = tuple(random.choice(self.local_peaks))
            # Randomly go in either direction
            if self.episodes_on_current_subject % 2:
                self.goal = self.end_coord
                self.gdt = self.gdt_start
                self.goal_distance_map = self.gdt_end
            else:
                self.goal = self.start_coord
                self.gdt = self.gdt_end
                self.goal_distance_map = self.gdt_start
        else:
            # Start at the end
            self.current_pos_vox = self.end_coord
            self.goal = self.start_coord
            self.gdt = self.gdt_end
            self.goal_distance_map = self.gdt_start

        self._start = self.current_pos_vox

        # Initialize path tracking
        self.cumulative_path_mask.zero_()
        self.cumulative_path_mask_pen[:] = 0
        self.current_target_mask = self._get_target_mask()
        self.target_voxels = int(self.current_target_mask.sum().item())
        if self.target_voxels <= 0:
            raise ValueError("The current episode has an empty path target.")
        self.path_voxels = 0
        self.path_target_intersection = 0
        self._add_path_segment(self.current_pos_vox)
        self.current_coverage = float(self._get_final_coverage())

        # Initialize various tracking variables
        self.tracking_path_history = [self.current_pos_vox]
        self.cum_reward = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.max_gdt_achieved = self.gdt[self.current_pos_vox]
        self.goal_gdt = self.gdt[self.goal]
        self.current_goal_distance = self.goal_distance_map[self.current_pos_vox]
        self.initial_goal_distance = self.current_goal_distance
        self.initial_euclidean_goal_distance = (
            dist(self.current_pos_vox, self.goal) * self.config.voxel_size_mm
        )
        if not isfinite(float(self.goal_gdt)):
            raise ValueError(
                f"Goal {self.goal} is unreachable in the selected GDT for "
                f"{self._current_subject_data['id']}."
            )
        if not isfinite(float(self.current_goal_distance)):
            raise ValueError(
                f"Start {self.current_pos_vox} is unreachable from goal {self.goal} for "
                f"{self._current_subject_data['id']}."
            )
        self.reward_map[tuple(self.local_peaks.T)] = 1
        err_text = f"{self.max_gdt_achieved} at {self.current_pos_vox}, {self.start_coord}, {self.end_coord}, {self.local_peaks}, ID: {self._current_subject_data['id']}, {self.gdt.shape}"
        if self.max_gdt_achieved < 0 or not np.isfinite(self.max_gdt_achieved):
            raise ValueError(f"Expected GDT>=0, got {err_text}")

        # --- Get Initial State Patches ---
        obs_dict = self._get_state_patches()

        # --- Update Internal Done Flag and Package Output ---
        self._is_done.fill_(False)
        reset_td = TensorDict(
            {
                "actor": obs_dict["actor"].unsqueeze(0),  # Add batch dim
                "context": obs_dict["context"].unsqueeze(0),
                "done": self._is_done.clone(),
                "terminated": self._is_done.clone(),
                "truncated": self._is_done.clone(),
                "info": {
                    "final_step_count": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_length": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_wall_gradient": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_coverage": self.placeholder_zeros.clone(),
                    "final_success": self.placeholder_zeros.clone(),
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
        self.steps_on_current_subject += 1
        # Extract Action
        action_normalized = tensordict.get("action").squeeze(0)

        # Interpret the action as a direction and guarantee a traversable local
        # move whenever the current component has a valid neighbour.
        action_vox_delta = self._project_action_to_allowed_displacement(action_normalized)

        # Execute Step Logic
        next_pos_vox = (
            self.current_pos_vox[0] + action_vox_delta[0],
            self.current_pos_vox[1] + action_vox_delta[1],
            self.current_pos_vox[2] + action_vox_delta[2],
        )

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
        is_next_pos_allowed = len(S) > 0

        if is_next_pos_allowed:
            self.current_distance_traveled += dist(next_pos_vox, self.current_pos_vox)
            self.tracking_path_history.append(next_pos_vox)
            self.current_pos_vox = next_pos_vox
            if S:
                self._add_path_segment(S)
                self.cumulative_path_mask_pen[S] = 1
                next_coverage = float(self._get_final_coverage())
                reward += coverage_potential_reward(
                    self.current_coverage,
                    next_coverage,
                    self.config.coverage_reward_scale,
                )
                self.current_coverage = next_coverage

        # Invalid actions leave the agent in place. Treating them as terminal
        # creates a cheap suicide policy whenever accumulated path costs can
        # exceed the fixed failure penalty.
        terminated, truncated = False, False
        termination_reason = TReason.NOT_DONE
        at_goal = is_next_pos_allowed and (
            dist(self.current_pos_vox, self.goal) < self.config.cumulative_path_radius_vox
        )
        coverage_for_decision = self.current_coverage
        solved_path = is_path_success(
            at_goal,
            coverage_for_decision,
            self.config.success_coverage_threshold,
        )
        if solved_path:
            terminated, termination_reason = True, TReason.GOAL_REACHED
        elif self.current_step_count >= self.config.max_episode_steps:
            # The horizon is part of this finite task: failing to reach the goal is terminal,
            # rather than an external time-limit truncation that should be bootstrapped.
            terminated, termination_reason = True, TReason.MAX_STEPS
        done = terminated | truncated

        # Final Reward Adjustment
        final_coverage = 0
        if done:
            final_coverage = coverage_for_decision
            reward += terminal_path_reward(
                float(final_coverage),
                at_goal,
                self.config.success_coverage_threshold,
                self.config.r_final,
                self.config.coverage_reward_scale
                + (
                    self.config.gdt_reward_scale
                    if self.config.use_immediate_gdt_reward
                    else 0.0
                )
                + self.config.r_val2,
            )

        # Get Next State Patches
        next_obs_dict = self._get_state_patches()
        # print(reward, S)

        # Update Internal Done Flag
        self._is_done[:] = done
        self.cum_reward += reward
        _reward = reward.view_as(self._is_done)  # B, T, 1

        if done and self.config.log_episode_ends:
            rew = self.cum_reward.cpu().item()
            print(
                "[DEBUG] Episode ended; "
                f"steps={self.current_step_count:04}; "
                f"cumulative_reward={'[bold green]' if rew > 0 else '[bold red]'}{rew:>10.1f}{'[/bold green]' if rew > 0 else '[/bold red]'}; "
                f"reason={'[bold green]' if termination_reason is TReason.GOAL_REACHED else '[bold red]'}{termination_reason}{'[/bold green]' if termination_reason is TReason.GOAL_REACHED else '[/bold red]'}; "
                f"final_coverage={final_coverage:.3f}; {id(self.start_coord)} {id(self.end_coord)} {id(self.goal)} {dist(next_pos_vox, self.goal):.0f}/{dist(self._start, self.goal):.0f}"
            )

        output_td = TensorDict(
            {
                "actor": next_obs_dict["actor"].unsqueeze(0),
                "context": next_obs_dict["context"].unsqueeze(0),
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
                    "final_success": torch.as_tensor(
                        solved_path, device=self.device, dtype=self.dtype
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

    def sample_perfect_action(self) -> torch.Tensor:
        """
        Finds the next best displacement based on the ground truth path.
        Returns the action delta in a suitable format (normalized [0, 1] range).
        """
        if self.gt_path_voxels is None or len(self.gt_path_voxels) == 0:
            raise ValueError("A non-empty ground-truth path is required.")

        current_pos_np = np.array(self.current_pos_vox)

        # Find the closest point in the ground truth path to the current position
        distances = np.linalg.norm(self.gt_path_voxels - current_pos_np, axis=1)
        closest_idx = np.argmin(distances)

        # Determine the target point:
        # If we are at or past the end of the path, the target is the last point.
        # Otherwise, the target is the next point in the path.
        if closest_idx >= len(self.gt_path_voxels) - 1:
            target_point = self.gt_path_voxels[-1]
        else:
            target_point = self.gt_path_voxels[closest_idx + 1]

        # Calculate the displacement vector
        displacement_vox = target_point - current_pos_np

        # Normalize the displacement to the range [-1, 1] based on max_step_vox
        # Ensure max_step_vox is not zero to avoid division by zero
        max_step_vox = self.config.max_step_vox
        if max_step_vox == 0:
            normalized_displacement = np.zeros_like(displacement_vox, dtype=np.float32)
        else:
            normalized_displacement = displacement_vox / max_step_vox

        # Convert from [-1, 1] range to [0, 1] range for the action spec
        # action_normalized = (normalized_displacement + 1) / 2
        # The action_spec is BoundedContinuous(low=0, high=1), so the action should be in [0, 1].
        # The _step method maps action_normalized from [0, 1] to [-max_step_vox, max_step_vox]
        # using (2 * action_normalized - 1) * max_step_vox.
        # So, to get a desired displacement `d`, we need:
        # d = (2 * action_normalized - 1) * max_step_vox
        # d / max_step_vox = 2 * action_normalized - 1
        # (d / max_step_vox) + 1 = 2 * action_normalized
        # action_normalized = ((d / max_step_vox) + 1) / 2
        action_normalized = (normalized_displacement + 1) / 2.0

        # Clamp values to ensure they are strictly within [0, 1] due to potential floating point inaccuracies
        action_tensor = (
            torch.from_numpy(action_normalized).to(self.device, dtype=self.dtype).clamp(0.0, 1.0)
        )
        return action_tensor


def get_first(x):
    return x[0]


def make_sb_env(
    config: Config,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_episodes_per_sample: int = 32,
    num_steps_per_sample: Optional[int] = None,
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
        num_steps_per_sample=num_steps_per_sample,
        device=device,
    )

    if check_env:
        check_env_specs(env)

    return env


class MRIPathEnv(SmallBowelEnv):
    """
    TorchRL-compatible environment for RL-based MRI path tracking.
    Subclasses SmallBowelEnv to reuse common functionalities.
    """

    def __init__(
        self,
        config: Config,
        dataset_iterator: Iterator,
        num_episodes_per_sample: int = 32,
        device: Optional[torch.device] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super().__init__(
            config=config,
            dataset_iterator=dataset_iterator,
            num_episodes_per_sample=num_episodes_per_sample,
            device=device,
            batch_size=batch_size,
        )
        self._current_path_idx = 0  # To iterate through multiple paths in a subject

    def _load_next_subject(self) -> bool:
        subject_data: dict = next(self.dataset_iterator)
        self._current_subject_data = subject_data
        self._current_path_idx = 0  # Reset path index for new subject

        # MRIPathDataset provides 'mri', 'small_bowel_seg', 'paths', 'start_coord' (list), 'end_coord' (list)
        # gdt_start, gdt_end, local_peaks are zeros from the dataset.
        self.update_data(
            image=subject_data["image"],  # Use mri as the main image
            seg=subject_data["seg"],  # Use small_bowel_seg as the main segmentation
            wall_map=subject_data["wall_map"],
            gdt_start=subject_data["gdt_start"],  # Zeros
            gdt_end=subject_data["gdt_end"],  # Zeros
            start_coord=subject_data["start_coord"],  # List of start coords
            end_coord=subject_data["end_coord"],  # List of end coords
            gt_path=None,  # No single gt_path for the whole subject
            spacing=subject_data.get("spacing"),
            image_affine=subject_data.get("image_affine"),
            local_peaks=subject_data.get("local_peaks"),  # Zeros
            paths=subject_data.get("paths"),  # All loaded paths
        )
        if self.image is None or self.seg is None or self.wall_map is None:
            raise RuntimeError(
                f"Critical data is None after loading subject {subject_data.get('id', 'N/A')}."
            )
        return True

    def update_data(
        self,
        image: np.ndarray,  # This is now MRI
        seg: np.ndarray,  # This is now small_bowel_seg
        wall_map: np.ndarray,
        gdt_start: np.ndarray,
        gdt_end: np.ndarray,
        start_coord: List[Coords],  # Now a list of start coords for each path
        end_coord: List[Coords],  # Now a list of end coords for each path
        gt_path: Optional[np.ndarray] = None,
        spacing: Optional[Spacing] = None,
        image_affine: Optional[np.ndarray] = None,
        local_peaks: Optional[np.ndarray] = None,
        paths: Optional[List[np.ndarray]] = None,  # New: list of all paths
    ):
        # Call parent's update_data for common processing
        super().update_data(
            image=image,
            seg=seg,
            wall_map=wall_map,
            gdt_start=gdt_start,  # These are zeros from MRIPathDataset
            gdt_end=gdt_end,  # These are zeros from MRIPathDataset
            start_coord=start_coord[0]
            if start_coord
            else (0, 0, 0),  # Use first path's start for initial setup
            end_coord=end_coord[0]
            if end_coord
            else (0, 0, 0),  # Use first path's end for initial setup
            gt_path=gt_path,
            spacing=spacing,
            image_affine=image_affine,
            local_peaks=local_peaks,  # These are zeros from MRIPathDataset
        )
        # Store all paths and their start/end points
        self.all_paths = paths
        self.all_start_coords = start_coord
        self.all_end_coords = end_coord

        # Override gdt and local_peaks to be zeros as per requirement
        self.gdt_start = np.zeros_like(self.image.numpy(force=True), dtype=np.float32)
        self.gdt_end = np.zeros_like(self.image.numpy(force=True), dtype=np.float32)
        self.local_peaks = np.zeros((0, 3), dtype=int)  # Kx3, K can be 0
        self.reward_map = np.zeros_like(self.gdt_start, dtype=np.uint8)  # Reset reward map

        # No duodenum or colon in this dataset
        self.duodenum = None
        self.colon = None

    def _reset(
        self,
        tensordict: Optional[TensorDictBase] = None,
        must_load_new_subject: Optional[bool] = None,
    ) -> TensorDictBase:
        if self._is_done.all():
            self.episodes_on_current_subject += 1
        if (
            self.episodes_on_current_subject >= self.num_episodes_per_sample
            or self.steps_on_current_subject >= self.num_steps_per_sample
            or must_load_new_subject
        ):
            self.episodes_on_current_subject = 0
            self.steps_on_current_subject = 0
            self._load_next_subject()
            # After loading new subject, _current_path_idx is reset to 0
            # and self.all_paths, self.all_start_coords, self.all_end_coords are updated.
        elif must_load_new_subject is False:
            self.episodes_on_current_subject = 0

        # Select the current path for this episode
        if not self.all_paths:
            raise ValueError("No paths available for the current subject.")

        # Cycle through paths for different episodes on the same subject
        self._current_path_idx = self.episodes_on_current_subject % len(self.all_paths)

        current_path = self.all_paths[self._current_path_idx]
        current_start_coord = self.all_start_coords[self._current_path_idx]
        current_end_coord = self.all_end_coords[self._current_path_idx]

        if current_path is None or current_start_coord is None or current_end_coord is None:
            # Handle cases where a path might be empty or failed to load
            print(
                f"Warning: Skipping path {self._current_path_idx} for subject {self._current_subject_data['id']} due to invalid data."
            )
            # Try to load next subject or path if current one is invalid
            # For simplicity, let's just reset again, which will eventually load a new subject
            return self._reset(tensordict, must_load_new_subject=True)

        self.current_step_count = 0
        self.current_distance_traveled = 0
        self.wall_gradient = 0

        self.current_pos_vox = tuple(current_start_coord)
        self.goal = tuple(current_end_coord)

        # As per requirement, gdt is always zeros for this dataset
        self.gdt = np.zeros_like(self.image.numpy(force=True), dtype=np.float32)
        self.max_gdt_achieved = 0.0  # Always 0 since GDT is zero
        self.goal_gdt = 0.0  # Always 0 since GDT is zero
        self.current_goal_distance = 0.0
        self.initial_goal_distance = 0.0
        self.initial_euclidean_goal_distance = (
            dist(self.current_pos_vox, self.goal) * self.config.voxel_size_mm
        )

        self._start = self.current_pos_vox

        self.cumulative_path_mask.zero_()
        # Use the current path as gt_path_vol for drawing
        self.gt_path_vol.zero_()
        if current_path is not None and current_path.shape[0] > 0:
            valid_indices = (
                (current_path[:, 0] >= 0)
                & (current_path[:, 0] < self.image.shape[0])
                & (current_path[:, 1] >= 0)
                & (current_path[:, 1] < self.image.shape[1])
                & (current_path[:, 2] >= 0)
                & (current_path[:, 2] < self.image.shape[2])
            )
            valid_gt_path = current_path[valid_indices]
            if valid_gt_path.shape[0] > 0:
                self.gt_path_vol[valid_gt_path[:, 0], valid_gt_path[:, 1], valid_gt_path[:, 2]] = (
                    1.0
                )

        self.current_target_mask = self._get_target_mask()
        self.target_voxels = int(self.current_target_mask.sum().item())
        if self.target_voxels <= 0:
            raise ValueError("The current MRI path has an empty segmentation target.")
        self.path_voxels = 0
        self.path_target_intersection = 0
        self._add_path_segment(tuple(self.current_pos_vox))
        self.current_coverage = float(self._get_final_coverage())

        self.tracking_path_history = [self.current_pos_vox]  # Full path tracking
        self.cum_reward = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # local_peaks is always zeros, so reward_map based on it is also zeros
        self.reward_map = np.zeros_like(self.gdt, dtype=np.uint8)
        actor_obs_data = self._get_state_patches()
        self._is_done.fill_(False)
        reset_td = TensorDict(
            {
                "actor": actor_obs_data["actor"].unsqueeze(0),
                "context": actor_obs_data["context"].unsqueeze(0),
                "done": self._is_done.clone(),
                "terminated": self._is_done.clone(),
                "truncated": self._is_done.clone(),
                "info": {
                    "final_step_count": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_length": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_wall_gradient": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_coverage": self.placeholder_zeros.clone(),
                    "final_success": self.placeholder_zeros.clone(),
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

    def _calculate_reward(self, action_vox: Coords, next_pos_vox: Coords) -> Tuple[float, Tuple]:
        rt = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if not any(action_vox):
            rt -= self.config.r_zero_mov
            return rt, ()
        if not self._is_valid_pos(next_pos_vox):
            rt -= self.config.r_val2
            return rt, ()

        S = line_nd(self.current_pos_vox, next_pos_vox, endpoint=True)
        if not bool(np.asarray(self.allowed_area[S]).all()):
            rt -= self.config.r_val2
            return rt, ()

        # GDT-based reward removed as gdt is always zero
        # next_gdt_val = self.gdt[next_pos_vox]
        # if next_gdt_val > self.max_gdt_achieved:
        #     delta = next_gdt_val - self.max_gdt_achieved
        #     rt += self.config.r_val2 * (
        #         (1 + delta / self.config.gdt_max_increase_theta)
        #         if delta < self.config.gdt_max_increase_theta
        #         else -1
        #     )
        #     if delta < self.config.gdt_max_increase_theta:
        #         dist_before = abs(self.goal_gdt - self.max_gdt_achieved)
        #         dist_after = abs(self.goal_gdt - next_gdt_val)
        #         phi_before = -dist_before
        #         phi_after = -dist_after
        #         shaping_bonus = self.config.gamma * phi_after - phi_before
        #         rt += 1 + shaping_bonus / self.config.gdt_max_increase_theta
        #     else:
        #         rt -= 1
        #     self.max_gdt_achieved = next_gdt_val

        rt -= self.config.step_penalty

        wall_val = self.wall_map[S].max()
        rt -= self.config.wall_penalty_scale * wall_val
        self.wall_gradient += wall_val

        rt -= self.config.r_val1 * self.seg[next_pos_vox].logical_not()
        return rt, S

    def _get_target_mask(self) -> torch.Tensor:
        return torch.from_numpy(self._current_subject_data["seg"] == self._current_path_idx + 1).to(
            self.device
        )

    def _get_final_coverage(self):
        return super()._get_final_coverage()

    def save_path(self, save_dir: Optional[Path] = None):
        cache_dir = save_dir or (Path("results") / self._current_subject_data["id"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save cumulative path mask
        nif = nib.nifti1.Nifti1Image(
            np.transpose(self.cumulative_path_mask.numpy(force=True), (2, 1, 0)),
            affine=self.image_affine,
        )
        nib.save(nif, cache_dir / f"cumulative_path_mask_path_{self._current_path_idx}.nii.gz")

        # Save tracking history
        tracking_history_path = cache_dir / f"tracking_history_path_{self._current_path_idx}.npy"
        np.savetxt(tracking_history_path, np.fliplr(self.tracking_path_history), fmt="%d")

        # Visualization
        plotter = pv.Plotter(off_screen=True)

        # Add small bowel segmentation
        plotter.add_volume(self.seg.bool().numpy(force=True) * 10, cmap=["blue"], opacity="linear")

        # Add the traversed path
        lines: pv.PolyData = pv.lines_from_points(
            np.array(self.tracking_path_history)
        )  # Ensure numpy array
        plotter.add_mesh(lines, line_width=10, cmap=["yellow"])
        plotter.add_points(np.array(self.tracking_path_history), color="blue", point_size=10)

        # Add the ground truth path for the current episode
        if self.gt_path_vol is not None and torch.sum(self.gt_path_vol) > 0:
            gt_coords = torch.argwhere(self.gt_path_vol > 0).cpu().numpy()
            plotter.add_points(gt_coords, color="red", point_size=5, render_points_as_spheres=True)
            plotter.add_text(
                f"GT Path {self._current_path_idx}", position="upper_left", color="red"
            )

        # Add start and end points for the current path
        plotter.add_points(
            np.array([self._start]), color="green", point_size=15, render_points_as_spheres=True
        )
        plotter.add_points(
            np.array([self.goal]), color="purple", point_size=15, render_points_as_spheres=True
        )
        plotter.add_text("Start (Green)", position="lower_left", color="green")
        plotter.add_text("Goal (Purple)", position="lower_right", color="purple")

        plotter.show_axes()
        plotter.view_xz()
        plotter.export_html(cache_dir / f"path_visualization_path_{self._current_path_idx}.html")
        plotter.close()


def make_mri_path_env(
    config: Config,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_episodes_per_sample: int = 32,
    check_env: bool = False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = getattr(config, "num_workers", 0)
    num_workers = 0
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
    env = MRIPathEnv(
        config=config,
        dataset_iterator=dataset_iterator,
        num_episodes_per_sample=num_episodes_per_sample,
        device=device,
        batch_size=torch.Size([1]),
    )
    if check_env:
        check_env_specs(env)
    return env
