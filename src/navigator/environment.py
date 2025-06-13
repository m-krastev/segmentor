"""
Environment implementation for Navigator's small bowel path tracking,
integrated with TorchRL. Simplified version without try-except blocks.
"""

from rich import print
from itertools import cycle
from math import dist, isfinite, atan2, pi  # Removed cos, sin
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Tuple
from collections import deque  # Added deque

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
POSITION_HISTORY_LENGTH = 128  # For LSTM input


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
    tracking_path_history: List[Coords]  # For full episode path, unbounded
    position_history: deque  # For LSTM observation, fixed length
    goal: Coords = (0, 0, 0)
    # current_agent_heading_vector removed

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
        self.num_episodes_per_sample = num_episodes_per_sample
        self.episodes_on_current_subject = num_episodes_per_sample

        self.dtype = (
            torch.bfloat16
            if config.use_bfloat16 and self.device.type == "cuda"
            else torch.float32
        )

        # Initialize position history deque
        self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)

        # --- Define Specs ---
        self.observation_spec = Composite(
            actor=UnboundedContinuous(
                shape=torch.Size([*batch_size, 4, *config.patch_size_vox]),
                dtype=self.dtype,
            ),
            aux=UnboundedContinuous(  # New spec for position history
                shape=torch.Size([
                    *batch_size,
                    POSITION_HISTORY_LENGTH,
                    3,
                ]),  # B x T x 3 (Z,Y,X)
                dtype=self.dtype,
            ),
            mask=UnboundedContinuous(
                shape=torch.Size([
                    *batch_size,
                    POSITION_HISTORY_LENGTH,
                ]),
                dtype=torch.bool,
            ),
            info=Composite(
                final_coverage=BoundedContinuous(
                    low=0, high=1, shape=torch.Size([*batch_size, 1]), dtype=self.dtype
                ),
                final_step_count=UnboundedContinuous(
                    shape=torch.Size([*batch_size, 1]), dtype=self.dtype
                ),
                final_length=UnboundedContinuous(
                    shape=torch.Size([*batch_size, 1]), dtype=self.dtype
                ),
                final_wall_gradient=UnboundedContinuous(
                    shape=torch.Size([*batch_size, 1]), dtype=self.dtype
                ),
                total_reward=UnboundedContinuous(
                    shape=torch.Size([*batch_size, 1]), dtype=self.dtype
                ),
                max_gdt_achieved=UnboundedContinuous(
                    shape=torch.Size([*batch_size, 1]), dtype=self.dtype
                ),
            ),
            shape=batch_size,
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedContinuous(
            low=0, high=1, shape=torch.Size([*batch_size, 3]), dtype=self.dtype
        )
        self.reward_spec = UnboundedContinuous(
            shape=torch.Size([*batch_size, 1]), dtype=self.dtype
        )
        self.done_spec = Composite(
            done=Binary(shape=torch.Size([*batch_size, 1]), dtype=torch.bool),
            terminated=Binary(shape=torch.Size([*batch_size, 1]), dtype=torch.bool),
            truncated=Binary(shape=torch.Size([*batch_size, 1]), dtype=torch.bool),
            shape=batch_size,
        )

        self.max_gdt_achieved = 0.0
        self.cum_reward = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._is_done = torch.zeros(
            *self.batch_size, 1, dtype=torch.bool, device=self.device
        )

        self.ct_transform = ClipTransform(30 - 160, 30 + 160)
        self.ct_transform.compile()
        self.wall_transform = ClipTransform(0.0, 0.1)
        self.wall_transform.compile()

        self.zeros_patch = torch.zeros(self.config.patch_size_vox, device=self.device)
        self.placeholder_zeros = torch.zeros_like(self._is_done, dtype=self.dtype)
        self.dilation = torch.nn.Sequential(*[
            BinaryDilation3D()
            for _ in range(config.cumulative_path_radius_vox // 2 + 1)
        ]).to(self.device)
        self.maxarea_dilation = torch.nn.Sequential(*[
            BinaryDilation3D() for _ in range(10)
        ]).to(self.device)
        self.dilation.compile()

    def _load_next_subject(self) -> bool:
        subject_data: dict = next(self.dataset_iterator)
        self._current_subject_data = subject_data
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
        assert (
            self.image is not None
            and self.seg is not None
            and self.wall_map is not None
        ), (
            f"Critical data is None after loading subject {subject_data.get('id', 'N/A')}."
        )
        return True

    def update_data(
        self,
        image: np.ndarray,
        seg: np.ndarray,
        wall_map: np.ndarray,
        gdt_start: np.ndarray,
        gdt_end: np.ndarray,
        start_coord: Coords,
        end_coord: Coords,
        gt_path: Optional[np.ndarray] = None,
        spacing: Optional[Spacing] = None,
        image_affine: Optional[np.ndarray] = None,
        local_peaks: Optional[np.ndarray] = None,
    ):
        self.image = self.ct_transform(torch.from_numpy(image).to(self.device)).to(
            self.dtype
        )
        self.seg = torch.from_numpy(seg).to(device=self.device, dtype=torch.uint8)
        self.seg = self.dilation(self.seg.unsqueeze(0).unsqueeze(0)).squeeze()
        self.seg_volume = torch.sum(self.seg).item()
        self.wall_map = self.wall_transform(
            torch.from_numpy(wall_map).to(device=self.device, dtype=self.dtype)
        )
        self.gdt_start = gdt_start
        self.gdt_end = gdt_end
        self.cumulative_path_mask = torch.zeros_like(self.seg)
        self.local_peaks = local_peaks[30:-30]
        self.reward_map = np.zeros_like(self.gdt_start, dtype=np.uint8)
        self.allowed_area = (
            self.maxarea_dilation(self.seg.unsqueeze(0).unsqueeze(0))
            .squeeze()
            .numpy(force=True)
        )
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.image_affine = image_affine if image_affine is not None else np.eye(4)
        self.start_coord = tuple(start_coord)
        self.end_coord = tuple(end_coord)
        self.gt_path_voxels = gt_path
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
            assert valid_gt_path.shape[0] > 0, "No valid GT path voxels found."
            self.gt_path_vol[
                valid_gt_path[:, 0], valid_gt_path[:, 1], valid_gt_path[:, 2]
            ] = 1.0

    def _get_actor_observation(self) -> Dict[str, torch.Tensor]:
        # Patches
        hist_len = len(self.tracking_path_history)
        img_patch = get_patch(
            self.image,
            self.tracking_path_history[-1 % hist_len],
            self.config.patch_size_vox,
        )
        img_patch_1 = get_patch(
            self.image,
            self.tracking_path_history[-2 % hist_len],
            self.config.patch_size_vox,
        )
        img_patch_2 = get_patch(
            self.image,
            self.tracking_path_history[-3 % hist_len],
            self.config.patch_size_vox,
        )
        cum_path_patch = get_patch(
            self.cumulative_path_mask, self.current_pos_vox, self.config.patch_size_vox
        )
        patches_tensor = torch.stack(
            [img_patch_2, img_patch_1, img_patch, cum_path_patch], axis=0
        )

        # Position Sequence
        # Coords are (z,y,x)
        position_sequence_tensor = torch.tensor(
            self.position_history, dtype=self.dtype, device=self.device
        )
        # Left-pad
        mask = torch.tensor(
            ([True] * (POSITION_HISTORY_LENGTH - hist_len) + [False] * hist_len) if hist_len < POSITION_HISTORY_LENGTH else [False]*POSITION_HISTORY_LENGTH, device=self.device
        )
        # print(mask.shape)

        return {
            "patches": patches_tensor,
            "position_sequence": position_sequence_tensor,
            "mask": mask
        }

    def _is_valid_pos(self, pos_vox: Coords) -> bool:
        s = self.image.shape
        return (
            (0 <= pos_vox[0] < s[0])
            and (0 <= pos_vox[1] < s[1])
            and (0 <= pos_vox[2] < s[2])
        )

    def _get_final_coverage(self) -> float:
        mask = self.cumulative_path_mask
        intersection = torch.sum(mask * self.seg)
        union = self.seg_volume + mask.sum()
        return (
            (2 * intersection / union).item() if union != 0 else 0.0
        )  # .item() for float

    def get_tracking_history(self) -> np.ndarray:
        return np.array(self.tracking_path_history)

    def get_tracking_mask(self) -> torch.Tensor:
        return self.cumulative_path_mask.clone()

    def save_path(self, save_dir: Optional[Path] = None):
        cache_dir = save_dir or (Path("results") / self._current_subject_data["id"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        nif = nib.nifti1.Nifti1Image(
            np.transpose(self.cumulative_path_mask.numpy(force=True), (2, 1, 0)),
            affine=self.image_affine,
        )
        nib.save(nif, cache_dir / "cumulative_path_mask.nii.gz")
        tracking_history_path = cache_dir / "tracking_history.npy"
        np.savetxt(
            tracking_history_path, np.fliplr(self.tracking_path_history), fmt="%d"
        )
        plotter = pv.Plotter()
        plotter.add_volume(
            self.seg.numpy(force=True) * 10, cmap="viridis", opacity="linear"
        )
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
        lines: pv.PolyData = pv.lines_from_points(
            np.array(self.tracking_path_history)
        )  # Ensure numpy array
        plotter.add_mesh(lines, line_width=10, cmap="viridis")
        plotter.add_points(
            np.array(self.tracking_path_history), color="blue", point_size=10
        )
        plotter.show_axes()
        plotter.export_html(cache_dir / "path.html")

    def _calculate_reward(
        self, action_vox: Coords, next_pos_vox: Coords
    ) -> Tuple[float, Tuple]:
        rt = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if not any(action_vox):
            rt -= self.config.r_zero_mov
            return rt, ()
        if not self._is_valid_pos(next_pos_vox) or not self.allowed_area[next_pos_vox]:
            rt -= self.config.r_zero_mov
            return rt, ()

        S = line_nd(self.current_pos_vox, next_pos_vox, endpoint=True)
        next_gdt_val = self.gdt[next_pos_vox]
        if next_gdt_val > self.max_gdt_achieved:
            delta = next_gdt_val - self.max_gdt_achieved
            rt += self.config.r_val2 * (
                (1 + delta / self.config.gdt_max_increase_theta)
                if delta < self.config.gdt_max_increase_theta
                else -1
            )
            if delta < self.config.gdt_max_increase_theta:
                dist_before = abs(self.goal_gdt - self.max_gdt_achieved)
                dist_after = abs(self.goal_gdt - next_gdt_val)
                phi_before = -dist_before
                phi_after = -dist_after
                shaping_bonus = self.config.gamma * phi_after - phi_before
                rt += 1 + shaping_bonus / self.config.gdt_max_increase_theta
            else:
                rt -= 1
            self.max_gdt_achieved = next_gdt_val

        phi_t = self.current_step_count - 1
        phi_tp1 = self.current_step_count
        rt += self.config.gamma * phi_tp1 - phi_t

        wall_val = self.wall_map[S].max()
        rt -= self.config.r_val2 * wall_val
        self.wall_gradient += wall_val

        coverage = self.cumulative_path_mask[S][3:]
        rt -= self.config.r_val1 * coverage.any()
        rt -= self.config.r_val1 * self.seg[next_pos_vox].logical_not()
        return rt, S

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, must_load_new_subject=False
    ) -> TensorDictBase:
        if self._is_done.all():
            self.episodes_on_current_subject += 1
        if (
            self.episodes_on_current_subject >= self.num_episodes_per_sample
            or must_load_new_subject
        ):
            self.episodes_on_current_subject = 0
            self._load_next_subject()
            self.goal = (0, 0, 0)  # Should be set by start_choice logic

        self.current_step_count = 0
        self.current_distance_traveled = 0
        self.wall_gradient = 0

        rand = random.randint(0, 9)
        if rand < 4:
            self.current_pos_vox = self.start_coord
            self.goal = self.end_coord
            self.gdt = self.gdt_start
        elif rand < 7:
            self.current_pos_vox = tuple(random.choice(self.local_peaks))
            if self.episodes_on_current_subject % 2:
                self.goal = self.end_coord
                self.gdt = self.gdt_start
            else:
                self.goal = self.start_coord
                self.gdt = self.gdt_end
        else:
            self.current_pos_vox = self.end_coord
            self.goal = self.start_coord
            self.gdt = self.gdt_end
        self._start = self.current_pos_vox

        self.cumulative_path_mask.zero_()
        draw_path_sphere_2(
            self.cumulative_path_mask,
            self.current_pos_vox,
            self.dilation,
            self.gt_path_vol,
        )

        # Initialize position_history
        for _ in range(POSITION_HISTORY_LENGTH - 1):
            self.position_history.append((0, 0, 0))  # Pad
        self.position_history.append(self.current_pos_vox)  # Add current position

        self.tracking_path_history = [self.current_pos_vox]  # Full path tracking
        self.cum_reward = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.max_gdt_achieved = self.gdt[self.current_pos_vox]
        self.goal_gdt = self.gdt.max()
        self.reward_map[tuple(self.local_peaks.T)] = 1
        assert self.max_gdt_achieved >= 0 and np.isfinite(self.max_gdt_achieved), (
            "GDT error."
        )

        actor_obs_data = self._get_actor_observation()
        self._is_done.fill_(False)
        reset_td = TensorDict(
            {
                "actor": actor_obs_data["patches"].unsqueeze(0),
                "aux": actor_obs_data["position_sequence"].unsqueeze(0),
                "mask": actor_obs_data["mask"].unsqueeze(0),
                "done": self._is_done.clone(),
                "terminated": self._is_done.clone(),
                "truncated": self._is_done.clone(),
                "info": {
                    "final_step_count": torch.zeros_like(
                        self._is_done, dtype=self.dtype
                    ),
                    "final_length": torch.zeros_like(self._is_done, dtype=self.dtype),
                    "final_wall_gradient": torch.zeros_like(
                        self._is_done, dtype=self.dtype
                    ),
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
        self.current_step_count += 1
        action_normalized = tensordict.get("action").squeeze(0)
        action_mapped = (2 * action_normalized - 1) * self.config.max_step_vox
        action_vox_delta = action_mapped.cpu().round().int().tolist()

        # Agent heading update logic removed as it's not part of observation anymore

        next_pos_vox = (
            self.current_pos_vox[0] + action_vox_delta[0],
            self.current_pos_vox[1] + action_vox_delta[1],
            self.current_pos_vox[2] + action_vox_delta[2],
        )
        is_next_pos_valid_seg = self._is_valid_pos(next_pos_vox)
        reward, S = self._calculate_reward(action_vox_delta, next_pos_vox)

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
            self.tracking_path_history.append(next_pos_vox)  # Full path
            self.current_pos_vox = next_pos_vox
            # Update position_history for LSTM
            self.position_history.append(self.current_pos_vox)

            draw_path_sphere_2(
                self.cumulative_path_mask, S, self.dilation, self.gt_path_vol
            )

        # truncated = terminated  # For now, treat all terminations as truncations for PPO
        done = terminated | truncated

        final_coverage = 0.0
        if done:
            final_coverage = self._get_final_coverage()
            # Coverage-based adjustment
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
                reward += self.config.r_final
                reward += self.config.r_final * multiplier
            elif termination_reason == TReason.OOB:
                reward -= self.config.r_final * (1 - multiplier)

            rew_val = (self.cum_reward + reward).item()  # .item() for float
            print(
                f"[DEBUG] Episode ended; steps={self.current_step_count:04}; "
                f"cumulative_reward={'[bold green]' if rew_val > 0 else '[bold red]'}{rew_val:>10.1f}{'[/bold green]' if rew_val > 0 else '[/bold red]'}; "
                f"reason={'[bold green]' if termination_reason is TReason.GOAL_REACHED else '[bold red]'}{termination_reason.value}{'[/bold green]' if termination_reason is TReason.GOAL_REACHED else '[/bold red]'}; "
                f"final_coverage={final_coverage:.3f}; dist_to_goal={dist(next_pos_vox, self.goal):.0f}/{dist(self._start, self.goal):.0f}"
            )

        next_actor_obs_data = self._get_actor_observation()
        self._is_done[:] = done
        self.cum_reward += reward
        _reward_tensor = reward.view_as(self._is_done)

        output_td = TensorDict(
            {
                "actor": next_actor_obs_data["patches"].unsqueeze(0),
                "aux": next_actor_obs_data["position_sequence"].unsqueeze(0),
                "mask": next_actor_obs_data["mask"].unsqueeze(0),
                "reward": _reward_tensor,
                "done": torch.as_tensor(done, device=self.device).view_as(
                    _reward_tensor
                ),
                "terminated": torch.as_tensor(terminated, device=self.device).view_as(
                    _reward_tensor
                ),
                "truncated": torch.as_tensor(truncated, device=self.device).view_as(
                    _reward_tensor
                ),
                "info": {
                    "final_coverage": torch.as_tensor(
                        final_coverage, device=self.device, dtype=self.dtype
                    ).view_as(_reward_tensor)
                    if done
                    else self.placeholder_zeros.clone(),
                    "final_step_count": torch.as_tensor(
                        self.current_step_count, device=self.device, dtype=self.dtype
                    ).view_as(_reward_tensor)
                    if done
                    else torch.as_tensor(
                        0, device=self.device, dtype=self.dtype
                    ).view_as(_reward_tensor),
                    "final_length": torch.as_tensor(
                        self.current_distance_traveled if done else 0,
                        device=self.device,
                        dtype=self.dtype,
                    ).view_as(_reward_tensor),
                    "final_wall_gradient": torch.as_tensor(
                        self.wall_gradient if done else 0,
                        device=self.device,
                        dtype=self.dtype,
                    ).view_as(_reward_tensor),
                    "total_reward": self.cum_reward.view_as(_reward_tensor)
                    if done
                    else self.placeholder_zeros.clone(),
                    "max_gdt_achieved": torch.as_tensor(
                        self.max_gdt_achieved, dtype=self.dtype, device=self.device
                    ).view_as(_reward_tensor),
                },
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return output_td

    def _set_seed(self, seed: Optional[int] = None):
        random.seed(seed)
        np.random.seed(seed)
        # self.rng = torch.manual_seed(seed) # rng not used, but good practice if it were


def get_first(x):
    return x[0]


def make_sb_env(
    config: Config,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_episodes_per_sample: int = 32,
    check_env: bool = False,
):
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
        batch_size=torch.Size([1]),  # Explicitly set batch size as list
    )
    if check_env:
        check_env_specs(env)
    return env
