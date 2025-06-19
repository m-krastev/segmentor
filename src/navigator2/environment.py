"""
Environment implementation for Navigator's small bowel path tracking,
compatible with the skrl framework.
"""

from math import dist
from pathlib import Path
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gymnasium
import nibabel as nib
import numpy as np
import pyvista as pv
import torch
from skimage.draw import line_nd

from navigator.config import Config
from navigator.utils import (
    ClipTransform,
    compute_gdt,
    draw_path_sphere,
    get_patch,
)



class SkrlSmallBowelEnv:
    """
    skrl-compatible environment for RL-based small bowel path tracking.
    """

    # --- Environment State Variables (similar to original) ---
    goal: Tuple[int, int, int]
    current_pos_vox: Tuple[int, int, int]
    start_coord: Tuple[int, int, int]
    end_coord: Tuple[int, int, int]
    image: torch.Tensor
    seg: torch.Tensor
    wall_map: torch.Tensor
    gt_path_voxels: Optional[np.ndarray]
    gt_path_vol: Optional[torch.Tensor]  # Can be None if GT not available
    cumulative_path_mask: torch.Tensor
    gdt: torch.Tensor
    reward_map: torch.Tensor
    spacing: Tuple[float, float, float]
    image_affine: np.ndarray
    current_step_count: int
    tracking_path_history: List[Tuple[int, int, int]]
    max_gdt_achieved: torch.Tensor
    current_gdt_val: torch.Tensor
    seg_volume: float
    seg_np: np.ndarray

    def __init__(
        self,
        config: Config,
        dataset_iterator: Iterator,  # Pass an iterator over your dataset
        device: Optional[torch.device] = None,
    ):
        """Initialize the small bowel environment for skrl."""
        # --- Call Wrapper Constructor ---
        # We are not wrapping an existing env object, so pass None
        # super().__init__()
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Store Config, Iterator, Device ---
        self.config = config
        self.dataset_iterator = dataset_iterator  # Store the iterator
        self._current_subject_data = None  # Store data for the current subject
        self.num_episodes_per_sample = config.num_episodes_per_sample
        self.episodes_on_current_subject = 0  # Counter for episodes on current subject

        # --- Define skrl/Gymnasium Spaces ---
        # Shape: [C, D, H, W] - assuming channel-first for PyTorch convolutions
        patch_shape = config.patch_size_vox
        actor_obs_shape = (3, *patch_shape)
        critic_obs_shape = (4, *patch_shape)  # Includes GT path patch if available

        # Use gymnasium.spaces.Dict for composite observations
        # self._observation_space = gymnasium.spaces.Dict({
        #     "actor": gymnasium.spaces.Box(
        #         low=-np.inf, high=np.inf, shape=actor_obs_shape, dtype=np.float32
        #     ),
        #     "critic": gymnasium.spaces.Box(
        #         low=-np.inf, high=np.inf, shape=critic_obs_shape, dtype=np.float32
        #     ),
        # })
        self._observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=actor_obs_shape, dtype=np.float32
        )  # For actor only, critic can be handled separately

        # State space can be the same as observation or potentially just the critic view
        self._state_space = self._observation_space  # Or adjust if needed

        # Action space: Box representing delta movements in 3D
        self._action_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(3,),  # ActionDim
            dtype=np.float32,
        )

        # --- skrl Properties ---
        self._num_envs = 1  # This wrapper manages a single environment instance
        self._num_agents = 1

        # --- Initialize Internal State (per-instance) ---
        # These need device placement
        self.max_gdt_achieved = torch.tensor(0.0, device=self.device)
        self.current_gdt_val = torch.tensor(0.0, device=self.device)

        # Helper attributes
        self.transform = ClipTransform(-150, 250).to(self.device)
        # self.transform.compile()
        self.zeros_patch = torch.zeros(self.config.patch_size_vox, device=self.device)

        # --- State variables to be initialized in reset ---
        self.image = None
        self.seg = None
        self.wall_map = None
        self.gdt_start = None
        self.gdt_end = None
        # ... and others like current_pos_vox, goal etc.

        # Flag to indicate if an episode is running (set in reset, cleared in step if done)
        self._episode_active = False

    # --- skrl Properties Implementation ---
    @property
    def observation_space(self) -> gymnasium.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        return self._action_space

    @property
    def state_space(self) -> gymnasium.spaces.Dict:
        # Return the defined state space, potentially same as obs_space
        return self._state_space

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def device(self) -> torch.device:
        return self._device

    def _load_next_subject(self) -> bool:
        """
        Loads the next subject's data from the iterator.
        Raises StopIteration if iterator is exhausted.
        Raises other exceptions if data loading/processing fails.
        """
        subject_data = next(self.dataset_iterator)  # Let StopIteration propagate
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

        # Check if critical data was loaded successfully
        if (
            self.image is None
            or self.seg is None
            or self.wall_map is None
            or self.gdt_start is None
            or self.gdt_end is None
        ):
            raise RuntimeError(
                f"Critical data is None after loading subject {subject_data.get('id', 'N/A')}."
            )

        self.episodes_on_current_subject = 0  # Reset counter for the new subject
        return True

    def update_data(
        self,
        image: np.ndarray,
        seg: np.ndarray,
        wall_map: np.ndarray,
        gdt_start: np.ndarray,
        gdt_end: np.ndarray,
        start_coord: Tuple[int, int, int],
        end_coord: Tuple[int, int, int],
        gt_path: Optional[np.ndarray] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
        image_affine: Optional[np.ndarray] = None,
        local_peaks: Optional[np.ndarray] = None,
    ):
        """
        Updates the environment's internal data stores using data from the dataset.
        Moves data to the correct device.
        """
        # Apply transform (clipping) and move to device
        self.image = self.transform(torch.from_numpy(image).float().to(self.device))

        self.seg = torch.from_numpy(seg).to(device=self.device, dtype=torch.uint8)
        self.seg_volume = torch.sum(self.seg).item()
        self.seg_np = seg  # Keep numpy version if needed for visualization or specific calcs

        self.wall_map = torch.from_numpy(wall_map).float().to(self.device)
        self.gdt_start = torch.from_numpy(gdt_start).float().to(self.device)
        self.gdt_end = torch.from_numpy(gdt_end).float().to(self.device)
        self.local_peaks = (
            local_peaks  # Keep as numpy for now? Or move to torch? Assume numpy for now.
        )

        # Store other metadata
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.image_affine = image_affine if image_affine is not None else np.eye(4)
        self.start_coord = tuple(map(int, start_coord))  # Ensure integer coordinates
        self.end_coord = tuple(map(int, end_coord))

        # Update ground truth path if provided
        self.gt_path_voxels = gt_path
        self._process_gt_path()  # This sets self.gt_path_vol (on device) or None

    def _process_gt_path(self):
        """Process ground truth path data. Creates self.gt_path_vol on device."""
        if self.gt_path_voxels is None or len(self.gt_path_voxels) == 0:
            self.gt_path_vol = None  # Explicitly set to None
            return

        self.gt_path_vol = torch.zeros_like(self.image, device=self.device)

        # Filter valid indices within image bounds
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
            print("Warning: No valid GT path voxels found within image bounds.")
            self.gt_path_vol = None  # Set back to None if no valid points
            return

        # Convert valid numpy indices to torch tensor for indexing
        self.gt_path_vol[tuple(valid_gt_path.T)] = 1.0

    def _get_state_patches(self) -> Dict[str, torch.Tensor]:
        """
        Get state patches centered at current position.
        Returns dict expected by observation/state space.
        Ensures tensors are on the correct device.
        """

        img_patch = get_patch(self.image, self.current_pos_vox, self.config.patch_size_vox)
        wall_patch = get_patch(self.wall_map, self.current_pos_vox, self.config.patch_size_vox)
        cum_path_patch = get_patch(
            self.cumulative_path_mask, self.current_pos_vox, self.config.patch_size_vox
        )

        actor_state = torch.stack([img_patch, wall_patch, cum_path_patch], dim=0)
        # # GT path can be None if not available, handle it
        # if self.gt_path_vol is not None:
        #     gt_path_patch = get_patch(
        #         self.gt_path_vol, self.current_pos_vox, self.config.patch_size_vox
        #     )
        # else:
        #     # Return a zero patch matching the expected shape/device/dtype
        #     gt_path_patch = torch.zeros(self.config.patch_size_vox, device=self.device)
        # critic_state = torch.stack([img_patch, wall_patch, cum_path_patch, gt_path_patch], dim=0)
        # return {"actor": actor_state, "critic": critic_state}
        return actor_state.unsqueeze(0)

    def _is_valid_pos(self, pos_vox: Tuple[int, int, int]) -> bool:
        """Check if a position is within the volume bounds."""
        s = self.image.shape
        return (0 <= pos_vox[0] < s[0]) and (0 <= pos_vox[1] < s[1]) and (0 <= pos_vox[2] < s[2])

    def _get_final_coverage(self) -> float:
        """Calculate coverage. Assumes tensors are valid and on device."""
        if self.seg_volume == 0:
            return 0.0
        intersection = torch.sum(self.cumulative_path_mask.float() * self.seg.float())
        return (intersection / self.seg_volume).item()

    def get_tracking_history(self) -> np.ndarray:
        """Get the history of tracked positions as a numpy array."""
        return np.array(self.tracking_path_history)

    def save_path_visualization(self, output_dir: str = "results"):
        """Saves visualization files (NIfTI, history, HTML plot)."""
        if not self._current_subject_data:
            print("Cannot save path visualization, no subject data loaded.")
            return
        if not hasattr(self, "cumulative_path_mask") or not hasattr(self, "tracking_path_history"):
            print("Cannot save path visualization, path tracking data missing.")
            return

        subject_id = self._current_subject_data.get("id", "unknown_subject")
        cache_dir = Path(output_dir) / subject_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save cumulative path mask
        path_mask_np = self.cumulative_path_mask.cpu().numpy()  # Move to CPU for numpy/nibabel
        nif = nib.nifti1.Nifti1Image(
            path_mask_np,
            affine=self.image_affine,
            header=nib.Nifti1Header(),
        )
        nif.header.set_zooms(self.spacing)
        nib.save(nif, cache_dir / "cumulative_path_mask.nii.gz")

        # Save the tracking history
        tracking_history_path = cache_dir / "tracking_history.npy"
        np.save(tracking_history_path, self.get_tracking_history())

        # PyVista visualization (optional, requires pyvista)
        try:
            plotter = pv.Plotter(off_screen=True)  # Use off_screen for non-interactive saving

            # Add volumes (use numpy arrays)
            if self.seg_np is not None:
                plotter.add_volume(
                    self.seg_np * 20, cmap="viridis", opacity="linear", name="small_bowel"
                )
            if (
                "colon" in self._current_subject_data
                and self._current_subject_data["colon"] is not None
            ):
                plotter.add_volume(
                    self._current_subject_data["colon"] * 80,
                    cmap="plasma",
                    opacity="linear",
                    name="colon",
                )
            if (
                "duodenum" in self._current_subject_data
                and self._current_subject_data["duodenum"] is not None
            ):
                plotter.add_volume(
                    self._current_subject_data["duodenum"] * 140,
                    cmap="magma",
                    opacity="linear",
                    name="duodenum",
                )

            # Add path lines and points
            path_points = self.get_tracking_history()
            if len(path_points) > 1:
                lines: pv.PolyData = pv.lines_from_points(path_points)
                plotter.add_mesh(
                    lines, line_width=5, color="cyan", name="path_line"
                )  # Use color instead of cmap
            if len(path_points) > 0:
                plotter.add_points(path_points, color="blue", point_size=8, name="path_points")

            plotter.show_axes()
            # plotter.export_html(cache_dir / "path_visualization.html") # HTML export
            plotter.screenshot(cache_dir / "path_visualization.png")  # Save screenshot
            plotter.close()  # Close plotter to free resources
        except Exception as e:
            print(f"Error during PyVista visualization: {e}")
            # Ensure plotter is closed even if error occurs
            if "plotter" in locals() and plotter:
                plotter.close()

    def _calculate_reward(
        self, action_vox_delta: Tuple[int, int, int], next_pos_vox: Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, Tuple]:  # Return reward as tensor
        """Calculate the reward for the current step. Returns tensor on device."""

        reward = torch.tensor(0.0, device=self.device)
        # Use _is_valid_pos first (cheaper check)
        if (
            not self._is_valid_pos(next_pos_vox)
            or not self.seg[next_pos_vox]
            or not any(action_vox_delta)
        ):
            reward -= self.config.r_val1
            return reward, tuple()

        # Calculate the line segment S using integer coordinates
        S_indices = line_nd(self.current_pos_vox, next_pos_vox, endpoint=True)

        # --- 2. GDT-based reward ---
        next_gdt_val = self.gdt[next_pos_vox]
        if next_gdt_val > self.max_gdt_achieved:
            delta = next_gdt_val - self.max_gdt_achieved
            # Penalty if too large (use tensor comparison)
            if delta > self.config.gdt_max_increase_theta:
                reward -= self.config.r_val2
            else:
                reward += self.config.r_val2 * delta / self.config.gdt_max_increase_theta
            # Update max_gdt_achieved (in-place update of tensor)
            self.max_gdt_achieved.fill_(next_gdt_val)

        # 2.5 Peaks-based reward (Optional - requires self.reward_map)
        # if hasattr(self, 'reward_map') and self.reward_map is not None:
        #     reward += self.reward_map[S_indices].sum() * self.config.r_peaks
        #     # Discard the reward for visited nodes (in-place update)
        #     self.reward_map[S_indices] = 0

        # --- 3. Wall-based penalty ---
        # Use tensor indexing for wall_map
        wall_penalty = self.wall_map[S_indices].mean()
        reward -= self.config.r_val2 * 2 * wall_penalty

        # --- 4. Revisiting penalty ---
        # Use tensor indexing for cumulative_path_mask
        reward -= self.config.r_val1 * self.cumulative_path_mask[S_indices].sum().bool()

        return reward, S_indices  # Return the numpy indices tuple S for path drawing

    # --- skrl Core Methods Implementation ---

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Resets the environment for skrl. Loads next subject if needed.
        Selects appropriate GDT based on start_choice.
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Initial observation dictionary and info dictionary.
        """

        # --- Subject Loading Logic ---
        # Increment episode counter *after* a successful run (tracked internally)
        # Check if we need to load a new subject
        should_load_new_subject = (
            self._current_subject_data is None  # First time call
            or self.episodes_on_current_subject >= self.num_episodes_per_sample
        )

        if should_load_new_subject:
            try:
                self._load_next_subject()
            except StopIteration:
                # Handle dataset exhaustion gracefully - maybe raise an error or signal completion
                raise StopIteration("Dataset iterator exhausted. Cannot reset environment.")
            # episodes_on_current_subject is reset inside _load_next_subject

        # --- Reset internal episode state ---
        self.current_step_count = 0
        self._episode_active = True  # Mark episode as started

        # Determine start position and select appropriate GDT (alternating start/end)
        if self.episodes_on_current_subject % 2 == 0:
            self.current_pos_vox = self.start_coord
            self.goal = self.end_coord
            self.gdt = self.gdt_start
        else:
            self.current_pos_vox = self.end_coord
            self.goal = self.start_coord
            self.gdt = self.gdt_end

        # Validate start position (simplified check using torch tensor)
        if not self._is_valid_pos(self.current_pos_vox) or not self.seg[self.current_pos_vox]:
            print(
                f"Warning: Chosen start pos {self.current_pos_vox} invalid/outside seg. Searching nearby..."
            )
            # Find valid voxels within the segmentation mask on the correct device
            valid_voxels = torch.nonzero(self.seg > 0, as_tuple=False)  # Get indices as Nx3 tensor
            if valid_voxels.shape[0] == 0:
                raise ValueError("Cannot reset: No valid voxels found in segmentation mask.")
            # Select a random valid voxel
            random_idx = random.randint(0, valid_voxels.shape[0] - 1)
            new_start_pos_tensor = valid_voxels[random_idx]
            self.current_pos_vox = tuple(new_start_pos_tensor.tolist())  # Convert back to tuple
            print(f"Resetting to random valid voxel: {self.current_pos_vox}")
            # Recompute GDT if start changed significantly? Or just use original goal?
            # For simplicity, keep original goal, but recompute GDT from new start
            self.goal = self.end_coord  # Or keep alternating goal based on original logic? Let's stick to original goal choice.
            self.gdt = compute_gdt(self.seg_np, self.current_pos_vox, self.spacing).to(self.device)

        # Initialize path tracking structures on the correct device
        self.cumulative_path_mask = torch.zeros_like(
            self.image, device=self.device, dtype=torch.uint8
        )

        # Draw initial path sphere at the starting position
        draw_path_sphere(
            self.cumulative_path_mask, self.current_pos_vox, self.config.cumulative_path_radius_vox
        )
        # Ensure mask is correct type after drawing if needed
        self.cumulative_path_mask = self.cumulative_path_mask.byte()  # Or keep as uint8

        self.tracking_path_history = [self.current_pos_vox]  # List of tuples

        # Initialize current GDT value using the selected GDT (ensure indices are valid)
        gdt_val_start = self.gdt[self.current_pos_vox]
        self.current_gdt_val.fill_(gdt_val_start)
        self.max_gdt_achieved.fill_(gdt_val_start)

        # --- Get Initial State Patches ---
        initial_obs_dict = self._get_state_patches()  # Returns dict of tensors on device

        # --- Prepare Output ---
        info = {
            "subject_id": self._current_subject_data.get("id", "N/A")
            if self._current_subject_data
            else "N/A"
        }

        # Observation dict already contains tensors on self.device
        return initial_obs_dict, info

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Performs a step in the environment using the provided actions.
        Args:
            actions (torch.Tensor): Actions tensor from the agent, shape (num_envs, action_dim), e.g., (1, 3).

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
            Next observation dict, reward tensor, terminated tensor, truncated tensor, info dict.
            Reward, terminated, truncated tensors have shape (num_envs,), e.g., (1,).
        """
        if not self._episode_active:
            raise RuntimeError("Cannot call step() before reset() or after an episode has ended.")

        # Ensure actions are on the correct device
        actions = actions.to(self.device)

        # --- Action Processing ---
        # Since num_envs is 1, squeeze the batch dimension
        action_normalized = actions.squeeze(0)  # Shape becomes (action_dim,) e.g., (3,)
        action_mapped = action_normalized * self.config.max_step_vox

        # Convert to integer voxel delta, ensuring result is tuple of ints
        action_vox_delta = tuple(torch.round(action_mapped).int().cpu().tolist())

        # --- Execute Step Logic ---
        self.current_step_count += 1
        # But the state update uses the rounded integer position
        next_pos_vox_int = (
            self.current_pos_vox[0] + action_vox_delta[0],
            self.current_pos_vox[1] + action_vox_delta[1],
            self.current_pos_vox[2] + action_vox_delta[2],
        )

        # --- Calculate reward and update state ---
        # Reward calculation needs the *potential* next integer position
        # _calculate_reward returns reward tensor and S (numpy indices tuple)
        reward_tensor, S_indices = self._calculate_reward(action_vox_delta, next_pos_vox_int)

        # Check if the *actual* next integer position is valid within segmentation
        is_next_pos_valid_seg = self._is_valid_pos(next_pos_vox_int) and self.seg[next_pos_vox_int]

        # Update current position *only if* the move is valid (within seg)
        # Otherwise, the agent stays in the same place but might receive penalty from reward calc
        if is_next_pos_valid_seg:
            self.current_pos_vox = next_pos_vox_int

            for vox in zip(*S_indices):
                # Ensure vox coords are within bounds before drawing
                draw_path_sphere(
                    self.cumulative_path_mask, vox, self.config.cumulative_path_radius_vox
                )

            # Ensure mask remains byte/uint8 type
            self.cumulative_path_mask = self.cumulative_path_mask.byte()

            # Update tracking history and current GDT value
            self.tracking_path_history.append(self.current_pos_vox)
            self.current_gdt_val.fill_(self.gdt[self.current_pos_vox])

        # --- Check Termination Conditions ---
        terminated, truncated = False, False
        termination_reason = ""
        # 1. Truncation: Max steps reached
        if self.current_step_count >= self.config.max_episode_steps:
            truncated, termination_reason = True, "max_steps"
        # 2. Termination: Agent moved out of segmentation or bounds
        elif not is_next_pos_valid_seg:
            terminated, termination_reason = True, "out_of_segmentation"
        # 3. Termination: Reached goal
        elif dist(self.current_pos_vox, self.goal) < self.config.cumulative_path_radius_vox:
            terminated, termination_reason = True, "reached_goal"

        # Determine overall 'done' flag for skrl
        done = terminated or truncated

        # --- Final Reward Adjustment ---
        final_coverage = 0.0
        if done:
            final_coverage = self._get_final_coverage()
            # Add bonus/penalty based on final coverage and termination reason
            reward_tensor += self.config.r_final * (
                final_coverage if termination_reason == "reached_goal" else (final_coverage - 1.0)
            )

        # --- Get Next State Patches ---
        next_obs_dict = self._get_state_patches()

        # --- Prepare Output for skrl ---
        # Ensure reward, terminated, truncated are tensors of shape (num_envs,) = (1,)
        # reward_tensor is already a scalar tensor on device
        reward_out = reward_tensor.unsqueeze(0)
        terminated_out = torch.as_tensor([terminated], device=self.device)
        truncated_out = torch.as_tensor([truncated], device=self.device)

        # Info dictionary
        info = {
            "step_count": self.current_step_count,
            "current_pos": self.current_pos_vox,
            "goal_pos": self.goal,
            "distance_to_goal": dist(self.current_pos_vox, self.goal),
            "current_gdt": self.current_gdt_val.item(),
            "max_gdt_achieved": self.max_gdt_achieved.item(),
            "termination_reason": termination_reason,
            "final_coverage": final_coverage if done else 0.0,
        }

        # If done, mark episode as inactive and increment subject episode counter
        if done:
            self._episode_active = False
            self.episodes_on_current_subject += 1
            # Optionally save visualization at the end of an episode
            # self.save_path_visualization()

        # next_obs_dict contains tensors on self.device
        return next_obs_dict, reward_out, terminated_out, truncated_out, info

    def render(self, *args, **kwargs) -> Any:
        """Rendering is not implemented for live display. Use save_path_visualization()."""
        # The original code saved plots to files. skrl's render usually implies live display.
        # We can either adapt save_path to return an image array (e.g., using pyvista offscreen)
        # or just state it's not implemented for live rendering.
        # print("Live rendering not available. Call save_path_visualization() to save artifacts.")
        # self.save_path_visualization() # Optionally trigger save on render call
        raise NotImplementedError("Live rendering not implemented. Use save_path_visualization().")

    def close(self) -> None:
        """Close the environment and release resources."""
        # Add any necessary cleanup here (e.g., closing files, stopping threads)
        print("Closing SmallBowelEnv.")
        # Reset internal state references if needed
        self.image = None
        self.seg = None
        self._current_subject_data = None
        self.dataset_iterator = None  # Allow garbage collection if iterator holds large refs

    def state(self) -> Dict[str, torch.Tensor]:
        """
        Get the current environment state dictionary (matches state_space).
        In this implementation, state is the same as observation.
        """
        # Ensure state is valid (e.g., after reset)
        if not self._episode_active and self.current_step_count == 0:
            # If called before first reset or after close, state might be invalid
            # Option 1: Raise error
            # raise RuntimeError("Cannot get state before environment reset.")
            # Option 2: Return zero state matching the space
            print(
                "Warning: state() called before reset or after episode end. Returning zero state."
            )
            zero_actor = torch.zeros(self.state_space["actor"].shape, device=self.device)
            zero_critic = torch.zeros(self.state_space["critic"].shape, device=self.device)
            return {"actor": zero_actor, "critic": zero_critic}

        # Normally, return the latest observation patches
        return self._get_state_patches()
