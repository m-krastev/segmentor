"""
Configuration classes and argument parsing for Navigator.
"""

import argparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union
import math
import torch


@dataclass
class Config:
    # --- Input/Output ---
    data_dir: str = "data"  # Directory containing dataset
    save_path: str = "ppo_small_bowel_tracker.pth"
    checkpoint_dir: str = "checkpoints"  # Directory to save checkpoints
    eval_only: bool = False  # Flag to run evaluation only
    load_from_checkpoint: Optional[str] = None  # Path to checkpoint for evaluation
    seed: int = 42 # Random seed for reproducibility

    # --- Dataset Parameters ---
    train_val_split: float = 0.8  # Fraction of data to use for training
    shuffle_dataset: bool = True  # Whether to shuffle dataset before splitting

    # --- Wandb Logging ---
    track_wandb: bool = True  # Flag to enable/disable wandb
    wandb_project_name: str = "SmallBowelTorchRL"
    wandb_entity: Optional[str] = None  # Your wandb username or team name (optional)
    wandb_run_name: Optional[str] = None  # Optional run name, defaults to auto-generated

    # --- Environment Hyperparameters ---
    voxel_size_mm: float = 1.5
    patch_size_mm: int = 60
    max_step_displacement_mm: float = 9.0
    use_immediate_gdt_reward: bool = False
    max_episode_steps: int = 1024
    cumulative_path_radius_mm: float = 6.0
    wall_map_sigmas: Tuple[int, ...] = (1, 3)

    # --- Reward Hyperparameters ---
    # Typically a penalty related to the game mechanics, e.g. zero movement, crossing walls, out of segmentation, etc.
    r_val1: float = 12.0
    # More active reward, e.g. moving towards the target, used along with the GDT
    r_val2: float = 6.0
    r_zero_mov: float = 400
    r_final: float = 100.0
    # Reward for passing through must-pass nodes?
    r_peaks: float = 16.0

    # --- Training Hyperparameters ---
    # For each subject, how many episodes to run before switching to the next one
    num_episodes_per_sample: int = 3_332_768
    total_timesteps: int = 10_000_000
    # Size of the buffer to store transitions
    frames_per_batch: int = 512
    learning_rate: float = 3e-5
    batch_size: int = 128  # Size of mini-batch for PPO update
    update_epochs: int = 5  # Number of PPO update epochs
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    # Entropy coefficient for exploration (higher values encourage exploration)
    ent_coef: float = 0.001
    # Value function coefficient (higher values encourage accurate value estimates)
    vf_coef: float = 0.5

    max_grad_norm: float = 0.25
    eval_interval: int = 1000  # Interval for evaluation
    save_freq: int = 1000  # Frequency to save model checkpoints
    metric_to_optimize: str = "validation/avg_coverage"

    # --- Training/Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Derived Parameters ---
    gdt_cell_length: float = field(init=False)
    max_step_vox: int = field(init=False)
    patch_size_vox: Tuple[int, int, int] = field(init=False)
    cumulative_path_radius_vox: int = field(init=False)
    gdt_max_increase_theta: float = field(init=False)

    def __post_init__(self):
        def mm_to_vox(dist_mm: float, voxel_dim_mm: float) -> int:
            """Convert millimeter distance to voxel units."""
            return int(dist_mm // voxel_dim_mm)

        self.gdt_cell_length = self.voxel_size_mm
        self.max_step_vox = mm_to_vox(self.max_step_displacement_mm, self.voxel_size_mm)
        patch_vox_dim = mm_to_vox(self.patch_size_mm, self.voxel_size_mm)
        self.patch_size_vox = (patch_vox_dim,) * 3
        self.cumulative_path_radius_vox = mm_to_vox(
            self.cumulative_path_radius_mm, self.voxel_size_mm
        )
        self.gdt_max_increase_theta = max(0.0, self.max_step_displacement_mm * 10 * math.sqrt(3))


def parse_args() -> Config:
    """
    Parse command line arguments and create a Config object.

    Returns:
        Config: Configuration object with parsed values.
    """
    parser = argparse.ArgumentParser(
        description="Deep Reinforcement Learning for Small Bowel Path Tracking"
    )
    default_config = Config()
    # Add arguments dynamically from Config fields
    for field_name, field_type in Config.__annotations__.items():
        if field_name in [
            "gdt_cell_length",
            "max_step_vox",
            "patch_size_vox",
            "cumulative_path_radius_vox",
            "gdt_max_increase_theta",
        ]:
            continue

        if field_name == "wall_map_sigmas":
            # Special case for wall_map_sigmas to accept a list of integers
            parser.add_argument(
                f"--{field_name.replace('_', '-')}",
                type=int,
                nargs="+",
                default=default_config.wall_map_sigmas,
                help=f"{field_name} (default: {default_config.wall_map_sigmas})",
            )
            continue

        # Handle default values and types
        default_val = getattr(default_config, field_name)
        arg_type = field_type
        required = False

        # Handle Optional type hint for argparse type
        if (
            hasattr(field_type, "__origin__")
            and field_type.__origin__ is Union
            and type(None) in field_type.__args__
        ):
            # Special case for wandb_entity and wandb_run_name which can be None
            if field_name in ["wandb_entity", "wandb_run_name"]:
                arg_type = str  # Expect string or nothing
            else:
                arg_type = field_type.__args__[0]

        if arg_type is bool:
            # Use BooleanOptionalAction for flags like --track-wandb / --no-track-wandb
            parser.add_argument(
                f"--{field_name.replace('_', '-')}",
                action=argparse.BooleanOptionalAction,
                default=default_val,
                help=f"{field_name} (default: {default_val})",
            )
        else:
            parser.add_argument(
                f"--{field_name.replace('_', '-')}",
                type=arg_type,
                default=default_val,
                required=required,
                help=f"{field_name} (default: {default_val})",
            )

    args = parser.parse_args()
    config_dict = {k: v for k, v in vars(args).items() if k in Config.__annotations__}
    config = Config(**config_dict)
    return config
