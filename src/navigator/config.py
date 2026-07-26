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
    reload_checkpoint_path: Optional[str] = None  # Path to checkpoint for evaluation
    seed: int = 42  # Random seed for reproducibility
    td3: bool = False
    train_gym_env: bool = False  # Dummy flag to train a gym environment

    # --- Dataset Parameters ---
    train_val_split: float = 0.8  # Fraction of data to use for training
    shuffle_dataset: bool = True  # Whether to shuffle dataset before splitting
    amp: bool = False
    amp_dtype: str = "bf16"

    # --- Wandb Logging ---
    track_wandb: bool = True  # Flag to enable/disable wandb
    wandb_project_name: str = "toydata"  # toydata
    wandb_entity: Optional[str] = None  # Your wandb username or team name (optional)
    wandb_run_name: Optional[str] = None  # Optional run name, defaults to auto-generated

    # --- Environment Hyperparameters ---
    voxel_size_mm: float = 1.0
    patch_size_mm: int = 16
    max_step_displacement_mm: float = 6
    use_immediate_gdt_reward: bool = True
    max_episode_steps: int = 2048
    cumulative_path_radius_mm: float = 6.0
    # Traversable space is the segmentation by default. A large dilation lets
    # the policy jump across nearby bowel loops and solve only the endpoint task.
    allowed_area_radius_mm: float = 0.0
    # wall_map_sigmas: Tuple[int, ...] = (1, 3)
    wall_map_sigmas: Tuple[int, ...] = (1,)

    # --- Reward Hyperparameters ---
    # Keep dense penalties on the same scale as one step of GDT progress. Large
    # per-step costs make deliberate early termination optimal.
    r_val1: float = 0.25
    r_val2: float = 1.0
    r_zero_mov: float = 1.0
    r_final: float = 50.0
    coverage_reward_scale: float = 50.0
    success_coverage_threshold: float = 0.55
    step_penalty: float = 0.01
    wall_penalty_scale: float = 0.1
    # Reward for passing through must-pass nodes
    r_peaks: float = 4.0
    r_val3: float = 0.1
    # An endpoint-directed prior bypasses visual navigation on the phantoms.
    goal_action_prior: float = 0.0
    log_episode_ends: bool = False

    # --- Training Hyperparameters ---
    # For each subject, how many episodes to run before switching to the next one (#16384)
    num_episodes_per_sample: int = 256  # 32768
    num_steps_per_sample: int = 8192
    behavior_cloning_epochs: int = 0
    behavior_cloning_learning_rate: float = 3e-4
    behavior_cloning_batch_size: int = 64
    behavior_cloning_max_policy_probability: float = 1.0
    # Write the code to force the agent to always move
    # num_episodes_per_sample: int = 32
    total_timesteps: int = 10_000_000
    # Size of the buffer to store transitions
    frames_per_batch: int = 4096
    learning_rate: float = 5e-5
    batch_size: int = 256  # Size of mini-batch for PPO update
    update_epochs: int = 5  # Number of PPO update epochs
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    # Entropy coefficient for exploration (higher values encourage exploration)
    ent_coef: float = 0.003
    # Value function coefficient (higher values encourage accurate value estimates)
    vf_coef: float = 0.5
    num_workers: int = 1

    max_grad_norm: float = 0.5
    eval_interval: int = 3000  # Interval for evaluation
    save_freq: int = 500  # Frequency to save model checkpoints
    metric_to_optimize: str = "validation/avg_coverage"

    # --- Training/Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Derived Parameters ---
    gdt_cell_length: float = field(init=False)
    max_step_vox: int = field(init=False)
    patch_size_vox: Tuple[int, int, int] = field(init=False)
    cumulative_path_radius_vox: int = field(init=False)
    allowed_area_radius_vox: int = field(init=False)
    gdt_max_increase_theta: float = field(init=False)
    observation_channels: int = field(init=False, default=4)
    context_features: int = field(init=False, default=5)

    def __post_init__(self):
        def mm_to_vox(dist_mm: float, voxel_dim_mm: float) -> int:
            """Convert millimeter distance to voxel units."""
            return int(dist_mm // voxel_dim_mm)

        if self.voxel_size_mm <= 0:
            raise ValueError("voxel_size_mm must be positive")
        if self.allowed_area_radius_mm < 0:
            raise ValueError("allowed_area_radius_mm must be non-negative")
        if not 0 <= self.success_coverage_threshold <= 1:
            raise ValueError("success_coverage_threshold must be between 0 and 1")
        if self.coverage_reward_scale < 0:
            raise ValueError("coverage_reward_scale must be non-negative")
        if self.step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")
        if self.behavior_cloning_epochs < 0:
            raise ValueError("behavior_cloning_epochs must be non-negative")
        if self.behavior_cloning_learning_rate <= 0:
            raise ValueError("behavior_cloning_learning_rate must be positive")
        if self.behavior_cloning_batch_size < 1:
            raise ValueError("behavior_cloning_batch_size must be positive")
        if not 0 <= self.behavior_cloning_max_policy_probability <= 1:
            raise ValueError(
                "behavior_cloning_max_policy_probability must be between 0 and 1"
            )

        self.checkpoint_dir = self.checkpoint_dir + "/" + self.data_dir
        self.gdt_cell_length = self.voxel_size_mm
        self.max_step_vox = mm_to_vox(self.max_step_displacement_mm, self.voxel_size_mm)
        patch_vox_dim = mm_to_vox(self.patch_size_mm, self.voxel_size_mm)
        self.patch_size_vox = (patch_vox_dim,) * 3
        self.cumulative_path_radius_vox = mm_to_vox(
            self.cumulative_path_radius_mm, self.voxel_size_mm
        )
        self.allowed_area_radius_vox = mm_to_vox(self.allowed_area_radius_mm, self.voxel_size_mm)
        if self.max_step_vox < 1:
            raise ValueError("max_step_displacement_mm must span at least one voxel")
        if patch_vox_dim < 8:
            raise ValueError("patch_size_mm must span at least eight voxels")
        if self.goal_action_prior < 0:
            raise ValueError("goal_action_prior must be non-negative")
        self.gdt_max_increase_theta = self.max_step_vox * self.voxel_size_mm * math.sqrt(3)


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
            "allowed_area_radius_vox",
            "gdt_max_increase_theta",
            "observation_channels",
            "context_features",
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
