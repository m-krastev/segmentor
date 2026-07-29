"""
Configuration classes and argument parsing for Navigator.
"""

import argparse
from dataclasses import dataclass, field
from itertools import product
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
    nnunet_raw_dir: Optional[str] = None
    nnunet_cache_dir: str = "results/navigator_nnunet/cache"
    nnunet_case_ids_file: Optional[str] = None
    nnunet_train_case_ids_file: Optional[str] = None
    nnunet_val_case_ids_file: Optional[str] = None
    # Directory containing one externally supplied native-XYZ seed coordinate
    # per case as <case_id>.txt. Annotation-free mode deliberately refuses to
    # derive seeds from organ labels or the GT path.
    nnunet_seed_dir: Optional[str] = None
    nnunet_generate_expert_path: bool = False
    amp: bool = False
    amp_dtype: str = "bf16"

    # --- Wandb Logging ---
    track_wandb: bool = True  # Flag to enable/disable wandb
    wandb_project_name: str = "toydata"  # toydata
    wandb_entity: Optional[str] = None  # Your wandb username or team name (optional)
    wandb_run_name: Optional[str] = None  # Optional run name, defaults to auto-generated

    # --- TensorBoard Logging ---
    track_tensorboard: bool = False
    # Exact run directory; otherwise a timestamped directory is created below
    # <checkpoint_dir>/tensorboard.
    tensorboard_log_dir: Optional[str] = None
    validation_save_paths: bool = False
    validation_output_dir: Optional[str] = None
    deterministic_action_statistic: str = "mean"

    # --- Environment Hyperparameters ---
    voxel_size_mm: float = 1.0
    patch_size_mm: int = 16
    max_step_displacement_mm: float = 6
    use_immediate_gdt_reward: bool = True
    max_episode_steps: int = 2048
    terminate_on_success: bool = True
    # Enforce that labels/endpoints cannot affect policy observations,
    # transitions, rewards, or termination. A single external start seed is
    # still required because a path tracker is undefined without one.
    annotation_free: bool = False
    # Training-only GT potentials with image-only policy inputs and
    # bounds-only action dynamics. This is deployable without labels but is
    # not annotation-free training.
    reward_supervised: bool = False
    observe_goal_distance: bool = False
    # Explicit supervised-input baseline. This exposes the local GT
    # segmentation patch to the policy and must never be described as
    # annotation-free or image-only.
    observe_segmentation: bool = False
    # ``navigation_filters`` is the current six-channel image-only state:
    # CT, four multiscale filter responses, and the dilated path tube.
    # ``shin_068_repaired`` is the controlled historical ablation: CT, the
    # original Meijering wall response, and an undilated centerline path. It
    # also removes absolute position/time from context and retains only the
    # explicitly allowed previous movement direction.
    policy_observation_contract: str = "navigation_filters"
    coverage_gated_goal_planner: bool = False
    # A 9 mm radius corresponds to an 18 mm diameter at the 1.5 mm nnU-Net
    # spacing, within the expected small-bowel caliber. Endpoint tolerance is a
    # separate localization criterion and must never be inferred from this.
    cumulative_path_radius_mm: float = 9.0
    endpoint_tolerance_mm: float = 3.0
    # Traversable space is the segmentation by default. A large dilation lets
    # the policy jump across nearby bowel loops and solve only the endpoint task.
    allowed_area_radius_mm: float = 0.0
    # wall_map_sigmas: Tuple[int, ...] = (1, 3)
    wall_map_sigmas: Tuple[int, ...] = (1,)
    navigation_filter_scales_mm: Tuple[float, ...] = (3.0, 6.0, 9.0)

    # --- Reward Hyperparameters ---
    # ``potential`` is the calibrated reward used by the current experiments.
    # ``shin_normalized`` is an explicit, unit-normalized implementation of
    # Shin & Summers (MICCAI 2022) Algorithm 1. The ``_guarded`` variant also
    # rejects background-crossing segments and requires the registered Dice
    # threshold for positive terminal reward. ``shin_normalized_repaired``
    # preserves that strict terminal condition but replaces the binary
    # off-target cliff with a physical distance penalty and gates GDT credit
    # for background-crossing segments. All Shin contracts use GT
    # segmentation/GDT in the reward and can never be called annotation-free.
    reward_contract: str = "potential"
    # Keep dense penalties on the same scale as one step of GDT progress. Large
    # per-step costs make deliberate early termination optimal.
    r_val1: float = 0.25
    r_val2: float = 1.0
    r_zero_mov: float = 1.0
    r_final: float = 50.0
    coverage_reward_scale: float = 50.0
    # Total return available for monotonic mask-constrained progress to the
    # requested endpoint. This is a telescoping potential, not a per-step bonus.
    gdt_reward_scale: float = 1.0
    # ``initial_distance`` preserves the historical total-return normalization.
    # ``max_step`` gives the same physical GDT displacement the same reward
    # across subjects and prevents long paths from erasing local progress.
    gdt_progress_normalization: str = "initial_distance"
    # Reward-only potential for recovering after an unconstrained action leaves
    # the supervised target. Zero preserves legacy behavior.
    target_recovery_reward_scale: float = 0.0
    # Persistent cost for distance from the endpoint-connected target. It is
    # evaluated over the complete action segment, not only at the endpoint.
    target_distance_penalty_scale: float = 0.0
    target_distance_penalty_radius_mm: float = 600.0
    # When enabled, a segment that crosses background cannot receive positive
    # GDT, Dice, or supervised episodic-cell shaping.
    gate_positive_shaping_on_target_segment: bool = False
    # Optional calibrated terminal protocol. A non-negative failure value
    # enables a fixed success bonus and explicit failure penalty. The default
    # negative sentinel preserves the historical coverage-scaled terminal.
    terminal_success_bonus: float = 0.0
    terminal_failure_penalty: float = -1.0
    success_coverage_threshold: float = 0.55
    step_penalty: float = 0.01
    # Penalize the fraction of a newly executed centerline segment that was
    # already visited. The mandatory starting voxel is excluded, and the
    # undilated agent-owned centerline is used so ordinary forward motion
    # inside the evaluation tube is not misclassified as a revisit.
    revisit_penalty_scale: float = 0.0
    wall_penalty_scale: float = 0.1
    # Image/self-state-only objectives used by annotation-free training.
    intrinsic_novelty_reward_scale: float = 0.05
    # E3B-inspired episodic first-visit bonus over controllable spatial cells.
    # This is label-free, decreases as cells are discovered, and is opt-in so
    # existing reward protocols remain reproducible.
    episodic_cell_reward_scale: float = 0.0
    episodic_cell_size_mm: float = 6.0
    curvature_penalty_scale: float = 0.02
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
    behavior_cloning_action_statistic: str = "mean"
    # Recurrent policies share one visual encoder between actor and critic.
    # "s5" is a dependency-free diagonal S5-style state-space baseline.
    memory_model: str = "none"
    memory_hidden_size: int = 256
    memory_num_layers: int = 1
    s5_state_size: int = 256
    recurrent_sequence_length: int = 64
    recurrent_backend: str = "pad"
    # "beta" reproduces the original continuous policy. "categorical"
    # assigns one joint category to each nonzero integer displacement.
    # "masked_categorical" uses the same joint support but renormalizes over
    # only the nonzero displacements whose endpoints remain inside the image.
    # "factorized_categorical" models the three exact integer coordinates
    # with independent categorical factors, avoiding a 728-way output head.
    action_distribution: str = "beta"
    # Write the code to force the agent to always move
    # num_episodes_per_sample: int = 32
    total_timesteps: int = 10_000_000
    # Size of the buffer to store transitions
    frames_per_batch: int = 4096
    learning_rate: float = 5e-5
    # Zero anneals over the complete run. A positive value reaches the minimum
    # learning rate after this many frames and stays there, preventing a longer
    # job from silently changing a validated short-run schedule.
    lr_anneal_timesteps: int = 0
    batch_size: int = 256  # Size of mini-batch for PPO update
    update_epochs: int = 5  # Number of PPO update epochs
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    # Stop the remaining PPO epochs for a rollout after this mean approximate
    # KL is exceeded. Zero disables the guard.
    target_kl: float = 0.0
    # Entropy coefficient for exploration (higher values encourage exploration)
    ent_coef: float = 0.003
    # Value function coefficient (higher values encourage accurate value estimates)
    vf_coef: float = 0.5
    # When enabled, critic gradients update only the value head; the shared
    # visual encoder and memory receive policy gradients exclusively.
    separate_actor_critic_losses: bool = False
    num_workers: int = 1

    max_grad_norm: float = 0.5
    eval_interval: int = 3000  # Interval for evaluation
    save_freq: int = 500  # Frequency to save model checkpoints
    metric_to_optimize: str = "validation/traversal_success_rate"

    # --- Training/Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Derived Parameters ---
    gdt_cell_length: float = field(init=False)
    max_step_vox: int = field(init=False)
    patch_size_vox: Tuple[int, int, int] = field(init=False)
    cumulative_path_radius_vox: int = field(init=False)
    endpoint_tolerance_vox: float = field(init=False)
    allowed_area_radius_vox: int = field(init=False)
    episodic_cell_size_vox: int = field(init=False)
    gdt_max_increase_theta: float = field(init=False)
    needs_target_distance: bool = field(init=False)
    observation_channels: int = field(init=False, default=5)
    # Legacy: time, geodesic progress, Dice coverage, normalized position (3),
    # previous direction (3), and goal direction (3). Annotation-free mode
    # derives a seven-value context in __post_init__.
    context_features: int = field(init=False, default=12)
    clean_policy_inputs: bool = field(init=False, default=False)
    action_displacements: tuple[tuple[int, int, int], ...] = field(
        init=False,
        default=(),
    )
    categorical_action_count: int = field(init=False, default=0)
    factorized_axis_action_count: int = field(init=False, default=0)

    def __post_init__(self):
        def mm_to_vox(dist_mm: float, voxel_dim_mm: float) -> int:
            """Convert millimeter distance to voxel units."""
            return int(dist_mm // voxel_dim_mm)

        if self.voxel_size_mm <= 0:
            raise ValueError("voxel_size_mm must be positive")
        if not 0 < self.train_val_split < 1:
            raise ValueError("train_val_split must be between zero and one")
        if self.allowed_area_radius_mm < 0:
            raise ValueError("allowed_area_radius_mm must be non-negative")
        if self.cumulative_path_radius_mm < 0:
            raise ValueError("cumulative_path_radius_mm must be non-negative")
        if self.endpoint_tolerance_mm < 0:
            raise ValueError("endpoint_tolerance_mm must be non-negative")
        if not 0 <= self.success_coverage_threshold <= 1:
            raise ValueError("success_coverage_threshold must be between 0 and 1")
        if self.coverage_reward_scale < 0:
            raise ValueError("coverage_reward_scale must be non-negative")
        if self.reward_contract not in {
            "potential",
            "shin_normalized",
            "shin_normalized_guarded",
            "shin_normalized_repaired",
        }:
            raise ValueError(
                "reward_contract must be one of: potential, shin_normalized, "
                "shin_normalized_guarded, shin_normalized_repaired"
            )
        if self.annotation_free and self.reward_contract.startswith(
            "shin_normalized"
        ):
            raise ValueError(
                "shin_normalized reward uses GT segmentation and GDT and is "
                "incompatible with annotation_free mode"
            )
        if self.gdt_reward_scale < 0:
            raise ValueError("gdt_reward_scale must be non-negative")
        if self.gdt_progress_normalization not in {
            "initial_distance",
            "max_step",
        }:
            raise ValueError(
                "gdt_progress_normalization must be either "
                "'initial_distance' or 'max_step'"
            )
        if self.target_recovery_reward_scale < 0:
            raise ValueError("target_recovery_reward_scale must be non-negative")
        if self.target_distance_penalty_scale < 0:
            raise ValueError("target_distance_penalty_scale must be non-negative")
        if self.target_distance_penalty_radius_mm <= 0:
            raise ValueError("target_distance_penalty_radius_mm must be positive")
        if self.terminal_success_bonus < 0:
            raise ValueError("terminal_success_bonus must be non-negative")
        if self.terminal_failure_penalty < -1:
            raise ValueError("terminal_failure_penalty must be at least -1")
        if self.step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")
        if self.revisit_penalty_scale < 0:
            raise ValueError("revisit_penalty_scale must be non-negative")
        if self.intrinsic_novelty_reward_scale < 0:
            raise ValueError("intrinsic_novelty_reward_scale must be non-negative")
        if self.episodic_cell_reward_scale < 0:
            raise ValueError("episodic_cell_reward_scale must be non-negative")
        if self.episodic_cell_size_mm <= 0:
            raise ValueError("episodic_cell_size_mm must be positive")
        if self.curvature_penalty_scale < 0:
            raise ValueError("curvature_penalty_scale must be non-negative")
        if not self.navigation_filter_scales_mm or any(
            scale <= 0 for scale in self.navigation_filter_scales_mm
        ):
            raise ValueError(
                "navigation_filter_scales_mm must contain positive values"
            )
        if self.policy_observation_contract not in {
            "navigation_filters",
            "shin_068_repaired",
        }:
            raise ValueError(
                "policy_observation_contract must be one of: "
                "navigation_filters, shin_068_repaired"
            )
        if (
            self.policy_observation_contract == "shin_068_repaired"
            and self.observe_segmentation
        ):
            raise ValueError(
                "shin_068_repaired policy observations cannot include the GT "
                "segmentation channel"
            )
        if (
            self.policy_observation_contract == "shin_068_repaired"
            and not (self.annotation_free or self.reward_supervised)
        ):
            raise ValueError(
                "shin_068_repaired policy observations require clean policy "
                "inputs via reward_supervised or annotation_free mode"
            )
        if self.behavior_cloning_epochs < 0:
            raise ValueError("behavior_cloning_epochs must be non-negative")
        if self.behavior_cloning_learning_rate <= 0:
            raise ValueError("behavior_cloning_learning_rate must be positive")
        if self.lr_anneal_timesteps < 0:
            raise ValueError("lr_anneal_timesteps must be non-negative")
        if self.target_kl < 0:
            raise ValueError("target_kl must be non-negative")
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be positive")
        if self.save_freq < 1:
            raise ValueError("save_freq must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if self.behavior_cloning_batch_size < 1:
            raise ValueError("behavior_cloning_batch_size must be positive")
        if not 0 <= self.behavior_cloning_max_policy_probability <= 1:
            raise ValueError("behavior_cloning_max_policy_probability must be between 0 and 1")
        if self.behavior_cloning_action_statistic not in {"mean", "mode"}:
            raise ValueError(
                "behavior_cloning_action_statistic must be either 'mean' or 'mode'"
            )
        if self.deterministic_action_statistic not in {"mean", "mode"}:
            raise ValueError(
                "deterministic_action_statistic must be either 'mean' or 'mode'"
            )
        if self.memory_model not in {"none", "gru", "s5"}:
            raise ValueError("memory_model must be one of: none, gru, s5")
        if self.memory_hidden_size < 1:
            raise ValueError("memory_hidden_size must be positive")
        if self.memory_num_layers < 1:
            raise ValueError("memory_num_layers must be positive")
        if self.s5_state_size < 1:
            raise ValueError("s5_state_size must be positive")
        if self.recurrent_sequence_length < 1:
            raise ValueError("recurrent_sequence_length must be positive")
        if self.recurrent_backend not in {"auto", "pad", "scan", "triton"}:
            raise ValueError(
                "recurrent_backend must be one of: auto, pad, scan, triton"
            )
        if self.action_distribution not in {
            "beta",
            "categorical",
            "masked_categorical",
            "factorized_categorical",
        }:
            raise ValueError(
                "action_distribution must be one of: beta, categorical, "
                "masked_categorical, factorized_categorical"
            )
        if self.action_distribution != "beta":
            if self.memory_model == "none":
                raise ValueError(
                    "categorical actions currently require a recurrent policy"
                )
            if self.deterministic_action_statistic != "mode":
                raise ValueError(
                    "categorical actions require deterministic mode evaluation"
                )
            if self.behavior_cloning_epochs:
                raise ValueError(
                    "categorical actions do not yet support behavior cloning"
                )
        if self.action_distribution == "masked_categorical" and not (
            self.annotation_free or self.reward_supervised
        ):
            raise ValueError(
                "masked_categorical requires clean bounds-only policy dynamics "
                "via reward_supervised or annotation_free mode"
            )
        if self.memory_model != "none" and self.td3:
            raise ValueError("Recurrent memory baselines currently support PPO only")
        if self.annotation_free and self.reward_supervised:
            raise ValueError(
                "annotation_free and reward_supervised are mutually exclusive"
            )
        if self.annotation_free:
            incompatible = {
                "observe_goal_distance": self.observe_goal_distance,
                "observe_segmentation": self.observe_segmentation,
                "coverage_gated_goal_planner": self.coverage_gated_goal_planner,
                "use_immediate_gdt_reward": self.use_immediate_gdt_reward,
                "terminate_on_success": self.terminate_on_success,
                "coverage_reward_scale": self.coverage_reward_scale != 0,
                "gdt_reward_scale": self.gdt_reward_scale != 0,
                "target_recovery_reward_scale": (
                    self.target_recovery_reward_scale != 0
                ),
                "target_distance_penalty_scale": (
                    self.target_distance_penalty_scale != 0
                ),
                "gate_positive_shaping_on_target_segment": (
                    self.gate_positive_shaping_on_target_segment
                ),
                "terminal_success_bonus": self.terminal_success_bonus != 0,
                "terminal_failure_penalty": self.terminal_failure_penalty >= 0,
                "r_final": self.r_final != 0,
                "r_val1": self.r_val1 != 0,
                "goal_action_prior": self.goal_action_prior != 0,
                "behavior_cloning_epochs": self.behavior_cloning_epochs != 0,
                "nnunet_generate_expert_path": self.nnunet_generate_expert_path,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "annotation_free mode forbids privileged options: "
                    + ", ".join(enabled)
                )
            if self.nnunet_raw_dir and not self.nnunet_seed_dir:
                raise ValueError(
                    "annotation_free nnU-Net training requires --nnunet-seed-dir "
                    "with externally supplied seed points"
                )
        if self.reward_supervised:
            incompatible = {
                "observe_goal_distance": self.observe_goal_distance,
                "coverage_gated_goal_planner": self.coverage_gated_goal_planner,
                "goal_action_prior": self.goal_action_prior != 0,
                "behavior_cloning_epochs": self.behavior_cloning_epochs != 0,
                "nnunet_generate_expert_path": self.nnunet_generate_expert_path,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "reward_supervised mode forbids privileged policy options: "
                    + ", ".join(enabled)
                )
            if self.nnunet_raw_dir and not self.nnunet_seed_dir:
                raise ValueError(
                    "reward_supervised nnU-Net training requires "
                    "--nnunet-seed-dir with externally supplied seed points"
                )

        self.checkpoint_dir = self.checkpoint_dir + "/" + self.data_dir
        self.gdt_cell_length = self.voxel_size_mm
        self.max_step_vox = mm_to_vox(self.max_step_displacement_mm, self.voxel_size_mm)
        self.action_displacements = tuple(
            displacement
            for displacement in product(
                range(-self.max_step_vox, self.max_step_vox + 1),
                repeat=3,
            )
            if any(displacement)
        )
        self.categorical_action_count = len(self.action_displacements)
        self.factorized_axis_action_count = 2 * self.max_step_vox + 1
        patch_vox_dim = mm_to_vox(self.patch_size_mm, self.voxel_size_mm)
        self.patch_size_vox = (patch_vox_dim,) * 3
        self.cumulative_path_radius_vox = mm_to_vox(
            self.cumulative_path_radius_mm, self.voxel_size_mm
        )
        self.endpoint_tolerance_vox = self.endpoint_tolerance_mm / self.voxel_size_mm
        self.allowed_area_radius_vox = mm_to_vox(self.allowed_area_radius_mm, self.voxel_size_mm)
        self.episodic_cell_size_vox = max(
            1,
            mm_to_vox(self.episodic_cell_size_mm, self.voxel_size_mm),
        )
        self.clean_policy_inputs = self.annotation_free or self.reward_supervised
        if self.clean_policy_inputs:
            if self.policy_observation_contract == "shin_068_repaired":
                # CT, original wall response, and undilated centerline path;
                # context is only the preceding movement direction.
                self.observation_channels = 3
                self.context_features = 3
            else:
                # Current CT, four physically scaled image-filter responses,
                # and the agent's own cumulative path.
                self.observation_channels = 6 + int(self.observe_segmentation)
                # Time, normalized position (3), and previous direction (3).
                self.context_features = 7
        else:
            self.observation_channels = 5 + int(self.observe_goal_distance)
            self.context_features = 12
        if self.max_step_vox < 1:
            raise ValueError("max_step_displacement_mm must span at least one voxel")
        if patch_vox_dim < 8:
            raise ValueError("patch_size_mm must span at least eight voxels")
        if self.goal_action_prior < 0:
            raise ValueError("goal_action_prior must be non-negative")
        self.gdt_max_increase_theta = self.max_step_vox * self.voxel_size_mm * math.sqrt(3)
        self.needs_target_distance = (
            self.reward_contract == "shin_normalized_repaired"
            or (
                self.reward_contract == "potential"
                and bool(
                    self.target_recovery_reward_scale
                    or self.target_distance_penalty_scale
                    or self.gate_positive_shaping_on_target_segment
                )
            )
        )


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
            "endpoint_tolerance_vox",
            "allowed_area_radius_vox",
            "episodic_cell_size_vox",
            "gdt_max_increase_theta",
            "needs_target_distance",
            "observation_channels",
            "context_features",
            "clean_policy_inputs",
            "action_displacements",
            "categorical_action_count",
            "factorized_axis_action_count",
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
        if field_name == "navigation_filter_scales_mm":
            parser.add_argument(
                f"--{field_name.replace('_', '-')}",
                type=float,
                nargs="+",
                default=default_config.navigation_filter_scales_mm,
                help=(
                    f"{field_name} "
                    f"(default: {default_config.navigation_filter_scales_mm})"
                ),
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
