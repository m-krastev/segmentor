import torch
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import argparse

# Use IterableDataset concept
from torch.utils.data import DataLoader

# TorchRL components
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict import TensorDict

# Your project components
from src.navigator.config import Config
from src.navigator.dataset import MRIPathDataset
from src.navigator.models.dummy_net import make_dummy_actor_critic
from src.navigator.environment import make_mri_path_env, POSITION_HISTORY_LENGTH
from src.navigator.utils import draw_path_sphere_2

torch.set_float32_matmul_precision("medium")
torch._dynamo.config.capture_dynamic_output_shape_ops = True


def inference_loop_mri_path(
    actor_module,
    config: Config,
    inference_dataset: MRIPathDataset,
    device: torch.device = None,
    output_dir: str | Path = "inference_results",
):
    """
    Inference loop for MRI Path Dataset.
    The agent tries to traverse a small bowel segment as outlined by the segmentation,
    starting from the first point of each path.
    """
    actor_module.eval()  # Set actor to evaluation mode
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create an inference environment instance using the MRIPathEnv
    inference_env = make_mri_path_env(config, inference_dataset, device, num_episodes_per_sample=1, check_env=False)

    num_inference_subjects = len(inference_dataset)

    print(f"Starting inference for {num_inference_subjects} subjects...")

    with (
        torch.no_grad(),
        set_exploration_type(ExplorationType.MODE),
    ):  # Use deterministic actions for inference
        for i in tqdm(range(num_inference_subjects), desc="Inference"):
            # For each subject, the MRIPathEnv will iterate through its multiple paths
            # We need to ensure we run an episode for each path within the subject.
            # The _reset method in MRIPathEnv handles cycling through paths.
            
            subject_data = inference_env._current_subject_data # Get current subject data
            if subject_data is None: # First iteration, or after previous subject finished
                inference_env._load_next_subject()
                subject_data = inference_env._current_subject_data

            num_paths_in_subject = len(subject_data["paths"])
            
            for path_idx_in_subject in range(num_paths_in_subject):
                try:
                    with torch.autocast(device.type, enabled=config.use_bfloat16):
                        # Reset the environment to load the specific path for this episode
                        # The _reset method in MRIPathEnv will automatically select the next path
                        # based on self.episodes_on_current_subject
                        tensordict = inference_env._reset(must_load_new_subject=False)
                        
                        # Ensure the correct path is loaded for this episode
                        inference_env._current_path_idx = path_idx_in_subject
                        inference_env.current_pos_vox = inference_env.all_start_coords[path_idx_in_subject]
                        inference_env.goal = inference_env.all_end_coords[path_idx_in_subject]
                        
                        # Re-initialize position history and cumulative mask for the new path
                        inference_env.position_history.clear()
                        for _ in range(POSITION_HISTORY_LENGTH - 1):
                            inference_env.position_history.append((0, 0, 0))
                        inference_env.position_history.append(inference_env.current_pos_vox)
                        inference_env.tracking_path_history = [inference_env.current_pos_vox]
                        inference_env.cumulative_path_mask.zero_()
                        
                        # Re-draw initial sphere for the new path
                        current_path_gt_vol = torch.zeros_like(inference_env.seg)
                        current_path_data = inference_env.all_paths[path_idx_in_subject]
                        if current_path_data is not None and current_path_data.shape[0] > 0:
                            valid_indices = (
                                (current_path_data[:, 0] >= 0)
                                & (current_path_data[:, 0] < inference_env.image.shape[0])
                                & (current_path_data[:, 1] >= 0)
                                & (current_path_data[:, 1] < inference_env.image.shape[1])
                                & (current_path_data[:, 2] >= 0)
                                & (current_path_data[:, 2] < inference_env.image.shape[2])
                            )
                            valid_gt_path = current_path_data[valid_indices]
                            if valid_gt_path.shape[0] > 0:
                                current_path_gt_vol[
                                    valid_gt_path[:, 0], valid_gt_path[:, 1], valid_gt_path[:, 2]
                                ] = 1.0
                        inference_env.gt_path_vol = current_path_gt_vol

                        draw_path_sphere_2(
                            inference_env.cumulative_path_mask,
                            inference_env.current_pos_vox,
                            inference_env.dilation,
                            inference_env.gt_path_vol,
                        )

                        # Perform rollout
                        rollout = inference_env.rollout(
                            config.max_episode_steps,
                            actor_module,
                            auto_reset=False,
                            tensordict=tensordict,
                        )

                    # Save the path and mask for this specific path
                    subject_output_dir = output_dir / subject_data["id"]
                    inference_env.save_path(save_dir=subject_output_dir)

                    print(
                        f"  Subject {subject_data['id']}, Path {path_idx_in_subject}: "
                        f"Steps={rollout['action'].shape[1]}, "
                        f"Reward={rollout['next', 'reward'].mean().item():.3f}, "
                        f"Coverage={inference_env._get_final_coverage():.3f}"
                    )

                except Exception as e:
                    print(f"Error during inference rollout for subject {subject_data['id']}, path {path_idx_in_subject}: {e}")
            
            # Manually increment episodes_on_current_subject to trigger next subject load
            # This is a bit of a hack, as the loop above handles paths within a subject.
            # The _reset logic will handle loading the next subject when this counter exceeds num_episodes_per_sample.
            inference_env.episodes_on_current_subject = inference_env.num_episodes_per_sample # Force next subject load

    inference_env.close()
    print("Inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform MRI Motility Inference")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root directory containing MRI path dataset (e.g., 'data/mri_motility')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results_mri_motility",
        help="Directory to save inference results (paths, visualizations)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (e.g., 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=500, # Default from SmallBowelEnv, adjust as needed
        help="Maximum number of steps per episode during inference",
    )
    parser.add_argument(
        "--patch_size_vox",
        type=int,
        nargs=3,
        default=[32, 32, 32], # Default from SmallBowelEnv, adjust as needed
        help="Patch size in voxels (Z Y X)",
    )
    parser.add_argument(
        "--cumulative_path_radius_vox",
        type=int,
        default=3, # Default from SmallBowelEnv, adjust as needed
        help="Radius for cumulative path mask dilation in voxels",
    )
    parser.add_argument(
        "--use_bfloat16",
        action="store_true",
        help="Use bfloat16 for mixed precision inference if available",
    )

    args = parser.parse_args()

    # Create a dummy Config object from argparse arguments
    class InferenceConfig(Config):
        def __init__(self, args):
            super().__init__()
            self.max_episode_steps = args.max_episode_steps
            self.patch_size_vox = tuple(args.patch_size_vox)
            self.cumulative_path_radius_vox = args.cumulative_path_radius_vox
            self.use_bfloat16 = args.use_bfloat16
            # Add other necessary config parameters if they are used in environment/dataset
            # For now, setting some defaults or assuming they are not critical for inference
            self.wall_map_sigmas = (1.0, 2.0, 3.0) # Example default, adjust if needed
            self.shuffle_dataset = False # No need to shuffle for inference
            self.num_workers = 0 # No need for multiple workers for inference
            self.max_step_vox = 3 # Example default, adjust if needed
            self.r_zero_mov = 0.01 # Example default, adjust if needed
            self.r_val1 = 0.01 # Example default, adjust if needed
            self.r_val2 = 0.01 # Example default, adjust if needed
            self.r_final = 1.0 # Example default, adjust if needed
            self.gdt_max_increase_theta = 10.0 # Example default, adjust if needed
            self.gamma = 0.99 # Example default, adjust if needed


    config = InferenceConfig(args)
    device = torch.device(args.device)

    # Load the dataset
    inference_dataset = MRIPathDataset(data_dir=args.data_dir, config=config)

    # Create a dummy environment to get observation_spec for model creation
    # This is a common pattern in TorchRL when loading models without full training setup
    dummy_env = make_mri_path_env(config, inference_dataset, device, num_episodes_per_sample=1, check_env=False)
    
    # Create actor module (policy)
    actor_module, _ = make_dummy_actor_critic(dummy_env.specs)
    actor_module.to(device)

    # Load checkpoint
    print(f"Loading actor from checkpoint: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        actor_module.load_state_dict(checkpoint["policy_module_state_dict"])
        print("Actor module loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        exit(1)
    except KeyError as e:
        print(f"Error loading checkpoint: Missing key {e}. Ensure 'policy_module_state_dict' is present.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading checkpoint: {e}")
        exit(1)

    # Run inference
    inference_loop_mri_path(
        actor_module=actor_module,
        config=config,
        inference_dataset=inference_dataset,
        device=device,
        output_dir=args.output_dir,
    )
