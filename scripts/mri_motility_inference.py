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
from navigator.config import Config, parse_args
from navigator.dataset import MRIPathDataset
from navigator.models import create_ppo_modules
from navigator.environment import make_mri_path_env
from navigator.utils import draw_path_sphere_2

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
    inference_env = make_mri_path_env(config, inference_dataset, device, num_episodes_per_sample=64, check_env=False)

    num_inference_subjects = len(inference_dataset)

    print(f"Starting inference for {num_inference_subjects} subjects...")

    with (
        torch.no_grad(),
        set_exploration_type(ExplorationType.MODE),
    ):
        for i in tqdm(range(num_inference_subjects), desc="Inference"):
            # Manually handle the reset... 
            inference_env._load_next_subject()
            subject_data = inference_env._current_subject_data
            num_paths_in_subject = len(subject_data["paths"])
            for path_idx_in_subject in range(num_paths_in_subject):
                try:
                    tensordict = inference_env._reset(must_load_new_subject=False)

                    # Ensure the correct path is loaded for this episode
                    inference_env._current_path_idx = path_idx_in_subject
                    inference_env.current_pos_vox = tuple(inference_env.all_start_coords[path_idx_in_subject])
                    inference_env.goal = tuple(inference_env.all_end_coords[path_idx_in_subject])
                    inference_env._start = inference_env.current_pos_vox
                    
                    # Re-initialize position history and cumulative mask for the new path
                    inference_env.tracking_path_history = [inference_env.current_pos_vox]
                    inference_env.cumulative_path_mask.zero_()
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
                    raise e
                    print(f"Error during inference rollout for subject {subject_data['id']}, path {path_idx_in_subject}: {e}")
            
            # Manually increment episodes_on_current_subject to trigger next subject load
            # This is a bit of a hack, as the loop above handles paths within a subject.
            # The _reset logic will handle loading the next subject when this counter exceeds num_episodes_per_sample.
            inference_env.episodes_on_current_subject = inference_env.num_episodes_per_sample # Force next subject load

    inference_env.close()
    print("Inference finished.")


if __name__ == "__main__":
    config = parse_args()
    device = torch.device(config.device)

    # Load the dataset
    inference_dataset = MRIPathDataset(data_dir=config.data_dir, config=config)

    # Create a dummy environment to get observation_spec for model creation
    # This is a common pattern in TorchRL when loading models without full training setup
    dummy_env = make_mri_path_env(config, inference_dataset, device, num_episodes_per_sample=1, check_env=False)
    
    # Create actor module (policy)
    actor_module, _ = create_ppo_modules(config, device, False, 4, 4)
    actor_module.to(device)

    # Load checkpoint
    print(f"Loading actor from checkpoint: {config.load_from_checkpoint}")
    try:
        checkpoint = torch.load(config.load_from_checkpoint, map_location=device)
        actor_module.load_state_dict(checkpoint["policy_module_state_dict"])
        print("Actor module loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {config.load_from_checkpoint}")
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
        output_dir="results/",
    )
