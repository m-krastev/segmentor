"""
Main execution script for Navigator's small bowel tracking.
"""

import os
import traceback
import torch
from torch.utils.data import Subset
import numpy as np


# Try importing wandb, handle if not available
try:
    import wandb
except ImportError:
    print("Warning: wandb not found. Wandb tracking will be disabled.")
    wandb = None

from .config import parse_args
from .dataset import SmallBowelDataset
from .train import train_torchrl, validation_loop_torchrl
from .models import create_ppo_modules
from .utils import seed_everything

def main():
    """Main entry point for the Navigator system."""

    # Parse command line arguments
    config = parse_args()
    print("Parsed configuration:")

    # Convert dataclass to dict for printing/wandb config
    config_dict = vars(config)
    print(config_dict)
    
    seed_everything(config.seed)

    # Create checkpoint directory if needed
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # --- Initialize Wandb ---
    run = None
    if config.track_wandb and wandb is not None:
        try:
            run = wandb.init(
                project=config.wandb_project_name,
                entity=config.wandb_entity,  # Optional: Your wandb user/team
                name=config.wandb_run_name,  # Optional: Defaults to auto-generated name
                sync_tensorboard=False,  # We are using wandb logging directly
                config=config_dict,  # Log hyperparameters
                monitor_gym=False,  # We are not using gym environment directly
                save_code=True,  # Save main script to wandb
            )
            print(f"Wandb run initialized: {run.url}")
        except Exception as e:
            print(f"Error initializing wandb: {e}. Wandb tracking disabled.")
            config.track_wandb = False  # Disable tracking if init fails

    dataset = SmallBowelDataset(
        data_dir=config.data_dir,
        config=config,
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- Setup ---
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # --- Dataset Splitting and Iterators ---
    train_size = int(len(dataset) * config.train_val_split)
    indices = np.arange(len(dataset))
    if config.shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    print(f"Val indices: {val_indices}, subjects: {[obj['id'] for obj in dataset.subjects[val_indices]]}")

    print(f"Training: {len(train_set)} subjects, Validation: {len(val_set)} subjects.")


    # --- Models ---
    policy_module, value_module = create_ppo_modules(config, config.device)

    # --- Start Training ---
    if config.eval_only:
        print("Evaluation mode: skipping training.")
        assert config.load_from_checkpoint, "Checkpoint must be provided for evaluation."

        data = torch.load(config.load_from_checkpoint)
        policy_module.load_state_dict(data["policy_state_dict"])
        validation_loop_torchrl(policy_module, config, val_set, config.device)
    else:
        train_torchrl(policy_module, value_module, config, train_set, val_set)
    # try:
    #     train_torchrl(config, dataset)
    # except Exception as e:
    #     print(f"\nAn error occurred during training: {e}")
    #     traceback.print_exc()  # Print detailed traceback
    # finally:
    #     run.finish()



if __name__ == "__main__":
    main()
