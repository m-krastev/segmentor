"""
Main execution script for Navigator's small bowel tracking.
"""

import os
import traceback
import torch
from torch.utils.data import Subset
import numpy as np
from tensordict import TensorDict

# Try importing wandb, handle if not available
try:
    import wandb
except ImportError:
    print("Warning: wandb not found. Wandb tracking will be disabled.")
    wandb = None

from .config import parse_args, Config
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
    print(f"Train indices\t({train_size:0>2}/{len(dataset)}): {train_indices}, subjects: {[dataset.subjects[idx]['id'] for idx in train_indices]}")
    print(f"Val indices \t({len(dataset)-train_size:0>2}/{len(dataset)}): {val_indices}, subjects: {[dataset.subjects[idx]['id'] for idx in val_indices]}")

    # --- Models ---
    in_act, in_crit = 4, 4
    policy_module, value_module = create_ppo_modules(config, config.device, qnets=config.td3, in_channels_actor=in_act, in_channels_critic=in_crit)

    # Init the lazy modules
    with torch.no_grad():
        dummy_input = TensorDict(
            {
                "actor": TensorDict( # This is the main observation for both actor and critic
                    {
                        "patches": torch.zeros(1, in_act, *config.patch_size_vox, device=config.device),
                        "agent_orientation": torch.zeros(1, 4, device=config.device),
                        "goal_direction_quat": torch.zeros(1, 4, device=config.device),
                    },
                    batch_size=[1],
                    device=config.device
                ),
                "action": torch.zeros(1, 3, device=config.device)
            },
            batch_size=[1],
            device=config.device
        )
        policy_module(dummy_input)
        value_module(dummy_input)
        print(f"Policy: {policy_module}")
    
    policy_module.compile(fullgraph=True, dynamic=False)
    value_module.compile()

    # Watch the model parameters
    # if config.track_wandb and wandb is not None:
    #     wandb.watch(policy_module, log="all")
    #     wandb.watch(value_module, log="all")

    torch.serialization.add_safe_globals([Config])
    # --- Start Training ---
    if config.eval_only:
        print("Evaluation mode: skipping training.")
        assert config.load_from_checkpoint, "Checkpoint must be provided for evaluation."

        data = torch.load(config.load_from_checkpoint)
        policy_module.load_state_dict(data["policy_module_state_dict"])
        validation_loop_torchrl(policy_module, config, val_set, config.device)
    elif config.train_gym_env:
        print("Training dummy Gym environment.")
        from .train_gym import train_gym_environment
        train_gym_environment(config)
    else:
        if config.load_from_checkpoint:
            print(f"Loading data from {config.load_from_checkpoint}")
            data = torch.load(config.load_from_checkpoint, weights_only=False)
            policy_module.load_state_dict(data["policy_module_state_dict"])
            value_module.load_state_dict(data["value_module_state_dict"])

        train_torchrl(policy_module, value_module, config, train_set, val_set, qnets=config.td3)
    # try:
    #     train_torchrl(config, dataset)
    # except Exception as e:
    #     print(f"\nAn error occurred during training: {e}")
    #     traceback.print_exc()  # Print detailed traceback
    # finally:
    #     run.finish()



if __name__ == "__main__":
    main()
