"""
Main execution script for Navigator's small bowel tracking.
"""

import os
import traceback
import torch


# Try importing wandb, handle if not available
try:
    import wandb
except ImportError:
    print("Warning: wandb not found. Wandb tracking will be disabled.")
    wandb = None

from .config import parse_args
from .dataset import SmallBowelDataset
from .train import train_torchrl


def main():
    """Main entry point for the Navigator system."""

    # Parse command line arguments
    config = parse_args()
    print("Parsed configuration:")

    # Convert dataclass to dict for printing/wandb config
    config_dict = vars(config)
    print(config_dict)

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
        preload=True,
        transform=None,
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- Start Training ---
    train_torchrl(config, dataset)
    # try:
    #     train_torchrl(config, dataset)
    # except Exception as e:
    #     print(f"\nAn error occurred during training: {e}")
    #     traceback.print_exc()  # Print detailed traceback
    # finally:
    #     # --- Finish Wandb Run ---
    #     # if config.track_wandb and run and wandb is not None:
    #     #     # Save final model before finishing
    #     #     final_save_path = config.save_path.replace(".pth", "_final.pth")
    #     #     save_dict = {
    #     #         "actor_state_dict": actor.state_dict(),
    #     #         "critic_state_dict": critic.state_dict(),
    #     #         "config": config,
    #     #     }
    #     #     torch.save(save_dict, final_save_path)
    #     #     print(f"Final model saved to {final_save_path}")
    #     #
    #     try:
    #         run.finish()
    #         print("Wandb run finished.")
    #     except Exception as e:
    #         print(f"Error finishing wandb run: {e}")


if __name__ == "__main__":
    main()
