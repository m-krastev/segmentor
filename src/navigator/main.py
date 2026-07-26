"""
Main execution script for Navigator's small bowel tracking.
"""

import os
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
from .dataset import NNUNetActualDataset, SmallBowelDataset
from .train import train_torchrl, validation_loop_torchrl
from .pretrain import pretrain_behavior_cloning
from .models import create_ppo_modules
from .utils import seed_everything


def main():
    """Main entry point for the Navigator system."""

    def read_case_ids(path: str) -> list[str]:
        with open(path) as case_file:
            case_ids = [
                line.strip()
                for line in case_file
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if not case_ids:
            raise ValueError(f"No case IDs found in {path}")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"Duplicate case IDs found in {path}")
        return case_ids

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

    if config.nnunet_raw_dir:
        case_ids = None
        explicit_train_ids = explicit_val_ids = None
        has_explicit_split = bool(
            config.nnunet_train_case_ids_file or config.nnunet_val_case_ids_file
        )
        if has_explicit_split:
            if not (
                config.nnunet_train_case_ids_file
                and config.nnunet_val_case_ids_file
            ):
                raise ValueError(
                    "Both --nnunet-train-case-ids-file and "
                    "--nnunet-val-case-ids-file are required."
                )
            if config.nnunet_case_ids_file:
                raise ValueError(
                    "--nnunet-case-ids-file cannot be combined with explicit "
                    "train/validation manifests."
                )
            explicit_train_ids = read_case_ids(config.nnunet_train_case_ids_file)
            explicit_val_ids = read_case_ids(config.nnunet_val_case_ids_file)
            overlap = set(explicit_train_ids) & set(explicit_val_ids)
            if overlap:
                raise ValueError(
                    f"Train/validation manifests overlap: {sorted(overlap)}"
                )
            case_ids = explicit_train_ids + explicit_val_ids
            print(
                "Using immutable nnU-Net split: "
                f"{len(explicit_train_ids)} train / {len(explicit_val_ids)} validation"
            )
        if config.nnunet_case_ids_file:
            case_ids = read_case_ids(config.nnunet_case_ids_file)
            print(
                f"Restricting nnU-Net dataset to {len(case_ids)} cases from "
                f"{config.nnunet_case_ids_file}"
            )
        dataset = NNUNetActualDataset(
            nnunet_raw=config.nnunet_raw_dir,
            cache_dir=config.nnunet_cache_dir,
            config=config,
            case_ids=case_ids,
        )
    else:
        dataset = SmallBowelDataset(
            data_dir=config.data_dir,
            config=config,
        )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- Setup ---
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # --- Dataset Splitting and Iterators ---
    if config.nnunet_raw_dir and explicit_train_ids is not None:
        index_by_id = {
            subject["id"]: index for index, subject in enumerate(dataset.subjects)
        }
        train_indices = np.asarray(
            [index_by_id[case_id] for case_id in explicit_train_ids],
            dtype=int,
        )
        val_indices = np.asarray(
            [index_by_id[case_id] for case_id in explicit_val_ids],
            dtype=int,
        )
        train_size = len(train_indices)
    else:
        train_size = int(len(dataset) * config.train_val_split)
        indices = np.arange(len(dataset))
        if config.shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    print(
        f"Train indices\t({train_size:0>2}/{len(dataset)}): {train_indices}, subjects: {[dataset.subjects[idx]['id'] for idx in train_indices]}"
    )
    print(
        f"Val indices \t({len(dataset) - train_size:0>2}/{len(dataset)}): {val_indices}, subjects: {[dataset.subjects[idx]['id'] for idx in val_indices]}"
    )

    # --- Models ---
    in_act = in_crit = config.observation_channels
    policy_module, value_module = create_ppo_modules(
        config,
        config.device,
        qnets=config.td3,
        in_channels_actor=in_act,
        in_channels_critic=in_crit,
    )

    # Init the lazy modules
    with torch.no_grad():
        dummy_input = TensorDict(
            {
                "actor": torch.zeros(1, in_act, *config.patch_size_vox, device=config.device),
                "context": torch.zeros(1, config.context_features, device=config.device),
                "action": torch.zeros(1, 3, device=config.device),
            },
        )
        policy_module(dummy_input)
        value_module(dummy_input)
        print(f"Policy: {policy_module}")

    # Keep the TensorDict wrappers eager. Compiling these top-level modules
    # currently crashes inside TorchDynamo's TensorDict dispatch handling
    # (PyTorch 2.10 / TensorDict 0.11); the underlying convolutions still use
    # optimized CUDA/cuDNN kernels.

    # Watch the model parameters
    # if config.track_wandb and wandb is not None:
    #     wandb.watch(policy_module, log="all")
    #     wandb.watch(value_module, log="all")

    torch.serialization.add_safe_globals([Config])
    # --- Start Training ---
    if config.eval_only:
        print("Evaluation mode: skipping training.")
        if not config.load_from_checkpoint:
            raise ValueError("Checkpoint must be provided for evaluation.")

        data = torch.load(config.load_from_checkpoint, weights_only=False)
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

        if config.behavior_cloning_epochs:
            pretrain_behavior_cloning(policy_module, config, train_set, config.device)
            pretraining_checkpoint = os.path.join(
                config.checkpoint_dir,
                "behavior_cloning_model.pth",
            )
            torch.save(
                {
                    "policy_module_state_dict": policy_module.state_dict(),
                    "value_module_state_dict": value_module.state_dict(),
                    "config": vars(config),
                },
                pretraining_checkpoint,
            )
            print(f"Behavior-cloning checkpoint saved to {pretraining_checkpoint}")
            validation_loop_torchrl(policy_module, config, val_set, config.device)
            policy_module.train()

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
