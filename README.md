# Segmentor

A deep learning project for medical image segmentation, focusing on bowel segmentation using various architectures including UNet, ViT, and attention-based models.

## Repository Structure

```bash
.
├── archive              # Legacy code and previous implementations
│   ├── BowelGraph/      # Graph-based bowel analysis
│   ├── MAE/             # Masked Autoencoder implementation
│   ├── UNet/            # UNet model implementations
│   ├── ViT/             # Vision Transformer models
│   └── utils/           # Shared utility functions
├── data                # Data processing utilities plus where raw data is stored
├── notebooks           # Jupyter notebooks for exploration and analysis
├── scripts             # Training and utility scripts
└── src                # Main source code
    ├── configs/        # Model configurations
    ├── model.py        # Core model implementations
    ├── ...
    └── utils/          # Utility functions
```

## Installation

First, create a virtual environment:

```bash
python -m venv .venv
# Recommend using uv instead: curl -LsSf https://astral.sh/uv/install.sh | sh
# uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
# uv pip install -e .

# Install nnUNet
git submodule update --init --recursive
cd nnUNet && pip install -e .
```

Finally, set up environment variables by creating a .env file in the root directory of the project. The `.env` file may contain variables such as `HF_TOKEN` and `WANDB_API_KEY` for Hugging Face and Weights & Biases, respectively. Furthermore, you must define the `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_base` variables for nnUNet data processing.

## Usage

### Local Training

For local training, you can use the training script directly:

```bash
python src/train.py --config configs/your_config.yaml
```

### Cluster Training (SLURM)

For training on a SLURM cluster, you can configure the `scripts/slurm_base.sh` script and submit it.

### Data Preprocessing

To prepare data for nnUNet format:

```bash
python data/nnunet_prep.py --data_dir /path/to/data --output_dir /path/to/output
```

### Training nnUNet

```bash
# Plan and preprocess
nnUNetv2_plan_and_preprocess -d 42 --verify_dataset_integrity -c 3d_fullres

# Train
nnUNetv2_train 42 3d_fullres 1 -device cuda --npz
```

## Requirements

- Python ≥ 3.10
- PyTorch
- CUDA-capable GPU (for training)
- For cluster usage: SLURM workload manager
