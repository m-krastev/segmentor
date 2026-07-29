# Small Bowel Centerline Extraction and Estimation Using Unsupervised Methods

While significant progress has been made in medical image segmentation and deep learning, accurately delineating the small bowel remains a complex challenge due to its intricate folding within the abdomen. This demanding and time-consuming task, known as small bowel centerline extraction, can take expert radiologists several hours to a full day to annotate for a single patient. Unlike simple segmentation, which often fails to capture the organ's proper topology, the centerline specifically maps the small bowel's winding, continuous path --- crucial for understanding its structure and connectivity, and vital for medical analysis. The inherent complexity of this task makes large-scale manual annotation efforts incredibly costly, underscoring the need for robust automated solutions that can overcome the limitations of conventional segmentation by focusing on this critical centerline.

We approach this task by reviewing the available literature, and proposing two methods for unsupervised small bowel centerline extraction  -- VoxGraph and VoxTrack, utilizing two distinctly different methodologies -- graph combinatorial optimization and Reinforcement Learning (RL). The benefit of these approaches lies in their ability to find solutions without relying on expert annotations. We discover these approaches can track segments of the small bowel centerline well, and can segment up to 650 mm of the full small bowel path correctly.


## Repository Structure

```bash
.
├── archive              # Legacy code for multi-step segmentation (e.g., GraphCentre, MAE, UNet, ViT)
├── data                 # Data processing utilities and storage for raw data
├── notebooks            # Jupyter notebooks for exploration, analysis, and graph-based approaches
│   ├── graph_approach.py # Script for the graph-based pathfinding approach
│   └── ...              # Other analysis and utility notebooks (e.g., compute_metrics.ipynb)
├── scripts              # Training and utility scripts (e.g., SLURM submission scripts)
└── src                  # Main source code
    ├── navigator/       # Reinforcement Learning (RL) approach for navigation
    │   ├── config.py    # Configuration for RL models and environment
    │   ├── dataset.py   # Dataset handling for the RL environment
    │   ├── environment.py # Definition of the RL environment
    │   ├── models/      # Actor-Critic models and other neural networks for RL
    │   ├── main.py      # Entry point for RL training and evaluation
    │   └── train.py     # Main training script for the RL agent
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
```

If you plan to use `nnUNet` for segmentation tasks, install it as a submodule:

```bash
git submodule update --init --recursive
cd nnUNet && pip install -e .
```

Finally, set up environment variables by creating a `.env` file in the root directory of the project. The `.env` file may contain variables such as `HF_TOKEN` and `WANDB_API_KEY` for Hugging Face and Weights & Biases, respectively. If using `nnUNet`, you must define the `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_base` variables for nnUNet data processing.

## Usage

### Graph-Based Approach

To run the graph-based pathfinding approach, execute the relevant script in the `notebooks` directory:

```bash
python notebooks/graph_approach.py
```

### Reinforcement Learning (RL) Approach (Navigator)

To train the RL agent for navigation:

```bash
uv run python -m navigator --data-dir <your_data> --patch-size-mm 32 --voxel-size-mm 1.5 --amp
```

For the preregistered real-data G1 experiment, run the versioned systemd
launchers in order:

```bash
scripts/run_navigator_g1_preflight.sh
systemctl --user status navigator-preflight-v1

scripts/run_navigator_g1_oracle.sh
systemctl --user status navigator-g1-oracle-validation-v1

scripts/run_navigator_g1_diagnostic.sh
journalctl --user -u navigator-g1-diagnostic-250k -f
```

These commands freeze disjoint train/validation/test manifests, verify the
validation oracle, and then run the 250k-step CUDA diagnostic with TensorBoard.
The test manifest is not consumed by either training or validation.

The bounded BOMOPI reproduction of the repaired historical `068dc4d`
configuration is launched with:

```bash
NAVIGATOR_TRAIN_SCRIPT=scripts/run_navigator_bomopi_068_repaired.sh \
  scripts/navigator_systemd.sh start navigator-bomopi-gru-068-repaired-256k-v1
```

It keeps the current GRU PPO implementation and fixed validation protocol, but
uses the registered 60-mm patch, 9-mm action, 6-mm path radius, CT/wall/thin-path
policy state, physically scaled GDT threshold, and strict endpoint-plus-Dice
success. GT segmentation and GDT supervise its reward only; they are never
policy inputs, so this is deployable without labels but not annotation-free
training. PPO minibatches are 16 because the paper's 32-sample minibatch does
not fit a 40-cubed 3-D encoder on the 15.5-GiB test GPU.

#### Data Preprocessing for nnUNet

To prepare data for nnUNet format:

```bash
python data/nnunet_prep.py --data_dir /path/to/data --output_dir /path/to/output
```

#### Training nnUNet

```bash
# Plan and preprocess
nnUNetv2_plan_and_preprocess -d 42 --verify_dataset_integrity -c 3d_fullres

# Train
nnUNetv2_train 42 3d_fullres 1 -device cuda --npz
```

### Cluster Training (SLURM)

For training on a SLURM cluster, you can configure the `scripts/slurm_base.sh` script.

## Requirements

- Python ≥ 3.10
- PyTorch
- CUDA-capable GPU (for training)
- For cluster usage: SLURM workload manager
