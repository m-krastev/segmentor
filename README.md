# Small bowel centreline extraction without centreline annotations

Code for the MSc thesis *Small Bowel Centerline Extraction and Estimation
Using Unsupervised Methods* (Matey Krastev, University of Amsterdam, 2025;
supervised by Dr Yunchao Yin and Prof. Martin R. Oswald, with Amsterdam UMC).
[Thesis](https://scripties.uba.uva.nl/search?id=record_56865)

The small bowel is usually segmented, but clinical use needs its centreline:
one ordered path from the duodenum to the ileocaecal junction. Expert path
annotation takes hours per patient, so this work compares two methods that
need only a CT scan and a small bowel segmentation:

- VoxGraph (`notebooks/graph_approach.py`): a reproduction and extension of
  the graph-theoretic tracker of Shin et al. A Meijering ridge filter marks
  bowel walls, SLIC supervoxels form a weighted graph, must-pass nodes are
  sampled inside the segmentation, and the path is found as a travelling
  salesman tour (simulated annealing, ACHCI, or exact solution with Concorde).
- VoxTrack (`src/navigator/`): a reinforcement-learning tracker trained with
  PPO in a 3D environment built for this work. The agent moves through local
  image patches and is rewarded for advancing along the bowel, measured by the
  geodesic distance transform of the segmentation.

On 21 contrast-enhanced CT scans, VoxGraph with the exact TSP solver tracks
the path correctly for 644.6 ± 135.8 mm on average (92.8% coverage), on par
with the Shin et al. baseline (625.2 ± 143.3 mm, 93.0%), and 574.2 mm when
it runs on predicted instead of manual segmentations (thesis, Table 2).

## Repository layout

```text
notebooks/
  graph_approach.py       VoxGraph pipeline (CLI)
  concorde_tsp.py         exact TSP with Concorde
  achci_tsp.py            ACHCI heuristic
  bayesian_optimization.py  hyperparameter search for VoxGraph
  compute_metrics.ipynb   evaluation tables and plots
src/navigator/            VoxTrack: environment, dataset, actor-critic models, PPO training
data/
  generate_phantom.py     synthetic bowel phantoms with known paths
  nnunet_prep.py          conversion to nnU-Net format for the segmentation model
slic/                     SLIC supervoxel package (uv workspace member)
scripts/                  Concorde/ACHCI batch runs, SLURM jobs, nnU-Net runs
archive/                  earlier segmentation experiments (U-Net, ViT, MAE)
```

## Installation

Python 3.12 or later.

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
# optional GPU extras: uv pip install -e ".[cuda]"
```

For the segmentation model, initialise the nnU-Net submodule:

```bash
git submodule update --init nnUNet
cd nnUNet && uv pip install -e .
```

Weights & Biases logging reads `WANDB_API_KEY` from the environment or a
`.env` file; nnU-Net needs `nnUNet_raw`, `nnUNet_preprocessed` and
`nnUNet_results`.

## Usage

VoxGraph on one scan:

```bash
python notebooks/graph_approach.py \
    --filename_ct scan.nii.gz \
    --filename_gt small_bowel_mask.nii.gz \
    --output out/ [--config params.json]
```

The exact solution is computed from the cached graph with
`scripts/run_concorde.sh <dataset_dir>`.

VoxTrack training (all fields of `src/navigator/config.py` are command-line
options):

```bash
python -m navigator --data-dir <data> --patch-size-mm 32 --voxel-size-mm 1.5 --amp
```

Synthetic test volumes with known paths (100 by default):

```bash
bash data/generate_phantoms.sh
```

nnU-Net segmentation model:

```bash
python data/nnunet_prep.py --data_dir /path/to/data --output_dir /path/to/output
nnUNetv2_plan_and_preprocess -d 42 --verify_dataset_integrity -c 3d_fullres
nnUNetv2_train 42 3d_fullres 1 -device cuda --npz
```

## Data

The CT scans and their segmentations are in-house data and are not included.
The phantom generator and the public TotalSegmentator dataset
(`data/filter_totalsegmentator.py`) can be used to run the pipelines without
them.
