#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

#SBATCH --job-name=segmentor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=23:59:00
#SBATCH --output=/scratch-shared/%u/logs/%x-%A.out
date

# Definitions
export HF_HOME="/scratch-shared/$USER"
export HF_TOKEN=
export GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

# make sure the correct modules are used and that the virtual environment is active
PROJECT_ROOT=/home/mkrastev1/segmentor
source $PROJECT_ROOT/scripts/slurm_setup.sh
setup $PROJECT_ROOT
cd $PROJECT_ROOT

# rest of the script
UNET_CONFIG="3d_fullres"
FOLD=1
DEVICE=cuda
DATASET_ID=42

# nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c $UNET_CONFIG -np 16 -pl nnUNetPlannerResEncL

# Training
# nnUNetv2_train $DATASET_ID $UNET_CONFIG $FOLD -device $DEVICE --npz -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDiceFocal

# export RAYON_NUM_THREADS=16
# cd notebooks
# python graph_approach_fail.py --sigmas 1 3 --thetav 3 --thetad 6 --supervoxel_size 216 --delta 5000 --use_rustworkx --precompute --filename_ct ../data/bowelseg/s0006/ct.nii.gz --filename_gt ../data/bowelseg/s0006/segmentations/small_bowel.nii.gz --start_volume ../data/bowelseg/s0006/segmentations/duodenum.nii.gz --end_volume ../data/bowelseg/s0006/segmentations/colon.nii.gz

# python bayesian_optimization.py
# python test.py


module load OpenMPI/5.0.3-GCC-13.3.0
python -m navigator --data-dir data/data

# python grl_pathtracking.py --nifti-path ./ct.nii.gz --seg-path ./small_bowel.nii.gz --duodenum-seg-path ./duodenum.nii.gz --colon-seg-path ./colon.nii.gz