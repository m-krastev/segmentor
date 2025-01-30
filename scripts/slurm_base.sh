#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

#SBATCH --job-name=segmentor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --output=/scratch-shared/%u/logs/%x-%A.out
date

# Definitions
export HF_HOME="/scratch-shared/$USER"
export HF_TOKEN=
export GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

# make sure the correct modules are used and that the virtual environment is active
PROJECT_ROOT=$HOME/segmentor
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
nnUNetv2_train $DATASET_ID $UNET_CONFIG $FOLD -device $DEVICE --npz -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDiceFocal
