#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

#SBATCH --job-name=segmentor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
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

#ignore: s0009, s0016, s0022, s0039, s0045, s0046, s0048, s0050, s0057
    
# rest of the script
UNET_CONFIG="3d_fullres"
FOLD=1
DEVICE=cuda
DATASET_ID=18

# nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c $UNET_CONFIG -np 16 -pl nnUNetPlannerResEncL

# Training
nnUNetv2_train $DATASET_ID $UNET_CONFIG $FOLD -device $DEVICE --npz -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDiceFocal

# export RAYON_NUM_THREADS=16
# cd notebooks
# python graph_approach_fail.py --sigmas 1 3 --thetav 3 --thetad 6 --supervoxel_size 216 --delta 5000 --use_rustworkx --precompute --filename_ct ../data/bowelseg/s0006/ct.nii.gz --filename_gt ../data/bowelseg/s0006/segmentations/small_bowel.nii.gz --start_volume ../data/bowelseg/s0006/segmentations/duodenum.nii.gz --end_volume ../data/bowelseg/s0006/segmentations/colon.nii.gz

# python bayesian_optimization.py
# python test.py


# python -m navigator --data-dir data/data # checkpoints/checkpoint_torchrl_7372800.pth # --load-from-checkpoint checkpoints/checkpoint_torchrl_2252800.pth

# cd notebooks
# python grl_pathtracking.py --nifti-path ./ct.nii.gz --seg-path ./small_bowel.nii.gz --duodenum-seg-path ./duodenum.nii.gz --colon-seg-path ./colon.nii.gz
