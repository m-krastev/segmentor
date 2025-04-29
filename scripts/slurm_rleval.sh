#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1

#SBATCH --job-name=segmentor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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
module load OpenMPI/5.0.3-GCC-13.3.0

python -m navigator --data-dir data/data --eval-only --load-from-checkpoint "$PROJECT_ROOT/checkpoints/final_model_torchrl.pth"
