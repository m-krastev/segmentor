#!/bin/bash

setup() {
    if [ -z $1 ]; then
        cwd=.
    else
        cwd=$1
    fi

    module purge
    module load 2024
    module load CUDA/12.6.0
    module load OpenMPI/5.0.3-NVHPC-24.9-CUDA-12.6.0
    module load Python/3.12.3-GCCcore-13.3.0

	source "$cwd/.venv/bin/activate"

	echo "Environment setup complete."
	echo "Python version: $(python --version)"
    echo "CUDA version: $(nvidia-smi --version)"

    # Source the .env file
    if [ -f "$cwd/.env" ]; then
        while read -r line; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

            # Evaluate the line with variable substitutions
            eval "export $line"
        done < "$cwd/.env"
    fi

    if [ ! -z $HF_TOKEN ]; then
        huggingface-cli login --token $HF_TOKEN
    fi

    if [ ! -z $WANDB_API_KEY ]; then
        wandb login $WANDB_API_KEY
    fi
}
