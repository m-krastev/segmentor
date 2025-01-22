#!/bin/bash

setup() {
    if [ -z $1 ]; then
        $1=.
    fi

	module purge
	
	module load 2024
    module load CUDA/12.6.0

	source "$1/.venv/bin/activate"
	export HF_HOME="/scratch-shared/$USER"

	echo "Environment setup complete."
	echo "Python version: $(python --version)"
    echo "CUDA version: $(nvidia-smi --version)"

    # Source the .env file
    if [ -f "$1/.env" ]; then
    while IFS='=' read -r key value; do
        # Remove leading/trailing whitespace from key and value
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        # Check if key is not empty and doesn't start with '#' (comment)
        if [[ -n "$key" && ! "$key" =~ ^# ]]; then
        export "$key"="$value" # Export the variable
        fi
    done < "$1/.env"
    fi

    if [ ! -z $HF_TOKEN ]; then
	    huggingface-cli login --token $HF_TOKEN
    fi

    if [! -z $WANDB_API_KEY]; then
        wandb login $WANDB_API_KEY
    fi
}
