#!/usr/bin/env bash

UNET_CONFIG="3d_fullres"
FOLD=1
DEVICE=cuda

nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c $UNET_CONFIG

# Training
nnUNetv2_train $DATASET_ID $UNET_CONFIG $FOLD -device $DEVICE --npz

# # Tune hyperparameters
# nnUNetv2_find_best_configuration $DATASET_ID -c CONFIGURATIONS 