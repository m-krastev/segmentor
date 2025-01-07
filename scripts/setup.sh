#!/bin/env bash

# Check if DATA_DIR is set
if [ -z "$DATA_DIR" ]; then
    echo "DATA_DIR is not set. Please set it to the directory where the data should be stored."
    exit 1
fi

if [ ! -d $DATA_DIR ]; then
    echo "Creating data directory at $DATA_DIR"
    echo "Warning: The directory is empty and should be populated with the dataset."
fi

# Unzip the dataset (assuming it is in the same directory as the script)
if [ ! -f $DATA_DIR/TotalsegmentatorMRI_dataset_v100.zip ]; then
    echo "Please put the dataset inside the data directory."
    echo "You can download it from: https://zenodo.org/records/11367005"
    # wget -O $DATA_DIR/TotalsegmentatorMRI_dataset_v100.zip https://www.dropbox.com/s/7z1q6zv5v5v6j5v/TotalsegmentatorMRI_dataset_v100.zip?dl=0
else
    unzip $DATA_DIR/TotalsegmentatorMRI_dataset_v100.zip -d $DATA_DIR/totalsegmentatormri
fi
