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


# Depending on the dataset specified in the args, download and unzip. If the dataset is not specified, do not do anything.
if [ -z "$1" ]; then
    echo "No dataset specified. Please specify the dataset to download."
    exit 1
fi

if [ $1 == "totalsegmentatormri" ]; then
    DATASET_LINK="https://zenodo.org/record/11367005/files/TotalsegmentatorMRI_dataset_v100.zip"
elif [ $1 == "totalsegmentatorct" ]; then
    DATASET_LINK="https://www.dropbox.com/scl/fi/oq0fsz8oauory204g8o6f/Totalsegmentator_dataset_v201.zip?rlkey=afnl2ixhqca2ukkf1v9p6jz7p&e=1&st=hmp1p0x3&dl=0"
else
    echo "Invalid dataset specified. Please specify either 'totalsegmentatormri' or 'totalsegmentator'."
    exit 1
fi

# Check if the dataset is already downloaded
if [ -d $DATA_DIR/$DATASET_NAME ]; then
    echo "Dataset already downloaded. Skipping download."
else
    echo "Downloading dataset from $DATASET_LINK"
    wget $DATASET_LINK -O $DATA_DIR/$1.zip
    echo "Unzipping dataset"
    unzip $DATA_DIR/$1.zip -d $DATA_DIR
    rm $DATA_DIR/$1.zip
fi