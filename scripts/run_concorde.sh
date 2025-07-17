#!/bin/bash

# Get dataset from the first argument
DATASET="$1"
if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

# Define base paths
BASE_DATA_DIR="$DATASET"
CONCORDE_SCRIPT="notebooks/concorde_tsp.py"

# Find all patient directories
PATIENT_DIRS=$(find "$BASE_DATA_DIR" -maxdepth 1 -mindepth 1 -type d)

# Loop through each patient directory
for PATIENT_DIR in $PATIENT_DIRS; do

    # Expand the patient name from the directory
    PATIENT_DIR=$(realpath "$PATIENT_DIR")
    PATIENT_NAME=$(basename "$PATIENT_DIR")
    INPUT_DIR="$PATIENT_DIR/cache"

    if [[ ! -d "$PATIENT_DIR" ]]; then
        echo "Skipping non-directory: $PATIENT_DIR"
        continue
    fi
    if [[ ! -f "$INPUT_DIR/rag2.json" ]]; then
        echo "rag2.json not found in $INPUT_DIR, skipping."
        continue
    fi
    if [[ ! -f "$INPUT_DIR/wall_map.nii.gz" ]]; then
        echo "wall_map.nii.gz not found in $INPUT_DIR, skipping."
        continue
    fi

    echo "Processing patient: $PATIENT_NAME"

    # Start can be found in the metrics.json in the parent directory
    METRICS_FILE="$PATIENT_DIR/metrics.json"
    if [ ! -f "$METRICS_FILE" ]; then
        echo "Metrics file not found for patient: $PATIENT_NAME"
        continue
    fi  
    START=$(jq -r '.metadata.start' "$METRICS_FILE")
    END=$(jq -r '.metadata.end' "$METRICS_FILE")

    # Execute concorde_tsp.py
    python "$CONCORDE_SCRIPT" --graph "$INPUT_DIR/rag2.json" \
        --reference_volume "$INPUT_DIR/wall_map.nii.gz" \
        --start "$START" \
        --end "$END" \
        --rounding_factor 1e6 \
        --comment "Concorde TSP for $PATIENT_NAME"

    echo "Finished processing patient: $PATIENT_NAME"
    echo "----------------------------------------"
done

echo "All patients processed."
