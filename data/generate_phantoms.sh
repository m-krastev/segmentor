#!/bin/bash

# filepath: generate_phantoms.sh

DIR=./phantoms

# Create a base output directory if it doesn't exist
mkdir -p $DIR

# Loop 100 times to generate 100 scans
for i in $(seq 1 100)
do
  echo "Generating phantom $i of 100..."

  # Generate random volume shapes (between 70 and 140 for each dimension)
  shape_x=$(( RANDOM % (140 - 70 + 1) + 70 ))
  shape_y=$(( RANDOM % (140 - 70 + 1) + 70 ))
  shape_z=$(( RANDOM % (140 - 70 + 1) + 70 ))
  volume_shape="${shape_x},${shape_y},${shape_z}"

  # Generate random number of control points (between 4 and 10)
  n_control_points=$(( RANDOM % (10 - 4 + 1) + 4 ))

  # Generate a random seed
  seed=$RANDOM

  # Zero-pad the patient ID
  patient_id_padded=$(printf "%03d" "$i")

  # Define output paths for the current iteration
  output_patient_folder="$DIR/patient_${patient_id_padded}"
  mkdir -p "$output_patient_folder/segmentations"
  mkdir -p "$output_patient_folder/cache"
  output_ct_path="${output_patient_folder}/ct.nii.gz"
  output_gt_path="${output_patient_folder}/segmentations/small_bowel.nii.gz"
  output_coords_path="${output_patient_folder}/cache/start_end.npy"
  output_path_path="${output_patient_folder}/path.npy"

  # Set a default for dilation_iter if not specified, or you can randomize it too
  dilation_iter=1 # Or randomize: dilation_iter=$((RANDOM % 5 + 1))

  # Run the phantom generation script
  python generate_phantom.py \
    --output_ct_path "$output_ct_path" \
    --output_gt_path "$output_gt_path" \
    --output_path_path "$output_path_path" \
    --output_coords_path "$output_coords_path" \
    --volume_shape "$volume_shape" \
    --tube_radius 3 \
    --n_control_points "$n_control_points" \
    --dilation_iterations "$dilation_iter" \
    --seed "$seed" \
    # --visualize # Comment out visualize if you don't want 100 pop-ups
    # --verbose   # Comment out verbose if you want less console output

  echo "Finished generating phantom $i. Output at: $output_patient_folder"
  echo "--------------------------------------------------"
done

echo "All 100 phantoms generated."