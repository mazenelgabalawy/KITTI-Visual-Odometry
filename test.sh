#!/bin/bash

# Define the array (Bash uses spaces, not commas)
sequence=("00")
delta=100
output_folder="./output_ORB_epipolar/" ## Change for relevant output folder

# Loop through each sequence
for sequence_id in "${sequence[@]}"; do
    echo "Processing sequence ${sequence_id}..."

    # Define directories and ensure output exists
    output_dir="${output_folder}${sequence_id}/"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Define file paths
    gt_file="./poses/${sequence_id}.txt"
    est_file="${output_dir}poses.txt"

    # Overwrite existing result files
    > "${output_dir}ape_results.txt"
    > "${output_dir}rpe_results.txt"    
    # Run APE - Note: --save_results uses .zip, not .txt
    echo "Delta: $delta" > "${output_dir}ape_results.txt"
    evo_ape kitti "$gt_file" "$est_file" -a --plot_mode xz \
        --save_plot "${output_dir}ape_plot.png" > "${output_dir}ape_results.txt"

    # Run RPE
    echo "Delta: $delta" > "${output_dir}rpe_results.txt"
    evo_rpe kitti "$gt_file" "$est_file" -a --delta $delta --plot_mode xz \
        --save_plot "${output_dir}rpe_plot.png" > "${output_dir}rpe_results.txt"


    
done
