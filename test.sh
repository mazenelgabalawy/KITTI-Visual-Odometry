#!/bin/bash

# 1. Define the array (Bash uses spaces, not commas)
sequence=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10")

for sequence_id in "${sequence[@]}"; do
    echo "Processing sequence ${sequence_id}..."

    # Define directories and ensure output exists
    output_dir="./output/${sequence_id}/"

    # Define file paths
    gt_file="./poses/${sequence_id}.txt"
    est_file="./output/${sequence_id}/poses.txt"

    # Run APE - Note: --save_results uses .zip, not .txt
    evo_ape kitti "$gt_file" "$est_file" -a --plot_mode xz \
        --save_plot "${output_dir}ape_plot.png" > "${output_dir}ape_results.txt"

    # Run RPE
    evo_rpe kitti "$gt_file" "$est_file" -a --delta 100 --plot_mode xz \
        --save_plot "${output_dir}rpe_plot.png" > "${output_dir}rpe_results.txt"
done
