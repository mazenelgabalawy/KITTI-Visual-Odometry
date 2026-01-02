#!/bin/bash

# Define the array (Bash uses spaces, not commas)
sequence=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10")
delta=100
output_folder="./output_ORB_epipolar/" ## Change for relevant output folder
old_files=("ape_results.txt" "rpe_results.txt" "ape_plot_map.png" "ape_plot_raw.png" 
            "rpe_plot_map.png" "rpe_plot_raw.png" 
            "trajectory_plot_trajectories.png" "trajectory_plot_xyz.png" "trajectory_plot_rpy.png"
)

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
    for file in "${old_files[@]}"; do
        if [ -f "${output_dir}${file}" ]; then
            rm "${output_dir}${file}"
        fi
    done

    # Run APE - Note: --save_results uses .zip, not .txt
    echo "Delta: $delta" > "${output_dir}ape_results.txt"
    evo_ape kitti "$gt_file" "$est_file" --plot_mode xz \
        --save_plot "${output_dir}ape_plot.png" > "${output_dir}ape_results.txt"

    # Run RPE
    echo "Delta: $delta" > "${output_dir}rpe_results.txt"
    evo_rpe kitti "$gt_file" "$est_file" --delta $delta --plot_mode xz \
        --save_plot "${output_dir}rpe_plot.png" > "${output_dir}rpe_results.txt"

    # Run trajactory plots
    evo_traj kitti "$gt_file" "$est_file" --plot_mode xz \
        --save_plot "${output_dir}trajectory_plot.png"

    # Remove unwanted files
    
    
done
