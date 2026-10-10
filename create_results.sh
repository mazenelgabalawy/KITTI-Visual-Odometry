#!/bin/bash

# Define the array (Bash uses spaces, not commas)
sequence=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10")
#sequence=("10")
delta=100 # RPE interval in meters along the ground-truth path
output_folder=$1 # argument 1: output folder path (e.g., ./results/constrained/)
sequence_number=$2 # argument 2 (optional): a single sequence (e.g., 01)


if [ -n "$sequence_number" ]; then
    sequence=("$sequence_number")
fi

# Ground truth poses are read from ./poses, or from ./ground_truth/poses if ./poses does not exist
gt_folder="./poses"
if [ ! -d "$gt_folder" ]; then
    gt_folder="./ground_truth/poses"
fi

old_files=("ape_results.txt" "ape_rotation_results.txt" "rpe_results.txt" "rpe_rotation_results.txt"
            "ape_plot_map.png" "ape_plot_raw.png" "rpe_plot_map.png" "rpe_plot_raw.png"
            "trajectory_plot_trajectories.png" "trajectory_plot_xyz.png" "trajectory_plot_rpy.png"
)

# Exit status: non-zero if any evaluation failed
status=0

# Loop through each sequence
for sequence_id in "${sequence[@]}"; do
    echo "Processing sequence ${sequence_id}..."

    # Define directories and ensure output exists
    output_dir="${output_folder%/}/${sequence_id}/"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Define file paths
    gt_file="${gt_folder}/${sequence_id}.txt"
    est_file="${output_dir}poses.txt"


    # Overwrite existing result files
    for file in "${old_files[@]}"; do
        if [ -f "${output_dir}${file}" ]; then
            rm "${output_dir}${file}"
        fi
    done

    # Run APE (no alignment): translation error in m, rotation error in rad
    evo_ape kitti "$gt_file" "$est_file" --plot_mode xz \
        --save_plot "${output_dir}ape_plot.png" > "${output_dir}ape_results.txt" || status=1
    evo_ape kitti "$gt_file" "$est_file" --pose_relation angle_rad \
        > "${output_dir}ape_rotation_results.txt" || status=1

    # Run RPE over 100 m: every frame is paired with the frame about 100 m ahead on the
    # ground-truth path (within 10%), so all estimates of a sequence use the same pairs
    evo_rpe kitti "$gt_file" "$est_file" --delta $delta --delta_unit m --all_pairs --pairs_from_reference --plot_mode xz \
        --save_plot "${output_dir}rpe_plot.png" > "${output_dir}rpe_results.txt" || status=1
    evo_rpe kitti "$gt_file" "$est_file" --delta $delta --delta_unit m --all_pairs --pairs_from_reference --pose_relation angle_rad \
        > "${output_dir}rpe_rotation_results.txt" || status=1

    # Run trajactory plots
    evo_traj kitti "$gt_file" "$est_file" --plot_mode xz \
        --save_plot "${output_dir}trajectory_plot.png" || status=1

done

exit $status
