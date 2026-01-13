# Setting up the Environment
Create a Python virtual environment and install all dependencies using:

```bash
pip install -r requirements.txt
```

Ensure you have the EVO tool installed for evaluation (see EVO documentation for installation instructions if needed).

# Running Base Pipeline
To run the base visual odometry pipeline, use the following command:

```bash
python VO_base.py
```

This script loads configuration from `config/base.yaml`. By default, it processes all 11 sequences (0-10) from the KITTI dataset and outputs results to the `output_base` directory. The KITTI dataset path is sepecified in the `dataset_path` parameter in `config/base.yaml`.

### Customizing the Base Pipeline
- **Sequences**: To specify which sequences to process, edit the `sequences` list in `config/base.yaml`. For example:
  ```
  sequences: [0, 4, 7]
  ```
  This will run only sequences 00, 04, and 07.
- **Output Directory**: To change the output folder (where poses.txt files are saved for each sequence), modify the `output_dir` parameter in `config/base.yaml`. For example:
  ```
  output_dir: output_base_custom
  ```
- **Other Parameters**: Additional tunable parameters include `nfeatures` (number of ORB features), `scale` (ORB scale factor), `nlevels` (ORB pyramid levels), `matching_norm` (e.g., NORM_HAMMING), `cross_check` (boolean for cross-checking matches), `pnp_confidence`, and `pnp_iterations_count`. Adjust these as needed for experimentation, but default values are provided in the YAML file.

The script will create subdirectories for each sequence in the specified output directory and save the estimated trajectory as `poses.txt`.

# Running Constrained Pipeline
To run the constrained visual odometry pipeline (which includes additional filters like depth thresholds, motion constraints, and grid-based feature clustering), use the following command:

```bash
python VO_constrained.py
```

This script loads configuration from `config/constrained.yaml`. By default, it processes all 11 sequences (0-10) from the KITTI dataset and outputs results to the `output_constrained` directory. The KITTI dataset path is sepecified in the `dataset_path` parameter in `config/constrained.yaml`.

### Customizing the Constrained Pipeline
- **Sequences**: To specify which sequences to process, edit the `sequences` list in `config/constrained.yaml`. For example:
  ```
  sequences: [0, 4, 7]
  ```
  This will run only sequences 00, 04, and 07.
- **Output Directory**: To change the output folder, modify the `output_dir` parameter in `config/constrained.yaml`. For example:
  ```
  output_dir: output_constrained_custom
  ```
- **Other Parameters**: This pipeline has more parameters for fine-tuning, including:
  - Feature detection: `nfeatures`, `scale`, `nlevels`, `grid_size` (e.g., [50, 30]), `max_per_grid`.
  - Depth and uncertainty filters: `min_depth_multiplier` (multiplied by baseline for minimum depth), `relative_depth_uncertainty_threshold`.
  - Motion constraints: `max_translation` (max allowed translation in meters), `max_pitch_deg` (max allowed pitch in degrees).
  - Matching: `matching_norm`, `temporal_inlier_threshold`, `temporal_cross_check`, `stereo_cross_check`, `stereo_pixel_threshold`, `stereo_max_hamming_distance`, `stereo_ratio_test`, `stereo_max_disparity`.
  - PnP: `pnp_confidence`, `pnp_iterations_count`.
  - Other: `orthogonalize_alpha` for planar motion orthogonalization.
  
  Default values are provided in the YAML file. Adjust them based on your needs, such as increasing `nfeatures` for denser matching or tightening `max_translation` for stricter motion constraints.

The script will create subdirectories for each sequence in the specified output directory and save the estimated trajectory as `poses.txt`. If a frame is skipped due to insufficient points or excessive motion, it interpolates using the previous transformation.

# Creating EVO Results
To evaluate the estimated trajectories using the EVO tool, run the following commands (assuming you have ground truth files from the KITTI dataset in a known path, e.g., `ground_truth/poses/`):

For Absolute Pose Error (APE) calculation:

```bash
evo_ape kitti ./path_to_ground_truth/poses/XX.txt ./path_to_estimated_trajectory/poses.txt -p --plot_mode xz
```

For Relative Pose Error (RPE) calculation (e.g., over 100 meters):

```bash
evo_rpe kitti ./path_to_ground_truth/poses/XX.txt ./path_to_estimated_trajectory/poses.txt -p --plot_mode xz --delta 100 --delta_unit m
```

Replace `--delta` and `--delta_unit` values as needed (e.g., use `--delta 1 --delta_unit f` for per-frame RPE).

For Trajectory comparison:

```bash
evo_traj kitti --ref ./path_to_ground_truth/poses/XX.txt ./path_to_estimated_trajectory/poses.txt -p --plot_mode xyz
```

Replace `XX` with the sequence number (e.g., 00), `./path_to_ground_truth/` with the actual ground truth directory, and `./path_to_estimated_trajectory/` with your output (e.g., `output_base/00/`).

# Saving All Results Using a Script
To automate running the above EVO commands for all (or specific) sequences and save the results (text files and plots) in the output folder, use the bash script:

```bash
./create_results.sh <path_to_output_folder>
```

Replace `<path_to_output_folder>` with the path to the folder containing the output data (e.g., the one specified in your YAML config), such as:

```bash
./create_results.sh ./output_constrained/
```

By default, the script processes all sequences (00-10) found in the output folder. To process a specific sequence, provide it as an additional argument:

```bash
./create_results.sh <path_to_output_folder> <sequence_name>
```

For example:

```bash
./create_results.sh ./output_constrained/ 01
```

The script assumes ground truth files are in a `ground_truth/poses/` directory relative to the script (update the script if your ground truth path differs). It will generate APE, RPE, and trajectory files/plots in each sequence's subdirectory.

# Viewing the Results
After running the EVO commands or the script, you can find the generated results in the specified output folder's sequence subdirectories. These include:
- Text files: e.g., `ape_results.txt`, `rpe_results.txt` with error metrics.
- Plots: e.g., `ape_plot_map.png` (APE error map), `trajectory_plot_xyz.png` (3D trajectory overlay), `rpe_plot.png` (RPE errors).

Example plots from sequence 00 in `output_constrained/`:

![APE Map](./output_constrained/00/ape_plot_map.png)
![Trajectory Plot](./output_constrained/00/trajectory_plot_xyz.png)

For interpretation:
- **APE** measures global consistency (total drift).
- **RPE** measures local accuracy (frame-to-frame errors).
- Lower errors indicate better odometry performance. Compare against KITTI benchmarks for context.
