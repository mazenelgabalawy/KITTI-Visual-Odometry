import os
from glob import glob
import numpy as np
import cv2
import yaml

from utils import *
from filter import *
from tqdm import tqdm

if __name__ == "__main__":
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'constrained.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    sequences = config['sequences']
    dataset_path = config['dataset_path']
    output_dir = config['output_dir']
    nfeatures = config['nfeatures']
    scale = config['scale']
    nlevels = config['nlevels']
    grid_size = config['grid_size']
    max_per_grid = config['max_per_grid']
    min_depth_multiplier = config['min_depth_multiplier']  # For min_depth = min_depth_multiplier * base_line
    max_translation = config['max_translation']
    max_pitch_deg = config['max_pitch_deg']  # Renamed for clarity, since it's used for pitch
    matching_norm = getattr(cv2, config['matching_norm'])  # Convert string to cv2 constant, e.g., 'NORM_HAMMING'
    temporal_inlier_threshold = config['temporal_inlier_threshold']
    temporal_cross_check = config['temporal_cross_check']
    stereo_cross_check = config['stereo_cross_check']
    stereo_pixel_threshold = config['stereo_pixel_threshold']
    stereo_max_hamming_distance = config['stereo_max_hamming_distance']
    stereo_ratio_test = config['stereo_ratio_test']
    stereo_max_disparity = config['stereo_max_disparity']
    relative_depth_uncertainty_threshold = config['relative_depth_uncertainty_threshold']
    min_abs_w = config['min_abs_w']  # Triangulated points with |w| < min_abs_w (beyond ~1/min_abs_w m) are rejected
    pnp_confidence = config['pnp_confidence']
    pnp_iterations_count = config['pnp_iterations_count']
    orthogonalize_alpha = config['orthogonalize_alpha']
    
    for i in sequences:
        print(f"Running filter on sequence {i:02d}")
        input_folder = os.path.join(os.path.dirname(__file__), dataset_path, f"{i:02d}")
        output_folder = os.path.join(os.path.dirname(__file__), output_dir, f"{i:02d}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        output_file = os.path.join(output_folder, "poses.txt")
        # Clear previous contents
        open(output_file, 'w').close()
        
        # Get length of sequence
        sequence_length = len(glob(os.path.join(input_folder, "image_0", "*.png")))
        
        # Load Camera Intrinsics and Extrinsics
        calibration_file = os.path.join(input_folder, "calib.txt")
        with open(calibration_file, 'r') as f:
            lines = f.readlines()
        P0 = np.array([float(x) for x in lines[0].strip().split()[1:]]).reshape(3, 4) # Projection matrix for left camera
        P1 = np.array([float(x) for x in lines[1].strip().split()[1:]]).reshape(3, 4) # Projection matrix for right camera
        
        # Decompose Projection Matrices to get Intrinsics and Extrinsics
        Kp_left, R_left, t_left = cv2.decomposeProjectionMatrix(P0)[:3]
        Kp_right, R_right, t_right = cv2.decomposeProjectionMatrix(P1)[:3]
        
        focal_length = Kp_left[0,0]
        
        # Normalize translation vector (t_left is zero vector for left camera)
        t_left = t_left[:3] / t_left[3]
        t_left = t_left.reshape(3)
        
        t_right = t_right[:3] / t_right[3]
        t_right = t_right.reshape(3)
        
        base_line = np.linalg.norm(t_right - t_left)
        
        # Compute Essential and Fundamental Matrices
        t_x = np.array([[0, -t_right[2], t_right[1]],
                        [t_right[2], 0, -t_right[0]],
                        [-t_right[1], t_right[0], 0]])
        E = t_x @ R_right  # Essential matrix
        F = np.linalg.inv(Kp_right).T @ E @ np.linalg.inv(Kp_left)  # Fundamental matrix
        
        # Initialize poses file
        C_k = np.eye(4)  # Initial pose (Identity matrix)
        T_prev = np.eye(3,4)  # Initial transformation
        # Save initial pose of the left camera
        save_transformation_matrix(output_file, C_k)
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scale, nlevels=nlevels)
        
        # Initialize previous images
        left_prev, right_prev = load_stereo_pair(0, input_folder)
        
        left_feat_prev, left_desc_prev = process_image(left_prev, nfeatures, scale, nlevels, grid_size, max_per_grid)
        right_feat_prev, right_desc_prev = process_image(right_prev, nfeatures, scale, nlevels, grid_size, max_per_grid)
                
        for k in tqdm(range(1, sequence_length)):
            left_curr, right_curr = load_stereo_pair(k, input_folder)
            
            left_feat_curr, left_desc_curr = process_image(left_curr, nfeatures, scale, nlevels, grid_size, max_per_grid)
            right_feat_curr, right_desc_curr = process_image(right_curr, nfeatures, scale, nlevels, grid_size, max_per_grid)
            
            # Match features between previous and current left images
            temporal_matches = BF_matcher(left_desc_prev, left_desc_curr, matching_norm, inliner_threshold=temporal_inlier_threshold, cross_check=temporal_cross_check)
                
            # Match features between current left and right images
            stereo_matches_prev = rectified_stereo_matcher(left_feat_prev, left_desc_prev, right_feat_prev, right_desc_prev,
                                                           cross_check=stereo_cross_check, pixel_threshold=stereo_pixel_threshold, max_hamming_distance=stereo_max_hamming_distance,
                                                           ratio_test=stereo_ratio_test, max_disparity=stereo_max_disparity)            
            
            # Filter matches that are exist in both temporal and stereo matches
            left_prev_pts, right_prev_pts, left_curr_pts = filter_temporal_and_stereo_features(temporal_matches, stereo_matches_prev,
                                                                                            left_feat_prev, right_feat_prev, left_feat_curr)
            
            if left_prev_pts.shape[0] == 0:
                print(f"Frame {k}: No matches found after filtering. Skipping frame.")
                # No correspondences; skip this frame
                left_prev = left_curr
                right_prev = right_curr
                
                left_feat_prev = left_feat_curr
                left_desc_prev = left_desc_curr
                
                right_feat_prev = right_feat_curr
                right_desc_prev = right_desc_curr
                
                # Constant velocity: propagate the pose with the previous transformation
                C_k = C_k @ np.vstack((T_prev, np.array([0,0,0,1])))
                save_transformation_matrix(output_file, C_k)
                continue
            
            # Triangulate 3D points from previous left and right images
            points_3D_hom = cv2.triangulatePoints(P0, P1, left_prev_pts.T, right_prev_pts.T)
            
            # Keep only finite points in front of both cameras
            valid_idx, points_3D = filter_valid_landmarks(points_3D_hom, P0, P1, min_abs_w)
            points_3D = points_3D[valid_idx]
            left_curr_pts = left_curr_pts[valid_idx]
            right_prev_pts = right_prev_pts[valid_idx]
            left_prev_pts = left_prev_pts[valid_idx]
            
            depths = points_3D[:, 2]
            sigma_z = np.square(depths)/(focal_length * base_line)
            
            # Filter points based on depth and uncertainty
            valid_idx = points_3D[:, 2] > min_depth_multiplier * base_line
            valid_idx = valid_idx & (sigma_z/depths < relative_depth_uncertainty_threshold)
            points_3D = points_3D[valid_idx]
            left_curr_pts = left_curr_pts[valid_idx]
            right_prev_pts = right_prev_pts[valid_idx]
            left_prev_pts = left_prev_pts[valid_idx]

            # Estimate camera pose using PnP
            # Ensure sufficient correspondences and correct dtypes
            if points_3D.shape[0] < 4:
                print(f"Frame {k}: Not enough points for PnP ({points_3D.shape[0]} points). Skipping frame.")
                # Not enough points for PnP; skip this frame
                left_prev = left_curr
                right_prev = right_curr
                
                left_feat_prev = left_feat_curr
                left_desc_prev = left_desc_curr
                
                right_feat_prev = right_feat_curr
                right_desc_prev = right_desc_curr
                
                # Constant velocity: propagate the pose with the previous transformation
                C_k = C_k @ np.vstack((T_prev, np.array([0,0,0,1])))
                save_transformation_matrix(output_file, C_k)
                continue

            points_3D = points_3D.astype(np.float32)
            left_curr_pts = left_curr_pts.astype(np.float32)

            _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D, left_curr_pts, Kp_left, None, confidence=pnp_confidence, iterationsCount=pnp_iterations_count)
            
            R_curr, _ = cv2.Rodrigues(rvec)
            t_curr = tvec.reshape(3)
            
            
            R_ortho = orthogonalize_to_planar_motion(R_curr, alpha=orthogonalize_alpha)
            
            # Put Rotation and Translation into KITTI format (x-right, y-down, z-forward)
            R_cam = R_ortho.T
            t_cam = -R_ortho.T @ t_curr
            
            roll, pitch, yaw = rotationMatrixToEulerAngles(R_cam)
            
                        
            # Check if the translation or pitch is too large
            if np.linalg.norm(t_cam) > max_translation or abs(pitch) > np.deg2rad(max_pitch_deg):
                print(f"Frame {k}: Translation too large ({np.linalg.norm(t_cam):.2f} m) or pitch too large ({np.rad2deg(pitch):.2f} degrees). Skipping frame.")
                
                # Skip this frame
                left_prev = left_curr
                right_prev = right_curr
                
                left_feat_prev = left_feat_curr
                left_desc_prev = left_desc_curr
                
                right_feat_prev = right_feat_curr
                right_desc_prev = right_desc_curr
                
                # Constant velocity: propagate the pose with the previous transformation
                C_k = C_k @ np.vstack((T_prev, np.array([0,0,0,1])))
                save_transformation_matrix(output_file, C_k)
                continue

            T = np.hstack((R_cam, t_cam.reshape(3,1)))
            # Update global pose
            C_k = C_k @ np.vstack((T, np.array([0,0,0,1])))
            
            # Save current pose
            save_transformation_matrix(output_file, C_k)
            
            # Copy previous images for next iteration
            left_prev = left_curr
            right_prev = right_curr
            
            left_feat_prev = left_feat_curr
            left_desc_prev = left_desc_curr
            
            right_feat_prev = right_feat_curr
            right_desc_prev = right_desc_curr
            
            # Copy transformation for constant-velocity propagation in case of skipping
            T_prev = T
