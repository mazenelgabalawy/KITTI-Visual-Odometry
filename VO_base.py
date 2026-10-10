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
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'base.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    sequences = config['sequences']
    dataset_path = config['dataset_path']
    output_dir = config['output_dir']
    nfeatures = config['nfeatures']
    scale = config['scale']
    nlevels = config['nlevels']
    matching_norm = getattr(cv2, config['matching_norm'])  # Convert string to cv2 constant, e.g., 'NORM_HAMMING'
    cross_check = config['cross_check']
    min_abs_w = config['min_abs_w']  # Triangulated points with |w| < min_abs_w (beyond ~1/min_abs_w m) are rejected
    pnp_confidence = config['pnp_confidence']
    pnp_iterations_count = config['pnp_iterations_count']
    pnp_min_inliers = config['pnp_min_inliers']
    
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
        
        # Normalize translation vector (t_left is zero vector for left camera)
        t_left = t_left[:3] / t_left[3]
        t_left = t_left.reshape(3)
        
        t_right = t_right[:3] / t_right[3]
        t_right = t_right.reshape(3)
        
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
                
        # Initialize previous images
        left_prev, right_prev = load_stereo_pair(0, input_folder)    
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scale, nlevels=nlevels)
            
        for k in tqdm(range(1, sequence_length)):
            left_curr, right_curr = load_stereo_pair(k, input_folder)
            
            # Detect and compute features and descriptors for current left image
            left_feat_prev, left_desc_prev = orb.detectAndCompute(left_prev, None)
            right_feat_prev, right_desc_prev = orb.detectAndCompute(right_prev, None)
            left_feat_curr, left_desc_curr = orb.detectAndCompute(left_curr, None)
            
            # Match features between previous and current left images (temporal matching) and return good matches
            bf = cv2.BFMatcher(matching_norm, crossCheck=cross_check)
            temporal_matches = bf.match(left_desc_prev, left_desc_curr)
                 
            # Match features between current left and right images (stereo matching)
            stereo_matches_prev = bf.match(left_desc_prev, right_desc_prev)
            
            # Filter matches that are exist in both temporal and stereo matches
            left_prev_pts, right_prev_pts, left_curr_pts = filter_temporal_and_stereo_features(temporal_matches, stereo_matches_prev,
                                                                                            left_feat_prev, right_feat_prev, left_feat_curr)
            
            if left_prev_pts.shape[0] == 0:
                print(f"Frame {k}: No matches found after filtering. Skipping frame.")
                # No correspondences; skip this frame
                left_prev = left_curr
                right_prev = right_curr
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
            
            # Ensure sufficient correspondences and correct dtypes
            if points_3D.shape[0] < 4:
                print(f"Frame {k}: Not enough points for PnP ({points_3D.shape[0]} points). Skipping frame.")
                # Not enough points for PnP; skip this frame
                left_prev = left_curr
                right_prev = right_curr
                # Constant velocity: propagate the pose with the previous transformation
                C_k = C_k @ np.vstack((T_prev, np.array([0,0,0,1])))
                save_transformation_matrix(output_file, C_k)
                continue

            points_3D = points_3D.astype(np.float32)
            left_curr_pts = left_curr_pts.astype(np.float32)

            # Estimate camera pose using PnP
            success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D, left_curr_pts, Kp_left, None,
                                                              confidence=pnp_confidence, iterationsCount=pnp_iterations_count)
            
            # Accept the pose only if PnP succeeded with a finite pose and enough inliers
            num_inliers = 0 if inliers is None else len(inliers)
            if not success or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()) or num_inliers < pnp_min_inliers:
                print(f"Frame {k}: PnP failed ({num_inliers} inliers). Skipping frame.")
                # PnP failed; skip this frame
                left_prev = left_curr
                right_prev = right_curr
                # Constant velocity: propagate the pose with the previous transformation
                C_k = C_k @ np.vstack((T_prev, np.array([0,0,0,1])))
                save_transformation_matrix(output_file, C_k)
                continue
            
            R_curr, _ = cv2.Rodrigues(rvec)
            t_curr = tvec.reshape(3)

            # Put Rotation and Translation into KITTI format (x-right, y-down, z-forward)
            R_cam = R_curr.T
            t_cam = -R_curr.T @ t_curr
            

            T = np.hstack((R_cam, t_cam.reshape(3,1)))
            # Update global pose
            C_k = C_k @ np.vstack((T, np.array([0,0,0,1])))
            
            # Save current pose
            save_transformation_matrix(output_file, C_k)
            
            # Copy previous images for next iteration
            left_prev = left_curr
            right_prev = right_curr
            
            # Copy transformation for constant-velocity propagation in case of skipping
            T_prev = T
