import os
from glob import glob
import numpy as np
import cv2

from utils import *
from filter import *
from tqdm import tqdm


if __name__ == "__main__":
    for i in range(0,11):
        print(f"Running filter on sequence {i:02d}")
        input_folder = os.path.dirname(__file__) + f"/sequences/{i:02d}/"
        output_folder = os.path.dirname(__file__) + f"/output_constrained/{i:02d}/"
        
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
        
        # print("Left Camera Intrinsics:\n", Kp_left)
        # print("Left Camera Rotation:\n", R_left)
        # print("Left Camera Translation:\n", t_left)
        
        # print("Right Camera Intrinsics:\n", Kp_right)
        # print("Right Camera Rotation:\n", R_right)
        # print("Right Camera Translation:\n", t_right)
        
        # Initialize previous images
        left_prev, right_prev = load_stereo_pair(0, input_folder)    

        method = 'ORB'  # Feature detection method
        if method == 'ORB':
            matching_norm = cv2.NORM_HAMMING
        else:
            matching_norm = cv2.NORM_L2
        
        nfeatures = 3000  # Number of features to detect
        scale = 1.6
        nlevels = 8
        
        grid_size = (30, 20)
        max_per_grid = 5
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor = scale, nlevels = nlevels)
            
        for k in tqdm(range(1, sequence_length)):
            left_curr, right_curr = load_stereo_pair(k, input_folder)
            
            # Detect and compute features and descriptors for current left image
            left_feat_prev, left_desc_prev = orb.detectAndCompute(left_prev, None)
            right_feat_prev, right_desc_prev = orb.detectAndCompute(right_prev, None)
            left_feat_curr, left_desc_curr = orb.detectAndCompute(left_curr, None)
            
            
            # Filter keypoints based on clustering to ensure uniform distribution
            left_feat_prev, left_desc_prev = cluster_keypoints(left_feat_prev, left_desc_prev, image=left_prev, grid_size=grid_size, max_per_grid=max_per_grid)
            right_feat_prev, right_desc_prev = cluster_keypoints(right_feat_prev, right_desc_prev, image=right_prev, grid_size=grid_size, max_per_grid=max_per_grid)
            left_feat_curr, left_desc_curr = cluster_keypoints(left_feat_curr, left_desc_curr, image=left_curr, grid_size=grid_size, max_per_grid=max_per_grid)
            
            # Match features between previous and current left images (temporal matching) and return good matches
            temporal_matches = BF_matcher(left_desc_prev, left_desc_curr, matching_norm, inliner_threshold=0.7, cross_check=True)
                
            # Match features between current left and right images (stereo matching)
            stereo_matches_prev = rectified_stereo_matcher(left_feat_prev, left_desc_prev, right_feat_prev, right_desc_prev,
                                                           cross_check=True, pixel_threshold=2.0, max_hamming_distance=30,
                                                           ratio_test=0.7, max_disparity=64.0)
            # Filter matches that are exist in both temporal and stereo matches
            left_prev_pts, right_prev_pts, left_curr_pts = filter_temporal_and_stereo_features(temporal_matches, stereo_matches_prev,
                                                                                            left_feat_prev, right_feat_prev, left_feat_curr)
            
            # Triangulate 3D points from previous left and right images
            points_3D_hom = cv2.triangulatePoints(P0, P1, left_prev_pts.T, right_prev_pts.T)
            points_3D = (points_3D_hom[:3, :] / points_3D_hom[3, :]).T  # Convert to non-homogeneous coordinates
            
            # Filter points with positive depth and depth < 100 meters
            valid_idx = points_3D[:, 2] > 0
            valid_idx = valid_idx & (points_3D[:, 2] < 100)
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
                
                # Interpolate pose using previous transformation
                C_k = C_k @ np.vstack((T_prev, np.array([0,0,0,1])))
                
                # Save previous pose again
                save_transformation_matrix(output_file, C_k)
                continue

            points_3D = points_3D.astype(np.float32)
            left_curr_pts = left_curr_pts.astype(np.float32)

            _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D, left_curr_pts, Kp_left, None)
            
            R_curr, _ = cv2.Rodrigues(rvec)
            t_curr = tvec.reshape(3)
            
            
            R_ortho = orthogonalize_to_planar_motion(R_curr, t_curr)
            
            # Put Rotation and Translation into KITTI format (x-right, y-down, z-forward)
            R_cam = R_ortho.T
            t_cam = -R_ortho.T @ t_curr
            
            roll, pitch, yaw = rotationMatrixToEulerAngles(R_cam)
                        
            # Check if the translation or pitch is too large
            if np.linalg.norm(t_cam) > 15.0 or pitch > np.deg2rad(30): # assuming 60km/h max speed and 30 degrees max pitch
                print(f"Frame {k}: Translation too large ({np.linalg.norm(t_cam):.2f} m) or pitch too large ({np.rad2deg(pitch):.2f} degrees). Skipping frame.")
                
                # Skip this frame; keep previous pose
                left_prev = left_curr
                right_prev = right_curr
                
                
                # Save previous pose again
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
            
            # Copy pose for interpolation in case of skipping
            T_prev = T
