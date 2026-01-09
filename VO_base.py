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
        output_folder = os.path.dirname(__file__) + f"/output_base/{i:02d}/"
        
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
        # Save initial pose of the left camera
        save_transformation_matrix(output_file, C_k)
                
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
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor = scale, nlevels = nlevels)
            
        for k in tqdm(range(1, sequence_length)):
            left_curr, right_curr = load_stereo_pair(k, input_folder)
            
            # Detect and compute features and descriptors for current left image
            left_feat_prev, left_desc_prev = orb.detectAndCompute(left_prev, None)
            right_feat_prev, right_desc_prev = orb.detectAndCompute(right_prev, None)
            left_feat_curr, left_desc_curr = orb.detectAndCompute(left_curr, None)
            
            # Match features between previous and current left images (temporal matching) and return good matches
            bf = cv2.BFMatcher(matching_norm, crossCheck=True)
            temporal_matches = bf.match(left_desc_prev, left_desc_curr)
                 
            # Match features between current left and right images (stereo matching)
            stereo_matches_prev = bf.match(left_desc_prev, right_desc_prev)
            
            # Filter matches that are exist in both temporal and stereo matches
            left_prev_pts, right_prev_pts, left_curr_pts = filter_temporal_and_stereo_features(temporal_matches, stereo_matches_prev,
                                                                                            left_feat_prev, right_feat_prev, left_feat_curr)
            
            # Triangulate 3D points from previous left and right images
            points_3D_hom = cv2.triangulatePoints(P0, P1, left_prev_pts.T, right_prev_pts.T)
            points_3D = (points_3D_hom[:3, :] / points_3D_hom[3, :]).T  # Convert to non-homogeneous coordinates
            

            # Estimate camera pose using PnP
            _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D, left_curr_pts, Kp_left, None)
            
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
