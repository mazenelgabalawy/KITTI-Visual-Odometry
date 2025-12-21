import os
from glob import glob
import numpy as np
import cv2

from utils import *
from filter import *


if __name__ == "__main__":
    input_folder = os.path.dirname(__file__) + "/sequences/00/"
    output_folder = os.path.dirname(__file__) + "/sequences/output/00"
    
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
    
    print("Left Camera Intrinsics:\n", Kp_left)
    print("Left Camera Rotation:\n", R_left)
    print("Left Camera Translation:\n", t_left)
    
    print("Right Camera Intrinsics:\n", Kp_right)
    print("Right Camera Rotation:\n", R_right)
    print("Right Camera Translation:\n", t_right)
    
    # Initialize previous images
    left_prev, right_prev = load_stereo_pair(0, input_folder)    

    for k in range(1, sequence_length):
        left_curr, right_curr = load_stereo_pair(k, input_folder)
        
        left_feat_prev, left_desc_prev = get_features(left_prev)
        right_feat_prev, right_desc_prev = get_features(right_prev)
        
        left_feat_curr, left_desc_curr = get_features(left_curr)
        right_feat_curr, right_desc_curr = get_features(right_curr)
        
        # Match features between previous and current left images (temporal matching) and return good matches
        temporal_matches = BF_matcher(left_desc_prev, left_desc_curr, cv2.NORM_HAMMING, inliner_threshold=0.7, cross_check=True)
              
        # Match features between current left and right images (stereo matching)
        stereo_matches_prev = BF_matcher(left_desc_prev, right_desc_prev, cv2.NORM_HAMMING, inliner_threshold=0.7, cross_check=True)
        # Filter stereo matches based on Epipolar constraint for rectified images
        stereo_matches_prev = filter_rectified_stereo_matches(stereo_matches_prev, left_feat_prev, right_feat_prev, threshold=2.0) 
        
        # Filter matches that are exist in both temporal and stereo matches
        left_prev_pts, right_prev_pts, left_curr_pts = filter_temporal_and_stereo_features(temporal_matches, stereo_matches_prev,
                                                                                           left_feat_prev, right_feat_prev, left_feat_curr)
        
        # Triangulate 3D points from previous left and right images
        points_3D_hom = cv2.triangulatePoints(P0, P1, left_prev_pts.T, right_prev_pts.T)
        points_3D = (points_3D_hom[:3, :] / points_3D_hom[3, :]).T  # Convert to non-homogeneous coordinates
        
        # print(f"Frame {k}: Triangulated {points_3D.shape[0]} points.")
        # print(f"Frame {k}: Triangulated Points Sample:\n", points_3D)
        
        # Estimate camera pose using PnP
        _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D, left_curr_pts, Kp_left, None)
        R_curr, _ = cv2.Rodrigues(rvec)
        t_curr = tvec.reshape(3)
        # print(f"Frame {k}: Estimated Rotation:\n", R_curr)
        # print(f"Frame {k}: Estimated Translation:\n", t_curr)
        
        T = np.hstack((R_curr, t_curr.reshape(3,1)))
        # Update global pose
        C_k = C_k @ np.vstack((T, np.array([0,0,0,1])))
        
        # Save current pose
        save_transformation_matrix(output_file, C_k)
        # # Plot matches for visualization
        # match_img = plot_matches(left_prev, left_feat_prev, left_curr, left_feat_curr, good_matches)
        # match_img = cv2.resize(match_img, (0,0), fx=0.75, fy=0.75)
        
        # cv2.imshow("Feature Matches", match_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Copy previous images for next iteration
        left_prev = left_curr
        right_prev = right_curr
