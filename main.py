import os
from glob import glob
import numpy as np
import cv2

from utils import *
from filter import *
from tqdm import tqdm


if __name__ == "__main__":
    for i in range(0,1):
        print(f"Running filter on sequence {i:02d}")
        input_folder = os.path.dirname(__file__) + f"/sequences/{i:02d}/"
        output_folder = os.path.dirname(__file__) + f"/output_ORB_epipolar/{i:02d}/"
        
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
        
        nfeatures = 2000  # Number of features to detect
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
            
            
            # Filter keypoints based on clustering to ensure uniform distribution
            left_feat_prev, left_desc_prev = cluster_keypoints(left_feat_prev, left_desc_prev, image=left_prev, grid_size=(30,20), max_per_grid=3)
            right_feat_prev, right_desc_prev = cluster_keypoints(right_feat_prev, right_desc_prev, image=right_prev, grid_size=(30,20), max_per_grid=3)
            left_feat_curr, left_desc_curr = cluster_keypoints(left_feat_curr, left_desc_curr, image=left_curr, grid_size=(30,20), max_per_grid=3)

            # # Plotting features of left and right images for visualization
            # left_feat_img = cv2.drawKeypoints(left_prev, left_feat_prev, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # right_feat_img = cv2.drawKeypoints(right_prev, right_feat_prev, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # feat_img = np.hstack((left_feat_img, right_feat_img))
            # feat_img = cv2.resize(feat_img, (0,0), fx=0.75, fy=0.75)
            # cv2.imshow("Features", feat_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # Match features between previous and current left images (temporal matching) and return good matches
            temporal_matches = BF_matcher(left_desc_prev, left_desc_curr, matching_norm, inliner_threshold=0.7, cross_check=True)
                
            # Match features between current left and right images (stereo matching)
            stereo_matches_prev = rectified_stereo_matcher(left_feat_prev, left_desc_prev, right_feat_prev, right_desc_prev,
                                                           cross_check=True, pixel_threshold=2.0, max_hamming_distance=30,
                                                           ratio_test=0.7, max_disparity=64.0)
            # Filter matches that are exist in both temporal and stereo matches
            left_prev_pts, right_prev_pts, left_curr_pts = filter_temporal_and_stereo_features(temporal_matches, stereo_matches_prev,
                                                                                            left_feat_prev, right_feat_prev, left_feat_curr)
            
            # left_feat_filtered = [left_feat_prev[m.queryIdx] for m in temporal_matches if m.queryIdx in [mp.queryIdx for mp in stereo_matches_prev]]
            # right_feat_filtered = [right_feat_prev[m.trainIdx] for m in stereo_matches_prev if m.queryIdx in [mt.queryIdx for mt in temporal_matches]]
            
            # # Plot filtered left and right features for visualization
            # left_feat_filtered_img = cv2.drawKeypoints(left_prev,left_feat_filtered, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # right_feat_filtered_img = cv2.drawKeypoints(right_prev,right_feat_filtered, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # filtered_feat_img = np.hstack((left_feat_filtered_img, right_feat_filtered_img))
            # filtered_feat_img = cv2.resize(filtered_feat_img, (0,0), fx=0.75, fy=0.75)
            # cv2.imshow("Filtered Features", filtered_feat_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
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
            
            # # Get maximum and minimum depth values for debugging
            # depths = points_3D[:, 2]
            # depths = np.argmax(depths), np.argmin(depths)
            # max_depth = points_3D[depths[0], 2]
            # min_depth = points_3D[depths[1], 2]
            # print(f"Frame {k}: Max Depth value: {max_depth}, Min Depth value: {min_depth}")
        
            
            # print(f"Frame {k}: Triangulated {points_3D.shape[0]} points.")
            # print(f"Frame {k}: Triangulated Points Sample:\n", points_3D)
            
            # Estimate camera pose using PnP
            # Ensure sufficient correspondences and correct dtypes
            if points_3D.shape[0] < 4:
                # Not enough points for PnP; skip this frame
                left_prev = left_curr
                right_prev = right_curr
                continue

            points_3D = points_3D.astype(np.float32)
            left_curr_pts = left_curr_pts.astype(np.float32)

            _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D, left_curr_pts, Kp_left, None)
            R_curr, _ = cv2.Rodrigues(rvec)
            t_curr = tvec.reshape(3)
            # print(f"Frame {k}: Estimated Rotation:\n", R_curr)
            # print(f"Frame {k}: Estimated Translation:\n", t_curr)
            
            # Put Rotation and Translation into KITTI format (x-right, y-down, z-forward)
            R_cam = R_curr.T
            t_cam = -R_curr.T @ t_curr

            T = np.hstack((R_cam, t_cam.reshape(3,1)))
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
