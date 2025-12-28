import os
import cv2
import numpy as np
from glob import glob

from utils import *
from filter import *

if __name__ == "__main__":
    
    nfeature = 2000
    nlevles = 8
    scale = 1.6
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    orb = cv2.ORB_create(nfeatures=nfeature, nlevels=nlevles, scaleFactor=scale)
    
    # Load images
    input_folder = os.path.dirname(__file__) + f"/sequences/{0:02d}/"
    output_folder = os.path.dirname(__file__) + f"/output_ORB_epipolar/{0:02d}/"
    
    left_curr, right_curr = load_stereo_pair(1500, input_folder)
    left_curr = clahe.apply(left_curr)
    right_curr = clahe.apply(right_curr)

    left_curr = cv2.cvtColor(left_curr, cv2.COLOR_GRAY2BGR)
    # Detect and compute ORB features
    keypoints_left, descriptors_left = orb.detectAndCompute(left_curr, None)
    
    # Draw keypoints on images
    left_with_keypoints = cv2.drawKeypoints(left_curr, keypoints_left, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # resize images
    # left_image = cv2.resize(left_curr, (0, 0), fx=0.75, fy=0.75)
    # left_with_keypoints = cv2.resize(left_with_keypoints, (0, 0), fx=0.75, fy=0.75)
    print(f"Number of ORB keypoints detected: {len(keypoints_left)}")
    
    # Filter keypoints based on response
    left_keypoints_filtered, descriptors_left_filtered = cluster_keypoints(keypoints_left, descriptors_left, image=left_curr, grid_size=(30,20), max_per_grid=3)
    print(f"Number of ORB keypoints after filtering: {len(left_keypoints_filtered)}")
    # Draw filtered keypoints on images
    keypoints_left_filtered_img = cv2.drawKeypoints(left_curr, left_keypoints_filtered, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display images with keypoints
    merge_image = np.vstack((left_curr, left_with_keypoints,keypoints_left_filtered_img))
    merge_image = cv2.resize(merge_image, (0, 0), fx=0.75, fy=0.75)
    cv2.imshow("Left Image and ORB Keypoints", merge_image)
    # cv2.imshow("ORB Keypoints", left_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
