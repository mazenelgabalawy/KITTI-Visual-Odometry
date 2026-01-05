import os
import cv2
import numpy as np
from glob import glob

from utils import *
from filter import *

if __name__ == "__main__":
    nfeatures = 1000
    scale = 1.6
    nlevels = 8
    
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor = scale, nlevels = nlevels)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    
    sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for _ in range(10):
        sequence = np.random.choice(sequences)
        sequence_length = len(glob(os.path.join(os.path.dirname(__file__) + f"/sequences/{sequence:02d}/", "image_0", "*.png")))
        image = np.random.randint(0, sequence_length)
        
        print(f"Processing sequence {sequence:02d}, image {image:06d}")
        
        input_folder = os.path.dirname(__file__) + f"/sequences/{sequence:02d}/"
        output_folder = os.path.dirname(__file__) + f"/keypoints/"
        os.makedirs(output_folder, exist_ok=True)
        
        left_img , _ = load_stereo_pair(image, input_folder)
        left_kp_orb, left_des_orb = orb.detectAndCompute(left_img, None)
        left_kp_sift, left_des_sift = sift.detectAndCompute(left_img, None)
        print(f"Detected {len(left_kp_sift)} SIFT keypoints in the left image.")
        print(f"Detected {len(left_kp_orb)} ORB keypoints in the left image.")
        # Draw keypoints on the image
        sift_kp = cv2.drawKeypoints(left_img, left_kp_sift, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        orb_kp = cv2.drawKeypoints(left_img, left_kp_orb, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        kp_img = np.vstack((sift_kp, orb_kp))
        
        # Add title to each side
        cv2.putText(kp_img, 'SIFT', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(kp_img, 'ORB', (50, sift_kp.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(output_folder, f"keypoints_comparison_{sequence:02d}_{image:06d}.png"), kp_img)
        