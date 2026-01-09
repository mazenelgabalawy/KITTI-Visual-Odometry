import numpy as np
import cv2
from filter import *
from utils import *

from glob import glob
import os
from tqdm import tqdm
import time

if __name__ == "__main__":
    
    
    # sequence = 0
    # # image_no = 0
    # input_folder = os.path.dirname(__file__) + f"/sequences/{sequence:02d}/"
    # output_folder = os.path.dirname(__file__) + f"/output_features/{sequence:02d}/"
    # os.makedirs(output_folder, exist_ok=True)
    # seq_length = len(glob(os.path.join(input_folder, "image_0", "*.png")))
    # start_time = time.time()

    # for image_no in tqdm(range(seq_length)):
        
    #     method = 'ORB'  # 'SIFT' or 'ORB'
    #     if method == 'SIFT':
    #         detector = cv2.SIFT_create(nfeatures=2000)
    #         norm = cv2.NORM_L2
    #     else:
    #         detector = cv2.ORB_create(nfeatures=2000)
    #         norm = cv2.NORM_HAMMING
        
    #     left_img = cv2.imread(f"sequences/{sequence:02d}/image_0/{image_no:06d}.png", cv2.IMREAD_GRAYSCALE)
    #     right_img = cv2.imread(f"sequences/{sequence:02d}/image_1/{image_no:06d}.png", cv2.IMREAD_GRAYSCALE)
        
    #     left_kp, left_desc = detector.detectAndCompute(left_img, None)
    #     right_kp, right_desc = detector.detectAndCompute(right_img, None)
        
    #     # matches = BF_matcher(left_desc, right_desc, norm_type=norm, inliner_threshold=0.7, cross_check=True)

    # end_time = time.time()
    # total_time = end_time - start_time
    # if total_time > 0:
    #     it_per_sec = seq_length / total_time
    #     print(f"Average iterations/second for sequence {sequence:02d}: {it_per_sec:.2f}")
    # else:
    #     print("Total time is too small to compute iterations/second.")

    nfeatures = 3000
    scale = 1.6
    nlevels = 8
    matching_norm = cv2.NORM_HAMMING
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor = scale, nlevels = nlevels)
    left_img , right_img = load_stereo_pair(0, os.path.dirname(__file__) + f"/sequences/00/")
    left_kp_orb, left_des_orb = orb.detectAndCompute(left_img, None)    
    left_kp_orb, left_des_orb = cluster_keypoints(left_kp_orb, left_des_orb, image=left_img, grid_size=(70,50), max_per_grid=3)
    right_kp_orb, right_des_orb = orb.detectAndCompute(right_img, None)    
    right_kp_orb, right_des_orb = cluster_keypoints(right_kp_orb, right_des_orb, image=right_img, grid_size=(70,50), max_per_grid=3)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_BF = bf.match(left_des_orb, right_des_orb)
    matches_stereo = rectified_stereo_matcher(left_kp_orb, left_des_orb, right_kp_orb, right_des_orb,
                                                           cross_check=True, pixel_threshold=2.0, max_hamming_distance=35,
                                                           ratio_test=0.8, max_disparity=64.0)
    
    print(f"Stereo matches : {len(matches_stereo)}")
    matches_BF_img = cv2.drawMatches(left_img, left_kp_orb, right_img, right_kp_orb, matches_BF, None, flags=cv2.DrawMatchesFlags_DEFAULT)    
    matches_stereo_img = cv2.drawMatches(left_img, left_kp_orb, right_img, right_kp_orb, matches_stereo, None, flags=cv2.DrawMatchesFlags_DEFAULT)
    
    matches_BF_img = cv2.resize(matches_BF_img,dsize=None,fx=0.7,fy=1)
    matches_stereo_img = cv2.resize(matches_stereo_img,dsize=None,fx=0.7,fy=1)
    
    # img = np.vstack((cv2.resize(matches_BF_img,dsize=None,fx=0.7,fy=0.7), cv2.resize(matches_stereo_img,dsize=None,fx=0.7,fy=0.7)))
    cv2.imwrite("matches_BF.png", matches_BF_img)
    cv2.imwrite("matches_stereo.png", matches_stereo_img)
    
    cv2.imshow("BF matching", matches_BF_img)
    cv2.imshow("Stereo matching", matches_stereo_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
