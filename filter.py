import numpy as np
import cv2


def get_features(image, nfeatures = 2000, method='ORB'):
    
    if method == 'ORB':
        # Initialize ORB detector
        orb = cv2.ORB.create(nfeatures=nfeatures)
        
        # Find the keypoints and descriptors with ORB
        keypoints, descriptors = orb.detectAndCompute(image, None)
    elif method == 'SIFT':
        # Initialize SIFT detector
        sift = cv2.SIFT.create(nfeatures=nfeatures)
        
        # Find the keypoints and descriptors with SIFT
        keypoints, descriptors = sift.detectAndCompute(image, None)
    else:
        raise ValueError("Unsupported feature detection method.")
    
    return keypoints, descriptors

def BF_matcher(desc1, desc2, norm_type, inliner_threshold = 0.8, cross_check=False):
    bf = cv2.BFMatcher(norm_type, crossCheck=cross_check)
    matches = bf.match(desc1, desc2)
    
    # Filter good matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = int(len(matches) * inliner_threshold)
    matches = matches[:num_good_matches]
    
    return matches


def filter_rectified_stereo_matches(stereo_matches, left_feat, right_feat, threshold=2.0):
    """
    Filter stereo matches for rectified images using horizontal disparity constraint.
    
    Args:
    - stereo_matches: List of cv2.DMatch objects (left_desc to right_desc).
    - left_feat: List of cv2.KeyPoint from left image.
    - right_feat: List of cv2.KeyPoint from right image.
    - threshold: Disparity threshold to consider a match valid (in pixels).
    
    Returns:
    - filtered_matches: List of cv2.DMatch objects that satisfy the disparity constraint.
    """
    filtered_matches = []
    
    y_left = np.array([left_feat[m.queryIdx].pt[1] for m in stereo_matches])
    y_right = np.array([right_feat[m.trainIdx].pt[1] for m in stereo_matches])
    
    dist = np.abs(y_left - y_right)
    
    idx = np.where(dist < threshold)[0]
    filtered_matches = [stereo_matches[i] for i in idx]
    
    return filtered_matches
        

def filter_temporal_and_stereo_features(temporal_matches, stereo_matches_prev, left_feat_prev, right_feat_prev, left_feat_curr):
    """
    Filters temporal matches to those with corresponding stereo matches in the previous frame.
    
    Args:
    - temporal_matches: List of cv2.DMatch objects (left_prev_desc to left_curr_desc).
    - stereo_matches_prev: List of cv2.DMatch objects (left_prev_desc to right_prev_desc).
    - left_feat_prev: List of cv2.KeyPoint from left_prev.
    - right_feat_prev: List of cv2.KeyPoint from right_prev.
    - left_feat_curr: List of cv2.KeyPoint from left_curr.
    
    Returns:
    - np.array of shape (N, 2): Points from left_prev for triangulation.
    - np.array of shape (N, 2): Corresponding points from right_prev for triangulation.
    - np.array of shape (N, 2): Corresponding 2D points from left_curr for PnP.
    """
    # Map left_prev indices to right_prev indices via stereo matches
    stereo_map = {m.queryIdx: m.trainIdx for m in stereo_matches_prev}
    
    # Filter temporal matches where left_prev (queryIdx) has a stereo match
    filtered_temporal = [m for m in temporal_matches if m.queryIdx in stereo_map]
    
    if not filtered_temporal:
        raise ValueError("No valid matches found after filtering.")
    
    # Extract indices as NumPy arrays
    query_indices = np.array([m.queryIdx for m in filtered_temporal])
    train_indices = np.array([m.trainIdx for m in filtered_temporal])
    right_indices = np.array([stereo_map[q] for q in query_indices])
    
    # Pre-extract all points as NumPy arrays (dtype float32 for cv2 compatibility)
    left_prev_all_pts = np.array([kp.pt for kp in left_feat_prev], dtype=np.float32)
    right_prev_all_pts = np.array([kp.pt for kp in right_feat_prev], dtype=np.float32)
    left_curr_all_pts = np.array([kp.pt for kp in left_feat_curr], dtype=np.float32)
    
    # Select points via vectorized indexing
    left_prev_pts = left_prev_all_pts[query_indices]
    right_prev_pts = right_prev_all_pts[right_indices]
    left_curr_pts = left_curr_all_pts[train_indices]
    
    return left_prev_pts, right_prev_pts, left_curr_pts
