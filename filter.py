import numpy as np
import cv2
import bisect

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

import cv2
import numpy as np
from collections import defaultdict

def cluster_keypoints(keypoints, descriptors, image, grid_size, max_per_grid=3):
    """
    Downsamples ORB keypoints to max_per_grid per cell by response, ensuring uniform distribution.
    Optionally filters close keypoints by dist_threshold before gridding.
    
    Args:
    - keypoints: List of cv2.KeyPoint (ORB).
    - descriptors: np.array (num_kp, 32) for ORB binary.
    - image: cv2 image (for shape).
    - grid_size: Tuple (cell_height, cell_width) in pixels.
    - max_per_grid: Max keypoints per grid cell.
    
    Returns:
    - selected_kps: List of cv2.KeyPoint.
    - selected_descs: np.array of descriptors.
    """
    if len(keypoints) == 0:
        return [], np.array([])
    
    height, width = image.shape[:2]
    
    # Extract positions
    positions = np.array([kp.pt for kp in keypoints])
    
    # Compute grid indices (row = y // cell_height, col = x // cell_width)
    rows = np.clip(positions[:, 1] // grid_size[0], 0, (height // grid_size[0]) - 1)
    cols = np.clip(positions[:, 0] // grid_size[1], 0, (width // grid_size[1]) - 1)
    
    # Group by grid cell
    grid_groups = defaultdict(list)
    for idx, (row, col) in enumerate(zip(rows, cols)):
        grid_groups[(row, col)].append(idx)
    
    # Select top max_per_grid by response per cell
    selected_indices = []
    for cell_indices in grid_groups.values():
        if len(cell_indices) > max_per_grid:
            sorted_indices = sorted(cell_indices, key=lambda i: keypoints[i].response, reverse=True)
            selected_indices.extend(sorted_indices[:max_per_grid])
        else:
            selected_indices.extend(cell_indices)
    
    # Extract selected
    selected_kps = [keypoints[i] for i in selected_indices]
    selected_descs = descriptors[selected_indices]
    
    return selected_kps, selected_descs

# def rectified_stereo_matcher(feat_left, desc_left, feat_right,desc_right, cross_check=False, pixel_threshold=2.0,
#                                 max_hamming_distance = 30, ratio_test= 0.7, max_disparity=64.0
#                              ):
#     """
#         Match features between two rectified stereo images with epipolar constraint.
#         Args:
#             - feat_left: Keypoints from the left image.
#             - feat_right: Keypoints from the right image.
#             - desc1: Descriptors from the left image.
#             - desc2: Descriptors from the right image.
#             - cross_check: Whether to use cross-checking.
#             - pixel_threshold: Maximum vertical pixel difference to consider a match valid.
#             - max_hamming_distance: Maximum Hamming distance for ORB descriptors.
#             - ratio_test: Lowe's ratio test threshold.
#             - max_disparity: Maximum allowed disparity (horizontal pixel difference).
#         Returns:
#             - matches: List of cv2.DMatch objects representing the good matches.
#     """
#     # Handle empty feature sets
#     if len(feat_left) == 0 or len(feat_right) == 0:
#         return []
    
#     # Sort right features by their y-coordinates
#     y_right = np.array([kp.pt[1] for kp in feat_right])
#     sorted_indices = np.argsort(y_right)
#     feat_right_sorted = [feat_right[i] for i in sorted_indices]
#     desc_right_sorted = desc_right[sorted_indices]
    
#     # Find matches between left and right descriptors
#     l2r_matches = []
#     for i, kp_left in enumerate(feat_left):
#         y_left = kp_left.pt[1]
        
#         row_start = y_left - pixel_threshold
#         row_end = y_left + pixel_threshold
        
#         # Find features in right image within the vertical threshold using binary search
#         indicies_mask = np.zeros(len(feat_right_sorted), dtype=bool)
#         indicies_mask[row_start:row_end + 1] = 1
        
#         valid_right_desc = desc_right_sorted[indicies_mask]
#         if len(valid_right_desc) == 0:
#             continue
        
#         matches = []
#         # Calculate Hamming distance between left descriptor and valid right descriptors
#         for j in range(len(valid_right_desc)):
#             distance = cv2.norm(desc_left[i], valid_right_desc[j], cv2.NORM_HAMMING, mask=None)
#             if distance <= max_hamming_distance:
#                 matches.append((j, distance))
#         # Apply lowe's ratio test (1-NN vs 2-NN)
#         matches = sorted(matches, key=lambda x: x[1])
#         if len(matches) >= 2:
#             if matches[0][1] < ratio_test * matches[1][1]:
#                 l2r_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=matches[0][0], _imgIdx=0, _distance=matches[0][1]))
#         else:
#             l2r_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=matches[0][0], _imgIdx=0, _distance=matches[0][1]))

#     # Filter matches based on disparity constraint
#     filtered_matches = []
#     for m in l2r_matches:
#         x_left = feat_left[m.queryIdx].pt[0]
#         x_right = feat_right_sorted[m.trainIdx].pt[0]
#         disparity = x_left - x_right
#         if 0 < disparity < max_disparity:
#             filtered_matches.append(m)
    
#     if cross_check != True:
#         return filtered_matches
    
#     # Cross-checking: ensure matches are mutual
#     r2l_matches = []
#     for j, kp_right in enumerate(feat_right):
#         y_right = kp_right.pt[1]
        
#         row_start = y_right - pixel_threshold
#         row_end = y_right + pixel_threshold
        
#         # Find features in left image within the vertical threshold using binary search
#         indicies_mask = np.zeros(len(feat_left), dtype=bool)
#         indicies_mask[row_start:row_end + 1] = 1
        
#         valid_left_desc = desc_left[indicies_mask]
#         if len(valid_left_desc) == 0:
#             continue
        
#         matches = []
#         # Calculate Hamming distance between right descriptor and valid left descriptors
#         for i in range(len(valid_left_desc)):
#             distance = cv2.norm(desc_right[j], valid_left_desc[i], cv2.NORM_HAMMING, mask=None)
#             if distance <= max_hamming_distance:
#                 matches.append((i, distance))
#         # Apply lowe's ratio test (1-NN vs 2-NN)
#         matches = sorted(matches, key=lambda x: x[1])
#         if len(matches) >= 2:
#             if matches[0][1] < ratio_test * matches[1][1]:
#                 r2l_matches.append(cv2.DMatch(_queryIdx=j, _trainIdx=matches[0][0], _imgIdx=0, _distance=matches[0][1]))
#         else:
#             r2l_matches.append(cv2.DMatch(_queryIdx=j, _trainIdx=matches[0][0], _imgIdx=0, _distance=matches[0][1]))
            
#     # Filter r2l matches based on disparity constraint
#     filtered_r2l_matches = []
#     for m in r2l_matches:
#         x_right = feat_right[m.queryIdx].pt[0]
#         x_left = feat_left[m.trainIdx].pt[0]
#         disparity = x_left - x_right
#         if 0 < disparity < max_disparity:
#             filtered_r2l_matches.append(m)
            
#     # Mutual matches
#     mutual_matches = []
#     for m1 in filtered_matches:
#         for m2 in filtered_r2l_matches:
#             if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
#                 mutual_matches.append(m1)
#                 break
#     return mutual_matches

def rectified_stereo_matcher(feat_left, desc_left, feat_right, desc_right, cross_check=False, pixel_threshold=2.0,
                             max_hamming_distance=30, ratio_test=0.7, max_disparity=64.0, min_disparity=1.0):
    """
    Match features between two rectified stereo images with epipolar constraint.
    Args:
        - feat_left: List of cv2.KeyPoint from the left image.
        - desc_left: np.array (n_left, 32) of uint8 ORB descriptors from left.
        - feat_right: List of cv2.KeyPoint from the right image.
        - desc_right: np.array (n_right, 32) of uint8 ORB descriptors from right.
        - cross_check: Whether to use cross-checking (mutual best matches).
        - pixel_threshold: Maximum vertical pixel difference for epipolar search.
        - max_hamming_distance: Maximum Hamming distance for ORB descriptors.
        - ratio_test: Lowe's ratio test threshold (best / second_best < threshold).
        - max_disparity: Maximum allowed disparity (u_left - u_right).
        - min_disparity: Minimum allowed disparity to avoid noise/infinite depth.
    Returns:
        - matches: List of cv2.DMatch objects representing the good matches.
    """
    if len(feat_left) == 0 or len(feat_right) == 0:
        return []

    # Pre-sort right keypoints by y for l2r
    y_right = np.array([kp.pt[1] for kp in feat_right])
    idx_right_sorted = np.argsort(y_right)
    y_right_sorted = y_right[idx_right_sorted]
    feat_right_sorted = np.array(feat_right)[idx_right_sorted]
    desc_right_sorted = desc_right[idx_right_sorted]

    # Left-to-right matches
    l2r_matches = []
    for query_idx, kp_query in enumerate(feat_left):
        desc_query = desc_left[query_idx].reshape(1, -1)  # (1, 32)
        row = kp_query.pt[1]
        low = bisect.bisect_left(y_right_sorted, row - pixel_threshold)
        high = bisect.bisect_right(y_right_sorted, row + pixel_threshold)
        if low == high:
            continue

        desc_train_subset = desc_right_sorted[low:high]  # (c, 32)
        feat_train_subset = feat_right_sorted[low:high]

        # Vectorized Hamming distance
        xor = np.bitwise_xor(desc_query, desc_train_subset)  # (c, 32)
        bits = np.unpackbits(xor, axis=1)  # (c, 256)
        dists = np.sum(bits, axis=1).astype(float)  # (c,)

        if len(dists) == 0:
            continue

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist > max_hamming_distance:
            continue

        # Ratio test if >=2 candidates
        if len(dists) >= 2:
            dists_sorted = np.partition(dists, 1)
            second_dist = dists_sorted[1]
            if min_dist / second_dist >= ratio_test:
                continue
        # For len==1, no ratio, but already checked max_dist

        # Disparity check
        u_query = kp_query.pt[0]
        u_train = feat_train_subset[min_idx].pt[0]
        disparity = u_query - u_train
        if disparity < min_disparity or disparity > max_disparity:
            continue

        # Match with original train idx
        train_idx_original = idx_right_sorted[low + min_idx]
        match = cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx_original, _distance=min_dist)
        l2r_matches.append(match)

    if not cross_check:
        return l2r_matches

    # Pre-sort left for r2l
    y_left = np.array([kp.pt[1] for kp in feat_left])
    idx_left_sorted = np.argsort(y_left)
    y_left_sorted = y_left[idx_left_sorted]
    feat_left_sorted = np.array(feat_left)[idx_left_sorted]
    desc_left_sorted = desc_left[idx_left_sorted]

    # Right-to-left matches
    r2l_matches = []
    for query_idx, kp_query in enumerate(feat_right):
        desc_query = desc_right[query_idx].reshape(1, -1)
        row = kp_query.pt[1]
        low = bisect.bisect_left(y_left_sorted, row - pixel_threshold)
        high = bisect.bisect_right(y_left_sorted, row + pixel_threshold)
        if low == high:
            continue

        desc_train_subset = desc_left_sorted[low:high]
        feat_train_subset = feat_left_sorted[low:high]

        xor = np.bitwise_xor(desc_query, desc_train_subset)
        bits = np.unpackbits(xor, axis=1)
        dists = np.sum(bits, axis=1).astype(float)

        if len(dists) == 0:
            continue

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist > max_hamming_distance:
            continue

        if len(dists) >= 2:
            dists_sorted = np.partition(dists, 1)
            second_dist = dists_sorted[1]
            if min_dist / second_dist >= ratio_test:
                continue

        u_query = kp_query.pt[0]
        u_train = feat_train_subset[min_idx].pt[0]
        disparity = u_train - u_query  # Reversed for r2l
        if disparity < min_disparity or disparity > max_disparity:
            continue

        # Match with original train idx (left)
        train_idx_original = idx_left_sorted[low + min_idx]
        match = cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx_original, _distance=min_dist)
        r2l_matches.append(match)

    # Mutual best matches (cross-check)
    r2l_backpointer = {m.queryIdx: m.trainIdx for m in r2l_matches}  # right -> left
    mutual_matches = [m for m in l2r_matches if m.trainIdx in r2l_backpointer and r2l_backpointer[m.trainIdx] == m.queryIdx]

    return mutual_matches


def BF_matcher(desc1, desc2, norm_type, inliner_threshold = 0.8, cross_check=False):
    """
        Brute-Force matcher with optional cross-checking and inlier filtering.
        Args:
            - desc1: Descriptors from the first image.
            - desc2: Descriptors from the second image.
            - norm_type: Norm type for matching (e.g., cv2.NORM_HAMMING for ORB, cv2.NORM_L2 for SIFT).
            - inliner_threshold: Fraction of top matches to keep as inliers.
            - cross_check: Whether to use cross-checking.
        Returns:
            - matches: List of cv2.DMatch objects representing the good matches.
    """
    # Defensive dtype casting based on norm type
    if norm_type in (cv2.NORM_HAMMING, cv2.NORM_HAMMING2):
        if desc1.dtype != np.uint8:
            desc1 = desc1.astype(np.uint8)
        if desc2.dtype != np.uint8:
            desc2 = desc2.astype(np.uint8)
    elif norm_type in (cv2.NORM_L2, cv2.NORM_L1):
        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)
    
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
