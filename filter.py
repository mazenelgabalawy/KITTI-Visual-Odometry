import numpy as np
import cv2
import bisect
from collections import defaultdict

def process_image(image, nfeatures, scale, nlevels, grid_size=(30,20), max_per_grid=5):
    """
        Process the image to detect and compute ORB features and descriptors,
        and cluster them to ensure uniform distribution across the image.
        
        Args:
            image (np.ndarray): Input image.
            nfeatures (int): Number of features to detect.
            scale (float): Scale factor for the ORB detector.
            nlevels (int): Number of pyramid levels for the ORB detector.
            grid_size (tuple): Size of the grid for clustering keypoints.
            max_per_grid (int): Maximum number of keypoints per grid cell.
        Returns:
            keypoints (list): List of detected keypoints.
            descriptors (np.ndarray): Corresponding descriptors for the keypoints.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor = scale, nlevels = nlevels)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    keypoints, descriptors = cluster_keypoints(keypoints, descriptors, image=image, grid_size=grid_size, max_per_grid=max_per_grid)
    return keypoints, descriptors

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
    
    # Compute grid indices (row = y // cell_height, col = x // cell_width). The number of cells
    # rounds up, so a partial row or column at the image border is a cell of its own
    n_rows = int(np.ceil(height / grid_size[0]))
    n_cols = int(np.ceil(width / grid_size[1]))
    rows = np.clip(positions[:, 1] // grid_size[0], 0, n_rows - 1)
    cols = np.clip(positions[:, 0] // grid_size[1], 0, n_cols - 1)
    
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

def rectified_stereo_matcher(feat_left, desc_left, feat_right, desc_right, cross_check=False, pixel_threshold=2.0,
                             max_hamming_distance=30, ratio_test=0.7, max_disparity=64.0, min_disparity=1.0):
    """
    Match features between two rectified stereo images with epipolar constraint.
    The candidates of a keypoint are the keypoints within the row band and the disparity range;
    the best match is the candidate with the smallest Hamming distance. With two or more candidates,
    the best must also pass the ratio test against the second-best candidate.
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

    # Precompute popcount LUT for fast Hamming distance
    popcount = np.array([bin(i).count('1') for i in range(256)], dtype=int)

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

        # Disparity check before ranking: only keypoints within the disparity range are candidates
        u_query = kp_query.pt[0]
        u_train = np.array([kp.pt[0] for kp in feat_train_subset])
        disparity = u_query - u_train
        candidates = np.flatnonzero((disparity >= min_disparity) & (disparity <= max_disparity))
        if len(candidates) == 0:
            continue

        # Vectorized Hamming distance using LUT
        xor = desc_query ^ desc_train_subset[candidates]  # (c, 32) uint8
        bit_counts = popcount[xor]  # (c, 32) int
        dists = np.sum(bit_counts, axis=1)  # (c,)

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist > max_hamming_distance:
            continue

        # Ratio test if >=2 candidates
        if len(dists) >= 2:
            dists_sorted = np.partition(dists, 1)
            second_dist = dists_sorted[1]
            if min_dist / (second_dist + 1e-9) >= ratio_test:
                continue
        # For len==1, no ratio, but already checked max_dist

        # Match with original train idx (fix: cast min_dist to float for cv2.DMatch)
        train_idx_original = idx_right_sorted[low + candidates[min_idx]]
        match = cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx_original, _distance=float(min_dist))
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

        # Disparity check before ranking (reversed for r2l)
        u_query = kp_query.pt[0]
        u_train = np.array([kp.pt[0] for kp in feat_train_subset])
        disparity = u_train - u_query
        candidates = np.flatnonzero((disparity >= min_disparity) & (disparity <= max_disparity))
        if len(candidates) == 0:
            continue

        # Vectorized Hamming distance using LUT (optimized)
        xor = desc_query ^ desc_train_subset[candidates]
        bit_counts = popcount[xor]
        dists = np.sum(bit_counts, axis=1)

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist > max_hamming_distance:
            continue

        if len(dists) >= 2:
            dists_sorted = np.partition(dists, 1)
            second_dist = dists_sorted[1]
            if min_dist / (second_dist + 1e-9) >= ratio_test:
                continue

        # Match with original train idx (left) (fix: cast min_dist to float for cv2.DMatch)
        train_idx_original = idx_left_sorted[low + candidates[min_idx]]
        match = cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx_original, _distance=float(min_dist))
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
    N is 0 when no temporal match has a stereo match.
    """
    # Map left_prev indices to right_prev indices via stereo matches
    stereo_map = {m.queryIdx: m.trainIdx for m in stereo_matches_prev}
    
    # Filter temporal matches where left_prev (queryIdx) has a stereo match
    filtered_temporal = [m for m in temporal_matches if m.queryIdx in stereo_map]
    
    if not filtered_temporal:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy(), empty.copy()
    
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


def filter_valid_landmarks(points_3D_hom, P0, P1, min_abs_w=1e-4):
    """
    Checks triangulated points before they are used for PnP.
    
    Args:
    - points_3D_hom: np.array of shape (4, N) from cv2.triangulatePoints. Each column has unit norm,
      so |w| is about 1/range (in 1/m) for points further than a few meters.
    - P0: Projection matrix of the left camera.
    - P1: Projection matrix of the right camera.
    - min_abs_w: Minimum |w|. 1e-4 rejects points beyond about 10 km (near-zero disparity).
    
    Returns:
    - valid_idx: np.array of shape (N,), True for points with finite coordinates, |w| >= min_abs_w,
      and positive depth in both cameras.
    - points_3D: np.array of shape (N, 3) of non-homogeneous points (nonfinite where w is 0).
    """
    w = points_3D_hom[3, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        points_3D = (points_3D_hom[:3, :] / w).T  # Convert to non-homogeneous coordinates
        
        # Depth of each point in the left and right camera
        points_3D_hom_normalized = np.hstack((points_3D, np.ones((len(points_3D), 1))))
        depth_left = points_3D_hom_normalized @ P0[2]
        depth_right = points_3D_hom_normalized @ P1[2]
        
        valid_idx = np.isfinite(points_3D_hom).all(axis=0) & np.isfinite(points_3D).all(axis=1)
        valid_idx = valid_idx & (w != 0) & (np.abs(w) >= min_abs_w)
        valid_idx = valid_idx & (depth_left > 0) & (depth_right > 0)
    
    return valid_idx, points_3D


def orthogonalize_to_planar_motion(R, alpha=0.8):
    """
        Orthogonalize rotation matrix R to planar motion by blending with optimal planar rotation.
        Args:
            - R: 3x3 rotation matrix from PnP.
            - alpha: Blending factor (0 < alpha <= 1) to control strength of planar constraint.
        Returns:
            - R_ortho: Adjusted rotation matrix constrained to planar motion.
    """
    
    # After getting R_curr from cv2.Rodrigues(rvec)
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt  # Closest orthonormal matrix
    if np.linalg.det(R_ortho) < 0:
        R_ortho = U @ np.diag([1, 1, -1]) @ Vt  # Ensure proper rotation (det=1)

    R_planar = R_ortho  # Replace original
    
    # Extract elements from R_ortho ( KITTI: x-right, y-down, z-forward; yaw around y)
    a = R_ortho[0,0] + R_ortho[2,2]  # For cos θ
    b = R_ortho[0,2] - R_ortho[2,0]  # For sin θ
    theta_opt = np.arctan2(b, a)  # Optimal yaw

    cos_theta = np.cos(theta_opt)
    sin_theta = np.sin(theta_opt)
    R_planar = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    # Blend with original to allow small pitch/roll (soft constraint: alpha=0.8 pulls toward planar)
    R_planar = alpha * R_planar + (1 - alpha) * R_ortho

    # Re-orthonormalize the blended R (optional but recommended)
    U, S, Vt = np.linalg.svd(R_planar)
    R_planar = U @ Vt
    if np.linalg.det(R_planar) < 0:
        R_planar = U @ np.diag([1, 1, -1]) @ Vt
        
    return R_planar
