import numpy as np
import cv2
import bisect
from scipy.optimize import least_squares

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
            if min_dist / (second_dist + 1e-9) >= ratio_test:
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
            if min_dist / (second_dist + 1e-9) >= ratio_test:
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


def planar_2pt_pnp(points3d, points2d, camera_matrix):
    if points3d.shape[0] != 2 or points2d.shape[0] != 2:
        return False, None, None
    
    # Normalize current 2D to bearings (x2, y2 = u', v')
    K_inv = np.linalg.inv(camera_matrix)
    homog = np.hstack((points2d, np.ones((2,1))))
    normalized_homog = K_inv @ homog.T
    p2_norm = (normalized_homog[:2] / normalized_homog[2]).T  # (2,2) rows as points [u', v']
    
    # Normalize prev 3D to bearings (x1, y1 = X/Z, Y/Z)
    depths = points3d[:, 2]
    if np.any(depths <= 0):
        return False, None, None
    p1_norm = (points3d[:, :2] / depths[:, np.newaxis]).T  # (2,2) rows as points [u, v]
    
    # Non-linear residuals from slide (for i=0,1: -x1 sin θ1 + x2 sin θ2 + y1 cos θ1 - y2 cos θ2 = 0)
    def residuals(params):
        theta1, theta2 = params
        res = np.zeros(2)
        for i in range(2):
            x1, y1 = p1_norm[0, i], p1_norm[1, i]
            x2, y2 = p2_norm[0, i], p2_norm[1, i]
            res[i] = -x1 * np.sin(theta1) + x2 * np.sin(theta2) + y1 * np.cos(theta1) - y2 * np.cos(theta2)
        return res
    
    # Optimize with least_squares (Gauss-Newton variant)
    initials = [[0, 0], [np.pi/2, np.pi/2], [np.pi, np.pi], [3*np.pi/2, 3*np.pi/2]]  # For 4 ambiguities
    valid_solutions = []
    
    for init in initials:
        result = least_squares(residuals, init, method='lm', ftol=1e-8, xtol=1e-8)
        if result.success and result.cost < 1e-6:
            theta1, theta2 = result.x
            theta = theta2 - theta1
            s, c = np.sin(theta), np.cos(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            
            # Recover absolute t (stereo: avg over points from λ [u', v', 1] = R X + t)
            t_candidates = []
            for i in range(2):
                q_hom = np.append(p2_norm[:, i], 1)
                rx = R @ points3d[i]
                v_norm = q_hom[1]  # Normalized v'
                Y = points3d[i, 1]
                if abs(v_norm) < 1e-6:
                    break
                lambda_i = Y / v_norm
                if lambda_i <= 0:
                    break
                t_i = lambda_i * q_hom - rx
                t_candidates.append(t_i[[0,2]])  # tx, tz
            
            if len(t_candidates) != 2:
                continue
            
            t_mean = np.mean(t_candidates, axis=0)
            if np.var(t_candidates, axis=0).max() > 1e-3:
                continue
            
            t = np.array([t_mean[0], 0, t_mean[1]]).reshape(3,1)
            
            # Validate positive depths
            transformed = R @ points3d.T + t
            if np.all(transformed[2] > 0):
                valid_solutions.append((R, t))
    
    if not valid_solutions:
        return False, None, None
    
    # Take best (lowest cost, or first)
    R, t = valid_solutions[0]
    return True, R, t
    

def RANSAC_pnp_two_point(triangulated_points, image_points, camera_matrix, outlier_ratio=0.5, confidence=0.99, rep_error_threshold=2.0):
    iterations = int(np.ceil(np.log(1 - confidence) / np.log(1 - (1 - outlier_ratio)**2)))
    iterations = min(iterations, 1000)
    
    N = triangulated_points.shape[0]
    best_inliers = []
    best_rvec = None
    best_tvec = None
    current_outlier = outlier_ratio
    
    for i in range(iterations):
        sample_indices = np.random.choice(N, 2, replace=False)
        sample_3D = triangulated_points[sample_indices]
        sample_2D = image_points[sample_indices]
        
        # Degeneracy check
        if np.linalg.norm(sample_3D[0] - sample_3D[1]) < 1e-3:
            continue
        
        valid, R, t = planar_2pt_pnp(sample_3D, sample_2D, camera_matrix)
        if not valid:
            continue
        
        rvec, _ = cv2.Rodrigues(R)  # Convert to vector
        
        projected_points, _ = cv2.projectPoints(triangulated_points, rvec, t, camera_matrix, None)
        projected_points = projected_points.reshape(-1, 2)
        
        errors = np.linalg.norm(projected_points - image_points, axis=1)
        inliers = np.where(errors < rep_error_threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_rvec = rvec
            best_tvec = t
            
            # Adaptive iterations
            current_outlier = 1 - len(inliers) / N
            if current_outlier < 1e-3:
                break
            remaining_iters = int(np.ceil(np.log(1 - confidence) / np.log(1 - (1 - current_outlier)**2)))
            iterations = min(i + remaining_iters, 1000)
    
    if len(best_inliers) < 4:
        return None, None, []  # No good model
    
    # Refinement on inliers
    inlier_3D = triangulated_points[best_inliers]
    inlier_2D = image_points[best_inliers]
    _, refined_rvec, refined_tvec = cv2.solvePnP(inlier_3D, inlier_2D, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=best_rvec, tvec=best_tvec)
    
    # Project to planar 3DoF
    refined_tvec[1] = 0
    R_ref, _ = cv2.Rodrigues(refined_rvec)
    theta = np.arctan2(R_ref[0,2], R_ref[0,0])
    R_planar = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    refined_rvec, _ = cv2.Rodrigues(R_planar)
    
    return refined_rvec, refined_tvec, best_inliers


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
