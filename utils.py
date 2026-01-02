import os
import cv2
import numpy as np

def load_stereo_pair(current_step, input_path):   
    left_image = cv2.imread(os.path.join(input_path, "image_0", f"{current_step:06d}.png"), cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(os.path.join(input_path, "image_1", f"{current_step:06d}.png"), cv2.IMREAD_GRAYSCALE)

    return left_image, right_image

def save_transformation_matrix(output_file, transformation_matrix: np.ndarray):
    # Ensure we have a numpy array and at least 3x4 content
    mat = np.asarray(transformation_matrix)

    # Take the top 3 rows and 4 columns (3x4), flatten row-major
    row = mat[:3, :4].reshape(-1)

    # Format as scientific notation with 6 decimals, space-separated
    line = ' '.join(f"{x:.6e}" for x in row)

    # Append to the file so this becomes the last line
    with open(output_file, 'a') as f:
        f.write(line + '\n')

def merge_images(left_prev, right_prev, left_curr, right_curr):
    top = np.hstack((left_prev, right_prev))
    bottom = np.hstack((left_curr, right_curr))
    merged = np.vstack((top, bottom))
    return merged

def plot_matches(img1, kp1, img2, kp2, matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
    