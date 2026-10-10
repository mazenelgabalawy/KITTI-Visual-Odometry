"""Pose / rotation transform helpers ported from utils.py and VO_base.py."""

from __future__ import annotations

import numpy as np


def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """Verbatim port of utils.rotationMatrixToEulerAngles (same gimbal-lock branch)."""
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


def invert_pose(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirrors VO_base.py L126-127: R_cam = R.T, t_cam = -R.T @ t."""
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def to_homogeneous(T34: np.ndarray) -> np.ndarray:
    """3x4 -> 4x4 by appending the [0, 0, 0, 1] row."""
    T34 = np.asarray(T34)
    bottom = np.array([[0, 0, 0, 1]], dtype=T34.dtype)
    return np.vstack((T34[:3, :4], bottom))


def compose_se3(C: np.ndarray, T: np.ndarray) -> np.ndarray:
    """C @ homogeneous(T). Mirrors VO_base.py L132: C_k @ vstack((T, [0,0,0,1]))."""
    T = np.asarray(T)
    if T.shape == (4, 4):
        T4 = T
    elif T.shape == (3, 4):
        T4 = to_homogeneous(T)
    else:
        raise ValueError(f"T must be 3x4 or 4x4, got shape {T.shape}")

    return C @ T4


def rotation_angle_between(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle in radians, in [0, pi], between rotation matrices Ra and Rb
    (broadcasts over leading dimensions).

    Uses ||Ra - Rb||_F = 2*sqrt(2)*sin(theta/2), which stays accurate for tiny
    angles where the trace/arccos form loses precision.
    """
    diff = np.asarray(Ra, dtype=float) - np.asarray(Rb, dtype=float)
    frob = np.sqrt(np.sum(diff * diff, axis=(-2, -1)))
    return 2.0 * np.arcsin(np.clip(frob / (2.0 * np.sqrt(2.0)), 0.0, 1.0))
