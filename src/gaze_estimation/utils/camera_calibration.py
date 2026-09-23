"""Camera intrinsics utilities: estimation from frame size and chessboard calibration."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ── Quick estimation ──────────────────────────────────────────────────────────

def estimate_camera_matrix(width: int, height: int) -> np.ndarray:
    """Estimate a reasonable pinhole camera matrix from image dimensions.

    Uses the common approximation: focal_length ≈ max(width, height).
    This is good enough for an uncalibrated baseline.

    Returns a (3, 3) float64 camera matrix.
    """
    focal = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[focal, 0.0, cx],
         [0.0, focal, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def zero_dist_coeffs() -> np.ndarray:
    """Return zero distortion coefficients (5,)."""
    return np.zeros(5, dtype=np.float64)


# ── Chessboard calibration ────────────────────────────────────────────────────

def calibrate_from_images(
    images: list[np.ndarray],
    board_cols: int = 9,
    board_rows: int = 6,
    square_size_mm: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate camera from a list of chessboard images.

    Args:
        images:         List of BGR frames containing the chessboard.
        board_cols:     Number of *inner* corners along the long axis.
        board_rows:     Number of *inner* corners along the short axis.
        square_size_mm: Physical size of one chessboard square in mm.

    Returns:
        (camera_matrix, dist_coeffs, rms_reprojection_error)

    Raises:
        ValueError: If fewer than 5 valid frames are found.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern_size = (board_cols, board_rows)

    # 3D object points in mm
    objp = np.zeros((board_cols * board_rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_shape: Optional[tuple] = None

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)

    if len(obj_points) < 5:
        raise ValueError(
            f"Only {len(obj_points)} valid chessboard frames found "
            "(need at least 5). Provide more diverse views."
        )

    assert img_shape is not None
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )
    return camera_matrix.astype(np.float64), dist_coeffs.flatten().astype(np.float64), float(rms)


# ── Undistort helpers ─────────────────────────────────────────────────────────

def precompute_undistort_maps(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute undistortion maps for fast per-frame remap.

    Returns (map1, map2) suitable for cv2.remap.
    """
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix,
        (width, height), cv2.CV_32FC1,
    )
    return map1, map2


def undistort_frame(
    frame: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
) -> np.ndarray:
    """Apply pre-computed undistortion maps to a frame."""
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def undistort_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Undistort a set of 2D pixel points.

    Args:
        points: (N, 2) float32/float64 array of pixel coordinates.

    Returns:
        (N, 2) undistorted pixel coordinates.
    """
    pts = points.astype(np.float64).reshape(-1, 1, 2)
    undis = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    return undis.reshape(-1, 2)


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_intrinsics(
    path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    """Save camera intrinsics to a NumPy .npz file."""
    np.savez(path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


def load_intrinsics(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics from a NumPy .npz file.

    Returns (camera_matrix, dist_coeffs).
    """
    data = np.load(path)
    return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)
