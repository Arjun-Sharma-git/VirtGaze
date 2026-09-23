"""3D geometry helpers: vector math, Euler conversions, ray-plane intersection."""

from __future__ import annotations

import math

import numpy as np

# ── Rotation matrix utilities ────────────────────────────────────────────────


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to [yaw, pitch, roll] in degrees.

    Uses the ZYX decomposition ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, so the
    returned angles are rotations about the **Z, Y and X** axes respectively:

    - ``yaw``   — rotation about Z
    - ``pitch`` — rotation about Y
    - ``roll``  — rotation about X

    Note that OpenCV camera coordinates are X-right, Y-down, Z-forward, so a
    physical "head turning left/right" rotation lands in the ``pitch`` slot
    when this is applied to a solvePnP result, and a head *tilt* lands in
    ``yaw``.  Consumers needing physical head axes must map them explicitly.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        yaw = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        roll = 0.0
    return np.array([yaw, pitch, roll], dtype=np.float64)


def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Convert [yaw, pitch, roll] in degrees to a 3x3 rotation matrix (ZYX)."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    Rz = np.array(
        [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]], dtype=np.float64
    )
    Ry = np.array(
        [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]], dtype=np.float64
    )
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]], dtype=np.float64
    )
    return Rz @ Ry @ Rx


# ── Ray utilities ────────────────────────────────────────────────────────────


def ray_to_angles(direction: np.ndarray) -> tuple[float, float]:
    """Convert a 3D unit direction vector to (yaw, pitch) in degrees.

    Convention: yaw is horizontal rotation around y-axis,
                pitch is vertical rotation around x-axis.
    """
    d = direction / (np.linalg.norm(direction) + 1e-12)
    pitch = math.degrees(math.asin(np.clip(-d[1], -1.0, 1.0)))
    yaw = math.degrees(math.atan2(d[0], d[2]))
    return float(yaw), float(pitch)


def angles_to_ray(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Convert (yaw, pitch) in degrees to a unit 3D direction vector."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    x = math.sin(y) * math.cos(p)
    y_ = -math.sin(p)
    z = math.cos(y) * math.cos(p)
    return np.array([x, y_, z], dtype=np.float64)


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of *v* (safe — returns zero vector if norm ~ 0)."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


# ── Ray-plane intersection ───────────────────────────────────────────────────


def ray_plane_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
) -> np.ndarray | None:
    """Compute the intersection of a 3D ray with a plane.

    Returns the 3D intersection point, or None if the ray is parallel to the plane.
    """
    denom = np.dot(ray_direction, plane_normal)
    if abs(denom) < 1e-6:
        return None
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    return ray_origin + t * ray_direction


# ── Angular / metric conversions ─────────────────────────────────────────────


def screen_to_angles(
    screen_x: float,
    screen_y: float,
    screen_width: int,
    screen_height: int,
    distance_mm: float,
    mm_per_px: float = 0.3,
) -> tuple[float, float]:
    """Convert a screen pixel position to gaze angles (yaw, pitch) in degrees.

    Uses the same right/up-positive convention as :func:`ray_to_angles` and
    :func:`angles_to_ray`, so the two can be compared directly (as
    :func:`~gaze_estimation.gaze.kappa_compensation.estimate_kappa` does).
    Note that ``pitch`` is therefore *negative* for a target below the screen
    centre, even though the pixel Y coordinate grows downwards.

    Assumes the user is at *distance_mm* from the screen centre.
    *mm_per_px* is approximate (0.3 mm/px for a 24" 1080p display).
    """
    x_mm = (screen_x - screen_width / 2.0) * mm_per_px
    y_mm = (screen_y - screen_height / 2.0) * mm_per_px
    yaw = math.degrees(math.atan2(x_mm, distance_mm))
    pitch = -math.degrees(math.atan2(y_mm, distance_mm))
    return float(yaw), float(pitch)


def angular_error_deg(
    pred_yaw: float, pred_pitch: float, true_yaw: float, true_pitch: float
) -> float:
    """Compute the angular error between two gaze directions in degrees."""

    def to_vec(y_deg: float, p_deg: float) -> np.ndarray:
        y, p = math.radians(y_deg), math.radians(p_deg)
        return np.array(
            [
                math.cos(p) * math.cos(y),
                math.cos(p) * math.sin(y),
                math.sin(p),
            ]
        )

    pred_vec = to_vec(pred_yaw, pred_pitch)
    true_vec = to_vec(true_yaw, true_pitch)
    cos_angle = float(np.dot(pred_vec, true_vec))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


# ── Eye aspect ratio ─────────────────────────────────────────────────────────


def eye_aspect_ratio(landmarks: np.ndarray, eye_indices: list) -> float:
    """Compute eye aspect ratio (EAR) for blink / openness detection.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where p1..p6 are the 6 eye landmarks (outer/inner corners + lids).
    """
    p = landmarks[eye_indices]
    A = float(np.linalg.norm(p[1] - p[5]))
    B = float(np.linalg.norm(p[2] - p[4]))
    C = float(np.linalg.norm(p[0] - p[3]))
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


# ── Circle fitting ───────────────────────────────────────────────────────────


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle to a set of 2D points using algebraic least-squares.

    Returns (cx, cy, radius).
    """
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = result
    radius = math.sqrt(max(cx**2 + cy**2 + c, 0.0))
    return float(cx), float(cy), float(radius)
