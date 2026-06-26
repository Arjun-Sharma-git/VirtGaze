"""Central data-packet and schema definitions shared across all pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ── Feature vector ────────────────────────────────────────────────────────────

FEATURE_KEYS: list[str] = [
    # 3D gaze (geometric)
    "gaze_yaw_left", "gaze_pitch_left",
    "gaze_yaw_right", "gaze_pitch_right",
    "gaze_yaw_avg", "gaze_pitch_avg",
    # Head pose
    "head_yaw", "head_pitch", "head_roll",
    "head_tx", "head_ty", "head_tz",
    # Iris 2D positions (normalised to [0,1])
    "left_iris_x", "left_iris_y",
    "right_iris_x", "right_iris_y",
    "left_iris_radius", "right_iris_radius",
    # Eye aspect ratios
    "left_ear", "right_ear",
    # Pupil-to-eye-corner vectors
    "left_pupil_to_inner_x", "left_pupil_to_inner_y",
    "left_pupil_to_outer_x", "left_pupil_to_outer_y",
    "right_pupil_to_inner_x", "right_pupil_to_inner_y",
    "right_pupil_to_outer_x", "right_pupil_to_outer_y",
    # Face position (normalised)
    "face_x", "face_y", "face_width", "face_height",
    # Interpupil distance (pixels)
    "interpupil_distance",
    # Confidence
    "landmark_confidence",
]
FEATURE_DIM: int = len(FEATURE_KEYS)  # 34


def empty_features() -> dict[str, float]:
    """Return a zeroed feature dict with all FEATURE_KEYS."""
    return {k: 0.0 for k in FEATURE_KEYS}


# ── Gaze state ────────────────────────────────────────────────────────────────

class GazeState(str, Enum):
    FIXATION = "fixation"
    SACCADE = "saccade"
    BLINK = "blink"
    LOST = "lost"


# ── Pipeline packets ──────────────────────────────────────────────────────────

@dataclass
class FramePacket:
    """Raw camera frame."""
    timestamp: float           # Unix timestamp (seconds)
    frame: np.ndarray          # BGR image, shape (H, W, 3)
    frame_id: int              # Monotonic counter


@dataclass
class FacePacket:
    """Face detection result."""
    timestamp: float
    frame: np.ndarray
    face_bbox: Optional[tuple]         # (x, y, w, h) or None
    detection_confidence: float
    face_landmarks_2d: Optional[np.ndarray]  # (6, 2) solvePnP keypoints or None


@dataclass
class MeshPacket:
    """Face mesh + iris landmarks."""
    timestamp: float
    frame: np.ndarray
    face_bbox: Optional[tuple]             # (x, y, w, h) or None
    mesh_468: np.ndarray                   # (468, 2) pixel coords
    iris_478: Optional[np.ndarray]         # (478, 2) when iris refinement enabled
    left_iris_center: Optional[tuple]      # (x, y) pixels
    right_iris_center: Optional[tuple]     # (x, y) pixels
    left_iris_radius: Optional[float]
    right_iris_radius: Optional[float]
    confidence: float


@dataclass
class HeadPose:
    """Head pose in camera frame."""
    rvec: np.ndarray              # Rotation vector (3,)
    tvec: np.ndarray              # Translation vector (3,)
    euler_angles: np.ndarray      # [yaw, pitch, roll] degrees
    rotation_matrix: np.ndarray   # (3, 3)


@dataclass
class PosePacket:
    """Head pose estimation result."""
    timestamp: float
    frame: np.ndarray
    mesh_468: np.ndarray
    iris_478: Optional[np.ndarray]
    left_iris_center: Optional[tuple]
    right_iris_center: Optional[tuple]
    left_iris_radius: Optional[float]
    right_iris_radius: Optional[float]
    head_pose: Optional[HeadPose]
    confidence: float


@dataclass
class GazeRay:
    """3D gaze ray in camera frame."""
    origin: np.ndarray     # (3,) 3D point
    direction: np.ndarray  # (3,) unit vector


@dataclass
class GazePacket:
    """3D gaze geometry result + full feature vector."""
    timestamp: float
    gaze_ray_left: Optional[GazeRay]
    gaze_ray_right: Optional[GazeRay]
    gaze_yaw: float
    gaze_pitch: float
    features: dict           # FEATURE_KEYS -> float
    head_pose: Optional[HeadPose]
    confidence: float


@dataclass
class PredictionPacket:
    """MLP / geometric screen prediction."""
    timestamp: float
    screen_x: float          # pixels
    screen_y: float          # pixels
    confidence: float
    raw_features: dict
    source: str              # "mlp" | "geometric" | "hold"


@dataclass
class GazeEstimate:
    """Final filtered gaze estimate (output of pipeline)."""
    timestamp: float
    screen_x: float           # Filtered screen X (pixels)
    screen_y: float           # Filtered screen Y (pixels)
    raw_x: float              # Pre-filter X
    raw_y: float              # Pre-filter Y
    velocity: float           # px/s
    confidence: float
    fixation_state: GazeState
    fixation_duration: float  # seconds in current state
    source: str               # "mlp" | "geometric" | "hold"
    latency_ms: float         # End-to-end latency for this frame


# ── Calibration ───────────────────────────────────────────────────────────────

@dataclass
class CalibrationSample:
    """One gaze sample collected during calibration."""
    features: dict           # FEATURE_KEYS -> float
    screen_x: float          # Target screen X (pixels)
    screen_y: float          # Target screen Y (pixels)
    timestamp: float


@dataclass
class CalibrationResult:
    """Result of a full calibration session."""
    samples: list             # list[CalibrationSample]
    mlp_weights_path: Optional[str]
    kappa_yaw: float
    kappa_pitch: float
    eyeball_radius: float
    bias_map: Optional[np.ndarray]
    timestamp: float
    screen_resolution: tuple  # (width, height)


# ── Fixation ──────────────────────────────────────────────────────────────────

@dataclass
class FixationInfo:
    """Fixation / saccade / blink state information."""
    state: GazeState
    velocity: float                    # px/s
    duration: float                    # seconds in current state
    fixation_point: Optional[tuple]    # (x, y) centroid if fixating
