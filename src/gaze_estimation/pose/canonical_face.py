"""3D canonical face model constants for solvePnP head-pose estimation."""

from __future__ import annotations

import numpy as np

# ── Canonical 3D face model (head coordinate frame, mm) ──────────────────────
# Average human face landmark positions used as PnP reference points.
# Indexed to match LANDMARK_INDICES below.
#
# The head frame uses the OpenCV camera convention: X towards the image right,
# Y down, Z away from the camera (the nose is the point nearest the camera).
# A face looking straight at the camera therefore has R = I, small Euler
# angles and tz > 0.  The values below were first written Y-up / Z-towards-
# the-viewer, which made a frontal face R = Rx(180 deg): head roll and the
# head-frame gaze yaw then sat on the +/-180 deg seam and flipped sign from
# frame to frame, which wrecked the calibrated model.
_CANONICAL_Y_UP = np.array(
    [
        [0.0, 0.0, 0.0],  # 0  Nose tip
        [0.0, -63.6, -12.5],  # 1  Chin
        [-43.3, 20.0, -5.0],  # 2  Left eye outer corner
        [43.3, 20.0, -5.0],  # 3  Right eye outer corner
        [-28.0, -28.0, -10.0],  # 4  Left mouth corner
        [28.0, -28.0, -10.0],  # 5  Right mouth corner
    ],
    dtype=np.float64,
)
CANONICAL_3D_POINTS = _CANONICAL_Y_UP * np.array([1.0, -1.0, -1.0])

# A frontal face about 60 cm from the camera: the solvePnP starting point when
# there is no previous pose, and the reset after an implausible solution.
DEFAULT_RVEC = np.zeros(3, dtype=np.float64)
DEFAULT_TVEC = np.array([0.0, 0.0, 600.0], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices (468-point model)
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Named mapping for documentation
SOLVEPNP_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# ── Eye landmarks (for EAR computation) ──────────────────────────────────────
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Inner eye corners (for pupil-to-corner vectors)
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# ── Iris landmark indices (in the 478-point MediaPipe model) ─────────────────
LEFT_IRIS_INDICES = list(range(468, 473))  # 468-472
RIGHT_IRIS_INDICES = list(range(473, 478))  # 473-477
