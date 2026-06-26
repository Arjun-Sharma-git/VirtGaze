"""3D canonical face model constants for solvePnP head-pose estimation."""
from __future__ import annotations

import numpy as np

# ── Canonical 3D face model (head coordinate frame, mm) ──────────────────────
# Average human face landmark positions used as PnP reference points.
# Indexed to match LANDMARK_INDICES below.

CANONICAL_3D_POINTS = np.array(
    [
        [0.0,    0.0,    0.0],    # 0  Nose tip
        [0.0,  -63.6,  -12.5],   # 1  Chin
        [-43.3,  20.0,   -5.0],  # 2  Left eye outer corner
        [43.3,   20.0,   -5.0],  # 3  Right eye outer corner
        [-28.0, -28.0,  -10.0],  # 4  Left mouth corner
        [28.0,  -28.0,  -10.0],  # 5  Right mouth corner
    ],
    dtype=np.float64,
)

# Corresponding MediaPipe Face Mesh landmark indices (468-point model)
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Named mapping for documentation
SOLVEPNP_LANDMARKS = {
    "nose_tip":        1,
    "chin":          152,
    "left_eye_outer":  33,
    "right_eye_outer": 263,
    "left_mouth":      61,
    "right_mouth":    291,
}

# ── Eye landmarks (for EAR computation) ──────────────────────────────────────
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Inner eye corners (for pupil-to-corner vectors)
LEFT_EYE_INNER  = 133
LEFT_EYE_OUTER  = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# ── Iris landmark indices (in the 478-point MediaPipe model) ─────────────────
LEFT_IRIS_INDICES  = list(range(468, 473))   # 468-472
RIGHT_IRIS_INDICES = list(range(473, 478))   # 473-477
