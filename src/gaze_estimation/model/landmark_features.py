"""Face-aligned eye-region landmark features (the EyeTrax approach).

Adapted from EyeTrax (https://github.com/ck-zhang/EyeTrax, MIT licence,
``eyetrax/gaze.py`` and ``eyetrax/constants.py``).

Instead of reconstructing a 3D eye model, the 478 MediaPipe landmarks
(normalised x, y, z) are expressed in a face-aligned frame:

* origin at the midpoint of the two outer eye corners (33, 263),
* X along the eye-corner line, Y towards the top of the head (10),
* scaled by the inter-eye distance,

and the eye-region subset (brows, lids, iris, eye surround) plus a few stable
face points is flattened, followed by the face frame's yaw / pitch / roll.
A linear model (ridge regression) on these features maps gaze to the screen
well, and the alignment makes them largely independent of where the face is
in the image.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# fmt: off
LEFT_EYE_INDICES = [
    107, 66, 105, 63, 70,
    55, 65, 52, 53, 46,
    468, 469, 470, 471, 472,
    133, 33,
    173, 157, 158, 159, 160, 161, 246,
    155, 154, 153, 145, 144, 163, 7,
    243, 190, 56, 28, 27, 29, 30, 247,
    130, 25, 110, 24, 23, 22, 26, 112,
    244, 189, 221, 222, 223, 224, 225, 113,
    226, 31, 228, 229, 230, 231, 232, 233,
    193, 245, 128, 121, 120, 119, 118, 117,
    111, 35, 124, 143, 156,
]

RIGHT_EYE_INDICES = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276,
    473, 476, 475, 474, 477,
    362, 263,
    398, 384, 385, 386, 387, 388, 466,
    382, 381, 380, 374, 373, 390, 249,
    463, 414, 286, 258, 257, 259, 260, 467,
    359, 255, 339, 254, 253, 252, 256, 341,
    464, 413, 441, 442, 443, 444, 445, 342,
    446, 261, 448, 449, 450, 451, 452, 453,
    417, 465, 357, 350, 349, 348, 347, 346,
    340, 265, 353, 372, 383,
]

MUTUAL_INDICES = [
    4,    # nose
    10,   # top of the face
    151,  # forehead
    9,    # between the brows
    152,  # chin
    234,  # far left
    454,  # far right
    58,   # left jaw
    288,  # right jaw
]
# fmt: on

SUBSET_INDICES = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
LANDMARK_FEATURE_DIM = len(SUBSET_INDICES) * 3 + 3

# Eye region only: iris points plus the eye corners and lid contours.  Brows,
# cheeks and jaw mostly carry head movement and expression; dropping them cut
# leave-one-target-out error from ~112 px to ~94 px on a 25-dot calibration.
EYE_REGION_LANDMARKS = (
    list(range(468, 478))
    + [33, 133, 159, 145, 160, 158, 153, 144]
    + [263, 362, 386, 374, 385, 387, 380, 373]
)


def landmark_columns(subset: str = "eye") -> Optional[list[int]]:
    """Feature-vector columns for *subset* ("eye" or "full"); None means all."""
    if str(subset).lower() == "full":
        return None
    cols: list[int] = []
    for i, lm in enumerate(SUBSET_INDICES):
        if lm in EYE_REGION_LANDMARKS:
            cols.extend((3 * i, 3 * i + 1, 3 * i + 2))
    n = LANDMARK_FEATURE_DIM
    cols.extend((n - 3, n - 2, n - 1))  # face yaw / pitch / roll
    return cols


_OUTER_LEFT = 33
_OUTER_RIGHT = 263
_TOP_OF_HEAD = 10


def extract_landmark_features(points: np.ndarray) -> Optional[np.ndarray]:
    """Return the face-aligned feature vector for one face, or ``None``.

    Args:
        points: ``(N, 3)`` MediaPipe landmarks (normalised x, y and z) with
                ``N >= 478`` (the iris points are required).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] <= max(SUBSET_INDICES):
        return None

    left = pts[_OUTER_LEFT]
    right = pts[_OUTER_RIGHT]
    centre = (left + right) / 2.0

    x_axis = right - left
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        return None
    x_axis /= x_norm
    y_axis = pts[_TOP_OF_HEAD] - centre
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-9:
        return None
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))

    aligned = (pts - centre) @ R  # == (R.T @ (p - centre).T).T
    # Both outer corners lie on the X axis, so their distance is x_norm.
    aligned /= x_norm

    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])

    return np.concatenate([aligned[SUBSET_INDICES].ravel(), [yaw, pitch, roll]]).astype(
        np.float32
    )
