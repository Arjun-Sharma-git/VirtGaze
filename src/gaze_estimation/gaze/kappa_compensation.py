"""Kappa angle estimation — offset between optical and visual axes."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from gaze_estimation.pipeline.schemas import CalibrationSample
from gaze_estimation.utils.geometry import screen_to_angles


def estimate_kappa(
    samples: List[CalibrationSample],
    screen_width: int,
    screen_height: int,
    distance_mm: float = 600.0,
    mm_per_px: float = 0.3,
) -> Tuple[float, float]:
    """Estimate the kappa angle (optical–visual axis offset) from calibration samples.

    For each sample:
    1. Compute the *target* gaze angles from the screen position.
    2. Compare against the *measured* geometric (optical) gaze angles.
    3. The median difference is the kappa angle for that eye.

    Args:
        samples:       List of CalibrationSample with known screen targets.
        screen_width:  Display resolution width in pixels.
        screen_height: Display resolution height in pixels.
        distance_mm:   Approximate viewing distance (mm). Defaults to 600 mm (≈24").
        mm_per_px:     mm per pixel; default 0.3 mm/px for 1080p at 24".

    Returns:
        (kappa_yaw_deg, kappa_pitch_deg)  — median offset in degrees.
    """
    kappa_yaws: List[float] = []
    kappa_pitches: List[float] = []

    for s in samples:
        target_yaw, target_pitch = screen_to_angles(
            s.screen_x, s.screen_y,
            screen_width, screen_height,
            distance_mm, mm_per_px,
        )
        optical_yaw = s.features.get("gaze_yaw_avg", 0.0)
        optical_pitch = s.features.get("gaze_pitch_avg", 0.0)

        kappa_yaws.append(target_yaw - optical_yaw)
        kappa_pitches.append(target_pitch - optical_pitch)

    if not kappa_yaws:
        return 0.0, 0.0

    kappa_yaw = float(np.median(kappa_yaws))
    kappa_pitch = float(np.median(kappa_pitches))
    return kappa_yaw, kappa_pitch


# Anthropometric ratio: the eyeball radius is roughly twice the iris radius.
# Human means: iris radius ≈ 5.8 mm, eyeball radius ≈ 12 mm.
_IRIS_TO_EYEBALL_RATIO = 0.48

# Plausible range for a human eyeball radius (mm).  Estimates outside this
# range are rejected as unreliable.
_EYEBALL_RADIUS_RANGE = (8.0, 20.0)


def fit_eyeball_radius(
    samples: List[CalibrationSample],
    default_radius: float = 12.0,
    focal_length_px: Optional[float] = None,
    frame_width_px: Optional[int] = None,
) -> float:
    """Estimate the eyeball radius (mm) from calibration data.

    The iris radius stored in the feature vector is *normalised by the frame
    width*, so it is dimensionless.  Converting it to millimetres requires the
    camera focal length and the frame width:

        L_iris_mm = r_norm * frame_width_px * z_mm / f_px

    where ``z_mm`` is the eye-to-camera distance, taken from the head-pose
    translation (``head_tz``).  The eyeball radius then follows from the
    anthropometric ratio ``R_eyeball ≈ L_iris / 0.48``.

    If the intrinsics are not supplied, or too little valid data is available,
    *default_radius* is returned.  (The previous implementation multiplied a
    normalised pixel radius by 1000 × 12, which was dimensionally meaningless
    and produced garbage radii that were persisted into the user profile.)

    Args:
        samples:         Calibration samples.
        default_radius:  Fallback eyeball radius in mm (12 mm is typical).
        focal_length_px: Camera focal length in pixels (``camera_matrix[0, 0]``).
        frame_width_px:  Capture frame width in pixels.

    Returns:
        Estimated eyeball radius in mm.
    """
    if not samples or focal_length_px is None or frame_width_px is None:
        return float(default_radius)
    if focal_length_px <= 0.0 or frame_width_px <= 0:
        return float(default_radius)

    estimates: List[float] = []
    for s in samples:
        z_mm = float(s.features.get("head_tz", 0.0))
        if z_mm <= 1.0:
            continue
        for key in ("left_iris_radius", "right_iris_radius"):
            r_norm = float(s.features.get(key, 0.0))
            if r_norm <= 1e-4:
                continue
            iris_mm = r_norm * float(frame_width_px) * z_mm / float(focal_length_px)
            estimates.append(iris_mm / _IRIS_TO_EYEBALL_RATIO)

    if not estimates:
        return float(default_radius)

    radius = float(np.median(estimates))
    lo, hi = _EYEBALL_RADIUS_RANGE
    if not np.isfinite(radius) or not lo <= radius <= hi:
        return float(default_radius)
    return radius
