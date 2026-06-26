"""Kappa angle estimation — offset between optical and visual axes."""
from __future__ import annotations

from typing import List, Tuple

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


def fit_eyeball_radius(
    samples: List[CalibrationSample],
    default_radius: float = 12.0,
) -> float:
    """Fit the eyeball radius from calibration data.

    Uses the relationship between iris radius (pixels) and gaze angle:
    as gaze angle increases, the apparent iris radius decreases.
    This is a simple average-based estimate; for production use a proper
    regression.

    Args:
        samples:        Calibration samples.
        default_radius: Return this if estimation fails (12 mm is typical).

    Returns:
        Estimated eyeball radius in mm.
    """
    radii = []
    for s in samples:
        lr = s.features.get("left_iris_radius", 0.0)
        rr = s.features.get("right_iris_radius", 0.0)
        # Only count when iris is visible (non-zero normalised radius)
        if lr > 1e-4:
            radii.append(lr)
        if rr > 1e-4:
            radii.append(rr)

    if not radii:
        return default_radius

    # Return the 25th percentile (near-frontal samples where iris is fullest)
    return float(np.percentile(radii, 25)) * 1000.0 * default_radius
