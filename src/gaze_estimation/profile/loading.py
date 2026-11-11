"""Attach a user's saved calibration (models, intrinsics, filters) to a pipeline.

Shared by the scripts that run the pipeline outside ``run_tracker.py``
(filter tuning, the accuracy check), so they use the user's models exactly the
way the tracker does.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from gaze_estimation.config.config import Config
from gaze_estimation.profile.profile_manager import ProfileManager


def apply_user_filters(config: Config, profile_mgr: ProfileManager, user: str) -> bool:
    """Apply the user's tuned filter settings to *config* (before ``start()``)."""
    from gaze_estimation.filtering.tuning import apply_filter_params

    return bool(apply_filter_params(config, profile_mgr.get_profile_filters_path(user)))


def attach_user_model(pipeline, config: Config, profile_mgr: ProfileManager, user: str) -> str:
    """Load the user's gaze model into *pipeline*; return ``"ridge"`` or ``"mlp"``.

    Mirrors the tracker: the landmark ridge model when the profile has one
    (unless ``inference.model: mlp``), otherwise the MLP.  Also applies the
    stored camera intrinsics and kappa.  Raises ``RuntimeError`` when the user
    has no calibration.
    """
    profile = profile_mgr.load_profile(user)
    if profile is None:
        raise RuntimeError(
            f"No calibration for '{user}'. Run scripts/run_calibration.py --user {user}"
        )
    if profile.camera_intrinsics:
        pipeline.set_camera_intrinsics(
            np.array(profile.camera_intrinsics, dtype=np.float64),
            np.array(profile.dist_coeffs, dtype=np.float64),
        )
    pipeline.update_kappa(profile.kappa_yaw, profile.kappa_pitch)

    ridge_path = profile_mgr.get_profile_ridge_path(user)
    want_ridge = str(config.get("inference.model", "ridge")).lower() != "mlp"
    if want_ridge and os.path.exists(ridge_path):
        from gaze_estimation.model.ridge import RidgeGazeModel

        pipeline.set_ridge_model(
            RidgeGazeModel.load(ridge_path, alpha=float(config.get("inference.ridge_alpha", 100.0)))
        )
        return "ridge"

    weights: Optional[str] = profile.mlp_weights_path
    if weights and os.path.exists(weights):
        from gaze_estimation.model.trainer import MLPTrainer

        model, trainer = MLPTrainer().load(weights, config=config)
        pipeline.set_model(model, trainer)
        return "mlp"
    raise RuntimeError("The profile has no trained model. Run scripts/run_calibration.py first.")
