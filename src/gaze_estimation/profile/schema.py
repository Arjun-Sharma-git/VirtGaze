"""Pydantic schema for user calibration profiles."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class UserProfile(BaseModel):
    """Persisted user profile including camera intrinsics and calibration results.

    Stored as JSON under ``profiles/<user_id>/profile.json``.
    """

    user_id: str
    screen_resolution: List[int]  # [width, height]
    camera_name: str
    camera_intrinsics: List[List[float]]  # 3×3 camera matrix as nested list
    dist_coeffs: List[float]  # 5 distortion coefficients
    calibration_samples: int  # Number of used calibration samples
    mlp_weights_path: str  # Path to .pt weight file
    onnx_model_path: Optional[str] = None  # Path to .onnx model (or None)
    kappa_yaw: float = 0.0
    kappa_pitch: float = 0.0
    eyeball_radius: float = 12.0
    bias_map_path: Optional[str] = None  # Path to .npz bias map
    last_session: str  # ISO 8601 datetime string
    calibration_grid: List[int] = [5, 5]  # [cols, rows]
    mlp_config: dict = {}  # input_dim, hidden_dims, …

    model_config = {"extra": "allow"}
