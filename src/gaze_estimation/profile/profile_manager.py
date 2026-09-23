"""ProfileManager: load/save/list user profiles for gaze calibration."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from gaze_estimation.profile.schema import UserProfile
from gaze_estimation.utils.logging import get_logger

_logger = get_logger("profile.manager")


class ProfileManager:
    """Manage per-user calibration profiles stored as JSON + binary files.

    Directory layout::

        profiles/
        └── <user_id>/
            ├── profile.json          # UserProfile metadata
            ├── mlp_weights.pt        # PyTorch weights
            ├── mlp_model.onnx        # ONNX export (optional)
            └── bias_map.npz          # BiasMap (optional)

    Args:
        profiles_dir: Root directory for all user profiles.
    """

    def __init__(self, profiles_dir: str = "profiles") -> None:
        self._root = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

    # ── Profile CRUD ──────────────────────────────────────────────────────

    def create_profile(
        self,
        user_id: str,
        screen_resolution: tuple,
        camera_name: str,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> str:
        """Create a new profile directory and return its path."""
        profile_dir = self._profile_dir(user_id)
        os.makedirs(profile_dir, exist_ok=True)
        _logger.info("Created profile directory: %s", profile_dir)
        return profile_dir

    def save_calibration(
        self,
        user_id: str,
        kappa_yaw: float,
        kappa_pitch: float,
        eyeball_radius: float,
        mlp_path: str,
        screen_resolution: tuple,
        camera_name: str,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        calibration_samples: int,
        onnx_path: Optional[str] = None,
        bias_map_path: Optional[str] = None,
        mlp_config: Optional[dict] = None,
        grid: tuple = (5, 5),
    ) -> str:
        """Persist calibration results for *user_id*.

        Returns the profile JSON path.
        """
        profile_dir = self._profile_dir(user_id)
        os.makedirs(profile_dir, exist_ok=True)

        intrinsics_list = (
            camera_matrix.tolist()
            if camera_matrix is not None
            else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        dist_list = dist_coeffs.tolist() if dist_coeffs is not None else [0.0] * 5

        profile = UserProfile(
            user_id=user_id,
            screen_resolution=list(screen_resolution),
            camera_name=camera_name,
            camera_intrinsics=intrinsics_list,
            dist_coeffs=dist_list,
            calibration_samples=calibration_samples,
            mlp_weights_path=mlp_path,
            onnx_model_path=onnx_path,
            kappa_yaw=kappa_yaw,
            kappa_pitch=kappa_pitch,
            eyeball_radius=eyeball_radius,
            bias_map_path=bias_map_path,
            last_session=datetime.now(timezone.utc).isoformat(),
            calibration_grid=list(grid),
            mlp_config=mlp_config or {},
        )

        json_path = os.path.join(profile_dir, "profile.json")
        with open(json_path, "w") as f:
            f.write(profile.model_dump_json(indent=2))

        _logger.info("Saved profile for user '%s' to %s", user_id, json_path)
        return json_path

    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load and return the UserProfile for *user_id*, or None if not found."""
        json_path = os.path.join(self._profile_dir(user_id), "profile.json")
        if not os.path.exists(json_path):
            return None
        with open(json_path) as f:
            data = json.load(f)
        return UserProfile(**data)

    def list_profiles(self) -> List[str]:
        """Return a list of all user IDs that have a saved profile."""
        if not os.path.isdir(self._root):
            return []
        return [
            name
            for name in os.listdir(self._root)
            if os.path.isdir(self._profile_dir(name))
            and os.path.exists(os.path.join(self._profile_dir(name), "profile.json"))
        ]

    def delete_profile(self, user_id: str) -> None:
        """Delete a user's profile directory."""
        import shutil

        profile_dir = self._profile_dir(user_id)
        if os.path.isdir(profile_dir):
            shutil.rmtree(profile_dir)
            _logger.info("Deleted profile: %s", user_id)

    # ── Recalibration checks ──────────────────────────────────────────────

    def needs_recalibration(
        self,
        user_id: str,
        current_camera: str,
        current_screen: tuple,
    ) -> bool:
        """Return True if a full recalibration is needed.

        Triggered when camera or screen resolution has changed since last calibration.
        """
        profile = self.load_profile(user_id)
        if profile is None:
            return True
        if profile.camera_name != current_camera:
            return True
        return list(profile.screen_resolution) != list(current_screen)

    def needs_quick_calibration(
        self,
        user_id: str,
        days_since_last: int = 7,
    ) -> bool:
        """Return True if a 5-point quick recalibration is recommended.

        Triggered after *days_since_last* days without recalibration.
        """
        profile = self.load_profile(user_id)
        if profile is None:
            return True
        try:
            last = datetime.fromisoformat(profile.last_session)
            now = datetime.now(timezone.utc)
            if last.tzinfo is None:
                return True
            delta = now - last
            return delta.days >= days_since_last
        except Exception:
            return True

    def get_profile_weights_path(self, user_id: str) -> str:
        """Return the default MLP weights path for a user."""
        return os.path.join(self._profile_dir(user_id), "mlp_weights.pt")

    def get_profile_onnx_path(self, user_id: str) -> str:
        """Return the default ONNX model path for a user."""
        return os.path.join(self._profile_dir(user_id), "mlp_model.onnx")

    def get_profile_bias_path(self, user_id: str) -> str:
        """Return the default bias map path for a user."""
        return os.path.join(self._profile_dir(user_id), "bias_map.npz")

    # ── Private ───────────────────────────────────────────────────────────

    def _profile_dir(self, user_id: str) -> str:
        return os.path.join(self._root, user_id)
