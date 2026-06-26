"""Feature extraction: builds the 34-dimensional FEATURE_KEYS vector."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gaze_estimation.pipeline.schemas import (
    FEATURE_KEYS, GazeRay, HeadPose, empty_features,
)
from gaze_estimation.pose.canonical_face import (
    LEFT_EYE_INDICES, LEFT_EYE_INNER, LEFT_EYE_OUTER,
    RIGHT_EYE_INDICES, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
)
from gaze_estimation.utils.geometry import eye_aspect_ratio, ray_to_angles


class FeatureExtractor:
    """Stateless feature extractor — builds one feature dict per frame."""

    def extract(
        self,
        mesh_468: np.ndarray,
        left_iris_center: Optional[Tuple],
        right_iris_center: Optional[Tuple],
        left_iris_radius: Optional[float],
        right_iris_radius: Optional[float],
        head_pose: Optional[HeadPose],
        gaze_ray_left: Optional[GazeRay],
        gaze_ray_right: Optional[GazeRay],
        frame_shape: Tuple,           # (H, W, C)
        face_bbox: Optional[Tuple],   # (x, y, w, h)
        confidence: float,
    ) -> dict:
        """Return a feature dict keyed by FEATURE_KEYS.

        All missing / unavailable values are filled with 0.0.
        """
        feats = empty_features()
        h, w = frame_shape[:2]

        # ----- 3D gaze angles -----------------------------------------------
        if gaze_ray_left is not None:
            yaw_l, pitch_l = ray_to_angles(gaze_ray_left.direction)
            feats["gaze_yaw_left"] = float(yaw_l)
            feats["gaze_pitch_left"] = float(pitch_l)
        if gaze_ray_right is not None:
            yaw_r, pitch_r = ray_to_angles(gaze_ray_right.direction)
            feats["gaze_yaw_right"] = float(yaw_r)
            feats["gaze_pitch_right"] = float(pitch_r)

        yaws = [feats["gaze_yaw_left"], feats["gaze_yaw_right"]]
        pitches = [feats["gaze_pitch_left"], feats["gaze_pitch_right"]]
        non_zero_yaws = [v for v in yaws if v != 0.0]
        non_zero_pitches = [v for v in pitches if v != 0.0]
        feats["gaze_yaw_avg"] = float(np.mean(non_zero_yaws)) if non_zero_yaws else 0.0
        feats["gaze_pitch_avg"] = float(np.mean(non_zero_pitches)) if non_zero_pitches else 0.0

        # ----- Head pose -----------------------------------------------------
        if head_pose is not None:
            euler = head_pose.euler_angles
            feats["head_yaw"] = float(euler[0])
            feats["head_pitch"] = float(euler[1])
            feats["head_roll"] = float(euler[2])
            feats["head_tx"] = float(head_pose.tvec[0])
            feats["head_ty"] = float(head_pose.tvec[1])
            feats["head_tz"] = float(head_pose.tvec[2])

        # ----- Iris 2D positions (normalised) --------------------------------
        if left_iris_center is not None:
            feats["left_iris_x"] = float(left_iris_center[0]) / max(w, 1)
            feats["left_iris_y"] = float(left_iris_center[1]) / max(h, 1)
        if right_iris_center is not None:
            feats["right_iris_x"] = float(right_iris_center[0]) / max(w, 1)
            feats["right_iris_y"] = float(right_iris_center[1]) / max(h, 1)

        # ----- Iris radii ----------------------------------------------------
        if left_iris_radius is not None:
            feats["left_iris_radius"] = float(left_iris_radius) / max(w, 1)
        if right_iris_radius is not None:
            feats["right_iris_radius"] = float(right_iris_radius) / max(w, 1)

        # ----- Eye aspect ratio ----------------------------------------------
        if mesh_468 is not None and len(mesh_468) >= 468:
            feats["left_ear"] = eye_aspect_ratio(mesh_468, LEFT_EYE_INDICES)
            feats["right_ear"] = eye_aspect_ratio(mesh_468, RIGHT_EYE_INDICES)

            # Pupil-to-eye-corner vectors
            if left_iris_center is not None:
                lc = np.array(left_iris_center, dtype=np.float32)
                inner = mesh_468[LEFT_EYE_INNER]
                outer = mesh_468[LEFT_EYE_OUTER]
                feats["left_pupil_to_inner_x"] = float(lc[0] - inner[0]) / max(w, 1)
                feats["left_pupil_to_inner_y"] = float(lc[1] - inner[1]) / max(h, 1)
                feats["left_pupil_to_outer_x"] = float(lc[0] - outer[0]) / max(w, 1)
                feats["left_pupil_to_outer_y"] = float(lc[1] - outer[1]) / max(h, 1)

            if right_iris_center is not None:
                rc = np.array(right_iris_center, dtype=np.float32)
                inner = mesh_468[RIGHT_EYE_INNER]
                outer = mesh_468[RIGHT_EYE_OUTER]
                feats["right_pupil_to_inner_x"] = float(rc[0] - inner[0]) / max(w, 1)
                feats["right_pupil_to_inner_y"] = float(rc[1] - inner[1]) / max(h, 1)
                feats["right_pupil_to_outer_x"] = float(rc[0] - outer[0]) / max(w, 1)
                feats["right_pupil_to_outer_y"] = float(rc[1] - outer[1]) / max(h, 1)

        # ----- Face bounding box (normalised) --------------------------------
        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            feats["face_x"] = float(fx) / max(w, 1)
            feats["face_y"] = float(fy) / max(h, 1)
            feats["face_width"] = float(fw) / max(w, 1)
            feats["face_height"] = float(fh) / max(h, 1)

        # ----- Interpupil distance -------------------------------------------
        if left_iris_center is not None and right_iris_center is not None:
            lc = np.array(left_iris_center, dtype=np.float32)
            rc = np.array(right_iris_center, dtype=np.float32)
            feats["interpupil_distance"] = float(np.linalg.norm(lc - rc)) / max(w, 1)

        # ----- Landmark confidence -------------------------------------------
        feats["landmark_confidence"] = float(confidence)

        return feats

    def to_vector(self, features: dict) -> np.ndarray:
        """Convert a feature dict to a (34,) float32 numpy array."""
        return np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)
