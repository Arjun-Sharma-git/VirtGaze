"""Tests for FeatureExtractor: correct keys, no NaN, proper normalisation."""
from __future__ import annotations

import numpy as np
import pytest

from gaze_estimation.gaze.gaze_features import FeatureExtractor
from gaze_estimation.pipeline.schemas import FEATURE_KEYS, FEATURE_DIM


@pytest.fixture
def extractor():
    return FeatureExtractor()


def test_feature_keys_count():
    assert FEATURE_DIM == 34
    assert len(FEATURE_KEYS) == 34


def test_extract_all_keys_present(extractor, mock_mesh_468, mock_head_pose):
    feats = extractor.extract(
        mesh_468=mock_mesh_468,
        left_iris_center=(320.0, 240.0),
        right_iris_center=(290.0, 235.0),
        left_iris_radius=15.0,
        right_iris_radius=14.5,
        head_pose=mock_head_pose,
        gaze_ray_left=None,
        gaze_ray_right=None,
        frame_shape=(480, 640, 3),
        face_bbox=(200, 150, 240, 200),
        confidence=0.9,
    )
    for k in FEATURE_KEYS:
        assert k in feats, f"Missing key: {k}"


def test_extract_no_nan(extractor, mock_mesh_468, mock_head_pose):
    feats = extractor.extract(
        mesh_468=mock_mesh_468,
        left_iris_center=(320.0, 240.0),
        right_iris_center=(290.0, 235.0),
        left_iris_radius=15.0,
        right_iris_radius=14.5,
        head_pose=mock_head_pose,
        gaze_ray_left=None,
        gaze_ray_right=None,
        frame_shape=(480, 640, 3),
        face_bbox=None,
        confidence=0.8,
    )
    for k, v in feats.items():
        assert not np.isnan(v), f"NaN in feature: {k}"


def test_extract_confidence_stored(extractor, mock_mesh_468, mock_head_pose):
    feats = extractor.extract(
        mesh_468=mock_mesh_468,
        left_iris_center=None, right_iris_center=None,
        left_iris_radius=None, right_iris_radius=None,
        head_pose=None,
        gaze_ray_left=None, gaze_ray_right=None,
        frame_shape=(480, 640, 3),
        face_bbox=None,
        confidence=0.77,
    )
    assert abs(feats["landmark_confidence"] - 0.77) < 1e-6


def test_to_vector(extractor, mock_mesh_468, mock_head_pose):
    feats = extractor.extract(
        mesh_468=mock_mesh_468,
        left_iris_center=(320.0, 240.0),
        right_iris_center=(290.0, 235.0),
        left_iris_radius=15.0, right_iris_radius=14.5,
        head_pose=mock_head_pose,
        gaze_ray_left=None, gaze_ray_right=None,
        frame_shape=(480, 640, 3),
        face_bbox=None,
        confidence=1.0,
    )
    vec = extractor.to_vector(feats)
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
