"""Tests for the EyeTrax-style landmark features and the ridge gaze model."""

from __future__ import annotations

import numpy as np
import pytest

from gaze_estimation.model.landmark_features import (
    LANDMARK_FEATURE_DIM,
    extract_landmark_features,
)
from gaze_estimation.model.ridge import RidgeGazeModel


def _face(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.3, 0.7, size=(478, 3))
    pts[33] = [0.40, 0.45, 0.0]  # outer eye corners
    pts[263] = [0.60, 0.45, 0.0]
    pts[10] = [0.50, 0.25, 0.0]  # top of the face
    return pts


def test_feature_vector_has_the_expected_size():
    feats = extract_landmark_features(_face())
    assert feats is not None
    assert feats.shape == (LANDMARK_FEATURE_DIM,)


def test_features_ignore_where_the_face_is_and_its_size():
    """Moving or scaling the face in the image must not change the features."""
    pts = _face()
    moved = (pts - 0.5) * 1.7 + np.array([0.6, 0.55, 0.1])

    np.testing.assert_allclose(
        extract_landmark_features(moved), extract_landmark_features(pts), atol=1e-5
    )


def test_features_need_the_iris_landmarks():
    assert extract_landmark_features(_face()[:468]) is None


def test_ridge_recovers_a_linear_mapping():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 12))
    W = rng.normal(size=(12, 2)) * 0.05
    Y_norm = 0.5 + X @ W
    model = RidgeGazeModel(alpha=1e-3).fit(X, Y_norm * [2560, 1440], 2560, 1440)

    np.testing.assert_allclose(model.predict(X), Y_norm, atol=1e-3)


def test_ridge_round_trips_and_refits_with_added_clicks(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 6))
    Y = rng.uniform(0, 1000, size=(50, 2))
    model = RidgeGazeModel().fit(X, Y, 1000, 1000)
    path = str(tmp_path / "ridge_model.npz")

    model.add_samples(X[0], [[10.0, 20.0]])
    model.refit()
    model.save(path)
    loaded = RidgeGazeModel.load(path)

    assert loaded.n_samples == 51
    np.testing.assert_allclose(loaded.predict(X[:3]), model.predict(X[:3]), atol=1e-9)


def test_untrained_ridge_refuses_to_predict():
    with pytest.raises(RuntimeError):
        RidgeGazeModel().predict(np.zeros((1, 3)))
