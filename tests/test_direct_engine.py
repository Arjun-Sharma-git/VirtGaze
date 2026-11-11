"""Tests for the direct (single-loop) gaze engine and its calibration helpers."""

from __future__ import annotations

import numpy as np

from gaze_estimation.config.config import Config
from gaze_estimation.direct.calibration import grid_points
from gaze_estimation.direct.engine import DirectGazeEngine, DirectSample, eye_aspect_ratios


class _FixedRidge:
    def predict(self, X):
        return np.array([[0.25, 0.75]])


def _engine() -> DirectGazeEngine:
    import queue
    import threading

    from gaze_estimation.pipeline.pipeline import FilterStage

    eng = DirectGazeEngine(Config.default(), 2000, 1000)
    eng._filter = FilterStage(queue.Queue(), eng._filter_out, threading.Event(),
                              config=Config.default())
    return eng


def _sample(t: float, feats=np.zeros(10)) -> DirectSample:
    return DirectSample(t, np.zeros((4, 4, 3), np.uint8), feats, 0.3, 0.3, 1.0)


def test_prediction_is_scaled_to_the_screen_and_filtered():
    eng = _engine()
    eng.set_ridge_model(_FixedRidge())
    tap = eng.tap_predictions()

    for i in range(20):
        eng._publish(_sample(1.0 + i / 30))

    raw = tap.get_nowait()
    assert (raw.screen_x, raw.screen_y) == (500.0, 750.0)
    est = eng.get_latest_estimate()
    assert abs(est.screen_x - 500.0) < 1.0 and abs(est.screen_y - 750.0) < 1.0


def test_no_face_gives_no_estimate_but_updates_the_sample():
    eng = _engine()
    eng.set_ridge_model(_FixedRidge())
    sub = eng.subscribe_samples()

    eng._publish(_sample(1.0, feats=None))

    assert eng.get_latest_estimate() is None
    assert eng.get_latest_sample().landmark_features is None
    assert sub.get_nowait().timestamp == 1.0


def test_serpentine_grid_reaches_the_margins():
    pts = grid_points(3, 3, 1000, 500, 0.05)

    xs = [p[0] for p in pts]
    assert xs[:3] == sorted(xs[:3]) and xs[3:6] == sorted(xs[3:6], reverse=True)
    assert min(xs) == 50.0 and max(xs) == 950.0
    assert min(p[1] for p in pts) == 25.0


def test_eye_aspect_ratio_is_lid_gap_over_width():
    pts = np.zeros((478, 3))
    pts[33, :2], pts[133, :2], pts[159, :2], pts[145, :2] = (0, 0), (1, 0), (0.5, -0.15), (0.5, 0.15)
    pts[263, :2], pts[362, :2], pts[386, :2], pts[374, :2] = (3, 0), (2, 0), (2.5, -0.05), (2.5, 0.05)

    left, right = eye_aspect_ratios(pts)

    assert abs(left - 0.3) < 1e-9 and abs(right - 0.1) < 1e-9


def test_lissajous_covers_the_screen_inside_the_margin():
    from gaze_estimation.direct.calibration import lissajous_point

    pts = np.array([lissajous_point(t, 2000, 1000, 0.05, 30.0) for t in np.linspace(0, 30, 3000)])

    assert pts[:, 0].min() >= 100 - 1 and pts[:, 0].max() <= 1900 + 1
    assert pts[:, 1].min() >= 50 - 1 and pts[:, 1].max() <= 950 + 1
    # reaches near every edge
    assert pts[:, 0].min() < 120 and pts[:, 0].max() > 1880
    assert pts[:, 1].min() < 70 and pts[:, 1].max() > 930


def test_unknown_calibration_mode_is_rejected():
    import pytest

    from gaze_estimation.direct.calibration import calibrate

    with pytest.raises(ValueError):
        calibrate(object(), 100, 100, {}, mode="nope")


def test_camera_mode_is_saved_with_the_model_and_applied(tmp_path):
    from gaze_estimation.direct.engine import apply_camera_mode, camera_mode
    from gaze_estimation.model.ridge import RidgeGazeModel

    cfg = Config.default()
    apply_camera_mode(cfg, "1280x720")
    assert camera_mode(cfg) == (1280, 720, "MJPG")

    rng = np.random.default_rng(0)
    model = RidgeGazeModel().fit(rng.normal(size=(30, 4)), rng.uniform(0, 100, (30, 2)), 100, 100)
    model.camera = camera_mode(cfg)
    model.save(str(tmp_path / "m.npz"))
    loaded = RidgeGazeModel.load(str(tmp_path / "m.npz"))
    assert loaded.camera == (1280, 720, "MJPG")

    eng = DirectGazeEngine(Config.default(), 100, 100)
    eng.set_ridge_model(loaded)
    assert camera_mode(eng._config) == (1280, 720, "MJPG")
