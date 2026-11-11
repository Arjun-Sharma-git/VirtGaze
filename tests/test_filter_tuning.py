"""Tests for fitting the UKF / One Euro parameters to a recorded session."""

from __future__ import annotations

import json

import numpy as np

from gaze_estimation.config.config import Config
from gaze_estimation.filtering.tuning import (
    TuningSession,
    apply_filter_params,
    interpolate_targets,
    save_filter_params,
    score,
    segments_to_flags,
    tune,
)


def _session(seed: int = 0, noise_px: float = 60.0) -> TuningSession:
    """30 fps: 3 s holding still, then 6 s of motion, plus measurement noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 9.0, 1 / 30) + 100.0
    target = np.empty((len(t), 2))
    hold = t < 103.0
    target[hold] = (800.0, 500.0)
    s = t[~hold] - 103.0
    target[~hold, 0] = 800 + 600 * np.sin(s * 1.5)
    target[~hold, 1] = 500 + 300 * np.sin(s * 3.0)
    raw = target + rng.normal(0, noise_px, size=target.shape)
    moving, weight = segments_to_flags(t, [(100.0, 103.0, "hold"), (103.0, 109.0, "move")])
    return TuningSession(t, raw, target, moving, weight, np.full((len(t), 2), 0.3))


def test_tuning_does_not_do_worse_than_the_defaults():
    session = _session()
    base = Config.default()

    result = tune(session, base, n_random=15, n_refine=10, steadiness=3.0)

    def objective(err, jit):
        return err + 3.0 * jit

    assert objective(result.error_px, result.jitter_px) <= objective(
        result.baseline["error_px"], result.baseline["jitter_px"]
    ) + 1e-6
    assert result.error_px == score(session, _with(base, result.params))[0]


def test_steadiness_trades_lag_for_less_jitter():
    session = _session()
    base = Config.default()

    loose = tune(session, base, n_random=20, n_refine=10, steadiness=0.0)
    steady = tune(session, base, n_random=20, n_refine=10, steadiness=20.0)

    assert steady.jitter_px < loose.jitter_px


def test_stabilizer_holds_the_dot_inside_its_radius():
    from gaze_estimation.filtering.tuning import jitter, with_params

    session = _session()
    base = Config.default()
    stable = with_params(base, {})  # independent copy
    stable.set("filtering.stabilizer_radius_px", 80.0)

    assert jitter(session, stable) < jitter(session, base)


def _with(base, params):
    from gaze_estimation.filtering.tuning import with_params

    return with_params(base, params)


def test_saved_parameters_are_applied_to_a_config(tmp_path):
    session = _session()
    result = tune(session, Config.default(), n_random=5, n_refine=0)
    path = str(tmp_path / "filters.json")

    save_filter_params(path, result)
    cfg = Config.default()
    applied = apply_filter_params(cfg, path)

    assert set(applied) == set(result.params)
    assert cfg.get("filtering.one_euro.beta") == result.params["one_euro.beta"]
    assert "stats" in json.loads(open(path).read())


def test_missing_tuning_file_changes_nothing(tmp_path):
    cfg = Config.default()
    before = dict(cfg.section("filtering")["ukf"])

    assert apply_filter_params(cfg, str(tmp_path / "nope.json")) is None
    assert cfg.section("filtering")["ukf"] == before


def test_settle_time_after_a_jump_is_ignored():
    t = np.array([0.0, 0.1, 0.5, 1.0])
    moving, weight = segments_to_flags(t, [(0.0, 2.0, "move")], settle_sec=0.4)

    assert moving.all()
    assert weight.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_target_positions_are_interpolated_to_frame_times():
    xy = interpolate_targets([0.5], [0.0, 1.0], np.array([[0.0, 0.0], [100.0, 50.0]]))
    np.testing.assert_allclose(xy, [[50.0, 25.0]])
