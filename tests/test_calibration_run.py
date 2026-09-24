"""Tests for the full and quick calibration *sessions*.

Both `run` methods are interactive, but they only need a queue of gaze packets —
no display, no camera, no user.  A fake clock removes the per-target settle delay,
so a whole session runs in milliseconds and the collected samples, callbacks,
kappa, bias map, and saved weights are all checkable.
"""

from __future__ import annotations

import queue
import time as stdlib_time
import types

import numpy as np
import pytest

import gaze_estimation.calibration.calibration_engine as engine_module
import gaze_estimation.calibration.quick_calibration as quick_module
from gaze_estimation.calibration.calibration_engine import CalibrationEngine
from gaze_estimation.calibration.quick_calibration import QuickCalibration
from gaze_estimation.model.trainer import MLPTrainer
from gaze_estimation.pipeline.schemas import FEATURE_KEYS, GazePacket, empty_features

W, H = 1920, 1080


def _trainer() -> MLPTrainer:
    return MLPTrainer(epochs=1, batch_size=4, hidden_dims=[8, 8], device="cpu")


def _packet(confidence=0.9, yaw=1.0, pitch=0.5, seed=0) -> GazePacket:
    rng = np.random.default_rng(seed)
    features = empty_features()
    for key in ("gaze_yaw_avg", "gaze_yaw_left", "gaze_yaw_right"):
        features[key] = yaw
    for key in ("gaze_pitch_avg", "gaze_pitch_left", "gaze_pitch_right"):
        features[key] = pitch
    features["landmark_confidence"] = confidence
    for key in FEATURE_KEYS:
        if features[key] == 0.0:
            features[key] = float(rng.normal() * 0.1)
    return GazePacket(
        timestamp=1.0,
        gaze_ray_left=None,
        gaze_ray_right=None,
        gaze_yaw=yaw,
        gaze_pitch=pitch,
        features=features,
        head_pose=None,
        confidence=confidence,
    )


class _FakeTime(types.ModuleType):
    """Clock for the calibration engine: no real sleeping, advancing monotonic."""

    def __init__(self) -> None:
        super().__init__("fake_time")
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def monotonic(self) -> float:
        self.now += 0.01
        return self.now

    def time(self) -> float:
        return 1_700_000_000.0


@pytest.fixture
def fast_engine_time(monkeypatch):
    """Fake clock + no-op drain so a pre-filled queue survives each target."""
    fake = _FakeTime()
    monkeypatch.setattr(engine_module, "time", fake)
    monkeypatch.setattr(engine_module, "_drain", lambda q, max_drain=100: None)
    return fake


@pytest.fixture
def fast_quick_time(monkeypatch):
    """QuickCalibration imports ``time`` locally, so patch the stdlib module."""
    monkeypatch.setattr(stdlib_time, "sleep", lambda seconds: None)
    monkeypatch.setattr(quick_module, "_drain", lambda q, max_drain=100: None)


def _filled_queue(n: int, **kwargs) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    for i in range(n):
        q.put_nowait(_packet(seed=i, **kwargs))
    return q


# ── CalibrationEngine.run ─────────────────────────────────────────────────────


def _engine(**kwargs) -> CalibrationEngine:
    kwargs.setdefault("grid_cols", 2)
    kwargs.setdefault("grid_rows", 2)
    kwargs.setdefault("samples_per_target", 5)
    kwargs.setdefault("mlp_trainer", _trainer())
    return CalibrationEngine(W, H, **kwargs)


def test_full_session_collects_samples_from_every_target(fast_engine_time):
    engine = _engine()

    result = engine.run(_filled_queue(20))

    assert len(result.samples) == 20  # 2x2 grid x 5 samples
    assert result.screen_resolution == (W, H)
    assert result.timestamp == pytest.approx(1_700_000_000.0)


def test_full_session_reports_kappa_and_eyeball_radius(fast_engine_time):
    result = _engine().run(_filled_queue(20))

    assert np.isfinite(result.kappa_yaw) and np.isfinite(result.kappa_pitch)
    assert result.eyeball_radius > 0.0


def test_full_session_builds_a_bias_map(fast_engine_time):
    assert _engine().run(_filled_queue(20)).bias_map is not None


def test_full_session_reports_no_path_when_nothing_was_saved(fast_engine_time):
    assert _engine().run(_filled_queue(20)).mlp_weights_path is None


def test_full_session_saves_and_reports_the_model_path(fast_engine_time, tmp_path):
    dest = tmp_path / "models" / "gaze.npz"

    result = _engine().run(_filled_queue(20), mlp_save_path=str(dest))

    assert result.mlp_weights_path == str(dest)
    assert dest.is_file()


def test_full_session_does_not_claim_a_path_when_no_samples_arrived(fast_engine_time, tmp_path):
    """An empty session must not report weights that were never written."""
    engine = _engine(target_duration_sec=0.001)
    dest = tmp_path / "gaze.npz"

    result = engine.run(queue.Queue(), mlp_save_path=str(dest))

    assert result.samples == []
    assert result.mlp_weights_path is None
    assert not dest.exists()


def test_full_session_calls_on_target_change_for_each_target(fast_engine_time):
    engine = _engine()
    visited: list[tuple] = []

    engine.run(_filled_queue(20), on_target_change=lambda x, y: visited.append((x, y)))

    assert len(visited) == 4
    assert set(visited) == set(engine.generate_targets())


def test_full_session_reports_progress_up_to_one(fast_engine_time):
    fractions: list[float] = []

    _engine().run(_filled_queue(20), on_progress=fractions.append)

    assert fractions == [0.25, 0.5, 0.75, 1.0]


def test_full_session_waits_for_each_target_to_settle(fast_engine_time):
    _engine().run(_filled_queue(20))

    assert fast_engine_time.sleeps == [0.3, 0.3, 0.3, 0.3]


def test_full_session_skips_low_confidence_packets(fast_engine_time):
    q: queue.Queue = queue.Queue()
    for i in range(10):
        q.put_nowait(_packet(confidence=0.1, seed=i))  # below the 0.3 gate
    for i in range(20):
        q.put_nowait(_packet(confidence=0.9, seed=100 + i))

    result = _engine().run(q)

    assert len(result.samples) == 20


def test_full_session_uses_the_camera_matrix_when_supplied(fast_engine_time):
    camera_matrix = np.array([[800.0, 0.0, W / 2], [0.0, 800.0, H / 2], [0.0, 0.0, 1.0]])
    engine = _engine(camera_matrix=camera_matrix, frame_width=1280)

    assert engine.run(_filled_queue(20)).eyeball_radius > 0.0


# ── QuickCalibration.run ──────────────────────────────────────────────────────


def _quick(**kwargs) -> QuickCalibration:
    kwargs.setdefault("samples_per_target", 2)
    kwargs.setdefault("trainer", _trainer())
    return QuickCalibration(W, H, **kwargs)


def _mlp():
    from gaze_estimation.model.mlp import GazeMLP

    return GazeMLP(hidden_dims=[8, 8])


def test_quick_session_collects_samples(fast_quick_time):
    result = _quick().run(_filled_queue(10), existing_model=None)

    assert len(result.samples) == 10  # 5 targets x 2 samples
    assert result.screen_resolution == (W, H)


def test_quick_session_reports_kappa_and_default_radius(fast_quick_time):
    result = _quick().run(_filled_queue(10), existing_model=_mlp())

    assert np.isfinite(result.kappa_yaw) and np.isfinite(result.kappa_pitch)
    assert result.eyeball_radius == 12.0  # unchanged by a quick recalibration


def test_quick_session_builds_a_bias_map(fast_quick_time):
    assert _quick().run(_filled_queue(10), existing_model=_mlp()).bias_map is not None


def test_quick_session_saves_the_fine_tuned_model(fast_quick_time, tmp_path):
    dest = tmp_path / "quick.npz"

    result = _quick().run(_filled_queue(10), existing_model=None, mlp_save_path=str(dest))

    assert result.mlp_weights_path == str(dest)
    assert dest.is_file()


def test_quick_session_without_samples_returns_documented_defaults(fast_quick_time):
    result = _quick(target_duration_sec=0.001).run(queue.Queue(), existing_model=None)

    assert result.samples == []
    assert result.mlp_weights_path is None
    assert result.kappa_yaw == 0.0 and result.kappa_pitch == 0.0
    assert result.eyeball_radius == 12.0
    assert result.bias_map is None


def test_quick_session_calls_on_target_change_for_each_target(fast_quick_time):
    calibration = _quick()
    visited: list[tuple] = []

    calibration.run(
        _filled_queue(10), existing_model=None, on_target_change=lambda x, y: visited.append((x, y))
    )

    assert len(visited) == 5
    assert set(visited) == set(calibration.generate_targets())


def test_quick_session_reports_progress_up_to_one(fast_quick_time):
    fractions: list[float] = []

    _quick().run(_filled_queue(10), existing_model=None, on_progress=fractions.append)

    assert fractions == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_quick_session_skips_low_confidence_packets(fast_quick_time):
    q: queue.Queue = queue.Queue()
    for i in range(5):
        q.put_nowait(_packet(confidence=0.1, seed=i))
    for i in range(10):
        q.put_nowait(_packet(confidence=0.9, seed=50 + i))

    assert len(_quick().run(q, existing_model=None).samples) == 10


def test_quick_session_honours_the_configured_target_count(fast_quick_time):
    calibration = _quick(points=9, samples_per_target=1)

    result = calibration.run(_filled_queue(9), existing_model=None)

    assert len(result.samples) == 9  # 3x3 grid x 1 sample


def test_quick_session_fine_tunes_the_existing_model_in_place(fast_quick_time):
    from gaze_estimation.model.mlp import GazeMLP

    model = GazeMLP(hidden_dims=[8, 8])
    result = _quick().run(_filled_queue(10), existing_model=model)

    assert result.samples
    # The supplied instance is updated rather than replaced.
    assert isinstance(model, GazeMLP)


# ── Sampling helpers ──────────────────────────────────────────────────────────


def test_collect_samples_polls_an_empty_queue_until_the_deadline(fast_engine_time):
    """A starved queue yields fewer samples instead of aborting the session."""
    engine = _engine(samples_per_target=5, target_duration_sec=0.05)

    samples = engine._collect_samples(_filled_queue(2), 100.0, 100.0)

    assert len(samples) == 2
    assert all(s.screen_x == 100.0 and s.screen_y == 100.0 for s in samples)


def test_drain_empties_the_queue():
    from gaze_estimation.calibration.calibration_engine import _drain

    q = _filled_queue(3)

    _drain(q)

    assert q.empty()


def test_drain_is_safe_on_an_empty_queue():
    from gaze_estimation.calibration.calibration_engine import _drain

    q: queue.Queue = queue.Queue()

    _drain(q)  # hits the Empty break

    assert q.empty()


def test_drain_respects_the_cap():
    from gaze_estimation.calibration.calibration_engine import _drain

    q = _filled_queue(5)

    _drain(q, max_drain=2)

    assert q.qsize() == 3
