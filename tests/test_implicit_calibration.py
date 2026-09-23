"""Tests for click-based implicit calibration wiring."""
from __future__ import annotations

from gaze_estimation.calibration.implicit_calibration import ImplicitCalibration
from gaze_estimation.pipeline.schemas import FEATURE_KEYS


class _OnlineTrainer:
    """Minimal stand-in exposing the OnlineTrainer surface used by the hook."""

    def __init__(self) -> None:
        self.fed: list = []
        self.retrain_result = False
        self.model = object()

    def feed_click(self, features, x, y, mouse_velocity=0.0,
                   click_velocity_threshold=500.0) -> None:
        self.fed.append((x, y))

    def maybe_retrain(self) -> bool:
        return self.retrain_result


def _features() -> dict:
    return {k: 0.0 for k in FEATURE_KEYS}


def test_fast_click_is_skipped():
    trainer = _OnlineTrainer()
    implicit = ImplicitCalibration(trainer)

    assert implicit.on_click(_features(), 100.0, 100.0, mouse_velocity=900.0) is False
    assert trainer.fed == []
    assert implicit.skipped_clicks == 1
    assert implicit.total_clicks == 0


def test_slow_click_is_accepted():
    trainer = _OnlineTrainer()
    implicit = ImplicitCalibration(trainer)

    assert implicit.on_click(_features(), 100.0, 100.0, mouse_velocity=10.0) is True
    assert trainer.fed == [(100.0, 100.0)]
    assert implicit.total_clicks == 1


def test_retrain_callback_invoked():
    trainer = _OnlineTrainer()
    trainer.retrain_result = True
    calls: list = []
    implicit = ImplicitCalibration(trainer, on_retrain=lambda: calls.append(1))

    implicit.on_click(_features(), 10.0, 10.0, mouse_velocity=0.0)

    assert calls == [1]
    assert implicit.retrain_count == 1


def test_retrain_callback_failure_is_contained():
    def _boom() -> None:
        raise RuntimeError("callback exploded")

    trainer = _OnlineTrainer()
    trainer.retrain_result = True
    implicit = ImplicitCalibration(trainer, on_retrain=_boom)

    # Must not propagate out of on_click()
    assert implicit.on_click(_features(), 10.0, 10.0, mouse_velocity=0.0) is True


def test_velocity_is_estimated_from_click_history():
    trainer = _OnlineTrainer()
    implicit = ImplicitCalibration(trainer, click_velocity_threshold=1.0)

    implicit.on_click(_features(), 0.0, 0.0)          # establishes the baseline
    accepted = implicit.on_click(_features(), 1000.0, 0.0)   # huge jump -> skipped

    assert accepted is False
    assert len(trainer.fed) == 1
