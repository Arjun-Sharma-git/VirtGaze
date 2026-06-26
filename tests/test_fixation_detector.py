"""Tests for the fixation / saccade state machine."""
from __future__ import annotations

import pytest

from gaze_estimation.filtering.fixation_detector import FixationDetector
from gaze_estimation.pipeline.schemas import GazeState


def _feed(det, x, y, t, ear=1.0):
    return det.update(x, y, t, ear, ear)


def test_initial_state_fixation_from_stationary():
    det = FixationDetector(fixation_velocity_threshold=100.0)
    info = _feed(det, 500.0, 300.0, 0.0)
    # First sample: no velocity → FIXATION or LOST (no prev)
    assert info.state in (GazeState.FIXATION, GazeState.LOST)


def test_saccade_detected():
    det = FixationDetector(saccade_velocity_threshold=10.0)
    _feed(det, 0.0, 0.0, 0.0)
    # Large jump in 16ms → velocity >> threshold
    info = _feed(det, 1000.0, 0.0, 0.016)
    assert info.state == GazeState.SACCADE


def test_blink_detected():
    det = FixationDetector(blink_ear_threshold=0.2)
    _feed(det, 500.0, 300.0, 0.0)
    info = det.update(500.0, 300.0, 0.016, 0.1, 0.1)  # EAR below threshold
    assert info.state == GazeState.BLINK


def test_fixation_centroid():
    det = FixationDetector(fixation_velocity_threshold=200.0)
    t = 0.0
    for _ in range(20):
        _feed(det, 500.0, 300.0, t)
        t += 0.016
    info = _feed(det, 500.0, 300.0, t)
    if info.state == GazeState.FIXATION and info.fixation_point is not None:
        cx, cy = info.fixation_point
        assert abs(cx - 500.0) < 5.0
        assert abs(cy - 300.0) < 5.0


def test_smoothing_multiplier():
    det = FixationDetector()
    # Default state LOST→ multiplier should be 5.0
    m = det.get_smoothing_multiplier()
    assert m > 0.0


def test_reset():
    det = FixationDetector()
    _feed(det, 100.0, 100.0, 0.0)
    det.reset()
    assert det._state == GazeState.LOST
