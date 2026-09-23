"""Tests for the One Euro Filter."""

from __future__ import annotations

from gaze_estimation.filtering.one_euro import OneEuroFilter, OneEuroFilter2D


def test_one_euro_convergence():
    """Filter should converge toward a constant input."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    t = 0.0
    val = 100.0
    for i in range(200):
        t += 0.016
        out = f.filter(val, t)
    assert abs(out - val) < 5.0, f"Expected convergence near {val}, got {out}"


def test_one_euro_first_value():
    """The very first filtered value should equal the input."""
    f = OneEuroFilter()
    out = f.filter(42.0, 0.0)
    assert abs(out - 42.0) < 1e-9


def test_one_euro_reset():
    f = OneEuroFilter()
    f.filter(10.0, 0.0)
    f.filter(10.0, 0.016)
    f.reset()
    out = f.filter(50.0, 0.0)
    assert abs(out - 50.0) < 1e-9


def test_one_euro_2d():
    f = OneEuroFilter2D(min_cutoff=1.0, beta=0.01)
    t = 0.0
    for _ in range(100):
        t += 0.016
        x, y = f.filter(200.0, 300.0, t)
    assert abs(x - 200.0) < 20.0
    assert abs(y - 300.0) < 20.0


def test_one_euro_responsiveness_at_high_velocity():
    """At high velocity, filter should respond quickly (low smoothing)."""
    f = OneEuroFilter(min_cutoff=0.1, beta=1.0)
    t = 0.0
    out = f.filter(0.0, 0.0)
    for i in range(50):
        t += 0.016
        out = f.filter(float(i) * 10, t)
    # Output should be reasonably close to the last fast-moving value
    # (not heavily smoothed)
    assert out > 100.0
