"""Tests for primary-monitor resolution detection.

``screeninfo`` is optional, so every branch is exercised by substituting the
module in ``sys.modules`` — including its absence, which is what a headless
machine (and this project's CI) looks like.
"""

from __future__ import annotations

import logging
import sys
import types

from gaze_estimation.utils.screen import DEFAULT_RESOLUTION, detect_screen_resolution


def _fake_screeninfo(monitors):
    """A stand-in screeninfo module exposing only ``get_monitors()``."""
    module = types.ModuleType("screeninfo")
    module.get_monitors = lambda: monitors
    return module


def _monitor(width, height):
    return types.SimpleNamespace(width=width, height=height)


def test_uses_the_first_monitor_resolution(monkeypatch):
    monkeypatch.setitem(sys.modules, "screeninfo", _fake_screeninfo([_monitor(2560, 1440)]))
    assert detect_screen_resolution() == (2560, 1440)


def test_additional_monitors_are_ignored(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "screeninfo",
        _fake_screeninfo([_monitor(2560, 1440), _monitor(800, 600)]),
    )
    assert detect_screen_resolution() == (2560, 1440)


def test_no_monitors_detected_returns_the_default(monkeypatch):
    monkeypatch.setitem(sys.modules, "screeninfo", _fake_screeninfo([]))
    assert detect_screen_resolution() == DEFAULT_RESOLUTION


def test_explicit_fallback_is_honoured(monkeypatch):
    monkeypatch.setitem(sys.modules, "screeninfo", _fake_screeninfo([]))
    assert detect_screen_resolution(fallback=(800, 600)) == (800, 600)


def test_missing_screeninfo_falls_back(monkeypatch):
    # A None entry makes ``import screeninfo`` raise ImportError.
    monkeypatch.setitem(sys.modules, "screeninfo", None)
    assert detect_screen_resolution() == DEFAULT_RESOLUTION


def test_detection_error_falls_back(monkeypatch):
    module = types.ModuleType("screeninfo")

    def _boom():
        raise RuntimeError("no DISPLAY")

    module.get_monitors = _boom
    monkeypatch.setitem(sys.modules, "screeninfo", module)
    assert detect_screen_resolution() == DEFAULT_RESOLUTION


def test_dimensions_are_coerced_to_int(monkeypatch):
    monkeypatch.setitem(sys.modules, "screeninfo", _fake_screeninfo([_monitor(1920.7, 1080.2)]))
    width, height = detect_screen_resolution()
    assert (width, height) == (1920, 1080)
    assert isinstance(width, int) and isinstance(height, int)


def test_fallback_is_logged(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "screeninfo", None)
    with caplog.at_level(logging.WARNING, logger="gaze_estimation.utils.screen"):
        detect_screen_resolution()
    assert "Screen detection unavailable" in caplog.text
