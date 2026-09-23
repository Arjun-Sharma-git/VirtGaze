"""Tests for the BGR frame overlay renderer.

Assertions are made on actual pixel values of a real numpy frame, so a broken
colour, coordinate mapping, or guard is caught rather than merely "did not
raise".
"""

from __future__ import annotations

import numpy as np
import pytest

from gaze_estimation.pipeline.schemas import GazeEstimate, GazeState, MeshPacket
from gaze_estimation.visualization.overlay import OverlayRenderer

W, H = 640, 480


@pytest.fixture
def frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


def _estimate(screen_x=960.0, screen_y=540.0, confidence=0.9, state=GazeState.FIXATION):
    return GazeEstimate(
        timestamp=0.0,
        screen_x=screen_x,
        screen_y=screen_y,
        raw_x=screen_x,
        raw_y=screen_y,
        velocity=5.0,
        confidence=confidence,
        fixation_state=state,
        fixation_duration=0.3,
        source="mlp",
        latency_ms=12.0,
    )


def _mesh(frame, mesh_468=None, iris=(True, True)):
    left_ok, right_ok = iris
    return MeshPacket(
        timestamp=0.0,
        frame=frame,
        face_bbox=(100, 100, 200, 200),
        mesh_468=np.zeros((3, 2)) if mesh_468 is None else mesh_468,
        iris_478=None,
        left_iris_center=(200.0, 220.0) if left_ok else None,
        right_iris_center=(300.0, 220.0) if right_ok else None,
        left_iris_radius=8.0 if left_ok else None,
        right_iris_radius=8.0 if right_ok else None,
        confidence=0.9,
    )


def _plain():
    """A renderer that draws nothing unless asked to."""
    return OverlayRenderer(draw_mesh=False, draw_iris=False, draw_fps=False, draw_latency=False)


def _has_color(image, color) -> bool:
    return bool(np.any(np.all(image == np.array(color, dtype=np.uint8), axis=-1)))


def _pixels_changed(image) -> bool:
    return bool(np.any(image != 0))


def test_copy_mode_leaves_the_input_frame_untouched(frame):
    out = _plain().render(frame, _estimate())

    assert out is not frame
    assert not _pixels_changed(frame)
    assert _pixels_changed(out)


def test_in_place_mode_returns_the_same_array(frame):
    renderer = OverlayRenderer(copy=False, draw_mesh=False, draw_iris=False)

    out = renderer.render(frame, _estimate())

    assert out is frame


def test_nothing_is_drawn_without_an_estimate_or_mesh(frame):
    out = OverlayRenderer().render(frame)

    assert not _pixels_changed(out)


def test_gaze_dot_is_drawn_at_the_scaled_screen_position(frame):
    out = _plain().render(frame, _estimate(960.0, 540.0))

    # screen 960x540 of 1920x1080 → centre of a 640x480 frame.
    assert out[H // 2, W // 2].tolist() == list(OverlayRenderer.COLOR_GAZE)


def test_gaze_dot_colour_follows_the_fixation_state(frame):
    out = _plain().render(frame, _estimate(state=GazeState.SACCADE))

    assert _has_color(out, OverlayRenderer.COLOR_STATE[GazeState.SACCADE])
    assert _has_color(out, OverlayRenderer.COLOR_GAZE)  # filled centre


def test_low_confidence_draws_no_gaze_dot(frame):
    """The dot is suppressed below 0.3 confidence; the HUD still reports it."""
    out = _plain().render(frame, _estimate(confidence=0.2))

    assert not _has_color(out, OverlayRenderer.COLOR_GAZE)
    assert not _has_color(out, OverlayRenderer.COLOR_STATE[GazeState.FIXATION])
    assert _pixels_changed(out)  # confidence/state text is unconditional


def test_offscreen_gaze_is_clipped_into_the_frame(frame):
    out = _plain().render(frame, _estimate(screen_x=99999.0, screen_y=540.0))

    assert out[H // 2, W - 1].tolist() == list(OverlayRenderer.COLOR_GAZE)


def test_mesh_is_drawn_only_when_enabled(frame):
    points = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])

    quiet = _plain().render(frame, None, mesh_packet=_mesh(frame, mesh_468=points))
    loud = OverlayRenderer(draw_mesh=True, draw_iris=False, draw_fps=False).render(
        frame, None, mesh_packet=_mesh(frame, mesh_468=points)
    )

    assert not _pixels_changed(quiet)
    assert _has_color(loud, OverlayRenderer.COLOR_MESH)


def test_mesh_without_landmarks_is_skipped(frame):
    packet = _mesh(frame)
    packet.mesh_468 = None

    out = OverlayRenderer(draw_mesh=True, draw_iris=False, draw_fps=False).render(
        frame, None, mesh_packet=packet
    )

    assert not _pixels_changed(out)


def test_iris_circles_are_drawn_when_present(frame):
    out = OverlayRenderer(draw_mesh=False, draw_fps=False).render(
        frame, None, mesh_packet=_mesh(frame)
    )

    assert _has_color(out, OverlayRenderer.COLOR_IRIS)


def test_iris_circles_are_skipped_without_landmarks(frame):
    out = OverlayRenderer(draw_mesh=False, draw_fps=False).render(
        frame, None, mesh_packet=_mesh(frame, iris=(False, False))
    )

    assert not _pixels_changed(out)


def test_hud_is_drawn_only_for_positive_values(frame):
    renderer = OverlayRenderer(draw_mesh=False, draw_iris=False)

    silent = renderer.render(frame, None, fps=0.0, latency_ms=0.0)
    loud = renderer.render(frame, None, fps=29.97, latency_ms=42.5)

    assert not _pixels_changed(silent)
    assert _pixels_changed(loud)


def test_hud_reports_confidence_and_state(frame):
    out = OverlayRenderer(draw_mesh=False, draw_iris=False).render(frame, _estimate())

    assert _pixels_changed(out)
