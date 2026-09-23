"""Tests for the pygame calibration window.

pygame is imported lazily inside :meth:`CalibrationUI.start`, and no wheel exists
for every interpreter (CI runs headless), so the entire render path is driven
through a recording stand-in injected into ``sys.modules``.  The fake takes
precedence over an installed pygame, so these tests behave identically with and
without the real package.
"""

from __future__ import annotations

import sys
import types

import pytest

from gaze_estimation.calibration.calibration_target import CalibrationTarget
from gaze_estimation.visualization.calibration_ui import CalibrationUI


class _Event:
    def __init__(self, type_: int, key: int = 0) -> None:
        self.type = type_
        self.key = key


class _FakeTextSurface:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_width(self) -> int:
        return len(self.text) * 10


class _FakeFont:
    def render(self, text, antialias, color, background=None):
        return _FakeTextSurface(text)


class _FakeClock:
    def __init__(self) -> None:
        self.ticks: list[int] = []

    def tick(self, framerate: int = 0) -> int:
        self.ticks.append(framerate)
        return 0


class _FakeSurface:
    """Records fill/blit calls instead of rasterising pixels."""

    def __init__(self) -> None:
        self.fills: list = []
        self.blits: list = []

    def fill(self, color) -> None:
        self.fills.append(color)

    def blit(self, source, dest) -> None:
        self.blits.append((source, dest))


class FakePygame(types.ModuleType):
    """Minimal pygame stand-in recording every call the UI makes."""

    QUIT = 256
    KEYDOWN = 768
    K_ESCAPE = 27
    FULLSCREEN = 0x8000

    def __init__(self, events=(), font_raises: bool = False, quit_after=None) -> None:
        super().__init__("pygame")
        self.init_calls = 0
        self.quit_calls = 0
        self.flip_calls = 0
        self.get_calls = 0
        self.captions: list[str] = []
        self.set_modes: list = []
        self.circles: list = []
        self.rects: list = []
        self.surface = _FakeSurface()
        self.clock = _FakeClock()
        self._font_raises = font_raises
        self._quit_after = quit_after
        self._queued = list(events)
        outer = self

        class _Display:
            def set_mode(self, size, flags=0, depth=0):
                outer.set_modes.append((size, flags))
                return outer.surface

            def set_caption(self, caption):
                outer.captions.append(caption)

            def flip(self):
                outer.flip_calls += 1

        class _Time:
            def Clock(self):
                return outer.clock

        class _Draw:
            def circle(self, surface, color, center, radius, width=0):
                outer.circles.append((color, center, radius))

            def rect(self, surface, color, rect, width=0):
                outer.rects.append((color, rect))

        class _FontModule:
            def SysFont(self, name, size):
                if outer._font_raises:
                    raise RuntimeError("no usable font")
                return _FakeFont()

        class _EventModule:
            def get(self):
                outer.get_calls += 1
                if outer._quit_after is not None and outer.get_calls >= outer._quit_after:
                    return [_Event(outer.QUIT)]
                queued, outer._queued = outer._queued, []
                return queued

        self.display = _Display()
        self.time = _Time()
        self.draw = _Draw()
        self.font = _FontModule()
        self.event = _EventModule()

    def init(self):
        self.init_calls += 1

    def quit(self):
        self.quit_calls += 1


@pytest.fixture
def fake_pygame(monkeypatch):
    """Install a fake pygame into ``sys.modules``; undone automatically."""

    def install(**kwargs) -> FakePygame:
        fake = FakePygame(**kwargs)
        monkeypatch.setitem(sys.modules, "pygame", fake)
        return fake

    return install


@pytest.fixture
def ui_started(fake_pygame):
    """A CalibrationUI with its (fake) window open, plus the fake pygame."""

    def make(**kwargs):
        pg = fake_pygame(**kwargs.pop("pygame", {}))
        ui = CalibrationUI(**kwargs)
        ui.start()
        return ui, pg

    return make


def test_start_opens_a_fullscreen_window(ui_started):
    ui, pg = ui_started(screen_width=800, screen_height=600)
    assert pg.init_calls == 1
    assert pg.set_modes == [((800, 600), FakePygame.FULLSCREEN)]
    assert pg.captions == ["Gaze Calibration"]
    assert ui.update() is True  # proves the render loop is armed


def test_start_failure_is_reported_and_non_fatal(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pygame", None)  # makes `import pygame` fail
    ui = CalibrationUI()

    ui.start()

    assert "Could not initialise pygame" in capsys.readouterr().out
    assert ui.update() is False


def test_update_before_start_returns_false():
    assert CalibrationUI().update() is False


def test_update_paints_background_target_and_progress(ui_started):
    ui, pg = ui_started(screen_width=800, screen_height=600)
    ui.set_target(400, 300)
    ui.set_progress(0.5)
    ui.set_message("Hi")

    assert ui.update() is True

    assert pg.surface.fills == [(30, 30, 30)]
    # 2 circles: the animated dot and its white highlight, at the same point.
    assert len(pg.circles) == 2
    dot_color, center, radius = pg.circles[0]
    assert all(0 <= channel <= 255 for channel in dot_color)
    assert abs(center[0] - 400) <= 15 and abs(center[1] - 300) <= 15
    assert pg.circles[1][1] == center and pg.circles[1][2] == 5
    # Progress bar: 60% of 800 wide, 12 tall, 50 px off the bottom.
    assert pg.rects == [
        ((80, 80, 80), (160, 550, 480, 12)),
        ((50, 200, 100), (160, 550, 240, 12)),
    ]
    text_surface, dest = pg.surface.blits[0]
    assert text_surface.text == "Hi"
    assert dest == (390, 510)
    assert pg.flip_calls == 1
    assert pg.clock.ticks == [60]


def test_zero_progress_draws_only_the_empty_bar(ui_started):
    ui, pg = ui_started(screen_width=800, screen_height=600)
    ui.set_progress(0.0)

    ui.update()

    assert len(pg.rects) == 1


def test_progress_is_clamped_to_the_unit_range():
    ui = CalibrationUI()
    ui.set_progress(-5.0)
    assert ui._progress == 0.0
    ui.set_progress(2.0)
    assert ui._progress == 1.0


def test_quit_event_ends_the_render_loop(fake_pygame):
    fake_pygame(events=[_Event(FakePygame.QUIT)])
    ui = CalibrationUI()
    ui.start()

    assert ui.update() is False


def test_escape_ends_the_render_loop(fake_pygame):
    fake_pygame(events=[_Event(FakePygame.KEYDOWN, FakePygame.K_ESCAPE)])
    ui = CalibrationUI()
    ui.start()

    assert ui.update() is False


def test_unrelated_events_are_ignored(fake_pygame):
    fake_pygame(events=[_Event(FakePygame.KEYDOWN, 1234)])
    ui = CalibrationUI()
    ui.start()

    assert ui.update() is True


def test_font_failure_does_not_break_rendering(ui_started):
    ui, pg = ui_started(pygame={"font_raises": True})
    ui.set_message("hi")

    assert ui.update() is True

    assert pg.surface.blits == []
    assert pg.flip_calls == 1


def test_stop_quits_pygame_and_blocks_further_updates(ui_started):
    ui, pg = ui_started()

    ui.stop()

    assert pg.quit_calls == 1
    assert ui.update() is False


def test_stop_before_start_is_safe():
    CalibrationUI().stop()  # must not raise


def test_render_loop_runs_until_the_condition_is_false(ui_started):
    ui, pg = ui_started()
    remaining = iter([True, True, False])

    ui.run_loop_while(lambda: next(remaining, False))

    assert pg.flip_calls == 2


def test_render_loop_stops_when_update_reports_quit(ui_started):
    ui, pg = ui_started(pygame={"quit_after": 3})

    ui.run_loop_while(lambda: True)

    assert pg.flip_calls == 2  # the third update saw QUIT before painting


def test_target_is_shared_and_centred_on_construction():
    ui = CalibrationUI(screen_width=800, screen_height=600)

    assert isinstance(ui.target, CalibrationTarget)
    state = ui.target.update()
    assert abs(state.x - 400) <= 15 and abs(state.y - 300) <= 15


def test_set_target_moves_the_shared_target():
    ui = CalibrationUI()

    ui.set_target(10, 20)

    state = ui.target.update()
    assert abs(state.x - 10) <= 15 and abs(state.y - 20) <= 15
