"""CalibrationUI: fullscreen pygame window for calibration target display."""
from __future__ import annotations

import time
from typing import Callable, Tuple

from gaze_estimation.calibration.calibration_target import CalibrationTarget


class CalibrationUI:
    """Fullscreen calibration window using pygame.

    Renders an animated target dot and a progress bar.

    Usage::

        ui = CalibrationUI(screen_width=1920, screen_height=1080)
        ui.start()
        ui.set_target(960, 540)
        ui.set_progress(0.5)
        # … run calibration …
        ui.stop()

    Args:
        screen_width:  Display resolution width (pixels).
        screen_height: Display resolution height (pixels).
        bg_color:      Background colour (R, G, B).
        dot_color:     Target dot colour.
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        bg_color: Tuple[int, int, int] = (30, 30, 30),
        dot_color: Tuple[int, int, int] = (255, 50, 50),
        base_radius: float = 20.0,
        pulse_amplitude: float = 0.3,
        pulse_freq_hz: float = 2.5,
        circular_radius: float = 15.0,
        circular_freq_hz: float = 1.0,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._bg_color = bg_color
        self._dot_color = dot_color

        # Animated target (pulse + circular drift) shared with the pipeline
        self._target = CalibrationTarget(
            base_radius=base_radius,
            pulse_amplitude=pulse_amplitude,
            pulse_freq_hz=pulse_freq_hz,
            circular_radius=circular_radius,
            circular_freq_hz=circular_freq_hz,
        )
        self._target.set_position(screen_width / 2, screen_height / 2)

        self._progress: float = 0.0
        self._message: str = "Look at the red dot"
        self._running = False
        self._screen = None
        self._clock = None
        self._pygame = None

    def start(self) -> None:
        """Initialise pygame and open the fullscreen window."""
        try:
            import pygame
            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height), pygame.FULLSCREEN
            )
            pygame.display.set_caption("Gaze Calibration")
            self._clock = pygame.time.Clock()
            self._running = True
        except Exception as exc:
            print(f"[CalibrationUI] Could not initialise pygame: {exc}")
            self._running = False

    def stop(self) -> None:
        """Close the pygame window."""
        if self._pygame is not None:
            self._pygame.quit()
        self._running = False

    def set_target(self, x: float, y: float) -> None:
        """Move the calibration target to (x, y) and restart its animation."""
        self._target.set_position(x, y)

    def set_progress(self, fraction: float) -> None:
        """Set the progress bar fill (0.0 – 1.0)."""
        self._progress = max(0.0, min(1.0, fraction))

    def set_message(self, msg: str) -> None:
        """Set the instruction text shown below the target."""
        self._message = msg

    def update(self) -> bool:
        """Render one frame.  Returns False if the user pressed ESC/quit."""
        if not self._running or self._pygame is None or self._screen is None:
            return False

        pg = self._pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return False

        # Background
        self._screen.fill(self._bg_color)

        # Animated target
        state = self._target.update(time.monotonic())
        radius = max(2, int(state.radius))
        ix, iy = int(state.x), int(state.y)
        alpha = max(0.0, min(1.0, state.alpha))
        color = tuple(int(c * alpha + bg * (1.0 - alpha))
                      for c, bg in zip(self._dot_color, self._bg_color))
        pg.draw.circle(self._screen, color, (ix, iy), radius)
        pg.draw.circle(self._screen, (255, 255, 255), (ix, iy), 5)

        # Progress bar
        bar_w = int(self.screen_width * 0.6)
        bar_h = 12
        bar_x = (self.screen_width - bar_w) // 2
        bar_y = self.screen_height - 50
        pg.draw.rect(self._screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * self._progress)
        if fill_w > 0:
            pg.draw.rect(self._screen, (50, 200, 100), (bar_x, bar_y, fill_w, bar_h))

        # Message
        try:
            font = pg.font.SysFont("Arial", 24)
            text_surf = font.render(self._message, True, (200, 200, 200))
            self._screen.blit(
                text_surf,
                (self.screen_width // 2 - text_surf.get_width() // 2,
                 self.screen_height - 90),
            )
        except Exception:
            pass

        pg.display.flip()
        if self._clock is not None:
            self._clock.tick(60)
        return True

    def run_loop_while(self, condition_fn: Callable[[], bool]) -> None:
        """Run the render loop until *condition_fn* returns False or ESC."""
        while condition_fn() and self.update():
            pass

    @property
    def target(self) -> CalibrationTarget:
        """The animated target shared with the calibration engine."""
        return self._target
