"""Primary-monitor resolution detection (best effort, optional dependency)."""
from __future__ import annotations

from typing import Tuple

from gaze_estimation.utils.logging import get_logger

_logger = get_logger("utils.screen")

DEFAULT_RESOLUTION: Tuple[int, int] = (1920, 1080)


def detect_screen_resolution(fallback: Tuple[int, int] = DEFAULT_RESOLUTION) -> Tuple[int, int]:
    """Return the primary monitor resolution as ``(width, height)``.

    Falls back to *fallback* when ``screeninfo`` is unavailable or cannot
    detect a monitor (e.g. headless machines).
    """
    try:
        import screeninfo  # type: ignore[import]

        monitors = screeninfo.get_monitors()
        if not monitors:
            return fallback
        monitor = monitors[0]
        return int(monitor.width), int(monitor.height)
    except Exception as exc:  # pragma: no cover - depends on host display
        _logger.warning("Screen detection unavailable (%s); using %s", exc, fallback)
        return fallback
