"""Calibration for the direct engine (the EyeTrax dot routine).

Waits until the face is tracked steadily (with a countdown), then for each dot
of a rows x cols grid: pulses it for ``settle_sec`` (eyes settle, nothing
recorded), then records ``capture_sec`` of frames while a ring counts down,
skipping blink frames.  Frames are matched to dots by capture timestamp, so
display and camera never get out of step.
"""

from __future__ import annotations

import math
import queue
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gaze_estimation.filtering.blink import BlinkDetector

WINDOW = "VirtGaze calibration"


def grid_points(
    rows: int, cols: int, sw: int, sh: int, margin: float, order: str = "serpentine"
) -> List[Tuple[float, float]]:
    xs = np.linspace(margin, 1.0 - margin, cols) * sw
    ys = np.linspace(margin, 1.0 - margin, rows) * sh
    if order == "serpentine":
        return [
            (float(x), float(y)) for r, y in enumerate(ys) for x in (xs if r % 2 == 0 else xs[::-1])
        ]
    return [(float(x), float(y)) for y in ys for x in xs]


class _Aborted(Exception):
    pass


def _show(sw: int, sh: int, draw) -> None:
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
    draw(canvas)
    cv2.imshow(WINDOW, canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        raise _Aborted


def run_direct_calibration(
    engine,
    sw: int,
    sh: int,
    rows: int = 5,
    cols: int = 5,
    margin: float = 0.03,
    settle_sec: float = 2.0,
    capture_sec: float = 1.0,
    wait_for_face_sec: float = 2.0,
    blink_ear_ratio: float = 0.8,
    order: str = "serpentine",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run the dot routine on a started :class:`DirectGazeEngine`.

    Returns ``(X, Y_px)`` (features, target pixels), or ``None`` on ESC.
    """
    samples = engine.subscribe_samples()
    blink = BlinkDetector(ratio=blink_ear_ratio)
    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    X: list = []
    Y: list = []
    try:
        _wait_for_face(engine, samples, sw, sh, wait_for_face_sec)
        for tx, ty in grid_points(rows, cols, sw, sh, margin, order):
            start = time.perf_counter()
            radius = 20
            while (el := time.perf_counter() - start) < settle_sec:
                radius = 15 + int(15 * abs(math.sin(2 * math.pi * el)))
                _show(sw, sh, lambda c: cv2.circle(c, (int(tx), int(ty)), radius, (0, 255, 0), -1))
            capture_start = time.perf_counter()
            _drain(samples)
            while (el := time.perf_counter() - capture_start) < capture_sec:
                ang = 360 * (1 - _ease(el / capture_sec))

                def draw(c, ang=ang):
                    cv2.circle(c, (int(tx), int(ty)), radius, (0, 255, 0), -1)
                    cv2.ellipse(c, (int(tx), int(ty)), (40, 40), 0, -90, -90 + ang,
                                (255, 255, 255), 4)

                _show(sw, sh, draw)
                _collect(samples, blink, capture_start, tx, ty, X, Y)
            _collect(samples, blink, capture_start, tx, ty, X, Y)
    except _Aborted:
        return None
    finally:
        engine.unsubscribe_samples(samples)
        cv2.destroyWindow(WINDOW)
    if not X:
        return None
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def _ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def _collect(samples, blink, since, tx, ty, X, Y) -> None:
    while True:
        try:
            s = samples.get_nowait()
        except queue.Empty:
            return
        if s.timestamp < since or s.landmark_features is None:
            continue
        if blink.is_blink(s.left_ear, s.right_ear):
            continue
        X.append(s.landmark_features)
        Y.append((tx, ty))


def _wait_for_face(engine, samples, sw: int, sh: int, seconds: float) -> None:
    """Countdown that starts once a face is seen and restarts if it is lost."""
    since = None
    while True:
        s = engine.get_latest_sample()
        face = s is not None and s.landmark_features is not None
        now = time.perf_counter()
        if face:
            since = since or now
            if now - since >= seconds:
                return
            ang = 360 * (1 - _ease((now - since) / max(seconds, 1e-6)))
            _show(sw, sh, lambda c: cv2.ellipse(
                c, (sw // 2, sh // 2), (50, 50), 0, -90, -90 + ang, (0, 255, 0), -1))
        else:
            since = None

            def msg(c):
                text = "Face not detected"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                cv2.putText(c, text, ((sw - w) // 2, (sh + h) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            _show(sw, sh, msg)
        _drain(samples)


def recenter(
    engine,
    ridge,
    sw: int,
    sh: int,
    settle_sec: float = 1.0,
    capture_sec: float = 1.5,
    blink_ear_ratio: float = 0.8,
) -> Optional[Tuple[float, float]]:
    """One-dot drift correction: look at the centre, then shift the model.

    Head position drifts between sessions (and within one); most of the
    resulting error is a constant offset.  This shows one dot, records
    ``capture_sec`` of frames and sets ``ridge.offset`` so they land on it.
    Returns the correction in pixels, or ``None`` if aborted / no face.
    """
    samples = engine.subscribe_samples()
    blink = BlinkDetector(ratio=blink_ear_ratio)
    cx, cy = sw / 2.0, sh / 2.0
    X: list = []
    Y: list = []
    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def label(c, text="Look at the dot"):
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(c, text, (sw // 2 - w // 2, sh // 2 + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (200, 200, 200), 2)

    try:
        start = time.perf_counter()
        while (el := time.perf_counter() - start) < settle_sec:
            r = 15 + int(15 * abs(math.sin(2 * math.pi * el)))

            def draw(c, r=r):
                cv2.circle(c, (int(cx), int(cy)), r, (0, 255, 0), -1)
                label(c)

            _show(sw, sh, draw)
        capture_start = time.perf_counter()
        _drain(samples)
        while (el := time.perf_counter() - capture_start) < capture_sec:
            ang = 360 * (1 - _ease(el / capture_sec))

            def draw(c, ang=ang):
                cv2.circle(c, (int(cx), int(cy)), 20, (0, 255, 0), -1)
                cv2.ellipse(c, (int(cx), int(cy)), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
                label(c)

            _show(sw, sh, draw)
            _collect(samples, blink, capture_start, cx, cy, X, Y)
        _collect(samples, blink, capture_start, cx, cy, X, Y)
    except _Aborted:
        return None
    finally:
        engine.unsubscribe_samples(samples)
        cv2.destroyWindow(WINDOW)
    if len(X) < 5:
        return None
    dx, dy = ridge.correct_drift(np.array(X), np.array(Y))
    return float(dx), float(dy)
