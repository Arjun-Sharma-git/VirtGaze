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

# Shown before each row when head variation is on.  Sampling the same gaze
# targets from slightly different head positions teaches the model to ignore
# small head movements, which otherwise show up as drift.
HEAD_PROMPTS = [
    "Keep your head where it is",
    "Move your head slightly LEFT, then keep it there",
    "Move your head slightly RIGHT, then keep it there",
    "Tilt your head slightly UP, then keep it there",
    "Tilt your head slightly DOWN, then keep it there",
    "Lean slightly CLOSER to the screen",
    "Lean slightly BACK from the screen",
]


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
    head_variation: bool = False,
    prompt_sec: float = 2.5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run the dot routine on a started :class:`DirectGazeEngine`.

    With *head_variation*, each row starts with a prompt to shift the head
    slightly (see :data:`HEAD_PROMPTS`).

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
        for i, (tx, ty) in enumerate(grid_points(rows, cols, sw, sh, margin, order)):
            if head_variation and i % cols == 0:
                _prompt(sw, sh, HEAD_PROMPTS[(i // cols) % len(HEAD_PROMPTS)], (tx, ty),
                        prompt_sec)
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


def _prompt(sw: int, sh: int, text: str, next_dot: tuple, seconds: float) -> None:
    """Show an instruction (and where the next dot will be) for *seconds*."""
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        def draw(c):
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.putText(c, text, ((sw - w) // 2, sh // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (230, 230, 230), 3)
            cv2.circle(c, (int(next_dot[0]), int(next_dot[1])), 8, (0, 120, 0), -1)

        _show(sw, sh, draw)


def lissajous_point(t: float, sw: int, sh: int, margin: float, period: float) -> tuple:
    """Position at time *t* on a 3:2 Lissajous figure covering the screen."""
    w = 2 * math.pi / period
    ax, ay = (0.5 - margin) * sw, (0.5 - margin) * sh
    return sw / 2 + ax * math.sin(3 * w * t + math.pi / 2), sh / 2 + ay * math.sin(2 * w * t)


def run_pursuit_calibration(
    engine,
    sw: int,
    sh: int,
    duration_sec: float = 30.0,
    margin: float = 0.03,
    period_sec: float = 30.0,
    lag_sec: float = 0.1,
    lock_on_sec: float = 1.5,
    blink_ear_ratio: float = 0.8,
    wait_for_face_sec: float = 2.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Smooth-pursuit calibration: follow a dot sweeping the whole screen.

    The dot traces a 3:2 Lissajous figure out to *margin* from the edges, so
    one pass covers the screen densely, edges included, and every frame is a
    sample (several times more data than still dots).  Each frame is paired
    with where the dot was *lag_sec* earlier: the eyes trail a moving target
    slightly, and the frame is captured a little before it is processed.

    Returns ``(X, Y_px)``, or ``None`` on ESC.
    """
    samples = engine.subscribe_samples()
    blink = BlinkDetector(ratio=blink_ear_ratio)
    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    X: list = []
    Y: list = []
    try:
        _wait_for_face(engine, samples, sw, sh, wait_for_face_sec)
        x0, y0 = lissajous_point(0.0, sw, sh, margin, period_sec)
        _prompt(sw, sh, "Follow the dot with your eyes", (x0, y0), 2.0)
        start = time.perf_counter()
        while time.perf_counter() - start < lock_on_sec:
            _show(sw, sh, lambda c: cv2.circle(c, (int(x0), int(y0)), 18, (0, 255, 0), -1))
        _drain(samples)
        start = time.perf_counter()
        while (el := time.perf_counter() - start) < duration_sec:
            x, y = lissajous_point(el, sw, sh, margin, period_sec)
            _show(sw, sh, lambda c: cv2.circle(c, (int(x), int(y)), 18, (0, 255, 0), -1))
            while True:
                try:
                    s = samples.get_nowait()
                except queue.Empty:
                    break
                t = s.timestamp - start - lag_sec
                if t < 0.5 or s.landmark_features is None:
                    continue
                if blink.is_blink(s.left_ear, s.right_ear):
                    continue
                X.append(s.landmark_features)
                Y.append(lissajous_point(t, sw, sh, margin, period_sec))
    except _Aborted:
        return None
    finally:
        engine.unsubscribe_samples(samples)
        cv2.destroyWindow(WINDOW)
    if not X:
        return None
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def calibrate(engine, sw: int, sh: int, cal_cfg: dict, mode: Optional[str] = None):
    """Run the configured routine(s): "dots", "pursuit" or "both" (dots first)."""
    mode = (mode or str(cal_cfg.get("mode", "dots"))).lower()
    margin = float(cal_cfg.get("edge_margin", 0.03))
    ratio = float(cal_cfg.get("blink_ear_ratio", 0.8))
    wait = float(cal_cfg.get("wait_for_face_sec", 2.0))
    parts = []
    if mode in ("dots", "both"):
        res = run_direct_calibration(
            engine, sw, sh,
            rows=int(cal_cfg.get("grid_rows", 5)),
            cols=int(cal_cfg.get("grid_cols", 5)),
            margin=margin,
            settle_sec=float(cal_cfg.get("settle_sec", 2.0)),
            capture_sec=float(cal_cfg.get("target_duration_sec", 1.0)),
            wait_for_face_sec=wait,
            blink_ear_ratio=ratio,
            order=str(cal_cfg.get("order", "serpentine")),
            head_variation=bool(cal_cfg.get("head_variation", False)),
        )
        if res is None:
            return None
        parts.append(res)
    if mode in ("pursuit", "both"):
        res = run_pursuit_calibration(
            engine, sw, sh,
            duration_sec=float(cal_cfg.get("pursuit_sec", 30.0)),
            margin=margin,
            period_sec=float(cal_cfg.get("pursuit_period_sec", 30.0)),
            lag_sec=float(cal_cfg.get("pursuit_lag_sec", 0.1)),
            blink_ear_ratio=ratio,
            wait_for_face_sec=0.0 if parts else wait,
        )
        if res is None:
            return None
        parts.append(res)
    if not parts:
        raise ValueError(f"unknown calibration mode {mode!r}")
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


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
