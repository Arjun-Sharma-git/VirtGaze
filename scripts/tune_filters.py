#!/usr/bin/env python3
"""Tune the Kalman (UKF) and One Euro filters to your eyes.

Shows a target that holds still at a few spots, then moves around the screen
at a speed that ramps slow -> fast -> slow (twice).  Follow it with your eyes.
The raw gaze is recorded and the filter parameters that best track the target
(least jitter while it holds, least lag while it moves) are searched offline,
then saved to ``profiles/<user>/filters.json``, which ``run_tracker.py``
applies automatically.

Usage::

    python scripts/tune_filters.py --user alice [--config PATH] [--duration 40]

Needs an existing calibration (run_calibration.py) for the user.  ESC aborts.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune the gaze smoothing filters")
    p.add_argument("--config", default="configs/example_config.yaml")
    p.add_argument("--user", default="default")
    p.add_argument(
        "--duration", type=float, default=40.0, help="Seconds of moving target (default 40)"
    )
    p.add_argument("--holds", type=int, default=8, help="Still targets before the motion")
    p.add_argument("--min-speed", type=float, default=150.0, help="Slowest target speed, px/s")
    p.add_argument("--max-speed", type=float, default=900.0, help="Fastest target speed, px/s")
    p.add_argument("--dry-run", action="store_true", help="Tune and report, but don't save")
    return p.parse_args()


def _lissajous(theta: float, sw: int, sh: int) -> tuple:
    """A figure that sweeps the whole screen (inside a 10% margin)."""
    ax, ay = sw * 0.4, sh * 0.4
    return sw / 2 + ax * math.sin(theta), sh / 2 + ay * math.sin(2 * theta + math.pi / 2)


def _speed_at(t: float, duration: float, vmin: float, vmax: float) -> float:
    """Slow -> fast -> slow, twice over *duration*."""
    phase = 2 * math.pi * (t / max(duration, 1e-6)) * 2
    return vmin + (vmax - vmin) * 0.5 * (1 - math.cos(phase))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import pygame

    from gaze_estimation.config.config import Config
    from gaze_estimation.filtering.tuning import (
        TuningSession,
        apply_filter_params,
        interpolate_targets,
        save_filter_params,
        segments_to_flags,
        tune,
    )
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.profile.loading import attach_user_model
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.utils.logging import setup_logging
    from gaze_estimation.utils.screen import detect_screen_resolution

    config = Config.from_yaml(args.config)
    setup_logging(level="WARNING")
    sw, sh = detect_screen_resolution()
    profile_mgr = ProfileManager(profiles_dir=str(config.get("profile.profiles_dir", "profiles")))
    filters_path = profile_mgr.get_profile_filters_path(args.user)
    # Start from the current tuning (if any), so a re-run refines it.
    apply_filter_params(config, filters_path)

    pipeline = GazeEstimationPipeline(config=config, screen_width=sw, screen_height=sh)
    try:
        attach_user_model(pipeline, config, profile_mgr, args.user)
    except RuntimeError as exc:
        sys.exit(str(exc))

    pipeline.start()
    tap = pipeline.tap_predictions()

    pygame.init()
    screen = pygame.display.set_mode((sw, sh), pygame.FULLSCREEN)
    pygame.display.set_caption("Filter tuning")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)

    target_t: list = []
    target_xy: list = []
    segments: list = []  # (start, end, "hold" | "move")
    aborted = False

    def frame(x: float, y: float, msg: str = "", radius: int = 14) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return False
        screen.fill((20, 20, 30))
        pygame.draw.circle(screen, (60, 220, 120), (int(x), int(y)), radius)
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 4)
        if msg:
            text = font.render(msg, True, (200, 200, 200))
            screen.blit(text, (sw // 2 - text.get_width() // 2, sh - 80))
        pygame.display.flip()
        target_t.append(time.perf_counter())
        target_xy.append((x, y))
        clock.tick(120)
        return True

    try:
        # Intro: get ready, eyes on the dot
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            if not frame(sw / 2, sh / 2, "Follow the dot with your eyes. Keep your head still."):
                raise KeyboardInterrupt
        # Phase 1: still targets (jitter)
        rng = np.random.default_rng()
        for _ in range(args.holds):
            x = float(rng.uniform(0.1, 0.9) * sw)
            y = float(rng.uniform(0.1, 0.9) * sh)
            start = time.perf_counter()
            while time.perf_counter() - start < 2.0:
                if not frame(x, y):
                    raise KeyboardInterrupt
            segments.append((start, time.perf_counter(), "hold"))
        # Phase 2: moving target with variable speed (lag)
        theta = 0.0
        x, y = _lissajous(theta, sw, sh)
        start = time.perf_counter()
        while time.perf_counter() - start < 1.0:  # lock on before it moves
            if not frame(x, y):
                raise KeyboardInterrupt
        start = prev = time.perf_counter()
        while (now := time.perf_counter()) - start < args.duration:
            dt = now - prev
            prev = now
            v = _speed_at(now - start, args.duration, args.min_speed, args.max_speed)
            # Advance along the curve at ~v px/s (speed / |d position / d theta|)
            dx = 0.4 * sw * math.cos(theta)
            dy = 0.4 * sh * 2 * math.cos(2 * theta + math.pi / 2)
            theta += v * dt / max(math.hypot(dx, dy), 1.0)
            x, y = _lissajous(theta, sw, sh)
            if not frame(x, y, f"{v:4.0f} px/s"):
                raise KeyboardInterrupt
        segments.append((start, time.perf_counter(), "move"))
    except KeyboardInterrupt:
        aborted = True
    finally:
        pygame.quit()
        pipeline.stop()

    if aborted:
        sys.exit("Aborted.")

    packets = []
    while True:
        try:
            packets.append(tap.get_nowait())
        except queue.Empty:
            break
    packets = [pk for pk in packets if pk.source in ("ridge", "mlp")]
    if len(packets) < 100:
        sys.exit(f"Only {len(packets)} usable gaze samples — is your face visible and lit?")

    ft = np.array([pk.timestamp for pk in packets])
    moving, weight = segments_to_flags(ft, segments)
    session = TuningSession(
        t=ft,
        raw=np.array([(pk.screen_x, pk.screen_y) for pk in packets]),
        target=interpolate_targets(ft, target_t, np.array(target_xy)),
        moving=moving,
        weight=weight,
        ears=np.array(
            [(pk.raw_features.get("left_ear", 1.0), pk.raw_features.get("right_ear", 1.0))
             for pk in packets]
        ),
    )
    print(f"[tune] {len(packets)} samples recorded; searching filter parameters …")

    def _progress(frac: float) -> None:
        print(f"\r[tune] {frac * 100:5.1f}%", end="", flush=True)

    result = tune(session, config, progress=_progress)
    print()
    b = result.baseline
    print(f"[tune] Before: {b['error_px']:.0f} px average error "
          f"(still {b['hold_error_px']:.0f}, moving {b['moving_error_px']:.0f})")
    print(f"[tune] After:  {result.error_px:.0f} px average error "
          f"(still {result.hold_error_px:.0f}, moving {result.moving_error_px:.0f})")
    for key, value in result.params.items():
        print(f"         filtering.{key} = {value:.4g}")
    if args.dry_run:
        print("[tune] --dry-run: not saved")
    else:
        save_filter_params(filters_path, result)
        print(f"[tune] Saved to {filters_path} (the tracker applies it automatically)")


if __name__ == "__main__":
    main()
