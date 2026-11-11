#!/usr/bin/env python3
"""Measure how accurate the gaze tracker is for a user, right now.

Runs the tracker exactly as ``run_tracker.py`` would (the user's model and
tuned filters) and shows test dots at positions the calibration never used:

1. **Still dots** (3x3 or 4x4 grid, shuffled): for each, the median gaze
   position after the eyes settle, its distance from the dot, and the jitter.
2. **Moving dot** (optional): the dot sweeps the screen at a few fixed speeds;
   reports the average distance between the gaze and the dot.

Prints a summary and saves it to ``profiles/<user>/accuracy/<time>.json`` so
runs can be compared over time.  ESC aborts.

Usage::

    python scripts/check_accuracy.py --user alice [--config PATH]
        [--points 9|16] [--hold 3] [--speeds 300 600] [--no-pursuit]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check gaze tracker accuracy")
    p.add_argument("--config", default="configs/example_config.yaml")
    p.add_argument("--user", default="default")
    p.add_argument(
        "--direct", action="store_true", help="Use the direct engine (direct_calibrate.py)"
    )
    p.add_argument("--points", type=int, choices=(9, 16), default=9, help="Still test dots")
    p.add_argument("--hold", type=float, default=3.0, help="Seconds per still dot")
    p.add_argument(
        "--speeds", type=float, nargs="*", default=[300.0, 600.0], help="Moving-dot speeds, px/s"
    )
    p.add_argument("--pursuit-sec", type=float, default=8.0, help="Seconds per speed")
    p.add_argument("--no-pursuit", action="store_true", help="Skip the moving-dot test")
    return p.parse_args()


class _Aborted(Exception):
    pass


def _grid(points: int, sw: int, sh: int) -> list:
    """Test positions (shuffled)."""
    # 9 points: inner area; 16 points: reaches out to 6% from the edges
    n, fr = (3, np.linspace(0.15, 0.85, 3)) if points == 9 else (4, np.linspace(0.06, 0.94, 4))
    pts = [(float(x * sw), float(y * sh)) for y in fr for x in fr]
    np.random.default_rng().shuffle(pts)
    return pts


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import pygame

    from gaze_estimation.config.config import Config
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.profile.loading import apply_user_filters, attach_user_model
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.utils.logging import setup_logging
    from gaze_estimation.utils.screen import detect_screen_resolution

    config = Config.from_yaml(args.config)
    setup_logging(level="WARNING")
    sw, sh = detect_screen_resolution()
    profile_mgr = ProfileManager(profiles_dir=str(config.get("profile.profiles_dir", "profiles")))
    tuned = apply_user_filters(config, profile_mgr, args.user)
    if args.direct:
        from gaze_estimation.direct import DirectGazeEngine

        pipeline = DirectGazeEngine(config, sw, sh)
    else:
        pipeline = GazeEstimationPipeline(config=config, screen_width=sw, screen_height=sh)
    try:
        model_kind = attach_user_model(pipeline, config, profile_mgr, args.user)
    except RuntimeError as exc:
        sys.exit(str(exc))
    pipeline.start()

    pygame.init()
    screen = pygame.display.set_mode((sw, sh), pygame.FULLSCREEN)
    pygame.display.set_caption("Accuracy check")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)

    def draw(x: float, y: float, msg: str = "") -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                raise _Aborted
        screen.fill((20, 20, 30))
        pygame.draw.circle(screen, (60, 220, 120), (int(x), int(y)), 14)
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 4)
        if msg:
            text = font.render(msg, True, (200, 200, 200))
            screen.blit(text, (sw // 2 - text.get_width() // 2, sh - 80))
        pygame.display.flip()
        clock.tick(120)

    def fresh_estimate(last_ts):
        e = pipeline.get_latest_estimate()
        if e is None or e.timestamp == last_ts or e.source not in ("ridge", "mlp"):
            return None
        return e

    still: list = []
    pursuit: list = []
    settle = min(1.0, args.hold / 2)
    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            draw(sw / 2, sh / 2, "Look at each dot. Keep your head still.")

        for tx, ty in _grid(args.points, sw, sh):
            t0 = time.perf_counter()
            raw, filt, last = [], [], None
            while (now := time.perf_counter()) - t0 < args.hold:
                draw(tx, ty)
                e = fresh_estimate(last)
                if e is not None:
                    last = e.timestamp
                    if now - t0 >= settle:
                        raw.append((e.raw_x, e.raw_y))
                        filt.append((e.screen_x, e.screen_y))
            if filt:
                f, r = np.array(filt), np.array(raw)
                med = np.median(f, axis=0)
                still.append(
                    {
                        "target": [tx, ty],
                        "gaze": med.tolist(),
                        "error_px": float(np.hypot(*(med - [tx, ty]))),
                        "raw_error_px": float(np.hypot(*(np.median(r, axis=0) - [tx, ty]))),
                        "jitter_px": float(np.median(np.linalg.norm(f - med, axis=1))),
                        "samples": len(f),
                    }
                )

        if not args.no_pursuit:
            for speed in args.speeds:
                theta, prev = 0.0, time.perf_counter()
                start = prev
                errs, last = [], None
                while (now := time.perf_counter()) - start < args.pursuit_sec + 1.0:
                    dt, prev = now - prev, now
                    if now - start >= 1.0:  # first second: hold still to lock on
                        dx = 0.35 * sw * math.cos(theta)
                        dy = 0.35 * sh * 2 * math.cos(2 * theta)
                        theta += speed * dt / max(math.hypot(dx, dy), 1.0)
                    x = sw / 2 + 0.35 * sw * math.sin(theta)
                    y = sh / 2 + 0.35 * sh * math.sin(2 * theta)
                    draw(x, y, f"Follow the dot  ({speed:.0f} px/s)")
                    e = fresh_estimate(last)
                    if e is not None and now - start >= 1.5:
                        last = e.timestamp
                        errs.append(float(np.hypot(e.screen_x - x, e.screen_y - y)))
                if errs:
                    pursuit.append(
                        {"speed_px_s": speed, "mean_error_px": float(np.mean(errs)),
                         "samples": len(errs)}
                    )
    except _Aborted:
        pygame.quit()
        pipeline.stop()
        sys.exit("Aborted.")
    pygame.quit()
    pipeline.stop()

    if not still:
        sys.exit("No gaze samples — is your face visible and lit?")

    errors = np.array([s["error_px"] for s in still])
    offset = np.median(np.array([np.subtract(s["gaze"], s["target"]) for s in still]), axis=0)
    centred = np.array(
        [np.hypot(*(np.subtract(s["gaze"], s["target"]) - offset)) for s in still]
    )
    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "user": args.user,
        "model": model_kind,
        "tuned_filters": tuned,
        "screen": [sw, sh],
        "median_error_px": float(np.median(errors)),
        "median_error_pct_width": float(100 * np.median(errors) / sw),
        "worst_error_px": float(errors.max()),
        "median_jitter_px": float(np.median([s["jitter_px"] for s in still])),
        "constant_offset_px": offset.tolist(),
        "median_error_without_offset_px": float(np.median(centred)),
        "still": still,
        "pursuit": pursuit,
    }

    print(f"\nAccuracy check for '{args.user}' ({model_kind} model, "
          f"{'tuned' if tuned else 'default'} filters, {sw}x{sh})")
    for s in sorted(still, key=lambda s: (s["target"][1], s["target"][0])):
        print(f"  dot ({s['target'][0]:5.0f},{s['target'][1]:5.0f})  gaze ({s['gaze'][0]:5.0f},"
              f"{s['gaze'][1]:5.0f})  off {s['error_px']:4.0f} px  jitter {s['jitter_px']:3.0f} px")
    print(f"  Median error : {summary['median_error_px']:.0f} px "
          f"({summary['median_error_pct_width']:.1f}% of width), worst {errors.max():.0f} px")
    print(f"  Median jitter: {summary['median_jitter_px']:.0f} px")
    print(f"  Shared offset: {offset[0]:+.0f} px, {offset[1]:+.0f} px  -> without it "
          f"{summary['median_error_without_offset_px']:.0f} px")
    if np.hypot(*offset) > 0.5 * summary["median_error_px"]:
        print("  Most of the error is one shift: run run_calibration.py --quick to correct it.")
    for p in pursuit:
        print(f"  Moving dot {p['speed_px_s']:4.0f} px/s: mean error {p['mean_error_px']:.0f} px")

    out_dir = os.path.join(os.path.dirname(profile_mgr.get_profile_ridge_path(args.user)),
                           "accuracy")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
