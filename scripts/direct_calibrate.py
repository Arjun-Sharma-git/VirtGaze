#!/usr/bin/env python3
"""Calibrate the direct engine (recommended): 5x5 dots, then save the model.

Waits for your face, then shows each dot (pulsing for ``settle_sec``, then
recording for ``capture_sec`` while a ring counts down).  Look at each dot
until it moves on.  The landmark ridge model is trained on the recorded frames
and saved to ``profiles/<user>/ridge_model.npz``, where run_direct.py,
check_accuracy.py --direct and tune_filters.py --direct pick it up.

Usage::

    python scripts/direct_calibrate.py --user alice [--config PATH]
        [--rows 5 --cols 5 --margin 0.03 --settle 2 --capture 1]
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct-mode calibration")
    p.add_argument("--config", default="configs/example_config.yaml")
    p.add_argument("--user", default="default")
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--margin", type=float, default=None, help="Edge margin, fraction")
    p.add_argument("--settle", type=float, default=None, help="Seconds before recording")
    p.add_argument("--capture", type=float, default=None, help="Seconds recorded per dot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import numpy as np

    from gaze_estimation.config.config import Config
    from gaze_estimation.direct import DirectGazeEngine
    from gaze_estimation.direct.calibration import run_direct_calibration
    from gaze_estimation.model.ridge import RidgeGazeModel
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.utils.logging import setup_logging
    from gaze_estimation.utils.screen import detect_screen_resolution

    config = Config.from_yaml(args.config)
    setup_logging(level="WARNING")
    cal = config.section("calibration")
    sw, sh = detect_screen_resolution()
    profile_mgr = ProfileManager(profiles_dir=str(config.get("profile.profiles_dir", "profiles")))

    engine = DirectGazeEngine(config, sw, sh)
    engine.start()
    try:
        result = run_direct_calibration(
            engine,
            sw,
            sh,
            rows=args.rows or int(cal.get("grid_rows", 5)),
            cols=args.cols or int(cal.get("grid_cols", 5)),
            margin=args.margin if args.margin is not None else float(cal.get("edge_margin", 0.03)),
            settle_sec=args.settle if args.settle is not None else float(cal.get("settle_sec", 2.0)),
            capture_sec=(
                args.capture if args.capture is not None
                else float(cal.get("target_duration_sec", 1.0))
            ),
            wait_for_face_sec=float(cal.get("wait_for_face_sec", 2.0)),
            blink_ear_ratio=float(cal.get("blink_ear_ratio", 0.8)),
            order=str(cal.get("order", "serpentine")),
        )
    finally:
        engine.stop()
    if result is None:
        sys.exit("Calibration aborted.")
    X, Y = result
    if len(X) < 20:
        sys.exit(f"Only {len(X)} usable frames — is your face visible and lit?")

    from gaze_estimation.model.landmark_features import landmark_columns

    model = RidgeGazeModel(
        alpha=float(config.get("inference.ridge_alpha", 1.0)),
        columns=landmark_columns(str(config.get("inference.landmark_subset", "eye"))),
    )
    model.fit(X, Y, sw, sh)
    err = np.linalg.norm(model.predict(X) * [sw, sh] - Y, axis=1)
    path = profile_mgr.get_profile_ridge_path(args.user)
    model.save(path)
    print(f"[calibration] {len(X)} frames from {len(np.unique(Y, axis=0))} dots; "
          f"fit error {np.median(err):.0f} px median")
    print(f"[calibration] Saved {path}")


if __name__ == "__main__":
    main()
