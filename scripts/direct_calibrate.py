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
    p.add_argument("--mode", choices=("dots", "pursuit", "both"), default=None,
                   help="Still dots, a moving dot to follow, or both (default: config)")
    p.add_argument("--pursuit-sec", type=float, default=None, help="Moving-dot duration")
    p.add_argument("--camera", default=None,
                   help="Camera mode WxH, e.g. 1280x720 (default: config); saved with the model")
    hv = p.add_mutually_exclusive_group()
    hv.add_argument("--head-variation", dest="head_variation", action="store_true",
                    default=None, help="Prompt small head moves between rows")
    hv.add_argument("--no-head-variation", dest="head_variation", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import numpy as np

    from gaze_estimation.config.config import Config
    from gaze_estimation.direct import DirectGazeEngine
    from gaze_estimation.direct.calibration import calibrate
    from gaze_estimation.model.ridge import RidgeGazeModel
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.utils.logging import setup_logging
    from gaze_estimation.utils.screen import detect_screen_resolution

    config = Config.from_yaml(args.config)
    from gaze_estimation.direct.engine import apply_camera_mode, camera_mode

    apply_camera_mode(config, args.camera)
    setup_logging(level="WARNING")
    cal = config.section("calibration")
    sw, sh = detect_screen_resolution()
    profile_mgr = ProfileManager(profiles_dir=str(config.get("profile.profiles_dir", "profiles")))

    engine = DirectGazeEngine(config, sw, sh)
    engine.start()
    overrides = {
        "grid_rows": args.rows, "grid_cols": args.cols, "edge_margin": args.margin,
        "settle_sec": args.settle, "target_duration_sec": args.capture,
        "pursuit_sec": args.pursuit_sec,
    }
    cal = {**cal, **{k: v for k, v in overrides.items() if v is not None}}
    if args.head_variation is not None:
        cal["head_variation"] = args.head_variation
    try:
        result = calibrate(engine, sw, sh, cal, mode=args.mode)
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
    model.camera = camera_mode(config)
    err = np.linalg.norm(model.predict(X) * [sw, sh] - Y, axis=1)
    path = profile_mgr.get_profile_ridge_path(args.user)
    model.save(path)
    print(f"[calibration] {len(X)} frames; "
          f"fit error {np.median(err):.0f} px median")
    print(f"[calibration] Saved {path}")


if __name__ == "__main__":
    main()
