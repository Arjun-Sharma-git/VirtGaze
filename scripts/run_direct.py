#!/usr/bin/env python3
"""Run the direct gaze tracker fullscreen (recommended).

Shows where you are looking as a dot on a fullscreen canvas, with a small
camera preview in the corner.  Uses the model from direct_calibrate.py and
your tuned filters (tune_filters.py).

With ``--implicit-calibration``, every left-click is taken as "I am looking
here": it is added to the model, which is refitted every
``adaptation.ridge_refit_every`` clicks and saved when you quit.

Keys: q / Esc quit.

Usage::

    python scripts/run_direct.py --user alice [--config PATH] [--implicit-calibration]
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct gaze tracker")
    p.add_argument("--config", default="configs/example_config.yaml")
    p.add_argument("--user", default="default")
    p.add_argument("--implicit-calibration", action="store_true",
                   help="Clicks where you look refine the model")
    p.add_argument("--no-preview", action="store_true", help="Hide the camera preview")
    p.add_argument("--no-recenter", action="store_true",
                   help="Skip the one-dot drift correction at start")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import cv2
    import numpy as np

    from gaze_estimation.config.config import Config
    from gaze_estimation.direct import DirectGazeEngine
    from gaze_estimation.model.ridge import RidgeGazeModel
    from gaze_estimation.pipeline.schemas import GazeState
    from gaze_estimation.profile.loading import apply_user_filters
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.utils.logging import setup_logging
    from gaze_estimation.utils.screen import detect_screen_resolution

    config = Config.from_yaml(args.config)
    setup_logging(level="WARNING")
    sw, sh = detect_screen_resolution()
    profile_mgr = ProfileManager(profiles_dir=str(config.get("profile.profiles_dir", "profiles")))
    ridge_path = profile_mgr.get_profile_ridge_path(args.user)
    if not os.path.exists(ridge_path):
        sys.exit(f"No model for '{args.user}'. Run scripts/direct_calibrate.py --user {args.user}")
    if apply_user_filters(config, profile_mgr, args.user):
        print(f"[direct] Using tuned filter settings for '{args.user}'")
    ridge = RidgeGazeModel.load(ridge_path, alpha=float(config.get("inference.ridge_alpha", 1.0)))
    if ridge.camera is not None:
        # Run the camera in the mode the model was calibrated with
        w, h, fcc = ridge.camera
        config.set("camera.width", w)
        config.set("camera.height", h)
        config.set("camera.fourcc", fcc)

    engine = DirectGazeEngine(config, sw, sh)
    engine.set_ridge_model(ridge)
    engine.start()
    if not args.no_recenter and bool(config.get("calibration.recenter_on_start", True)):
        from gaze_estimation.direct.calibration import recenter

        shift = recenter(engine, ridge, sw, sh)
        if shift is None:
            print("[direct] Re-centring skipped (no face or aborted)")
        else:
            print(f"[direct] Re-centred: {shift[0]:+.0f} px, {shift[1]:+.0f} px")

    win = "VirtGaze"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    clicks: list = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONUP:
            clicks.append((x, y))

    cv2.setMouseCallback(win, on_mouse)
    refit_every = int(config.get("adaptation.ridge_refit_every", 5))
    added = pending = 0
    colors = {GazeState.FIXATION: (60, 220, 120), GazeState.SACCADE: (60, 160, 255),
              GazeState.BLINK: (120, 120, 120)}
    print("[direct] Running. Press q or Esc in the window to quit.")
    try:
        while True:
            canvas = np.full((sh, sw, 3), 25, dtype=np.uint8)
            sample = engine.get_latest_sample()
            if sample is not None and not args.no_preview:
                pv = cv2.resize(sample.frame, (320, 240))
                canvas[sh - 250:sh - 10, 10:330] = pv
            est = engine.get_latest_estimate()
            if est is not None:
                c = colors.get(est.fixation_state, (60, 220, 120))
                cv2.circle(canvas, (int(est.screen_x), int(est.screen_y)), 18, c, -1)
                cv2.circle(canvas, (int(est.screen_x), int(est.screen_y)), 24, (255, 255, 255), 2)
            hud = f"FPS {engine.fps:4.1f}"
            if args.implicit_calibration:
                hud += f"   clicks added {added}"
            face = sample is not None and sample.landmark_features is not None
            if not face:
                hud += "   (no face)"
            cv2.putText(canvas, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

            while args.implicit_calibration and clicks:
                cx, cy = clicks.pop(0)
                s = engine.get_latest_sample()
                if s is None or s.landmark_features is None:
                    continue
                ridge.add_samples(s.landmark_features, [[float(cx), float(cy)]])
                added += 1
                pending += 1
                if pending >= refit_every:
                    ridge.refit()
                    ridge.save(ridge_path)
                    pending = 0

            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        cv2.destroyAllWindows()
        if pending:
            ridge.refit()
            ridge.save(ridge_path)
        if args.implicit_calibration:
            print(f"[direct] {added} clicks added ({ridge.n_samples} samples); model saved.")


if __name__ == "__main__":
    main()
