#!/usr/bin/env python3
"""Real-time gaze tracker entry point.

Usage:
    python scripts/run_tracker.py [--config PATH] [--user USER_ID] [--debug]
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time 3D gaze tracker")
    p.add_argument("--config", default="configs/example_config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--user", default="default", help="User profile ID")
    p.add_argument("--debug", action="store_true", help="Show debug overlay")
    p.add_argument("--no-display", action="store_true", help="Suppress OpenCV window")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import here so the script works even without full package install
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from gaze_estimation.config.config import Config
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.utils.logging import setup_logging
    from gaze_estimation.visualization.overlay import OverlayRenderer

    config = Config.from_yaml(args.config)
    setup_logging(level=str(config.get("logging.level", "INFO")))

    profile_mgr = ProfileManager(
        profiles_dir=str(config.get("profile.profiles_dir", "profiles"))
    )

    pipeline = GazeEstimationPipeline(config=config)

    # Load existing calibration if available
    profile = profile_mgr.load_profile(args.user)
    if profile is not None and profile.mlp_weights_path:
        try:
            from gaze_estimation.model.trainer import MLPTrainer
            trainer = MLPTrainer()
            model, trainer = trainer.load(profile.mlp_weights_path)
            pipeline.set_model(model, trainer)
            pipeline.update_kappa(profile.kappa_yaw, profile.kappa_pitch)
            print(f"[tracker] Loaded calibration profile for user '{args.user}'")
        except Exception as exc:
            print(f"[tracker] Could not load calibration: {exc}")
    else:
        print(f"[tracker] No calibration profile for '{args.user}'. "
              "Run: python scripts/run_calibration.py --user", args.user)

    pipeline.start()
    renderer = OverlayRenderer(draw_iris=True, draw_mesh=args.debug)

    print("[tracker] Running. Press 'q' in the OpenCV window or Ctrl+C to stop.")
    t_last_fps_print = time.time()
    try:
        while True:
            estimate = pipeline.get_latest_estimate()

            if not args.no_display:
                # Grab the last frame if available (best-effort)
                frame_shown = False
                if estimate is not None:
                    dummy = None  # We don't hold a direct reference to the raw frame here
                if not frame_shown:
                    # Minimal display: show a blank window with gaze info
                    import numpy as np
                    blank = np.zeros((100, 600, 3), dtype="uint8")
                    if estimate is not None:
                        txt = (f"Gaze: ({estimate.screen_x:.0f}, {estimate.screen_y:.0f})  "
                               f"State: {estimate.fixation_state.value}  "
                               f"Conf: {estimate.confidence:.2f}  "
                               f"FPS: {pipeline.fps:.1f}")
                        cv2.putText(blank, txt, (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("Gaze Tracker", blank)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            if time.time() - t_last_fps_print > 5.0:
                print(f"[tracker] FPS={pipeline.fps:.1f}")
                t_last_fps_print = time.time()

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[tracker] Stopped.")


if __name__ == "__main__":
    main()
