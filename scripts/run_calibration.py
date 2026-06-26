#!/usr/bin/env python3
"""Standalone calibration session.

Usage:
    python scripts/run_calibration.py [--user USER_ID] [--quick] [--config PATH]
"""
from __future__ import annotations

import argparse
import sys
import os
import threading


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gaze calibration session")
    p.add_argument("--user", default="default", help="User profile ID")
    p.add_argument("--quick", action="store_true", help="Run 5-point quick recalibration")
    p.add_argument("--config", default="configs/example_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from gaze_estimation.config.config import Config
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.visualization.calibration_ui import CalibrationUI
    from gaze_estimation.utils.logging import setup_logging

    config = Config.from_yaml(args.config)
    setup_logging(level="INFO")

    profile_mgr = ProfileManager(
        profiles_dir=str(config.get("profile.profiles_dir", "profiles"))
    )

    import screeninfo
    try:
        monitor = screeninfo.get_monitors()[0]
        sw, sh = monitor.width, monitor.height
    except Exception:
        sw, sh = 1920, 1080

    pipeline = GazeEstimationPipeline(config=config, screen_width=sw, screen_height=sh)
    pipeline.start()

    weights_path = profile_mgr.get_profile_weights_path(args.user)
    done_event = threading.Event()

    if args.quick:
        from gaze_estimation.calibration.quick_calibration import QuickCalibration
        from gaze_estimation.model.trainer import MLPTrainer
        profile = profile_mgr.load_profile(args.user)
        if profile is None or not os.path.exists(profile.mlp_weights_path):
            print("[calibration] No existing profile — running full calibration instead.")
            args.quick = False
        else:
            trainer = MLPTrainer()
            model, trainer = trainer.load(profile.mlp_weights_path)
            qc = QuickCalibration(sw, sh, trainer=trainer)
            ui = CalibrationUI(screen_width=sw, screen_height=sh)
            ui.start()

            def _on_target(tx, ty):
                ui.set_target(tx, ty)

            def _on_progress(frac):
                ui.set_progress(frac)
                ui.update()

            result = qc.run(
                gaze_queue=pipeline.get_gaze_queue(),
                existing_model=model,
                on_target_change=_on_target,
                on_progress=_on_progress,
                mlp_save_path=weights_path,
            )
            ui.stop()
            pipeline.update_kappa(result.kappa_yaw, result.kappa_pitch)
            if result.mlp_weights_path:
                pipeline.set_model(model, trainer)
            done_event.set()

    if not args.quick and not done_event.is_set():
        from gaze_estimation.calibration.calibration_engine import CalibrationEngine
        from gaze_estimation.model.trainer import MLPTrainer
        engine = CalibrationEngine(sw, sh)
        ui = CalibrationUI(screen_width=sw, screen_height=sh)
        ui.start()

        def _on_target(tx, ty):
            ui.set_target(tx, ty)
            ui.update()

        def _on_progress(frac):
            ui.set_progress(frac)
            ui.update()

        result = engine.run(
            gaze_queue=pipeline.get_gaze_queue(),
            on_target_change=_on_target,
            on_progress=_on_progress,
            mlp_save_path=weights_path,
        )
        ui.stop()

        from gaze_estimation.utils.camera_calibration import estimate_camera_matrix, zero_dist_coeffs
        cm_w = int(config.get("camera.width", 1280))
        cm_h = int(config.get("camera.height", 720))
        cm = estimate_camera_matrix(cm_w, cm_h)
        dc = zero_dist_coeffs()

        profile_mgr.save_calibration(
            user_id=args.user,
            kappa_yaw=result.kappa_yaw,
            kappa_pitch=result.kappa_pitch,
            eyeball_radius=result.eyeball_radius,
            mlp_path=weights_path,
            screen_resolution=(sw, sh),
            camera_name=f"camera_{int(config.get('camera.index', 0))}",
            camera_matrix=cm,
            dist_coeffs=dc,
            calibration_samples=len(result.samples),
        )
        print(f"[calibration] Saved calibration for user '{args.user}'")

    pipeline.stop()
    print("[calibration] Done.")


if __name__ == "__main__":
    main()
