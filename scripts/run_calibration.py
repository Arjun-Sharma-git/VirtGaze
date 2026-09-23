#!/usr/bin/env python3
"""Standalone calibration session.

Usage:
    python scripts/run_calibration.py [--user USER_ID] [--quick] [--config PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
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
    from gaze_estimation.utils.camera_calibration import (
        estimate_camera_matrix,
        zero_dist_coeffs,
    )
    from gaze_estimation.utils.logging import setup_logging
    from gaze_estimation.utils.screen import detect_screen_resolution
    from gaze_estimation.visualization.calibration_ui import CalibrationUI

    config = Config.from_yaml(args.config)
    setup_logging(level="INFO")

    profile_mgr = ProfileManager(
        profiles_dir=str(config.get("profile.profiles_dir", "profiles"))
    )

    sw, sh = detect_screen_resolution()

    pipeline = GazeEstimationPipeline(config=config, screen_width=sw, screen_height=sh)
    pipeline.start()

    weights_path = profile_mgr.get_profile_weights_path(args.user)
    bias_path = profile_mgr.get_profile_bias_path(args.user)
    done_event = threading.Event()

    # Shared intrinsics so every calibration path persists the same data.
    cm_w = int(config.get("camera.width", 1280))
    cm_h = int(config.get("camera.height", 720))
    camera_matrix = estimate_camera_matrix(cm_w, cm_h)
    dist_coeffs = zero_dist_coeffs()
    camera_name = f"camera_{int(config.get('camera.index', 0))}"

    bias_cfg = config.section("bias_map")
    bias_cols = int(bias_cfg.get("cols", 40))
    bias_rows = int(bias_cfg.get("rows", 20))
    bias_smoothing = float(bias_cfg.get("smoothing_sigma", 1.0))

    def _apply_to_running_pipeline(result) -> None:
        """Attach the freshly-trained model + kappa to the live pipeline."""
        pipeline.update_kappa(result.kappa_yaw, result.kappa_pitch)
        if result.mlp_weights_path and os.path.exists(result.mlp_weights_path):
            from gaze_estimation.model.trainer import MLPTrainer
            trainer = MLPTrainer()
            model, trainer = trainer.load(result.mlp_weights_path)
            pipeline.set_model(model, trainer)
        if result.bias_map is not None:
            pipeline.set_bias_map(result.bias_map)

    def _persist(result) -> None:
        """Persist calibration results so the next session reloads them."""
        saved_bias_path = None
        if result.bias_map is not None:
            try:
                result.bias_map.save(bias_path)
                saved_bias_path = bias_path
            except Exception as exc:
                print(f"[calibration] Could not save bias map: {exc}")

        profile_mgr.save_calibration(
            user_id=args.user,
            kappa_yaw=result.kappa_yaw,
            kappa_pitch=result.kappa_pitch,
            eyeball_radius=result.eyeball_radius,
            mlp_path=weights_path,
            screen_resolution=(sw, sh),
            camera_name=camera_name,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            calibration_samples=len(result.samples),
            bias_map_path=saved_bias_path,
            grid=(5, 5) if not args.quick else (3, 3),
        )
        print(f"[calibration] Saved calibration for user '{args.user}'")

    if args.quick:
        from gaze_estimation.calibration.quick_calibration import QuickCalibration
        from gaze_estimation.model.trainer import MLPTrainer
        profile = profile_mgr.load_profile(args.user)
        if profile is None or not profile.mlp_weights_path or not os.path.exists(profile.mlp_weights_path):
            print("[calibration] No existing profile — running full calibration instead.")
            args.quick = False
        else:
            trainer = MLPTrainer()
            model, trainer = trainer.load(profile.mlp_weights_path)
            qc = QuickCalibration(sw, sh, trainer=trainer)
            ui = CalibrationUI(screen_width=sw, screen_height=sh)
            ui.start()

            def _quick_on_target(tx, ty):
                ui.set_target(tx, ty)
                ui.update()

            def _quick_on_progress(frac):
                ui.set_progress(frac)
                ui.update()

            result = qc.run(
                gaze_queue=pipeline.get_gaze_queue(),
                existing_model=model,
                on_target_change=_quick_on_target,
                on_progress=_quick_on_progress,
                mlp_save_path=weights_path,
            )
            ui.stop()
            _apply_to_running_pipeline(result)
            _persist(result)
            done_event.set()

    if not args.quick and not done_event.is_set():
        from gaze_estimation.calibration.calibration_engine import CalibrationEngine
        engine = CalibrationEngine(
            sw, sh,
            camera_matrix=camera_matrix,
            frame_width=cm_w,
            bias_map_cols=bias_cols,
            bias_map_rows=bias_rows,
            bias_map_smoothing=bias_smoothing,
        )
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
        _apply_to_running_pipeline(result)
        _persist(result)

    pipeline.stop()
    print("[calibration] Done.")


if __name__ == "__main__":
    main()
