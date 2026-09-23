#!/usr/bin/env python3
"""Real-time gaze tracker entry point.

Usage:
    python scripts/run_tracker.py [--config PATH] [--user USER_ID] [--debug]
        [--implicit-calibration] [--no-display]

``--implicit-calibration`` enables click-based online adaptation: each
deliberate left-click in the (fullscreen) preview window is treated as a
gaze-directed click and used to fine-tune the personalised model in the
background.  The window is fullscreen so mouse coordinates match screen pixels.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

WINDOW_NAME = "Gaze Tracker"
# Throttling for logging.save_debug_frames: at most one frame every 0.5 s and
# 200 files total, so a long session cannot fill the disk.
_DEBUG_SAVE_INTERVAL_SEC = 0.5
_MAX_DEBUG_FRAMES = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time 3D gaze tracker")
    p.add_argument("--config", default="configs/example_config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--user", default="default", help="User profile ID")
    p.add_argument("--debug", action="store_true", help="Draw face mesh in the overlay")
    p.add_argument("--no-display", action="store_true", help="Suppress the preview window")
    p.add_argument("--implicit-calibration", action="store_true",
                   help="Enable click-based online adaptation (fullscreen window)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from gaze_estimation.config.config import Config
    from gaze_estimation.correction.bias_map import BiasMap
    from gaze_estimation.model.backend import create_predictor
    from gaze_estimation.model.trainer import MLPTrainer
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.utils.camera_calibration import load_intrinsics
    from gaze_estimation.utils.logging import setup_logging
    from gaze_estimation.utils.screen import detect_screen_resolution
    from gaze_estimation.visualization.overlay import OverlayRenderer

    config = Config.from_yaml(args.config)
    setup_logging(level=str(config.get("logging.level", "INFO")))

    screen_w, screen_h = detect_screen_resolution()
    profile_mgr = ProfileManager(
        profiles_dir=str(config.get("profile.profiles_dir", "profiles"))
    )
    pipeline = GazeEstimationPipeline(
        config=config, screen_width=screen_w, screen_height=screen_h
    )

    # ── Camera intrinsics (chessboard file, else the stored profile) ──────
    intrinsics_path = str(config.get("camera.intrinsics_path", "") or "")
    profile = profile_mgr.load_profile(args.user)
    if intrinsics_path and os.path.exists(intrinsics_path):
        try:
            cam, dist = load_intrinsics(intrinsics_path)
            pipeline.set_camera_intrinsics(cam, dist)
            print(f"[tracker] Loaded camera intrinsics from {intrinsics_path}")
        except Exception as exc:
            print(f"[tracker] Could not load intrinsics: {exc}")
    elif profile is not None and profile.camera_intrinsics:
        pipeline.set_camera_intrinsics(
            np.array(profile.camera_intrinsics, dtype=np.float64),
            np.array(profile.dist_coeffs, dtype=np.float64),
        )

    # ── Staleness hint (profile.quick_calib_interval_days) ────────────────
    if profile is not None:
        if profile_mgr.needs_quick_calibration(
            args.user,
            days_since_last=int(config.get("profile.quick_calib_interval_days", 7)),
        ):
            print(
                "[tracker] Profile is older than "
                f"{int(config.get('profile.quick_calib_interval_days', 7))} days — "
                f"consider: python scripts/run_calibration.py --user {args.user} --quick"
            )

    # ── Personalised model ────────────────────────────────────────────────
    model = trainer = None
    weights_path = None
    if profile is not None and profile.mlp_weights_path:
        weights_path = profile.mlp_weights_path
        try:
            trainer = MLPTrainer()
            model, trainer = trainer.load(weights_path)
            print(f"[tracker] Loaded calibration profile for user '{args.user}'")
        except Exception as exc:
            model = trainer = None
            print(f"[tracker] Could not load calibration: {exc}")

    predictor = None
    backend_name = "torch"
    if model is not None:
        backend = str(config.get("inference.backend", "torch")).lower()
        model_path = None
        if backend == "onnx":
            model_path = (
                profile.onnx_model_path or profile_mgr.get_profile_onnx_path(args.user)
            )
            if not os.path.exists(model_path):
                print(f"[tracker] ONNX model not found at {model_path} "
                      "(export it with scripts/export_model.py)")
        elif backend == "tensorrt":
            model_path = str(config.get("inference.tensorrt_engine_path", "")) or None

        predictor, backend_name = create_predictor(
            backend,
            model=model,
            model_path=model_path,
            device=str(config.get("inference.device", "CPU")),
            input_dim=int(config.get("mlp.input_dim", 34)),
        )
        if backend_name != backend:
            print(f"[tracker] Backend '{backend}' unavailable — using '{backend_name}'")

        pipeline.set_model(model, trainer, predictor=predictor, backend=backend_name)
        pipeline.update_kappa(profile.kappa_yaw, profile.kappa_pitch)

        # Residual spatial bias correction (built during calibration)
        bias_path = profile_mgr.get_profile_bias_path(args.user)
        if os.path.exists(bias_path):
            try:
                bias_map = BiasMap()
                bias_map.load(bias_path)
                pipeline.set_bias_map(bias_map)
                print("[tracker] Applied spatial bias correction")
            except Exception as exc:
                print(f"[tracker] Could not load bias map: {exc}")
    else:
        print(f"[tracker] No calibration profile for '{args.user}'. "
              f"Run: python scripts/run_calibration.py --user {args.user}")

    # ── Optional implicit (click-based) calibration ───────────────────────
    online_trainer = None
    implicit = None
    if args.implicit_calibration:
        if model is None or trainer is None:
            print("[tracker] Implicit calibration needs an existing profile — skipping.")
        else:
            from gaze_estimation.adaptation.adaptation_buffer import AdaptationBuffer
            from gaze_estimation.adaptation.online_trainer import OnlineTrainer
            from gaze_estimation.calibration.implicit_calibration import ImplicitCalibration

            buffer = AdaptationBuffer(
                max_size=int(config.get("adaptation.buffer_max_size", 5000))
            )
            online_trainer = OnlineTrainer(
                model,
                buffer,
                trainer,
                retrain_threshold=int(config.get("adaptation.retrain_threshold", 50)),
                retrain_epochs=int(config.get("adaptation.retrain_epochs", 2)),
                learning_rate=float(config.get("adaptation.retrain_lr", 1e-4)),
                batch_size=int(config.get("adaptation.retrain_batch_size", 16)),
                screen_width=screen_w,
                screen_height=screen_h,
            )

            def _on_retrain() -> None:
                """Publish the fine-tuned model and persist its weights."""
                pipeline.set_model(
                    online_trainer.model, trainer,
                    predictor=predictor, backend=backend_name,
                )
                if weights_path and bool(config.get("profile.auto_save", True)):
                    try:
                        trainer.save(online_trainer.model, weights_path)
                    except Exception as exc:
                        print(f"[tracker] Could not save adapted weights: {exc}")

            implicit = ImplicitCalibration(
                online_trainer,
                click_velocity_threshold=float(
                    config.get("adaptation.click_velocity_threshold", 500.0)
                ),
                on_retrain=_on_retrain,
            )
            print("[tracker] Implicit calibration enabled — click where you are looking.")

    pipeline.start()

    # ── Preview window ────────────────────────────────────────────────────
    log_cfg = config.section("logging")
    log_fps = bool(log_cfg.get("log_fps", True))
    log_latency = bool(log_cfg.get("log_latency", True))
    renderer = OverlayRenderer(
        draw_iris=True,
        draw_mesh=args.debug,
        draw_fps=log_fps,
        draw_latency=log_latency,
    )
    debug_dir = (
        "debug_frames" if bool(log_cfg.get("save_debug_frames", False)) else None
    )
    debug_saved = 0
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[tracker] Saving debug frames to {debug_dir}/ "
              f"(max {_MAX_DEBUG_FRAMES} @ {1 / _DEBUG_SAVE_INTERVAL_SEC:.0f} fps)")
    pending_clicks: list = []

    def _on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONUP:
            pending_clicks.append((x, y))

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if args.implicit_calibration:
            # Fullscreen so window pixels == screen pixels for click targets
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

    print("[tracker] Running. Press 'q' in the window or Ctrl+C to stop.")
    t_last_fps_print = time.perf_counter()
    t_last_debug_save = time.perf_counter()
    clicks_used = 0
    try:
        while True:
            estimate = pipeline.get_latest_estimate()
            mesh_packet = pipeline.get_latest_mesh_packet()

            # ── Implicit calibration: consume queued clicks ────────────────
            while implicit is not None and pending_clicks:
                cx, cy = pending_clicks.pop(0)
                packet = pipeline.get_latest_gaze_packet()
                if packet is not None and packet.confidence >= 0.3:
                    if implicit.on_click(dict(packet.features), float(cx), float(cy)):
                        clicks_used += 1
                else:
                    print("[tracker] Click ignored: no confident gaze sample")

            if not args.no_display:
                if mesh_packet is not None:
                    frame = mesh_packet.frame
                else:
                    frame = np.zeros((240, 640, 3), dtype=np.uint8)
                display = renderer.render(
                    frame,
                    estimate=estimate,
                    mesh_packet=mesh_packet,
                    fps=pipeline.fps,
                    latency_ms=estimate.latency_ms if estimate else 0.0,
                )
                # ── Optional debug frame dump (logging.save_debug_frames) ──
                if debug_dir is not None and debug_saved < _MAX_DEBUG_FRAMES:
                    if time.perf_counter() - t_last_debug_save >= _DEBUG_SAVE_INTERVAL_SEC:
                        t_last_debug_save = time.perf_counter()
                        path = os.path.join(debug_dir, f"frame_{debug_saved:06d}.jpg")
                        try:
                            cv2.imwrite(path, display)
                            debug_saved += 1
                        except Exception as exc:
                            print(f"[tracker] Could not write debug frame: {exc}")
                if implicit is not None:
                    # Match the screen resolution so clicks map 1:1 to pixels
                    display = cv2.resize(display, (screen_w, screen_h))
                    info = (f"clicks={clicks_used} accepted  "
                            f"retrains={implicit.retrain_count}")
                    cv2.putText(display, info, (10, display.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                try:
                    cv2.imshow(WINDOW_NAME, display)
                except cv2.error as exc:
                    print(f"[tracker] Display unavailable ({exc}) — disabling window")
                    args.no_display = True
                else:
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

            if log_fps and time.perf_counter() - t_last_fps_print > 5.0:
                print(f"[tracker] FPS={pipeline.fps:.1f}")
                t_last_fps_print = time.perf_counter()

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if implicit is not None:
            print(f"[tracker] Implicit calibration: {implicit.total_clicks} clicks "
                  f"accepted, {implicit.skipped_clicks} skipped, "
                  f"{implicit.retrain_count} fine-tunes")
        print("[tracker] Stopped.")


if __name__ == "__main__":
    main()
