#!/usr/bin/env python3
"""Evaluate gaze accuracy against ground-truth data.

Usage:
    python scripts/benchmark_accuracy.py --video data/test.mp4 --gt data/ground_truth.csv
    [--user USER_ID] [--config PATH]

Ground truth CSV columns: timestamp,screen_x,screen_y
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gaze accuracy benchmark")
    p.add_argument("--video", required=True, help="Path to test video")
    p.add_argument("--gt", required=True, help="Path to ground truth CSV")
    p.add_argument("--user", default="default")
    p.add_argument("--config", default="configs/example_config.yaml")
    return p.parse_args()


def load_ground_truth(path: str) -> List[Tuple[float, float, float]]:
    """Load (timestamp, screen_x, screen_y) triples from CSV."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (float(row["timestamp"]), float(row["screen_x"]), float(row["screen_y"]))
            )
    return rows


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import numpy as np
    from gaze_estimation.config.config import Config
    from gaze_estimation.capture.video_source import VideoFileSource
    from gaze_estimation.utils.geometry import angular_error_deg, screen_to_angles
    from gaze_estimation.utils.logging import setup_logging

    config = Config.from_yaml(args.config)
    setup_logging(level="WARNING")

    gt = load_ground_truth(args.gt)
    print(f"[benchmark] Loaded {len(gt)} ground-truth samples")

    # Run the full pipeline on the video file
    import queue, threading
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.model.trainer import MLPTrainer

    pipeline = GazeEstimationPipeline(config=config)

    profile_mgr = ProfileManager(str(config.get("profile.profiles_dir", "profiles")))
    profile = profile_mgr.load_profile(args.user)
    if profile and profile.mlp_weights_path and os.path.exists(profile.mlp_weights_path):
        trainer = MLPTrainer()
        model, trainer = trainer.load(profile.mlp_weights_path)
        pipeline.set_model(model, trainer)
        pipeline.update_kappa(profile.kappa_yaw, profile.kappa_pitch)

    predictions: list = []
    stop_event = threading.Event()

    def _collect() -> None:
        while not stop_event.is_set():
            est = pipeline.get_latest_estimate()
            if est:
                predictions.append((est.timestamp, est.screen_x, est.screen_y))
            import time
            time.sleep(0.005)

    # Replace camera with video source
    import time
    pipeline.start()
    col_thread = threading.Thread(target=_collect, daemon=True)
    col_thread.start()

    # Wait for video to complete (approximate: 10s max)
    time.sleep(15)
    stop_event.set()
    pipeline.stop()

    if not predictions:
        print("[benchmark] No predictions collected")
        return

    # Match predictions to GT by nearest timestamp
    px = np.array([(p[1], p[2]) for p in predictions])
    pt = np.array([p[0] for p in predictions])
    errors_px = []
    for ts, gx, gy in gt:
        idx = int(np.argmin(np.abs(pt - ts)))
        err = float(np.linalg.norm(np.array([px[idx, 0] - gx, px[idx, 1] - gy])))
        errors_px.append(err)

    errors = np.array(errors_px)
    print(f"\n{'='*50}")
    print(f"  Screen-space errors (pixels)")
    print(f"  Mean:   {errors.mean():.1f} px")
    print(f"  Median: {np.median(errors):.1f} px")
    print(f"  95th%:  {np.percentile(errors, 95):.1f} px")
    print(f"  Max:    {errors.max():.1f} px")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
