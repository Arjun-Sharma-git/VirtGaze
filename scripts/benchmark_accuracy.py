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


def match_predictions(predictions, ground_truth):
    """Pair collected predictions with ground-truth rows.

    Prefers nearest-timestamp matching when the two time ranges overlap.  Video
    replay produces monotonic timestamps which do not relate to the CSV's
    wall-clock times, so in that case the two sequences are aligned by index.

    Returns:
        (pred_xy, gt_xy) arrays of shape (N, 2).
    """
    import numpy as np

    px = np.array([(p[1], p[2]) for p in predictions], dtype=np.float64)
    pt = np.array([p[0] for p in predictions], dtype=np.float64)
    gt_ts = np.array([g[0] for g in ground_truth], dtype=np.float64)
    gt_xy = np.array([(g[1], g[2]) for g in ground_truth], dtype=np.float64)

    overlap = (
        pt.size > 0 and gt_ts.size > 0
        and pt.max() >= gt_ts.min() and pt.min() <= gt_ts.max()
    )
    if overlap:
        idx = np.array([int(np.argmin(np.abs(pt - ts))) for ts in gt_ts])
    else:
        print("[benchmark] Timestamp ranges do not overlap — aligning by index")
        idx = np.linspace(0, len(predictions) - 1, len(ground_truth)).astype(int)
    return px[idx], gt_xy


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import threading
    import time

    import numpy as np

    from gaze_estimation.capture.video_source import VideoFileSource
    from gaze_estimation.config.config import Config
    from gaze_estimation.model.trainer import MLPTrainer
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.profile.profile_manager import ProfileManager
    from gaze_estimation.utils.logging import setup_logging

    config = Config.from_yaml(args.config)
    setup_logging(level="WARNING")

    gt = load_ground_truth(args.gt)
    print(f"[benchmark] Loaded {len(gt)} ground-truth samples")

    pipeline = GazeEstimationPipeline(config=config)

    profile_mgr = ProfileManager(str(config.get("profile.profiles_dir", "profiles")))
    profile = profile_mgr.load_profile(args.user)
    if profile and profile.mlp_weights_path and os.path.exists(profile.mlp_weights_path):
        trainer = MLPTrainer()
        model, trainer = trainer.load(profile.mlp_weights_path)
        pipeline.set_model(model, trainer)
        pipeline.update_kappa(profile.kappa_yaw, profile.kappa_pitch)

    # Replay the video file instead of opening a webcam.
    source = VideoFileSource(
        video_path=args.video,
        output_queue=pipeline.frame_queue,
        stop_event=pipeline.stop_event,
        loop=False,
        realtime=False,
    )

    predictions: list = []
    stop_event = threading.Event()
    last_ts: list = [None]

    def _collect() -> None:
        while not stop_event.is_set():
            est = pipeline.get_latest_estimate()
            # Only record each estimate once (the poller runs faster than the
            # pipeline produces new estimates).
            if est is not None and est.timestamp != last_ts[0]:
                predictions.append((est.timestamp, est.screen_x, est.screen_y))
                last_ts[0] = est.timestamp
            time.sleep(0.005)

    pipeline.start(frame_source=source)
    col_thread = threading.Thread(target=_collect, daemon=True)
    col_thread.start()

    # Wait for the video to finish replaying (bounded).
    deadline = time.time() + 300.0
    while source.is_alive() and time.time() < deadline and not stop_event.is_set():
        time.sleep(0.1)
    time.sleep(1.0)  # let the pipeline flush the final frames

    stop_event.set()
    col_thread.join(timeout=1.0)
    pipeline.stop()

    if not predictions:
        print("[benchmark] No predictions collected")
        return

    pred_xy, gt_xy = match_predictions(predictions, gt)
    errors = np.linalg.norm(pred_xy - gt_xy, axis=1)

    print(f"\n{'='*50}")
    print(f"  Screen-space errors (pixels, n={len(errors)})")
    print(f"  Mean:   {errors.mean():.1f} px")
    print(f"  Median: {np.median(errors):.1f} px")
    print(f"  95th%:  {np.percentile(errors, 95):.1f} px")
    print(f"  Max:    {errors.max():.1f} px")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
