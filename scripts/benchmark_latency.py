#!/usr/bin/env python3
"""Measure per-stage and end-to-end pipeline latency.

Usage:
    python scripts/benchmark_latency.py [--frames 200] [--config PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline latency benchmark")
    p.add_argument("--frames", type=int, default=200, help="Frames to benchmark")
    p.add_argument("--config", default="configs/example_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import numpy as np
    from gaze_estimation.config.config import Config
    from gaze_estimation.pipeline.pipeline import GazeEstimationPipeline
    from gaze_estimation.utils.logging import setup_logging

    config = Config.from_yaml(args.config)
    setup_logging(level="WARNING")

    pipeline = GazeEstimationPipeline(config=config)
    pipeline.start()

    latencies = []
    start_ts_map: dict = {}
    collected = 0
    t_start = time.time()

    print(f"[latency] Running {args.frames} frames …")
    while collected < args.frames and (time.time() - t_start) < 60:
        est = pipeline.get_latest_estimate()
        if est is not None and est.latency_ms > 0.0:
            latencies.append(est.latency_ms)
            collected += 1
        time.sleep(0.005)

    pipeline.stop()

    if not latencies:
        print("[latency] No data collected")
        return

    a = np.array(latencies)
    print(f"\n{'='*50}")
    print(f"  End-to-end latency ({len(a)} samples):")
    print(f"  Mean:      {a.mean():.1f} ms")
    print(f"  Median:    {np.median(a):.1f} ms")
    print(f"  95th %%:   {np.percentile(a, 95):.1f} ms")
    print(f"  Max:       {a.max():.1f} ms")
    print(f"  Pipeline FPS: {pipeline.fps:.1f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
