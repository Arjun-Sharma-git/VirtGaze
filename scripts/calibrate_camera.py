#!/usr/bin/env python3
"""Chessboard-based camera intrinsics calibration.

Usage:
    python scripts/calibrate_camera.py [--board-cols 9] [--board-rows 6]
        [--square-size 25.0] [--output data/camera_intrinsics.npz]
        [--camera 0] [--frames 30]

Hold a chessboard in front of the camera and press SPACE to capture a frame,
ESC to finish and compute intrinsics.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera intrinsics calibration")
    p.add_argument("--board-cols", type=int, default=9, help="Inner corners (columns)")
    p.add_argument("--board-rows", type=int, default=6, help="Inner corners (rows)")
    p.add_argument("--square-size", type=float, default=25.0, help="Square size mm")
    p.add_argument("--output", default="data/camera_intrinsics.npz")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--frames", type=int, default=30, help="Target number of frames")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    import cv2

    from gaze_estimation.utils.camera_calibration import calibrate_from_images, save_intrinsics

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[calibrate_camera] Cannot open camera {args.camera}")
        sys.exit(1)

    frames = []
    print(f"[calibrate_camera] Collecting up to {args.frames} frames.")
    print("  Press SPACE to capture, ESC to finish.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (args.board_cols, args.board_rows), flags=cv2.CALIB_CB_FAST_CHECK
        )
        if found:
            cv2.drawChessboardCorners(display, (args.board_cols, args.board_rows), corners, found)

        n = len(frames)
        cv2.putText(
            display,
            f"Captured: {n}/{args.frames}  [SPC=capture  ESC=done]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32 and found:  # SPACE
            frames.append(frame.copy())
            print(f"  Captured frame {len(frames)}/{args.frames}")
            if len(frames) >= args.frames:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) < 5:
        print(f"[calibrate_camera] Only {len(frames)} frames — need at least 5. Aborted.")
        sys.exit(1)

    print(f"[calibrate_camera] Computing intrinsics from {len(frames)} frames …")
    try:
        cm, dc, rms = calibrate_from_images(
            frames, args.board_cols, args.board_rows, args.square_size
        )
    except ValueError as e:
        print(f"[calibrate_camera] Calibration failed: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_intrinsics(args.output, cm, dc)
    print(f"[calibrate_camera] Saved to {args.output}  (RMS reprojection error: {rms:.4f} px)")


if __name__ == "__main__":
    main()
