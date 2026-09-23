#!/usr/bin/env python3
"""Export a trained GazeMLP to ONNX (or TensorRT).

Usage:
    python scripts/export_model.py --user USER_ID [--output models/out.onnx] [--format onnx|tensorrt]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export GazeMLP to ONNX / TensorRT")
    p.add_argument("--user", default="default")
    p.add_argument("--output", default=None, help="Destination file path")
    p.add_argument("--format", choices=["onnx", "tensorrt"], default="onnx")
    p.add_argument("--config", default="configs/example_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from gaze_estimation.config.config import Config
    from gaze_estimation.model.onnx_export import export_to_onnx
    from gaze_estimation.model.trainer import MLPTrainer
    from gaze_estimation.profile.profile_manager import ProfileManager

    config = Config.from_yaml(args.config)
    profile_mgr = ProfileManager(profiles_dir=str(config.get("profile.profiles_dir", "profiles")))
    profile = profile_mgr.load_profile(args.user)
    if profile is None:
        print(f"[export] No profile found for user '{args.user}'")
        sys.exit(1)

    weights_path = profile.mlp_weights_path
    if not weights_path or not os.path.exists(weights_path):
        print(f"[export] Weights file not found: {weights_path}")
        sys.exit(1)

    trainer = MLPTrainer()
    model, _ = trainer.load(weights_path)

    out_path = args.output or profile_mgr.get_profile_onnx_path(args.user)

    if args.format == "onnx":
        export_to_onnx(model, out_path, input_dim=model.input_dim)
        print(f"[export] ONNX model saved to {out_path}")
    else:
        # Convert via trtexec.  Run it as an argv list (no shell), so a path
        # containing spaces or shell metacharacters cannot be reinterpreted as
        # a command.
        onnx_path = out_path.replace(".trt", ".onnx")
        export_to_onnx(model, onnx_path, input_dim=model.input_dim)
        trt_path = out_path if out_path.endswith(".trt") else out_path.replace(".onnx", ".trt")
        cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={trt_path}", "--fp16"]
        print(f"[export] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print("[export] trtexec not found.  Is TensorRT installed and on PATH?")
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            print(f"[export] trtexec failed (exit {exc.returncode}).  Is TensorRT installed?")
            sys.exit(1)
        print(f"[export] TensorRT engine saved to {trt_path}")


if __name__ == "__main__":
    main()
