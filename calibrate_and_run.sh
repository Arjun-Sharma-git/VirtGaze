#!/usr/bin/env bash
# Full calibration, then the gaze tracker fullscreen.
#   ./calibrate_and_run.sh [user] [extra run_direct.py options]
# Uses .venv and configs/example_config_rocm.yaml (override with CONFIG=...).
set -euo pipefail
cd "$(dirname "$0")"
USER_ID="${1:-$USER}"
shift || true
CONFIG="${CONFIG:-configs/example_config_rocm.yaml}"

.venv/bin/python scripts/direct_calibrate.py --user "$USER_ID" --config "$CONFIG"
exec .venv/bin/python scripts/run_direct.py --user "$USER_ID" --config "$CONFIG" "$@"
