#!/usr/bin/env bash
# Run the tracker fullscreen with the saved calibration (no calibration step).
#   ./run.sh [user] [extra run_tracker.py options, e.g. --implicit-calibration]
# Uses .venv and configs/example_config_rocm.yaml (override with CONFIG=...).
set -euo pipefail
cd "$(dirname "$0")"
USER_ID="${1:-$USER}"
shift || true
CONFIG="${CONFIG:-configs/example_config_rocm.yaml}"

if [[ ! -f "profiles/$USER_ID/profile.json" ]]; then
  echo "No calibration for '$USER_ID' yet: run ./calibrate_and_run.sh $USER_ID first." >&2
  exit 1
fi
exec .venv/bin/python scripts/run_tracker.py --user "$USER_ID" --config "$CONFIG" --fullscreen "$@"
