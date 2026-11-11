#!/bin/bash
# Controlled lambda=5 stress test. All other settings come from the exact
# lambda=2.5 launcher, including the physical AE and NURD-off warm start.
set -euo pipefail
if (( $# != 2 )); then
  echo "Usage: bash slurm/launch_physics_decay8_lambda5_control.sh BASE_RUN NEW_RUN_TAG" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/launch_physics_decay8_lambda25_control.sh" "$1" "$2" 5.0
