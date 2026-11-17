#!/bin/bash
# One opt-in AE-only control. Does not alter existing campaigns/checkpoints.
set -euo pipefail
if (( $# != 1 )) || [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_ae_random_control.sh UNIQUE_TAG" >&2
  exit 2
fi
TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submission" >&2
  exit 2
fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CONVERTED="$BASE/data/train3M_v1/converted_v1"
BASELINE_AE="$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
CONTROL_OUT="$BASE/outputs/${TAG}_AE_random_control"
AE_EPOCHS="${AE_EPOCHS:-150}"
AE_PATIENCE="${AE_PATIENCE:-25}"
if [[ ! "$AE_EPOCHS" =~ ^[1-9][0-9]*$ || ! "$AE_PATIENCE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: positive integer AE_EPOCHS and AE_PATIENCE required" >&2
  exit 2
fi
for path in "$CONVERTED/FULL_VALIDATION_PASSED" "$BASELINE_AE" "$CONVERTED/train/sample.json" \
            "$CONVERTED/weight_train.npz" "$CONVERTED/test/sample.json" "$CONVERTED/weight_test.npz"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing input $path" >&2
    exit 2
  fi
done
if [[ -e "$CONTROL_OUT" ]]; then
  echo "ERROR: refusing to overwrite $CONTROL_OUT" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
export BASE CODE_DIR CODE_COMMIT CONVERTED BASELINE_AE CONTROL_OUT AE_EPOCHS AE_PATIENCE
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${TAG}_AEcontrol" --export=ALL \
  "$SCRIPT_DIR/submit_ae_random_control.sbatch")"
echo "AE-only job: $JOB"
echo "Baseline preserved: $BASELINE_AE"
echo "Output: $CONTROL_OUT"
echo "New AE: $CONTROL_OUT/ae_random_category/checkpoint_ae.pth"
echo "Log: $BASE/logs/${TAG}_AEcontrol-${JOB}.out"
echo "GPU request: gpu:1; no explicit partition, GPU model, or node constraint"
echo "One AE training, then frozen AE-only rate comparison; no NURD training."
