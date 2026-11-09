#!/bin/bash
# One-time Mequinna domain-transfer view; normal decay8 pipeline is untouched.
set -euo pipefail
if (( $# != 2 )); then
  echo "Usage: bash slurm/launch_decay8_parent4_legacy_eval.sh NURD_EXPERIMENT UNIQUE_OUTPUT_TAG" >&2
  exit 2
fi
NURD_EXP="$1"
OUTPUT_TAG="$2"
for tag in "$NURD_EXP" "$OUTPUT_TAG"; do
  if [[ ! "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid tag: $tag" >&2
    exit 2
  fi
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit tracked changes before submitting" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
AE_CKPT="$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
MODEL_CKPT="$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_main.pth.tar"
LEGACY_TRAIN="$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt"
LEGACY_TEST="$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt"
LEGACY_TRAIN_WEIGHTS="$MEQUINNA_DIR/weight_train.pt"
LEGACY_TEST_WEIGHTS="$MEQUINNA_DIR/weight_test.pt"
LEGACY_OUT="$BASE/outputs/${OUTPUT_TAG}_legacy_parent4"
for path in "$AE_CKPT" "$MODEL_CKPT" "$LEGACY_TRAIN" "$LEGACY_TEST" \
            "$LEGACY_TRAIN_WEIGHTS" "$LEGACY_TEST_WEIGHTS"; do
  test -s "$path" || { echo "Missing $path" >&2; exit 2; }
done
test ! -e "$LEGACY_OUT"
export BASE CODE_DIR CODE_COMMIT AE_CKPT MODEL_CKPT LEGACY_TRAIN LEGACY_TEST
export LEGACY_TRAIN_WEIGHTS LEGACY_TEST_WEIGHTS LEGACY_OUT
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${OUTPUT_TAG}_legacy_parent4" \
  --export=ALL "$SCRIPT_DIR/submit_decay8_parent4_legacy_eval.sbatch")"
echo "Job: $JOB"
echo "Output: $LEGACY_OUT"
echo "Log: $BASE/logs/${OUTPUT_TAG}_legacy_parent4-${JOB}.out"
