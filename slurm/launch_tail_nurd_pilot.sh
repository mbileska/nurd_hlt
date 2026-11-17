#!/bin/bash
# Matched, fixed-update continuations; all original checkpoints stay untouched.
set -euo pipefail
if (( $# != 1 )) || [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_tail_nurd_pilot.sh UNIQUE_TAG" >&2
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
SOURCE_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_train3m_physics_decay8_lambda1_20261107_132516/checkpoint_main.pth.tar"
ORIGINAL_AE="$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
AE_CACHE_DIR="${AE_CACHE_DIR:-$BASE/outputs/ae_phys_randomsplit_20261117_150546_AE_random_control}"
PILOT_OUT="$BASE/outputs/${TAG}_tail_nurd_pilot"
PILOT_UPDATES="${PILOT_UPDATES:-4000}"
PILOT_BATCH_SIZE="${PILOT_BATCH_SIZE:-512}"
if [[ ! "$PILOT_UPDATES" =~ ^[1-9][0-9]*$ || ! "$PILOT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || (( PILOT_BATCH_SIZE < 2 )); then
  echo "ERROR: positive PILOT_UPDATES and PILOT_BATCH_SIZE >= 2 required" >&2
  exit 2
fi
for path in "$CONVERTED/FULL_VALIDATION_PASSED" "$SOURCE_CKPT" "$ORIGINAL_AE" \
            "$CONVERTED/train/sample.json" "$CONVERTED/weight_train.npz" \
            "$CONVERTED/test/sample.json" "$CONVERTED/weight_test.npz" \
            "$AE_CACHE_DIR/AE_control_summary.json" "$AE_CACHE_DIR/frozen_AE_scores.npz"; do
  if [[ ! -s "$path" && ! -e "$path" ]]; then
    echo "ERROR: missing input $path" >&2
    exit 2
  fi
done
if [[ -e "$PILOT_OUT" ]]; then
  echo "ERROR: refusing to overwrite $PILOT_OUT" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
export BASE CODE_DIR CODE_COMMIT CONVERTED SOURCE_CKPT ORIGINAL_AE AE_CACHE_DIR PILOT_OUT PILOT_UPDATES PILOT_BATCH_SIZE
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${TAG}_tailpilot" --export=ALL \
  --output="$BASE/logs/${TAG}_tailpilot-%j.out" --error="$BASE/logs/${TAG}_tailpilot-%j.err" \
  "$SCRIPT_DIR/submit_tail_nurd_pilot.sbatch")"
echo "Pilot job: $JOB"
echo "Output: $PILOT_OUT"
echo "Log: $BASE/logs/${TAG}_tailpilot-${JOB}.out"
echo "Source: $SOURCE_CKPT (copied, never overwritten)"
echo "Updates per arm: $PILOT_UPDATES; batch: $PILOT_BATCH_SIZE; lambda=1; SupCon=0.3"
echo "Original AE frozen; AE-unseen calibration; common outer held-out check only after both final checkpoints."
echo "GPU request: gpu:1; no explicit partition, GPU model, or node constraint"
