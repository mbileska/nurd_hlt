#!/bin/bash
# Preparation only: CPU AE scoring and class/ESS report. No training submission.
set -euo pipefail
if (( $# != 1 )) || [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_tail_pilot_preparation.sh UNIQUE_TAG" >&2
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
MODEL_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_train3m_physics_decay8_lambda1_20261107_132516/checkpoint_main.pth.tar"
AE_CKPT="$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
OUTDIR="$BASE/outputs/${TAG}_pilot_preparation"
CODE_COMMIT="$(git rev-parse HEAD)"
test -e "$CONVERTED/FULL_VALIDATION_PASSED"
for path in "$MODEL_CKPT" "$AE_CKPT" "$CONVERTED/train/sample.json" "$CONVERTED/weight_train.npz"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite $OUTDIR" >&2
  exit 2
fi
export BASE CODE_DIR CODE_COMMIT CONVERTED MODEL_CKPT AE_CKPT OUTDIR
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${TAG}_prep" --export=ALL \
  "$SCRIPT_DIR/submit_tail_pilot_preparation.sbatch")"
echo "Preparation job: $JOB"
echo "Output: $OUTDIR"
echo "Log: $BASE/logs/${TAG}_prep-${JOB}.out"
echo "CPU AE inference only; no held-out sample, classifier inference, or training."
