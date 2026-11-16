#!/bin/bash
# Usual full-sample eval, twice; no training or AE selection.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash slurm/launch_abcd_axis_comparison.sh UNIQUE_TAG [gpu|cpu]" >&2
  exit 2
fi
TAG="$1"
RESOURCE="${2:-gpu}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ( "$RESOURCE" != gpu && "$RESOURCE" != cpu ) ]]; then
  echo "ERROR: invalid tag or resource (gpu/cpu)" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submitting" >&2
  exit 2
fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
TRAIN3M_DIR="${TRAIN3M_DIR:-$BASE/data/train3M_v1}"
TRAIN3M_CONVERTED_DIR="${TRAIN3M_CONVERTED_DIR:-$TRAIN3M_DIR/converted_v1}"
# Lock both modes to the validated full converted sample, even if the caller
# previously sourced a legacy or AE-tail campaign's environment.
TRAIN_PT="$TRAIN3M_CONVERTED_DIR/train/sample.json"
GEN_WEIGHT_TRAIN="$TRAIN3M_CONVERTED_DIR/weight_train.npz"
REFERENCE_PT="$TRAIN_PT"
REFERENCE_WEIGHTS="$GEN_WEIGHT_TRAIN"
HELDOUT_PT="$TRAIN3M_CONVERTED_DIR/test/sample.json"
HELDOUT_WEIGHTS="$TRAIN3M_CONVERTED_DIR/weight_test.npz"
source "$SCRIPT_DIR/train3m_environment.sh"
CODE_COMMIT="$(git rev-parse HEAD)"
AE_CKPT="${AE_CKPT:-$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth}"
MODEL_CKPT="${MODEL_CKPT:-$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_train3m_physics_decay8_lambda1_20261107_132516/checkpoint_main.pth.tar}"
OUTDIR="$BASE/outputs/${TAG}_axis_comparison"
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite $OUTDIR" >&2
  exit 2
fi
for path in "$AE_CKPT" "$MODEL_CKPT" "$REFERENCE_PT" "$REFERENCE_WEIGHTS" "$HELDOUT_PT" "$HELDOUT_WEIGHTS"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing input: $path" >&2
    exit 2
  fi
done
SIGNAL_PT="${SIGNAL_PT:-$BASE/data/hlt_signal_TpTp.pt}"
if [[ "${INCLUDE_SIGNAL:-1}" == 0 || ! -s "$SIGNAL_PT" ]]; then
  SIGNAL_PT=""
fi
RESOURCE_ARGS=(--partition=gpu --gres=gpu:1 --time=08:00:00)
if [[ "$RESOURCE" == cpu ]]; then
  RESOURCE_ARGS=(--partition=cpu --time=24:00:00)
fi
export BASE CODE_DIR CODE_COMMIT AE_CKPT MODEL_CKPT
export REFERENCE_PT REFERENCE_WEIGHTS HELDOUT_PT HELDOUT_WEIGHTS
export OUTDIR RESOURCE SIGNAL_PT
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable "${RESOURCE_ARGS[@]}" --job-name="${TAG}_axes" \
  --export=ALL "$SCRIPT_DIR/submit_abcd_axis_comparison.sbatch")"
echo "Job: $JOB"
echo "Classifier: $MODEL_CKPT"
echo "Output: $OUTDIR"
echo "QCD-MD plots: $OUTDIR/qcd_md/held-out/plots"
echo "1-P(QCD) plots: $OUTDIR/one_minus_pqcd/held-out/plots"
echo "Summary: $OUTDIR/axis_comparison.csv"
echo "Log: $BASE/logs/${TAG}_axes-${JOB}.out"
echo "Resources: $RESOURCE (no A100 requirement)"
