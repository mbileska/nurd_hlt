#!/bin/bash
#SBATCH --job-name=nurd_validation_scan
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

module load anaconda3/2025.12
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
ANALYSIS_COMMIT="${ANALYSIS_COMMIT:?ANALYSIS_COMMIT must be supplied}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be supplied}"
EPOCHS_CSV="${EPOCHS_CSV:-}"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
AE_CKPT="$BASE/checkpoints/hlt/hlt/ae_engineer_$RUN_TAG/checkpoint_ae.pth"
NURD_DIR="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_$RUN_TAG"
TRAIN_PT="$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt"
TRAIN_WEIGHTS="$MEQUINNA_DIR/weight_train.pt"
OUTDIR="$BASE/outputs/${RUN_TAG}_validation_checkpoint_scan"

mkdir -p "$BASE/logs" "$BASE/outputs" "$BASE/matplotlib"
export MPLCONFIGDIR="$BASE/matplotlib"
cd "$CODE_DIR"
if [[ "$(git rev-parse HEAD)" != "$ANALYSIS_COMMIT" ]]; then
  echo "ERROR: analysis code changed after checkpoint-scan submission."
  exit 2
fi
for input in "$AE_CKPT" "$NURD_DIR" "$TRAIN_PT" "$TRAIN_WEIGHTS"; do
  if [[ ! -e "$input" ]]; then
    echo "ERROR: required input is missing: $input"
    exit 2
  fi
done

EPOCH_ARGS=()
if [[ -n "$EPOCHS_CSV" ]]; then
  IFS=',' read -r -a EPOCH_VALUES <<< "$EPOCHS_CSV"
  EPOCH_ARGS=(--epochs "${EPOCH_VALUES[@]}")
fi

python scripts/scan_validation_checkpoints.py \
  --checkpoint-dir "$NURD_DIR" \
  --ae-ckpt "$AE_CKPT" \
  --data "$TRAIN_PT" \
  --gen-weight-path "$TRAIN_WEIGHTS" \
  --outdir "$OUTDIR" \
  --qcd-label 1 \
  --n-pca 6 \
  "${EPOCH_ARGS[@]}"

echo "VALIDATION CHECKPOINT SCAN DONE"
echo "Summary: $OUTDIR/validation_checkpoint_scan.json"
echo "Plots:   $OUTDIR/plots"
