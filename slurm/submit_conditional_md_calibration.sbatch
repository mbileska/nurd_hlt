#!/bin/bash
#SBATCH --job-name=conditional_md_validation
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

export PS1="${PS1-}"
set -euo pipefail

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

BASE="${BASE:?BASE must be supplied}"
CODE_DIR="${CODE_DIR:?CODE_DIR must be supplied}"
ANALYSIS_COMMIT="${ANALYSIS_COMMIT:?ANALYSIS_COMMIT must be supplied}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT must be supplied}"
AE_CKPT="${AE_CKPT:?AE_CKPT must be supplied}"
TRAIN_PT="${TRAIN_PT:?TRAIN_PT must be supplied}"
TRAIN_WEIGHTS="${TRAIN_WEIGHTS:?TRAIN_WEIGHTS must be supplied}"
OUTDIR="${OUTDIR:?OUTDIR must be supplied}"
AE_BINS="${AE_BINS:-20}"
MIN_FIT_BIN_ESS="${MIN_FIT_BIN_ESS:-200}"

mkdir -p "$BASE/logs" "$BASE/matplotlib"
export MPLCONFIGDIR="$BASE/matplotlib"
cd "$CODE_DIR"
if [[ "$(git rev-parse HEAD)" != "$ANALYSIS_COMMIT" ]]; then
  echo "ERROR: analysis code changed after submission." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked analysis code has uncommitted changes." >&2
  exit 2
fi
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite output: $OUTDIR" >&2
  exit 2
fi
for input in "$CHECKPOINT" "$AE_CKPT" "$TRAIN_PT" "$TRAIN_WEIGHTS"; do
  if [[ ! -s "$input" ]]; then
    echo "ERROR: required input is missing: $input" >&2
    exit 2
  fi
done

python -m scripts.run_conditional_md_calibration \
  --checkpoint "$CHECKPOINT" \
  --ae-ckpt "$AE_CKPT" \
  --data "$TRAIN_PT" \
  --gen-weight-path "$TRAIN_WEIGHTS" \
  --outdir "$OUTDIR" \
  --qcd-label 1 \
  --n-pca 6 \
  --ae-bins "$AE_BINS" \
  --min-fit-bin-ess "$MIN_FIT_BIN_ESS"

echo "CONDITIONAL MD VALIDATION DONE"
echo "Report: $OUTDIR/conditional_md_report.json"
echo "Plots:  $OUTDIR/plots"
