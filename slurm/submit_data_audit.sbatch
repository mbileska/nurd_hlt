#!/bin/bash
#SBATCH --job-name=hlt_data_audit
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -eo pipefail

module load anaconda3/2025.12
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
TRAIN_PT="${TRAIN_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt}"
TEST_PT="${TEST_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt}"
TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$MEQUINNA_DIR/weight_train.pt}"
TEST_WEIGHTS="${TEST_WEIGHTS:-$MEQUINNA_DIR/weight_test.pt}"
LEGACY_PT="${LEGACY_PT:-$BASE/data/hlt_smcocktail_test.pt}"
INCLUDE_LEGACY="${INCLUDE_LEGACY:-1}"
OUT_ROOT="${OUT_ROOT:-$CODE_DIR/data_plots}"
AUDIT_NAME="${AUDIT_NAME:-data_audit_${SLURM_JOB_ID}}"
OUTDIR="$OUT_ROOT/$AUDIT_NAME"
CHUNK_SIZE="${CHUNK_SIZE:-8192}"
EVENT_PLOT_SAMPLE="${EVENT_PLOT_SAMPLE:-100000}"
CANDIDATE_PLOT_SAMPLE="${CANDIDATE_PLOT_SAMPLE:-300000}"
WRITE_PDF="${WRITE_PDF:-0}"

mkdir -p "$BASE/logs" "$BASE/matplotlib" "$OUT_ROOT"
export MPLCONFIGDIR="$BASE/matplotlib"

for input in "$TRAIN_PT" "$TEST_PT" "$TRAIN_WEIGHTS" "$TEST_WEIGHTS"; do
  if [[ ! -s "$input" ]]; then
    echo "ERROR: required audit input is missing: $input"
    exit 2
  fi
done
if [[ "$INCLUDE_LEGACY" == "1" && ! -s "$LEGACY_PT" ]]; then
  echo "ERROR: requested legacy input is missing: $LEGACY_PT"
  exit 2
fi
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite existing audit output: $OUTDIR"
  exit 2
fi

cd "$CODE_DIR"
ARGS=(
  --dataset train "$TRAIN_PT" "$TRAIN_WEIGHTS"
  --dataset test "$TEST_PT" "$TEST_WEIGHTS"
  --output-dir "$OUTDIR"
  --chunk-size "$CHUNK_SIZE"
  --event-plot-sample "$EVENT_PLOT_SAMPLE"
  --candidate-plot-sample "$CANDIDATE_PLOT_SAMPLE"
)
if [[ "$INCLUDE_LEGACY" == "1" ]]; then
  ARGS+=(--dataset legacy "$LEGACY_PT" -)
fi
if [[ "$WRITE_PDF" == "1" ]]; then
  ARGS+=(--pdf)
fi

echo "=== HLT data audit started: $(date) ==="
echo "Code:       $CODE_DIR"
echo "Train:      $TRAIN_PT"
echo "Test:       $TEST_PT"
echo "Legacy:     $LEGACY_PT (enabled=$INCLUDE_LEGACY)"
echo "Output:     $OUTDIR"
echo "Slurm job:  $SLURM_JOB_ID"

/usr/bin/time -v python scripts/audit_hlt_data.py "${ARGS[@]}"

echo "=== HLT data audit finished: $(date) ==="
echo "Report: $OUTDIR/report.md"
echo "Plots:  $OUTDIR"
find "$OUTDIR" -type f | sort
