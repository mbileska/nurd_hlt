#!/bin/bash
#SBATCH --job-name=frozen_representation_audit
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

export PS1="${PS1-}"
set -euo pipefail
module load anaconda3/2026.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
BASE="${BASE:?}"
CODE_DIR="${CODE_DIR:?}"
ANALYSIS_COMMIT="${ANALYSIS_COMMIT:?}"
PURE_RUN_PREFIX="${PURE_RUN_PREFIX:?}"
AE_CKPT="${AE_CKPT:?}"
TRAIN_PT="${TRAIN_PT:?}"
TRAIN_WEIGHTS="${TRAIN_WEIGHTS:?}"
OUTDIR="${OUTDIR:?}"
mkdir -p "$BASE/matplotlib"
export MPLCONFIGDIR="$BASE/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
cd "$CODE_DIR"
if [[ "$(git rev-parse HEAD)" != "$ANALYSIS_COMMIT" ]]; then
  echo "ERROR: analysis code changed after submission." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked analysis code has uncommitted changes." >&2
  exit 2
fi
checkpoints=()
for suffix in 01 03 05; do
  checkpoints+=("$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_${PURE_RUN_PREFIX}_sc${suffix}/checkpoint_main.pth.tar")
done
python -m scripts.audit_frozen_representations \
  --checkpoints "${checkpoints[@]}" \
  --ae-ckpt "$AE_CKPT" --data "$TRAIN_PT" --gen-weight-path "$TRAIN_WEIGHTS" \
  --outdir "$OUTDIR" --jobs "$OMP_NUM_THREADS"
echo "FROZEN REPRESENTATION AUDIT COMPLETE: $OUTDIR"
