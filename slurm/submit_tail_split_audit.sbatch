#!/bin/bash
#SBATCH --job-name=tail_split_audit
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
set -eo pipefail
module load "${ANACONDA_MODULE:-anaconda3/2026.7}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u
cd "${CODE_DIR:?}"
test "$(git rev-parse HEAD)" = "$CODE_COMMIT"
git diff --quiet
git diff --cached --quiet
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$BASE/matplotlib"
mkdir -p "$MPLCONFIGDIR"
python -m scripts.audit_tail_pilot_split \
  --preparation-dir "$PREP_DIR" --ae-checkpoint "$AE_CKPT" --weights "$WEIGHTS" \
  --output-dir "$OUTDIR"
