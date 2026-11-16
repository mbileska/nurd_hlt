#!/bin/bash
#SBATCH --job-name=ae_tail_validation
#SBATCH --partition=cpu
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

export PS1="${PS1-}"
set -euo pipefail
module load anaconda3/2026.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
cd "${CODE_DIR:?}"
if [[ "$(git rev-parse HEAD)" != "${CODE_COMMIT:?}" ]]; then
  echo "ERROR: code changed after submission" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked code has uncommitted changes" >&2
  exit 2
fi
mkdir -p "$BASE/matplotlib"
export MPLCONFIGDIR="$BASE/matplotlib" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
python -m scripts.evaluate_ae_tail_validation \
  --checkpoint "$CHECKPOINT" --ae-checkpoint "$AE_CKPT" \
  --training-manifest "$TRAIN_PT" --training-weights "$TRAIN_WEIGHTS" \
  --qcd-score-cache "$QCD_CACHE" --qcd-cache-report "$CACHE_REPORT" \
  --output-dir "$OUTDIR" --tail-fraction .1 \
  --selection-measure "$SELECTION_MEASURE" --threads "$OMP_NUM_THREADS"
