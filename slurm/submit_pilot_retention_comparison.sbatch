#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -eo pipefail
module load "${ANACONDA_MODULE:-anaconda3/2026.7}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u
cd "${CODE_DIR:?}"
if [[ "$(git rev-parse HEAD)" != "$CODE_COMMIT" ]] || ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: source changed after submission" >&2; exit 2
fi
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="$BASE/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$MPLCONFIGDIR"
python -m scripts.compare_pilot_retentions \
  --pilot-dir "$PILOT_DIR" --ae-cache-dir "$AE_CACHE_DIR" \
  --sample "$CONVERTED/train/sample.json" --weights "$CONVERTED/weight_train.npz" \
  --heldout "$CONVERTED/test/sample.json" --heldout-weights "$CONVERTED/weight_test.npz" \
  --output-dir "$COMPARISON_OUT" --device cuda --batch-size 512 --threads "$OMP_NUM_THREADS" \
  --minimums 50 50 50 "$D_MIN_ESS" --code-commit "$CODE_COMMIT"
