#!/bin/bash
#SBATCH --job-name=double_nurd_eval
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -eo pipefail
module load "${ANACONDA_MODULE:-anaconda3/2026.7}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u
cd "${CODE_DIR:?}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="$BASE/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$MPLCONFIGDIR"
python -c 'import torch; assert torch.cuda.is_available(), "Requested GPU is not visible"'
python -m scripts.evaluate_double_nurd_pilot \
  --experiment-dir "$EXPERIMENT_OUT" --ae-cache-dir "$AE_CACHE_DIR" \
  --sample "$CONVERTED/train/sample.json" --weights "$CONVERTED/weight_train.npz" \
  --heldout "$CONVERTED/test/sample.json" --heldout-weights "$CONVERTED/weight_test.npz" \
  --output-dir "$EVALUATION_OUT" --minimums 50 50 50 "$D_MIN_ESS" \
  --device cuda --threads "$OMP_NUM_THREADS" --code-commit "$CODE_COMMIT"
