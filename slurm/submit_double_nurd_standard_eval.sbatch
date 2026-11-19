#!/bin/bash
#SBATCH --job-name=double_nurd_standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# GPU resources are supplied by the launcher for STAGE=infer only.
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
COMMON=(--experiment-dir "$EXPERIMENT_DIR" --comparison-dir "$COMPARISON_DIR"
  --heldout "$CONVERTED/test/sample.json" --heldout-weights "$CONVERTED/weight_test.npz"
  --minimums 50 50 50 "$D_MIN_ESS" --threads "$OMP_NUM_THREADS" --code-commit "$CODE_COMMIT")
if [[ "${STAGE:?}" == infer ]]; then
  python -c 'import torch; assert torch.cuda.is_available(), "Requested GPU is not visible"'
  python -m scripts.standard_double_nurd_eval "${COMMON[@]}" \
    --device cuda --inference-only --output-dir "$PREDICTION_OUT"
elif [[ "$STAGE" == plots ]]; then
  python -m scripts.standard_double_nurd_eval "${COMMON[@]}" \
    --device cpu --prediction-cache "$PREDICTION_OUT" --output-dir "$STANDARD_OUT"
else
  echo "ERROR: STAGE must be infer or plots" >&2; exit 2
fi
