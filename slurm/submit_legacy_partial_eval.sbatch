#!/bin/bash
# Evaluation only. Launcher supplies resources and job dependencies.
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
if [[ "${STAGE:?}" == score ]]; then
  python -c 'import torch; assert torch.cuda.is_available(), "Requested GPU is not visible"; print(torch.cuda.get_device_name())'
  python -m scripts.evaluate_legacy_double_nurd score --experiment-dir "$LEGACY_OUT" \
    --sample "$LEGACY_TRAIN" --weights "$LEGACY_WEIGHTS" \
    --heldout "$LEGACY_TEST" --heldout-weights "$LEGACY_TEST_WEIGHTS" \
    --output-dir "$LEGACY_SCORES" --device cuda --threads "$OMP_NUM_THREADS" \
    --allow-partial-training --code-commit "$CODE_COMMIT"
elif [[ "$STAGE" == plots ]]; then
  python -m scripts.evaluate_legacy_double_nurd plots --experiment-dir "$LEGACY_OUT" \
    --score-dir "$LEGACY_SCORES" --golden-summary "$GOLDEN_SUMMARY" \
    --output-dir "$LEGACY_EVAL" --device cpu --threads "$OMP_NUM_THREADS" --code-commit "$CODE_COMMIT"
else
  echo "ERROR: unknown evaluation stage $STAGE" >&2; exit 2
fi
