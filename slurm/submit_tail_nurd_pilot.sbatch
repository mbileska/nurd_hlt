#!/bin/bash
#SBATCH --job-name=tail_nurd_pilot
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -eo pipefail
module load "${ANACONDA_MODULE:-anaconda3/2026.7}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u
cd "${CODE_DIR:?}"
if [[ "$(git rev-parse HEAD)" != "$CODE_COMMIT" ]] || ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: source changed after submission" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$BASE/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$MPLCONFIGDIR"
python -c 'import torch; assert torch.cuda.is_available(), "Requested Slurm GPU is not visible"; print(torch.cuda.get_device_name())'
python -m scripts.run_tail_nurd_pilot \
  --checkpoint "$SOURCE_CKPT" --ae-checkpoint "$ORIGINAL_AE" \
  --sample "$CONVERTED/train/sample.json" --weights "$CONVERTED/weight_train.npz" \
  --ae-cache-dir "$AE_CACHE_DIR" \
  --heldout "$CONVERTED/test/sample.json" --heldout-weights "$CONVERTED/weight_test.npz" \
  --output-dir "$PILOT_OUT" --code-commit "$CODE_COMMIT" \
  --updates "$PILOT_UPDATES" --batch-size "$PILOT_BATCH_SIZE"
