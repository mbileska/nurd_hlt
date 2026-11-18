#!/bin/bash
#SBATCH --job-name=double_nurd_train
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
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
python -c 'import torch; assert torch.cuda.is_available(), "Requested GPU is not visible"; print(torch.cuda.get_device_name())'
python -m scripts.run_double_nurd_pilot \
  --checkpoint "$SOURCE_CKPT" --ae-checkpoint "$ORIGINAL_AE" \
  --sample "$CONVERTED/train/sample.json" --weights "$CONVERTED/weight_train.npz" \
  --ae-cache-dir "$AE_CACHE_DIR" --output-dir "$EXPERIMENT_OUT" \
  --updates "$PILOT_UPDATES" --batch-size "$PILOT_BATCH_SIZE" \
  --ramp-updates "$TAIL_RAMP_UPDATES" --min-post-ramp-updates "$MIN_POST_RAMP_UPDATES" \
  --device cuda --threads "$OMP_NUM_THREADS" --code-commit "$CODE_COMMIT"
