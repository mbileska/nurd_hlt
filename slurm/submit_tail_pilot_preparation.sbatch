#!/bin/bash
#SBATCH --job-name=tail_pilot_prep
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err

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
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
python -m scripts.prepare_tail_pilot \
  --checkpoint "$MODEL_CKPT" --ae-checkpoint "$AE_CKPT" \
  --sample "$CONVERTED/train/sample.json" --weights "$CONVERTED/weight_train.npz" \
  --output-dir "$OUTDIR" --threads "$OMP_NUM_THREADS"
