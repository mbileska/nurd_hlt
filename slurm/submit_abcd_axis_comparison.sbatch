#!/bin/bash
#SBATCH --job-name=abcd_axes
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Module/Conda initialization can read PS1; don't enable nounset until afterward.
set -eo pipefail
module load "${ANACONDA_MODULE:-anaconda3/2026.7}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u
cd "${CODE_DIR:?}"
if [[ "$(git rev-parse HEAD)" != "${CODE_COMMIT:?}" ]] || \
   ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: code changed after submission" >&2
  exit 2
fi
export MPLCONFIGDIR="$BASE/matplotlib"
export MPLBACKEND=Agg
export WANDB_DIR="$BASE/wandb"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$MPLCONFIGDIR" "$WANDB_DIR"
if [[ "$RESOURCE" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  python -c 'import torch; assert torch.cuda.is_available(), "Slurm GPU is not visible to PyTorch"'
fi
SIGNAL_ARGS=()
if [[ -n "$SIGNAL_PT" ]]; then
  SIGNAL_ARGS=(--signal "$SIGNAL_PT")
fi
python -m scripts.compare_abcd_axes \
  --checkpoint "$MODEL_CKPT" --ae-checkpoint "$AE_CKPT" \
  --reference "$REFERENCE_PT" --reference-weights "$REFERENCE_WEIGHTS" \
  --heldout "$HELDOUT_PT" --heldout-weights "$HELDOUT_WEIGHTS" \
  --output-dir "$OUTDIR" --code-commit "$CODE_COMMIT" "${SIGNAL_ARGS[@]}"
echo "FULL-SAMPLE AXIS COMPARISON DONE: $OUTDIR/axis_comparison.json"
