#!/bin/bash
#SBATCH --job-name=ae_tail_standard_plots
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
BATCH_SIZE=128
if [[ "$DEVICE" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  BATCH_SIZE=512
fi
SIGNAL_ARGS=()
if [[ -n "$SIGNAL_PT" ]]; then
  SIGNAL_ARGS=(--signal "$SIGNAL_PT")
fi
python -m scripts.plot_ae_tail_eval_suite \
  --tail-dir "$TAIL_DIR" --output-dir "$OUTDIR" --device "$DEVICE" \
  --threads "$OMP_NUM_THREADS" --batch-size "$BATCH_SIZE" "${SIGNAL_ARGS[@]}"
