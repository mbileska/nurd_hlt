#!/bin/bash
#SBATCH --job-name=decay8_legacy_parent4
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
set -euo pipefail
module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
export WANDB_MODE=offline
export WANDB_DIR="$BASE/wandb"
export MPLCONFIGDIR="$BASE/matplotlib"
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"
cd "${CODE_DIR:?}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT:?}"
test ! -e "${LEGACY_OUT:?}"
python scripts/evaluate_decay8_parent4_legacy.py \
  --checkpoint "$MODEL_CKPT" --ae-checkpoint "$AE_CKPT" \
  --reference "$LEGACY_TRAIN" --reference-weights "$LEGACY_TRAIN_WEIGHTS" \
  --test "$LEGACY_TEST" --test-weights "$LEGACY_TEST_WEIGHTS" \
  --output-dir "$LEGACY_OUT"
echo "LEGACY PARENT4 EVALUATION DONE: $LEGACY_OUT"
