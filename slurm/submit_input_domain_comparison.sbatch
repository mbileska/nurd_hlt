#!/bin/bash
#SBATCH --job-name=input_domain
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
set -euo pipefail
module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
export MPLCONFIGDIR="$BASE/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "${CODE_DIR:?}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT:?}"
test ! -e "${DOMAIN_OUT:?}"
python scripts/compare_input_domains.py \
  --legacy "$LEGACY_SAMPLE" --legacy-weights "$LEGACY_WEIGHTS" \
  --new "$NEW_SAMPLE" --new-weights "$NEW_WEIGHTS" \
  --ae-checkpoint "$AE_CKPT" --output-dir "$DOMAIN_OUT"
echo "INPUT DOMAIN COMPARISON DONE: $DOMAIN_OUT"
