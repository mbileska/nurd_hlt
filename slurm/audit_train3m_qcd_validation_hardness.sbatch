#!/bin/bash
#SBATCH --job-name=train3m_qcd_valhard
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail
export PS1="${PS1-}"

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
RUN_TAG="${RUN_TAG:?Pass RUN_TAG with sbatch --export=ALL,RUN_TAG=...}"
cd "$CODE_DIR"
source slurm/train3m_environment.sh

AE_CKPT="$BASE/checkpoints/hlt/hlt/ae_engineer_${RUN_TAG}/checkpoint_ae.pth"
test -s "$AE_CKPT"
export MPLCONFIGDIR="$BASE/matplotlib"
mkdir -p "$MPLCONFIGDIR"

for stage in baseline nurd; do
  CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_${RUN_TAG}_${stage}/checkpoint_main.pth.tar"
  EVAL_SUMMARY="$BASE/outputs/${RUN_TAG}_${stage}_eval/evaluation_summary.json"
  OUTPUT_DIR="$BASE/outputs/${RUN_TAG}_qcd_validation_hardness_${SLURM_JOB_ID}/${stage}"
  test -s "$CKPT"
  test -s "$EVAL_SUMMARY"
  echo "Validation-only hardness audit: $stage at $(date)"
  python scripts/audit_qcd_validation_hardness.py \
    --checkpoint "$CKPT" \
    --ae-checkpoint "$AE_CKPT" \
    --training-manifest "$TRAIN_PT" \
    --training-weights "$GEN_WEIGHT_TRAIN" \
    --evaluation-summary "$EVAL_SUMMARY" \
    --output-dir "$OUTPUT_DIR"
done

echo "VALIDATION HARDNESS AUDIT DONE: $BASE/outputs/${RUN_TAG}_qcd_validation_hardness_${SLURM_JOB_ID}"
