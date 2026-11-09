#!/bin/bash
#SBATCH --job-name=qcd_lambda_pair_audit
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
export MPLCONFIGDIR="${BASE}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "${CODE_DIR:?}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT:?}"
source slurm/train3m_environment.sh
test ! -e "${AUDIT_OUT:?}"

AE="$BASE/checkpoints/hlt/hlt/ae_engineer_${SOURCE_RUN}/checkpoint_ae.pth"
BASELINE_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_${SOURCE_RUN}_nurd/checkpoint_main.pth.tar"
CANDIDATE_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_${CANDIDATE_RUN}/checkpoint_main.pth.tar"
BASELINE_EVAL="$BASE/outputs/${SOURCE_RUN}_nurd_eval/evaluation_summary.json"
CANDIDATE_EVAL="$BASE/outputs/${CANDIDATE_RUN}_eval_stat_valid/evaluation_summary.json"
for path in "$AE" "$BASELINE_CKPT" "$CANDIDATE_CKPT" "$BASELINE_EVAL" "$CANDIDATE_EVAL"; do
  test -s "$path"
done
mkdir -p "$AUDIT_OUT"

for role in baseline candidate; do
  if [[ "$role" == baseline ]]; then
    CKPT="$BASELINE_CKPT"
    EVAL="$BASELINE_EVAL"
  else
    CKPT="$CANDIDATE_CKPT"
    EVAL="$CANDIDATE_EVAL"
  fi
  python scripts/audit_qcd_abcd_composition.py \
    --checkpoint "$CKPT" --ae-checkpoint "$AE" \
    --reference "$TRAIN_PT" --reference-weights "$GEN_WEIGHT_TRAIN" \
    --heldout "$HELDOUT_PT" --heldout-weights "$HELDOUT_WEIGHTS" \
    --evaluation-summary "$EVAL" --output-dir "$AUDIT_OUT/${role}_heldout"
  python scripts/audit_qcd_validation_hardness.py \
    --checkpoint "$CKPT" --ae-checkpoint "$AE" \
    --training-manifest "$TRAIN_PT" --training-weights "$GEN_WEIGHT_TRAIN" \
    --evaluation-summary "$EVAL" --output-dir "$AUDIT_OUT/${role}_validation"
done

python scripts/compare_qcd_score_audit.py \
  --baseline-heldout "$AUDIT_OUT/baseline_heldout/qcd_events.npz" \
  --candidate-heldout "$AUDIT_OUT/candidate_heldout/qcd_events.npz" \
  --baseline-validation "$AUDIT_OUT/baseline_validation/validation_qcd_scores.npz" \
  --candidate-validation "$AUDIT_OUT/candidate_validation/validation_qcd_scores.npz" \
  --baseline-evaluation "$BASELINE_EVAL" \
  --candidate-evaluation "$CANDIDATE_EVAL" \
  --output-dir "$AUDIT_OUT/paired"
echo "PAIRED AUDIT DONE: $AUDIT_OUT/paired"
