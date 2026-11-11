#!/bin/bash
# Full train3M decay8 continuation: same physical AE and baseline classifier,
# same λ=1 schedule/weights/critic, changing only the maximum information λ.
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash slurm/launch_physics_decay8_lambda25_control.sh BASE_RUN NEW_RUN_TAG [LAMBDA_MAX=2.5]" >&2
  exit 2
fi
BASE_RUN="$1"
NEW_RUN_TAG="$2"
LAMBDA_MAX="${3:-2.5}"
if [[ ! "$LAMBDA_MAX" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "LAMBDA_MAX must be a nonnegative decimal number" >&2
  exit 2
fi
for tag in "$BASE_RUN" "$NEW_RUN_TAG"; do
  if [[ ! "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run tag: $tag" >&2
    exit 2
  fi
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
source "$SCRIPT_DIR/train3m_environment.sh"
if [[ "$TRAIN_PT" != "$TRAIN3M_CONVERTED_DIR/train/sample.json" \
   || "$GEN_WEIGHT_TRAIN" != "$TRAIN3M_CONVERTED_DIR/weight_train.npz" \
   || "$HELDOUT_PT" != "$TRAIN3M_CONVERTED_DIR/test/sample.json" \
   || "$HELDOUT_WEIGHTS" != "$TRAIN3M_CONVERTED_DIR/weight_test.npz" \
   || "$REFERENCE_PT" != "$TRAIN_PT" \
   || "$REFERENCE_WEIGHTS" != "$GEN_WEIGHT_TRAIN" ]]; then
  echo "Expected validated full train3M data and physics weights" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit tracked changes before submitting" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
AE_EXP="ae_engineer_${BASE_RUN}"
NURD_EXP="hlt_nurd_engineer_${NEW_RUN_TAG}"
WARM_START_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_${BASE_RUN}_baseline/checkpoint_main.pth.tar"
test -s "$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
test -s "$WARM_START_CKPT"
test ! -e "$BASE/checkpoints/hlt/hlt/$NURD_EXP"
test ! -e "$BASE/outputs/${NEW_RUN_TAG}_eval_stat_valid"
export BASE CODE_DIR CODE_COMMIT AE_EXP NURD_EXP WARM_START_CKPT
export RUN_TAG="$NEW_RUN_TAG" REUSE_AE=1 AE_WEIGHTING_SCHEME=physics TARGET_SCHEME=decay8
export CONTRAST_WEIGHT=0.3 NURD_ENABLED=1 CRITIC_START_EPOCH=0
export CRITIC_SHUFFLE_MODE=weighted_within_class
export CRITIC_WARMUP_EPOCHS=5 LAMBDA_INFO_RAMP_EPOCHS=15
export LAMBDA_INFO="$LAMBDA_MAX" SELECTION_AFTER_RAMP=1 TRAIN_LR=3e-5
export BALANCE_STRATA=1 BALANCE_BINNING=log_fixed BALANCE_CLIP_QUANTILE=0.995
export NURD_EPOCHS=80 LR_SCHEDULE_EPOCHS=80 NURD_PATIENCE=20
export MIN_POST_RAMP_EPOCHS=25 CHECKPOINT_EVERY=2
export TRAIN_PT GEN_WEIGHT_TRAIN HELDOUT_PT HELDOUT_WEIGHTS REFERENCE_PT REFERENCE_WEIGHTS
export STATISTICALLY_VALID_CLOSURE=1 OUTPUT_ROOT="$BASE/outputs/${NEW_RUN_TAG}_eval_stat_valid"
mkdir -p "$BASE/logs"
TRAIN_JOB="$(sbatch --parsable --job-name="${NEW_RUN_TAG}_train" \
  --export=ALL "$SCRIPT_DIR/submit_engineer_train.sbatch")"
EVAL_JOB="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" \
  --job-name="${NEW_RUN_TAG}_eval" --export=ALL \
  "$SCRIPT_DIR/submit_engineer_dual_eval.sbatch")"
echo "Training job: $TRAIN_JOB"
echo "Evaluation job: $EVAL_JOB"
echo "AE: $BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
echo "Selected classifier: $BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_main.pth.tar"
echo "Evaluation: $OUTPUT_ROOT/evaluation_summary.json"
echo "Schedule: warm-up 5, ramp 15 to lambda=$LAMBDA_MAX, >=25 post-ramp epochs, 80 maximum"
