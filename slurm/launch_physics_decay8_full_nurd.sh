#!/bin/bash
# Fresh physics AE -> decay8 baseline + evaluation -> matched NURD + evaluation.
# All five jobs are chained afterok, so only one campaign job runs at a time.
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash slurm/launch_physics_decay8_full_nurd.sh RUN_TAG [SUPCON_WEIGHT]"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
source "$SCRIPT_DIR/train3m_environment.sh"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submitting."
  exit 2
fi

RUN_TAG="$1"
SUPCON_WEIGHT="${2:-0.3}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: invalid run tag."
  exit 2
fi
if ! [[ "$SUPCON_WEIGHT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "ERROR: SUPCON_WEIGHT must be a non-negative decimal."
  exit 2
fi

CODE_COMMIT="$(git rev-parse HEAD)"
AE_EXP="ae_engineer_${RUN_TAG}"
BASELINE_TAG="${RUN_TAG}_baseline"
NURD_TAG="${RUN_TAG}_nurd"
BASELINE_EXP="hlt_nurd_engineer_${BASELINE_TAG}"
NURD_EXP="hlt_nurd_engineer_${NURD_TAG}"

AE_EPOCHS="${AE_EPOCHS:-150}"
AE_PATIENCE="${AE_PATIENCE:-25}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-80}"
NURD_EPOCHS="${NURD_EPOCHS:-80}"
NURD_PATIENCE="${NURD_PATIENCE:-20}"
CRITIC_WARMUP_EPOCHS="${CRITIC_WARMUP_EPOCHS:-5}"
LAMBDA_INFO_RAMP_EPOCHS="${LAMBDA_INFO_RAMP_EPOCHS:-15}"
MIN_POST_RAMP_EPOCHS="${MIN_POST_RAMP_EPOCHS:-25}"
LAMBDA_INFO="${LAMBDA_INFO:-0.1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"

for name in AE_EPOCHS AE_PATIENCE BASELINE_EPOCHS NURD_EPOCHS \
  NURD_PATIENCE CRITIC_WARMUP_EPOCHS LAMBDA_INFO_RAMP_EPOCHS \
  MIN_POST_RAMP_EPOCHS CHECKPOINT_EVERY; do
  value="${!name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $name must be a non-negative integer."
    exit 2
  fi
done
if (( AE_EPOCHS < 1 || BASELINE_EPOCHS < 1 || NURD_EPOCHS < 1 \
      || AE_PATIENCE < 1 || NURD_PATIENCE < 1 || CHECKPOINT_EVERY < 1 \
      || NURD_EPOCHS < CRITIC_WARMUP_EPOCHS + LAMBDA_INFO_RAMP_EPOCHS \
                         + MIN_POST_RAMP_EPOCHS )); then
  echo "ERROR: epoch limits, patience, or checkpoint interval are inconsistent."
  exit 2
fi

for output in \
  "$BASE/checkpoints/hlt/hlt/$AE_EXP" \
  "$BASE/checkpoints/hlt/hlt/$BASELINE_EXP" \
  "$BASE/checkpoints/hlt/hlt/$NURD_EXP" \
  "$BASE/outputs/${BASELINE_TAG}_eval" \
  "$BASE/outputs/${NURD_TAG}_eval"; do
  if [[ -e "$output" ]]; then
    echo "ERROR: refusing to overwrite: $output"
    exit 2
  fi
done

mkdir -p "$BASE/logs"
export BASE CODE_DIR CODE_COMMIT AE_EXP AE_EPOCHS AE_PATIENCE
export TRAIN_PT GEN_WEIGHT_TRAIN HELDOUT_PT HELDOUT_WEIGHTS
export REFERENCE_PT REFERENCE_WEIGHTS
export STATISTICALLY_VALID_CLOSURE=1 TARGET_SCHEME=decay8
export AE_WEIGHTING_SCHEME=physics REUSE_AE=1 BALANCE_STRATA=1
export BALANCE_BINNING=log_fixed BALANCE_CLIP_QUANTILE=0.995
export CHECKPOINT_EVERY CRITIC_WARMUP_EPOCHS LAMBDA_INFO_RAMP_EPOCHS
export LAMBDA_INFO NURD_PATIENCE MIN_POST_RAMP_EPOCHS

AE_JOB="$(sbatch --parsable --job-name="${RUN_TAG}_ae" \
  "$SCRIPT_DIR/submit_train3m_physics_ae.sbatch")"

BASELINE_JOB="$(sbatch --parsable --dependency="afterok:$AE_JOB" \
  --job-name="${BASELINE_TAG}_train" \
  --export=ALL,RUN_TAG="$BASELINE_TAG",NURD_EXP="$BASELINE_EXP",CONTRAST_WEIGHT="$SUPCON_WEIGHT",NURD_EPOCHS="$BASELINE_EPOCHS",LR_SCHEDULE_EPOCHS="$BASELINE_EPOCHS",NURD_ENABLED=0,CRITIC_START_EPOCH=0,CRITIC_SHUFFLE_MODE=weighted_within_class,CRITIC_WARMUP_EPOCHS=0,LAMBDA_INFO_RAMP_EPOCHS=0,SELECTION_AFTER_RAMP=0,WARM_START_CKPT=,TRAIN_LR=1e-4,MIN_POST_RAMP_EPOCHS=0 \
  "$SCRIPT_DIR/submit_engineer_train.sbatch")"

BASELINE_EVAL_JOB="$(sbatch --parsable --dependency="afterok:$BASELINE_JOB" \
  --job-name="${BASELINE_TAG}_eval" \
  --export=ALL,RUN_TAG="$BASELINE_TAG",NURD_EXP="$BASELINE_EXP" \
  "$SCRIPT_DIR/submit_engineer_dual_eval.sbatch")"

NURD_JOB="$(sbatch --parsable --dependency="afterok:$BASELINE_EVAL_JOB" \
  --job-name="${NURD_TAG}_train" \
  --export=ALL,RUN_TAG="$NURD_TAG",NURD_EXP="$NURD_EXP",CONTRAST_WEIGHT="$SUPCON_WEIGHT",NURD_EPOCHS="$NURD_EPOCHS",LR_SCHEDULE_EPOCHS="$NURD_EPOCHS",NURD_ENABLED=1,CRITIC_START_EPOCH=0,CRITIC_SHUFFLE_MODE=weighted_within_class,SELECTION_AFTER_RAMP=1,WARM_START_CKPT="$BASE/checkpoints/hlt/hlt/$BASELINE_EXP/checkpoint_main.pth.tar",TRAIN_LR=3e-5 \
  "$SCRIPT_DIR/submit_engineer_train.sbatch")"

NURD_EVAL_JOB="$(sbatch --parsable --dependency="afterok:$NURD_JOB" \
  --job-name="${NURD_TAG}_eval" \
  --export=ALL,RUN_TAG="$NURD_TAG",NURD_EXP="$NURD_EXP" \
  "$SCRIPT_DIR/submit_engineer_dual_eval.sbatch")"

echo "Campaign: $RUN_TAG ($CODE_COMMIT)"
echo "AE: physics weights, $AE_EPOCHS max epochs, $AE_PATIENCE patience"
echo "Targets: decay8; classifier: class-only balance, SupCon=$SUPCON_WEIGHT"
echo "Baseline: $BASELINE_EPOCHS max epochs, NURD off"
echo "NURD: $NURD_EPOCHS max epochs; warm-up=$CRITIC_WARMUP_EPOCHS, ramp=$LAMBDA_INFO_RAMP_EPOCHS, lambda=$LAMBDA_INFO"
echo "Earliest NURD early stop: epoch $((CRITIC_WARMUP_EPOCHS + LAMBDA_INFO_RAMP_EPOCHS + MIN_POST_RAMP_EPOCHS))"
echo "Periodic checkpoint interval: $CHECKPOINT_EVERY epochs"
echo "AE job: $AE_JOB"
echo "Baseline job: $BASELINE_JOB"
echo "Baseline evaluation job: $BASELINE_EVAL_JOB"
echo "NURD job: $NURD_JOB"
echo "NURD evaluation job: $NURD_EVAL_JOB"
echo "Held-out summaries: $BASE/outputs/{${BASELINE_TAG},${NURD_TAG}}_eval/evaluation_summary.json"
