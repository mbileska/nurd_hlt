#!/bin/bash
# Continue a validated classifier-only train3M baseline with safe NURD defaults.

set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash slurm/launch_warm_start_nurd.sh BASELINE_TAG NEW_RUN_TAG [SUPCON_WEIGHT]"
  exit 2
fi

BASELINE_TAG="$1"
NEW_RUN_TAG="$2"
SUPCON_WEIGHT="${3:-0.3}"
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export AE_EXP="${AE_EXP:-ae_engineer_$BASELINE_TAG}"
export REUSE_AE="${REUSE_AE:-1}"
export WARM_START_CKPT="${WARM_START_CKPT:-$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_$BASELINE_TAG/checkpoint_main.pth.tar}"
export TARGET_SCHEME="${TARGET_SCHEME:-decay8}"
export NURD_ENABLED="${NURD_ENABLED:-1}"
export CRITIC_START_EPOCH="${CRITIC_START_EPOCH:-0}"
export CRITIC_SHUFFLE_MODE="${CRITIC_SHUFFLE_MODE:-weighted_within_class}"
export CRITIC_WARMUP_EPOCHS="${CRITIC_WARMUP_EPOCHS:-3}"
export LAMBDA_INFO_RAMP_EPOCHS="${LAMBDA_INFO_RAMP_EPOCHS:-10}"
export LAMBDA_INFO="${LAMBDA_INFO:-0.1}"
export SELECTION_AFTER_RAMP="${SELECTION_AFTER_RAMP:-1}"
export TRAIN_LR="${TRAIN_LR:-3e-5}"
export NURD_EPOCHS="${NURD_EPOCHS:-30}"
export LR_SCHEDULE_EPOCHS="${LR_SCHEDULE_EPOCHS:-$NURD_EPOCHS}"
export BALANCE_STRATA="${BALANCE_STRATA:-20}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
export STATISTICALLY_VALID_CLOSURE="${STATISTICALLY_VALID_CLOSURE:-1}"

for checkpoint in \
  "$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth" \
  "$WARM_START_CKPT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: required baseline checkpoint is missing: $checkpoint"
    exit 2
  fi
done

echo "Warm-start baseline: $BASELINE_TAG"
echo "New continuation:    $NEW_RUN_TAG"
echo "AE checkpoint:       $BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
echo "Classifier checkpoint: $WARM_START_CKPT"
echo "Schedule: critic warm-up=$CRITIC_WARMUP_EPOCHS, ramp=$LAMBDA_INFO_RAMP_EPOCHS, lambda_max=$LAMBDA_INFO"

exec bash "$SCRIPT_DIR/launch_train3m_campaign.sh" +  "$NEW_RUN_TAG" "$SUPCON_WEIGHT"
