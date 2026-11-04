#!/bin/bash
# Continue a completed parent4 classifier-only control with a full, gradual
# NURD schedule while preserving its data split, AE and event-weight contract.

set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash slurm/launch_parent4_full_nurd.sh BASELINE_TAG NEW_RUN_TAG [SUPCON_WEIGHT]"
  exit 2
fi

BASELINE_TAG="$1"
NEW_RUN_TAG="$2"
SUPCON_WEIGHT="${3:-0.3}"
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

source "$SCRIPT_DIR/train3m_environment.sh"

# Force the exact parent4 control contract. In particular, keep class-only
# balancing: changing back to 20 AE-score strata would confound NURD with a
# second, implicit decorrelation mechanism and collapse the effective sample.
export AE_EXP="${PARENT_BALANCED_AE_EXP:-ae_engineer_train3m_v1_decay8_ae_parentbal_20261030_142528}"
export AE_WEIGHTING_SCHEME=parent_balanced
export REUSE_AE=1
export WARM_START_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_$BASELINE_TAG/checkpoint_main.pth.tar"
export TARGET_SCHEME=parent4
export BALANCE_STRATA=1
export BALANCE_BINNING=log_fixed
export BALANCE_CLIP_QUANTILE=0.995

export NURD_ENABLED=1
export CRITIC_START_EPOCH=0
export CRITIC_SHUFFLE_MODE=weighted_within_class
export CRITIC_WARMUP_EPOCHS="${CRITIC_WARMUP_EPOCHS_OVERRIDE:-5}"
export LAMBDA_INFO_RAMP_EPOCHS="${LAMBDA_INFO_RAMP_EPOCHS_OVERRIDE:-15}"
export LAMBDA_INFO="${LAMBDA_INFO_OVERRIDE:-0.1}"
export SELECTION_AFTER_RAMP=1
export TRAIN_LR="${TRAIN_LR_OVERRIDE:-3e-5}"
export NURD_EPOCHS="${NURD_EPOCHS_OVERRIDE:-40}"
export LR_SCHEDULE_EPOCHS="$NURD_EPOCHS"
export CHECKPOINT_EVERY=1
export STATISTICALLY_VALID_CLOSURE=1

AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
for checkpoint in "$AE_CKPT" "$WARM_START_CKPT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: required warm-start checkpoint is missing: $checkpoint"
    exit 2
  fi
done

echo "Launching full parent4 NURD continuation"
echo "  Baseline: $BASELINE_TAG"
echo "  Continuation: $NEW_RUN_TAG"
echo "  AE: $AE_EXP"
echo "  Targets: parent4"
echo "  Weights: class-only, 25% mass per parent, no AE-score strata"
echo "  Schedule: critic warm-up=$CRITIC_WARMUP_EPOCHS epochs"
echo "  Schedule: lambda ramp=$LAMBDA_INFO_RAMP_EPOCHS epochs"
echo "  Schedule: lambda max=$LAMBDA_INFO"
echo "  Continuation epochs=$NURD_EPOCHS, lr=$TRAIN_LR"
echo "  Selection begins only after critic warm-up and lambda ramp"
echo "  Periodic checkpoints: every epoch"

exec bash "$SCRIPT_DIR/launch_engineer_campaign.sh" \
  "$NEW_RUN_TAG" "$SUPCON_WEIGHT"
