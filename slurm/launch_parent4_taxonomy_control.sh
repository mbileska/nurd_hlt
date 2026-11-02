#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash slurm/launch_parent4_taxonomy_control.sh RUN_TAG [SUPCON_WEIGHT]"
  exit 2
fi

# This is a taxonomy-only control. Reuse the validated train3M tensors and
# already-trained parent-balanced AE, and force every classifier setting that
# could otherwise make the comparison with decay8 ambiguous.
source "$SCRIPT_DIR/train3m_environment.sh"

export AE_EXP="${PARENT_BALANCED_AE_EXP:-ae_engineer_train3m_v1_decay8_ae_parentbal_20261030_142528}"
export AE_WEIGHTING_SCHEME=parent_balanced
export TARGET_SCHEME=parent4
export REUSE_AE=1
export NURD_ENABLED=0
export NURD_EPOCHS="${CLASSIFIER_EPOCHS:-50}"
export LR_SCHEDULE_EPOCHS="$NURD_EPOCHS"
export BALANCE_STRATA=1
export BALANCE_BINNING=log_fixed
export BALANCE_CLIP_QUANTILE=0.995
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
export STATISTICALLY_VALID_CLOSURE=1
export WARM_START_CKPT=""
export CRITIC_WARMUP_EPOCHS=0
export LAMBDA_INFO_RAMP_EPOCHS=0
export SELECTION_AFTER_RAMP=0

RUN_TAG="$1"
SUPCON_WEIGHT="${2:-0.3}"
AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"

if [[ ! -s "$AE_CKPT" ]]; then
  echo "ERROR: parent-balanced AE checkpoint is missing: $AE_CKPT"
  exit 2
fi

echo "Launching parent4 taxonomy control"
echo "  Run tag: $RUN_TAG"
echo "  Data: validated train3M tensors"
echo "  Reused AE: $AE_EXP"
echo "  Targets: parent4 (Z+jets, QCD, ttbar, W+jets)"
echo "  Classifier weights: equal target-class mass, physics ratios within class"
echo "  AE-score strata: disabled (BALANCE_STRATA=1)"
echo "  NURD: disabled"
echo "  Classifier epochs: $NURD_EPOCHS"
echo "  Evaluation: dependent statistically filtered held-out + legacy control"

exec bash "$SCRIPT_DIR/launch_engineer_campaign.sh" \
  "$RUN_TAG" "$SUPCON_WEIGHT"
