#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash slurm/launch_parent_balanced_ae_ablation.sh RUN_TAG [SUPCON_WEIGHT]"
  exit 2
fi

# Use the validated train3M tensors regardless of the caller's shell defaults.
source "$SCRIPT_DIR/train3m_environment.sh"

export AE_WEIGHTING_SCHEME=parent_balanced
export TARGET_SCHEME=decay8
export REUSE_AE=0
export NURD_ENABLED=0
export NURD_EPOCHS="${CLASSIFIER_EPOCHS:-50}"
export LR_SCHEDULE_EPOCHS="$NURD_EPOCHS"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
export STATISTICALLY_VALID_CLOSURE=1

RUN_TAG="$1"
SUPCON_WEIGHT="${2:-0.3}"

echo "Launching controlled parent-balanced AE ablation"
echo "  Run tag: $RUN_TAG"
echo "  AE: parent-balanced exposure, full train3M sample"
echo "  Classifier: decay8, $NURD_EPOCHS epochs, NURD disabled"
echo "  Evaluation: one dependent statistically filtered held-out job"

exec bash "$SCRIPT_DIR/launch_engineer_campaign.sh" \
  "$RUN_TAG" "$SUPCON_WEIGHT"
