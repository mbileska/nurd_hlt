#!/bin/bash
# Full-data NURD continuation: the Nov-04 decay8 control with only lambda changed.
# This submits training only. Inspect validation checkpoints before held-out eval.
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: bash slurm/launch_physics_decay8_lambda1_control.sh BASE_RUN NEW_RUN_TAG" >&2
  exit 2
fi

BASE_RUN="$1"
NEW_RUN_TAG="$2"
for value in "$BASE_RUN" "$NEW_RUN_TAG"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: run tags may contain only letters, numbers, dot, underscore, dash." >&2
    exit 2
  fi
done
if [[ "$BASE_RUN" == "$NEW_RUN_TAG" ]]; then
  echo "ERROR: continuation tag must differ from baseline run tag." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
source "$SCRIPT_DIR/train3m_environment.sh"

# A stale shell export must not quietly turn this full train3M control into a
# different dataset or weight experiment.
if [[ "$TRAIN_PT" != "$TRAIN3M_CONVERTED_DIR/train/sample.json" \
   || "$GEN_WEIGHT_TRAIN" != "$TRAIN3M_CONVERTED_DIR/weight_train.npz" ]]; then
  echo "ERROR: this control requires the validated full train3M train split and weights." >&2
  exit 2
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submitting training." >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"

AE_EXP="ae_engineer_${BASE_RUN}"
AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
WARM_START_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_${BASE_RUN}_baseline/checkpoint_main.pth.tar"
NURD_EXP="hlt_nurd_engineer_${NEW_RUN_TAG}"
NEW_CKPT_DIR="$BASE/checkpoints/hlt/hlt/$NURD_EXP"
for input in "$AE_CKPT" "$WARM_START_CKPT"; do
  if [[ ! -s "$input" ]]; then
    echo "ERROR: required checkpoint is missing: $input" >&2
    exit 2
  fi
done
if [[ -e "$NEW_CKPT_DIR" || -e "$BASE/outputs/${NEW_RUN_TAG}_eval" \
   || -e "$BASE/logs/${NEW_RUN_TAG}_train.log" ]]; then
  echo "ERROR: refusing to reuse existing run tag: $NEW_RUN_TAG" >&2
  exit 2
fi

# Match slurm/launch_physics_decay8_full_nurd.sh exactly, except lambda=1.0
# and more frequent periodic checkpointing. In particular, retain class-only
# weighting, the decay8 taxonomy, the same AE, the same baseline model, and
# the same critic/optimizer schedule and random seed (in submit script).
export BASE CODE_DIR CODE_COMMIT
export RUN_TAG="$NEW_RUN_TAG" AE_EXP NURD_EXP WARM_START_CKPT
export REUSE_AE=1 AE_WEIGHTING_SCHEME=physics TARGET_SCHEME=decay8
export CONTRAST_WEIGHT=0.3 NURD_ENABLED=1 CRITIC_START_EPOCH=0
export CRITIC_SHUFFLE_MODE=weighted_within_class
export CRITIC_WARMUP_EPOCHS=5 LAMBDA_INFO_RAMP_EPOCHS=15
export LAMBDA_INFO=1.0 SELECTION_AFTER_RAMP=1 TRAIN_LR=3e-5
export BALANCE_STRATA=1 BALANCE_BINNING=log_fixed
export BALANCE_CLIP_QUANTILE=0.995
export NURD_EPOCHS=80 LR_SCHEDULE_EPOCHS=80 NURD_PATIENCE=20
export MIN_POST_RAMP_EPOCHS=25 CHECKPOINT_EVERY=2
export TRAIN_PT GEN_WEIGHT_TRAIN HELDOUT_PT HELDOUT_WEIGHTS

mkdir -p "$BASE/logs"
TRAIN_JOB="$(sbatch --parsable --job-name="${NEW_RUN_TAG}_train" \
  --export=ALL "$SCRIPT_DIR/submit_engineer_train.sbatch")"

echo "Full-data decay8 NURD lambda=1 control"
echo "Source AE:       $AE_CKPT"
echo "Source baseline: $WARM_START_CKPT"
echo "Data:            $TRAIN_PT"
echo "Weights:         $GEN_WEIGHT_TRAIN"
echo "New experiment:  $NURD_EXP"
echo "Commit:          $CODE_COMMIT"
echo "Only changed training hyperparameter: lambda_info 0.1 -> 1.0"
echo "Checkpoints:     every 2 epochs (instead of every 5)"
echo "Schedule:        warm-up 5, ramp 15, >=25 full-strength epochs; 80 max"
echo "Training job:    $TRAIN_JOB"
echo "No held-out evaluation submitted."
echo "Monitor: squeue -j $TRAIN_JOB"
echo "Log: $NEW_CKPT_DIR/train.log"
