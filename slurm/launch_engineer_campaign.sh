#!/bin/bash
set -euo pipefail

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
mkdir -p "$BASE/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

if (( $# == 3 )) && [[ "$3" == "--diagnostic" ]]; then
  exec bash "$SCRIPT_DIR/launch_diagnostic_campaign.sh" "$1" "$2"
fi
if (( $# != 2 )); then
  echo "Usage: bash slurm/launch_engineer_campaign.sh RUN_TAG SUPCON_WEIGHT [--diagnostic]"
  echo "Example: bash slurm/launch_engineer_campaign.sh engineer_continuous_supcon040 0.4"
  echo "Diagnostic: bash slurm/launch_engineer_campaign.sh chain_check 0.3 --diagnostic"
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit or discard tracked changes before launching."
  exit 2
fi

CODE_COMMIT="$(git rev-parse HEAD)"
RUN_TAG="$1"
CONTRAST_WEIGHT="$2"
NURD_EPOCHS="${NURD_EPOCHS:-40}"
LR_SCHEDULE_EPOCHS="${LR_SCHEDULE_EPOCHS:-$NURD_EPOCHS}"
NURD_ENABLED="${NURD_ENABLED:-0}"
CRITIC_START_EPOCH="${CRITIC_START_EPOCH:-0}"
CRITIC_SHUFFLE_MODE="${CRITIC_SHUFFLE_MODE:-global}"
CRITIC_WARMUP_EPOCHS="${CRITIC_WARMUP_EPOCHS:-0}"
LAMBDA_INFO_RAMP_EPOCHS="${LAMBDA_INFO_RAMP_EPOCHS:-0}"
SELECTION_AFTER_RAMP="${SELECTION_AFTER_RAMP:-0}"
LAMBDA_INFO="${LAMBDA_INFO:-1.0}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
WARM_START_CKPT="${WARM_START_CKPT:-}"
REUSE_AE="${REUSE_AE:-0}"
BALANCE_BINNING="${BALANCE_BINNING:-log_fixed}"
BALANCE_CLIP_QUANTILE="${BALANCE_CLIP_QUANTILE:-0.995}"
BALANCE_STRATA="${BALANCE_STRATA:-20}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
STATISTICALLY_VALID_CLOSURE="${STATISTICALLY_VALID_CLOSURE:-1}"
TARGET_SCHEME="${TARGET_SCHEME:-parent4}"
AE_WEIGHTING_SCHEME="${AE_WEIGHTING_SCHEME:-physics}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run tag may contain only letters, numbers, dot, underscore, and dash."
  exit 2
fi
if [[ ! "$CONTRAST_WEIGHT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  echo "ERROR: SUPCON_WEIGHT must be a non-negative number."
  exit 2
fi
for value_name in NURD_EPOCHS LR_SCHEDULE_EPOCHS CRITIC_START_EPOCH CHECKPOINT_EVERY CRITIC_WARMUP_EPOCHS LAMBDA_INFO_RAMP_EPOCHS BALANCE_STRATA; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer."
    exit 2
  fi
done
for switch_name in NURD_ENABLED SELECTION_AFTER_RAMP REUSE_AE; do
  switch_value="${!switch_name}"
  if [[ "$switch_value" != "0" && "$switch_value" != "1" ]]; then
    echo "ERROR: $switch_name must be zero or one."
    exit 2
  fi
done
for float_name in LAMBDA_INFO TRAIN_LR; do
  float_value="${!float_name}"
  if [[ ! "$float_value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
    echo "ERROR: $float_name must be a non-negative number."
    exit 2
  fi
done
if [[ "$REUSE_AE" == "1" && -z "${AE_EXP:-}" ]]; then
  echo "ERROR: REUSE_AE=1 requires AE_EXP to name an existing AE experiment."
  exit 2
fi
if [[ "$CRITIC_SHUFFLE_MODE" != "weighted_within_class" \
      && "$CRITIC_SHUFFLE_MODE" != "global" ]]; then
  echo "ERROR: CRITIC_SHUFFLE_MODE must be weighted_within_class or global."
  exit 2
fi
if [[ "$BALANCE_BINNING" != "log_fixed" \
      && "$BALANCE_BINNING" != "weighted_quantile" ]]; then
  echo "ERROR: BALANCE_BINNING must be log_fixed or weighted_quantile."
  exit 2
fi
if [[ ! "$BALANCE_CLIP_QUANTILE" =~ ^(0([.][0-9]+)?|1([.]0*)?)$ ]]; then
  echo "ERROR: BALANCE_CLIP_QUANTILE must be a decimal in [0, 1]."
  exit 2
fi
if ! awk -v value="$BALANCE_CLIP_QUANTILE" \
    'BEGIN { exit !(value >= 0.5 && value <= 1.0) }'; then
  echo "ERROR: BALANCE_CLIP_QUANTILE must lie in [0.5, 1.0]."
  exit 2
fi
if [[ "$STATISTICALLY_VALID_CLOSURE" != "0" \
      && "$STATISTICALLY_VALID_CLOSURE" != "1" ]]; then
  echo "ERROR: STATISTICALLY_VALID_CLOSURE must be zero or one."
  exit 2
fi
if [[ "$TARGET_SCHEME" != "parent4" && "$TARGET_SCHEME" != "decay8" ]]; then
  echo "ERROR: TARGET_SCHEME must be parent4 or decay8."
  exit 2
fi
if [[ "$AE_WEIGHTING_SCHEME" != "physics" \
      && "$AE_WEIGHTING_SCHEME" != "parent_balanced" ]]; then
  echo "ERROR: AE_WEIGHTING_SCHEME must be physics or parent_balanced."
  exit 2
fi
if (( NURD_EPOCHS < 1 )); then
  echo "ERROR: NURD_EPOCHS must be at least one."
  exit 2
fi
if (( LR_SCHEDULE_EPOCHS < 1 )); then
  echo "ERROR: LR_SCHEDULE_EPOCHS must be at least one."
  exit 2
fi
if (( BALANCE_STRATA < 1 )); then
  echo "ERROR: BALANCE_STRATA must be at least one."
  exit 2
fi
if (( NURD_ENABLED == 1 && CRITIC_START_EPOCH > NURD_EPOCHS )); then
  echo "ERROR: CRITIC_START_EPOCH must not exceed NURD_EPOCHS when NURD is enabled."
  exit 2
fi
AE_EXP="${AE_EXP:-ae_engineer_$RUN_TAG}"
NURD_EXP="${NURD_EXP:-hlt_nurd_engineer_$RUN_TAG}"

outputs=(
  "$BASE/checkpoints/hlt/hlt/$NURD_EXP"
  "$BASE/outputs/${RUN_TAG}_eval"
)
if [[ "$REUSE_AE" == "0" ]]; then
  outputs+=("$BASE/checkpoints/hlt/hlt/$AE_EXP")
elif [[ ! -s "$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth" ]]; then
  echo "ERROR: requested reusable AE checkpoint is missing."
  exit 2
fi
if [[ -n "$WARM_START_CKPT" && ! -s "$WARM_START_CKPT" ]]; then
  echo "ERROR: warm-start checkpoint is missing: $WARM_START_CKPT"
  exit 2
fi
for output in "${outputs[@]}"; do
  if [[ -e "$output" ]]; then
    echo "ERROR: refusing to overwrite existing campaign output: $output"
    exit 2
  fi
done

TRAIN_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_train" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP",CONTRAST_WEIGHT="$CONTRAST_WEIGHT",NURD_EPOCHS="$NURD_EPOCHS",LR_SCHEDULE_EPOCHS="$LR_SCHEDULE_EPOCHS",NURD_ENABLED="$NURD_ENABLED",CRITIC_START_EPOCH="$CRITIC_START_EPOCH",CRITIC_SHUFFLE_MODE="$CRITIC_SHUFFLE_MODE",CRITIC_WARMUP_EPOCHS="$CRITIC_WARMUP_EPOCHS",LAMBDA_INFO_RAMP_EPOCHS="$LAMBDA_INFO_RAMP_EPOCHS",SELECTION_AFTER_RAMP="$SELECTION_AFTER_RAMP",LAMBDA_INFO="$LAMBDA_INFO",TRAIN_LR="$TRAIN_LR",WARM_START_CKPT="$WARM_START_CKPT",REUSE_AE="$REUSE_AE",BALANCE_STRATA="$BALANCE_STRATA",BALANCE_BINNING="$BALANCE_BINNING",BALANCE_CLIP_QUANTILE="$BALANCE_CLIP_QUANTILE",CHECKPOINT_EVERY="$CHECKPOINT_EVERY",TARGET_SCHEME="$TARGET_SCHEME",AE_WEIGHTING_SCHEME="$AE_WEIGHTING_SCHEME" \
  slurm/submit_engineer_train.sbatch)"

EVAL_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_eval" \
  --dependency="afterok:$TRAIN_JOB" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP",STATISTICALLY_VALID_CLOSURE="$STATISTICALLY_VALID_CLOSURE",TARGET_SCHEME="$TARGET_SCHEME" \
  slurm/submit_engineer_dual_eval.sbatch)"

echo "Run tag:       $RUN_TAG"
echo "Commit:        $CODE_COMMIT"
echo "AE experiment: $AE_EXP"
echo "NURD:          $NURD_EXP"
echo "SupCon weight: $CONTRAST_WEIGHT"
echo "NURD maximum epochs: $NURD_EPOCHS"
echo "NURD LR schedule epochs: $LR_SCHEDULE_EPOCHS"
echo "NURD enabled: $NURD_ENABLED"
echo "Target scheme: $TARGET_SCHEME"
echo "AE weighting scheme: $AE_WEIGHTING_SCHEME"
echo "Critic start epoch: $CRITIC_START_EPOCH (0 means epoch 1)"
echo "Critic shuffle: $CRITIC_SHUFFLE_MODE"
echo "Critic-only warm-up: $CRITIC_WARMUP_EPOCHS"
echo "Lambda ramp epochs: $LAMBDA_INFO_RAMP_EPOCHS"
echo "Maximum information lambda: $LAMBDA_INFO"
echo "Selection after ramp: $SELECTION_AFTER_RAMP"
echo "Training learning rate: $TRAIN_LR"
echo "Warm-start checkpoint: ${WARM_START_CKPT:-none}"
echo "Reuse AE: $REUSE_AE"
echo "Nuisance balance strata: $BALANCE_STRATA"
echo "Nuisance weight binning: $BALANCE_BINNING"
echo "Effective-weight clip quantile: $BALANCE_CLIP_QUANTILE"
echo "Checkpoint interval: $CHECKPOINT_EVERY"
echo "Statistically valid evaluation: $STATISTICALLY_VALID_CLOSURE"
echo "Outputs:       $BASE/outputs/${RUN_TAG}_eval/{held-out,legacy}"
echo "Training job:  $TRAIN_JOB"
echo "Evaluation job: $EVAL_JOB"
echo
echo "Monitor with:"
echo "  squeue -j $TRAIN_JOB,$EVAL_JOB"
echo "  sacct -X -j $TRAIN_JOB,$EVAL_JOB --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit"
