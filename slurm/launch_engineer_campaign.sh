#!/bin/bash
set -euo pipefail

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
mkdir -p "$BASE/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

if (( $# != 2 )); then
  echo "Usage: bash slurm/launch_engineer_campaign.sh RUN_TAG SUPCON_WEIGHT"
  echo "Example: bash slurm/launch_engineer_campaign.sh engineer_continuous_supcon040 0.4"
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
BALANCE_BINNING="${BALANCE_BINNING:-log_fixed}"
BALANCE_CLIP_QUANTILE="${BALANCE_CLIP_QUANTILE:-0.995}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
STATISTICALLY_VALID_CLOSURE="${STATISTICALLY_VALID_CLOSURE:-1}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run tag may contain only letters, numbers, dot, underscore, and dash."
  exit 2
fi
if [[ ! "$CONTRAST_WEIGHT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  echo "ERROR: SUPCON_WEIGHT must be a non-negative number."
  exit 2
fi
for value_name in NURD_EPOCHS LR_SCHEDULE_EPOCHS CRITIC_START_EPOCH CHECKPOINT_EVERY; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer."
    exit 2
  fi
done
if [[ "$NURD_ENABLED" != "0" && "$NURD_ENABLED" != "1" ]]; then
  echo "ERROR: NURD_ENABLED must be zero or one."
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
if (( NURD_EPOCHS < 1 )); then
  echo "ERROR: NURD_EPOCHS must be at least one."
  exit 2
fi
if (( LR_SCHEDULE_EPOCHS < 1 )); then
  echo "ERROR: LR_SCHEDULE_EPOCHS must be at least one."
  exit 2
fi
if (( NURD_ENABLED == 1 && CRITIC_START_EPOCH > NURD_EPOCHS )); then
  echo "ERROR: CRITIC_START_EPOCH must not exceed NURD_EPOCHS when NURD is enabled."
  exit 2
fi
AE_EXP="ae_engineer_$RUN_TAG"
NURD_EXP="hlt_nurd_engineer_$RUN_TAG"

for output in \
  "$BASE/checkpoints/hlt/hlt/$AE_EXP" \
  "$BASE/checkpoints/hlt/hlt/$NURD_EXP" \
  "$BASE/outputs/${RUN_TAG}_eval"; do
  if [[ -e "$output" ]]; then
    echo "ERROR: refusing to overwrite existing campaign output: $output"
    exit 2
  fi
done

TRAIN_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_train" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP",CONTRAST_WEIGHT="$CONTRAST_WEIGHT",NURD_EPOCHS="$NURD_EPOCHS",LR_SCHEDULE_EPOCHS="$LR_SCHEDULE_EPOCHS",NURD_ENABLED="$NURD_ENABLED",CRITIC_START_EPOCH="$CRITIC_START_EPOCH",CRITIC_SHUFFLE_MODE="$CRITIC_SHUFFLE_MODE",BALANCE_BINNING="$BALANCE_BINNING",BALANCE_CLIP_QUANTILE="$BALANCE_CLIP_QUANTILE",CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  slurm/submit_engineer_train.sbatch)"

EVAL_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_eval" \
  --dependency="afterok:$TRAIN_JOB" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP",STATISTICALLY_VALID_CLOSURE="$STATISTICALLY_VALID_CLOSURE" \
  slurm/submit_engineer_dual_eval.sbatch)"

echo "Run tag:       $RUN_TAG"
echo "Commit:        $CODE_COMMIT"
echo "AE experiment: $AE_EXP"
echo "NURD:          $NURD_EXP"
echo "SupCon weight: $CONTRAST_WEIGHT"
echo "NURD maximum epochs: $NURD_EPOCHS"
echo "NURD LR schedule epochs: $LR_SCHEDULE_EPOCHS"
echo "NURD enabled: $NURD_ENABLED"
echo "Critic start epoch: $CRITIC_START_EPOCH (0 means epoch 1)"
echo "Critic shuffle: $CRITIC_SHUFFLE_MODE"
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
