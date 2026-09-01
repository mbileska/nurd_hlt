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
INFO_WARMUP_EPOCHS="${INFO_WARMUP_EPOCHS:-0}"
INFO_RAMP_EPOCHS="${INFO_RAMP_EPOCHS:-0}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run tag may contain only letters, numbers, dot, underscore, and dash."
  exit 2
fi
if [[ ! "$CONTRAST_WEIGHT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  echo "ERROR: SUPCON_WEIGHT must be a non-negative number."
  exit 2
fi
for value_name in NURD_EPOCHS LR_SCHEDULE_EPOCHS INFO_WARMUP_EPOCHS INFO_RAMP_EPOCHS; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer."
    exit 2
  fi
done
if (( NURD_EPOCHS < 1 )); then
  echo "ERROR: NURD_EPOCHS must be at least one."
  exit 2
fi
if (( LR_SCHEDULE_EPOCHS < 1 )); then
  echo "ERROR: LR_SCHEDULE_EPOCHS must be at least one."
  exit 2
fi
if (( INFO_RAMP_EPOCHS > 0 )); then
  SELECTION_START_EPOCH=$((INFO_WARMUP_EPOCHS + INFO_RAMP_EPOCHS))
else
  SELECTION_START_EPOCH=$((INFO_WARMUP_EPOCHS + 1))
fi
if (( SELECTION_START_EPOCH > NURD_EPOCHS )); then
  echo "ERROR: the information schedule does not finish within NURD_EPOCHS."
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
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP",CONTRAST_WEIGHT="$CONTRAST_WEIGHT",NURD_EPOCHS="$NURD_EPOCHS",LR_SCHEDULE_EPOCHS="$LR_SCHEDULE_EPOCHS",INFO_WARMUP_EPOCHS="$INFO_WARMUP_EPOCHS",INFO_RAMP_EPOCHS="$INFO_RAMP_EPOCHS" \
  slurm/submit_engineer_train.sbatch)"

EVAL_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_eval" \
  --dependency="afterok:$TRAIN_JOB" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP" \
  slurm/submit_engineer_dual_eval.sbatch)"

echo "Run tag:       $RUN_TAG"
echo "Commit:        $CODE_COMMIT"
echo "AE experiment: $AE_EXP"
echo "NURD:          $NURD_EXP"
echo "SupCon weight: $CONTRAST_WEIGHT"
echo "NURD maximum epochs: $NURD_EPOCHS"
echo "NURD LR schedule epochs: $LR_SCHEDULE_EPOCHS"
echo "Information warm-up/ramp: $INFO_WARMUP_EPOCHS/$INFO_RAMP_EPOCHS"
echo "Outputs:       $BASE/outputs/${RUN_TAG}_eval/{held-out,legacy}"
echo "Training job:  $TRAIN_JOB"
echo "Evaluation job: $EVAL_JOB"
echo
echo "Monitor with:"
echo "  squeue -j $TRAIN_JOB,$EVAL_JOB"
echo "  sacct -X -j $TRAIN_JOB,$EVAL_JOB --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit"
