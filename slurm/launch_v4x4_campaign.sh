#!/bin/bash
# Submit one exact, fresh weighted V4x4 training job and its dependent dual eval.
#
# The campaign deliberately does not read run/checkpoint/model configuration from
# the caller's environment.  Interactive shells commonly retain exported values
# from earlier runs; accepting those values can silently redirect a new campaign
# into an old checkpoint directory or alter the training profile.

set -euo pipefail

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

if (( $# > 1 )); then
  echo "Usage: bash slurm/launch_v4x4_campaign.sh [run_tag]"
  exit 2
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked code has uncommitted changes. Commit/pull before submitting."
  exit 2
fi

CODE_COMMIT="$(git rev-parse HEAD)"
RUN_TAG="${1:-weighted_v4_x4_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run_tag may contain only letters, digits, dot, underscore, and dash."
  exit 2
fi
AE_EXP="ae_pretrain_$RUN_TAG"
NURD_EXP="hlt_nurd_closure_bs4096_$RUN_TAG"
EVAL_NAME="${RUN_TAG}_eval"
SKIP_AE="0"
AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
CAMPAIGN_CONTRACT="weighted_v4_x4_fresh_v1"
EVAL_CONTRACT="dual_qcd_v1"

for output in \
  "$BASE/checkpoints/hlt/hlt/$AE_EXP" \
  "$BASE/checkpoints/hlt/hlt/$NURD_EXP" \
  "$BASE/outputs/$EVAL_NAME"; do
  if [[ -e "$output" ]]; then
    echo "ERROR: refusing to overwrite an existing campaign output: $output"
    exit 2
  fi
done

TRAIN_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_train" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",CAMPAIGN_CONTRACT="$CAMPAIGN_CONTRACT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP",SKIP_AE="$SKIP_AE",AE_CKPT="$AE_CKPT" \
  slurm/submit_train.sbatch)"

EVAL_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_eval" \
  --dependency="afterok:$TRAIN_JOB" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",EVAL_CONTRACT="$EVAL_CONTRACT",NURD_EXP="$NURD_EXP",AE_CKPT="$AE_CKPT",EVAL_NAME="$EVAL_NAME" \
  slurm/submit_eval_latest.sbatch)"

echo "Run tag:    $RUN_TAG"
echo "Commit:     $CODE_COMMIT"
echo "Mode:       fresh AE + weighted V4x4 NURD"
echo "AE:         $AE_CKPT"
echo "NURD:       $NURD_EXP"
echo "Evaluation: $BASE/outputs/$EVAL_NAME/{held-out,legacy}"
echo "Training job:   $TRAIN_JOB"
echo "Evaluation job: $EVAL_JOB"
echo
echo "Monitor with:"
echo "  squeue -j $TRAIN_JOB,$EVAL_JOB"
echo "  sacct -X -j $TRAIN_JOB,$EVAL_JOB --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit"
