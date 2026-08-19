#!/bin/bash
# Submit one exact weighted V4x4 training job and its dependent dual evaluation.
# Set SKIP_AE=1 and AE_CKPT=/absolute/path to reuse an existing weighted AE;
# otherwise the campaign trains its own AE from the Mequinna training sample.

set -euo pipefail

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-$HOME/nurd_hlt}"
cd "$CODE_DIR"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked code has uncommitted changes. Commit/pull before submitting."
  exit 2
fi

CODE_COMMIT="$(git rev-parse HEAD)"
RUN_TAG="${RUN_TAG:-weighted_v4_x4_$(date +%Y%m%d_%H%M%S)}"
AE_EXP="${AE_EXP:-ae_pretrain_$RUN_TAG}"
NURD_EXP="${NURD_EXP:-hlt_nurd_closure_bs4096_$RUN_TAG}"
EVAL_NAME="${EVAL_NAME:-${RUN_TAG}_eval}"
SKIP_AE="${SKIP_AE:-0}"
AE_CKPT="${AE_CKPT:-$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth}"

if [[ "$SKIP_AE" == "1" && ! -s "$AE_CKPT" ]]; then
  echo "ERROR: SKIP_AE=1 but the exact AE checkpoint is missing: $AE_CKPT"
  exit 2
fi

TRAIN_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_train" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP",SKIP_AE="$SKIP_AE",AE_CKPT="$AE_CKPT" \
  slurm/submit_train.sbatch)"

EVAL_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_eval" \
  --dependency="afterok:$TRAIN_JOB" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",NURD_EXP="$NURD_EXP",AE_CKPT="$AE_CKPT",EVAL_NAME="$EVAL_NAME" \
  slurm/submit_eval_latest.sbatch)"

echo "Run tag:    $RUN_TAG"
echo "Commit:     $CODE_COMMIT"
echo "AE:         $AE_CKPT"
echo "NURD:       $NURD_EXP"
echo "Evaluation: $BASE/outputs/$EVAL_NAME/{held-out,legacy}"
echo "Training job:   $TRAIN_JOB"
echo "Evaluation job: $EVAL_JOB"
echo
echo "Monitor with:"
echo "  squeue -j $TRAIN_JOB,$EVAL_JOB"
echo "  sacct -X -j $TRAIN_JOB,$EVAL_JOB --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit"
