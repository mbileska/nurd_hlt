#!/bin/bash
# Relaunch only the dual evaluation for a completed engineer campaign.

set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: bash slurm/launch_engineer_eval.sh <existing_run_tag>"
  exit 2
fi

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
RUN_TAG="$1"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run tag may contain only letters, numbers, dot, underscore, and dash."
  exit 2
fi

mkdir -p "$BASE/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source files have uncommitted changes."
  exit 2
fi

CODE_COMMIT="$(git rev-parse HEAD)"
AE_EXP="${AE_EXP:-ae_engineer_$RUN_TAG}"
NURD_EXP="${NURD_EXP:-hlt_nurd_engineer_$RUN_TAG}"
AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
NURD_CKPT="$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_main.pth.tar"
OUTPUT_ROOT="$BASE/outputs/${RUN_TAG}_eval"

for checkpoint in "$AE_CKPT" "$NURD_CKPT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: completed training checkpoint is missing: $checkpoint"
    exit 2
  fi
done
if [[ -s "$OUTPUT_ROOT/evaluation_summary.json" ]]; then
  echo "ERROR: this evaluation is already complete: $OUTPUT_ROOT/evaluation_summary.json"
  exit 2
fi

EVAL_JOB="$(sbatch --parsable \
  --job-name="${RUN_TAG}_eval_retry" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP" \
  slurm/submit_engineer_dual_eval.sbatch)"

echo "Evaluation-only retry submitted."
echo "Run tag: $RUN_TAG"
echo "Commit:  $CODE_COMMIT"
echo "Job:     $EVAL_JOB"
echo "Output:  $OUTPUT_ROOT/{held-out,legacy}"
echo "Monitor: squeue -j $EVAL_JOB"
echo "Accounting: sacct -X -j $EVAL_JOB --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit"
