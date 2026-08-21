#!/bin/bash
set -euo pipefail

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
mkdir -p "$BASE/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

if (( $# > 1 )); then
  echo "Usage: bash slurm/launch_engineer_campaign.sh [run_tag]"
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit or discard tracked changes before launching."
  exit 2
fi

CODE_COMMIT="$(git rev-parse HEAD)"
RUN_TAG="${1:-engineer_continuous_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run tag may contain only letters, numbers, dot, underscore, and dash."
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
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",AE_EXP="$AE_EXP",NURD_EXP="$NURD_EXP" \
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
echo "Outputs:       $BASE/outputs/${RUN_TAG}_eval/{held-out,legacy}"
echo "Training job:  $TRAIN_JOB"
echo "Evaluation job: $EVAL_JOB"
echo
echo "Monitor with:"
echo "  squeue -j $TRAIN_JOB,$EVAL_JOB"
echo "  sacct -X -j $TRAIN_JOB,$EVAL_JOB --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit"
