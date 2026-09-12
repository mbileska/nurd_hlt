#!/bin/bash
set -euo pipefail

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

if (( $# < 1 )); then
  echo "Usage: bash slurm/launch_validation_checkpoint_scan.sh RUN_TAG [EPOCH ...]"
  echo "Example: bash slurm/launch_validation_checkpoint_scan.sh engineer_continuous_supcon030_conditional_shuffle_v1 5 10 15 20 25 28 30"
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit or discard tracked changes before launching."
  exit 2
fi

RUN_TAG="$1"
shift
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run tag may contain only letters, numbers, dot, underscore, and dash."
  exit 2
fi
EPOCHS_CSV=""
if (( $# > 0 )); then
  for epoch in "$@"; do
    if [[ ! "$epoch" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: epochs must be positive integers."
      exit 2
    fi
  done
  EPOCHS_CSV="$(IFS=,; echo "$*")"
fi

for input in \
  "$BASE/checkpoints/hlt/hlt/ae_engineer_$RUN_TAG/checkpoint_ae.pth" \
  "$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_$RUN_TAG"; do
  if [[ ! -e "$input" ]]; then
    echo "ERROR: required trained-model input is missing: $input"
    exit 2
  fi
done
OUTDIR="$BASE/outputs/${RUN_TAG}_validation_checkpoint_scan"
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite existing scan output: $OUTDIR"
  exit 2
fi

ANALYSIS_COMMIT="$(git rev-parse HEAD)"
JOB_ID="$(sbatch --parsable \
  --job-name="${RUN_TAG}_valscan" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",ANALYSIS_COMMIT="$ANALYSIS_COMMIT",RUN_TAG="$RUN_TAG",EPOCHS_CSV="$EPOCHS_CSV" \
  slurm/submit_validation_checkpoint_scan.sbatch)"

echo "Run tag:        $RUN_TAG"
echo "Analysis commit: $ANALYSIS_COMMIT"
echo "Epochs:         ${EPOCHS_CSV:-all saved epochs}"
echo "Protocol:       training rows fit MD; validation rows report closure"
echo "Held-out data:  NOT ACCESSED"
echo "Output:         $OUTDIR"
echo "Job:            $JOB_ID"
echo
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  sacct -X -j $JOB_ID --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit"
