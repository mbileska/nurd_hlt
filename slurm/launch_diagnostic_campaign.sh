#!/bin/bash
set -euo pipefail

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

if (( $# != 2 )); then
  echo "Usage: bash slurm/launch_diagnostic_campaign.sh RUN_TAG SUPCON_WEIGHT"
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit or discard tracked changes before launching."
  exit 2
fi

RUN_TAG="$1"
CONTRAST_WEIGHT="$2"
CODE_COMMIT="$(git rev-parse HEAD)"
DIAG_MAX_EVENTS="${DIAG_MAX_EVENTS:-120000}"
DIAG_AE_EPOCHS="${DIAG_AE_EPOCHS:-5}"
DIAG_SUPCON_EPOCHS="${DIAG_SUPCON_EPOCHS:-5}"
DIAG_NURD_EPOCHS="${DIAG_NURD_EPOCHS:-3}"
DIAG_CRITIC_PROBE_EPOCHS="${DIAG_CRITIC_PROBE_EPOCHS:-3}"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-4096}"
BALANCE_BINNING="${BALANCE_BINNING:-log_fixed}"
BALANCE_CLIP_QUANTILE="${BALANCE_CLIP_QUANTILE:-0.995}"
BALANCE_STRATA="${BALANCE_STRATA:-20}"
TARGET_SCHEME="${TARGET_SCHEME:-parent4}"
CRITIC_SHUFFLE_MODE="${CRITIC_SHUFFLE_MODE:-global}"
LAMBDA_INFO="${LAMBDA_INFO:-1.0}"
CRITIC_WARMUP_EPOCHS="${CRITIC_WARMUP_EPOCHS:-0}"
LAMBDA_INFO_RAMP_EPOCHS="${LAMBDA_INFO_RAMP_EPOCHS:-0}"
SELECTION_AFTER_RAMP="${SELECTION_AFTER_RAMP:-0}"

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: run tag may contain only letters, numbers, dot, underscore, and dash."
  exit 2
fi
if [[ ! "$CONTRAST_WEIGHT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  echo "ERROR: SUPCON_WEIGHT must be a non-negative number."
  exit 2
fi
if [[ "$TARGET_SCHEME" != "parent4" && "$TARGET_SCHEME" != "decay8" ]]; then
  echo "ERROR: TARGET_SCHEME must be parent4 or decay8."
  exit 2
fi
if [[ "$CRITIC_SHUFFLE_MODE" != "global" \
      && "$CRITIC_SHUFFLE_MODE" != "weighted_within_class" ]]; then
  echo "ERROR: CRITIC_SHUFFLE_MODE must be global or weighted_within_class."
  exit 2
fi
if [[ "$SELECTION_AFTER_RAMP" != "0" && "$SELECTION_AFTER_RAMP" != "1" ]]; then
  echo "ERROR: SELECTION_AFTER_RAMP must be zero or one."
  exit 2
fi
if [[ ! "$LAMBDA_INFO" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  echo "ERROR: LAMBDA_INFO must be a non-negative number."
  exit 2
fi
for name in DIAG_MAX_EVENTS DIAG_AE_EPOCHS DIAG_SUPCON_EPOCHS DIAG_NURD_EPOCHS DIAG_CRITIC_PROBE_EPOCHS DIAG_BATCH_SIZE BALANCE_STRATA; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $name must be a positive integer."
    exit 2
  fi
done
for name in CRITIC_WARMUP_EPOCHS LAMBDA_INFO_RAMP_EPOCHS; do
  value="${!name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $name must be a non-negative integer."
    exit 2
  fi
done

DIAG_ROOT="$BASE/outputs/${RUN_TAG}_diagnostic"
AE_DIR="$BASE/checkpoints/hlt/hlt/ae_diagnostic_$RUN_TAG"
NURD_DIR="$BASE/checkpoints/hlt/hlt/hlt_diagnostic_$RUN_TAG"
for output in "$DIAG_ROOT" "$AE_DIR" "$NURD_DIR"; do
  if [[ -e "$output" ]]; then
    echo "ERROR: refusing to overwrite diagnostic output: $output"
    exit 2
  fi
done
mkdir -p "$BASE/logs"

JOB_ID="$(sbatch --parsable \
  --job-name="${RUN_TAG}_diagnostic" \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR",CODE_COMMIT="$CODE_COMMIT",RUN_TAG="$RUN_TAG",CONTRAST_WEIGHT="$CONTRAST_WEIGHT",DIAG_MAX_EVENTS="$DIAG_MAX_EVENTS",DIAG_AE_EPOCHS="$DIAG_AE_EPOCHS",DIAG_SUPCON_EPOCHS="$DIAG_SUPCON_EPOCHS",DIAG_NURD_EPOCHS="$DIAG_NURD_EPOCHS",DIAG_CRITIC_PROBE_EPOCHS="$DIAG_CRITIC_PROBE_EPOCHS",DIAG_BATCH_SIZE="$DIAG_BATCH_SIZE",BALANCE_BINNING="$BALANCE_BINNING",BALANCE_CLIP_QUANTILE="$BALANCE_CLIP_QUANTILE",BALANCE_STRATA="$BALANCE_STRATA",TARGET_SCHEME="$TARGET_SCHEME",CRITIC_SHUFFLE_MODE="$CRITIC_SHUFFLE_MODE",LAMBDA_INFO="$LAMBDA_INFO",CRITIC_WARMUP_EPOCHS="$CRITIC_WARMUP_EPOCHS",LAMBDA_INFO_RAMP_EPOCHS="$LAMBDA_INFO_RAMP_EPOCHS",SELECTION_AFTER_RAMP="$SELECTION_AFTER_RAMP" \
  slurm/submit_diagnostic_campaign.sbatch)"

echo "Diagnostic run: $RUN_TAG"
echo "Commit:         $CODE_COMMIT"
echo "Job:            $JOB_ID"
echo "Output:         $DIAG_ROOT"
echo "Stages:         AE=$DIAG_AE_EPOCHS, SupCon-only=$DIAG_SUPCON_EPOCHS, NURD=$DIAG_NURD_EPOCHS, frozen-critic=$DIAG_CRITIC_PROBE_EPOCHS"
echo "Subset events:  $DIAG_MAX_EVENTS"
echo "Target scheme:  $TARGET_SCHEME"
echo "Critic shuffle: $CRITIC_SHUFFLE_MODE"
echo "NURD schedule:  warmup=$CRITIC_WARMUP_EPOCHS ramp=$LAMBDA_INFO_RAMP_EPOCHS lambda_max=$LAMBDA_INFO"
echo
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  sacct -X -j $JOB_ID --format=JobID,JobName%40,State,ExitCode,Elapsed,Timelimit"
