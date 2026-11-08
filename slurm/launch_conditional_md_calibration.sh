#!/bin/bash
# Submit a train-QCD-fitted conditional MD calibration; validation QCD only.
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash slurm/launch_conditional_md_calibration.sh RUN_TAG AE_EXP [CHECKPOINT_FILENAME]" >&2
  exit 2
fi
RUN_TAG="$1"
AE_EXP="$2"
CHECKPOINT_FILENAME="${3:-checkpoint_main.pth.tar}"
for value in "$RUN_TAG" "$AE_EXP" "$CHECKPOINT_FILENAME"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: tags and checkpoint filename may contain only alphanumerics, dot, underscore, dash." >&2
    exit 2
  fi
done
if [[ "$CHECKPOINT_FILENAME" != checkpoint*.pth.tar ]]; then
  echo "ERROR: checkpoint filename must be a checkpoint*.pth.tar file." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
source "$SCRIPT_DIR/train3m_environment.sh"
if [[ "$TRAIN_PT" != "$TRAIN3M_CONVERTED_DIR/train/sample.json" \
   || "$TRAIN_WEIGHTS" != "$TRAIN3M_CONVERTED_DIR/weight_train.npz" ]]; then
  echo "ERROR: this analysis requires the validated train3M outer-training split." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked code changes before submitting analysis." >&2
  exit 2
fi
ANALYSIS_COMMIT="$(git rev-parse HEAD)"
CHECKPOINT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_$RUN_TAG/$CHECKPOINT_FILENAME"
AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
for input in "$CHECKPOINT" "$AE_CKPT"; do
  if [[ ! -s "$input" ]]; then
    echo "ERROR: missing checkpoint: $input" >&2
    exit 2
  fi
done
SCAN_TAG="conditional_md_$(date +%Y%m%d_%H%M%S)"
OUTDIR="$BASE/outputs/${RUN_TAG}_${SCAN_TAG}"
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite output: $OUTDIR" >&2
  exit 2
fi
AE_BINS=20
MIN_FIT_BIN_ESS=200
export BASE CODE_DIR ANALYSIS_COMMIT CHECKPOINT AE_CKPT TRAIN_PT TRAIN_WEIGHTS
export OUTDIR AE_BINS MIN_FIT_BIN_ESS
mkdir -p "$BASE/logs"
JOB_ID="$(sbatch --parsable --job-name="${RUN_TAG}_condmd" \
  --export=ALL "$SCRIPT_DIR/submit_conditional_md_calibration.sbatch")"

echo "Conditional MD validation: $RUN_TAG"
echo "Checkpoint:    $CHECKPOINT"
echo "AE:            $AE_CKPT"
echo "Physics weights: $TRAIN_WEIGHTS"
echo "Fit:           training QCD only, 20 equal-weight AE bins; minimum ESS 200/bin"
echo "Report:        disjoint saved validation QCD only; outer test untouched"
echo "Output:        $OUTDIR"
echo "Job:           $JOB_ID"
echo "Monitor:       sacct -X -j $JOB_ID --format=JobID,State,ExitCode,Elapsed"
