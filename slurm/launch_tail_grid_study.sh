#!/bin/bash
set -euo pipefail
if (( $# != 2 )) || [[ ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_tail_grid_study.sh PILOT_DIR UNIQUE_TAG" >&2
  exit 2
fi
PILOT_DIR="$1"
TAG="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submission" >&2
  exit 2
fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CONVERTED="$BASE/data/train3M_v1/converted_v1"
AE_CACHE_DIR="${AE_CACHE_DIR:-$BASE/outputs/ae_phys_randomsplit_20261117_150546_AE_random_control}"
PILOT_BRANCH="${PILOT_BRANCH:-tail}"
case "$PILOT_BRANCH" in
  tail|full) CHECKPOINT="$PILOT_DIR/$PILOT_BRANCH/checkpoint_final.pth.tar" ;;
  source) CHECKPOINT="$PILOT_DIR/source_lambda1_checkpoint.pth.tar" ;;
  *) echo "ERROR: PILOT_BRANCH must be tail, full, or source" >&2; exit 2 ;;
esac
for path in "$PILOT_DIR/pilot_comparison.json" "$PILOT_DIR/pilot_roles.npz" \
            "$PILOT_DIR/${PILOT_BRANCH}_checking_QCD.npz" "$CHECKPOINT" \
            "$AE_CACHE_DIR/frozen_AE_scores.npz" "$CONVERTED/train/sample.json" \
            "$CONVERTED/weight_train.npz" "$CONVERTED/test/sample.json" "$CONVERTED/weight_test.npz"; do
  if [[ ! -s "$path" ]]; then echo "ERROR: missing/empty input $path" >&2; exit 2; fi
done
PILOT_DIR="$(cd "$PILOT_DIR" && pwd -P)"
STUDY_OUT="$BASE/outputs/${TAG}_tail_grid_study"
if [[ -e "$STUDY_OUT" ]]; then echo "ERROR: refusing to overwrite $STUDY_OUT" >&2; exit 2; fi
CODE_COMMIT="$(git rev-parse HEAD)"
export BASE CONVERTED PILOT_DIR AE_CACHE_DIR PILOT_BRANCH STUDY_OUT CODE_DIR CODE_COMMIT
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${TAG}_tailgrid" --export=ALL \
  --output="$BASE/logs/${TAG}_tailgrid-%j.out" --error="$BASE/logs/${TAG}_tailgrid-%j.err" \
  "$SCRIPT_DIR/submit_tail_grid_study.sbatch")"
echo "Grid study job: $JOB"
echo "Frozen model branch: $PILOT_BRANCH"
echo "Output: $STUDY_OUT"
echo "Log: $BASE/logs/${TAG}_tailgrid-${JOB}.out"
echo "No training. Generic GPU inference; no explicit partition, model, or node constraint."
