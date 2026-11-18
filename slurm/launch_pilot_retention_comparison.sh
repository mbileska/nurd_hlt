#!/bin/bash
set -euo pipefail
if (( $# != 2 )) || [[ ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_pilot_retention_comparison.sh PILOT_DIR UNIQUE_TAG" >&2
  exit 2
fi
PILOT_DIR="$1"
TAG="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submission" >&2; exit 2
fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CONVERTED="$BASE/data/train3M_v1/converted_v1"
AE_CACHE_DIR="${AE_CACHE_DIR:-$BASE/outputs/ae_phys_randomsplit_20261117_150546_AE_random_control}"
D_MIN_ESS="${D_MIN_ESS:-100}"
if [[ ! "$D_MIN_ESS" =~ ^[1-9][0-9]*$ ]]; then echo "ERROR: positive integer D_MIN_ESS required" >&2; exit 2; fi
for path in "$PILOT_DIR/pilot_comparison.json" "$PILOT_DIR/pilot_roles.npz" \
            "$PILOT_DIR/full/checkpoint_final.pth.tar" "$PILOT_DIR/tail/checkpoint_final.pth.tar" \
            "$PILOT_DIR/full_checking_QCD.npz" "$PILOT_DIR/tail_checking_QCD.npz" \
            "$AE_CACHE_DIR/frozen_AE_scores.npz" "$CONVERTED/train/sample.json" \
            "$CONVERTED/weight_train.npz" "$CONVERTED/test/sample.json" "$CONVERTED/weight_test.npz"; do
  if [[ ! -s "$path" ]]; then echo "ERROR: missing/empty input $path" >&2; exit 2; fi
done
PILOT_DIR="$(cd "$PILOT_DIR" && pwd -P)"
COMPARISON_OUT="$BASE/outputs/${TAG}_retention_comparison"
if [[ -e "$COMPARISON_OUT" ]]; then echo "ERROR: refusing to overwrite $COMPARISON_OUT" >&2; exit 2; fi
CODE_COMMIT="$(git rev-parse HEAD)"
export BASE CONVERTED PILOT_DIR AE_CACHE_DIR COMPARISON_OUT CODE_DIR CODE_COMMIT D_MIN_ESS
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${TAG}_retentions" --export=ALL \
  --output="$BASE/logs/${TAG}_retentions-%j.out" --error="$BASE/logs/${TAG}_retentions-%j.err" \
  "$SCRIPT_DIR/submit_pilot_retention_comparison.sbatch")"
echo "Comparison job: $JOB"
echo "Output: $COMPARISON_OUT"
echo "Log: $BASE/logs/${TAG}_retentions-${JOB}.out"
echo "Models: full and tail (frozen); retention: 100%, 20%, 10%; overall QCD A targets: 0.5%, 1%, 2%."
echo "ESS requirements: A/B/C=50, D=$D_MIN_ESS; no changes to earlier studies."
echo "No training. One generic GPU; no explicit partition, GPU model, or node constraint."
