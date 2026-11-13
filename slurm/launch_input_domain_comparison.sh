#!/bin/bash
# Held-out Mequinna vs train3M input-domain comparison; no training or tuning.
set -euo pipefail
if (( $# != 1 )); then
  echo "Usage: bash slurm/launch_input_domain_comparison.sh UNIQUE_TAG" >&2
  exit 2
fi
TAG="$1"
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid output tag: $TAG" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit tracked changes before submitting" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
LEGACY_SAMPLE="${LEGACY_SAMPLE:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt}"
LEGACY_WEIGHTS="${LEGACY_WEIGHTS:-$MEQUINNA_DIR/weight_test.pt}"
NEW_SAMPLE="${NEW_SAMPLE:-$BASE/data/train3M_v1/converted_v1/test/sample.json}"
NEW_WEIGHTS="${NEW_WEIGHTS:-$BASE/data/train3M_v1/converted_v1/weight_test.npz}"
AE_CKPT="${AE_CKPT:-$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth}"
DOMAIN_OUT="$BASE/outputs/${TAG}_input_domain_comparison"
for path in "$LEGACY_SAMPLE" "$LEGACY_WEIGHTS" "$NEW_SAMPLE" "$NEW_WEIGHTS" "$AE_CKPT"; do
  test -s "$path" || { echo "Missing input: $path" >&2; exit 2; }
done
test ! -e "$DOMAIN_OUT" || { echo "Refusing to overwrite $DOMAIN_OUT" >&2; exit 2; }
export BASE CODE_DIR CODE_COMMIT LEGACY_SAMPLE LEGACY_WEIGHTS NEW_SAMPLE NEW_WEIGHTS AE_CKPT DOMAIN_OUT
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${TAG}_domain" --export=ALL \
  "$SCRIPT_DIR/submit_input_domain_comparison.sbatch")"
echo "Job: $JOB"
echo "Output: $DOMAIN_OUT"
echo "Log: $BASE/logs/${TAG}_domain-${JOB}.out"
