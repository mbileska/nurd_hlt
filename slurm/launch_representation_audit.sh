#!/bin/bash
# One GPU job audits all three completed pure-SupCon checkpoints sequentially.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash slurm/launch_representation_audit.sh PURE_RUN_PREFIX [AUDIT_TAG]" >&2
  exit 2
fi
PURE_RUN_PREFIX="$1"
AUDIT_TAG="${2:-${PURE_RUN_PREFIX}_repaudit_$(date +%Y%m%d_%H%M%S)}"
for value in "$PURE_RUN_PREFIX" "$AUDIT_TAG"; do
  if [[ ! "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: invalid tag: $value" >&2
    exit 2
  fi
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
source "$SCRIPT_DIR/train3m_environment.sh"
if [[ "$TRAIN_PT" != "$TRAIN3M_CONVERTED_DIR/train/sample.json" \
   || "$TRAIN_WEIGHTS" != "$TRAIN3M_CONVERTED_DIR/weight_train.npz" ]]; then
  echo "ERROR: use the validated train3M outer-training manifest and weights." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked analysis changes before submission." >&2
  exit 2
fi
ANALYSIS_COMMIT="$(git rev-parse HEAD)"
AE_CKPT="$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
for suffix in 01 03 05; do
  checkpoint="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_${PURE_RUN_PREFIX}_sc${suffix}/checkpoint_main.pth.tar"
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: missing checkpoint: $checkpoint" >&2
    exit 2
  fi
done
if [[ ! -s "$AE_CKPT" ]]; then
  echo "ERROR: missing physics AE: $AE_CKPT" >&2
  exit 2
fi
OUTDIR="$BASE/outputs/$AUDIT_TAG"
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite $OUTDIR" >&2
  exit 2
fi
mkdir -p "$BASE/logs"
export BASE CODE_DIR ANALYSIS_COMMIT PURE_RUN_PREFIX AE_CKPT TRAIN_PT TRAIN_WEIGHTS OUTDIR
JOB="$(sbatch --parsable --job-name="${AUDIT_TAG}_repaudit" \
  --export=ALL "$SCRIPT_DIR/submit_representation_audit.sbatch")"
echo "Job: $JOB"
echo "Output: $OUTDIR"
echo "Summary: $OUTDIR/audit_summary.json"
echo "Probe table: $OUTDIR/probe_metrics.csv"
echo "Log: $BASE/logs/${AUDIT_TAG}_repaudit-${JOB}.out"
echo "Frozen networks; saved training fit / validation report; held-out unopened."
