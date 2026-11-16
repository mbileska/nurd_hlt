#!/bin/bash
# CPU-only lambda=1 validation-tail closure using cached raw QCD MD scores.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash slurm/launch_ae_tail_validation.sh UNIQUE_TAG [raw-all|physics-qcd]" >&2
  exit 2
fi
TAG="$1"
SELECTION_MEASURE="${2:-raw-all}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: invalid tag" >&2
  exit 2
fi
if [[ "$SELECTION_MEASURE" != raw-all && "$SELECTION_MEASURE" != physics-qcd ]]; then
  echo "ERROR: selection measure must be raw-all or physics-qcd" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
source "$SCRIPT_DIR/train3m_environment.sh"
if [[ "$TRAIN_PT" != "$TRAIN3M_CONVERTED_DIR/train/sample.json" \
   || "$TRAIN_WEIGHTS" != "$TRAIN3M_CONVERTED_DIR/weight_train.npz" ]]; then
  echo "ERROR: use the validated outer-training manifest and weights" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submitting" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
RUN=train3m_physics_decay8_lambda1_20261107_132516
CHECKPOINT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_${RUN}/checkpoint_main.pth.tar"
AE_CKPT="$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
CACHE_DIR="${AE_TAIL_CACHE_DIR:-$BASE/outputs/${RUN}_conditional_md_20261108_154347}"
QCD_CACHE="$CACHE_DIR/validation_qcd_scores.npz"
CACHE_REPORT="$CACHE_DIR/conditional_md_report.json"
OUTDIR="$BASE/outputs/${TAG}_ae_tail_validation"
for path in "$CHECKPOINT" "$AE_CKPT" "$QCD_CACHE" "$CACHE_REPORT"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required input is missing: $path" >&2
    echo "This fast CPU analysis requires the existing lambda=1 raw-score cache; it will not silently rerun PF inference." >&2
    exit 2
  fi
done
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite $OUTDIR" >&2
  exit 2
fi
mkdir -p "$BASE/logs"
export BASE CODE_DIR CODE_COMMIT CHECKPOINT AE_CKPT TRAIN_PT TRAIN_WEIGHTS
export QCD_CACHE CACHE_REPORT SELECTION_MEASURE OUTDIR
JOB="$(sbatch --parsable --partition=cpu --job-name="${TAG}_aetail" \
  --export=ALL "$SCRIPT_DIR/submit_ae_tail_validation.sbatch")"
echo "Job: $JOB"
echo "Baseline: $RUN (selected checkpoint, no retraining)"
echo "Selection: top 10%; $SELECTION_MEASURE"
echo "Resources: CPU only; cached uncalibrated QCD MD; AE inference only"
echo "Output: $OUTDIR"
echo "Summary: $OUTDIR/tail_summary.json"
echo "Log: $BASE/logs/${TAG}_aetail-${JOB}.out"
