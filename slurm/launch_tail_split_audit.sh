#!/bin/bash
set -euo pipefail
if (( $# != 2 )) || [[ ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_tail_split_audit.sh PREPARATION_DIR_OR_latest UNIQUE_TAG" >&2
  exit 2
fi
PREP_DIR="$1"
TAG="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submission" >&2
  exit 2
fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
if [[ "$PREP_DIR" == latest ]]; then
  PREP_DIR="$(python - "$BASE/outputs" <<'PY'
from pathlib import Path
import sys
reports=sorted(Path(sys.argv[1]).glob('*_pilot_preparation/preparation_report.json'),key=lambda p:p.stat().st_mtime,reverse=True)
if not reports:
    raise SystemExit('No completed pilot preparation found')
print(reports[0].parent)
PY
)"
fi
PREP_DIR="$(cd "$PREP_DIR" && pwd -P)"
for file in preparation_report.json pilot_selection.npz; do
  if [[ ! -s "$PREP_DIR/$file" ]]; then
    echo "ERROR: preparation is incomplete; missing $PREP_DIR/$file" >&2
    echo "Inspect the preparation job stderr. If pilot_preparation.pth and pilot_selection.npz exist, recover the report without inference." >&2
    exit 2
  fi
done
AE_CKPT="$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
WEIGHTS="$BASE/data/train3M_v1/converted_v1/weight_train.npz"
OUTDIR="$BASE/outputs/${TAG}_split_audit"
CODE_COMMIT="$(git rev-parse HEAD)"
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite $OUTDIR" >&2
  exit 2
fi
for path in "$AE_CKPT" "$WEIGHTS"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required audit input missing: $path" >&2
    exit 2
  fi
done
export BASE CODE_DIR CODE_COMMIT PREP_DIR AE_CKPT WEIGHTS OUTDIR
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${TAG}_split_audit" --export=ALL \
  "$SCRIPT_DIR/submit_tail_split_audit.sbatch")"
echo "Audit job: $JOB"
echo "Preparation: $PREP_DIR"
echo "Output: $OUTDIR"
echo "Log: $BASE/logs/${TAG}_split_audit-${JOB}.out"
echo "No training, PF-encoder inference, or held-out evaluation."
