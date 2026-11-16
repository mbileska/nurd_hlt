#!/bin/bash
# Read-only standard plot suite for a completed physics-all validation-tail job.
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash slurm/launch_ae_tail_standard_plots.sh TAIL_OUTPUT_DIR_OR_latest UNIQUE_TAG [cpu|gpu]" >&2
  exit 2
fi
TAIL_DIR="$1"
TAG="$2"
RESOURCE="${3:-cpu}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ( "$RESOURCE" != cpu && "$RESOURCE" != gpu ) ]]; then
  echo "ERROR: invalid tag or resource (cpu/gpu)" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submitting" >&2
  exit 2
fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_COMMIT="$(git rev-parse HEAD)"
if [[ "$TAIL_DIR" == latest ]]; then
  TAIL_DIR="$(python - "$BASE/outputs" <<'PY'
import json
from pathlib import Path
import sys
reports = sorted(Path(sys.argv[1]).glob('*_ae_tail_validation/tail_summary.json'),
                 key=lambda path: path.stat().st_mtime, reverse=True)
for path in reports:
    try:
        report = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if report.get('selection', {}).get('measure') == 'physics-all':
        print(path.parent)
        break
else:
    raise SystemExit('No completed physics-all AE-tail report found. Wait for the tail-selection job first.')
PY
)"
fi
for name in tail_summary.json validation_ae_selection.npz validation_qcd_scores.npz closure_scan_arrays.npz; do
  if [[ ! -s "$TAIL_DIR/$name" ]]; then
    echo "ERROR: incomplete source tail output: $TAIL_DIR/$name" >&2
    exit 2
  fi
done
TAIL_DIR="$(cd "$TAIL_DIR" && pwd -P)"
OUTDIR="$BASE/outputs/${TAG}_standard_plots"
if [[ -e "$OUTDIR" ]]; then
  echo "ERROR: refusing to overwrite $OUTDIR" >&2
  exit 2
fi
SIGNAL_PT="${SIGNAL_PT:-$BASE/data/hlt_signal_TpTp.pt}"
if [[ "${INCLUDE_SIGNAL:-1}" == 0 || ! -s "$SIGNAL_PT" ]]; then
  SIGNAL_PT=""
fi
DEVICE=cpu
RESOURCE_ARGS=(--partition=cpu --time=24:00:00)
if [[ "$RESOURCE" == gpu ]]; then
  DEVICE=cuda
  RESOURCE_ARGS=(--partition=gpu --gres=gpu:1 --time=08:00:00)
fi
export BASE CODE_DIR CODE_COMMIT TAIL_DIR OUTDIR DEVICE SIGNAL_PT
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable "${RESOURCE_ARGS[@]}" --job-name="${TAG}_tailplots" \
  --export=ALL "$SCRIPT_DIR/submit_ae_tail_standard_plots.sbatch")"
echo "Job: $JOB"
echo "Source selection: $TAIL_DIR"
echo "Output: $OUTDIR/plots"
echo "Log: $BASE/logs/${TAG}_tailplots-${JOB}.out"
echo "Resources: $RESOURCE (generic GPU only if requested; no A100 constraint)"
echo "Frozen-model inference and plots only; no training, new cuts, or held-out backgrounds."
