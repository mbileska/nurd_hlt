#!/bin/bash
set -euo pipefail
if (( $# != 1 )); then
  echo "Usage: bash slurm/launch_qcd_lambda_pair_audit.sh UNIQUE_TAG" >&2
  exit 2
fi
AUDIT_TAG="$1"
if [[ ! "$AUDIT_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid tag" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
source "$SCRIPT_DIR/train3m_environment.sh"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit tracked changes before submitting" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
SOURCE_RUN=train3m_physics_decay8_full_20261104_142222
CANDIDATE_RUN=train3m_physics_decay8_lambda1_20261107_132516
AUDIT_OUT="$BASE/outputs/$AUDIT_TAG"
test ! -e "$AUDIT_OUT"
export BASE CODE_DIR CODE_COMMIT SOURCE_RUN CANDIDATE_RUN AUDIT_OUT
mkdir -p "$BASE/logs"
JOB="$(sbatch --parsable --job-name="${AUDIT_TAG}_qcd_audit" \
  --export=ALL "$SCRIPT_DIR/submit_qcd_lambda_pair_audit.sbatch")"
echo "Job: $JOB"
echo "Output: $AUDIT_OUT/paired"
echo "Log: $BASE/logs/${AUDIT_TAG}_qcd_audit-${JOB}.out"
