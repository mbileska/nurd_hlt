#!/bin/bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$PWD}"
if [[ ! -f "$CODE_DIR/scripts/audit_hlt_data.py" || \
      ! -f "$CODE_DIR/slurm/submit_data_audit.sbatch" ]]; then
  echo "ERROR: launch this from the nurd_hlt repository, or set CODE_DIR explicitly."
  exit 2
fi

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
mkdir -p "$BASE/logs" "$CODE_DIR/data_plots"

JOB_ID="$(sbatch --parsable \
  --export=ALL,BASE="$BASE",CODE_DIR="$CODE_DIR" \
  "$CODE_DIR/slurm/submit_data_audit.sbatch")"

echo "Data-audit job: $JOB_ID"
echo "Output:         $CODE_DIR/data_plots/data_audit_$JOB_ID"
echo "Monitor:        squeue -j $JOB_ID"
echo "Accounting:     sacct -X -j $JOB_ID --format=JobID,JobName%25,State,ExitCode,Elapsed,Timelimit,MaxRSS"
echo "Log:            $BASE/logs/hlt_data_audit-$JOB_ID.out"
