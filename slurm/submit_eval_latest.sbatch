#!/bin/bash
#SBATCH --job-name=nurd_eval_dual
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -eo pipefail

module load anaconda3/2025.12
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"

# Seal the dual-evaluation protocol when invoked by the campaign launcher.
# In particular, discard a stale CKPT: it would otherwise take precedence over
# the exact new NURD_EXP even though the launcher supplied the right run name.
if [[ "${EVAL_CONTRACT:-}" == "dual_qcd_v1" ]]; then
  unset MEQUINNA_DIR HELDOUT_TEST_PT HELDOUT_REFERENCE_PT
  unset HELDOUT_TEST_WEIGHTS HELDOUT_REFERENCE_WEIGHTS LEGACY_TEST_PT
  unset SIGNAL_PT SIGNAL_WEIGHTS EVAL_BATCH_SIZE AE_BATCH_SIZE N_PCA
  unset DCOR_SAMPLE_SIZE BASELINE_LABELS SCAN_PERCENT_MIN SCAN_PERCENT_MAX
  unset SCAN_PERCENT_STEPS MIN_A MIN_A_FRAC MIN_D MIN_REGION_FRAC
  unset MAX_RATIO_UNC SELECTION_FOLDS SELECTION_STAT_WEIGHT
  unset SELECTION_NEIGHBOR_WEIGHT SELECTION_NEIGHBOR_RADIUS SYNC_WANDB
  unset CKPT OUTDIR_ROOT
fi

MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
HELDOUT_TEST_PT="${HELDOUT_TEST_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt}"
HELDOUT_REFERENCE_PT="${HELDOUT_REFERENCE_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt}"
HELDOUT_TEST_WEIGHTS="${HELDOUT_TEST_WEIGHTS:-$MEQUINNA_DIR/weight_test.pt}"
HELDOUT_REFERENCE_WEIGHTS="${HELDOUT_REFERENCE_WEIGHTS:-$MEQUINNA_DIR/weight_train.pt}"
LEGACY_TEST_PT="${LEGACY_TEST_PT:-$BASE/data/hlt_smcocktail_test.pt}"
SIGNAL_PT="${SIGNAL_PT:-$BASE/data/hlt_signal_TpTp.pt}"
SIGNAL_WEIGHTS="${SIGNAL_WEIGHTS:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-4096}"
N_PCA="${N_PCA:-6}"
DCOR_SAMPLE_SIZE="${DCOR_SAMPLE_SIZE:-2000}"
BASELINE_LABELS="${BASELINE_LABELS:-0,1,2,3}"
SCAN_PERCENT_MIN="${SCAN_PERCENT_MIN:-0.50}"
SCAN_PERCENT_MAX="${SCAN_PERCENT_MAX:-0.98}"
SCAN_PERCENT_STEPS="${SCAN_PERCENT_STEPS:-48}"
MIN_A="${MIN_A:-50}"
MIN_A_FRAC="${MIN_A_FRAC:-0.10}"
MIN_D="${MIN_D:-500}"
MIN_REGION_FRAC="${MIN_REGION_FRAC:-0.01}"
MAX_RATIO_UNC="${MAX_RATIO_UNC:-0.15}"
SELECTION_FOLDS="${SELECTION_FOLDS:-5}"
SELECTION_STAT_WEIGHT="${SELECTION_STAT_WEIGHT:-0.5}"
SELECTION_NEIGHBOR_WEIGHT="${SELECTION_NEIGHBOR_WEIGHT:-1.0}"
SELECTION_NEIGHBOR_RADIUS="${SELECTION_NEIGHBOR_RADIUS:-1}"
SYNC_WANDB="${SYNC_WANDB:-0}"

mkdir -p "$BASE/logs" "$BASE/wandb" "$BASE/matplotlib" "$BASE/outputs"
export WANDB_DIR="$BASE/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MPLCONFIGDIR="$BASE/matplotlib"

cd "$CODE_DIR"
if [[ -e checkpoints && ! -L checkpoints ]]; then
  echo "ERROR: $CODE_DIR/checkpoints exists and is not a symlink."
  exit 2
fi
ln -sfnT "$BASE/checkpoints" checkpoints

if [[ -z "${CKPT:-}" ]]; then
  if [[ -z "${NURD_EXP:-}" ]]; then
    echo "ERROR: set exact NURD_EXP or CKPT; wildcard/latest discovery is disabled."
    exit 2
  fi
  NURD_DIR="$BASE/checkpoints/hlt/hlt/$NURD_EXP"
  CKPT="$NURD_DIR/checkpoint_abcd.pth.tar"
fi
if [[ ! -s "${CKPT:-}" ]]; then
  echo "ERROR: statistically eligible ABCD checkpoint not found: ${CKPT:-unset}"
  echo "The dual evaluation never falls back to a classifier-loss main checkpoint."
  exit 2
fi
NURD_EXP="${NURD_EXP:-$(basename "$(dirname "$CKPT")")}" # exact directory name

python - "$CKPT" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
if int(payload.get("checkpoint_contract_version", 0)) < 2:
    raise SystemExit("ERROR: dual evaluation requires a contract-v2 checkpoint.")
if not payload.get("selection_metric"):
    raise SystemExit(
        "ERROR: checkpoint was not selected by the constrained ABCD metric.")
PY

if [[ -z "${AE_CKPT:-}" ]]; then
  AE_CKPT="$(python - "$CKPT" "$BASE" "$CODE_DIR" <<'PY'
import os
import sys
import torch

checkpoint, base, code_dir = sys.argv[1:]
payload = torch.load(checkpoint, map_location="cpu")
path = payload.get("config", {}).get("ae_ckpt", "")
if path and not os.path.isabs(path):
    path = os.path.normpath(os.path.join(code_dir, path))
if path and not os.path.exists(path) and "/checkpoints/" in path:
    path = os.path.join(base, "checkpoints", path.split("/checkpoints/", 1)[1])
print(path)
PY
)"
fi
if [[ ! -s "${AE_CKPT:-}" ]]; then
  echo "ERROR: the checkpoint's exact AE could not be resolved; set AE_CKPT explicitly."
  echo "AE_CKPT=${AE_CKPT:-unset}"
  exit 2
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
CHECKPOINT_COMMIT="$(python - "$CKPT" <<'PY'
import sys
import torch
payload = torch.load(sys.argv[1], map_location="cpu")
print(payload.get("config", {}).get("code_commit", ""))
PY
)"
if [[ -z "$CHECKPOINT_COMMIT" || "$CURRENT_COMMIT" != "$CHECKPOINT_COMMIT" ]]; then
  echo "ERROR: code/checkpoint provenance mismatch."
  echo "Current:    $CURRENT_COMMIT"
  echo "Checkpoint: ${CHECKPOINT_COMMIT:-missing}"
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked code has uncommitted changes."
  exit 2
fi

EVAL_NAME="${EVAL_NAME:-${NURD_EXP}_eval_${SLURM_JOB_ID:-manual}}"
if [[ ! "$EVAL_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: EVAL_NAME may contain only letters, numbers, dot, underscore, and dash."
  exit 2
fi
OUTDIR_ROOT="${OUTDIR_ROOT:-$BASE/outputs/$EVAL_NAME}"
if [[ -d "$OUTDIR_ROOT" && -n "$(find "$OUTDIR_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: evaluation output already exists and is non-empty: $OUTDIR_ROOT"
  exit 2
fi
mkdir -p "$OUTDIR_ROOT/held-out" "$OUTDIR_ROOT/legacy"

for path in \
  "$HELDOUT_TEST_PT" "$HELDOUT_REFERENCE_PT" \
  "$HELDOUT_TEST_WEIGHTS" "$HELDOUT_REFERENCE_WEIGHTS" \
  "$LEGACY_TEST_PT"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required evaluation input is missing: $path"
    exit 2
  fi
done

HELDOUT_SIGNAL_ARGS=()
LEGACY_SIGNAL_ARGS=()
if [[ -s "$SIGNAL_PT" ]]; then
  HELDOUT_SIGNAL_ARGS=(--signal_pt "$SIGNAL_PT")
  LEGACY_SIGNAL_ARGS=(--signal_pt "$SIGNAL_PT")
  if [[ -n "$SIGNAL_WEIGHTS" && -s "$SIGNAL_WEIGHTS" ]]; then
    HELDOUT_SIGNAL_ARGS+=(--signal_weights "$SIGNAL_WEIGHTS")
  fi
fi

JOB_TOKEN="${SLURM_JOB_ID:-manual}"
COMMON_ARGS=(
  --ckpt "$CKPT"
  --ae_ckpt "$AE_CKPT"
  --n_pca "$N_PCA"
  --batch_size "$EVAL_BATCH_SIZE"
  --ae_batch_size "$AE_BATCH_SIZE"
  --dcor_sample_size "$DCOR_SAMPLE_SIZE"
  --baseline_labels "$BASELINE_LABELS"
  --abcd_scope qcd
  --score_mode qcd_md
  --min_A "$MIN_A"
  --min_D "$MIN_D"
  --scan_percent_min "$SCAN_PERCENT_MIN"
  --scan_percent_max "$SCAN_PERCENT_MAX"
  --scan_percent_steps "$SCAN_PERCENT_STEPS"
  --wandb_project nurd-ood-hlt
)

echo "=== Dual QCD evaluation started: $(date) ==="
echo "NURD_EXP: $NURD_EXP"
echo "CKPT: $CKPT"
echo "AE_CKPT: $AE_CKPT"
echo "CODE_COMMIT: $CURRENT_COMMIT"
echo "OUTPUT_ROOT: $OUTDIR_ROOT"

echo "=== Protocol 1/2: held-out weighted Mequinna report ==="
python eval_abcd_nurd.py \
  "${COMMON_ARGS[@]}" \
  --evaluation_protocol heldout \
  --test_pt "$HELDOUT_TEST_PT" \
  --test_weights "$HELDOUT_TEST_WEIGHTS" \
  --reference_pt "$HELDOUT_REFERENCE_PT" \
  --reference_weights "$HELDOUT_REFERENCE_WEIGHTS" \
  --outdir "$OUTDIR_ROOT/held-out" \
  --min_A_frac "$MIN_A_FRAC" \
  --min_region_frac "$MIN_REGION_FRAC" \
  --max_ratio_unc "$MAX_RATIO_UNC" \
  --selection_folds "$SELECTION_FOLDS" \
  --selection_stat_weight "$SELECTION_STAT_WEIGHT" \
  --selection_neighbor_weight "$SELECTION_NEIGHBOR_WEIGHT" \
  --selection_neighbor_radius "$SELECTION_NEIGHBOR_RADIUS" \
  --wandb_run_name "${EVAL_NAME}_held_out" \
  --resume_run_id "dual_${JOB_TOKEN}_held_out" \
  "${HELDOUT_SIGNAL_ARGS[@]}"

echo "=== Protocol 2/2: origin/main legacy QCD report ==="
python eval_abcd_nurd.py \
  "${COMMON_ARGS[@]}" \
  --evaluation_protocol legacy_main_qcd \
  --test_pt "$LEGACY_TEST_PT" \
  --outdir "$OUTDIR_ROOT/legacy" \
  --wandb_run_name "${EVAL_NAME}_legacy" \
  --resume_run_id "dual_${JOB_TOKEN}_legacy" \
  "${LEGACY_SIGNAL_ARGS[@]}"

python scripts/summarize_dual_eval.py \
  --root "$OUTDIR_ROOT" \
  --ckpt "$CKPT" \
  --ae-ckpt "$AE_CKPT" \
  --nurd-exp "$NURD_EXP" \
  --code-commit "$CURRENT_COMMIT"

echo "=== Evaluation outputs ==="
find "$OUTDIR_ROOT" -maxdepth 2 -type f | sort
echo "Primary result: $OUTDIR_ROOT/held-out"
echo "Legacy comparison: $OUTDIR_ROOT/legacy"
echo "Combined summary: $OUTDIR_ROOT/evaluation_summary.json"

if [[ "$SYNC_WANDB" == "1" ]]; then
  mapfile -t RUN_DIRS < <(find "$WANDB_DIR/wandb" -maxdepth 1 -type d \
    \( -name "offline-run-*-dual_${JOB_TOKEN}_held_out" \
       -o -name "offline-run-*-dual_${JOB_TOKEN}_legacy" \) | sort)
  for run_dir in "${RUN_DIRS[@]}"; do
    wandb sync "$run_dir" || echo "WARNING: wandb sync failed for $run_dir"
  done
fi

echo "DUAL QCD EVAL DONE"
echo "=== Dual QCD evaluation finished: $(date) ==="
