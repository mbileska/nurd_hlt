#!/bin/bash
#SBATCH --job-name=nurd_eval_latest
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=04:00:00
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
TEST_PT="${TEST_PT:-$BASE/data/hlt_smcocktail_test.pt}"
SIGNAL_PT="${SIGNAL_PT:-$BASE/data/hlt_signal_TpTp.pt}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-4096}"
N_PCA="${N_PCA:-6}"
CORR_SAMPLE_SIZE="${CORR_SAMPLE_SIZE:-50000}"
DCOR_SAMPLE_SIZE="${DCOR_SAMPLE_SIZE:-2000}"
CLOSURE_HOLDOUT_FRAC="${CLOSURE_HOLDOUT_FRAC:-0.5}"
CLOSURE_SPLIT_SEED="${CLOSURE_SPLIT_SEED:-42}"
NURD_GLOB="${NURD_GLOB:-hlt_nurd_closure_bs4096_*}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-optimized}"
SYNC_WANDB="${SYNC_WANDB:-0}"
PREFER_ABCD_CKPT="${PREFER_ABCD_CKPT:-0}"
PREFER_CLOSURE_CKPT="${PREFER_CLOSURE_CKPT:-0}"
SCAN_PERCENT_MIN="${SCAN_PERCENT_MIN:-0.50}"
SCAN_PERCENT_MAX="${SCAN_PERCENT_MAX:-0.98}"
SCAN_PERCENT_STEPS="${SCAN_PERCENT_STEPS:-48}"
CLOSURE_CURVE_PERCENT_MIN="${CLOSURE_CURVE_PERCENT_MIN:-$SCAN_PERCENT_MIN}"
CLOSURE_CURVE_PERCENT_MAX="${CLOSURE_CURVE_PERCENT_MAX:-$SCAN_PERCENT_MAX}"
CLOSURE_CURVE_PERCENT_STEPS="${CLOSURE_CURVE_PERCENT_STEPS:-$SCAN_PERCENT_STEPS}"
CLOSURE_CURVE_XMIN="${CLOSURE_CURVE_XMIN:-}"
CLOSURE_CURVE_XMAX="${CLOSURE_CURVE_XMAX:-}"

mkdir -p "$BASE/logs" "$BASE/wandb" "$BASE/matplotlib" "$BASE/outputs"

export WANDB_DIR="$BASE/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MPLCONFIGDIR="$BASE/matplotlib"

cd "$CODE_DIR"

if [[ -e checkpoints && ! -L checkpoints ]]; then
  echo "ERROR: $CODE_DIR/checkpoints exists and is not a symlink."
  echo "Move it or back it up, then link checkpoints to $BASE/checkpoints."
  exit 2
fi
ln -sfnT "$BASE/checkpoints" checkpoints

if [[ -z "${CKPT:-}" ]]; then
  if [[ "$PREFER_ABCD_CKPT" == "1" ]]; then
    CKPT="$(find "$BASE/checkpoints/hlt/hlt" -path "*/$NURD_GLOB/checkpoint_abcd.pth.tar" \
      -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
  fi
  if [[ -z "${CKPT:-}" && "$PREFER_CLOSURE_CKPT" == "1" ]]; then
    CKPT="$(find "$BASE/checkpoints/hlt/hlt" -path "*/$NURD_GLOB/checkpoint_closure.pth.tar" \
      -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
  fi
  if [[ -z "${CKPT:-}" ]]; then
    CKPT="$(find "$BASE/checkpoints/hlt/hlt" -path "*/$NURD_GLOB/checkpoint_main_*.pth.tar" \
      -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
  fi
fi
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  echo "ERROR: no NURD checkpoint found under $BASE/checkpoints/hlt/hlt matching $NURD_GLOB"
  exit 2
fi

NURD_EXP="${NURD_EXP:-$(basename "$(dirname "$CKPT")")}"
CKPT_FILE="$(basename "$CKPT")"
CKPT_TAG="${CKPT_FILE#checkpoint_main_}"
CKPT_TAG="${CKPT_TAG%.pth.tar}"

if [[ -z "${AE_CKPT:-}" ]]; then
  AE_CKPT="$(python - "$CKPT" "$BASE" "$CODE_DIR" <<'PY'
import os
import sys
import torch

ckpt_path, base, code_dir = sys.argv[1:4]
ckpt = torch.load(ckpt_path, map_location="cpu")
ae_ckpt = ckpt.get("config", {}).get("ae_ckpt", "")
if ae_ckpt and not os.path.isabs(ae_ckpt):
    ae_ckpt = os.path.normpath(os.path.join(code_dir, ae_ckpt))
if ae_ckpt and not os.path.exists(ae_ckpt) and "/checkpoints/" in ae_ckpt:
    rel = ae_ckpt.split("/checkpoints/", 1)[1]
    ae_ckpt = os.path.join(base, "checkpoints", rel)
print(ae_ckpt)
PY
)"
fi

if [[ -z "${AE_CKPT:-}" || ! -f "$AE_CKPT" ]]; then
  AE_DIR="$(ls -td "$BASE"/checkpoints/hlt/hlt/ae_pretrain_* 2>/dev/null | head -1)"
  AE_CKPT="$AE_DIR/checkpoint_ae.pth"
fi
if [[ ! -f "$AE_CKPT" ]]; then
  echo "ERROR: AE checkpoint not found. Set AE_CKPT explicitly."
  echo "Resolved AE_CKPT=$AE_CKPT"
  exit 2
fi

SAFE_PREFIX="$(echo "$WANDB_NAME_PREFIX" | tr -c 'A-Za-z0-9_-' '_')"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${SAFE_PREFIX}_${NURD_EXP}_ckpt_${CKPT_TAG}_abcd}"
WANDB_RUN_ID="${WANDB_RUN_ID:-eval_${CKPT_TAG}_${RUN_STAMP}}"
OUTDIR="${OUTDIR:-$BASE/outputs/abcd_${NURD_EXP}_ckpt_${CKPT_TAG}_${SAFE_PREFIX}}"

if [[ ! -f "$TEST_PT" ]]; then
  echo "ERROR: test sample not found: $TEST_PT"
  exit 2
fi

SIGNAL_ARGS=()
if [[ -n "${SIGNAL_PT:-}" && -f "$SIGNAL_PT" ]]; then
  SIGNAL_ARGS=(--signal_pt "$SIGNAL_PT")
fi

EXTRA_ARGS=()
if [[ "${MIN_MD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--min_md)
fi
if [[ "${SKIP_PCA_MD_PLOTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_pca_md_plots)
fi
if [[ "${SKIP_EMBEDDING_PCA:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_embedding_pca)
fi
if [[ -n "$CLOSURE_CURVE_XMIN" ]]; then
  EXTRA_ARGS+=(--closure_curve_xmin "$CLOSURE_CURVE_XMIN")
fi
if [[ -n "$CLOSURE_CURVE_XMAX" ]]; then
  EXTRA_ARGS+=(--closure_curve_xmax "$CLOSURE_CURVE_XMAX")
fi

echo "=== Latest ABCD eval started: $(date) ==="
echo "Host: $(hostname)"
echo "Code: $CODE_DIR"
echo "Results: $OUTDIR"
echo "AE_CKPT: $AE_CKPT"
echo "CKPT: $CKPT"
echo "NURD_EXP: $NURD_EXP"
echo "CKPT_TAG: $CKPT_TAG"
echo "TEST_PT: $TEST_PT"
echo "SIGNAL_PT: ${SIGNAL_PT:-}"
echo "EVAL_BATCH_SIZE: $EVAL_BATCH_SIZE"
echo "AE_BATCH_SIZE: $AE_BATCH_SIZE"
echo "N_PCA: $N_PCA"
echo "CORR_SAMPLE_SIZE: $CORR_SAMPLE_SIZE"
echo "DCOR_SAMPLE_SIZE: $DCOR_SAMPLE_SIZE"
echo "CLOSURE_HOLDOUT_FRAC: $CLOSURE_HOLDOUT_FRAC"
echo "CLOSURE_SPLIT_SEED: $CLOSURE_SPLIT_SEED"
echo "SCAN_PERCENT_MIN: $SCAN_PERCENT_MIN"
echo "SCAN_PERCENT_MAX: $SCAN_PERCENT_MAX"
echo "SCAN_PERCENT_STEPS: $SCAN_PERCENT_STEPS"
echo "CLOSURE_CURVE_PERCENT_MIN: $CLOSURE_CURVE_PERCENT_MIN"
echo "CLOSURE_CURVE_PERCENT_MAX: $CLOSURE_CURVE_PERCENT_MAX"
echo "CLOSURE_CURVE_PERCENT_STEPS: $CLOSURE_CURVE_PERCENT_STEPS"
echo "CLOSURE_CURVE_XMIN: $CLOSURE_CURVE_XMIN"
echo "CLOSURE_CURVE_XMAX: $CLOSURE_CURVE_XMAX"
echo "NURD_GLOB: $NURD_GLOB"
echo "PREFER_ABCD_CKPT: $PREFER_ABCD_CKPT"
echo "PREFER_CLOSURE_CKPT: $PREFER_CLOSURE_CKPT"
echo "WANDB_RUN_NAME: $WANDB_RUN_NAME"
echo "WANDB_RUN_ID: $WANDB_RUN_ID"
echo "SYNC_WANDB: $SYNC_WANDB"
echo "W&B mode: $WANDB_MODE"

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import numpy, sklearn, scipy, matplotlib, wandb; print('eval imports ok')"

python eval_abcd_nurd.py \
  --ckpt "$CKPT" \
  --ae_ckpt "$AE_CKPT" \
  --test_pt "$TEST_PT" \
  "${SIGNAL_ARGS[@]}" \
  --outdir "$OUTDIR" \
  --wandb_project nurd-ood-hlt \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --resume_run_id "$WANDB_RUN_ID" \
  --n_pca "$N_PCA" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --ae_batch_size "$AE_BATCH_SIZE" \
  --corr_sample_size "$CORR_SAMPLE_SIZE" \
  --dcor_sample_size "$DCOR_SAMPLE_SIZE" \
  --scan_percent_min "$SCAN_PERCENT_MIN" \
  --scan_percent_max "$SCAN_PERCENT_MAX" \
  --scan_percent_steps "$SCAN_PERCENT_STEPS" \
  --closure_curve_percent_min "$CLOSURE_CURVE_PERCENT_MIN" \
  --closure_curve_percent_max "$CLOSURE_CURVE_PERCENT_MAX" \
  --closure_curve_percent_steps "$CLOSURE_CURVE_PERCENT_STEPS" \
  --closure_holdout_frac "$CLOSURE_HOLDOUT_FRAC" \
  --closure_split_seed "$CLOSURE_SPLIT_SEED" \
  "${EXTRA_ARGS[@]}"

RUN_DIR="$(find "$WANDB_DIR/wandb" -maxdepth 1 -type d -name "offline-run-*-$WANDB_RUN_ID" | sort | tail -1)"

echo "=== ABCD outputs ==="
find "$OUTDIR" -maxdepth 2 -type f | sort
echo "W&B offline run: $RUN_DIR"
if [[ -n "$RUN_DIR" ]]; then
  echo "W&B sync command: wandb sync $RUN_DIR"
fi

if [[ "$SYNC_WANDB" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.secrets/wandb_api_key" ]]; then
    export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_api_key")"
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARNING: SYNC_WANDB=1 but WANDB_API_KEY is unset and $HOME/.secrets/wandb_api_key is missing."
  elif [[ -z "$RUN_DIR" ]]; then
    echo "WARNING: SYNC_WANDB=1 but no offline run directory was found for $WANDB_RUN_ID."
  else
    wandb sync "$RUN_DIR" || echo "WARNING: wandb sync failed; rerun the printed sync command from a login node."
  fi
fi

echo "ABCD EVAL DONE"
echo "=== Latest ABCD eval finished: $(date) ==="
