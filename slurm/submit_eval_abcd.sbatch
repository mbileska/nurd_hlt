#!/bin/bash
#SBATCH --job-name=nurd_eval
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
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
TEST_PT="${TEST_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt}"
REFERENCE_PT="${REFERENCE_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt}"
TEST_WEIGHTS="${TEST_WEIGHTS:-$MEQUINNA_DIR/weight_test.pt}"
REFERENCE_WEIGHTS="${REFERENCE_WEIGHTS:-$MEQUINNA_DIR/weight_train.pt}"
SIGNAL_WEIGHTS="${SIGNAL_WEIGHTS:-}"
SIGNAL_PT="${SIGNAL_PT:-$BASE/data/hlt_signal_TpTp.pt}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-4096}"
N_PCA="${N_PCA:-6}"
CORR_SAMPLE_SIZE="${CORR_SAMPLE_SIZE:-50000}"
DCOR_SAMPLE_SIZE="${DCOR_SAMPLE_SIZE:-2000}"
CLOSURE_HOLDOUT_FRAC="${CLOSURE_HOLDOUT_FRAC:-0.5}"
CLOSURE_SPLIT_SEED="${CLOSURE_SPLIT_SEED:-42}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
PREFER_ABCD_CKPT="${PREFER_ABCD_CKPT:-1}"
PREFER_CLOSURE_CKPT="${PREFER_CLOSURE_CKPT:-0}"
BASELINE_LABELS="${BASELINE_LABELS:-0,1,2,3}"
ABCD_SCOPE="${ABCD_SCOPE:-qcd}"
SCORE_MODE="${SCORE_MODE:-qcd_md}"
MIN_MD="${MIN_MD:-0}"
MIN_A="${MIN_A:-50}"
MIN_A_FRAC="${MIN_A_FRAC:-0.10}"
MIN_D="${MIN_D:-500}"
MIN_REGION_FRAC="${MIN_REGION_FRAC:-0.01}"
MAX_RATIO_UNC="${MAX_RATIO_UNC:-0.15}"
SELECTION_FOLDS="${SELECTION_FOLDS:-5}"
SELECTION_STAT_WEIGHT="${SELECTION_STAT_WEIGHT:-0.5}"
SELECTION_NEIGHBOR_WEIGHT="${SELECTION_NEIGHBOR_WEIGHT:-1.0}"
SELECTION_NEIGHBOR_RADIUS="${SELECTION_NEIGHBOR_RADIUS:-1}"
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

if [[ -z "${AE_CKPT:-}" ]]; then
  if [[ -n "${AE_EXP:-}" ]]; then
    AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
  else
    AE_DIR="$(ls -td "$BASE"/checkpoints/hlt/hlt/ae_pretrain_* 2>/dev/null | head -1)"
    AE_CKPT="$AE_DIR/checkpoint_ae.pth"
  fi
fi

if [[ -z "${CKPT:-}" ]]; then
  if [[ -n "${NURD_EXP:-}" ]]; then
    NURD_DIR="$BASE/checkpoints/hlt/hlt/$NURD_EXP"
  else
    NURD_DIR="$(ls -td "$BASE"/checkpoints/hlt/hlt/hlt_nurd_closure_bs4096_* 2>/dev/null | head -1)"
    NURD_EXP="$(basename "$NURD_DIR")"
  fi
  if [[ "$PREFER_ABCD_CKPT" == "1" && -f "$NURD_DIR/checkpoint_abcd.pth.tar" ]]; then
    CKPT="$NURD_DIR/checkpoint_abcd.pth.tar"
  elif [[ "$PREFER_CLOSURE_CKPT" == "1" && -f "$NURD_DIR/checkpoint_closure.pth.tar" ]]; then
    CKPT="$NURD_DIR/checkpoint_closure.pth.tar"
  else
    CKPT="$(ls -t "$NURD_DIR"/checkpoint_main_*.pth.tar 2>/dev/null | head -1)"
  fi
fi

if [[ -z "${NURD_EXP:-}" ]]; then
  NURD_EXP="$(basename "$(dirname "$CKPT")")"
fi

OUTDIR="${OUTDIR:-$BASE/outputs/abcd_${NURD_EXP}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${NURD_EXP}_abcd}"

if [[ ! -f "$AE_CKPT" ]]; then
  echo "ERROR: AE checkpoint not found: $AE_CKPT"
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: NURD checkpoint not found: $CKPT"
  exit 2
fi
if [[ ! -f "$TEST_PT" ]]; then
  echo "ERROR: test sample not found: $TEST_PT"
  exit 2
fi
if [[ ! -f "$REFERENCE_PT" ]]; then
  echo "ERROR: training reference sample not found: $REFERENCE_PT"
  exit 2
fi
if [[ ! -f "$TEST_WEIGHTS" || ! -f "$REFERENCE_WEIGHTS" ]]; then
  echo "ERROR: generator-weight files are missing."
  exit 2
fi

SIGNAL_ARGS=()
if [[ -n "${SIGNAL_PT:-}" && -f "$SIGNAL_PT" ]]; then
  SIGNAL_ARGS=(--signal_pt "$SIGNAL_PT")
  if [[ -n "$SIGNAL_WEIGHTS" && -f "$SIGNAL_WEIGHTS" ]]; then
    SIGNAL_ARGS+=(--signal_weights "$SIGNAL_WEIGHTS")
  fi
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
if [[ "${SKIP_NUISANCE_AUDITOR:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_nuisance_auditor)
fi
if [[ -n "$CLOSURE_CURVE_XMIN" ]]; then
  EXTRA_ARGS+=(--closure_curve_xmin "$CLOSURE_CURVE_XMIN")
fi
if [[ -n "$CLOSURE_CURVE_XMAX" ]]; then
  EXTRA_ARGS+=(--closure_curve_xmax "$CLOSURE_CURVE_XMAX")
fi

echo "=== ABCD eval started: $(date) ==="
echo "Host: $(hostname)"
echo "Code: $CODE_DIR"
echo "Results: $OUTDIR"
echo "AE_CKPT: $AE_CKPT"
echo "CKPT: $CKPT"
echo "TEST_PT: $TEST_PT"
echo "REFERENCE_PT: $REFERENCE_PT"
echo "TEST_WEIGHTS: $TEST_WEIGHTS"
echo "REFERENCE_WEIGHTS: $REFERENCE_WEIGHTS"
echo "SIGNAL_PT: ${SIGNAL_PT:-}"
echo "EVAL_BATCH_SIZE: $EVAL_BATCH_SIZE"
echo "AE_BATCH_SIZE: $AE_BATCH_SIZE"
echo "N_PCA: $N_PCA"
echo "CORR_SAMPLE_SIZE: $CORR_SAMPLE_SIZE"
echo "DCOR_SAMPLE_SIZE: $DCOR_SAMPLE_SIZE"
echo "CLOSURE_HOLDOUT_FRAC: $CLOSURE_HOLDOUT_FRAC"
echo "CLOSURE_SPLIT_SEED: $CLOSURE_SPLIT_SEED"
echo "BASELINE_LABELS: $BASELINE_LABELS"
echo "ABCD_SCOPE: $ABCD_SCOPE"
echo "SCORE_MODE: $SCORE_MODE"
echo "MIN_MD: $MIN_MD"
echo "MIN_A: $MIN_A"
echo "MIN_A_FRAC: $MIN_A_FRAC"
echo "MIN_D: $MIN_D"
echo "MIN_REGION_FRAC: $MIN_REGION_FRAC"
echo "MAX_RATIO_UNC: $MAX_RATIO_UNC"
echo "SELECTION_FOLDS: $SELECTION_FOLDS"
echo "SELECTION_STAT_WEIGHT: $SELECTION_STAT_WEIGHT"
echo "SELECTION_NEIGHBOR_WEIGHT: $SELECTION_NEIGHBOR_WEIGHT"
echo "SELECTION_NEIGHBOR_RADIUS: $SELECTION_NEIGHBOR_RADIUS"
echo "SCAN_PERCENT_MIN: $SCAN_PERCENT_MIN"
echo "SCAN_PERCENT_MAX: $SCAN_PERCENT_MAX"
echo "SCAN_PERCENT_STEPS: $SCAN_PERCENT_STEPS"
echo "CLOSURE_CURVE_PERCENT_MIN: $CLOSURE_CURVE_PERCENT_MIN"
echo "CLOSURE_CURVE_PERCENT_MAX: $CLOSURE_CURVE_PERCENT_MAX"
echo "CLOSURE_CURVE_PERCENT_STEPS: $CLOSURE_CURVE_PERCENT_STEPS"
echo "CLOSURE_CURVE_XMIN: $CLOSURE_CURVE_XMIN"
echo "CLOSURE_CURVE_XMAX: $CLOSURE_CURVE_XMAX"
echo "PREFER_ABCD_CKPT: $PREFER_ABCD_CKPT"
echo "PREFER_CLOSURE_CKPT: $PREFER_CLOSURE_CKPT"
echo "WANDB_RUN_NAME: $WANDB_RUN_NAME"
echo "WANDB_RUN_ID: ${WANDB_RUN_ID:-}"
echo "W&B mode: $WANDB_MODE"

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import numpy, sklearn, scipy, matplotlib, wandb; print('eval imports ok')"

RESUME_ARGS=()
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  RESUME_ARGS+=(--resume_run_id "$WANDB_RUN_ID")
fi

python eval_abcd_nurd.py \
  --ckpt "$CKPT" \
  --ae_ckpt "$AE_CKPT" \
  --test_pt "$TEST_PT" \
  --test_weights "$TEST_WEIGHTS" \
  --reference_pt "$REFERENCE_PT" \
  --reference_weights "$REFERENCE_WEIGHTS" \
  "${SIGNAL_ARGS[@]}" \
  --outdir "$OUTDIR" \
  --wandb_project nurd-ood-hlt \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --n_pca "$N_PCA" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --ae_batch_size "$AE_BATCH_SIZE" \
  --corr_sample_size "$CORR_SAMPLE_SIZE" \
  --dcor_sample_size "$DCOR_SAMPLE_SIZE" \
  --baseline_labels "$BASELINE_LABELS" \
  --abcd_scope "$ABCD_SCOPE" \
  --score_mode "$SCORE_MODE" \
  --min_A "$MIN_A" \
  --min_A_frac "$MIN_A_FRAC" \
  --min_D "$MIN_D" \
  --min_region_frac "$MIN_REGION_FRAC" \
  --max_ratio_unc "$MAX_RATIO_UNC" \
  --selection_folds "$SELECTION_FOLDS" \
  --selection_stat_weight "$SELECTION_STAT_WEIGHT" \
  --selection_neighbor_weight "$SELECTION_NEIGHBOR_WEIGHT" \
  --selection_neighbor_radius "$SELECTION_NEIGHBOR_RADIUS" \
  --scan_percent_min "$SCAN_PERCENT_MIN" \
  --scan_percent_max "$SCAN_PERCENT_MAX" \
  --scan_percent_steps "$SCAN_PERCENT_STEPS" \
  --closure_curve_percent_min "$CLOSURE_CURVE_PERCENT_MIN" \
  --closure_curve_percent_max "$CLOSURE_CURVE_PERCENT_MAX" \
  --closure_curve_percent_steps "$CLOSURE_CURVE_PERCENT_STEPS" \
  --closure_holdout_frac "$CLOSURE_HOLDOUT_FRAC" \
  --closure_split_seed "$CLOSURE_SPLIT_SEED" \
  "${RESUME_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

echo "=== ABCD outputs ==="
find "$OUTDIR" -maxdepth 2 -type f | sort
echo "ABCD EVAL DONE"
echo "=== ABCD eval finished: $(date) ==="
