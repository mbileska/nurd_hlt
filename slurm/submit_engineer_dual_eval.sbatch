#!/bin/bash
#SBATCH --job-name=nurd_engineer_eval
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

export PS1="${PS1-}"
set -euo pipefail

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be supplied by the launcher}"
CODE_COMMIT="${CODE_COMMIT:?CODE_COMMIT must be supplied by the launcher}"
AE_EXP="${AE_EXP:-ae_engineer_$RUN_TAG}"
NURD_EXP="${NURD_EXP:-hlt_nurd_engineer_$RUN_TAG}"
AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
NURD_CKPT="$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_main.pth.tar"
HELDOUT_PT="${HELDOUT_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt}"
HELDOUT_WEIGHTS="${HELDOUT_WEIGHTS:-$MEQUINNA_DIR/weight_test.pt}"
REFERENCE_PT="${REFERENCE_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt}"
REFERENCE_WEIGHTS="${REFERENCE_WEIGHTS:-$MEQUINNA_DIR/weight_train.pt}"
LEGACY_PT="${LEGACY_PT:-$BASE/data/hlt_smcocktail_test.pt}"
SIGNAL_PT="${SIGNAL_PT:-$BASE/data/hlt_signal_TpTp.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BASE/outputs/${RUN_TAG}_eval}"
STATISTICALLY_VALID_CLOSURE="${STATISTICALLY_VALID_CLOSURE:-0}"
TARGET_SCHEME="${TARGET_SCHEME:-parent4}"

mkdir -p "$BASE/logs" "$BASE/outputs" "$BASE/wandb" "$OUTPUT_ROOT/held-out"
if [[ "$TARGET_SCHEME" == "parent4" ]]; then
  mkdir -p "$OUTPUT_ROOT/legacy"
fi
export WANDB_DIR="$BASE/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MPLCONFIGDIR="$BASE/matplotlib"
mkdir -p "$MPLCONFIGDIR"

cd "$CODE_DIR"
if [[ "$(git rev-parse HEAD)" != "$CODE_COMMIT" ]]; then
  echo "ERROR: code changed after campaign submission."
  exit 2
fi
required_inputs=(
  "$AE_CKPT" "$NURD_CKPT" "$HELDOUT_PT" "$HELDOUT_WEIGHTS"
  "$REFERENCE_PT" "$REFERENCE_WEIGHTS"
)
if [[ "$TARGET_SCHEME" == "parent4" ]]; then
  required_inputs+=("$LEGACY_PT")
fi
for input in "${required_inputs[@]}"; do
  if [[ ! -s "$input" ]]; then
    echo "ERROR: required evaluation input is missing: $input"
    exit 2
  fi
done

SIGNAL_ARGS=()
if [[ -s "$SIGNAL_PT" ]]; then
  SIGNAL_ARGS=(--signal_pt "$SIGNAL_PT")
fi

STATISTICS_ARGS=()
WANDB_EVAL_SUFFIX=""
if [[ "$STATISTICALLY_VALID_CLOSURE" == "1" ]]; then
  STATISTICS_ARGS=(--statistically_valid_closure)
  WANDB_EVAL_SUFFIX="_stat_valid"
fi

python eval_abcd_nurd.py \
  --ckpt "$NURD_CKPT" \
  --ae_ckpt "$AE_CKPT" \
  --test_pt "$HELDOUT_PT" \
  --gen_weight_path "$HELDOUT_WEIGHTS" \
  --reference_pt "$REFERENCE_PT" \
  --reference_weight_path "$REFERENCE_WEIGHTS" \
  --outdir "$OUTPUT_ROOT/held-out" \
  --n_pca 6 \
  --qcd_label 1 \
  --target_scheme "$TARGET_SCHEME" \
  --wandb_run_name "${RUN_TAG}_held_out${WANDB_EVAL_SUFFIX}" \
  "${STATISTICS_ARGS[@]}" \
  "${SIGNAL_ARGS[@]}"

SUMMARY_ARGS=()
if [[ "$TARGET_SCHEME" == "parent4" ]]; then
  python eval_abcd_nurd.py \
    --ckpt "$NURD_CKPT" \
    --ae_ckpt "$AE_CKPT" \
    --test_pt "$LEGACY_PT" \
    --outdir "$OUTPUT_ROOT/legacy" \
    --n_pca 6 \
    --qcd_label 1 \
    --target_scheme "$TARGET_SCHEME" \
    --wandb_run_name "${RUN_TAG}_legacy${WANDB_EVAL_SUFFIX}" \
    "${STATISTICS_ARGS[@]}" \
    "${SIGNAL_ARGS[@]}"
else
  echo "Skipping legacy evaluation: legacy tensor has no background_category."
  SUMMARY_ARGS=(--skip-legacy)
fi

python scripts/summarize_dual_eval.py \
  --root "$OUTPUT_ROOT" \
  --ckpt "$NURD_CKPT" \
  --ae-ckpt "$AE_CKPT" \
  --code-commit "$CODE_COMMIT" \
  "${SUMMARY_ARGS[@]}"

echo "DUAL EVALUATION DONE"
echo "Held-out: $OUTPUT_ROOT/held-out"
echo "Legacy:   $OUTPUT_ROOT/legacy"
echo "Summary:  $OUTPUT_ROOT/evaluation_summary.json"
