#!/bin/bash
#SBATCH --job-name=nurd_diagnostic
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=della-i12g[1-2],della-i14g[1-20]

export PS1="${PS1-}"
set -euo pipefail

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
TRAIN_PT="${TRAIN_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt}"
GEN_WEIGHT_TRAIN="${GEN_WEIGHT_TRAIN:-$MEQUINNA_DIR/weight_train.pt}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be supplied by the launcher}"
CODE_COMMIT="${CODE_COMMIT:?CODE_COMMIT must be supplied by the launcher}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-0.3}"
DIAG_MAX_EVENTS="${DIAG_MAX_EVENTS:-120000}"
DIAG_AE_EPOCHS="${DIAG_AE_EPOCHS:-5}"
DIAG_SUPCON_EPOCHS="${DIAG_SUPCON_EPOCHS:-5}"
DIAG_NURD_EPOCHS="${DIAG_NURD_EPOCHS:-3}"
DIAG_CRITIC_PROBE_EPOCHS="${DIAG_CRITIC_PROBE_EPOCHS:-3}"
DIAG_BATCH_SIZE="${DIAG_BATCH_SIZE:-4096}"
BALANCE_BINNING="${BALANCE_BINNING:-log_fixed}"
BALANCE_CLIP_QUANTILE="${BALANCE_CLIP_QUANTILE:-0.995}"
BALANCE_STRATA="${BALANCE_STRATA:-20}"
TOTAL_HLT_EPOCHS=$((DIAG_SUPCON_EPOCHS + DIAG_NURD_EPOCHS))
CRITIC_START_EPOCH=$((DIAG_SUPCON_EPOCHS + 1))

DIAG_ROOT="$BASE/outputs/${RUN_TAG}_diagnostic"
INPUT_DIR="$DIAG_ROOT/inputs"
SUBSET_PT="$INPUT_DIR/hlt_diagnostic_train.pt"
SUBSET_WEIGHTS="$INPUT_DIR/weight_diagnostic_train.pt"
SUBSET_METADATA="$INPUT_DIR/subset_metadata.json"
AE_EXP="ae_diagnostic_$RUN_TAG"
NURD_EXP="hlt_diagnostic_$RUN_TAG"
AE_DIR="$BASE/checkpoints/hlt/hlt/$AE_EXP"
NURD_DIR="$BASE/checkpoints/hlt/hlt/$NURD_EXP"
AE_CKPT="$AE_DIR/checkpoint_ae.pth"

mkdir -p "$BASE/checkpoints" "$BASE/logs" "$BASE/outputs" "$INPUT_DIR"
cd "$CODE_DIR"
if [[ "$(git rev-parse HEAD)" != "$CODE_COMMIT" ]]; then
  echo "ERROR: checked-out commit differs from the submitted diagnostic."
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source files changed after diagnostic submission."
  exit 2
fi
if [[ -e checkpoints && ! -L checkpoints ]]; then
  echo "ERROR: $CODE_DIR/checkpoints exists and is not a symlink."
  exit 2
fi
ln -sfnT "$BASE/checkpoints" checkpoints

echo "=== HLT staged training diagnostic ==="
echo "Run tag: $RUN_TAG"
echo "Commit: $CODE_COMMIT"
echo "Subset events: $DIAG_MAX_EVENTS"
echo "AE epochs: $DIAG_AE_EPOCHS"
echo "SupCon/classifier-only epochs: $DIAG_SUPCON_EPOCHS"
echo "Short NURD epochs: $DIAG_NURD_EPOCHS"
echo "Frozen critic probe epochs: $DIAG_CRITIC_PROBE_EPOCHS"
echo "Output: $DIAG_ROOT"

python scripts/create_diagnostic_subset.py \
  --data "$TRAIN_PT" \
  --weights "$GEN_WEIGHT_TRAIN" \
  --output-data "$SUBSET_PT" \
  --output-weights "$SUBSET_WEIGHTS" \
  --metadata "$SUBSET_METADATA" \
  --max-events "$DIAG_MAX_EVENTS" \
  --seed 42

python train_ae.py \
  --data "$SUBSET_PT" \
  --gen_weight_path "$SUBSET_WEIGHTS" \
  --generator_weight_label 1 \
  --epochs "$DIAG_AE_EPOCHS" \
  --patience "$DIAG_AE_EPOCHS" \
  --batch_size "$DIAG_BATCH_SIZE" \
  --lr 1e-3 \
  --latent_dim 16 \
  --enc_nodes 512 256 \
  --dec_nodes 256 512 \
  --exp_name "$AE_EXP" \
  --project_name hlt \
  --local_testing 1 \
  --manualSeed 42

python train_hlt.py \
  --data "$SUBSET_PT" \
  --gen_weight_path "$SUBSET_WEIGHTS" \
  --ae_ckpt "$AE_CKPT" \
  --qcd_label 1 \
  --balance_strata "$BALANCE_STRATA" \
  --balance_binning "$BALANCE_BINNING" \
  --balance_clip_quantile "$BALANCE_CLIP_QUANTILE" \
  --epochs "$TOTAL_HLT_EPOCHS" \
  --lr_schedule_epochs "$TOTAL_HLT_EPOCHS" \
  --batch_size "$DIAG_BATCH_SIZE" \
  --num_workers 0 \
  --lr 1e-4 \
  --critic_lr 1e-3 \
  --critic_steps 3 \
  --nurd_enabled 1 \
  --critic_start_epoch "$CRITIC_START_EPOCH" \
  --critic_shuffle_mode global \
  --lambda_info 1.0 \
  --contrast_weight "$CONTRAST_WEIGHT" \
  --weight_decay 5e-3 \
  --embed_size 128 \
  --latent_dim 6 \
  --proj_dim 6 \
  --num_heads 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --linear_dim 16 \
  --patience "$TOTAL_HLT_EPOCHS" \
  --checkpoint_every 1 \
  --exp_name "$NURD_EXP" \
  --project_name hlt \
  --local_testing 1 \
  --manualSeed 42 \
  --code_commit "$CODE_COMMIT"

python scripts/run_training_diagnostics.py \
  --data "$SUBSET_PT" \
  --gen-weight-path "$SUBSET_WEIGHTS" \
  --ae-ckpt "$AE_CKPT" \
  --checkpoint-dir "$NURD_DIR" \
  --baseline-epoch "$DIAG_SUPCON_EPOCHS" \
  --nurd-epoch "$TOTAL_HLT_EPOCHS" \
  --outdir "$DIAG_ROOT" \
  --qcd-label 1 \
  --balance-strata "$BALANCE_STRATA" \
  --balance-binning "$BALANCE_BINNING" \
  --balance-clip-quantile "$BALANCE_CLIP_QUANTILE" \
  --batch-size 1024 \
  --ae-batch-size "$DIAG_BATCH_SIZE" \
  --critic-probe-epochs "$DIAG_CRITIC_PROBE_EPOCHS" \
  --seed 42

python - "$DIAG_ROOT" "$CODE_COMMIT" "$RUN_TAG" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {
    "run_tag": sys.argv[3],
    "code_commit": sys.argv[2],
    "status": "complete",
    "summary": str(root / "diagnostic_summary.json"),
    "plots": str(root / "plots"),
}
(root / "campaign_manifest.json").write_text(json.dumps(payload, indent=2))
PY

echo "DIAGNOSTIC DONE"
echo "SUMMARY=$DIAG_ROOT/diagnostic_summary.json"
echo "PLOTS=$DIAG_ROOT/plots"
echo "Optional W&B upload:"
echo "  python scripts/upload_training_diagnostics.py $DIAG_ROOT"
