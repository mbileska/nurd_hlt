#!/bin/bash
#SBATCH --job-name=nurd_engineer_train
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=della-i12g[1-2],della-i14g[1-20]

set -euo pipefail

module load anaconda3/2025.12
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
TRAIN_PT="${TRAIN_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt}"
GEN_WEIGHT_TRAIN="${GEN_WEIGHT_TRAIN:-$MEQUINNA_DIR/weight_train.pt}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be supplied by the launcher}"
CODE_COMMIT="${CODE_COMMIT:?CODE_COMMIT must be supplied by the launcher}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:?CONTRAST_WEIGHT must be supplied by the launcher}"
NURD_EPOCHS="${NURD_EPOCHS:-40}"
LR_SCHEDULE_EPOCHS="${LR_SCHEDULE_EPOCHS:-$NURD_EPOCHS}"
INFO_WARMUP_EPOCHS="${INFO_WARMUP_EPOCHS:-0}"
INFO_RAMP_EPOCHS="${INFO_RAMP_EPOCHS:-0}"
CRITIC_SHUFFLE_MODE="${CRITIC_SHUFFLE_MODE:-weighted_within_class}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
AE_EXP="${AE_EXP:-ae_engineer_$RUN_TAG}"
NURD_EXP="${NURD_EXP:-hlt_nurd_engineer_$RUN_TAG}"
AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"

mkdir -p "$BASE/checkpoints" "$BASE/logs" "$BASE/outputs" "$BASE/wandb"
export WANDB_DIR="$BASE/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CODE_DIR"
if [[ "$(git rev-parse HEAD)" != "$CODE_COMMIT" ]]; then
  echo "ERROR: checked-out commit differs from the submitted campaign."
  echo "Expected: $CODE_COMMIT"
  echo "Current:  $(git rev-parse HEAD)"
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source files have uncommitted changes."
  exit 2
fi
if [[ -e checkpoints && ! -L checkpoints ]]; then
  echo "ERROR: $CODE_DIR/checkpoints exists and is not a symlink."
  exit 2
fi
ln -sfnT "$BASE/checkpoints" checkpoints

for input in "$TRAIN_PT" "$GEN_WEIGHT_TRAIN"; do
  if [[ ! -s "$input" ]]; then
    echo "ERROR: required input is missing: $input"
    exit 2
  fi
done

echo "=== Corrected continuous-NURD campaign ==="
echo "Run tag: $RUN_TAG"
echo "Commit: $CODE_COMMIT"
echo "Data: $TRAIN_PT"
echo "Weights: $GEN_WEIGHT_TRAIN"
echo "AE experiment: $AE_EXP"
echo "NURD experiment: $NURD_EXP"
echo "SupCon weight: $CONTRAST_WEIGHT"
echo "NURD maximum epochs: $NURD_EPOCHS"
echo "NURD LR schedule epochs: $LR_SCHEDULE_EPOCHS"
echo "Information warm-up/ramp: $INFO_WARMUP_EPOCHS/$INFO_RAMP_EPOCHS"
echo "Critic shuffle: $CRITIC_SHUFFLE_MODE"
echo "Checkpoint interval: $CHECKPOINT_EVERY"

python train_ae.py \
  --data "$TRAIN_PT" \
  --gen_weight_path "$GEN_WEIGHT_TRAIN" \
  --qcd_label 1 \
  --epochs 100 \
  --batch_size 4096 \
  --lr 1e-3 \
  --latent_dim 16 \
  --enc_nodes 512 256 \
  --dec_nodes 256 512 \
  --exp_name "$AE_EXP" \
  --project_name hlt \
  --manualSeed 42

if [[ ! -s "$AE_CKPT" ]]; then
  echo "ERROR: AE checkpoint was not produced: $AE_CKPT"
  exit 2
fi

python train_hlt.py \
  --data "$TRAIN_PT" \
  --gen_weight_path "$GEN_WEIGHT_TRAIN" \
  --ae_ckpt "$AE_CKPT" \
  --qcd_label 1 \
  --balance_strata 20 \
  --epochs "$NURD_EPOCHS" \
  --lr_schedule_epochs "$LR_SCHEDULE_EPOCHS" \
  --batch_size 4096 \
  --num_workers 0 \
  --lr 1e-4 \
  --critic_lr 1e-3 \
  --critic_steps 3 \
  --critic_shuffle_mode "$CRITIC_SHUFFLE_MODE" \
  --lambda_info 1.0 \
  --info_warmup_epochs "$INFO_WARMUP_EPOCHS" \
  --info_ramp_epochs "$INFO_RAMP_EPOCHS" \
  --contrast_weight "$CONTRAST_WEIGHT" \
  --weight_decay 5e-3 \
  --embed_size 128 \
  --latent_dim 6 \
  --proj_dim 6 \
  --num_heads 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --linear_dim 16 \
  --patience 15 \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  --exp_name "$NURD_EXP" \
  --project_name hlt \
  --manualSeed 42 \
  --code_commit "$CODE_COMMIT"

NURD_CKPT="$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_main.pth.tar"
if [[ ! -s "$NURD_CKPT" ]]; then
  echo "ERROR: NURD checkpoint was not produced: $NURD_CKPT"
  exit 2
fi
echo "TRAINING DONE"
echo "AE_CKPT=$AE_CKPT"
echo "NURD_CKPT=$NURD_CKPT"
