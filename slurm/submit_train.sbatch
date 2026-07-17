#!/bin/bash
#SBATCH --job-name=nurd_hlt_train
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -eo pipefail

module load anaconda3/2025.12
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
TRAIN_PT="${TRAIN_PT:-$BASE/data/hlt_smcocktail_train.pt}"

mkdir -p "$BASE/checkpoints" "$BASE/logs" "$BASE/outputs" "$BASE/wandb"

export WANDB_DIR="$BASE/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$WANDB_MODE" != "offline" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -f "$HOME/.secrets/wandb_api_key" ]]; then
    export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_api_key")"
  else
    echo "ERROR: WANDB_API_KEY is unset and $HOME/.secrets/wandb_api_key does not exist."
    exit 2
  fi
fi

cd "$CODE_DIR"

if [[ -e checkpoints && ! -L checkpoints ]]; then
  echo "ERROR: $CODE_DIR/checkpoints exists and is not a symlink."
  echo "Move it or back it up, then link checkpoints to $BASE/checkpoints."
  exit 2
fi
ln -sfnT "$BASE/checkpoints" checkpoints

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
AE_EXP="${AE_EXP:-ae_pretrain_$RUN_TAG}"
NURD_EXP="${NURD_EXP:-hlt_nurd_closure_bs4096_$RUN_TAG}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
AE_CKPT="${AE_CKPT:-checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth}"

echo "=== Full training started: $(date) ==="
echo "Host: $(hostname)"
echo "Code: $CODE_DIR"
echo "Scratch: $BASE"
echo "Train data: $TRAIN_PT"
echo "AE_EXP=$AE_EXP"
echo "NURD_EXP=$NURD_EXP"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "AE_CKPT=$AE_CKPT"
echo "W&B project: nurd-ood-hlt"
echo "W&B mode: $WANDB_MODE"

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import numpy, sklearn, scipy, matplotlib, wandb; print('training/eval imports ok')"
python -c "import torch; x=torch.load('$TRAIN_PT',map_location='cpu'); print('keys', x.keys()); print('pf', x['pf'].shape); print('obj', x['obj'].shape); print('label', x['label'].shape)"

if [[ "${SKIP_AE:-0}" == "1" ]]; then
  if [[ ! -f "$AE_CKPT" ]]; then
    echo "ERROR: SKIP_AE=1 but AE_CKPT does not exist: $AE_CKPT"
    exit 2
  fi
  echo "=== Step 1: AE pretraining skipped; using $AE_CKPT ==="
else
  echo "=== Step 1: AE pretraining ==="
  python train_ae.py \
    --data "$TRAIN_PT" \
    --epochs 100 \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-3 \
    --latent_dim 16 \
    --enc_nodes 512 256 \
    --dec_nodes 256 512 \
    --exp_name "$AE_EXP" \
    --project_name hlt
fi

echo "=== Step 2: NURD closure training ==="
python train_hlt.py \
  --data "$TRAIN_PT" \
  --ae_ckpt "$AE_CKPT" \
  --epochs 100 \
  --batch_size "$BATCH_SIZE" \
  --lr 1e-4 \
  --weight_decay 5e-3 \
  --n_bins 10 \
  --reweight 1 \
  --joint_indep 1 \
  --contrast_weight 0.2 \
  --contrast_temp 0.05 \
  --embed_size 128 \
  --latent_dim 6 \
  --proj_dim 6 \
  --num_heads 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --linear_dim 16 \
  --critic_schedule warmup \
  --critic_warmup_epochs 7 \
  --critic_ramp_epochs 10 \
  --critic_train_frac 0.2 \
  --critic_lr_multiplier 10.0 \
  --n_critic_steps_per_batch 3 \
  --_lambda 0.01 \
  --no_mi_norm \
  --offload_critic_graph \
  --closure_weight 0.1 \
  --qcd_label 1 \
  --exp_name "$NURD_EXP" \
  --project_name hlt \
  --manualSeed 42

echo "=== Training outputs ==="
ls -lh "checkpoints/hlt/hlt/$AE_EXP/"
ls -lh "checkpoints/hlt/hlt/$NURD_EXP/"
echo "TRAINING DONE"
echo "=== Full training finished: $(date) ==="
