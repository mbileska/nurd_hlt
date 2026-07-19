#!/bin/bash
#SBATCH --job-name=nurd_hlt_train
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=della-i12g[1-2],della-i14g[1-20]

set -eo pipefail

module load anaconda3/2025.12
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
set -u

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
TRAIN_PT="${TRAIN_PT:-$BASE/data/hlt_smcocktail_train.pt}"
MIN_GPU_MEM_GB="${MIN_GPU_MEM_GB:-75}"

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
AE_EPOCHS="${AE_EPOCHS:-100}"
NURD_EPOCHS="${NURD_EPOCHS:-85}"
CLOSURE_WEIGHT="${CLOSURE_WEIGHT:-0.5}"
CLOSURE_WEIGHT_START="${CLOSURE_WEIGHT_START:-0.0}"
CLOSURE_RAMP_EPOCHS="${CLOSURE_RAMP_EPOCHS:-15}"
CLOSURE_LOSS_TYPE="${CLOSURE_LOSS_TYPE:-dcorr_profile}"
NUISANCE_BIN_SCOPE="${NUISANCE_BIN_SCOPE:-qcd}"
CRITIC_SCOPE="${CRITIC_SCOPE:-qcd}"
CRITIC_TYPE="${CRITIC_TYPE:-density_ratio}"
CRITIC_PENALTY_TYPE="${CRITIC_PENALTY_TYPE:-confusion}"
N_CRITIC_STEPS="${N_CRITIC_STEPS:-2}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-0.05}"
CONTRAST_WEIGHT_START="${CONTRAST_WEIGHT_START:-0.15}"
CONTRAST_RAMP_EPOCHS="${CONTRAST_RAMP_EPOCHS:-40}"
MAX_WEIGHT_RATIO="${MAX_WEIGHT_RATIO:-10.0}"
MD_PROXY_TYPE="${MD_PROXY_TYPE:-ema}"
MD_EMA_MOMENTUM="${MD_EMA_MOMENTUM:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-0}"
AE_CKPT="${AE_CKPT:-checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth}"

echo "=== Full training started: $(date) ==="
echo "Host: $(hostname)"
echo "Code: $CODE_DIR"
echo "Scratch: $BASE"
echo "Train data: $TRAIN_PT"
echo "AE_EXP=$AE_EXP"
echo "NURD_EXP=$NURD_EXP"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "AE_EPOCHS=$AE_EPOCHS"
echo "NURD_EPOCHS=$NURD_EPOCHS"
echo "AE_CKPT=$AE_CKPT"
echo "CRITIC_SCOPE=$CRITIC_SCOPE"
echo "CRITIC_TYPE=$CRITIC_TYPE"
echo "CRITIC_PENALTY_TYPE=$CRITIC_PENALTY_TYPE"
echo "N_CRITIC_STEPS=$N_CRITIC_STEPS"
echo "NUISANCE_BIN_SCOPE=$NUISANCE_BIN_SCOPE"
echo "CONTRAST_WEIGHT=$CONTRAST_WEIGHT"
echo "CONTRAST_WEIGHT_START=$CONTRAST_WEIGHT_START"
echo "CONTRAST_RAMP_EPOCHS=$CONTRAST_RAMP_EPOCHS"
echo "CLOSURE_WEIGHT=$CLOSURE_WEIGHT"
echo "CLOSURE_WEIGHT_START=$CLOSURE_WEIGHT_START"
echo "CLOSURE_RAMP_EPOCHS=$CLOSURE_RAMP_EPOCHS"
echo "CLOSURE_LOSS_TYPE=$CLOSURE_LOSS_TYPE"
echo "MAX_WEIGHT_RATIO=$MAX_WEIGHT_RATIO"
echo "MD_PROXY_TYPE=$MD_PROXY_TYPE"
echo "MD_EMA_MOMENTUM=$MD_EMA_MOMENTUM"
echo "W&B project: nurd-ood-hlt"
echo "W&B mode: $WANDB_MODE"
echo "Minimum GPU memory: ${MIN_GPU_MEM_GB} GiB"

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import torch, sys; assert torch.cuda.is_available(), 'CUDA is not available'; props=torch.cuda.get_device_properties(0); mem_gib=props.total_memory/1024**3; print(f'GPU memory: {mem_gib:.2f} GiB'); sys.exit(0 if mem_gib >= float('$MIN_GPU_MEM_GB') else 42)"
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
    --epochs "$AE_EPOCHS" \
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
  --epochs "$NURD_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr 1e-4 \
  --weight_decay 5e-3 \
  --n_bins 10 \
  --nuisance_bin_scope "$NUISANCE_BIN_SCOPE" \
  --reweight 1 \
  --joint_indep 1 \
  --max_weight_ratio "$MAX_WEIGHT_RATIO" \
  --contrast_weight "$CONTRAST_WEIGHT" \
  --contrast_weight_start "$CONTRAST_WEIGHT_START" \
  --contrast_ramp_epochs "$CONTRAST_RAMP_EPOCHS" \
  --contrast_temp 0.05 \
  --embed_size 128 \
  --latent_dim 6 \
  --proj_dim 6 \
  --num_heads 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --linear_dim 16 \
  --critic_schedule warmup \
  --critic_type "$CRITIC_TYPE" \
  --critic_penalty_type "$CRITIC_PENALTY_TYPE" \
  --critic_warmup_epochs 7 \
  --critic_ramp_epochs 10 \
  --critic_train_frac 0.2 \
  --critic_lr_multiplier 10.0 \
  --n_critic_steps_per_batch "$N_CRITIC_STEPS" \
  --critic_scope "$CRITIC_SCOPE" \
  --_lambda 0.01 \
  --no_mi_norm \
  --closure_weight "$CLOSURE_WEIGHT" \
  --closure_weight_start "$CLOSURE_WEIGHT_START" \
  --closure_ramp_epochs "$CLOSURE_RAMP_EPOCHS" \
  --closure_loss_type "$CLOSURE_LOSS_TYPE" \
  --closure_corr_weight 0.25 \
  --closure_dcorr_weight 1.0 \
  --closure_profile_weight 0.5 \
  --closure_profile_bins 8 \
  --closure_profile_tail_weight 2.0 \
  --closure_dcorr_max_samples 512 \
  --md_proxy_type "$MD_PROXY_TYPE" \
  --md_ema_momentum "$MD_EMA_MOMENTUM" \
  --qcd_label 1 \
  --exp_name "$NURD_EXP" \
  --project_name hlt \
  --manualSeed 42

echo "=== Training outputs ==="
ls -lh "checkpoints/hlt/hlt/$AE_EXP/"
ls -lh "checkpoints/hlt/hlt/$NURD_EXP/"
echo "TRAINING DONE"
echo "=== Full training finished: $(date) ==="
