#!/bin/bash
#SBATCH --job-name=nurd_hlt_train
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
TRAIN_PT="${TRAIN_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt}"
GEN_WEIGHT_TRAIN="${GEN_WEIGHT_TRAIN:-$MEQUINNA_DIR/weight_train.pt}"
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

RUN_TAG="${RUN_TAG:-experimental_global_nurd_calibrated_v9_$(date +%Y%m%d_%H%M%S)}"
AE_EXP="${AE_EXP:-ae_pretrain_$RUN_TAG}"
NURD_EXP="${NURD_EXP:-hlt_nurd_closure_bs4096_$RUN_TAG}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
AE_EPOCHS="${AE_EPOCHS:-100}"
NURD_EPOCHS="${NURD_EPOCHS:-180}"
REPRESENTATION_EPOCHS="${REPRESENTATION_EPOCHS:-140}"
QCD_FINETUNE_EPOCHS="${QCD_FINETUNE_EPOCHS:-40}"
FINETUNE_LR_MULTIPLIER="${FINETUNE_LR_MULTIPLIER:-0.25}"
FINETUNE_ANCHOR_WEIGHT="${FINETUNE_ANCHOR_WEIGHT:-0.01}"
FINETUNE_MAX_BALANCED_ACC_DROP="${FINETUNE_MAX_BALANCED_ACC_DROP:-0.03}"
FINETUNE_MAX_CLASS_ACC_DROP="${FINETUNE_MAX_CLASS_ACC_DROP:-0.05}"
CLOSURE_WEIGHT="${CLOSURE_WEIGHT:-0.5}"
CLOSURE_WEIGHT_START="${CLOSURE_WEIGHT_START:-0.0}"
CLOSURE_RAMP_EPOCHS="${CLOSURE_RAMP_EPOCHS:-20}"
CLOSURE_LOSS_TYPE="${CLOSURE_LOSS_TYPE:-hybrid}"
CLOSURE_SCOPE="${CLOSURE_SCOPE:-qcd}"
CLOSURE_CLASS_MIN_EVENTS="${CLOSURE_CLASS_MIN_EVENTS:-20}"
CLOSURE_CLASS_WEIGHTING="${CLOSURE_CLASS_WEIGHTING:-equal}"
CLOSURE_SCORE_MODE="${CLOSURE_SCORE_MODE:-own_class}"
CLOSURE_REVERSE_PROFILE_WEIGHT="${CLOSURE_REVERSE_PROFILE_WEIGHT:-0.5}"
CLOSURE_TAIL_ABCD_WEIGHT="${CLOSURE_TAIL_ABCD_WEIGHT:-1.0}"
CLOSURE_TAIL_QUANTILES="${CLOSURE_TAIL_QUANTILES:-0.50,0.65,0.80,0.90}"
CLOSURE_TAIL_SCALE="${CLOSURE_TAIL_SCALE:-12.0}"
CLOSURE_TAIL_MIN_EVENTS="${CLOSURE_TAIL_MIN_EVENTS:-5}"
CLOSURE_TAIL_FOCUS_WEIGHT="${CLOSURE_TAIL_FOCUS_WEIGHT:-2.0}"
VAL_ABCD_QUANTILES="${VAL_ABCD_QUANTILES:-0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.92}"
VAL_ABCD_MIN_EVENTS="${VAL_ABCD_MIN_EVENTS:-20}"
VAL_ABCD_MIN_REGION_FRAC="${VAL_ABCD_MIN_REGION_FRAC:-0.005}"
ABCD_CKPT_LOSS_TOL="${ABCD_CKPT_LOSS_TOL:-1.10}"
ABCD_CKPT_MIN_EPOCH="${ABCD_CKPT_MIN_EPOCH:-40}"
BASELINE_LABELS="${BASELINE_LABELS:-0,1,2,3}"
NUISANCE_BIN_SCOPE="${NUISANCE_BIN_SCOPE:-all}"
CRITIC_SCOPE="${CRITIC_SCOPE:-all}"
N_BINS="${N_BINS:-50}"
CRITIC_TYPE="${CRITIC_TYPE:-continuous_density_ratio}"
CRITIC_BIN_RESOLUTIONS="${CRITIC_BIN_RESOLUTIONS:-50}"
CRITIC_PENALTY_TYPE="${CRITIC_PENALTY_TYPE:-ratio_to_one}"
CRITIC_SHUFFLE="${CRITIC_SHUFFLE:-global}"
N_CRITIC_STEPS="${N_CRITIC_STEPS:-2}"
QCD_BATCH_FRACTION="${QCD_BATCH_FRACTION:-0.0}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-0.02}"
CONTRAST_WEIGHT_START="${CONTRAST_WEIGHT_START:-0.15}"
CONTRAST_RAMP_EPOCHS="${CONTRAST_RAMP_EPOCHS:-40}"
MAX_WEIGHT_RATIO="${MAX_WEIGHT_RATIO:-10.0}"
MD_PROXY_TYPE="${MD_PROXY_TYPE:-epoch}"
MD_EMA_MOMENTUM="${MD_EMA_MOMENTUM:-0.01}"
MD_PROXY_SHRINKAGE="${MD_PROXY_SHRINKAGE:-0.05}"
VAL_MD_FOLDS="${VAL_MD_FOLDS:-2}"
ABCD_CKPT_SMOOTHING_EPOCHS="${ABCD_CKPT_SMOOTHING_EPOCHS:-5}"
VAL_ABCD_MIN_EFFECTIVE_EVENTS="${VAL_ABCD_MIN_EFFECTIVE_EVENTS:-20}"
VAL_ABCD_MAX_RATIO_UNC="${VAL_ABCD_MAX_RATIO_UNC:-0.15}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAINING_MEASURE="${TRAINING_MEASURE:-class_balanced_physical}"
LAMBDA="${LAMBDA:-0.05}"
CLOSURE_LATENT_DCORR_WEIGHT="${CLOSURE_LATENT_DCORR_WEIGHT:-1.5}"
MIN_VAL_BALANCED_ACC="${MIN_VAL_BALANCED_ACC:-0.55}"
MIN_VAL_CLASS_ACC="${MIN_VAL_CLASS_ACC:-0.25}"
AE_CKPT="${AE_CKPT:-checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth}"

if (( REPRESENTATION_EPOCHS + QCD_FINETUNE_EPOCHS != NURD_EPOCHS )); then
  echo "ERROR: REPRESENTATION_EPOCHS + QCD_FINETUNE_EPOCHS must equal NURD_EPOCHS."
  exit 2
fi

echo "=== Full training started: $(date) ==="
echo "Host: $(hostname)"
echo "Code: $CODE_DIR"
echo "Scratch: $BASE"
echo "Train data: $TRAIN_PT"
echo "Generator weights: $GEN_WEIGHT_TRAIN"
echo "AE_EXP=$AE_EXP"
echo "NURD_EXP=$NURD_EXP"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "AE_EPOCHS=$AE_EPOCHS"
echo "NURD_EPOCHS=$NURD_EPOCHS"
echo "REPRESENTATION_EPOCHS=$REPRESENTATION_EPOCHS"
echo "QCD_FINETUNE_EPOCHS=$QCD_FINETUNE_EPOCHS"
echo "FINETUNE_LR_MULTIPLIER=$FINETUNE_LR_MULTIPLIER"
echo "FINETUNE_ANCHOR_WEIGHT=$FINETUNE_ANCHOR_WEIGHT"
echo "FINETUNE_MAX_BALANCED_ACC_DROP=$FINETUNE_MAX_BALANCED_ACC_DROP"
echo "FINETUNE_MAX_CLASS_ACC_DROP=$FINETUNE_MAX_CLASS_ACC_DROP"
echo "AE_CKPT=$AE_CKPT"
echo "BASELINE_LABELS=$BASELINE_LABELS"
echo "CRITIC_SCOPE=$CRITIC_SCOPE"
echo "CRITIC_TYPE=$CRITIC_TYPE"
echo "CRITIC_BIN_RESOLUTIONS=$CRITIC_BIN_RESOLUTIONS"
echo "CRITIC_PENALTY_TYPE=$CRITIC_PENALTY_TYPE"
echo "CRITIC_SHUFFLE=$CRITIC_SHUFFLE"
echo "N_CRITIC_STEPS=$N_CRITIC_STEPS"
echo "LAMBDA=$LAMBDA"
echo "N_BINS=$N_BINS"
echo "TRAINING_MEASURE=$TRAINING_MEASURE"
echo "QCD_BATCH_FRACTION=$QCD_BATCH_FRACTION"
echo "NUISANCE_BIN_SCOPE=$NUISANCE_BIN_SCOPE"
echo "CONTRAST_WEIGHT=$CONTRAST_WEIGHT"
echo "CONTRAST_WEIGHT_START=$CONTRAST_WEIGHT_START"
echo "CONTRAST_RAMP_EPOCHS=$CONTRAST_RAMP_EPOCHS"
echo "CLOSURE_WEIGHT=$CLOSURE_WEIGHT"
echo "CLOSURE_WEIGHT_START=$CLOSURE_WEIGHT_START"
echo "CLOSURE_RAMP_EPOCHS=$CLOSURE_RAMP_EPOCHS"
echo "CLOSURE_LOSS_TYPE=$CLOSURE_LOSS_TYPE"
echo "CLOSURE_SCOPE=$CLOSURE_SCOPE"
echo "CLOSURE_CLASS_MIN_EVENTS=$CLOSURE_CLASS_MIN_EVENTS"
echo "CLOSURE_CLASS_WEIGHTING=$CLOSURE_CLASS_WEIGHTING"
echo "CLOSURE_SCORE_MODE=$CLOSURE_SCORE_MODE"
echo "CLOSURE_LATENT_DCORR_WEIGHT=$CLOSURE_LATENT_DCORR_WEIGHT"
echo "CLOSURE_REVERSE_PROFILE_WEIGHT=$CLOSURE_REVERSE_PROFILE_WEIGHT"
echo "CLOSURE_TAIL_ABCD_WEIGHT=$CLOSURE_TAIL_ABCD_WEIGHT"
echo "CLOSURE_TAIL_QUANTILES=$CLOSURE_TAIL_QUANTILES"
echo "CLOSURE_TAIL_SCALE=$CLOSURE_TAIL_SCALE"
echo "CLOSURE_TAIL_MIN_EVENTS=$CLOSURE_TAIL_MIN_EVENTS"
echo "CLOSURE_TAIL_FOCUS_WEIGHT=$CLOSURE_TAIL_FOCUS_WEIGHT"
echo "VAL_ABCD_QUANTILES=$VAL_ABCD_QUANTILES"
echo "VAL_ABCD_MIN_EVENTS=$VAL_ABCD_MIN_EVENTS"
echo "VAL_ABCD_MIN_REGION_FRAC=$VAL_ABCD_MIN_REGION_FRAC"
echo "ABCD_CKPT_LOSS_TOL=$ABCD_CKPT_LOSS_TOL"
echo "ABCD_CKPT_MIN_EPOCH=$ABCD_CKPT_MIN_EPOCH"
echo "MAX_WEIGHT_RATIO=$MAX_WEIGHT_RATIO"
echo "MD_PROXY_TYPE=$MD_PROXY_TYPE"
echo "MD_EMA_MOMENTUM=$MD_EMA_MOMENTUM"
echo "MD_PROXY_SHRINKAGE=$MD_PROXY_SHRINKAGE"
echo "VAL_MD_FOLDS=$VAL_MD_FOLDS"
echo "ABCD_CKPT_SMOOTHING_EPOCHS=$ABCD_CKPT_SMOOTHING_EPOCHS"
echo "VAL_ABCD_MIN_EFFECTIVE_EVENTS=$VAL_ABCD_MIN_EFFECTIVE_EVENTS"
echo "VAL_ABCD_MAX_RATIO_UNC=$VAL_ABCD_MAX_RATIO_UNC"
echo "MIN_VAL_BALANCED_ACC=$MIN_VAL_BALANCED_ACC"
echo "MIN_VAL_CLASS_ACC=$MIN_VAL_CLASS_ACC"
echo "W&B project: nurd-ood-hlt"
echo "W&B mode: $WANDB_MODE"
echo "Minimum GPU memory: ${MIN_GPU_MEM_GB} GiB"

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import torch, sys; assert torch.cuda.is_available(), 'CUDA is not available'; props=torch.cuda.get_device_properties(0); mem_gib=props.total_memory/1024**3; print(f'GPU memory: {mem_gib:.2f} GiB'); sys.exit(0 if mem_gib >= float('$MIN_GPU_MEM_GB') else 42)"
python -c "import numpy, sklearn, scipy, matplotlib, wandb; print('training/eval imports ok')"
python -c "import torch; x=torch.load('$TRAIN_PT',map_location='cpu'); print('keys', x.keys()); print('pf', x['pf'].shape); print('obj', x['obj'].shape); print('label', x['label'].shape)"
python -c "import torch; from utils.event_weights import load_event_weights; x=torch.load('$TRAIN_PT',map_location='cpu'); _,m=load_event_weights('$GEN_WEIGHT_TRAIN',x); print('generator weight metadata',m)"

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
    --gen_weights "$GEN_WEIGHT_TRAIN" \
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
  --gen_weights "$GEN_WEIGHT_TRAIN" \
  --training_measure "$TRAINING_MEASURE" \
  --ae_ckpt "$AE_CKPT" \
  --epochs "$NURD_EPOCHS" \
  --representation_epochs "$REPRESENTATION_EPOCHS" \
  --qcd_finetune_epochs "$QCD_FINETUNE_EPOCHS" \
  --finetune_lr_multiplier "$FINETUNE_LR_MULTIPLIER" \
  --finetune_anchor_weight "$FINETUNE_ANCHOR_WEIGHT" \
  --finetune_max_balanced_acc_drop "$FINETUNE_MAX_BALANCED_ACC_DROP" \
  --finetune_max_class_acc_drop "$FINETUNE_MAX_CLASS_ACC_DROP" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr 1e-4 \
  --weight_decay 5e-3 \
  --n_bins "$N_BINS" \
  --qcd_batch_fraction "$QCD_BATCH_FRACTION" \
  --baseline_labels "$BASELINE_LABELS" \
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
  --critic_bin_resolutions "$CRITIC_BIN_RESOLUTIONS" \
  --critic_penalty_type "$CRITIC_PENALTY_TYPE" \
  --critic_shuffle "$CRITIC_SHUFFLE" \
  --critic_warmup_epochs 7 \
  --critic_ramp_epochs 10 \
  --critic_train_frac 1.0 \
  --critic_lr_multiplier 10.0 \
  --n_critic_steps_per_batch "$N_CRITIC_STEPS" \
  --critic_scope "$CRITIC_SCOPE" \
  --_lambda "$LAMBDA" \
  --no_mi_norm \
  --closure_weight "$CLOSURE_WEIGHT" \
  --closure_scope "$CLOSURE_SCOPE" \
  --closure_class_min_events "$CLOSURE_CLASS_MIN_EVENTS" \
  --closure_class_weighting "$CLOSURE_CLASS_WEIGHTING" \
  --closure_score_mode "$CLOSURE_SCORE_MODE" \
  --closure_weight_start "$CLOSURE_WEIGHT_START" \
  --closure_ramp_epochs "$CLOSURE_RAMP_EPOCHS" \
  --closure_loss_type "$CLOSURE_LOSS_TYPE" \
  --closure_corr_weight 0.25 \
  --closure_dcorr_weight 1.0 \
  --closure_latent_dcorr_weight "$CLOSURE_LATENT_DCORR_WEIGHT" \
  --closure_profile_weight 0.5 \
  --closure_reverse_profile_weight "$CLOSURE_REVERSE_PROFILE_WEIGHT" \
  --closure_profile_bins 8 \
  --closure_profile_tail_weight 2.0 \
  --closure_dcorr_max_samples 512 \
  --closure_tail_abcd_weight "$CLOSURE_TAIL_ABCD_WEIGHT" \
  --closure_tail_quantiles "$CLOSURE_TAIL_QUANTILES" \
  --closure_tail_scale "$CLOSURE_TAIL_SCALE" \
  --closure_tail_min_events "$CLOSURE_TAIL_MIN_EVENTS" \
  --closure_tail_focus_weight "$CLOSURE_TAIL_FOCUS_WEIGHT" \
  --abcd_ckpt_loss_tol "$ABCD_CKPT_LOSS_TOL" \
  --abcd_ckpt_min_epoch "$ABCD_CKPT_MIN_EPOCH" \
  --val_abcd_quantiles "$VAL_ABCD_QUANTILES" \
  --val_abcd_min_events "$VAL_ABCD_MIN_EVENTS" \
  --val_abcd_min_region_frac "$VAL_ABCD_MIN_REGION_FRAC" \
  --val_abcd_min_effective_events "$VAL_ABCD_MIN_EFFECTIVE_EVENTS" \
  --val_abcd_max_ratio_unc "$VAL_ABCD_MAX_RATIO_UNC" \
  --val_md_folds "$VAL_MD_FOLDS" \
  --abcd_ckpt_smoothing_epochs "$ABCD_CKPT_SMOOTHING_EPOCHS" \
  --min_val_balanced_acc "$MIN_VAL_BALANCED_ACC" \
  --min_val_class_acc "$MIN_VAL_CLASS_ACC" \
  --md_proxy_type "$MD_PROXY_TYPE" \
  --md_ema_momentum "$MD_EMA_MOMENTUM" \
  --md_proxy_shrinkage "$MD_PROXY_SHRINKAGE" \
  --qcd_label 1 \
  --exp_name "$NURD_EXP" \
  --project_name hlt \
  --manualSeed 42

echo "=== Training outputs ==="
ls -lh "checkpoints/hlt/hlt/$AE_EXP/"
ls -lh "checkpoints/hlt/hlt/$NURD_EXP/"
echo "TRAINING DONE"
echo "=== Full training finished: $(date) ==="
