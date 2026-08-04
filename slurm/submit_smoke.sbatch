#!/bin/bash
#SBATCH --job-name=nurd_smoke
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1

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

mkdir -p "$BASE/checkpoints" "$BASE/logs" "$BASE/outputs" "$BASE/wandb"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CODE_DIR"

if [[ -e checkpoints && ! -L checkpoints ]]; then
  echo "ERROR: $CODE_DIR/checkpoints exists and is not a symlink."
  echo "Move it or back it up, then link checkpoints to $BASE/checkpoints."
  exit 2
fi
ln -sfnT "$BASE/checkpoints" checkpoints

echo "=== Smoke test started: $(date) ==="
echo "Host: $(hostname)"
echo "Code: $CODE_DIR"
echo "Scratch: $BASE"
echo "Train data: $TRAIN_PT"
echo "Generator weights: $GEN_WEIGHT_TRAIN"

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import numpy, sklearn, scipy; print('core imports ok')"
python -c "import torch; x=torch.load('$TRAIN_PT',map_location='cpu'); print('keys', x.keys()); print('pf', x['pf'].shape); print('obj', x['obj'].shape); print('label', x['label'].shape)"

echo "=== Step 1: AE smoke ==="
python train_ae.py \
  --data "$TRAIN_PT" \
  --gen_weights "$GEN_WEIGHT_TRAIN" \
  --epochs 1 \
  --batch_size 64 \
  --max_events 2000 \
  --local_testing 1 \
  --exp_name smoke_ae \
  --project_name hlt

echo "=== Step 2: NURD smoke ==="
python train_hlt.py \
  --data "$TRAIN_PT" \
  --gen_weights "$GEN_WEIGHT_TRAIN" \
  --ae_ckpt checkpoints/hlt/hlt/smoke_ae/checkpoint_ae.pth \
  --epochs 1 \
  --batch_size 64 \
  --max_events 2000 \
  --local_testing 1 \
  --training_measure class_balanced_physical \
  --critic_schedule warmup \
  --critic_type continuous_density_ratio \
  --critic_bin_resolutions 50 \
  --critic_penalty_type ratio_to_one \
  --critic_scope qcd \
  --critic_shuffle global \
  --critic_train_frac 1.0 \
  --n_critic_steps_per_batch 2 \
  --baseline_labels 0,1,2,3 \
  --nuisance_bin_scope qcd \
  --n_bins 50 \
  --qcd_batch_fraction 0 \
  --reweight 1 \
  --joint_indep 1 \
  --max_weight_ratio 10.0 \
  --closure_weight 0.1 \
  --closure_loss_type hybrid \
  --closure_latent_dcorr_weight 1.5 \
  --closure_scope qcd \
  --closure_score_mode own_class \
  --closure_class_min_events 5 \
  --closure_tail_quantiles 0.5,0.7,0.9 \
  --val_md_folds 2 \
  --abcd_ckpt_smoothing_epochs 1 \
  --min_val_balanced_acc 0.0 \
  --min_val_class_acc 0.0 \
  --md_proxy_type ema \
  --exp_name smoke_nurd \
  --project_name hlt \
  --manualSeed 42

echo "=== Smoke outputs ==="
ls -lh checkpoints/hlt/hlt/smoke_ae/
ls -lh checkpoints/hlt/hlt/smoke_nurd/
echo "SMOKE DONE"
echo "=== Smoke test finished: $(date) ==="
