#!/bin/bash
#SBATCH --job-name=train3m_physics_ae
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=20:00:00
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
CODE_COMMIT="${CODE_COMMIT:?}"
AE_EXP="${AE_EXP:?}"
AE_EPOCHS="${AE_EPOCHS:?}"
AE_PATIENCE="${AE_PATIENCE:?}"

cd "$CODE_DIR"
source slurm/train3m_environment.sh
if [[ "$(git rev-parse HEAD)" != "$CODE_COMMIT" ]]; then
  echo "ERROR: code changed after campaign submission."
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked code has uncommitted changes."
  exit 2
fi
if [[ -e checkpoints && ! -L checkpoints ]]; then
  echo "ERROR: checkpoints must be a symlink to BASE/checkpoints."
  exit 2
fi
mkdir -p "$BASE/checkpoints" "$BASE/wandb"
ln -sfnT "$BASE/checkpoints" checkpoints
export WANDB_DIR="$BASE/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

AE_CKPT="$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth"
if [[ -e "$BASE/checkpoints/hlt/hlt/$AE_EXP" ]]; then
  echo "ERROR: refusing to overwrite AE experiment: $AE_EXP"
  exit 2
fi

echo "Fresh full-train3M AE: $AE_EXP"
echo "Weighting: physics (cross section x generator; no class balancing)"
echo "Maximum epochs: $AE_EPOCHS; patience: $AE_PATIENCE"
python train_ae.py \
  --data "$TRAIN_PT" \
  --gen_weight_path "$GEN_WEIGHT_TRAIN" \
  --generator_weight_label 1 \
  --ae_weighting_scheme physics \
  --epochs "$AE_EPOCHS" \
  --patience "$AE_PATIENCE" \
  --batch_size 4096 \
  --lr 1e-3 \
  --latent_dim 16 \
  --enc_nodes 512 256 \
  --dec_nodes 256 512 \
  --exp_name "$AE_EXP" \
  --project_name hlt \
  --manualSeed 42

test -s "$AE_CKPT"
echo "AE DONE: $AE_CKPT"
