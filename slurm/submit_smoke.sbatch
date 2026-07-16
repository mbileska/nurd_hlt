#!/bin/bash
#SBATCH --job-name=nurd_smoke
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

module load anaconda3/2025.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
TRAIN_PT="${TRAIN_PT:-$BASE/data/hlt_smcocktail_train.pt}"

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

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import numpy, sklearn, scipy; print('core imports ok')"
python -c "import torch; x=torch.load('$TRAIN_PT',map_location='cpu'); print('keys', x.keys()); print('pf', x['pf'].shape); print('obj', x['obj'].shape); print('label', x['label'].shape)"

echo "=== Step 1: AE smoke ==="
python train_ae.py \
  --data "$TRAIN_PT" \
  --epochs 1 \
  --batch_size 64 \
  --max_events 2000 \
  --local_testing 1 \
  --exp_name smoke_ae \
  --project_name hlt

echo "=== Step 2: NURD smoke ==="
python train_hlt.py \
  --data "$TRAIN_PT" \
  --ae_ckpt checkpoints/hlt/hlt/smoke_ae/checkpoint_ae.pth \
  --epochs 1 \
  --batch_size 64 \
  --max_events 2000 \
  --local_testing 1 \
  --critic_schedule warmup \
  --reweight 1 \
  --joint_indep 1 \
  --exp_name smoke_nurd \
  --project_name hlt \
  --manualSeed 42

echo "=== Smoke outputs ==="
ls -lh checkpoints/hlt/hlt/smoke_ae/
ls -lh checkpoints/hlt/hlt/smoke_nurd/
echo "SMOKE DONE"
echo "=== Smoke test finished: $(date) ==="
