#!/bin/bash
#SBATCH --job-name=hlt_event_displays
#SBATCH --output=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

module load anaconda3/2025.12
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_DIR="${CODE_DIR:-/home/mb7126/nurd_hlt}"
RUN_TAG="${RUN_TAG:-engineer_continuous_v1}"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
AE_EXP="${AE_EXP:-ae_engineer_$RUN_TAG}"
NURD_EXP="${NURD_EXP:-hlt_nurd_engineer_$RUN_TAG}"

AE_CKPT="${AE_CKPT:-$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth}"
NURD_CKPT="${NURD_CKPT:-$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_main.pth.tar}"
TEST_PT="${TEST_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt}"
TEST_WEIGHTS="${TEST_WEIGHTS:-$MEQUINNA_DIR/weight_test.pt}"
REFERENCE_PT="${REFERENCE_PT:-$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt}"
REFERENCE_WEIGHTS="${REFERENCE_WEIGHTS:-$MEQUINNA_DIR/weight_train.pt}"
SIGNAL_PT="${SIGNAL_PT:-$BASE/data/hlt_signal_TpTp.pt}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-$BASE/outputs/${RUN_TAG}_eval/held-out/abcd_thresholds.json}"
OUTDIR="${OUTDIR:-$CODE_DIR/event_displays/generated/${RUN_TAG}_${SLURM_JOB_ID}}"

mkdir -p "$BASE/logs" "$BASE/matplotlib"
export MPLCONFIGDIR="$BASE/matplotlib"

for input in "$AE_CKPT" "$NURD_CKPT" "$TEST_PT" "$TEST_WEIGHTS" \
  "$REFERENCE_PT" "$REFERENCE_WEIGHTS" "$SIGNAL_PT" "$THRESHOLDS_JSON"; do
  if [[ ! -s "$input" ]]; then
    echo "ERROR: required input is missing: $input"
    exit 2
  fi
done

cd "$CODE_DIR"
python event_displays/generate_event_displays.py \
  --ckpt "$NURD_CKPT" \
  --ae-ckpt "$AE_CKPT" \
  --test-pt "$TEST_PT" \
  --test-weight-path "$TEST_WEIGHTS" \
  --reference-pt "$REFERENCE_PT" \
  --reference-weight-path "$REFERENCE_WEIGHTS" \
  --signal-pt "$SIGNAL_PT" \
  --thresholds-json "$THRESHOLDS_JSON" \
  --outdir "$OUTDIR" \
  --n-displays "${N_DISPLAYS:-3}" \
  --seed "${SEED:-42}" \
  --batch-size "${BATCH_SIZE:-512}"

echo "EVENT DISPLAYS DONE"
echo "Output: $OUTDIR"
