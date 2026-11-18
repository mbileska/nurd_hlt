#!/bin/bash
# One matched training job (two arms sequentially), then one dependent evaluation.
set -euo pipefail
if (( $# != 1 )) || [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_double_nurd_experiment.sh UNIQUE_TAG" >&2
  exit 2
fi
TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if [[ "$(git branch --show-current)" != double-nurd-experiment ]]; then
  echo "ERROR: launch from branch double-nurd-experiment" >&2; exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submission" >&2; exit 2
fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CONVERTED="$BASE/data/train3M_v1/converted_v1"
SOURCE_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_train3m_physics_decay8_lambda1_20261107_132516/checkpoint_main.pth.tar"
ORIGINAL_AE="$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
AE_CACHE_DIR="${AE_CACHE_DIR:-$BASE/outputs/ae_phys_randomsplit_20261117_150546_AE_random_control}"
EXPERIMENT_OUT="$BASE/outputs/${TAG}_double_nurd"
EVALUATION_OUT="$BASE/outputs/${TAG}_double_nurd_eval"
PILOT_UPDATES="${PILOT_UPDATES:-6000}"
PILOT_BATCH_SIZE="${PILOT_BATCH_SIZE:-512}"
TAIL_RAMP_UPDATES="${TAIL_RAMP_UPDATES:-1000}"
MIN_POST_RAMP_UPDATES="${MIN_POST_RAMP_UPDATES:-3000}"
D_MIN_ESS="${D_MIN_ESS:-100}"
for name in PILOT_UPDATES PILOT_BATCH_SIZE MIN_POST_RAMP_UPDATES D_MIN_ESS; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then echo "ERROR: positive integer $name required" >&2; exit 2; fi
done
if [[ ! "$TAIL_RAMP_UPDATES" =~ ^[0-9]+$ ]] || (( PILOT_BATCH_SIZE < 2 || PILOT_UPDATES < TAIL_RAMP_UPDATES + MIN_POST_RAMP_UPDATES )); then
  echo "ERROR: budget must cover the entire tail ramp plus the minimum full-strength period" >&2; exit 2
fi
for path in "$SOURCE_CKPT" "$ORIGINAL_AE" "$CONVERTED/train/sample.json" \
            "$CONVERTED/weight_train.npz" "$CONVERTED/test/sample.json" "$CONVERTED/weight_test.npz" \
            "$AE_CACHE_DIR/AE_control_summary.json" "$AE_CACHE_DIR/frozen_AE_scores.npz"; do
  if [[ ! -s "$path" ]]; then echo "ERROR: missing/empty input $path" >&2; exit 2; fi
done
if [[ ! -e "$CONVERTED/FULL_VALIDATION_PASSED" ]]; then
  echo "ERROR: converted dataset has no FULL_VALIDATION_PASSED marker" >&2; exit 2
fi
if [[ -e "$EXPERIMENT_OUT" || -e "$EVALUATION_OUT" ]]; then
  echo "ERROR: tag already exists; choose a new tag. Nothing will be overwritten." >&2; exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
# Isolate queued jobs from subsequent branch switches/edits in the checkout.
SNAPSHOT="$BASE/code_snapshots/${TAG}_double_nurd_${CODE_COMMIT:0:8}"
if [[ -e "$SNAPSHOT" ]]; then echo "ERROR: code snapshot already exists: $SNAPSHOT" >&2; exit 2; fi
mkdir -p "$SNAPSHOT" "$BASE/logs"
git archive HEAD | tar -x -C "$SNAPSHOT"
CODE_DIR="$SNAPSHOT"
export BASE CODE_DIR CODE_COMMIT CONVERTED SOURCE_CKPT ORIGINAL_AE AE_CACHE_DIR EXPERIMENT_OUT EVALUATION_OUT
export PILOT_UPDATES PILOT_BATCH_SIZE TAIL_RAMP_UPDATES MIN_POST_RAMP_UPDATES D_MIN_ESS
TRAIN_JOB="$(sbatch --parsable --job-name="${TAG}_doubletrain" --export=ALL \
  --output="$BASE/logs/${TAG}_doubletrain-%j.out" --error="$BASE/logs/${TAG}_doubletrain-%j.err" \
  "$SNAPSHOT/slurm/submit_double_nurd_train.sbatch")"
if ! EVAL_JOB="$(sbatch --parsable --dependency="afterok:${TRAIN_JOB%%;*}" \
  --job-name="${TAG}_doubleeval" --export=ALL \
  --output="$BASE/logs/${TAG}_doubleeval-%j.out" --error="$BASE/logs/${TAG}_doubleeval-%j.err" \
  "$SNAPSHOT/slurm/submit_double_nurd_eval.sbatch")"; then
  echo "ERROR: training $TRAIN_JOB was submitted, but evaluation submission failed." >&2
  echo "Recover with: sbatch --dependency=afterok:${TRAIN_JOB%%;*} --export=ALL $SNAPSHOT/slurm/submit_double_nurd_eval.sbatch" >&2
  exit 1
fi
echo "Training job: $TRAIN_JOB"
echo "Evaluation job: $EVAL_JOB"
echo "Training: $EXPERIMENT_OUT"
echo "Evaluation: $EVALUATION_OUT"
echo "Code snapshot: $SNAPSHOT (commit $CODE_COMMIT)"
echo "Both arms start from: $SOURCE_CKPT"
echo "Same frozen physical AE; same decay8 CE + SupCon=0.3 training sample and weights."
echo "Global critic lambda=1; double arm adds tail critic ramp 0->1 over $TAIL_RAMP_UPDATES updates."
echo "Critic-only warm-up: 300; encoder updates per arm: $PILOT_UPDATES; full-strength tail updates: $((PILOT_UPDATES-TAIL_RAMP_UPDATES))."
echo "No CE early stopping. Save every 2000 updates and final. Full + fixed top-10% checks; D ESS >= $D_MIN_ESS."
echo "Each job requests one generic GPU; no explicit partition or GPU-model constraint."
