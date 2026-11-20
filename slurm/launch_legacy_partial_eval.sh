#!/bin/bash
# Evaluate existing best Mequinna checkpoints, including interrupted trainings.
set -euo pipefail
if (( $# != 2 )); then
  echo "Usage: bash slurm/launch_legacy_partial_eval.sh ORIGINAL_TRAINING_TAG NEW_EVAL_TAG" >&2; exit 2
fi
RUN="$1"
TAG="$2"
for tag in "$RUN" "$TAG"; do
  if [[ ! "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then echo "ERROR: invalid tag $tag" >&2; exit 2; fi
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR/.."
if ! git diff --quiet || ! git diff --cached --quiet; then echo "ERROR: commit tracked changes before submitting" >&2; exit 2; fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CODE_COMMIT="$(git rev-parse HEAD)"
LEGACY_OUT="$BASE/outputs/${RUN}_legacy_double_nurd"
PREPARATION="$LEGACY_OUT/preparation_complete.json"
if [[ ! -s "$PREPARATION" ]]; then echo "ERROR: missing preparation $PREPARATION" >&2; exit 2; fi
LEGACY_TRAIN="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sample"])' "$PREPARATION")"
LEGACY_WEIGHTS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["weights"])' "$PREPARATION")"
SOURCE_COMMIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "$PREPARATION")"
MEQUINNA_DIR="${MEQUINNA_DIR:-$(dirname "$LEGACY_TRAIN")}"
LEGACY_TEST="$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt"
LEGACY_TEST_WEIGHTS="$MEQUINNA_DIR/weight_test.pt"
GOLDEN_SUMMARY="${GOLDEN_SUMMARY:-$BASE/code_snapshots/${RUN}_legacy_driver_${SOURCE_COMMIT:0:8}/golden_evaluation_summary.json}"
LEGACY_SCORES="$BASE/outputs/${TAG}_legacy_partial_scores"
LEGACY_EVAL="$BASE/outputs/${TAG}_legacy_partial_eval"
for path in "$LEGACY_TRAIN" "$LEGACY_WEIGHTS" "$LEGACY_TEST" "$LEGACY_TEST_WEIGHTS" "$GOLDEN_SUMMARY" \
            "$LEGACY_OUT/prepared_recipe.pth.tar" "$LEGACY_OUT/golden_checkpoint.pth.tar" \
            "$LEGACY_OUT/original_checkpoint_ae.pth"; do
  if [[ ! -s "$path" ]]; then echo "ERROR: missing/empty $path" >&2; exit 2; fi
done
for arm in single double; do
  for name in checkpoint_main.pth.tar training_history.json; do
    if [[ ! -s "$LEGACY_OUT/$arm/$name" ]]; then echo "ERROR: missing $arm/$name" >&2; exit 2; fi
  done
done
CODE_DIR="$BASE/code_snapshots/${TAG}_partial_eval_${CODE_COMMIT:0:8}"
for path in "$CODE_DIR" "$LEGACY_SCORES" "$LEGACY_EVAL"; do
  if [[ -e "$path" ]]; then echo "ERROR: refusing to overwrite $path; choose a fresh evaluation tag" >&2; exit 2; fi
done
mkdir -p "$CODE_DIR" "$BASE/logs"
git archive HEAD | tar -x -C "$CODE_DIR"
cp -p "$GOLDEN_SUMMARY" "$CODE_DIR/golden_evaluation_summary.json"
GOLDEN_SUMMARY="$CODE_DIR/golden_evaluation_summary.json"
export BASE CODE_DIR CODE_COMMIT LEGACY_OUT LEGACY_SCORES LEGACY_EVAL
export LEGACY_TRAIN LEGACY_WEIGHTS LEGACY_TEST LEGACY_TEST_WEIGHTS GOLDEN_SUMMARY
SCORE_JOB="$(sbatch --parsable --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=08:00:00 \
  --job-name="${TAG}_partial_score" --export=ALL,STAGE=score \
  --output="$BASE/logs/${TAG}_partial_score-%j.out" --error="$BASE/logs/${TAG}_partial_score-%j.err" \
  "$CODE_DIR/slurm/submit_legacy_partial_eval.sbatch")"
echo "Scoring job: $SCORE_JOB"
if ! PLOT_JOB="$(sbatch --parsable --partition=cpu --cpus-per-task=2 --mem=16G --time=08:00:00 \
  --dependency="afterok:${SCORE_JOB%%;*}" --job-name="${TAG}_partial_plots" --export=ALL,STAGE=plots \
  --output="$BASE/logs/${TAG}_partial_plots-%j.out" --error="$BASE/logs/${TAG}_partial_plots-%j.err" \
  "$CODE_DIR/slurm/submit_legacy_partial_eval.sbatch")"; then
  echo "ERROR: scoring was submitted, but plotting submission failed." >&2
  echo "Recover in this shell: sbatch --partition=cpu --cpus-per-task=2 --mem=16G --time=08:00:00 --dependency=afterok:${SCORE_JOB%%;*} --export=ALL,STAGE=plots $CODE_DIR/slurm/submit_legacy_partial_eval.sbatch" >&2
  exit 1
fi
echo "Plot job: $PLOT_JOB"
echo "Scores: $LEGACY_SCORES"
echo "Evaluation: $LEGACY_EVAL"
echo "Logs: $BASE/logs/${TAG}_partial_{score,plots}-JOB.{out,err}"
echo "Evaluation ONLY: existing post-ramp best validation-CE checkpoints + frozen golden; no training or completion markers."
echo "All six standard parent4 plot suites: {golden,single,double}/{f100,f010}/plots/."
