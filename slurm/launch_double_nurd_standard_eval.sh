#!/bin/bash
# Inference only on a generic GPU, followed by CPU rendering of all eight suites.
set -euo pipefail
if (( $# != 2 )) || [[ ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_double_nurd_standard_eval.sh ORIGINAL_EXPERIMENT_TAG NEW_EVAL_TAG" >&2
  exit 2
fi
RUN="$1"
TAG="$2"
if [[ ! "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then echo "Invalid original tag" >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: commit tracked changes before submitting" >&2; exit 2
fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
CONVERTED="$BASE/data/train3M_v1/converted_v1"
EXPERIMENT_DIR="$BASE/outputs/${RUN}_double_nurd"
COMPARISON_DIR="$BASE/outputs/${RUN}_double_nurd_eval"
STANDARD_OUT="$BASE/outputs/${TAG}_standard_eval"
PREDICTION_OUT="$BASE/outputs/${TAG}_standard_inference"
D_MIN_ESS="${D_MIN_ESS:-100}"
if [[ ! "$D_MIN_ESS" =~ ^[1-9][0-9]*$ ]]; then echo "ERROR: positive D_MIN_ESS required" >&2; exit 2; fi
for path in "$EXPERIMENT_DIR/training_complete.json" "$COMPARISON_DIR/double_nurd_comparison.json" \
            "$CONVERTED/test/sample.json" "$CONVERTED/weight_test.npz"; do
  if [[ ! -s "$path" ]]; then echo "ERROR: missing/empty $path" >&2; exit 2; fi
done
for arm in global double; do
  for path in "$EXPERIMENT_DIR/$arm/checkpoint_final.pth.tar" \
              "$COMPARISON_DIR/${arm}_calibration_QCD.npz" "$COMPARISON_DIR/${arm}_checking_all_classes.npz"; do
    if [[ ! -s "$path" ]]; then echo "ERROR: missing/empty $path" >&2; exit 2; fi
  done
done
if [[ -e "$STANDARD_OUT" || -e "$PREDICTION_OUT" ]]; then echo "ERROR: choose a new output tag" >&2; exit 2; fi
CODE_COMMIT="$(git rev-parse HEAD)"
SNAPSHOT="$BASE/code_snapshots/${TAG}_standard_${CODE_COMMIT:0:8}"
if [[ -e "$SNAPSHOT" ]]; then echo "ERROR: snapshot already exists" >&2; exit 2; fi
mkdir -p "$SNAPSHOT" "$BASE/logs"
git archive HEAD | tar -x -C "$SNAPSHOT"
CODE_DIR="$SNAPSHOT"
export BASE CONVERTED EXPERIMENT_DIR COMPARISON_DIR STANDARD_OUT PREDICTION_OUT D_MIN_ESS CODE_DIR CODE_COMMIT
INFER_JOB="$(sbatch --parsable --job-name="${TAG}_stdinfer" --export=ALL,STAGE=infer \
  --gres=gpu:1 --time=04:00:00 \
  --output="$BASE/logs/${TAG}_stdinfer-%j.out" --error="$BASE/logs/${TAG}_stdinfer-%j.err" \
  "$SNAPSHOT/slurm/submit_double_nurd_standard_eval.sbatch")"
if ! PLOT_JOB="$(sbatch --parsable --dependency="afterok:${INFER_JOB%%;*}" \
  --partition=cpu --job-name="${TAG}_stdplots" --export=ALL,STAGE=plots \
  --output="$BASE/logs/${TAG}_stdplots-%j.out" --error="$BASE/logs/${TAG}_stdplots-%j.err" \
  "$SNAPSHOT/slurm/submit_double_nurd_standard_eval.sbatch")"; then
  echo "ERROR: inference $INFER_JOB was submitted; plot job submission failed." >&2
  echo "Recover in this shell: sbatch --partition=cpu --dependency=afterok:${INFER_JOB%%;*} --export=ALL,STAGE=plots $SNAPSHOT/slurm/submit_double_nurd_standard_eval.sbatch" >&2
  exit 1
fi
echo "Inference job: $INFER_JOB"
echo "Plot job: $PLOT_JOB"
echo "Predictions: $PREDICTION_OUT"
echo "Results: $STANDARD_OUT"
echo "Suites: {global,double}/{f100,f010}/{decay8,parent4}/plots/"
echo "Frozen models + AE/MD scores; original physics weights; no retraining or MD refit."
echo "Standard 50--98% percentile calibration cut selection; A/B/C ESS >=50, D >=$D_MIN_ESS."
echo "One generic GPU for checking inference only; CPU rendering releases GPU before KDE/PCA plots."
