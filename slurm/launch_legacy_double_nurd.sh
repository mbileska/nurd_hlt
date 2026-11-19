#!/bin/bash
# Historical full parent4 training recipe + additive tail critic. No train3M inputs.
set -euo pipefail
if (( $# != 1 )) || [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Usage: bash slurm/launch_legacy_double_nurd.sh UNIQUE_TAG" >&2; exit 2
fi
TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR/.."
if ! git diff --quiet || ! git diff --cached --quiet; then echo "ERROR: commit tracked changes before submission" >&2; exit 2; fi
BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
OLD_COMMIT=b5b2272939cfe8898c7001c5b3abfcfb8c94ef4e
CODE_COMMIT="$(git rev-parse HEAD)"
MEQUINNA_DIR="${MEQUINNA_DIR:-$BASE/data/mequinna_1M_noZB}"
LEGACY_TRAIN="$MEQUINNA_DIR/hlt_smcocktail_mequinna_train.pt"
LEGACY_WEIGHTS="$MEQUINNA_DIR/weight_train.pt"
LEGACY_TEST="$MEQUINNA_DIR/hlt_smcocktail_mequinna_test.pt"
LEGACY_TEST_WEIGHTS="$MEQUINNA_DIR/weight_test.pt"
GOLDEN_CKPT="$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_engineer_continuous_supcon030/checkpoint_main.pth.tar"
GOLDEN_AE="$BASE/checkpoints/hlt/hlt/ae_engineer_engineer_continuous_supcon030/checkpoint_ae.pth"
GOLDEN_SUMMARY="${GOLDEN_SUMMARY:-$BASE/outputs/engineer_continuous_supcon030_eval_stat_valid/evaluation_summary.json}"
LEGACY_OUT="$BASE/outputs/${TAG}_legacy_double_nurd"
LEGACY_SCORES="$BASE/outputs/${TAG}_legacy_scores"
LEGACY_EVAL="$BASE/outputs/${TAG}_legacy_eval"
LEGACY_EPOCHS="${LEGACY_EPOCHS:-40}"
if [[ ! "$LEGACY_EPOCHS" =~ ^[1-9][0-9]*$ ]] || (( LEGACY_EPOCHS < 40 )); then
  echo "ERROR: LEGACY_EPOCHS must be >=40 (5 warm-up +10 ramp +25 full-strength epochs)" >&2; exit 2
fi
for path in "$LEGACY_TRAIN" "$LEGACY_WEIGHTS" "$LEGACY_TEST" "$LEGACY_TEST_WEIGHTS" \
            "$GOLDEN_CKPT" "$GOLDEN_AE" "$GOLDEN_SUMMARY"; do
  if [[ ! -s "$path" ]]; then echo "ERROR: missing/empty input $path" >&2; exit 2; fi
done
for path in "$LEGACY_OUT" "$LEGACY_SCORES" "$LEGACY_EVAL"; do
  if [[ -e "$path" ]]; then echo "ERROR: refusing to overwrite $path" >&2; exit 2; fi
done
CODE_DIR="$BASE/code_snapshots/${TAG}_legacy_driver_${CODE_COMMIT:0:8}"
OLD_CODE_DIR="$BASE/code_snapshots/${TAG}_golden_b5b2272"
if [[ -e "$CODE_DIR" || -e "$OLD_CODE_DIR" ]]; then echo "ERROR: code snapshot exists; use a fresh tag" >&2; exit 2; fi
git cat-file -e "$OLD_COMMIT^{commit}"
mkdir -p "$CODE_DIR" "$OLD_CODE_DIR" "$BASE/logs"
git archive HEAD | tar -x -C "$CODE_DIR"
git archive "$OLD_COMMIT" train_hlt.py dataset models utils | tar -x -C "$OLD_CODE_DIR"
cp -p "$GOLDEN_SUMMARY" "$CODE_DIR/golden_evaluation_summary.json"
GOLDEN_SUMMARY="$CODE_DIR/golden_evaluation_summary.json"
export BASE CODE_COMMIT CODE_DIR OLD_CODE_DIR LEGACY_TRAIN LEGACY_WEIGHTS LEGACY_TEST LEGACY_TEST_WEIGHTS
export GOLDEN_CKPT GOLDEN_AE GOLDEN_SUMMARY LEGACY_OUT LEGACY_SCORES LEGACY_EVAL LEGACY_EPOCHS
submit() {
  local stage="$1" dependency="$2"
  local flags=(--gres=gpu:1 --mem=64G --time=20:00:00)
  if [[ "$stage" == prepare ]]; then flags=(--gres=gpu:1 --mem=64G --time=04:00:00); fi
  if [[ "$stage" == score ]]; then flags=(--gres=gpu:1 --mem=64G --time=08:00:00); fi
  if [[ "$stage" == plots ]]; then flags=(--partition=cpu --cpus-per-task=2 --mem=16G --time=08:00:00); fi
  if [[ -n "$dependency" ]]; then flags+=(--dependency="afterok:$dependency"); fi
  sbatch --parsable "${flags[@]}" --job-name="${TAG}_${stage}" --export="ALL,STAGE=$stage" \
    --output="$BASE/logs/${TAG}_${stage}-%j.out" --error="$BASE/logs/${TAG}_${stage}-%j.err" \
    "$CODE_DIR/slurm/submit_legacy_double_nurd.sbatch"
}
PREP_JOB="$(submit prepare '')"
echo "Preparation job: $PREP_JOB"
SINGLE_JOB="$(submit single "${PREP_JOB%%;*}")"
echo "Single training job: $SINGLE_JOB"
DOUBLE_JOB="$(submit double "${PREP_JOB%%;*}")"
echo "Double training job: $DOUBLE_JOB"
SCORE_JOB="$(submit score "${SINGLE_JOB%%;*}:${DOUBLE_JOB%%;*}")"
echo "Scoring job: $SCORE_JOB"
PLOT_JOB="$(submit plots "${SCORE_JOB%%;*}")"
echo "Plot job: $PLOT_JOB"
echo "Training: $LEGACY_OUT"
echo "Scores: $LEGACY_SCORES"
echo "Evaluation: $LEGACY_EVAL"
echo "Exactly TWO fresh parent4 trainings; frozen golden is an evaluation reference, not another training."
echo "Old AE, 20 unclipped quantile strata, original batch 4096 and SupCon=.3 retained from the source config."
echo "Both arms: $LEGACY_EPOCHS full epochs, original 40-epoch cosine LR then floor; no early stopping."
echo "Tail penalty: 0 for 5 epochs, ramp to1 over10, >=25 post-ramp epochs. Global lambda remains1 from epoch1."
echo "Full and frozen top10% standard plots for {golden,single,double}; full D ESS>=500, tail D ESS>=100."
echo "Generic GPU requests without GPU-model or partition constraints; CPU job renders plots."
echo "Activation checkpointing retains original batch/architecture, reducing VRAM at extra compute cost."
