#!/bin/bash
# Three full-data, CE-free SupCon controls. Jobs run sequentially to use one GPU.
set -euo pipefail
if (( $# != 1 )); then
  echo "Usage: bash slurm/launch_pure_supcon_sweep.sh UNIQUE_PREFIX" >&2
  exit 2
fi
PREFIX="$1"
if [[ ! "$PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid run prefix: $PREFIX" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"
source "$SCRIPT_DIR/train3m_environment.sh"

export CE_WEIGHT=0 NURD_ENABLED=0 REUSE_AE=1
export TARGET_SCHEME=decay8 BALANCE_STRATA=1
export STATISTICALLY_VALID_CLOSURE=1 PLOT_PROJECTOR_KDE=1
export AE_WEIGHTING_SCHEME=physics
export AE_EXP="ae_engineer_train3m_physics_decay8_full_20261104_142222"
export NURD_EPOCHS="${PURE_SUPCON_EPOCHS:-50}"
export LR_SCHEDULE_EPOCHS="$NURD_EPOCHS"
export NURD_PATIENCE="${PURE_SUPCON_PATIENCE:-15}"
export CHECKPOINT_EVERY="${PURE_SUPCON_CHECKPOINT_EVERY:-10}"
export TRAIN_LR="${PURE_SUPCON_LR:-1e-4}"
export WARM_START_CKPT=
test -s "$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth" || {
  echo "Missing frozen physics AE checkpoint: $AE_EXP" >&2
  exit 2
}

previous_eval=""
for specification in '01:0.1' '03:0.3' '05:0.5'; do
  suffix="${specification%%:*}"
  weight="${specification#*:}"
  run_tag="${PREFIX}_sc${suffix}"
  export TRAIN_DEPENDENCY="$previous_eval"
  launch_output="$(bash "$SCRIPT_DIR/launch_engineer_campaign.sh" \
    "$run_tag" "$weight")"
  printf '%s\n' "$launch_output"
  previous_eval="$(printf '%s\n' "$launch_output" | awk '$1 == "Evaluation" && $2 == "job:" {print $3}')"
  if [[ ! "$previous_eval" =~ ^[0-9]+$ ]]; then
    echo "Could not parse evaluation job ID for $run_tag" >&2
    exit 2
  fi
  echo "Projector KDE: $BASE/outputs/${run_tag}_eval/held-out/plots/projector_pca_kde_by_class.png"
  echo "Score KDE: $BASE/outputs/${run_tag}_eval/held-out/plots/hist2d_pca_md_kde.png"
done
echo "All three controls submitted in sequence; the first starts when scheduled."
