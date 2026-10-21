#!/bin/bash
# Source this file before launch helpers that should use the new train3M sample.

BASE="${BASE:-/scratch/gpfs/IOJALVO/mb7126/nurd_hlt}"
TRAIN3M_DIR="${TRAIN3M_DIR:-$BASE/data/train3M_v1}"

export BASE TRAIN3M_DIR
export TRAIN_PT="${TRAIN_PT:-$TRAIN3M_DIR/train/sample.json}"
export GEN_WEIGHT_TRAIN="${GEN_WEIGHT_TRAIN:-$TRAIN3M_DIR/weight_train.npz}"
export HELDOUT_PT="${HELDOUT_PT:-$TRAIN3M_DIR/test/sample.json}"
export HELDOUT_WEIGHTS="${HELDOUT_WEIGHTS:-$TRAIN3M_DIR/weight_test.npz}"
export REFERENCE_PT="${REFERENCE_PT:-$TRAIN_PT}"
export REFERENCE_WEIGHTS="${REFERENCE_WEIGHTS:-$GEN_WEIGHT_TRAIN}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$GEN_WEIGHT_TRAIN}"

for train3m_input in \
  "$TRAIN_PT" "$GEN_WEIGHT_TRAIN" "$HELDOUT_PT" "$HELDOUT_WEIGHTS"; do
  if [[ ! -s "$train3m_input" ]]; then
    echo "ERROR: train3M input is missing: $train3m_input" >&2
    return 2 2>/dev/null || exit 2
  fi
done
