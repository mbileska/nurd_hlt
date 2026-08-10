# NURD HLT Anomaly Detection

This branch trains a two-axis HLT anomaly detector:

- Axis 1: object-feature autoencoder reconstruction loss.
- Axis 2 for the primary closure result: a frozen conditional-CDF transform of
  QCD-referenced, whitened Mahalanobis distance from the PF-candidate encoder.
  Raw QCD MD is saved and evaluated beside the transformed score.
- Learned backgrounds: `DY=0`, `QCD=1`, `TT=2`, and `WJets=3`.
- Primary report: ABCD closure on independent QCD test events.

Classification, supervised contrastive learning, exact NURD reweighting, and
the nuisance critic use all four backgrounds. Only the direct second-stage
closure objective and primary ABCD report are QCD-scoped.

## What Changed From `main`

### Data And Weights

- The legacy campaign uses `hlt_smcocktail_train.pt` and
  `hlt_smcocktail_test.pt`. The new Mequinna campaign is stored separately under
  `data/mequinna_1M_noZB/`; never overwrite the legacy files because that makes
  checkpoint provenance ambiguous.
- The Mequinna release also provides event-aligned `weight_train.pt` and
  `weight_test.pt` generator weights. These are physics generator weights, not
  the NURD label/nuisance weights described below.
- Generator weights are required by the default Mequinna Slurm campaign. The
  AE and final physics yields use the physical measure directly. NURD training
  preserves generator weights within each class but normalizes the four class
  masses equally, preventing the 99% QCD cross-section prior from turning the
  classifier into an always-QCD predictor.
- The campaign does not truncate the new files: 90% of every Mequinna training
  event trains the AE/NURD models, the remaining stratified 10% selects
  checkpoints and ABCD thresholds, and every independent test event is used
  once for the final report. Test events never enter optimization or reference
  fitting.
- AE reconstruction is computed once. The train/validation split is made before
  fitting nuisance preprocessing.
- Fifty generator-weighted AE-loss nuisance bins are fitted from all training
  backgrounds for exact NURD reweighting, matching the original global NURD
  factorization. A separate continuous all-background weighted CDF coordinate
  is used by the critic, so critic resolution is not limited by bin count.
  Both are fitted on training data only and reused for validation.
- Exact NURD weights are fitted on training data only and reused unchanged for
  validation.
- Weight clipping now preserves both the configured cap and a sample-weighted
  training mean of one.
- Nuisance edges and the fitted weight table are stored in every NURD
  checkpoint.

### Training Objective

- All-background classifier and supervised contrastive losses are unchanged in
  scope: all four backgrounds teach the encoder their structure.
- A weighted sampler draws equal class mass while preserving the physical
  generator-weight distribution within each class. The same number of events
  and Transformer batches are processed per epoch as before.
- A continuous all-background density-ratio critic distinguishes real
  `(latent, class, weighted-AE-CDF)` tuples from shuffled-nuisance tuples. It
  takes two updates on reused encoder activations.
- The critic and encoder-side critic penalty now use the exact NURD weights.
  The prior class-balanced sampler previously caused them to use unit weights,
  so critic confusion did not certify independence under the randomized NURD
  measure.
- The encoder uses the bounded `ratio_to_one` objective, making the learned
  density ratio approach one without an unbounded adversarial CE objective.
- Supervised contrastive learning weights both anchors and comparison events by
  exact NURD weights after sampling; generator weights are not multiplied a
  second time.
- Direct QCD closure regularization acts on continuous AE loss and QCD MD. The
  default restores the best-performing v3/v4 hybrid objective: weighted
  log-correlation, distance correlation, forward/reverse profile flatness, and
  a soft tail copula grid. The fixed-size dCorr calculation now samples from
  the physical generator measure instead of retaining extreme weights on a
  uniform 512-row subsample.
- A second continuous distance-correlation term acts on the full six-dimensional
  QCD latent, not only its EMA-MD radius. This removes AE information that a
  downstream Mahalanobis score could recover even when scalar proxy correlation
  is small.
- Training is explicitly staged. The first 140 epochs learn an all-background
  representation with global NURD. The final 40 epochs retain those objectives
  and add QCD closure regularization at a lower learning rate.
- QCD closure batches use one epoch-frozen MD reference produced by the prior
  frozen validation pass. This replaces statistics accumulated while the
  encoder was changing and aligns the proxy more closely with evaluation.
- Fine-tuning includes a small parameter anchor to the representation
  checkpoint. Closure checkpoints are rejected if balanced accuracy drops by
  more than 3 percentage points or any class drops by more than 5 points.
- Contrastive weight decreases from `0.15` to `0.02` during representation.
  Closure is zero in that stage and ramps from `0` to `0.5` during fine-tuning.

### Selection And Evaluation

- `checkpoint_main_*` is selected by class-balanced validation NURD loss.
- No main or closure checkpoint is accepted below 55% physical balanced
  accuracy or 25% accuracy in any individual background class. This explicitly
  prevents both the v6/v7 always-QCD collapse and a hidden single-class collapse.
- Validation QCD MD is two-fold cross-fitted with weighted shrinkage covariance.
  Fold assignment balances generator-weight mass, so a few large-weight events
  cannot make one reference fold statistically much weaker than the other.
- `checkpoint_abcd.pth.tar` and `checkpoint_closure.pth.tar` are selected by a
  broad cross-fitted QCD MD score: p90 plus tail and median log-nonclosure,
  subject to validation-loss, effective-region-count, and propagated-ratio-
  uncertainty guards. Selection starts after epoch 40 and saves the exact epoch
  that produced the improving score; an older rolling-median implementation
  could save a different current state than the historical score described.
- Final threshold selection uses the untouched model-validation portion of the
  training file. The complete test file is report-only.
- ABCD cuts use weighted quantiles. `A/B/C/D` are generator-weighted yields,
  ratio uncertainty uses per-region `sumw2`, and raw event-count minima are
  retained as a guard against a few high-weight events.
- The primary `qcd_conditional_cdf` evaluation divides untouched training-file
  validation events into disjoint calibration and threshold-tuning halves. The
  calibration half fits weighted QCD `P(MD <= m | AE=x)`; the transform is then
  frozen, thresholds are selected on the other half, and the independent test
  file is report-only. This targets residual nonlinear/heteroscedastic closure
  without fitting the report sample.
- Evaluation saves `conditional_cdf_calibration.npz`, calibrated and raw event
  scores, raw and calibrated correlations/grids, and an overlaid raw closure
  curve. `SCORE_MODE=qcd_md` remains the untransformed apples-to-apples check.
- The default scan requires at least 10% of tuning QCD in region A, at least 1%
  in every region, and at most 15% propagated ratio uncertainty. The looser
  uncertainty ceiling is necessary for generator-weighted samples whose
  effective event count is much smaller than the raw row count; uncertainty is
  still penalized in threshold selection and reported. Five stability folds are
  balanced by generator-weight mass rather than raw rows.
- Evaluation trains a fresh nonlinear nuisance auditor after freezing the
  encoder. Training and metrics now follow generator-weighted QCD rather than
  raw row counts; its test-QCD accuracy/AUC/CE diagnose residual AE information.
- `calibrated_union`, `min_md`, and other all-background scores remain available
  as secondary studies. They are not the default QCD-closure axis.

These changes are intended to improve broad QCD closure, not guarantee a
particular result. Rank models from independent `report_at_selected`, the
closure curve, and grid median/p90 together, not from a tuned point alone.

## One-Time Della Setup

```bash
module load anaconda3/2025.12
conda activate disco

cd /home/mb7126/nurd_hlt
git fetch origin
git switch experimental
git pull --ff-only

export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
mkdir -p "$BASE"/{checkpoints,logs,outputs,wandb,matplotlib}
```

Verify the environment:

```bash
python -c "import torch, wandb, numpy, sklearn, scipy, matplotlib; print('imports ok'); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Mequinna Dataset

Keep the new release in a versioned scratch directory:

```text
$BASE/data/mequinna_1M_noZB/hlt_smcocktail_mequinna_train.pt
$BASE/data/mequinna_1M_noZB/hlt_smcocktail_mequinna_test.pt
$BASE/data/mequinna_1M_noZB/weight_train.pt
$BASE/data/mequinna_1M_noZB/weight_test.pt
```

The CERN source is:

```text
/eos/user/e/escheull/smcocktail_1M_noZB/
```

If the files are not already on Della, run from a Della login node:

```bash
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
export DATA_DIR=$BASE/data/mequinna_1M_noZB
export EOS_DIR=/eos/user/e/escheull/smcocktail_1M_noZB
mkdir -p "$DATA_DIR"

rsync -ahP mbileska@lxplus.cern.ch:$EOS_DIR/hlt_smcocktail_mequinna_train.pt "$DATA_DIR/"
rsync -ahP mbileska@lxplus.cern.ch:$EOS_DIR/hlt_smcocktail_mequinna_test.pt "$DATA_DIR/"
rsync -ahP mbileska@lxplus.cern.ch:$EOS_DIR/genweight_lookup/weight_train.pt "$DATA_DIR/"
rsync -ahP mbileska@lxplus.cern.ch:$EOS_DIR/genweight_lookup/weight_test.pt "$DATA_DIR/"
```

Validate event alignment, finite values, signs, total weight, and effective
sample size with the same loader used by training:

```bash
export DATA_DIR=$BASE/data/mequinna_1M_noZB
python - <<'PY'
import os
import torch
from utils.event_weights import load_event_weights

root = os.environ["DATA_DIR"]
for split in ("train", "test"):
    sample = torch.load(
        os.path.join(root, f"hlt_smcocktail_mequinna_{split}.pt"),
        map_location="cpu",
    )
    print(f"\n{split}: keys={list(sample)}")
    for key, value in sample.items():
        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
    _, metadata = load_event_weights(
        os.path.join(root, f"weight_{split}.pt"), sample)
    print("  generator weights:", metadata)
PY
```

Negative weights are rejected with a clear error because these positive
weighted classification and closure losses do not implement signed-weight
statistics. The smoke test now validates the real generator-weight path:

```bash
sbatch slurm/submit_smoke.sbatch
```

Verify the legacy data when reproducing an older campaign:

```bash
export TRAIN_PT=$BASE/data/hlt_smcocktail_train.pt
export TEST_PT=$BASE/data/hlt_smcocktail_test.pt
export SIGNAL_PT=$BASE/data/hlt_signal_TpTp.pt

python -c "import torch; x=torch.load('$TRAIN_PT',map_location='cpu'); print(x.keys()); print(x['pf'].shape,x['obj'].shape,x['label'].shape); print(torch.unique(x['label'],return_counts=True))"
python -c "import torch; x=torch.load('$TEST_PT',map_location='cpu'); print(x.keys()); print(x['pf'].shape,x['obj'].shape,x['label'].shape)"
python -c "import torch; x=torch.load('$SIGNAL_PT',map_location='cpu'); print(x.keys()); print(x['pf'].shape,x['obj'].shape,x['label'].shape)"
```

Do not install the repository's old full `requirements.txt` over a working
CUDA environment. Install only a package that is actually missing.

Before submissions, remove inherited overrides:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG NURD_GLOB
unset WANDB_RUN_NAME WANDB_RUN_ID ABCD_SCOPE SCORE_MODE MIN_MD
unset PREFER_ABCD_CKPT PREFER_CLOSURE_CKPT
unset TRAIN_PT TEST_PT REFERENCE_PT GEN_WEIGHT_TRAIN TEST_WEIGHTS REFERENCE_WEIGHTS
unset REPRESENTATION_EPOCHS QCD_FINETUNE_EPOCHS FINETUNE_LR_MULTIPLIER
unset FINETUNE_ANCHOR_WEIGHT CONDITIONAL_CDF_BINS CONDITIONAL_CDF_QUANTILES
```

## Smoke Test

Submit:

```bash
sbatch slurm/submit_smoke.sbatch
```

Inspect:

```bash
JOB=<smoke_job_id>
squeue -j "$JOB"
tail -f "$BASE/logs/nurd_smoke-$JOB.out"
tail -f "$BASE/logs/nurd_smoke-$JOB.err"
```

Success requires `SMOKE DONE` and no traceback. The smoke test is only a
one-epoch integration check; the full staged settings are verified in the full
training log header.

## Full Training

Submit the default campaign:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG NURD_GLOB
unset WANDB_RUN_NAME WANDB_RUN_ID
sbatch slurm/submit_train.sbatch
```

Defaults:

```text
wall time             12:00:00 hard limit
GPU                   1 A100 with at least 75 GiB
CPU                   8 cores, 48 GiB RAM
batch size            4096
AE epochs             100
NURD epochs           180: 140 representation + 40 QCD fine-tune
nuisance bins         50 weighted all-background quantiles for exact NURD weights
continuous nuisance   weighted all-background AE-loss CDF in [0,1]
training measure      equal class mass; physical generator weights within class
critic                continuous global density-ratio critic, exact NURD weighted
critic penalty        bounded ratio_to_one
closure               QCD hybrid losses vs prior-validation frozen MD reference
fine-tune guard        parameter anchor plus relative balanced/per-class accuracy gates
checkpoint guard      balanced accuracy >= 0.55 and every class >= 0.25
checkpoint MD         weight-balanced two-fold cross-fitted shrinkage covariance
ABCD yields           generator weighted, uncertainty from sumw2
```

The 180-epoch split leaves margin inside the 12-hour hard limit on an 80 GiB
A100. Slurm terminates the job at 12 hours; it cannot consume a GPU indefinitely.

Monitor:

```bash
JOB=<training_job_id>
squeue -j "$JOB" -o "%.18i %.9P %.24j %.8T %.10M %.20R"
tail -f "$BASE/logs/nurd_hlt_train-$JOB.out"
tail -f "$BASE/logs/nurd_hlt_train-$JOB.err"
```

The `.out` header should contain:

```text
CRITIC_SCOPE=all
CRITIC_TYPE=continuous_density_ratio
CRITIC_BIN_RESOLUTIONS=50
CRITIC_PENALTY_TYPE=ratio_to_one
CRITIC_SHUFFLE=global
LAMBDA=0.05
N_BINS=50
TRAINING_MEASURE=class_balanced_physical
QCD_BATCH_FRACTION=0.0
NUISANCE_BIN_SCOPE=all
CLOSURE_SCOPE=qcd
CLOSURE_SCORE_MODE=own_class
CLOSURE_LATENT_DCORR_WEIGHT=1.5
NURD_EPOCHS=180
REPRESENTATION_EPOCHS=140
QCD_FINETUNE_EPOCHS=40
FINETUNE_LR_MULTIPLIER=0.25
FINETUNE_ANCHOR_WEIGHT=0.01
MD_PROXY_TYPE=epoch
CLOSURE_LOSS_TYPE=hybrid
CLOSURE_WEIGHT=0.5
CONTRAST_WEIGHT=0.02
MIN_VAL_BALANCED_ACC=0.55
MIN_VAL_CLASS_ACC=0.25
```

Training is complete only when the output contains `TRAINING DONE`.

Recover the exact experiment names:

```bash
grep -E '^AE_EXP=|^NURD_EXP=' "$BASE/logs/nurd_hlt_train-$JOB.out"
```

Checkpoints are written to:

```text
$BASE/checkpoints/hlt/hlt/<AE_EXP>/checkpoint_ae.pth
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/checkpoint_main_*.pth.tar
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/checkpoint_representation.pth.tar
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/checkpoint_abcd.pth.tar
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/checkpoint_closure.pth.tar
```

To reuse a completed AE:

```bash
unset CKPT OUTDIR NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
AE_EXP=<exact_ae_exp> SKIP_AE=1 sbatch slurm/submit_train.sbatch
```

Do not increase `NUM_WORKERS`, CPU memory, or batch size without measuring the
result. `BATCH_SIZE=3072` is the fallback for lower VRAM, but it is a distinct
optimization experiment.

## Primary QCD Evaluation

The latest-eval script defaults to:

```text
NURD_GLOB=hlt_nurd_closure_bs4096_experimental_global_nurd_calibrated_v9_*
PREFER_ABCD_CKPT=1
ABCD_SCOPE=qcd
SCORE_MODE=qcd_conditional_cdf
CONDITIONAL_CDF_BINS=20
CONDITIONAL_CDF_QUANTILES=257
MIN_A_FRAC=0.10
MIN_REGION_FRAC=0.01
MAX_RATIO_UNC=0.15
SELECTION_FOLDS=5
SELECTION_STAT_WEIGHT=0.5
SELECTION_NEIGHBOR_WEIGHT=1.0
SCAN_PERCENT_MAX=0.98
```

Evaluate the latest closure-selected checkpoint:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG NURD_GLOB
unset WANDB_RUN_NAME WANDB_RUN_ID ABCD_SCOPE SCORE_MODE
unset PREFER_ABCD_CKPT PREFER_CLOSURE_CKPT
sbatch slurm/submit_eval_latest.sbatch
```

Evaluate the latest validation-loss checkpoint for comparison:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP NURD_GLOB
PREFER_ABCD_CKPT=0 PREFER_CLOSURE_CKPT=0 \
  WANDB_NAME_PREFIX=experimental_global_nurd_calibrated_v9_main \
  sbatch slurm/submit_eval_latest.sbatch
```

Monitor and find the output directory:

```bash
JOB=<eval_job_id>
squeue -j "$JOB"
tail -f "$BASE/logs/nurd_eval_latest-$JOB.out"
tail -f "$BASE/logs/nurd_eval_latest-$JOB.err"

OUT=$(grep '^Results:' "$BASE/logs/nurd_eval_latest-$JOB.out" | sed 's/^Results: //')
ls -lh "$OUT"
ls -lh "$OUT/plots"
cat "$OUT/diagnostics.json"
```

Use these fields to compare models:

```text
abcd_selection.report_at_selected.ratio
abcd_selection.report_at_selected.nonclosure
abcd_selection.report_at_selected.ratio_unc
abcd_grid.median_abs_nonclosure
abcd_grid.p90_abs_nonclosure
closure_curve
correlations.qcd
raw_qcd_md_correlations.qcd
raw_qcd_md_abcd
nuisance_auditor
signal_at_selected
```

Do not rank models from `tune_best` or `report_best_for_reference`: both are
optimization/reference diagnostics, not independent selected-point results.

## Evaluate A Specific Checkpoint

```bash
NURD_EXP=<exact_nurd_exp>
AE_EXP=<exact_ae_exp>

CKPT=$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_abcd.pth.tar
AE_CKPT=$BASE/checkpoints/hlt/hlt/$AE_EXP/checkpoint_ae.pth
OUTDIR=$BASE/outputs/manual_${NURD_EXP}_qcd_conditional_cdf

CKPT="$CKPT" AE_CKPT="$AE_CKPT" OUTDIR="$OUTDIR" \
ABCD_SCOPE=qcd SCORE_MODE=qcd_conditional_cdf \
  sbatch slurm/submit_eval_latest.sbatch
```

Run the raw legacy QCD-MD axis on the same checkpoint:

```bash
CKPT="$CKPT" AE_CKPT="$AE_CKPT" \
OUTDIR=$BASE/outputs/manual_${NURD_EXP}_raw_qcd_md \
ABCD_SCOPE=qcd SCORE_MODE=qcd_md \
WANDB_NAME_PREFIX=raw_qcd_md \
  sbatch slurm/submit_eval_latest.sbatch
```

For an older checkpoint without saved nuisance edges, add
`SKIP_NUISANCE_AUDITOR=1`.

## Secondary All-Background Studies

Keep QCD as the closure population and change only axis 2:

```bash
ABCD_SCOPE=qcd SCORE_MODE=calibrated_union \
  WANDB_NAME_PREFIX=experimental_calibrated_union \
  sbatch slurm/submit_eval_latest.sbatch

ABCD_SCOPE=qcd SCORE_MODE=min_md \
  WANDB_NAME_PREFIX=experimental_min_md \
  sbatch slurm/submit_eval_latest.sbatch

ABCD_SCOPE=qcd SCORE_MODE=mixture_nll \
  WANDB_NAME_PREFIX=experimental_mixture_nll \
  sbatch slurm/submit_eval_latest.sbatch
```

To measure closure on all known backgrounds:

```bash
ABCD_SCOPE=all_baselines SCORE_MODE=calibrated_union \
  WANDB_NAME_PREFIX=experimental_all_baselines \
  sbatch slurm/submit_eval_latest.sbatch
```

These are secondary questions. Quote the leakage-safe
`ABCD_SCOPE=qcd SCORE_MODE=qcd_conditional_cdf` result for the primary campaign,
and always report `SCORE_MODE=qcd_md` beside it as the raw model comparison.

## W&B

Jobs use offline W&B by default. Each evaluation log prints its exact sync
command. From a login node:

```bash
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)
wandb sync <offline-run-directory-printed-by-the-job>
```

Use `SYNC_WANDB=1` only when compute nodes can reach W&B. Offline mode avoids
training failures caused by network timeouts.
