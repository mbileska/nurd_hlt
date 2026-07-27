# NURD HLT Anomaly Detection

This branch trains a two-axis HLT anomaly detector:

- Axis 1: object-feature autoencoder reconstruction loss.
- Axis 2 for the primary closure result: QCD-referenced, PCA-whitened
  Mahalanobis distance from the PF-candidate encoder.
- Learned backgrounds: `DY=0`, `QCD=1`, `TT=2`, and `WJets=3`.
- Primary report: ABCD closure on independent QCD test events.

Classification and supervised contrastive learning still use all four
backgrounds. Only the nuisance-removal and closure objectives are QCD-scoped,
because the primary ABCD estimate is evaluated on QCD.

## What Changed From `main`

### Data And Weights

- AE reconstruction is computed once. The train/validation split is made before
  fitting nuisance preprocessing.
- Twenty AE-loss nuisance bins are fitted from training QCD only, then the same
  edges are applied to every training and validation event. Validation data
  cannot influence the nuisance definition.
- Exact NURD weights are fitted on training data only and reused unchanged for
  validation.
- Weight clipping now preserves both the configured cap and a sample-weighted
  training mean of one.
- Nuisance edges and the fitted weight table are stored in every NURD
  checkpoint.

### Training Objective

- All-background classifier and supervised contrastive losses are unchanged in
  scope: all four backgrounds teach the encoder their structure.
- The density-ratio critic is trained on QCD every batch. It distinguishes real
  `(latent, AE-bin)` pairs from globally shuffled QCD pairs.
- The encoder minimizes a uniform-target density-ratio penalty. Its minimum is
  a learned density ratio of one; unlike the old raw-logit objective, it is
  bounded below and invariant to a common logit shift.
- Direct QCD closure regularization acts on continuous AE loss and QCD MD. It
  combines log-correlation, distance correlation, forward/reverse profile
  flatness, and soft tail-ABCD terms.
- The QCD MD proxy uses lagged EMA first and second moments. It scores a batch
  with the previous reference and updates afterward. This fixes the old
  covariance averaging error and prevents the reference from absorbing the
  batch before it is scored.
- Contrastive weight decreases from `0.15` to `0.02` over 40 epochs. Closure
  weight increases from `0` to `1.0` over 15 epochs.

### Selection And Evaluation

- `checkpoint_main_*` is selected by validation NURD loss.
- `checkpoint_abcd.pth.tar` and `checkpoint_closure.pth.tar` are selected by a
  broad QCD MD closure score: p90 plus tail and median log-nonclosure, subject
  to a validation-loss tolerance.
- Final threshold selection uses the untouched model-validation portion of the
  training file. The complete test file is report-only.
- The default working-point scan requires at least 5% of tuning QCD in region A
  and penalizes statistical uncertainty and unstable neighboring grid points.
  This avoids selecting an apparently perfect but sparse fluctuation.
- Evaluation trains a fresh nonlinear nuisance auditor after freezing the
  encoder. Its test-QCD accuracy/AUC/CE diagnose residual AE-bin information.
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

Verify the data:

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

Success requires `SMOKE DONE`, no traceback, and `Critic scope: qcd`.

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
NURD epochs           200
nuisance bins         20, fitted on training QCD
critic                QCD density ratio, one update per batch
closure               QCD AE loss vs lagged QCD MD
```

The previous 150-epoch jobs finished in about 6.5 hours. Two hundred NURD
epochs give the lower learning-rate tail more time while retaining margin under
the 12-hour hard limit. Slurm terminates the job at 12 hours; it cannot consume
a GPU indefinitely.

Monitor:

```bash
JOB=<training_job_id>
squeue -j "$JOB" -o "%.18i %.9P %.24j %.8T %.10M %.20R"
tail -f "$BASE/logs/nurd_hlt_train-$JOB.out"
tail -f "$BASE/logs/nurd_hlt_train-$JOB.err"
```

The `.out` header should contain:

```text
CRITIC_SCOPE=qcd
CRITIC_TYPE=density_ratio
CRITIC_PENALTY_TYPE=ratio_to_one
CRITIC_SHUFFLE=global
NUISANCE_BIN_SCOPE=qcd
CLOSURE_SCOPE=qcd
CLOSURE_SCORE_MODE=own_class
NURD_EPOCHS=200
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
NURD_GLOB=hlt_nurd_closure_bs4096_experimental_qcd_v4_*
PREFER_ABCD_CKPT=1
ABCD_SCOPE=qcd
SCORE_MODE=qcd_md
MIN_A_FRAC=0.05
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
  WANDB_NAME_PREFIX=experimental_qcd_v4_main \
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
OUTDIR=$BASE/outputs/manual_${NURD_EXP}_qcd_md

CKPT="$CKPT" AE_CKPT="$AE_CKPT" OUTDIR="$OUTDIR" \
ABCD_SCOPE=qcd SCORE_MODE=qcd_md \
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

These are secondary questions. They should not replace the primary apples-to-
apples `ABCD_SCOPE=qcd SCORE_MODE=qcd_md` comparison.

## W&B

Jobs use offline W&B by default. Each evaluation log prints its exact sync
command. From a login node:

```bash
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)
wandb sync <offline-run-directory-printed-by-the-job>
```

Use `SYNC_WANDB=1` only when compute nodes can reach W&B. Offline mode avoids
training failures caused by network timeouts.
