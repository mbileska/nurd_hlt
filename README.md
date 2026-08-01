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

- The legacy campaign uses `hlt_smcocktail_train.pt` and
  `hlt_smcocktail_test.pt`. The new Mequinna campaign is stored separately under
  `data/mequinna_1M_noZB/`; never overwrite the legacy files because that makes
  checkpoint provenance ambiguous.
- The Mequinna release also provides event-aligned `weight_train.pt` and
  `weight_test.pt` generator weights. These are physics generator weights, not
  the NURD label/nuisance weights described below.
- Generator weights are now required by the default Mequinna Slurm campaign.
  They are applied consistently to AE and NURD losses, nuisance-bin quantiles,
  NURD frequency estimates, critic/closure objectives, latent references,
  validation checkpoint selection, and final ABCD yields.
- AE reconstruction is computed once. The train/validation split is made before
  fitting nuisance preprocessing.
- Twenty generator-weighted AE-loss nuisance bins are fitted from training QCD only, then the same
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
- Natural shuffled batches are used by default; QCD-rich sampling remains an
  explicit optional experiment.
- A QCD density-ratio critic distinguishes real `(latent, AE-bin)` pairs from
  shuffled pairs. It takes one update per selected batch and is weighted by the
  physical generator measure, but not by NURD label/nuisance weights.
- The encoder uses the bounded `ratio_to_one` objective, making the learned
  density ratio approach one without an unbounded adversarial CE objective.
- Direct QCD closure regularization acts on continuous AE loss and QCD MD. It
  combines weighted log-correlation, distance correlation, and forward profile
  flatness. The higher-variance v5 reverse-profile/copula terms remain optional.
- The QCD MD proxy is a real online EMA. A batch is scored against the previous
  detached reference before that batch updates the weighted moments, avoiding
  self-scoring while tracking the changing encoder.
- Contrastive weight decreases from `0.15` to `0.05` over 40 epochs. Closure
  weight increases from `0` to `0.5` over 15 epochs.

### Selection And Evaluation

- `checkpoint_main_*` is selected by validation NURD loss.
- Validation QCD MD is two-fold cross-fitted with weighted shrinkage covariance,
  matching final evaluation geometry while ensuring no validation event helps
  define its own MD. Closure checkpoints use a five-epoch rolling median.
- `checkpoint_abcd.pth.tar` and `checkpoint_closure.pth.tar` are selected by a
  broad cross-fitted QCD MD score: p90 plus tail and median log-nonclosure,
  subject to a validation-loss tolerance.
- Final threshold selection uses the untouched model-validation portion of the
  training file. The complete test file is report-only.
- ABCD cuts use weighted quantiles. `A/B/C/D` are generator-weighted yields,
  ratio uncertainty uses per-region `sumw2`, and raw event-count minima are
  retained as a guard against a few high-weight events.
- For QCD MD, all untouched model-validation QCD tune thresholds because this
  score does not use empirical tail calibration. The independent test file
  remains report-only.
- The default scan requires at least 10% of tuning QCD in region A, at least 1%
  in every region, and at most 5% propagated ratio uncertainty. Selection uses
  five-fold and neighboring-grid stability instead of the closure of one cell.
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
nuisance bins         20 weighted quantiles, fitted on training QCD
training batches      natural shuffled all-background batches
critic                QCD density-ratio critic, one update per batch
critic penalty        bounded ratio_to_one
closure               weighted QCD dCorr/profile vs online EMA QCD MD
checkpoint MD         two-fold cross-fitted weighted shrinkage covariance
ABCD yields           generator weighted, uncertainty from sumw2
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
CRITIC_BIN_RESOLUTIONS=20
CRITIC_PENALTY_TYPE=ratio_to_one
CRITIC_SHUFFLE=global
N_BINS=20
QCD_BATCH_FRACTION=0.0
NUISANCE_BIN_SCOPE=qcd
CLOSURE_SCOPE=qcd
CLOSURE_SCORE_MODE=own_class
NURD_EPOCHS=200
MD_PROXY_TYPE=ema
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
NURD_GLOB=hlt_nurd_closure_bs4096_experimental_qcd_weighted_v6_*
PREFER_ABCD_CKPT=1
ABCD_SCOPE=qcd
SCORE_MODE=qcd_md
MIN_A_FRAC=0.10
MIN_REGION_FRAC=0.01
MAX_RATIO_UNC=0.05
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
  WANDB_NAME_PREFIX=experimental_qcd_weighted_v6_main \
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
