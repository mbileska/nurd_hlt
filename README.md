# NURD HLT QCD closure

This branch contains the corrected weighted V4×4 campaign.

- Axis 1 is object-feature autoencoder reconstruction loss.
- Axis 2 is QCD-referenced, PCA-whitened Mahalanobis distance from the
  PF-candidate encoder.
- `DY=0`, `QCD=1`, `TT=2`, and `WJets=3` all train the classifier and
  supervised contrastive representation.
- Nuisance removal and the direct closure objective are QCD-scoped.
- The primary result is generator-weighted QCD closure on the independent
  Mequinna test sample.

## Fixed evaluation contract

Every Slurm evaluation runs exactly two protocols for the same NURD and AE
checkpoints:

```text
$BASE/outputs/<EVAL_NAME>/
├── held-out/   primary weighted Mequinna result
├── legacy/     origin/main historical QCD comparison
└── evaluation_summary.json
```

`held-out/` is the only primary result. Its latent reference, checkpoint
selection, threshold selection, and reporting events are disjoint:

```text
Mequinna training file
├── 90% reference fitting and model training
├──  5% checkpoint validation
└──  5% ABCD threshold tuning

Mequinna test file
└── 100% report-only held-out events
```

The AE and NURD use the same three training-file roles, balanced by
generator-weight mass within each background class. The threshold-tuning 5%
is used by neither training nor checkpoint selection. Exact row indices are
saved in both checkpoints and must match during NURD training and evaluation.

`legacy/` exactly reproduces the relevant `origin/main` behavior on
`$BASE/data/hlt_smcocktail_test.pt`: unweighted QCD-only MD is fitted on that
complete test sample, and the ABCD working point is optimized and reported on
the same QCD events. It is an oracle compatibility result, not a held-out
measurement.

Evaluation refuses to run when:

- the NURD experiment/checkpoint is ambiguous;
- the AE digest does not match the NURD checkpoint;
- the code commit differs from the training commit;
- the Mequinna reference sample or weight digest differs from training;
- held-out reference and report files are the same;
- the output directory is already non-empty.

The campaign also refuses to evaluate a classifier-loss fallback. Training
must produce `checkpoint_abcd.pth.tar` under the constrained broad-closure
selection rules; otherwise the training job fails and the dependent eval is
not launched.

## Weighted V4×4 model

V4 used 20 AE nuisance bins. This profile uses 80 generator-weighted QCD
quantile bins and does not add extra critics or critic resolutions.

The V4 architecture, loss coefficients, schedules, natural all-background
batches, one critic update per batch, and 200 epochs are retained. Required
weighted-data corrections are:

- generator weights define AE training, nuisance quantiles, NURD frequencies,
  physical QCD critic/closure objectives, references, and ABCD yields;
- the density-ratio critic's joint and shuffled samples use the same physical
  QCD measure;
- stochastic AE, CE, and SupCon losses use fixed full-training normalizers
  instead of biased random-batch self-normalization;
- the complete V4 closure objective is evaluated analytically with generator
  weights; physical resampling is disabled because that was a later V7
  experiment and adds avoidable stochastic variance;
- the intended V4 reverse profile is active with its conditioner gradient fixed;
- the EMA QCD reference decays weighted sufficient statistics, so influence is
  proportional to batch generator-weight mass;
- checkpoint MD is two-fold cross-fitted using the V4 covariance definition;
- closure checkpoint selection begins at epoch 20, uses a five-epoch rolling
  median, requires at least 20 effective events per ABCD region, and rejects
  propagated ratio uncertainty above 0.15;
- object features receive identical nonfinite-value handling in AE training,
  NURD training, and evaluation;
- unknown training arguments are fatal.

The final held-out threshold scan requires at least 10% of tuning QCD in A,
1% in every ABCD region, propagated ratio uncertainty below 15%, five-fold
physical-mass stability, and local grid stability.

## Della setup

```bash
module load anaconda3/2025.12
conda activate disco

cd ~/nurd_hlt
git fetch origin
git switch wip-mila-test
git pull --ff-only

export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
mkdir -p "$BASE"/{checkpoints,logs,outputs,wandb,matplotlib}
```

Required data:

```text
$BASE/data/mequinna_1M_noZB/hlt_smcocktail_mequinna_train.pt
$BASE/data/mequinna_1M_noZB/hlt_smcocktail_mequinna_test.pt
$BASE/data/mequinna_1M_noZB/weight_train.pt
$BASE/data/mequinna_1M_noZB/weight_test.pt
$BASE/data/hlt_smcocktail_test.pt
$BASE/data/hlt_signal_TpTp.pt              optional
```

Inspect the weight contract before spending GPU time:

```bash
python - <<'PY'
import os
import torch
from utils.event_weights import load_event_weights

root = os.path.join(os.environ["BASE"], "data", "mequinna_1M_noZB")
for split in ("train", "test"):
    sample = torch.load(
        os.path.join(root, f"hlt_smcocktail_mequinna_{split}.pt"),
        map_location="cpu",
    )
    _, metadata = load_event_weights(
        os.path.join(root, f"weight_{split}.pt"), sample)
    print(split, metadata)
PY
```

Negative or nonfinite weights and length/event-ID mismatches are fatal. A plain
weight tensor has no event IDs, so source-level row alignment cannot be proven;
the loader marks this as `alignment_verified: false` and checkpoints its exact
content digest.

## Smoke test

```bash
sbatch slurm/submit_smoke.sbatch
```

Success requires `SMOKE DONE`, a main checkpoint, and no traceback.

## Launch training and both evaluations

The recommended command generates one run tag in the login shell and submits
an evaluation with an `afterok` dependency on that exact training job:

```bash
cd ~/nurd_hlt
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
bash slurm/launch_v4x4_campaign.sh
```

The launcher prints the exact run tag, checkpoint directory, output directory,
training job ID, and evaluation job ID. Training has an 18-hour allocation so
a roughly 12-hour run is not killed at the boundary. Evaluation has eight hours
for both protocols.

Monitor the IDs printed by the launcher:

```bash
squeue -j <train_job>,<eval_job>
sacct -X -j <train_job>,<eval_job> \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit
```

Training is complete only when its log contains `TRAINING DONE`. Evaluation is
complete only when its log contains `DUAL QCD EVAL DONE`.

### Reuse an exact AE

Only reuse an AE produced by this contract from the same code commit, new
Mequinna sample, generator weights, and exact 90/5/5 row partition:

```bash
export SKIP_AE=1
export AE_CKPT=$BASE/checkpoints/hlt/hlt/<exact_ae_exp>/checkpoint_ae.pth
bash slurm/launch_v4x4_campaign.sh
```

The AE SHA256 is embedded in every NURD checkpoint and verified during both
evaluations.

## Launch both evaluations for an existing completed run

Use an exact experiment name. Wildcards, newest-directory fallback, and
main-checkpoint fallback are intentionally disabled:

```bash
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
export NURD_EXP=hlt_nurd_closure_bs4096_<exact_run_tag>
export EVAL_NAME=<exact_run_tag>_eval

sbatch --export=ALL,BASE="$BASE",NURD_EXP="$NURD_EXP",EVAL_NAME="$EVAL_NAME" \
  slurm/submit_eval_latest.sbatch
```

If the AE path saved in the NURD checkpoint is no longer resolvable, provide
the exact file explicitly:

```bash
export AE_CKPT=$BASE/checkpoints/hlt/hlt/<exact_ae_exp>/checkpoint_ae.pth
sbatch --export=ALL,BASE="$BASE",NURD_EXP="$NURD_EXP",EVAL_NAME="$EVAL_NAME",AE_CKPT="$AE_CKPT" \
  slurm/submit_eval_latest.sbatch
```

## Reading results

Start with:

```bash
cat "$BASE/outputs/$EVAL_NAME/evaluation_summary.json"
```

For the primary result, inspect:

```text
held-out/diagnostics.json
  abcd_selection.report_at_selected
  abcd_grid.median_abs_nonclosure
  abcd_grid.p90_abs_nonclosure
  closure_curve
  correlations.qcd
  nuisance_auditor
```

The selected working point is chosen only on the dedicated Mequinna threshold
split. `report_at_selected` applies that frozen point to the independent test
sample. `report_best_for_reference` is descriptive only and must not be quoted
as the held-out result.

For historical comparison, use `legacy/diagnostics.json` and remember that it
is deliberately same-sample and unweighted.

## W&B

Jobs default to offline W&B. Set `SYNC_WANDB=1` only when compute nodes can
reach W&B, or sync the two printed offline run directories later from a login
node.
