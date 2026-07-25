# NURD HLT Anomaly Detection

This branch trains a two-axis HLT anomaly detector and evaluates ABCD closure
without using the test sample to define the latent-space score or select the
ABCD thresholds.

- Axis 1: object-feature autoencoder reconstruction loss.
- Axis 2: a calibrated all-background anomaly score from the PF-candidate
  contrastive encoder.
- Learned background classes: `DY=0`, `QCD=1`, `TT=2`, and `WJets=3`.
- Primary closure report: QCD events in the independent test file.

The recommended workflow is implemented on branch `experimental`.

## What Changed From `main`

The changes are cumulative. They include the Della workflow developed on the
earlier `wip-mila`, `wip-mila-test`, and `wip-mila-all-baselines` branches, plus
the final score/evaluation contract introduced on `experimental`.

### Data And Memory

- AE reconstruction scores are computed once before the train/validation split.
- NURD weights and AE nuisance values are stored in the dataset instead of being
  recomputed in the training loop.
- The AE scaler saved during AE training is reused by NURD training and eval.
- Validation no longer drops its final batch.
- The projector is evaluated only when the contrastive loss needs it.
- Critic updates reuse detached encoder activations instead of running extra
  Transformer forwards.
- DataLoader workers default to zero, training uses at most 64 GB CPU memory,
  and full training requires an A100 with at least 75 GiB VRAM.

### NURD And Closure Training

- AE nuisance bins are defined separately inside each background class.
  Therefore bin 9 means the high-AE tail of that class, rather than one global
  AE-loss interval dominated by a different class.
- Per-class quantile bins intentionally make the discrete class/bin marginals
  close to independent, so the exact NURD weights may be close to one. The
  conditional critic and continuous closure losses then carry most of the
  decorrelation work; `w_cv` and `w_ess` make this visible in the logs.
- Ten nuisance bins do not impose a 10-bin limit on the final closure result.
  The direct distance-correlation, profile, and soft ABCD losses operate on
  continuous AE and MD proxy values.
- The density-ratio critic compares real `(latent, class, AE-bin)` triples with
  triples whose AE bin is shuffled within the same class. This targets
  conditional dependence between the representation and AE loss without letting
  the critic win from class-prior differences.
- The encoder uses a bounded critic-confusion penalty. Critic accuracy and
  chance-normalized cross entropy are logged as diagnostics; they are not
  treated as proof of closure.
- Classification and supervised contrastive losses still train on all four
  backgrounds. QCD-only closure evaluation does not turn the encoder into a
  QCD-only model.
- Direct closure regularization uses per-class EMA Mahalanobis references and
  combines correlation, distance-correlation, profile-flatness, reverse-profile,
  and soft tail-ABCD terms.
- `experimental` adds a smooth all-class union score to training. The default
  loss includes:
  - auxiliary own-class MD closure for all four backgrounds;
  - union-score closure on QCD, matching the final “unusual to every known
    background” use case.
- Contrastive and closure weights are scheduled so class structure forms before
  the full closure penalty is applied.
- `checkpoint_main_*` is selected by validation loss.
  `checkpoint_abcd.pth.tar` is selected by a validation closure score combining
  the QCD union score with auxiliary per-class closure, subject to a validation
  loss tolerance.

### Final Score And Evaluation

`ABCD_SCOPE` and `SCORE_MODE` are independent:

- `ABCD_SCOPE` selects the events on which thresholds and closure are measured.
- `SCORE_MODE` defines axis 2 for every event.

The recommended combination is:

```text
ABCD_SCOPE=qcd
SCORE_MODE=calibrated_union
```

Evaluation now has three disjoint roles:

1. The training portion of `hlt_smcocktail_train.pt` fits a Ledoit-Wolf
   shrinkage mean/covariance reference for each background class.
2. Half of the original model-validation split calibrates each class MD into an
   empirical tail probability.
3. The other half of model validation selects ABCD thresholds. The complete
   `hlt_smcocktail_test.pt` is opened only for the final report.

For class `c`, evaluation computes `p_c`, the probability that a reference event
from class `c` has an MD at least as large as the event being scored. The default
axis is:

```text
calibrated_union = -log(max_c p_c)
```

It is large only when the event is atypical for every learned background.
Unlike raw min-MD, class covariance scales and priors cannot make one class win
only because its distances have a different numerical scale.

Evaluation also saves:

- raw MD, empirical tail probability, and Gaussian NLL for every class;
- classifier, Gaussian-likelihood, and calibrated-typicality class assignments;
- assignment confusion matrices on known test backgrounds;
- QCD, all-background, and per-class Pearson/Spearman/distance correlation;
- selected-point closure, closure uncertainty, grid median/p90, and closure
  curve;
- `event_scores.npz`, `diagnostics.json`, `abcd_thresholds.json`, and plots.

The older scores remain available for controlled comparisons:

```text
qcd_md, min_md, mixture_nll, classifier_routed, gaussian_routed
```

## Della Setup

Use code in home and large files in scratch:

```bash
module load anaconda3/2025.12
conda activate disco

cd /home/mb7126/nurd_hlt
git fetch origin
git switch experimental
git pull --ff-only

export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
mkdir -p $BASE/{checkpoints,logs,outputs,wandb,matplotlib}
```

Verify the environment:

```bash
python -c "import torch, wandb, numpy, sklearn, scipy, matplotlib; print('imports ok'); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Do not install the old full `requirements.txt` blindly on Della. Install only a
missing HLT package in the active environment. Full training requires a
CUDA-enabled PyTorch build.

Verify all three data files:

```bash
export TRAIN_PT=$BASE/data/hlt_smcocktail_train.pt
export TEST_PT=$BASE/data/hlt_smcocktail_test.pt
export SIGNAL_PT=$BASE/data/hlt_signal_TpTp.pt

python -c "import torch; x=torch.load('$TRAIN_PT',map_location='cpu'); print(x.keys()); print(x['pf'].shape,x['obj'].shape,x['label'].shape); print(torch.unique(x['label'],return_counts=True))"
python -c "import torch; x=torch.load('$TEST_PT',map_location='cpu'); print(x.keys()); print(x['pf'].shape,x['obj'].shape,x['label'].shape)"
python -c "import torch; x=torch.load('$SIGNAL_PT',map_location='cpu'); print(x.keys()); print(x['pf'].shape,x['obj'].shape,x['label'].shape)"
```

Before every submission, clear stale exported choices. Slurm inherits the
current shell environment.

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG
unset WANDB_RUN_NAME WANDB_RUN_ID NURD_GLOB
unset ABCD_SCOPE SCORE_MODE MIN_MD
```

## Smoke Test

Run this after pulling:

```bash
sbatch slurm/submit_smoke.sbatch
```

Monitor it:

```bash
JOB=<job_id>
squeue -j $JOB
tail -f $BASE/logs/nurd_smoke-$JOB.out
tail -f $BASE/logs/nurd_smoke-$JOB.err
```

Success requires both:

```text
SMOKE DONE
Critic scope: baselines
```

## Full Training

The recommended campaign trains the AE for 100 epochs and NURD for 150 epochs.
The AE has consistently converged by 100 epochs; the NURD schedule uses the
longer run. The job has a hard 10-hour Slurm limit.

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG
unset WANDB_RUN_NAME WANDB_RUN_ID NURD_GLOB
sbatch slurm/submit_train.sbatch
```

The default job requests:

```text
1 x A100 with at least 75 GiB
8 CPU cores
64 GB CPU memory
10:00:00 wall time
batch size 4096
AE epochs 100
NURD epochs 150
```

Monitor:

```bash
JOB=<training_job_id>
squeue -j $JOB -o "%.18i %.9P %.24j %.8T %.10M %.20R"
tail -f $BASE/logs/nurd_hlt_train-$JOB.out
tail -f $BASE/logs/nurd_hlt_train-$JOB.err
```

The `.out` file must show these defaults:

```text
CRITIC_SCOPE=baselines
CRITIC_TYPE=density_ratio
CRITIC_SHUFFLE=within_label
NUISANCE_BIN_SCOPE=per_class
CLOSURE_SCOPE=baselines
CLOSURE_SCORE_MODE=hybrid
CLOSURE_UNION_SCOPE=qcd
AE_EPOCHS=100
NURD_EPOCHS=150
```

The run has finished only when the output contains:

```text
TRAINING DONE
```

Find the exact experiment names later:

```bash
grep -E '^AE_EXP=|^NURD_EXP=' $BASE/logs/nurd_hlt_train-$JOB.out
```

Outputs:

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

Do not increase `NUM_WORKERS` or Slurm memory without measuring it first.
For lower VRAM, use `BATCH_SIZE=3072`; this changes optimization statistics and
should be treated as a separate experiment.

## Recommended Evaluation

The latest-eval script searches only experiment names beginning with
`hlt_nurd_closure_bs4096_experimental_`, preventing an older branch from being
selected accidentally.

Evaluate the latest validation-loss checkpoint:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
unset NURD_GLOB MIN_MD
ABCD_SCOPE=qcd SCORE_MODE=calibrated_union \
  sbatch slurm/submit_eval_latest.sbatch
```

Evaluate the closure-selected checkpoint:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
unset NURD_GLOB MIN_MD
PREFER_ABCD_CKPT=1 ABCD_SCOPE=qcd SCORE_MODE=calibrated_union \
  sbatch slurm/submit_eval_latest.sbatch
```

Monitor:

```bash
JOB=<eval_job_id>
squeue -j $JOB
tail -f $BASE/logs/nurd_eval_latest-$JOB.out
tail -f $BASE/logs/nurd_eval_latest-$JOB.err
```

The log prints the exact `CKPT`, `AE_CKPT`, `REFERENCE_PT`, `SCORE_MODE`, and
`Results` path. Inspect:

```bash
OUT=$(grep '^Results:' $BASE/logs/nurd_eval_latest-$JOB.out | sed 's/^Results: //')
ls -lh "$OUT"
ls -lh "$OUT/plots"
cat "$OUT/diagnostics.json"
cat "$OUT/abcd_thresholds.json"
```

The final result is:

```text
diagnostics.json
  -> abcd_selection
  -> report_at_selected
  -> ratio
  -> nonclosure
  -> ratio_unc
```

Also compare:

```text
abcd_grid.median_abs_nonclosure
abcd_grid.p90_abs_nonclosure
closure_curve
correlations.qcd
signal_at_selected
class_assignment
```

Do not rank models from `tune_best` or `report_best_for_reference`. The first is
the threshold-selection sample; the second scans the final test report and is
diagnostic only.

## Controlled Score And Scope Comparisons

Keep the test population QCD and change only the score:

```bash
ABCD_SCOPE=qcd SCORE_MODE=qcd_md \
  WANDB_NAME_PREFIX=experimental_qcd_md \
  sbatch slurm/submit_eval_latest.sbatch

ABCD_SCOPE=qcd SCORE_MODE=min_md \
  WANDB_NAME_PREFIX=experimental_raw_min_md \
  sbatch slurm/submit_eval_latest.sbatch

ABCD_SCOPE=qcd SCORE_MODE=mixture_nll \
  WANDB_NAME_PREFIX=experimental_mixture_nll \
  sbatch slurm/submit_eval_latest.sbatch
```

Keep the calibrated union score and change only the closure population:

```bash
ABCD_SCOPE=all_baselines SCORE_MODE=calibrated_union \
  MIN_A_FRAC=0.02 SELECTION_STAT_WEIGHT=0.5 \
  WANDB_NAME_PREFIX=experimental_all_baselines \
  sbatch slurm/submit_eval_latest.sbatch
```

To evaluate a specific run, set it explicitly:

```bash
NURD_EXP=<exact_nurd_exp>
CKPT=$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_abcd.pth.tar
AE_CKPT=$BASE/checkpoints/hlt/hlt/<exact_ae_exp>/checkpoint_ae.pth
OUTDIR=$BASE/outputs/manual_${NURD_EXP}_calibrated_union_qcd

CKPT="$CKPT" AE_CKPT="$AE_CKPT" OUTDIR="$OUTDIR" \
ABCD_SCOPE=qcd SCORE_MODE=calibrated_union \
  sbatch slurm/submit_eval_latest.sbatch
```

## W&B

Jobs run offline by default. Evaluation prints the exact sync command. To sync
one completed run from a login node:

```bash
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)
wandb sync <offline-run-directory-printed-by-the-job>
```

Set `SYNC_WANDB=1` on an eval submission only when compute nodes can reach W&B.
Offline mode is the reliable default.

## Interpretation

- QCD closure with `calibrated_union` asks: among true QCD events, is AE loss
  independent enough from the score “unusual to every known background” for
  sidebands to predict region A?
- Class assignment asks a different question: which known background reference
  best explains the event? Use the saved classifier, Gaussian, and typicality
  routes; raw MD argmin alone is not a calibrated class probability.
- A low critic accuracy is necessary evidence that the adversary is confused,
  but continuous correlation, closure grids, and independent test closure are
  the deciding diagnostics.
- A single working point can close by statistical accident. Quote its
  uncertainty together with the closure grid and curve.
