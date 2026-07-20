# NURD HLT Anomaly Detection

This repo trains and evaluates a two-axis HLT anomaly-detection analysis using
ABCD background estimation.

- Axis 1: autoencoder reconstruction loss from object-level event features.
- Axis 2: Mahalanobis distance in the contrastive HLT latent space.
- Main target on this branch: closure for all baseline backgrounds in the ABCD
  plane (`DY`, `QCD`, `TT`, and `WJets`).

For Della, the detailed runbook is also in
[`slurm/README_della.md`](slurm/README_della.md). The commands below are the
standard workflow for the current branch.

## Analysis Idea

The autoencoder (`train_ae.py`) learns to reconstruct object-level inputs. Its
reconstruction loss is used as an anomaly-like score.

The contrastive model (`train_hlt.py`, `models/hlt_con.py`) takes PF candidates
and learns a low-dimensional latent space using cross-entropy and supervised
contrastive loss. Evaluation (`eval_abcd_nurd.py`) turns that latent space into a
Mahalanobis-distance score.

NURD is the decorrelation part. It tries to remove AE-loss information from the
contrastive score for the background population used by the ABCD estimate. In
this branch the Slurm training default is all-baseline: per-class AE nuisance
bins, `CRITIC_SCOPE=baselines`, `CRITIC_SHUFFLE=within_label`, and
`CLOSURE_SCOPE=baselines`.

The default critic is the NURD density-ratio critic: a small network sees
`(latent, class label, AE-loss bin)` and tries to classify real triples from
triples with the AE-loss bin shuffled. If it can tell real from shuffled, the
latent representation still contains nuisance information. The encoder is
penalized with a bounded confusion objective, so real and shuffled triples become
indistinguishable. The older direct bin-prediction critic is still available with
`CRITIC_TYPE=bin_pred`.

The current training default defines AE nuisance bins inside each baseline
class, uses an EMA class whitening proxy for the Mahalanobis-distance axis, and
adds a direct per-baseline closure loss with distance-correlation,
profile-flatness, reverse-profile, and soft tail-ABCD terms.

Closure means the ABCD estimate agrees with the true QCD yield in region A:

```text
predicted A = B * C / D
nonclosure = (predicted A - true A) / true A
```

Conceptually, good closure means the two axes are independent enough for every
baseline class, and for their mixture, that sidebands B, C, and D can predict
the signal-like region A.

## Data

The training and eval scripts expect `.pt` files containing:

- `pf`: PF-candidate tensor used by the contrastive model.
- `obj`: object-level tensor used by the autoencoder.
- `label`: event class label.
- `eventid`: event identifier.

The Della scratch layout used by the Slurm scripts is:

```text
/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_smcocktail_train.pt
/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_smcocktail_test.pt
/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_signal_TpTp.pt
```

Labels used by the eval code are:

```text
0 = DY
1 = QCD
2 = TT
3 = WJets
```

## Della Setup From Scratch

Keep code in home and large outputs in scratch:

```bash
module load anaconda3/2025.12
conda activate disco

cd /home/mb7126/nurd_hlt
git switch wip-mila-all-baselines
git pull --ff-only

export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
mkdir -p $BASE/{checkpoints,logs,outputs,wandb,matplotlib}
```

Do not blindly install the old `requirements.txt` on Della. For this HLT path,
the needed packages are:

```bash
python -c "import torch, wandb, numpy, sklearn, scipy, matplotlib; print('imports ok'); print(torch.__version__, torch.version.cuda)"
```

If an import is missing in a fresh environment, install only the missing HLT
dependencies:

```bash
python -m pip install wandb numpy scikit-learn scipy matplotlib
```

If `torch` itself is missing, install the Della-supported CUDA PyTorch package
for the active Python environment. A CPU-only Torch install is not useful for
the full HLT training.

Verify the data before submitting jobs:

```bash
python -c "import torch; x=torch.load('$BASE/data/hlt_smcocktail_train.pt',map_location='cpu'); print(x.keys()); print(x['pf'].shape, x['obj'].shape, x['label'].shape); print(torch.unique(x['label'], return_counts=True))"
python -c "import torch; x=torch.load('$BASE/data/hlt_smcocktail_test.pt',map_location='cpu'); print(x.keys()); print(x['pf'].shape, x['obj'].shape, x['label'].shape)"
python -c "import torch; x=torch.load('$BASE/data/hlt_signal_TpTp.pt',map_location='cpu'); print(x.keys()); print(x['pf'].shape, x['obj'].shape, x['label'].shape)"
```

Before every Slurm submission, clear stale variables. This avoids accidentally
evaluating an old checkpoint:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
```

## Smoke Test

Submit:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
sbatch slurm/submit_smoke.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f $BASE/logs/nurd_smoke-<JOBID>.out
tail -f $BASE/logs/nurd_smoke-<JOBID>.err
```

Expected signs:

```text
CUDA available: True
NVIDIA A100...
Using AE normalization scaler saved in the AE checkpoint.
Critic scope: qcd
Saving checkpoint
SMOKE DONE
```

Expected outputs:

```bash
ls -lh $BASE/checkpoints/hlt/hlt/smoke_ae/
ls -lh $BASE/checkpoints/hlt/hlt/smoke_nurd/
```

## Full Training

Submit a fresh AE + NURD run:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
sbatch slurm/submit_train.sbatch
```

The default `RUN_TAG` is `all_baselines_<timestamp>`, so each submission gets a
fresh AE/NURD experiment name unless you override it.

The training job requests 10 hours, 64 GB CPU memory, and one A100. It exits
early if the allocated GPU has less than 75 GiB memory.

Monitor:

```bash
squeue -u $USER
tail -f $BASE/logs/nurd_hlt_train-<JOBID>.out
tail -f $BASE/logs/nurd_hlt_train-<JOBID>.err
```

The `.out` log prints:

```text
AE_EXP=...
NURD_EXP=...
BATCH_SIZE=4096
AE_EPOCHS=150
NURD_EPOCHS=150
CRITIC_SCOPE=baselines
CRITIC_TYPE=density_ratio
CRITIC_PENALTY_TYPE=confusion
CRITIC_SHUFFLE=within_label
BASELINE_LABELS=0,1,2,3
NUISANCE_BIN_SCOPE=per_class
CLOSURE_LOSS_TYPE=hybrid
CLOSURE_SCOPE=baselines
CLOSURE_WEIGHT=0.8
CONTRAST_WEIGHT=0.03
MD_PROXY_TYPE=ema
```

Checkpoints are written to:

```text
$BASE/checkpoints/hlt/hlt/<AE_EXP>/checkpoint_ae.pth
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/checkpoint_main_*.pth.tar
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/checkpoint_abcd.pth.tar
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/checkpoint_closure.pth.tar
```

`checkpoint_main_*.pth.tar` is selected by validation classification loss.
`checkpoint_abcd.pth.tar` is selected by the best validation all-baseline
proxy-ABCD grid score while keeping validation loss close to the best loss.
`checkpoint_closure.pth.tar` is kept as an alias of the ABCD-selected
checkpoint for older scripts.

To reuse an existing AE checkpoint and train only NURD:

```bash
unset CKPT OUTDIR AE_CKPT NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
AE_EXP=<existing_ae_exp> SKIP_AE=1 sbatch slurm/submit_train.sbatch
```

Useful training toggles:

```bash
CRITIC_SCOPE=baselines sbatch slurm/submit_train.sbatch # default, all-baseline critic
CRITIC_SCOPE=qcd sbatch slurm/submit_train.sbatch       # old QCD-only critic
CRITIC_SCOPE=all sbatch slurm/submit_train.sbatch       # all events, no baseline mask
CRITIC_TYPE=density_ratio sbatch slurm/submit_train.sbatch
CRITIC_TYPE=bin_pred sbatch slurm/submit_train.sbatch   # older direct-bin critic
CRITIC_SHUFFLE=within_label sbatch slurm/submit_train.sbatch # default conditional shuffle
CRITIC_SHUFFLE=global sbatch slurm/submit_train.sbatch  # older density-ratio shuffle
CLOSURE_LOSS_TYPE=hybrid sbatch slurm/submit_train.sbatch # default: dcorr + profiles + soft tail ABCD
CLOSURE_LOSS_TYPE=dcorr_profile sbatch slurm/submit_train.sbatch # no soft tail ABCD term
CLOSURE_LOSS_TYPE=corr sbatch slurm/submit_train.sbatch # cheaper Pearson-only closure loss
CLOSURE_LOSS_TYPE=abcd sbatch slurm/submit_train.sbatch # older random-cut batch proxy
CLOSURE_SCOPE=baselines sbatch slurm/submit_train.sbatch # default, closure loss over 0,1,2,3
CLOSURE_SCOPE=qcd sbatch slurm/submit_train.sbatch     # QCD-only closure loss
CRITIC_PENALTY_TYPE=logit_ratio sbatch slurm/submit_train.sbatch # previous HLT critic penalty
NUISANCE_BIN_SCOPE=per_class sbatch slurm/submit_train.sbatch # default per-baseline AE bins
NUISANCE_BIN_SCOPE=qcd sbatch slurm/submit_train.sbatch # old QCD AE nuisance bins
CLOSURE_WEIGHT=0.5 sbatch slurm/submit_train.sbatch    # weaker closure proxy
BATCH_SIZE=3072 sbatch slurm/submit_train.sbatch       # lower GPU memory
AE_EPOCHS=100 NURD_EPOCHS=100 sbatch slurm/submit_train.sbatch
```

W&B runs offline by default.

## Find A Checkpoint

List the newest real NURD checkpoints:

```bash
find $BASE/checkpoints/hlt/hlt -path '*/hlt_nurd_closure_bs4096_*/checkpoint_main_*.pth.tar' \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort -r | head -10
```

Recover experiment names from a training job:

```bash
JOB=<training_job_id>
grep -E '^AE_EXP=|^NURD_EXP=|^BATCH_SIZE=|^CRITIC_SCOPE=' $BASE/logs/nurd_hlt_train-$JOB.out
```

Select the newest checkpoint in one experiment:

```bash
NURD_EXP=<nurd_exp_from_log>
CKPT=$(ls -t $BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_main_*.pth.tar | head -1)
echo "$CKPT"
```

Select the closure-selected checkpoint in one experiment:

```bash
NURD_EXP=<nurd_exp_from_log>
CKPT=$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_abcd.pth.tar
echo "$CKPT"
```

`checkpoint_closure.pth.tar` points to the same ABCD-selected model for
backward compatibility.

## Evaluation

Use `slurm/submit_eval_latest.sbatch` for normal evaluations. It prints the
exact checkpoint, AE checkpoint, result directory, W&B run name, and W&B sync
command into the Slurm `.out` log.

### New Default: Held-Out All-Baseline Closure

This is the method to quote for this branch. Thresholds are optimized on one
deterministic stratified split of all baseline backgrounds and closure is
reported on the held-out split. The score uses min-MD across `DY`, `QCD`, `TT`,
and `WJets`, requires a minimum A-region fraction, and penalizes high
statistical uncertainty in the selected ABCD point.

Evaluate the newest real checkpoint:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
WANDB_NAME_PREFIX=all_baselines_heldout_eval sbatch slurm/submit_eval_latest.sbatch
```

Evaluate the newest ABCD-selected checkpoint instead of the normal
validation-loss checkpoint:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
PREFER_ABCD_CKPT=1 WANDB_NAME_PREFIX=all_baselines_abcd_ckpt \
  sbatch slurm/submit_eval_latest.sbatch
```

Evaluate a specific checkpoint:

```bash
unset OUTDIR WANDB_RUN_NAME WANDB_RUN_ID
CKPT=/path/to/checkpoint_main_YYYYMMDD_HHMMSS.pth.tar
AE_CKPT=/path/to/checkpoint_ae.pth
OUTDIR=$BASE/outputs/abcd_manual_heldout_$(date +%Y%m%d_%H%M%S)

CKPT="$CKPT" AE_CKPT="$AE_CKPT" OUTDIR="$OUTDIR" WANDB_NAME_PREFIX=all_baselines_heldout_eval \
  sbatch slurm/submit_eval_latest.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f $BASE/logs/nurd_eval_latest-<JOBID>.out
tail -f $BASE/logs/nurd_eval_latest-<JOBID>.err
```

Inspect outputs:

```bash
OUT=<the Results path printed in the eval .out log>
ls -lh $OUT
ls -lh $OUT/plots
cat $OUT/diagnostics.json
cat $OUT/abcd_thresholds.json
```

All-baseline eval defaults can be made explicit:

```bash
ABCD_SCOPE=all_baselines BASELINE_LABELS=0,1,2,3 MIN_MD=1 MIN_A_FRAC=0.02 \
  SELECTION_STAT_WEIGHT=0.5 SCAN_PERCENT_MAX=0.95 \
  sbatch slurm/submit_eval_latest.sbatch
```

To reproduce the previous QCD-only held-out evaluation:

```bash
ABCD_SCOPE=qcd MIN_MD=0 MIN_A_FRAC=0 SELECTION_STAT_WEIGHT=0 \
  WANDB_NAME_PREFIX=qcd_heldout_eval sbatch slurm/submit_eval_latest.sbatch
```

### Old Toggle: Same-Sample Closure

The old method optimizes thresholds and reports closure on the same events. It
is useful for comparison but can be too optimistic.

Run the old method on the newest checkpoint:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
CLOSURE_HOLDOUT_FRAC=0 WANDB_NAME_PREFIX=old_same_sample_eval \
  sbatch slurm/submit_eval_latest.sbatch
```

Run the old method on a specific checkpoint:

```bash
CKPT=/path/to/checkpoint_main_YYYYMMDD_HHMMSS.pth.tar
AE_CKPT=/path/to/checkpoint_ae.pth
OUTDIR=$BASE/outputs/abcd_manual_old_$(date +%Y%m%d_%H%M%S)

CKPT="$CKPT" AE_CKPT="$AE_CKPT" OUTDIR="$OUTDIR" CLOSURE_HOLDOUT_FRAC=0 \
  WANDB_NAME_PREFIX=old_same_sample_eval sbatch slurm/submit_eval_latest.sbatch
```

### Compare New And Old On The Same Checkpoint

```bash
unset OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID

CKPT=$(find $BASE/checkpoints/hlt/hlt -path '*/hlt_nurd_closure_bs4096_*/checkpoint_main_*.pth.tar' \
  -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')

echo "Using CKPT=$CKPT"

CKPT="$CKPT" WANDB_NAME_PREFIX=all_baselines_heldout_eval \
  OUTDIR=$BASE/outputs/abcd_compare_heldout_$(date +%Y%m%d_%H%M%S) \
  sbatch slurm/submit_eval_latest.sbatch

CKPT="$CKPT" CLOSURE_HOLDOUT_FRAC=0 WANDB_NAME_PREFIX=old_same_sample_eval \
  OUTDIR=$BASE/outputs/abcd_compare_old_$(date +%Y%m%d_%H%M%S) \
  sbatch slurm/submit_eval_latest.sbatch
```

Compare:

```bash
cat <heldout_OUTDIR>/diagnostics.json
cat <old_OUTDIR>/diagnostics.json
ls -lh <heldout_OUTDIR>/plots
ls -lh <old_OUTDIR>/plots
```

Key metrics:

- `ABCD/nonclosure`: main held-out closure number when
  `CLOSURE_HOLDOUT_FRAC > 0`. It is `predicted_A / true_A - 1`.
- `ABCD/legacy_nonclosure`: old convention, `(true_A - predicted_A) /
  predicted_A`, saved only for comparing to older outputs.
- `ABCD/ratio_pred_over_true`: the direct closure ratio.
- `ABCD/tune_nonclosure`: threshold-tuning split result.
- `ABCD/report_best_nonclosure`: best possible held-out point, for reference.
- `ABCD/grid_median_abs_nonclosure` and `ABCD/grid_p90_abs_nonclosure`: closure
  stability across the scan.
- `ABCD/scope_all_baselines`: `1` means thresholds were selected/reported on
  the all-baseline mixture.
- `diagnostics.json -> abcd_selection.report_at_selected_per_class`: selected
  ABCD closure for each baseline class.
- `diagnostics.json -> signal_at_selected`: TpTp counts and efficiencies in
  A/B/C/D at the selected thresholds.
- `Corr/all_baselines_*`: decorrelation diagnostics for the baseline mixture.
- `diagnostics.json -> per_class_correlations`: per-baseline decorrelation.
- `Corr/qcd_pearson`, `Corr/qcd_spearman`, `Corr/qcd_distance`: QCD
  decorrelation diagnostics.
- `Closure/tail_le_2pct_mean_ratio`: high-score tail prediction quality.

## W&B

Training and eval use `WANDB_MODE=offline` by default because compute nodes may
not initialize online W&B reliably. Nothing is uploaded until you sync.

Save the API key once:

```bash
mkdir -p ~/.secrets
chmod 700 ~/.secrets
nano ~/.secrets/wandb_api_key
chmod 600 ~/.secrets/wandb_api_key
```

Sync the exact run printed by a job:

```bash
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)
wandb sync <offline-run-dir-printed-in-the-log>
```

Sync every offline run under scratch:

```bash
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)
find $BASE/wandb -type d -name 'offline-run-*' -print0 | xargs -0 -n1 wandb sync
```

Try automatic sync at the end of eval:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
SYNC_WANDB=1 WANDB_NAME_PREFIX=heldout_eval sbatch slurm/submit_eval_latest.sbatch
```

If compute-node sync fails, rerun the printed `wandb sync ...` command from a
login node.

## Direct Python Entry Points

These are useful outside Slurm or for debugging small jobs:

```bash
python train_ae.py --data "$BASE/data/hlt_smcocktail_train.pt" --epochs 1 --batch_size 64 --max_events 2000 --local_testing 1 --exp_name smoke_ae --project_name hlt

python train_hlt.py \
  --data "$BASE/data/hlt_smcocktail_train.pt" \
  --ae_ckpt checkpoints/hlt/hlt/smoke_ae/checkpoint_ae.pth \
  --epochs 1 --batch_size 64 --max_events 2000 --local_testing 1 \
  --critic_scope qcd --critic_schedule warmup --critic_type density_ratio \
  --critic_penalty_type confusion --nuisance_bin_scope qcd \
  --closure_weight 0.1 --closure_loss_type dcorr_profile --md_proxy_type ema \
  --reweight 1 --joint_indep 1 \
  --exp_name smoke_nurd --project_name hlt
```

Held-out eval directly:

```bash
python eval_abcd_nurd.py \
  --ckpt "$CKPT" \
  --ae_ckpt "$AE_CKPT" \
  --test_pt "$BASE/data/hlt_smcocktail_test.pt" \
  --signal_pt "$BASE/data/hlt_signal_TpTp.pt" \
  --outdir "$BASE/outputs/abcd_manual" \
  --closure_holdout_frac 0.5 \
  --n_pca 6
```

Old same-sample eval directly:

```bash
python eval_abcd_nurd.py \
  --ckpt "$CKPT" \
  --ae_ckpt "$AE_CKPT" \
  --test_pt "$BASE/data/hlt_smcocktail_test.pt" \
  --signal_pt "$BASE/data/hlt_signal_TpTp.pt" \
  --outdir "$BASE/outputs/abcd_manual_old" \
  --closure_holdout_frac 0 \
  --n_pca 6
```

## Datacard For CMS Combine

After an eval run, reuse its thresholds:

```bash
python make_datacard_ttbar.py \
  --ckpt "$CKPT" \
  --ae_ckpt "$AE_CKPT" \
  --test_pt "$BASE/data/hlt_smcocktail_test.pt" \
  --eval_json "$OUT/abcd_thresholds.json" \
  --outdir "$BASE/outputs/datacard_<run>"
```

Run Combine from the datacard directory:

```bash
combine -M Significance datacard_ttbar.txt -t -1 --expectSignal 1
combine -M AsymptoticLimits datacard_ttbar.txt -t -1
```
