# NURD HLT Anomaly Detection

This repo trains and evaluates a two-axis HLT anomaly-detection analysis using
ABCD background estimation.

- Axis 1: autoencoder reconstruction loss from object-level event features.
- Axis 2: Mahalanobis distance in the contrastive HLT latent space.
- Main target: QCD closure in the ABCD plane.

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
this branch the Slurm training default is `CRITIC_SCOPE=qcd`, so the adversarial
critic targets QCD, which is the class used for closure.

The critic is a small network that tries to predict the binned AE reconstruction
loss from the contrastive latent representation. If the critic can predict the
AE-loss bin, the latent representation still contains nuisance information. The
encoder is penalized for allowing that, so the two ABCD axes become less
correlated.

Closure means the ABCD estimate agrees with the true QCD yield in region A:

```text
predicted A = B * C / D
nonclosure = (predicted A - true A) / true A
```

Conceptually, good closure means the two axes are independent enough for QCD
that sidebands B, C, and D can predict the signal-like region A.

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
git switch wip-mila-test
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
RUN_TAG=qcdcritic_$(date +%Y%m%d_%H%M%S) sbatch slurm/submit_train.sbatch
```

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
CRITIC_SCOPE=qcd
```

Checkpoints are written to:

```text
$BASE/checkpoints/hlt/hlt/<AE_EXP>/checkpoint_ae.pth
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/checkpoint_main_*.pth.tar
```

To reuse an existing AE checkpoint and train only NURD:

```bash
unset CKPT OUTDIR AE_CKPT NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
AE_EXP=<existing_ae_exp> SKIP_AE=1 RUN_TAG=qcdcritic_$(date +%Y%m%d_%H%M%S) sbatch slurm/submit_train.sbatch
```

Useful training toggles:

```bash
CRITIC_SCOPE=qcd sbatch slurm/submit_train.sbatch      # default, QCD-only critic
CRITIC_SCOPE=all sbatch slurm/submit_train.sbatch      # older all-class critic
CLOSURE_WEIGHT=0.3 sbatch slurm/submit_train.sbatch    # stronger closure proxy
BATCH_SIZE=3072 sbatch slurm/submit_train.sbatch       # lower GPU memory
AE_EPOCHS=100 NURD_EPOCHS=100 sbatch slurm/submit_train.sbatch
```

The full training job requests one A100 and exits early unless the allocated GPU
has at least 75 GiB memory. W&B runs offline by default.

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

## Evaluation

Use `slurm/submit_eval_latest.sbatch` for normal evaluations. It prints the
exact checkpoint, AE checkpoint, result directory, W&B run name, and W&B sync
command into the Slurm `.out` log.

### New Default: Held-Out Closure

This is the method to quote. Thresholds are optimized on one deterministic QCD
split and closure is reported on the held-out QCD split.

Evaluate the newest real checkpoint:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
WANDB_NAME_PREFIX=heldout_eval sbatch slurm/submit_eval_latest.sbatch
```

Evaluate a specific checkpoint:

```bash
unset OUTDIR WANDB_RUN_NAME WANDB_RUN_ID
CKPT=/path/to/checkpoint_main_YYYYMMDD_HHMMSS.pth.tar
AE_CKPT=/path/to/checkpoint_ae.pth
OUTDIR=$BASE/outputs/abcd_manual_heldout_$(date +%Y%m%d_%H%M%S)

CKPT="$CKPT" AE_CKPT="$AE_CKPT" OUTDIR="$OUTDIR" WANDB_NAME_PREFIX=heldout_eval \
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

### Old Toggle: Same-Sample Closure

The old method optimizes thresholds and reports closure on the same QCD events.
It is useful for comparison but can be too optimistic.

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

CKPT="$CKPT" WANDB_NAME_PREFIX=heldout_eval \
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
  `CLOSURE_HOLDOUT_FRAC > 0`.
- `ABCD/tune_nonclosure`: threshold-tuning split result.
- `ABCD/report_best_nonclosure`: best possible held-out point, for reference.
- `ABCD/grid_median_abs_nonclosure` and `ABCD/grid_p90_abs_nonclosure`: closure
  stability across the scan.
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
  --critic_scope qcd --critic_schedule warmup \
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
