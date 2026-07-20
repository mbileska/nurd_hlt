# Della HLT Runbook

Use this as the standard workflow on Della for this branch. It keeps code in
`/home/mb7126/nurd_hlt` and all large files, checkpoints, logs, W&B files, and
plots in scratch.

## 0. One-Time Assumptions

- Repo: `/home/mb7126/nurd_hlt`
- Scratch base: `/scratch/gpfs/IOJALVO/mb7126/nurd_hlt`
- Conda env: `disco`
- Data:
  - `/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_smcocktail_train.pt`
  - `/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_smcocktail_test.pt`
  - `/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_signal_TpTp.pt`
- Optional W&B key: `~/.secrets/wandb_api_key`

Load the environment and update code:

```bash
module load anaconda3/2025.12
conda activate disco
cd /home/mb7126/nurd_hlt

git switch wip-mila-all-baselines
git pull --ff-only
```

Create scratch directories:

```bash
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
mkdir -p $BASE/{checkpoints,logs,outputs,wandb,matplotlib}
```

Check Python dependencies:

```bash
python -c "import torch, wandb, numpy, sklearn, scipy, matplotlib; print('imports ok'); print(torch.__version__, torch.version.cuda)"
```

Do not blindly install the old top-level `requirements.txt` on Della. If a
non-Torch package is missing, install only the missing HLT dependency. If
`torch` is missing, install a Della-supported CUDA PyTorch package; CPU-only
Torch is not useful for full training.

Check the data files:

```bash
python -c "import torch; x=torch.load('$BASE/data/hlt_smcocktail_train.pt',map_location='cpu'); print(x.keys()); print(x['pf'].shape, x['obj'].shape, x['label'].shape); print(torch.unique(x['label'], return_counts=True))"
python -c "import torch; x=torch.load('$BASE/data/hlt_smcocktail_test.pt',map_location='cpu'); print(x.keys()); print(x['pf'].shape, x['obj'].shape, x['label'].shape)"
python -c "import torch; x=torch.load('$BASE/data/hlt_signal_TpTp.pt',map_location='cpu'); print(x.keys()); print(x['pf'].shape, x['obj'].shape, x['label'].shape)"
```

Before every submission, clear old environment variables. Slurm exports your
current shell environment, so stale values can make jobs reuse old checkpoints
or old output directories.

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
```

## 1. Smoke Test

Run this after pulling code changes or changing the environment:

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

Smoke outputs:

```bash
ls -lh $BASE/checkpoints/hlt/hlt/smoke_ae/
ls -lh $BASE/checkpoints/hlt/hlt/smoke_nurd/
```

## 2. Full Training

The default `RUN_TAG` includes `all_baselines` plus a timestamp. Use a fresh
tag when overriding it so yesterday's checkpoints cannot mix with today's
training.

Train AE and NURD from scratch:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
sbatch slurm/submit_train.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f $BASE/logs/nurd_hlt_train-<JOBID>.out
tail -f $BASE/logs/nurd_hlt_train-<JOBID>.err
```

The `.out` log prints the exact experiment names:

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

Save those values. The checkpoints will be under:

```text
$BASE/checkpoints/hlt/hlt/<AE_EXP>/
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/
```

The normal model checkpoint is `checkpoint_main_*.pth.tar`, selected by
validation classification loss. Training also writes `checkpoint_abcd.pth.tar`,
selected by a validation all-baseline proxy-ABCD grid score while the validation
loss stays close to the best loss. `checkpoint_closure.pth.tar` is kept as an
alias of the ABCD-selected checkpoint for older scripts.

Reuse an existing AE and train only NURD:

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
CRITIC_TYPE=bin_pred sbatch slurm/submit_train.sbatch   # old direct-bin critic
CRITIC_SHUFFLE=within_label sbatch slurm/submit_train.sbatch # default conditional shuffle
CRITIC_SHUFFLE=global sbatch slurm/submit_train.sbatch  # older density-ratio shuffle
CLOSURE_LOSS_TYPE=hybrid sbatch slurm/submit_train.sbatch # default: dcorr + profiles + soft tail ABCD
CLOSURE_LOSS_TYPE=dcorr_profile sbatch slurm/submit_train.sbatch # no soft tail ABCD term
CLOSURE_LOSS_TYPE=corr sbatch slurm/submit_train.sbatch # cheaper Pearson-only closure loss
CLOSURE_LOSS_TYPE=abcd sbatch slurm/submit_train.sbatch # old random-cut batch proxy
CLOSURE_SCOPE=baselines sbatch slurm/submit_train.sbatch # default, closure over 0,1,2,3
CLOSURE_SCOPE=qcd sbatch slurm/submit_train.sbatch     # QCD-only closure loss
CRITIC_PENALTY_TYPE=logit_ratio sbatch slurm/submit_train.sbatch # previous HLT critic penalty
NUISANCE_BIN_SCOPE=per_class sbatch slurm/submit_train.sbatch # default per-baseline AE bins
NUISANCE_BIN_SCOPE=qcd sbatch slurm/submit_train.sbatch # old QCD AE nuisance bins
CLOSURE_WEIGHT=0.5 sbatch slurm/submit_train.sbatch    # weaker closure loss
BATCH_SIZE=3072 sbatch slurm/submit_train.sbatch       # lower memory
AE_EPOCHS=100 NURD_EPOCHS=100 sbatch slurm/submit_train.sbatch
```

The training job requests 10 hours, 64 GB CPU memory, and one A100. It exits
early unless the GPU has at least 75 GiB memory. It runs W&B in offline mode by
default.

## 3. Find The Checkpoint You Want

Newest real NURD checkpoint:

```bash
find $BASE/checkpoints/hlt/hlt -path '*/hlt_nurd_closure_bs4096_*/checkpoint_main_*.pth.tar' \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort -r | head -10
```

Get `AE_EXP` and `NURD_EXP` from a training job:

```bash
JOB=<training_job_id>
grep -E '^AE_EXP=|^NURD_EXP=|^BATCH_SIZE=|^CRITIC_SCOPE=' $BASE/logs/nurd_hlt_train-$JOB.out
```

Get newest checkpoint from that NURD experiment:

```bash
NURD_EXP=<nurd_exp_from_log>
CKPT=$(ls -t $BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_main_*.pth.tar | head -1)
echo "$CKPT"
```

Get the closure-selected checkpoint from that NURD experiment:

```bash
NURD_EXP=<nurd_exp_from_log>
CKPT=$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_abcd.pth.tar
echo "$CKPT"
```

## 4. Evaluation: Held-Out All-Baseline Method

This is the default and the method to quote for this branch. It chooses ABCD
thresholds on one deterministic stratified split of all baseline backgrounds
and reports closure on the held-out split. The score uses min-MD across
`DY`, `QCD`, `TT`, and `WJets`, requires a minimum A-region fraction, and
penalizes high statistical uncertainty in the selected ABCD point.

Evaluate the newest real checkpoint automatically:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
WANDB_NAME_PREFIX=all_baselines_heldout_eval sbatch slurm/submit_eval_latest.sbatch
```

Evaluate the newest ABCD-selected checkpoint instead:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
PREFER_ABCD_CKPT=1 WANDB_NAME_PREFIX=all_baselines_abcd_ckpt \
  sbatch slurm/submit_eval_latest.sbatch
```

Evaluate a specific checkpoint explicitly:

```bash
unset OUTDIR WANDB_RUN_NAME WANDB_RUN_ID
CKPT=/path/to/checkpoint_main_YYYYMMDD_HHMMSS.pth.tar
AE_CKPT=/path/to/checkpoint_ae.pth
OUTDIR=$BASE/outputs/abcd_manual_heldout_$(date +%Y%m%d_%H%M%S)

CKPT="$CKPT" AE_CKPT="$AE_CKPT" OUTDIR="$OUTDIR" WANDB_NAME_PREFIX=all_baselines_heldout_eval \
  sbatch slurm/submit_eval_latest.sbatch
```

Make the all-baseline defaults explicit:

```bash
ABCD_SCOPE=all_baselines BASELINE_LABELS=0,1,2,3 MIN_MD=1 MIN_A_FRAC=0.02 \
  SELECTION_STAT_WEIGHT=0.5 SCAN_PERCENT_MAX=0.95 \
  sbatch slurm/submit_eval_latest.sbatch
```

Reproduce the previous QCD-only held-out method:

```bash
ABCD_SCOPE=qcd MIN_MD=0 MIN_A_FRAC=0 SELECTION_STAT_WEIGHT=0 \
  WANDB_NAME_PREFIX=qcd_heldout_eval sbatch slurm/submit_eval_latest.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f $BASE/logs/nurd_eval_latest-<JOBID>.out
tail -f $BASE/logs/nurd_eval_latest-<JOBID>.err
```

When done, the `.out` log prints:

```text
Results: <OUTDIR>
CKPT: <exact checkpoint>
AE_CKPT: <exact AE checkpoint>
W&B sync command: wandb sync ...
```

Inspect outputs:

```bash
OUT=<the Results path from the eval .out log>
ls -lh $OUT
ls -lh $OUT/plots
cat $OUT/diagnostics.json
cat $OUT/abcd_thresholds.json
```

Important held-out eval numbers:

- `ABCD/nonclosure`: selected thresholds reported on the held-out all-baseline
  mixture. This is the main closure number.
- `ABCD/tune_nonclosure`: selected thresholds measured on the tuning split.
  This is useful for debugging but optimistic.
- `ABCD/report_best_nonclosure`: best possible point on the held-out split.
  This is a reference only, not a final number.
- `Closure/tail_le_2pct_mean_ratio`: tight-tail average of
  `Predicted Bkg / True Bkg`.
- `Corr/qcd_pearson`, `Corr/qcd_spearman`, `Corr/qcd_distance`: QCD
  decorrelation diagnostics.

## 5. Evaluation: Old Same-Sample Method

The old method optimizes thresholds and reports closure on the same events. It
can make the optimized red point look too good. Use it only as a comparison.

Evaluate the newest checkpoint with the old method:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
CLOSURE_HOLDOUT_FRAC=0 WANDB_NAME_PREFIX=old_same_sample_eval \
  sbatch slurm/submit_eval_latest.sbatch
```

Evaluate the same explicit checkpoint with the old method:

```bash
CKPT=/path/to/checkpoint_main_YYYYMMDD_HHMMSS.pth.tar
AE_CKPT=/path/to/checkpoint_ae.pth
OUTDIR=$BASE/outputs/abcd_manual_old_$(date +%Y%m%d_%H%M%S)

CKPT="$CKPT" AE_CKPT="$AE_CKPT" OUTDIR="$OUTDIR" CLOSURE_HOLDOUT_FRAC=0 \
  WANDB_NAME_PREFIX=old_same_sample_eval sbatch slurm/submit_eval_latest.sbatch
```

## 6. Run Both Eval Methods On The Same Checkpoint

This is the cleanest way to compare old vs new.

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

Compare the two `diagnostics.json` files and the two
`plots/cut_and_count_bkg_check.png` plots.

## 7. W&B Sync

Training and eval run with `WANDB_MODE=offline` by default. Nothing is uploaded
unless you sync.

Sync the exact eval run from its `.out` log:

```bash
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)
wandb sync <offline-run-dir-printed-in-the-eval-log>
```

Sync every offline run under scratch:

```bash
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)
find $BASE/wandb -type d -name 'offline-run-*' -print0 | xargs -0 -n1 wandb sync
```

Try automatic sync at the end of an eval job:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
SYNC_WANDB=1 WANDB_NAME_PREFIX=heldout_eval sbatch slurm/submit_eval_latest.sbatch
```

If compute-node sync fails, rerun the printed `wandb sync ...` command from a
login node.

## 8. Quick Interpretation

- Use held-out `ABCD/nonclosure` as the main closure number.
- In current outputs, `ABCD/nonclosure` means `predicted_A / true_A - 1`.
  `ABCD/legacy_nonclosure` is saved only for comparison with older outputs.
- Use old same-sample eval only to understand how much the previous method was
  over-optimizing.
- A good red point means the selected ABCD working point generalizes.
- A low tight-tail ratio in the closure curve means the model still
  underpredicts QCD in the high-AE/high-MD tail.
- Do not quote only the optimized red point without the held-out and grid
  diagnostics.
