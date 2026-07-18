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

git switch wip-mila-test
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

Always use a fresh `RUN_TAG`. This prevents mixing yesterday's checkpoints with
today's training.

Train AE and NURD from scratch:

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

The `.out` log prints the exact experiment names:

```text
AE_EXP=...
NURD_EXP=...
BATCH_SIZE=4096
CRITIC_SCOPE=qcd
CRITIC_TYPE=density_ratio
CLOSURE_LOSS_TYPE=corr
```

Save those values. The checkpoints will be under:

```text
$BASE/checkpoints/hlt/hlt/<AE_EXP>/
$BASE/checkpoints/hlt/hlt/<NURD_EXP>/
```

The normal model checkpoint is `checkpoint_main_*.pth.tar`, selected by
validation classification loss. Training also writes
`checkpoint_closure.pth.tar`, selected by the smallest validation QCD
AE-vs-proxy-MD correlation while the validation loss stays close to the best
loss.

Reuse an existing AE and train only NURD:

```bash
unset CKPT OUTDIR AE_CKPT NURD_EXP RUN_TAG WANDB_RUN_NAME WANDB_RUN_ID
AE_EXP=<existing_ae_exp> SKIP_AE=1 RUN_TAG=qcdcritic_$(date +%Y%m%d_%H%M%S) sbatch slurm/submit_train.sbatch
```

Useful training toggles:

```bash
CRITIC_SCOPE=qcd sbatch slurm/submit_train.sbatch      # default, targets QCD closure
CRITIC_SCOPE=all sbatch slurm/submit_train.sbatch      # old all-class critic
CRITIC_TYPE=density_ratio sbatch slurm/submit_train.sbatch
CRITIC_TYPE=bin_pred sbatch slurm/submit_train.sbatch   # old direct-bin critic
CLOSURE_LOSS_TYPE=corr sbatch slurm/submit_train.sbatch # default, cheaper and closer to eval axes
CLOSURE_LOSS_TYPE=abcd sbatch slurm/submit_train.sbatch # old random-cut batch proxy
CLOSURE_WEIGHT=0.3 sbatch slurm/submit_train.sbatch    # stronger closure loss
BATCH_SIZE=3072 sbatch slurm/submit_train.sbatch       # lower memory
AE_EPOCHS=100 NURD_EPOCHS=100 sbatch slurm/submit_train.sbatch
```

The training job requests 7 hours, 64 GB CPU memory, and one A100. It exits
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
CKPT=$BASE/checkpoints/hlt/hlt/$NURD_EXP/checkpoint_closure.pth.tar
echo "$CKPT"
```

## 4. Evaluation: New Correct Held-Out Method

This is the default and the method to quote. It chooses ABCD thresholds on one
deterministic half of QCD and reports closure on the held-out half.

Evaluate the newest real checkpoint automatically:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
WANDB_NAME_PREFIX=heldout_eval sbatch slurm/submit_eval_latest.sbatch
```

Evaluate the newest closure-selected checkpoint instead:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
PREFER_CLOSURE_CKPT=1 WANDB_NAME_PREFIX=heldout_eval_closure_ckpt \
  sbatch slurm/submit_eval_latest.sbatch
```

Evaluate a specific checkpoint explicitly:

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

- `ABCD/nonclosure`: selected thresholds reported on held-out QCD. This is the
  main closure number.
- `ABCD/tune_nonclosure`: selected thresholds measured on the tuning split.
  This is useful for debugging but optimistic.
- `ABCD/report_best_nonclosure`: best possible point on the held-out split.
  This is a reference only, not a final number.
- `Closure/tail_le_2pct_mean_ratio`: tight-tail average of
  `Predicted Bkg / True Bkg`.
- `Corr/qcd_pearson`, `Corr/qcd_spearman`, `Corr/qcd_distance`: QCD
  decorrelation diagnostics.

## 5. Evaluation: Old Same-Sample Method

The old method optimizes thresholds and reports closure on the same QCD events.
It can make the optimized red point look too good. Use it only as a comparison.

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
