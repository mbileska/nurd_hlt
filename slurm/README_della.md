# Della Runbook

The authoritative end-to-end instructions for branch `experimental` are in
[`../README.md`](../README.md). This short file is a command reference.

## Update And Verify

```bash
module load anaconda3/2025.12
conda activate disco
cd /home/mb7126/nurd_hlt
git fetch origin
git switch experimental
git pull --ff-only

export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
python -c "import torch,wandb,numpy,sklearn,scipy,matplotlib; print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
```

Clear inherited selections before every submission:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP RUN_TAG
unset WANDB_RUN_NAME WANDB_RUN_ID NURD_GLOB
unset ABCD_SCOPE SCORE_MODE MIN_MD
```

## Smoke

```bash
sbatch slurm/submit_smoke.sbatch
```

Success is `SMOKE DONE` in:

```text
$BASE/logs/nurd_smoke-<job_id>.out
```

## Train

```bash
sbatch slurm/submit_train.sbatch
```

Defaults: 100 AE epochs, 200 NURD epochs, a 12-hour hard limit, one 80 GB A100,
8 CPU cores, 48 GB CPU memory, and zero DataLoader workers. Training uses
40 QCD-defined nuisance bins, a 10/20/40-bin direct critic, 25% QCD batches
with importance correction, and cross-fitted validation MD.

Monitor:

```bash
JOB=<training_job_id>
squeue -j $JOB
tail -f $BASE/logs/nurd_hlt_train-$JOB.out
tail -f $BASE/logs/nurd_hlt_train-$JOB.err
```

## Evaluate

Recommended independent-test QCD closure with QCD Mahalanobis distance:

```bash
sbatch slurm/submit_eval_latest.sbatch
```

Closure-selected checkpoint:

```bash
PREFER_ABCD_CKPT=1 ABCD_SCOPE=qcd SCORE_MODE=qcd_md \
  sbatch slurm/submit_eval_latest.sbatch
```

The eval job fits class references on the training split, calibrates them on
half of the original validation split, selects thresholds on the other half,
and reports once on the independent test file.

Inspect:

```bash
JOB=<eval_job_id>
OUT=$(grep '^Results:' $BASE/logs/nurd_eval_latest-$JOB.out | sed 's/^Results: //')
cat "$OUT/diagnostics.json"
cat "$OUT/abcd_thresholds.json"
ls -lh "$OUT/plots"
```

See the main README for explicit checkpoint selection, score comparisons,
all-baseline closure, interpretation, and W&B synchronization.
