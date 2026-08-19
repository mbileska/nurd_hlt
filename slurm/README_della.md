# Della command reference

The authoritative weighted V4×4 contract is in [`../README.md`](../README.md).

```bash
module load anaconda3/2025.12
conda activate disco
cd ~/nurd_hlt
git fetch origin
git switch wip-mila-test
git pull --ff-only
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
```

Smoke test:

```bash
sbatch slurm/submit_smoke.sbatch
```

Submit one exact training job and its dependent held-out plus legacy evaluation:

```bash
bash slurm/launch_v4x4_campaign.sh
```

This is a fresh, sealed campaign: old exported run names, checkpoint paths, and
training/evaluation hyperparameters are ignored. The launcher trains a new AE
and refuses to overwrite an existing campaign directory.

The launcher prints both job IDs and the result directory:

```text
$BASE/outputs/<EVAL_NAME>/held-out/
$BASE/outputs/<EVAL_NAME>/legacy/
$BASE/outputs/<EVAL_NAME>/evaluation_summary.json
```

Evaluate an already completed contract-v2 run:

```bash
export NURD_EXP=hlt_nurd_closure_bs4096_<exact_run_tag>
export EVAL_NAME=<exact_run_tag>_eval
sbatch --export=ALL,BASE="$BASE",NURD_EXP="$NURD_EXP",EVAL_NAME="$EVAL_NAME" \
  slurm/submit_eval_latest.sbatch
```

Wildcards, newest-run fallback, newest-AE fallback, main-checkpoint fallback,
and output overwrites are disabled. `NURD_EXP` must be exact and contain an
eligible `checkpoint_abcd.pth.tar`. The held-out result is primary; `legacy/`
is the unweighted, same-sample QCD protocol from `origin/main`.
