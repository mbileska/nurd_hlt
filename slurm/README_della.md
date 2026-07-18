# Della Slurm Jobs

These jobs assume:

- Repo: `/home/mb7126/nurd_hlt`
- Scratch base: `/scratch/gpfs/IOJALVO/mb7126/nurd_hlt`
- Data:
  - `/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_smcocktail_train.pt`
  - `/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_smcocktail_test.pt`
  - `/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/hlt_signal_TpTp.pt`
- Conda env: `disco`
- W&B API key for full training: `~/.secrets/wandb_api_key`

## Before Submitting

```bash
module load anaconda3/2025.12
conda activate disco
cd /home/mb7126/nurd_hlt

export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
mkdir -p $BASE/{checkpoints,logs,outputs,wandb}
```

Required Python packages for the HLT train/eval path:

```bash
python -m pip install torch wandb numpy scikit-learn scipy matplotlib
```

Current smoke/full training does not require `torchvision`.

Check that the data can be read:

```bash
python -c "import torch; x=torch.load('$BASE/data/hlt_smcocktail_train.pt',map_location='cpu'); print(x.keys()); print(x['pf'].shape, x['obj'].shape, x['label'].shape); print(torch.unique(x['label'], return_counts=True))"
```

## Smoke Test

Submit:

```bash
sbatch slurm/submit_smoke.sbatch
```

Inspect:

```bash
squeue -u $USER
tail -f /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/nurd_smoke-<JOBID>.out
tail -f /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/nurd_smoke-<JOBID>.err
```

Expected log signs:

```text
CUDA available: True
NVIDIA A100...
Epoch 1/1
[train] NURD weight groups=...
Saving checkpoint
SMOKE DONE
```

Expected files:

```bash
ls -lh checkpoints/hlt/hlt/smoke_ae/
ls -lh checkpoints/hlt/hlt/smoke_nurd/
```

You should see `checkpoint_ae.pth` and at least one `checkpoint_main_*.pth.tar`.

## Full Training

Submit only after the smoke test passes:

```bash
sbatch slurm/submit_train.sbatch
```

Inspect:

```bash
squeue -u $USER
tail -f /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/nurd_hlt_train-<JOBID>.out
tail -f /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/nurd_hlt_train-<JOBID>.err
```

The full job prints `AE_EXP=...` and `NURD_EXP=...`. The checkpoints are under:

```text
/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/checkpoints/hlt/hlt/<AE_EXP>/
/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/checkpoints/hlt/hlt/<NURD_EXP>/
```

W&B project: `nurd-ood-hlt`.

To reuse an existing AE checkpoint and rerun only the NURD stage:

```bash
AE_EXP=ae_pretrain_20260717_102917 SKIP_AE=1 sbatch slurm/submit_train.sbatch
```

The NURD stage keeps batch size 4096 and uses `--offload_critic_graph` to
preserve the original critic-penalty second forward while saving its autograd
tensors on CPU. The job excludes the Della `della-i*` A100 nodes seen in
`sinfo` and exits early unless the allocated GPU has at least 75 GiB memory.

On this test branch the Slurm job defaults to `CRITIC_SCOPE=qcd`, so the
critic targets the QCD background used by ABCD closure. To reproduce the older
all-class critic behavior:

```bash
CRITIC_SCOPE=all sbatch slurm/submit_train.sbatch
```

Useful knobs that do not require editing the script:

```bash
SKIP_AE=1 AE_EXP=<existing_ae_exp> sbatch slurm/submit_train.sbatch
AE_EPOCHS=100 NURD_EPOCHS=100 sbatch slurm/submit_train.sbatch
CLOSURE_WEIGHT=0.2 sbatch slurm/submit_train.sbatch
```

If batch size 4096 still runs out of GPU memory on an 80 GB A100:

```bash
AE_EXP=ae_pretrain_20260717_102917 SKIP_AE=1 BATCH_SIZE=3072 sbatch slurm/submit_train.sbatch
```

The full Slurm job uses W&B offline mode by default because Della compute
nodes may not be able to initialize online W&B reliably. Metrics are still
written under the scratch W&B directory and can be synced later.

After the job finishes, find offline runs:

```bash
find /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/wandb -type d -name 'offline-run-*'
```

Sync them from a session that can reach W&B:

```bash
find /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/wandb -type d -name 'offline-run-*' -print0 | xargs -0 -n1 wandb sync
```

## ABCD / Closure Evaluation

Run eval through Slurm; interactive login-node eval can be killed by the
cluster. Results are written under scratch:
`/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/outputs/abcd_<NURD_EXP>/`.

```bash
NURD_EXP=hlt_nurd_closure_bs4096_20260717_190314 sbatch slurm/submit_eval_abcd.sbatch
```

If `NURD_EXP` is omitted, the script uses the newest
`hlt_nurd_closure_bs4096_*` checkpoint directory.

Inspect:

```bash
tail -f /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/nurd_eval-<JOBID>.out
tail -f /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/logs/nurd_eval-<JOBID>.err
```

Key outputs:

```bash
ls -lh /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/outputs/abcd_<NURD_EXP>/
ls -lh /scratch/gpfs/IOJALVO/mb7126/nurd_hlt/outputs/abcd_<NURD_EXP>/plots/
```

The most important non-plot files are:

```text
abcd_thresholds.json
diagnostics.json
```

For decorrelation, inspect `diagnostics.json` and W&B keys
`Corr/qcd_pearson`, `Corr/qcd_spearman`, `Corr/qcd_distance`. For closure,
do not look only at the optimized `ABCD/nonclosure`; also check
`ABCD/grid_median_abs_nonclosure` and `ABCD/grid_p90_abs_nonclosure`.

## Latest Checkpoint Eval

To automatically evaluate the newest `checkpoint_main_*.pth.tar` under
scratch, use:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
sbatch slurm/submit_eval_latest.sbatch
```

The wrapper prints the exact `CKPT`, `AE_CKPT`, result directory, W&B run name,
and W&B sync command into the Slurm `.out` log. By default the W&B run name
starts with `optimized_`.

To sync that eval run automatically at the end of the Slurm job:

```bash
unset CKPT OUTDIR AE_CKPT AE_EXP NURD_EXP WANDB_RUN_NAME WANDB_RUN_ID
SYNC_WANDB=1 sbatch slurm/submit_eval_latest.sbatch
```

If compute-node W&B sync fails, use the printed `W&B sync command` from the
`.out` log on a login node.
