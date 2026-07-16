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
