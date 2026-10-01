# HLT NURD anomaly detection

This is the HLT-only implementation of the two-axis ABCD anomaly-detection
pipeline. The branch was cut from the frozen golden implementation
`golden-supcon030-v1` (`affb2f1`); cleanup removes retired experiments but
does not change the retained model, training, weighting, or evaluation code.

## Pipeline

1. `train_ae.py` trains an autoencoder on QCD only. Its weighted
   reconstruction error is the continuous nuisance/AE axis.
2. `train_hlt.py` trains the four-background classifier and its latent
   representation from PF-candidate tensors.
3. A real-vs-shuffled density-ratio critic sees
   `(latent, continuous AE score, class label)`. The encoder is penalized when
   its latent representation retains AE information.
4. `eval_abcd_nurd.py` converts the latent vector to the Mahalanobis-distance
   (MD) axis and measures weighted QCD ABCD closure against the AE axis.

The default campaign is the golden setup: SupCon 0.3, 20 training-only
nuisance strata for estimating weights, global uniform nuisance shuffling,
three critic updates per encoder update, immediate information weight 1, and
no ramp.

Generator weights are loaded from `weight_train.pt`/`weight_test.pt`.
Training combines them with class and nuisance-stratum balancing through
`utils/hlt_weights.py`; the resulting effective weights are used consistently
by AE training, classification, SupCon, critic training, and the encoder
information loss.

## Canonical Della run

From the repository root:

```bash
module load anaconda3/2025.12
conda activate disco
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt

bash slurm/launch_engineer_campaign.sh <new_run_tag> 0.3
```

This retrains both the AE and NURD model, then submits one dependent dual
evaluation. Outputs are written to:

```text
$BASE/checkpoints/hlt/hlt/ae_engineer_<run_tag>/
$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_<run_tag>/
$BASE/outputs/<run_tag>_eval/
├── held-out/
├── legacy/
└── evaluation_summary.json
```

The held-out evaluation fits the MD reference on saved training rows, selects
thresholds on saved validation rows, and reports closure on the independent
weighted Mequinna test set. The legacy evaluation preserves the old unweighted
same-sample comparison; it is diagnostic, not the primary result.

Monitor the printed job IDs with:

```bash
squeue -j <training_job>,<evaluation_job>
sacct -X -j <training_job>,<evaluation_job> \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit
```

If training finished but evaluation did not, reuse the checkpoints:

```bash
bash slurm/launch_engineer_eval.sh <existing_run_tag> stat-valid
```

That command writes a non-overwriting result to
`$BASE/outputs/<run_tag>_eval_stat_valid/{held-out,legacy}`.

## Retained experiment controls

The launcher keeps recent comparisons available without changing source:

```bash
NURD_EPOCHS=80 \
LR_SCHEDULE_EPOCHS=40 \
INFO_WARMUP_EPOCHS=5 \
INFO_RAMP_EPOCHS=10 \
CHECKPOINT_EVERY=5 \
CRITIC_SHUFFLE_MODE=global \
STATISTICALLY_VALID_CLOSURE=1 \
  bash slurm/launch_engineer_campaign.sh <new_run_tag> 0.3
```

Defaults reproduce the golden run: `NURD_EPOCHS=40`,
`LR_SCHEDULE_EPOCHS=40`, zero warm-up/ramp,
`CRITIC_SHUFFLE_MODE=global`, `CHECKPOINT_EVERY=5`, and statistically valid
closure enabled. `weighted_within_class` remains available only as the recent
conditional-shuffle comparison.

To compare saved epochs without touching held-out test data:

```bash
bash slurm/launch_validation_checkpoint_scan.sh <run_tag> 5 10 15 20 25 30 35 40
```

## Repository map

- `train_ae.py`, `train_hlt.py`: the only training entry points.
- `eval_abcd_nurd.py`: the only ABCD evaluation implementation.
- `dataset/hlt_smcocktail_dataset.py`: HLT tensor preprocessing and datasets.
- `models/`: AE, classifier/encoder, and critic definitions.
- `utils/hlt_weights.py`: generator/effective weighting and split metadata.
- `utils/hlt_density_ratio.py`: critic sampling and density-ratio losses.
- `slurm/`: canonical training, dual-evaluation, and checkpoint-scan jobs.
- `scripts/`: dual-evaluation summary and validation checkpoint scan.
- `event_displays/`: matched QCD, false-positive QCD, and TpTp displays.
- `tests/`: regression tests for data, weights, critic, schedules, evaluation,
  and checkpoint selection.

## Local checks

```bash
python -m pytest -q
bash -n slurm/*.sh slurm/*.sbatch event_displays/*.sbatch
```

The Della jobs use the existing `disco` conda environment. The compact
`requirements.txt` lists only direct Python dependencies for this retained
pipeline.
