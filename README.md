# HLT NURD anomaly detection

This is the HLT-only implementation of the two-axis ABCD anomaly-detection
pipeline. The branch was cut from the frozen golden implementation
`golden-supcon030-v1` (`affb2f1`). The initial cleanup removed retired
experiments without changing the retained pipeline; subsequent changes are
documented here, beginning with generator-only all-class AE training.

## Pipeline

1. `train_ae.py` trains an autoencoder on events from all four background
   classes. Its loss uses generator weights only—without class or nuisance
   balancing—and its reconstruction error is the continuous nuisance/AE axis.
2. `train_hlt.py` trains the four-background classifier and its latent
   representation from PF-candidate tensors.
3. A real-vs-shuffled density-ratio critic sees
   `(latent, continuous AE score, class label)`. The encoder is penalized when
   its latent representation retains AE information. A master NURD switch can
   disable both critic training and this penalty, or activate both sharply at
   a selected epoch.
4. `eval_abcd_nurd.py` converts the latent vector to the Mahalanobis-distance
   (MD) axis and measures weighted QCD ABCD closure against the AE axis.

The campaign uses SupCon 0.3, 20 training-only nuisance strata, global uniform
nuisance shuffling, and three critic updates per encoder update when NURD is
active. NURD is disabled by default. When enabled, start epoch `0` activates it
in epoch 1; a positive start epoch produces a sharp off-to-on transition with
no ramp. Weight estimation uses fixed-width bins in a training-fitted scaled
`log1p(AE loss)` coordinate and clips the top 0.5% of effective event weights
separately in each class before restoring equal class mass. The AE weighting
is the all-class, generator-only update described above.

Generator weights are loaded from `weight_train.pt`/`weight_test.pt`. The AE
uses those physics weights directly across the full all-class sample (non-QCD
rows have unit generator weight). NURD training separately combines generator
weights with class and nuisance-stratum balancing through
`utils/hlt_weights.py`; those NURD effective weights are used consistently by
classification, SupCon, critic training, and the encoder information loss.

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
NURD_ENABLED=1 \
CRITIC_START_EPOCH=6 \
BALANCE_BINNING=log_fixed \
BALANCE_CLIP_QUANTILE=0.995 \
CHECKPOINT_EVERY=5 \
CRITIC_SHUFFLE_MODE=global \
STATISTICALLY_VALID_CLOSURE=1 \
  bash slurm/launch_engineer_campaign.sh <new_run_tag> 0.3
```

Current defaults use `NURD_EPOCHS=40`, `LR_SCHEDULE_EPOCHS=40`,
`NURD_ENABLED=0`, `CRITIC_START_EPOCH=0`,
`BALANCE_BINNING=log_fixed`, `BALANCE_CLIP_QUANTILE=0.995`,
`CRITIC_SHUFFLE_MODE=global`, `CHECKPOINT_EVERY=5`, and statistically valid
closure. With NURD disabled, training is classification plus the requested
SupCon loss and the critic is never optimized. To enable NURD immediately:

```bash
NURD_ENABLED=1 CRITIC_START_EPOCH=0 \
  bash slurm/launch_engineer_campaign.sh <new_run_tag> 0.3
```

To let classification/SupCon train alone for epochs 1--5 and switch NURD on
sharply in epoch 6:

```bash
NURD_ENABLED=1 CRITIC_START_EPOCH=6 \
  bash slurm/launch_engineer_campaign.sh <new_run_tag> 0.3
```

To recover the earlier weight estimator for a controlled comparison, use
`BALANCE_BINNING=weighted_quantile BALANCE_CLIP_QUANTILE=1.0`.
`weighted_within_class` remains available only as the recent
conditional-shuffle comparison.

To compare saved epochs without touching held-out test data:

```bash
bash slurm/launch_validation_checkpoint_scan.sh <run_tag> 5 10 15 20 25 30 35 40
```

## Staged diagnostic mode

Diagnostic mode uses the same AE, encoder, losses, critic, preprocessing, and
weight code as production, but runs them on a deterministic class-stratified
subset and saves every short stage. It does not open the held-out test sample
and it does not submit the normal dual evaluation.

```bash
bash slurm/launch_engineer_campaign.sh <new_diagnostic_tag> 0.3 --diagnostic
```

The default sequence is five AE epochs, five classifier/SupCon-only epochs,
three epochs with the global NURD critic sharply enabled, and a separate
three-epoch critic probe applied identically to frozen representations before
and after NURD. Defaults can be changed without editing source, for example:

```bash
DIAG_MAX_EVENTS=200000 \
DIAG_AE_EPOCHS=8 \
DIAG_SUPCON_EPOCHS=6 \
DIAG_NURD_EPOCHS=4 \
DIAG_CRITIC_PROBE_EPOCHS=4 \
  bash slurm/launch_engineer_campaign.sh <new_diagnostic_tag> 0.3 --diagnostic
```

The complete report is written to
`$BASE/outputs/<tag>_diagnostic/`. It contains the selected diagnostic inputs,
`diagnostic_summary.json`, a campaign manifest, and plot groups for data/AE,
SupCon-only, the frozen critic probe, post-NURD behavior, and direct comparison.
Nothing is sent to W&B during the run. Upload the completed report later with:

```bash
python scripts/upload_training_diagnostics.py \
  "$BASE/outputs/<tag>_diagnostic"
```

The diagnostic closure plots use only the saved training/validation split and
are intended to locate bugs, not to select a result after inspecting held-out
data.

## Repository map

- `train_ae.py`, `train_hlt.py`: the only training entry points.
- `eval_abcd_nurd.py`: the only ABCD evaluation implementation.
- `dataset/hlt_smcocktail_dataset.py`: HLT tensor preprocessing and datasets.
- `models/`: AE, classifier/encoder, and critic definitions.
- `utils/hlt_weights.py`: generator/effective weighting and split metadata.
- `utils/hlt_density_ratio.py`: critic sampling and density-ratio losses.
- `slurm/`: canonical training, diagnostic, dual-evaluation, and checkpoint-scan jobs.
- `scripts/`: diagnostic drivers, dual-evaluation summary, and validation scan.
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
