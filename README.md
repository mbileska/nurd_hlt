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

Generator/physics weights are loaded from `.pt` or `.npz` payloads. New
production payloads declare `weight_scope=all_events`, carry the composite
event ID, and are applied to every background row only after exact alignment
with the converted sample. Bare legacy tensors keep their historical QCD-only
meaning. The AE uses the physics weights directly across the full all-class
sample. NURD training separately combines them with class and nuisance-stratum balancing through
`utils/hlt_weights.py`; those NURD effective weights are used consistently by
classification, SupCon, critic training, and the encoder information loss.

## 2025 train3M conversion

`scripts/convert_background_nanoaod.py` is the production bridge from the
validated 2,008-file NanoAOD campaign to this repository. It streams ROOT
files, verifies every row against the composite
`(Background_sample, run, luminosityBlock, event)` ID in the physics-weight
NPZ, and makes a deterministic 80/20 split stratified by the fine background
category. It writes memory-mapped NumPy arrays plus `sample.json`, not a
second monolithic `.pt` copy.

The data contract is:

- `pf`: 400 highest-pT `ScoutingParticle` candidates with
  `[pt, eta, phi, dxy, dxysig, is_pf, pdgId]`; `is_pf=1` for active rows and
  zero for padding. This intentionally avoids guessing the undocumented
  high-level-object duplication present in the old Mequinna PF tensor.
- `obj`: 10 PF jets, 4 vertex muons, 4 electrons, 4 photons, and 1 MET row,
  each `[pt, eta, phi, type]`, matching the legacy 23-slot AE layout.
- `label`: NURD labels `Z=0`, `QCD=1`, `ttbar=2`, `W=3`.
- `background_category`: the original exclusive generator-level subcategory;
  it is retained for audits and splitting but is not a training target.
- `event_id`: the four-column composite ID used to prove weight alignment.
- `weight_{train,test}.npz`: the exact all-background physics measure sliced
  with the same split. The cross section and generator-weight calculation was
  already performed by the CMSSW exporter; conversion never recomputes or
  class-balances it.

The conversion report records multiplicity truncation and the retained
scalar-pT fraction. The validator independently checks finiteness, padding,
pT ordering, PDG IDs, split uniqueness, category coverage, and exact
data/weight event-ID alignment.

### 1. Copy the standalone converter from Della to login.hep.wisc.edu

Run on Della after pulling this branch:

```bash
cd /home/mb7126/nurd_hlt
git switch dismembered-nurd-implementation
git pull --ff-only origin dismembered-nurd-implementation

export FALCON=mbileska@login.hep.wisc.edu
export REMOTE_WORK=/afs/hep.wisc.edu/user/mbileska/axo_sample_work/CMSSW_16_1_0_pre4/src/DPGAnalysis/L1TNanoAOD/test/axoScoutingNano

scp scripts/convert_background_nanoaod.py \
  "$FALCON:$REMOTE_WORK/convert_background_nanoaod.py"
```

Then log into Falcon and prepare its existing CMSSW area:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc14
cd /afs/hep.wisc.edu/user/mbileska/axo_sample_work/CMSSW_16_1_0_pre4/src
cmsenv
export PYTHONNOUSERSITE=1

cd DPGAnalysis/L1TNanoAOD/test/axoScoutingNano
python3 -c 'import numpy, awkward, uproot; print("conversion environment OK")'
```

### 2. Run a small QCD-prefix smoke conversion

```bash
export CMSSW_WORK=/afs/hep.wisc.edu/user/mbileska/axo_sample_work/CMSSW_16_1_0_pre4/src/DPGAnalysis/L1TNanoAOD/test/axoScoutingNano
export FILELIST="$CMSSW_WORK/production_filelists/all.txt"
export FULL_WEIGHTS="$CMSSW_WORK/production_reports/background_physics_weights.npz"
export SMOKE_OUT=/hdfs/store/user/mbileska/anomalyNano/2025/nurd_tensor_smoke_qcd_v1

python3 convert_background_nanoaod.py @"$FILELIST" \
  --weights "$FULL_WEIGHTS" \
  --output-dir "$SMOKE_OUT" \
  --max-files 3 --seed 42 --test-fraction 0.2

ls -lh "$SMOKE_OUT/conversion_report.json"
```

The smoke sample is deliberately a prefix of the canonical file list so its
rows still align with the prefix of the full weight payload.

### 3. Convert and validate all 2,582,935 events in tmux

Start a persistent shell with `tmux new -s train3m_convert`, repeat the Falcon
CMSSW setup in step 1 inside it, and run from `axoScoutingNano`:

```bash
export CMSSW_WORK=/afs/hep.wisc.edu/user/mbileska/axo_sample_work/CMSSW_16_1_0_pre4/src/DPGAnalysis/L1TNanoAOD/test/axoScoutingNano
export FILELIST="$CMSSW_WORK/production_filelists/all.txt"
export FULL_WEIGHTS="$CMSSW_WORK/production_reports/background_physics_weights.npz"
export CONVERTED=/hdfs/store/user/mbileska/anomalyNano/2025/nurd_tensor_v1
set -o pipefail

time python3 convert_background_nanoaod.py @"$FILELIST" \
  --weights "$FULL_WEIGHTS" \
  --output-dir "$CONVERTED" \
  --expected-files 2008 \
  --expected-events 2582935 \
  --seed 42 --test-fraction 0.2 \
  2>&1 | tee "$CMSSW_WORK/production_reports/nurd_tensor_conversion.log"

ls -lh "$CONVERTED/conversion_report.json"
du -sh "$CONVERTED"
```

The full PyTorch-based validator is run on Della after the copy, where the
NURD environment and repository exist. The standalone converter itself still
checks every ROOT/weight event ID and every converted chunk before marking the
conversion complete.

The converter refuses to overwrite a non-empty destination. If an interrupted
attempt leaves `.conversion_incomplete`, move that whole directory aside and
start with a fresh destination rather than mixing partial and final arrays.

### 4. Copy to Della and validate there

Run on Della:

```bash
module load anaconda3/2025.12
conda activate disco
cd /home/mb7126/nurd_hlt
git switch dismembered-nurd-implementation
git pull --ff-only origin dismembered-nurd-implementation

export DATA_DIR=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/train3M_v1
mkdir -p "$DATA_DIR"
scp -rp \
  mbileska@login.hep.wisc.edu:/hdfs/store/user/mbileska/anomalyNano/2025/nurd_tensor_v1/. \
  "$DATA_DIR/"

python scripts/validate_converted_backgrounds.py \
  --input-dir "$DATA_DIR" \
  --expected-files 2008 \
  --expected-events 2582935 \
  --expected-categories 0-23,27-36
```

### 5. Launch the diagnostic, then the full AE/NURD campaign

```bash
cd /home/mb7126/nurd_hlt
module load anaconda3/2025.12
conda activate disco

# Short end-to-end test on a deterministic subset.
bash slurm/launch_train3m_campaign.sh train3m_v1_diagnostic 0.3 --diagnostic

# After the diagnostic report is healthy, train with NURD enabled from epoch 6.
NURD_ENABLED=1 CRITIC_START_EPOCH=6 \
  bash slurm/launch_train3m_campaign.sh train3m_v1_nurd 0.3
```

For a later evaluation retry or checkpoint scan, first run
`source slurm/train3m_environment.sh`; this exports the train/test manifests
and their matching NPZ weights so the generic launchers cannot silently fall
back to the old Mequinna inputs.

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
