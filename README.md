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
export SMOKE_OUT=/hdfs/store/user/mbileska/anomalyNano/2025/nurd_tensor_smoke_qcd_v2
export SMOKE_STAGE=/scratch/mbileska/nurd_tensor_staging

python3 convert_background_nanoaod.py @"$FILELIST" \
  --weights "$FULL_WEIGHTS" \
  --output-dir "$SMOKE_OUT" \
  --staging-dir "$SMOKE_STAGE" \
  --max-files 3 --seed 42 --test-fraction 0.2

ls -lh "$SMOKE_OUT/conversion_report.json"
```

The smoke sample is deliberately a prefix of the canonical file list so its
rows still align with the prefix of the full weight payload.

### 3. Convert all 2,582,935 events through HTCondor

The full conversion is too intensive for a Falcon login node. From
`axoScoutingNano`, create a self-contained batch wrapper and submit file:

```bash
export CMSSW_WORK=/afs/hep.wisc.edu/user/mbileska/axo_sample_work/CMSSW_16_1_0_pre4/src/DPGAnalysis/L1TNanoAOD/test/axoScoutingNano

cat > "$CMSSW_WORK/run_nurd_tensor_conversion.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc14
cd /afs/hep.wisc.edu/user/mbileska/axo_sample_work/CMSSW_16_1_0_pre4/src
eval "$(scram runtime -sh)"
export PYTHONNOUSERSITE=1
cd DPGAnalysis/L1TNanoAOD/test/axoScoutingNano

python3 convert_background_nanoaod.py @production_filelists/all.txt \
  --weights production_reports/background_physics_weights.npz \
  --output-dir /hdfs/store/user/mbileska/anomalyNano/2025/nurd_tensor_v1 \
  --staging-dir "${_CONDOR_SCRATCH_DIR:?HTCondor scratch directory is not set}" \
  --expected-files 2008 --expected-events 2582935 \
  --seed 42 --test-fraction 0.2
EOF
chmod 700 "$CMSSW_WORK/run_nurd_tensor_conversion.sh"

cat > "$CMSSW_WORK/nurd_tensor_conversion.sub" <<EOF
universe = vanilla
executable = $CMSSW_WORK/run_nurd_tensor_conversion.sh
initialdir = $CMSSW_WORK
output = $CMSSW_WORK/production_reports/nurd_tensor_conversion.out
error = $CMSSW_WORK/production_reports/nurd_tensor_conversion.err
log = $CMSSW_WORK/production_reports/nurd_tensor_conversion.condor.log
request_cpus = 1
request_memory = 8000MB
request_disk = 45GB
should_transfer_files = NO
queue
EOF

export CONVERTED=/hdfs/store/user/mbileska/anomalyNano/2025/nurd_tensor_v1
if [[ -e "$CONVERTED" ]]; then
  echo "Refusing to submit: output already exists at $CONVERTED"
else
  condor_submit "$CMSSW_WORK/nurd_tensor_conversion.sub"
fi
```

Monitor it with `condor_q "$USER"`. After it leaves the queue, require both a
zero exit code in the Condor log and a final report:

```bash
grep -E "return value|Job terminated" production_reports/nurd_tensor_conversion.condor.log
tail -50 production_reports/nurd_tensor_conversion.err
ls -lh /hdfs/store/user/mbileska/anomalyNano/2025/nurd_tensor_v1/conversion_report.json
du -sh /hdfs/store/user/mbileska/anomalyNano/2025/nurd_tensor_v1
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

export DATA_DIR=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/data/train3M_v1/converted_v1
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

# Opt-in comparison: split W/Z by generated decay channel (eight targets).
# The default remains TARGET_SCHEME=parent4 for the original baseline.
TARGET_SCHEME=decay8 \
  bash slurm/launch_train3m_campaign.sh train3m_v1_decay8_diagnostic 0.3 --diagnostic

# After establishing a classifier-only baseline, continue from that exact
# checkpoint with a fresh conditional critic and gradual information ramp.
BASELINE_TAG=<completed_decay8_supcon_tag>
AE_EXP="ae_engineer_${BASELINE_TAG}" \
REUSE_AE=1 \
WARM_START_CKPT="/scratch/gpfs/IOJALVO/mb7126/nurd_hlt/checkpoints/hlt/hlt/hlt_nurd_engineer_${BASELINE_TAG}/checkpoint_main.pth.tar" \
TARGET_SCHEME=decay8 \
NURD_ENABLED=1 \
CRITIC_START_EPOCH=0 \
CRITIC_SHUFFLE_MODE=weighted_within_class \
CRITIC_WARMUP_EPOCHS=3 \
LAMBDA_INFO_RAMP_EPOCHS=10 \
LAMBDA_INFO=0.1 \
SELECTION_AFTER_RAMP=1 \
TRAIN_LR=3e-5 \
NURD_EPOCHS=30 \
LR_SCHEDULE_EPOCHS=30 \
CHECKPOINT_EVERY=1 \
  bash slurm/launch_train3m_campaign.sh train3m_v1_decay8_nurd_ramp 0.3
```

The same recommended configuration is available through a shorter wrapper:

```bash
bash slurm/launch_warm_start_nurd.sh \
  <completed_decay8_supcon_tag> \
  <new_nurd_run_tag> \
  0.3
```

Every wrapper default remains overrideable. For example,
`LAMBDA_INFO_RAMP_EPOCHS=0` switches the penalty on sharply after the
configured critic warm-up, while `LAMBDA_INFO=0.05` lowers the final pressure.
Set both warm-up and ramp lengths to zero to reproduce immediate activation.

`decay8` maps the stored fine-category metadata to QCD, ttbar,
W$\to e\nu/\mu\nu/\tau\nu$, and Z$\to ee/\mu\mu/\tau\tau$. The W/Z HT
bins remain within their decay class, while QCD hard-pT bins and ttbar decay
modes remain diagnostic strata. No NanoAOD regeneration or tensor conversion
is performed. Checkpoints record the selected taxonomy, and evaluation derives
the same labels from `background_category`. The legacy tensor has no such
metadata, so decay8 campaigns intentionally skip the legacy comparison and
evaluate only the independent converted holdout.

The continuation launcher refuses incompatible warm starts. It verifies the
exact data signature, aligned physics-weight checksum, AE checksum, target
taxonomy, train/validation row indices, architecture, and weighting settings.
Only the classifier/encoder parameters are restored; the critic and both
optimizers start fresh. `REUSE_AE=1` avoids retraining the validated AE.
During `CRITIC_WARMUP_EPOCHS` the critic trains while the encoder information
coefficient is zero. The coefficient then rises linearly to `LAMBDA_INFO`
over `LAMBDA_INFO_RAMP_EPOCHS`. With `SELECTION_AFTER_RAMP=1`, best-checkpoint
selection begins only after the full requested penalty is reached; periodic
snapshots retain the entire trajectory.

For a later evaluation retry or checkpoint scan, first run
`source slurm/train3m_environment.sh`; this exports the train/test manifests
and their matching NPZ weights so the generic launchers cannot silently fall
back to the old Mequinna inputs. The launcher also requires the validator's
`converted_v1/FULL_VALIDATION_PASSED` marker before it will submit any job.

### Parent-balanced AE ablation

The AE optimization measure is independently selectable with
`AE_WEIGHTING_SCHEME=physics|parent_balanced`. `physics` is the default and
preserves the literal cross-section mixture. `parent_balanced` fits factors on
the saved AE training split so QCD, W+jets, Z+jets, and ttbar each contribute
25% of the AE loss and feature-scaling measure. It preserves generator-weight
ratios inside each parent and applies the training factors unchanged to
validation. The original aligned physics weights remain in the checkpoint and
are still used for physics-weighted diagnostics and final evaluation.

The controlled full comparison has a dedicated launcher:

```bash
bash slurm/launch_parent_balanced_ae_ablation.sh \
  train3m_v1_decay8_ae_parentbal_v1 0.3
```

It uses the validated train3M tensors, trains a new parent-balanced AE, trains
the decay8 classifier for 50 epochs with NURD disabled, and submits one
dependent statistically filtered held-out evaluation. Existing outputs are
never overwritten. The AE checkpoint records its scheme, fitted parent
factors, exposure and physics class masses, and both exposure-weighted and
physics-weighted validation losses.

For the follow-up classifier isolation test, `BALANCE_STRATA=1` has explicit
class-only semantics: it equalizes target-class mass but does not use AE score
bins to estimate event weights. Values of two or more retain the continuous
nuisance-stratified behavior.

The taxonomy-only parent4 control reuses the completed parent-balanced AE,
uses class-only classifier weighting, and disables NURD. The target map is the
intended modeling change. Class balancing is consequently refit over the four
parents, giving each parent 25% exposure instead of allowing W and Z to gain
extra total mass merely because decay8 gives each three child labels. By
default it expects the established parent-balanced AE experiment;
`PARENT_BALANCED_AE_EXP` can name an equivalent validated checkpoint:

```bash
bash slurm/launch_parent4_taxonomy_control.sh \
  train3m_v1_parent4_parentbal_classonly_v1 0.3
```

The classifier starts from the aligned all-event physics weights, then
equalizes the four parent totals. Inside each parent it preserves the
generator-weight ratios and clips only the top 0.5% of effective weights, as
in the decay8 class-only control. Fine decay, HT, and QCD hard-pT categories
remain in the converted metadata for stratified splitting and diagnostics;
they are not discarded by the parent4 target map.

After that classifier-only control has completed, the full parent4 NURD
continuation is launched with:

```bash
bash slurm/launch_parent4_full_nurd.sh \
  <completed_parent4_baseline_tag> <new_parent4_nurd_tag> 0.3
```

This reuses both the parent-balanced AE and the selected classifier checkpoint.
It retains class-only weighting (25% mass per parent), uses within-class
weighted critic shuffling, trains the critic alone for five epochs, ramps the
information coefficient to 0.1 over fifteen epochs, and then trains for twenty
epochs at full strength.  The continuation learning rate is `3e-5`, checkpoint
selection is disabled until the warm-up and ramp finish, and every epoch is
saved for validation-only checkpoint scans. Environment overrides ending in
`_OVERRIDE` expose the schedule intentionally without allowing unrelated shell
settings to contaminate the parent4 contract.

## Full physics-AE / decay8 / NURD comparison

`slurm/launch_physics_decay8_full_nurd.sh` submits five sequential jobs on the
validated train3M split: a **fresh physics-weighted AE**, a decay8
classifier/SupCon baseline, its statistically filtered held-out evaluation, a
warm-started NURD continuation, and a second held-out evaluation. The AE uses
the aligned cross-section-times-generator weights with no parent balancing.
The classifier, SupCon, and NURD losses use class-only balance fitted on the
training split (`BALANCE_STRATA=1`), preserving physical ratios within each
decay target. The outer held-out split always uses the original physics weights.

On Della after pulling the campaign commit:

```bash
cd /home/mb7126/nurd_hlt
git pull --ff-only origin dismembered-nurd-implementation
RUN_TAG="train3m_physics_decay8_full_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_physics_decay8_full_nurd.sh "$RUN_TAG" 0.3
```

Defaults are AE 150 epochs maximum (patience 25), classifier-only 80 maximum,
then NURD 80 maximum (patience 20). The NURD critic trains for five epochs
without encoder penalty, the penalty ramps over fifteen epochs to 0.1, and
early stopping is forbidden until **25 more full-strength epochs** have been
completed (earliest stop epoch 45). Best/last and every-fifth-epoch classifier
checkpoints are kept. Each stage is a separate 20-hour Slurm allocation and
starts only when its predecessor exits successfully; the five jobs do not
occupy GPUs concurrently. Increase a limit via `AE_EPOCHS`,
`BASELINE_EPOCHS`, or `NURD_EPOCHS` before launching, but check the per-job
walltime. The two held-out evaluations are for the predeclared comparison,
not for choosing a post-ramp checkpoint; inspect validation first if tuning.

### Full-data lambda=1 NURD control, with interim validation scans

This control reuses the **same physics-weighted AE and selected NURD-off
decay8 classifier** from `train3m_physics_decay8_full_20261104_142222`.
It trains on the complete 2,066,347-event outer-training split with the same
class-only loss weights, split, architecture, seed, optimizer, five-epoch
critic warm-up, fifteen-epoch ramp, and 25-epoch minimum after the ramp as the
lambda=0.1 continuation. The controlled training change is
`LAMBDA_INFO=1.0`; model/critic snapshots are saved every **two** epochs. The
trainer still reports classification validation every epoch. No held-out
evaluation is submitted automatically: scan saved checkpoints on the saved
training/validation split first.

```bash
cd /home/mb7126/nurd_hlt
git pull --ff-only origin dismembered-nurd-implementation
BASE_RUN=train3m_physics_decay8_full_20261104_142222
RUN_TAG="train3m_physics_decay8_lambda1_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_physics_decay8_lambda1_control.sh "$BASE_RUN" "$RUN_TAG"
```

Once the training log says a periodic checkpoint has been saved, a small
validation-only scan can run while training continues. For example, after
epochs 20 and 30 have finished:

```bash
source slurm/train3m_environment.sh
export AE_EXP="ae_engineer_${BASE_RUN}"
bash slurm/launch_validation_checkpoint_scan.sh "$RUN_TAG" 20 30
```

The scan fits QCD MD on saved training rows and reports weighted QCD closure
on disjoint saved validation rows; it does not open the outer held-out test.
Each scan is a separate GPU job, so pick a few informative epochs rather than
scanning every two-epoch checkpoint. The training allocation is capped at
20 hours; periodic checkpoints make partial progress inspectable, but the
trainer currently has no optimizer-state resume if that cap is reached.

### Score-level conditional MD calibration diagnostic

`scripts/run_conditional_md_calibration.py` tests whether the observed AE--MD
dependence is stable enough to calibrate **without retraining either model**.
It fits the usual physics-weighted QCD PCA/MD reference and a piecewise
weighted empirical CDF of MD conditional on log10(AE loss), using only QCD
events in the checkpoint's saved *training* split. The default 20 AE bins
carry equal QCD physics mass and must each have ESS at least 200. The fitted
map is frozen before application to the disjoint saved *validation* QCD rows.

The report compares raw and calibrated ABCD nonclosure with identical region
ESS requirements and also reports the median on grid cells valid in **both**
scans. It saves bin-level effective statistics, a portable calibration NPZ,
validation scores/weights, and side-by-side grid and curve plots. It never
opens the outer test split, and its outcome does not establish signal
sensitivity: the transformation uses AE loss and can change signal ranking.
Do not tune bin count or select a final analysis on the outer test sample.

On Della, to compare the selected full-campaign lambda=0.1 and lambda=1.0
checkpoints under this exact validation protocol:

```bash
cd /home/mb7126/nurd_hlt
git pull --ff-only origin dismembered-nurd-implementation
BASE_RUN=train3m_physics_decay8_full_20261104_142222
AE_EXP="ae_engineer_${BASE_RUN}"
bash slurm/launch_conditional_md_calibration.sh "${BASE_RUN}_nurd" "$AE_EXP"
bash slurm/launch_conditional_md_calibration.sh \
  train3m_physics_decay8_lambda1_20261107_132516 "$AE_EXP"
```

Each command submits one validation-only GPU job. Its printed output directory
contains `conditional_md_report.json`, `conditional_md_fit_train_qcd.npz`,
`validation_qcd_scores.npz`, and `plots/`. Raw-vs-calibrated closure on the
same validation events is the meaningful first test; compare the two models
only after checking their bin ESS and the full closure curves.

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

Those sharp-switch commands are retained for reproducing earlier baselines.
For a new physics result, prefer the warm-started conditional-ramp workflow
above. Setting `LAMBDA_INFO_RAMP_EPOCHS=0` preserves the historical on/off
switch, so the ramp remains a toggleable experiment control.

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

The safer conditional-ramp diagnostic is available without source edits:

```bash
TARGET_SCHEME=decay8 \
DIAG_SUPCON_EPOCHS=30 \
DIAG_NURD_EPOCHS=15 \
CRITIC_SHUFFLE_MODE=weighted_within_class \
CRITIC_WARMUP_EPOCHS=3 \
LAMBDA_INFO_RAMP_EPOCHS=7 \
LAMBDA_INFO=0.1 \
SELECTION_AFTER_RAMP=1 \
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

## Train3M lambda and legacy-domain audits (opt-in)

From the Della checkout after pulling this commit:

```bash
BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
bash slurm/launch_qcd_lambda_pair_audit.sh qcd_lambda01_vs_1_$(date +%Y%m%d_%H%M%S)
bash slurm/launch_physics_decay8_lambda25_control.sh \
  train3m_physics_decay8_full_20261104_142222 \
  train3m_physics_decay8_lambda25_$(date +%Y%m%d_%H%M%S)
bash slurm/launch_decay8_parent4_legacy_eval.sh \
  hlt_nurd_engineer_train3m_physics_decay8_lambda1_20261107_132516 \
  decay8_lambda1_mequinna_$(date +%Y%m%d_%H%M%S)
```

The first job recomputes per-event AE and QCD-only MD scores for both
checkpoints on the identical validated train3M test and validation rows. It
checks exact event IDs, weights, AE scores, and saved ABCD region yields. The
paired report under `$BASE/outputs/<audit_tag>/paired/` compares fixed
validation-fitted cuts at matched weighted QCD A efficiencies, performs a
paired Poisson event bootstrap, and shows validation-to-test weighted MD
shifts within QCD 15--50 GeV and reconstructed HT thirds. It is diagnostic:
the outer test set has already been inspected, and this does not replace a
new blind sample or account for cross-section/systematic uncertainty.

The second launcher reuses the exact physical-weighted AE and baseline
classifier, with class-only classifier balancing. Relative to the lambda-one
control, it changes only the maximum information penalty from 1 to 2.5. It
submits training plus an after-success statistically-valid held-out evaluation.
The checkpoint is under `$BASE/checkpoints/hlt/hlt/hlt_nurd_engineer_<tag>/`;
regular closure plots and `evaluation_summary.json` go under
`$BASE/outputs/<tag>_eval_stat_valid/`.

The last job is a standalone Mequinna domain-transfer diagnostic. It does not
retrain or relabel a checkpoint. It adds decay8 probabilities into parent4
probabilities, refits the QCD MD reference on legacy training QCD, chooses
thresholds on a disjoint legacy validation split, and reports on the legacy
test split. Results and regular closure plots go under
`$BASE/outputs/<tag>_legacy_parent4/`. Bare legacy weights remain QCD-only
with unit weights on non-QCD rows; there are no event IDs to verify alignment.
Do not compare its closure directly to train3M held-out closure as though the
input domains and physics measures were identical.

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
