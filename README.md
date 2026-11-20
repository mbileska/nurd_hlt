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

To test whether the closure trend continues at lambda 5 without changing
SupCon or the model/weight contract, use
`slurm/launch_physics_decay8_lambda5_control.sh` with the same baseline run
and a fresh tag. This is a stress test: the selected lambda-2.5 checkpoint
closed much better but had only 31.3% decay8 validation balanced accuracy,
with several W/tau classes nearly absent. Do not change SupCon in the same
run or infer improved overall performance from closure alone.

The last job is a standalone Mequinna domain-transfer diagnostic. It does not
retrain or relabel a checkpoint. It adds decay8 probabilities into parent4
probabilities, refits the QCD MD reference on legacy training QCD, chooses
thresholds on a disjoint legacy validation split, and reports on the legacy
test split. Results and regular closure plots go under
`$BASE/outputs/<tag>_legacy_parent4/`. Bare legacy weights remain QCD-only
with unit weights on non-QCD rows; there are no event IDs to verify alignment.
Do not compare its closure directly to train3M held-out closure as though the
input domains and physics measures were identical.

## Pure-SupCon representation controls (opt-in)

`CE_WEIGHT=0` disables classification CE and freezes the classifier head;
`NURD_ENABLED=0` disables the critic. The encoder and projector are trained
only by supervised contrastive loss. The best checkpoint minimizes a
deterministic, mixed-class **validation SupCon loss**, not validation CE.
Classifier accuracy and the logit axis are not meaningful for these runs.

On Della, use the validated train3M split and the existing physics-weighted AE:

```bash
cd /home/mb7126/nurd_hlt
bash slurm/launch_pure_supcon_sweep.sh pure_sc_$(date +%Y%m%d_%H%M%S)
```

The launcher queues 0.1, 0.3, and 0.5 coefficients sequentially, each followed
by the ordinary weighted held-out AE-vs-QCD-MD evaluation. The usual
`hist2d_pca_md_kde.png` is in score space; the additional
`projector_pca_kde_by_class.png` directly visualizes the SupCon-optimized
representation. The latter fits PCA only for descriptive display and is not
used for model or threshold selection. Because CE is zero, changing the
SupCon coefficient only rescales the sole objective; it is **not** a test of
the CE/SupCon balance. Any differences largely reflect optimization scale
and its interaction with weight decay.

### Frozen encoder/projector separation audit

After all three pure-SupCon runs finish, submit **one** GPU analysis job:

```bash
bash slurm/launch_representation_audit.sh train3m_puresc_20261113_174126
```

This does not train or alter either network. It recovers each checkpoint's saved
training/validation indices, checks the input/physics-weight/AE fingerprints,
and requires identical splits across the three models. The outer test is not
opened. All saved training rows are used for a fixed-C multinomial logistic
probe, with a training-weight-fitted StandardScaler and the checkpoint's
training-fitted class/physics effective weights. Validation accuracy, macro
recall, and confusion matrices are reported under raw, training-effective, and
physical measures; these measures must not be conflated.

kNN uses a fixed seed, at most 5,000 uniformly selected training examples per
decay class, and k=25. The main vote is inverse-distance times the training
effective event weight, corrected for the support's class-dependent inclusion
probability. An inverse-distance-only control uses exactly the same neighbors.
Encoder kNN uses training-standardized coordinates; projector kNN uses the
normalized projector output directly (Euclidean/cosine-equivalent ranking).
No k, C, or score orientation is selected on validation. Probe convergence and
representation covariance eigenvalues are recorded.

Parent4 reports both summed decay probabilities and mapped decay argmax. This
is an aggregation diagnostic, **not** training a new parent4 probe. MD fits use
all training QCD with unmodified physics weights in each representation, with
the existing eigenvalue floor of 1e-6. Both the QCD reference and the AE/MD
percentile thresholds are training-fitted; disjoint validation reports weighted
ABCD closure. These grids are not directly identical to older held-out scans
whose percentile thresholds came from validation.

The launcher prints the output directory. It contains `audit_summary.json`,
`probe_metrics.csv`, and one directory per run, each with eight PNGs:

- `linear_confusion_{raw,training_effective}.png` and
  `knn_confusion_{raw,training_effective}.png`: encoder/projector by decay8/parent4;
- `md_discrimination.png`: raw and physics-weighted class-vs-QCD ROC AUC;
- `md_closure_validation.png`: matched reference-percentile grids and curves;
- `md_distributions.png`: per-class physics-weighted score distributions;
- `md_qcd_ae_plane.png`: genuinely physics-weighted QCD AE-vs-MD histograms.

Compact probe/MD fits and event-aligned validation embeddings, probabilities,
scores, weights, categories, and IDs are saved in NPZ files. No PF tensors or
training embeddings are copied. Good class separation does not by itself imply
QCD closure or signal sensitivity; this audit distinguishes representation,
scalar-score, and weighting effects before another network training run.

## AE-tail validation closure (CPU-only)

For a full-network baseline, the default is the selected lambda=1 decay8 model
`train3m_physics_decay8_lambda1_20261107_132516` (epoch 22), not a pure-SupCon
model. The existing physical AE and full-training-QCD MD reference are frozen.

```bash
bash slurm/launch_ae_tail_validation.sh lambda1_top10_$(date +%Y%m%d_%H%M%S)
```

One CPU job (4 cores, 12 GB, two-hour limit; **no GPU**) scores the AE on the
saved validation rows. It reuses `raw_md` from the existing lambda=1 cache under
`outputs/train3m_physics_decay8_lambda1_20261107_132516_conditional_md_20261108_154347/`.
Only the uncalibrated MD field is used; conditional calibration is **not**
applied. Cache checkpoint/AE hashes, row membership, physics weights, reference
population counts, and recomputed AE scores are checked before interpretation.
If the cache is missing or incompatible, the job refuses to proceed rather than
silently doing full CPU PF inference. `AE_TAIL_CACHE_DIR` can name another
compatible conditional-MD cache directory.

The default is now **`physics-all`**: use the original aligned, all-event
generator × cross-section weights to define the physical mixture of all four
generated parents (eight decay targets). Compute the mixture's weighted AE
90th percentile, then apply that one AE boundary to every validation event.
The weights are applied to the percentile distribution **before selection**,
not added only after ranking raw rows. No class balance, nuisance balance,
additional genWeight multiplication, score/input multiplication, or random
resampling is performed. Legacy QCD-only weight files are rejected in this
mode. Boundary ties are included and the achieved weighted fraction is reported.
The retained 10% refers to the physical mixture's mass, not the number of stored
MC rows. This cocktail is not a replacement for real zero-bias data.

Explicit opt-in controls remain available: `physics-qcd` computes the boundary
from weighted QCD alone, while `raw-all` retains exactly the highest-scoring 10%
of raw rows with deterministic source-row tie breaking. Neither is the default:

```bash
bash slurm/launch_ae_tail_validation.sh lambda1_physical_tail physics-qcd
```

All modes are exploratory validation diagnostics. Within the retained QCD
population, the AE and MD thresholds are rescanned at conditional weighted
percentiles 0.50--0.98. The original tail boundary is **not** used as the ABCD
AE split, which would empty C and D. The fixed reported working point is the
conditional median/median, not the best closure cell. The reference remains
the original, unrestricted training QCD reference; it is not refit in the tail.
Closure uses physics weights with the existing per-region ESS minimums
A/B/C=50 and D=500. Insufficient cells remain masked even if the entire tail
grid fails the requirements.

Results are in `outputs/<tag>_ae_tail_validation/`: `tail_summary.json`, six
plots under `plots/`, event-aligned AE-selection and QCD-score NPZ files, scan
arrays, and a category/region composition CSV when QCD survives. The plots are
`closure_scan_2d.png`, `cut_and_count_bkg_check.png`,
`closure_effective_statistics.png`, `qcd_ae_md.png`,
`selection_retention.png`, and `ae_selection.png`. The selection figure now has
three clearly separated panels using identical full-range AE bins:

- the generator/cross-section-weighted mixture, with one common all-parent
  normalization (so relative parent rates are preserved);
- weighted per-class shapes, normalized separately and labeled as shapes;
- raw per-class MC coverage, explicitly labeled as **not** the physical mixture.

Its dashed boundary is the all-class AE preselection, not the later ABCD split.
`ae_selection_histograms.npz` stores the numerical bin contents, including the
weighted mixture total and each parent's/decay class's contribution. The run
also writes `abcd_cut_coordinates.csv` with the actual AE/MD cut values in the
full and selected populations, and annotates their median splits on the plots.
The summary records the achieved mixture retention and original weight checksum.
No classification probe,
signal inference, model training, or held-out sample access occurs. Improved
tail closure would apply only to that selected population, not establish
global factorization or signal sensitivity.

Use a separate code checkout if other jobs are still queued/running: existing
launchers check their submitted commit and deliberately fail if their checkout
has changed underneath them.

### The complete usual evaluation plot suite for the selected tail

After a `physics-all` tail job completes, generate the normal per-class plots,
KDE, PCA, profiles, and standalone closure figures without changing the models,
AE selection, or saved ABCD thresholds:

```bash
bash slurm/launch_ae_tail_standard_plots.sh latest lambda1_tail_visuals cpu
# Or use a specific source directory and optionally any generic GPU:
bash slurm/launch_ae_tail_standard_plots.sh \
  "$BASE/outputs/<tag>_ae_tail_validation" lambda1_tail_visuals_gpu gpu
```

`latest` selects the newest completed **physics-all** tail report, not the
newest raw-MC diagnostic. The launcher prints the exact source directory.
The default is CPU-only (24-hour walltime cap, 4 cores, 48 GB); `gpu` requests
one generic GPU with an eight-hour cap, never specifically an A100. This job
is more work than the quick tail scan: the old scalar-QCD cache does not store
the all-class latents. It performs frozen inference on the full training QCD
reference and selected validation events, verifies the rebuilt QCD MD against
the original cache, then keeps the cached QCD values exactly. There is **no
network training** and no held-out background access. Benchmark TpTp overlays
are included if `data/hlt_signal_TpTp.pt` exists; `INCLUDE_SIGNAL=0` disables them.
The signal receives the same background-defined AE cut and does not affect the
mixture, reference, or percentile selection.

The output `outputs/<plot-tag>_standard_plots/plots/` contains the ordinary
filenames, including `hist2d_pca_md_kde.png`, `hist2d_pca_md_scatter.png`,
`hist2d_by_class_combined.png`, `hist2d_bkg.png`, each `hist2d_<class>.png`,
`pca_scatter_embeddings.png`, `pca_corner.png`, the four `profile_*.png` figures,
`closure_scan_2d.png`, and `cut_and_count_bkg_check.png` (plus `hist2d_TpTp.png`
when signal survives). Figure canvases, fonts, DPI, class colors, log axes,
and KDE contour heights match `eval_abcd_nurd.py`: score plots 8x6 inches,
histograms 6x5, closure grid 7.5x6.5, 200 DPI. The corner retains its usual wide
layout. Titles explicitly say validation, not held-out.

Unlike the historical raw-count displays, histograms and KDEs use the original
physics weights; profiles use weighted means and effective-statistics errors.
Class KDEs are conditional **shapes**, not relative parent yields. Scatter/PCA
plots use fixed-seed, per-class physics-weighted draws for display only;
resampled points are never used in ABCD, fitting, KDE, or uncertainty estimates.
The PCA reference is fitted on physics-weighted training QCD, not selected
validation. Closure retains the original effective-count mask. Standard display
limits (100% closure color cap, ratio axis 0--1.5) are preserved; the source
summary's numerical values remain uncapped.

`plot_suite_metadata.json` records the weighting/display protocol and cache
agreement; `selected_validation_scores.npz` and `training_qcd_reference.npz`
retain the compact all-class scores/latents and fitted reference for future
plotting. No PF tensors are copied and no source tail outputs are overwritten.

## Full-sample QCD-MD versus 1-P(QCD) evaluation

This standalone, evaluation-only control defaults to the selected full-model
lambda=1 checkpoint (`train3m_physics_decay8_lambda1_20261107_132516`) and its
exact physics-weighted AE. It runs `eval_abcd_nurd.py` twice, changing axis 2
only via the existing `--axis2_logit` flag. There is **no top-10% selection,
resampling, conditional calibration, or retraining**. Both modes use the full
converted train3M held-out sample and original aligned physics weights.
QCD MD is fit on the saved training QCD; each mode selects its own usual cuts
on saved validation QCD, then applies those numerical cuts to held-out data.
This deliberately reproduces ordinary evaluation, not matched efficiencies.

```bash
cd /home/mb7126/nurd_hlt
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
TAG="lambda1_md_vs_pqcd_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_abcd_axis_comparison.sh "$TAG" gpu
```

The GPU is generic (no A100 constraint); pass `cpu` instead for CPU-only
inference. One job evaluates the modes sequentially. `MODEL_CKPT` and
`AE_CKPT` may explicitly override the checkpoint paths; the evaluator checks
the exact AE hash and saved reference split. The optional existing TpTp
benchmark gets the usual plots if available; use `INCLUDE_SIGNAL=0` to omit it.

Results are in `$BASE/outputs/${TAG}_axis_comparison/`:

- `qcd_md/held-out/plots/` and `one_minus_pqcd/held-out/plots/`: usual evaluator figures.
- Each mode also has `diagnostics.json` and `abcd_thresholds.json` under `held-out/`.
- `axis_comparison.csv` and `.json`: closure, predicted/observed with sumw2
  uncertainty, thresholds, grid summaries, and actual weighted QCD A efficiency.
- `comparison_manifest.json`: commands and immutable input file hashes.

The comparison fails if accepted event masks, sample signatures, physical
weights, validation selection masks, or conserved QCD totals differ. The
normal evaluator's existing plot/count conventions are retained: raw-count
histograms/KDEs are not substituted for physics-weighted closure. Grid
summaries are descriptive; held-out data never choose the working point.
No existing output is overwritten.

## Training-defined top-10% pilot preparation (no training)

```bash
bash slurm/launch_tail_pilot_preparation.sh "lambda1_tailprep_$(date +%Y%m%d_%H%M%S)"
```

This small CPU job scores only the frozen AE, not the PF encoder. It uses
the lambda=1 checkpoint's exact saved classifier train/validation partition
of `converted_v1/train/sample.json`. The 90th percentile is fit on **training
rows only**, using original all-background physics weights without class
balancing. The same numerical cut is applied to validation; no held-out file
is read. Training nuisance arithmetic is used, and progress/ETA is printed.

Results under `$BASE/outputs/<tag>_pilot_preparation/` include:

- `preparation_report.json`: frozen threshold, achieved retention including
  boundary ties, class-coverage checks, provenance, physical and optimization
  ESS, and low-ESS warnings. `class_coverage_ready` is not a closure guarantee.
- `class_statistics.csv`: all eight classes in each selected split, raw
  counts, weighted yields, physical ESS/N, and class-balanced statistics.
- `pilot_selection.npz`: original and selected split indices, event IDs,
  AE scores, physical weights, and proposed effective optimization weights.
- `pilot_preparation.pth`: train-fitted class-only balancing/clipping spec
  and provenance. It does not contain a newly trained network.

The optimization-weight preview uses the existing class-only recipe with
0.995 clipping, fit on selected training rows and reused on validation. It
never changes the physical weights used for selection or evaluation. No
training jobs are submitted; inspect the report before launching the pilot.

If an older preparation wrote `pilot_selection.npz`, `pilot_preparation.pth`,
and the CSV but failed to write JSON (e.g. a NumPy 2 float32 serialization
error), do not repeat inference. Recover the missing report with:

```bash
python -m scripts.recover_tail_preparation_report \
  --preparation-dir /path/to/existing_pilot_preparation \
  --ae-checkpoint "$BASE/checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth"
```

Recovery verifies checkpoint hashes, original cached weight/ID checksums,
saved and selected indices, the frozen threshold, and per-class statistics.
It creates only the missing JSON and refuses to overwrite an existing report.
The NPZ, PTH, CSV, scores, weights, and split remain unchanged. Future
preparations explicitly normalize NumPy scalar serialization and validate
JSON before writing output artifacts.

## Frozen preparation train/validation split audit

```bash
bash slurm/launch_tail_split_audit.sh latest "lambda1_split_audit_$(date +%Y%m%d_%H%M%S)"
```

`latest` resolves a completed pilot preparation. An explicit preparation
directory can be used instead. The audit requests one CPU core, 8 GB, and
30 minutes, with no explicit partition or GPU request. It reads the original
outer-training manifest and the prepared AE scores, never held-out data.
Only a small deterministic object sample is rescored for arithmetic parity;
there is no PF-encoder inference, retraining, or replacement split.

The audit fails on checkpoint/sample/weight/ID/label/split inconsistencies,
duplicate composite IDs, unreproducible selection, scaler differences, or
AE scoring mismatches. It checks reproduction of the saved classifier split,
seed sensitivity, and reconstructs AE membership from saved AE split metadata.
Historical AE checkpoints did not store row indices: reconstruction is labeled
as such and checked against saved parent-process mass summaries, not presented
as independently proven historical membership.

Outputs in `$BASE/outputs/<tag>_split_audit/`:

- `split_audit_summary.json`: integrity checks, scoring parity, QCD retention,
  overlap, composition-adjusted comparisons, and uncertainty summaries.
- CSVs for all QCD AE-input coordinates, category/weight-stratum composition,
  and reconstructed AE/classifier membership overlap.
- Eight plots under `plots/`: weighted AE and hardness CDFs, largest input
  shifts, membership retention, category composition, retention by weight,
  bootstrap gap, and randomized-assignment reference gap.
- `uncertainty_replicates.npz`: bootstrap and diagnostic resplit results.

The event-level Poisson bootstrap refits the all-background training threshold
in each replicate while holding the trained AE fixed. It describes conditional
MC precision, not uncertainty from retraining or a formal null test for the
weight-balanced splitter. Random fine-category-stratified reassignments of
frozen scores are only a diagnostic: both sides mix AE-seen and AE-unseen rows
and are not new independent validation. Common-composition adjustments are
descriptive only and do not change physical evaluation weights.

## Opt-in random-category AE-only control

```bash
TAG="ae_phys_randomsplit_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_ae_random_control.sh "$TAG"
```

This stays on the current branch and does not change the default legacy AE
split or any classifier/NURD campaign. One generic GPU is requested, with no
explicit partition, GPU model, exclusion list, or node constraint. The job
requests 4 CPU cores, 16 GB host memory, and a six-hour limit. It trains one
fresh AE, not SupCon/classifier/NURD. The architecture is copied from the
existing physical AE; optimizer defaults are those of the original full AE
campaign: Adam 1e-3, batch 4096, cosine schedule, 150 maximum epochs, patience
25. `AE_EPOCHS`/`AE_PATIENCE` may explicitly override those budgets.

The complete original outer-training sample is randomly partitioned **by
event within each fine generator category**, with no weight sorting:

- 90% AE fitting/scaler fitting;
- 5% validation for epoch selection/early stopping;
- 2.5% threshold calibration;
- 2.5% independent internal rate check.

Fractions are approximate for small categories; every role gets at least
one event per category (categories with fewer than four events are rejected).
Physics weights remain unchanged; no class balancing or top-10% training is
performed. Exact indices, category/weight/sample fingerprints, and run config
are archived. The training interface is opt-in via `train_ae.py --split_plan
... --output_dir ...`; a control output directory must not already exist.

Results live under `$BASE/outputs/${TAG}_AE_random_control/`:

- `baseline_checkpoint_ae.pth`: exact snapshot of the original AE; source and
  snapshot hashes are verified before declaring completion.
- `ae_random_category/checkpoint_ae.pth`: selected new AE, separate from all
  existing checkpoints, plus epoch history and training log.
- `random_split_plan.pth`, `control_manifest.json`: immutable split plan and
  explicit command/budget/provenance.
- `AE_control_summary.json`, `rate_comparison.csv`, `plots/`: frozen AE-only
  rate transfer, learning curve, and same-event held-out comparison.
- `frozen_AE_scores.npz`: old/new scores in original row order with event IDs.

After freezing the new AE, each AE defines its own physical-mixture 90th
percentile on a reference it did not use for gradient training. The old AE
uses its reconstructed original validation pool (also used historically for
epoch selection); the new AE uses its independently reserved calibration
pool. Both numerical thresholds are then reported on the **same outer
held-out events**, never used to select the new epoch or either threshold.
This is a frozen comparison on previously inspected MC, not a new blind test.
The control changes the role layout as well as randomizing assignment:
the original AE used 90/10%, whereas the new one separates early stopping,
calibration, and checking. It is not an isolated test of the seed alone.

Within-model train/unseen differences and acceptance transfer are meaningful;
absolute standardized MSE between AEs is not directly comparable because
their training-fitted scalers differ. Fixed-cut errors do not include finite
calibration uncertainty. No ABCD closure or signal-sensitivity claim is made,
and the new AE is never substituted into old NURD evaluations automatically.

## Matched original-AE full/tail NURD pilot

```bash
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
TAG="lambda1_matched_tail_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_tail_nurd_pilot.sh "$TAG"
```

This standalone pilot preserves the original physics-weighted AE and the
selected full-model lambda=1 checkpoint (Nov. 7 run, epoch 22). It does not
replace either with the random-split AE control. The completed control's
`frozen_AE_scores.npz` supplies **original-AE scores only**, verified against
the source AE hash, event IDs, and a small direct AE scoring parity check.
The default cache is
`$BASE/outputs/ae_phys_randomsplit_20261117_150546_AE_random_control`;
`AE_CACHE_DIR` can point to an equivalent completed control directory.

The fixed all-background physical-mixture 90th-percentile threshold is fit
on the **original AE's gradient-unseen validation rows**, not on its
reconstruction-training rows. AE membership is archived when available or
reconstructed with the original saved seed, fraction, parent labels, and
weights, as checked by the preceding split audit. Original AE validation
also selected the AE epoch: it is gradient-unseen, not wholly independent
of AE model selection. Some calibration rows were seen historically by the
source classifier; that overlap is reported, not called a blind sample.

Both new classifier continuations exclude every calibration row. The full
arm uses the remaining source-classifier training rows; the tail arm uses
their subset above the frozen AE threshold. Thus no formerly held-back
classifier rows are promoted into training. Both arms clone the exact same
encoder, projector, classifier **and critic** states. Fresh Adam states,
constant LR 3e-5, lambda=1 from the first update, SupCon=0.3, original nuisance
standardization, and weighted-within-class critic sampling are common.
Class-only physics-weighted balancing and 0.995 per-class weight clipping are
fit on each training population separately; no AE-loss-stratum balancing or
AE training is introduced.

The default is **4,000 encoder updates per arm**, batch 512, with the source
critic-update ratio retained. This is a short adaptation pilot, not another
full training or an equal-number-of-epochs comparison. Batch size differs
from the historical 4096 but is common between arms; SupCon is batch-dependent.
`PILOT_UPDATES` and `PILOT_BATCH_SIZE` override the common budget. There is no
new ramp, early stopping, or checking-based epoch selection. Training metrics
are logged every 500 updates; only update-2000 and final checkpoints are saved
with the defaults. One GPU runs both arms sequentially (4 CPU cores, 48 GB host
memory, 12-hour limit); no partition, GPU model, or node constraint is requested.

Only **after both final checkpoints exist**, the job opens the common outer
held-out sample and applies the same numerical AE threshold. This is the
pilot's checking sample, separate from calibration. Previously inspected MC
is an exploratory check, not a fresh blind result. The frozen source classifier
is also evaluated to distinguish common continuation drift from tail focusing.

For each model, QCD MD uses the same full eligible training-QCD reference
population, refit in that model's encoder latent. Calibration-tail QCD fixes
an AE median cut and an MD cut targeting 10% joint A-region efficiency within
the tail. No cut minimizes checking nonclosure. Classification is measured
on identical selected checking events with a common full-training-fitted
weight specification; ABCD always uses original physical weights. A paired
event bootstrap compares tail minus full nonclosure with cuts fixed; it does
not include calibration, retraining, or systematic uncertainties. Nominal
region ESS requirements remain A/B/C >= 50 and D >= 500; failing them is
reported, not hidden by loosening the requirements. Descriptive grids never
select a checkpoint or a threshold.

Outputs in `$BASE/outputs/${TAG}_tail_nurd_pilot/`:

- `pilot_manifest.json`, `pilot_roles.npz`, `class_statistics.csv`: source
  lineage, explicit role indices, frozen threshold, class counts and physical ESS.
- Source AE/classifier snapshots; original checkpoint paths are never overwritten.
- `full/` and `tail/`: `checkpoint_final.pth.tar`, sparse update checkpoints,
  `training_history.json`, and `training_weights.json` (optimization ESS/class mass).
- `pilot_comparison.json`: source/full/tail classification and matched-cut
  checking closure, region ESS, acceptance transfer, and paired uncertainty.
- Three `*_checking_QCD.npz` files: identical checking IDs, AE/MD scores,
  physical weights, and region membership for independent recounting.
- Eight figures in `plots/`: source/full/tail weighted score planes and grids,
  matched-point comparison, and per-class classification comparison. Grid
  colors use `|predicted/observed - 1|`; the legacy JSON `absolute_nonclosure`
  instead divides the yield difference by the predicted yield.

Pilot checkpoints are explicitly tagged. Ordinary `eval_abcd_nurd.py` refuses
them rather than silently reusing the source checkpoint's incompatible
calibration roles; use the standalone pilot comparison above.

## Frozen pilot cut-position study and KDE plots

```bash
PILOT="$BASE/outputs/lambda1_matched_tail_20261117_181652_tail_nurd_pilot"
TAG="tail_cutstudy_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_tail_grid_study.sh "$PILOT" "$TAG"
```

This is frozen inference and cut analysis, **not training**. By default it uses
the tail-focused final pilot checkpoint. `PILOT_BRANCH=full` or `source` can
explicitly study a fixed control instead; no model is automatically selected.
The original all-background AE threshold, selected calibration/checking roles,
physical event weights, and each model's previously fitted training-QCD MD
mean/whitening remain fixed. There is no MD refit, score regression, conditional
calibration transform, rotation of axes, or removal of hard QCD.

Calibration QCD alone defines a 91x91 grid of AE/MD percentiles from 0.05 to
0.95 within the existing AE tail. A candidate must pass the nominal A/B/C>=50,
D>=500 region ESS requirements and have QCD A efficiency between 2% and 25%
of the tail throughout a 3x3 neighborhood. The objective is the neighborhood's
median `|predicted/observed-1|` plus central ratio uncertainty. This is a
closure/stability search, **not a signal-sensitivity optimum**. The report also
flags whether all neighboring ratios are within 20% of one; a best available
candidate need not have acceptable closure.

`frozen_cut_plan.json` is written before checking inference or opening its
background manifest. If no stable candidate exists, the result stays `null`;
the statistics criteria are not silently weakened. Original pilot cuts then
appear only as an explicitly labeled reference in plots. Checking evaluates
the frozen candidate, reuses identical numerical grid thresholds for diagnostic
plots, and cannot choose a different cut. Its QCD scores/IDs/weights are checked
against the pilot's saved QCD cache and original point. A fixed-cut paired
bootstrap compares the candidate to that original point; calibration-selection,
retraining and systematic uncertainties are not included. All of this remains
exploratory on previously inspected MC, not a new blind confirmation.

One generic GPU is requested for the missing frozen calibration-QCD and
checking-all-class inference (4 CPU cores, 48 GB host memory, six-hour job
limit). No explicit partition, GPU model or node constraint is used. KDEs use
original physical weights once, with no weighted-resampling approximation.
Display limits cover the weighted 0.1--99.9% range for QCD or the union of those
ranges across classes. These are view limits only: all valid events still enter
KDE fitting, region counts and saved score arrays. KDE shapes are normalized
per class and do not show their relative expected rates. No signal efficiency
or sensitivity claim is made without a benchmark signal evaluation.

Results in `$BASE/outputs/${TAG}_tail_grid_study/`:

- `frozen_cut_plan.json`, `grid_study_summary.json`: calibration-only selection,
  checking result, original-point recount, uncertainty, feasibility and provenance.
- `calibration_QCD_scores.npz`, `checking_all_class_scores.npz`: reusable scores,
  event IDs, weights, labels and (for checking) encoder latent vectors.
- `calibration_grid.npz`, `checking_grid.npz`, `cut_scan.csv`: numerical cut
  locations and grid yields, ESS, efficiencies and ratios.
- Ten plots under `plots/`: calibration/checking closure scans, A-efficiency
  maps, closure-versus-efficiency curves, weighted QCD cut plane, class KDEs,
  per-class score planes, score marginals, PCA scatter and ABCD region yields.

White scan cells fail region-statistics requirements; they do not mean zero
nonclosure. Scan colors use `|predicted/observed-1|`, while the legacy
`absolute_nonclosure` JSON field divides the yield difference by the prediction.
Checking plots never rank cuts or models, even if another cell looks better.

## Frozen 100% / 20% / 10% retention comparison

```bash
PILOT="$BASE/outputs/lambda1_matched_tail_20261117_181652_tail_nurd_pilot"
TAG="pilot_retentions_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_pilot_retention_comparison.sh "$PILOT" "$TAG"
```

This standalone evaluation compares **both saved full/tail pilot checkpoints**
at three AE preselections. No network is trained, no AE or MD reference is
refit, and original checkpoints and prior study defaults remain untouched.
100% means no AE preselection, not pooling training/calibration/checking events.
20% and 10% are the physics-weighted all-background calibration-mixture upper
tails. The calibration pool is the complete original AE-gradient-unseen pool;
all its rows were excluded from both pilot continuations. The 10% threshold
must reproduce the pilot's frozen threshold. Original AE validation also selected
its epoch, and the source classifier historically saw some calibration rows;
this remains the same exploratory checkpoint lineage, not new blind MC.

The new study's default region requirements are **A/B/C ESS >= 50 and D ESS >=
100**, relaxed from the earlier D>=500. `D_MIN_ESS` may explicitly override D
before submission. Every point reports its actual ESS, yield uncertainty,
validity under the new requirements, and whether it also passes the old D>=500
rule. No thresholds in earlier evaluators are changed.

Comparisons use **overall QCD A-region acceptance relative to the original
unselected QCD mixture**, not acceptance within each tail. The primary target
is 1%, with 0.5% and 2% as additional diagnostics. For every calibration-derived
AE cut, the MD cut is fit on calibration high-AE QCD to match that overall A
yield. This yields an iso-acceptance curve: cut positions can vary without
changing the intended overall background budget. Weighted event discreteness
is allowed up to 5% relative mismatch; otherwise the point is ineligible.

At each target and model/retention combination, three consecutive AE-cut
positions must all meet the statistics and acceptance criteria. The candidate
minimizes their median `|predicted/observed-1|` plus central ratio uncertainty.
If no candidate qualifies, it remains null. This is not an unconstrained scan
allowed to win simply by accepting substantially more background.

**All six model/retention plans, including all three acceptance targets, are
saved before opening checking events.** The same numerical cuts are applied
unchanged on checking; its actual A acceptance is reported, not retuned to hit
1%. Each model's frozen original training-QCD mean/whitening is reused. The
old top-10% checking QCD scores/IDs/weights must reproduce the existing pilot
cache. All grids on checking are descriptive only, never used to pick a new
cut, model or retained fraction.

One generic GPU job performs full-range calibration-QCD and full-range
checking inference once per model, reusing the scores for all fractions.
No explicit partition, GPU model or node constraint is requested (4 CPU cores,
48 GB host memory, six-hour limit). Exact physics weights remain unchanged;
there is no AE-loss-stratum balancing, conditional calibration transform,
hard-QCD removal, or MC event replication to inflate effective statistics.
The 10%-adapted model is merely evaluated at 20%/100%, not newly trained there.

Results in `$BASE/outputs/${TAG}_retention_comparison/`:

- `frozen_comparison_plan.json`: calibration thresholds, all candidate choices,
  matching tolerance, role/provenance checks and statistics requirements.
- `retention_comparison.json`, `comparison.csv`: matched-rate closure,
  realized calibration/checking acceptance, region yields/ESS/uncertainties,
  all-class preselection statistics, and primary paired full-vs-tail-model
  bootstrap results. Fixed-cut bootstrap omits calibration-selection,
  retraining and systematic uncertainties.
- `full_calibration_QCD.npz`, `tail_calibration_QCD.npz`, and the two
  `*_checking_all_classes.npz` files: complete reusable scores, IDs, weights,
  categories/labels and checking latent vectors, with no duplicated fraction
  score dumps.
- Per-model/per-fraction grid NPZs and 14 figures under `plots/`: closure at
  matched overall acceptance, retention/rate transfer, effective statistics,
  calibration/checking closure maps, iso-acceptance cut curves, weighted QCD
  planes and region yields, plus six standard-size class KDEs in
  `plots/{full,tail}/{f100,f020,f010}/`.

KDEs apply original physical weights once and normalize each class shape;
they do not show relative process rates. Weighted display quantiles affect
view limits only. Blank scan cells fail statistics, not zero nonclosure.
No signal sample is evaluated: these results cannot establish a sensitivity
optimum or alone justify choosing an analysis retention threshold on real data.

## Double-NURD experiment (branch `double-nurd-experiment`)

This is a **matched fixed-update continuation**, not a new AE training or two
independent anomaly networks. It preserves the original physical AE and the
full lambda=1 checkpoint, and trains two arms sequentially on one generic GPU:

| Arm | Supervised training | Dependence objectives |
|---|---|---|
| `global` | Full eligible decay8 sample; CE + 0.3 SupCon | Global physical QCD, lambda=1 |
| `double` | Exactly the same sample, weights and starting model | Same global critic + independent tail-QCD critic, ramping to lambda=1 |

The tail is the **existing frozen top-10% physical all-background AE cut**,
defined on AE-gradient-unseen calibration rows. Both arms exclude every
calibration row from new optimization. The tail critic's real anchors AND
shuffled nuisance donors are restricted to tail training QCD. Global and tail
critics independently estimate real-versus-product density ratios between
the encoder latent and fixed standardized AE loss. The encoder minimizes

```
L = L_CE + 0.3 L_SupCon + L_info_global + lambda_tail(t) L_info_tail
```

The control omits the final term. There is no conditional-CDF transformation,
MD-score substitution, AE refitting, hard-QCD deletion, or held-out training.

Weighting is deliberately explicit:

- CE/SupCon: the same training-fitted class-only physical weights, with the
  existing per-class 0.995 clipping, fitted ONCE on common eligible rows.
- Critics and encoder information losses: the original **unclipped physical
  QCD measure** in their respective full/tail training populations, sampled
  with replacement proportional to physical weight and then given UNIT loss
  weights. Weights enter once, not in both sampling and the loss. Repeated
  training draws do not increase physical effective event statistics.
- Every ABCD yield, threshold quantile, MD reference and KDE: original physical
  weights, without classifier class factors or clipping. Cross sections are
  already included in the weight file; they are not applied a second time.

Both arms clone the source model and global critic; the tail critic is
independent and initially fresh. Adam states restart identically. Critics
receive 300 warm-up updates with encoder parameters AND BN buffers frozen.
The global coefficient remains 1; the tail coefficient reaches 1 over the
first 1,000 encoder updates. Default budget is **6,000 encoder updates per
arm**, hence **5,000 updates strictly after the tail ramp**. No CE-based early
stopping or best-checkpoint selection cuts this short. Save every 2,000
updates plus final; logs every 100 updates include timing and rough ETA.
Critics train on the same eval-mode representation used for gradient-enabled
information forwards; CE/SupCon use train mode. Both arms use batch 512,
constant LR 3e-5, three critic steps per encoder update and gradient norm cap 10.

On Della, from a checkout of this branch:

```bash
cd /home/mb7126/nurd_hlt
module load anaconda3/2026.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
TAG="double_nurd_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_double_nurd_experiment.sh "$TAG" | tee "/tmp/${TAG}.launch"
```

The launcher verifies the source/AE/cache/dataset paths, refuses existing
outputs, and archives the committed code so branch switches cannot change a
queued job. It submits **one training job for BOTH arms** and **one dependent
evaluation job**. Both request `gpu:1`, four CPUs and 48 GB host memory, with
no explicit partition, GPU model or node constraint. Time limits are 20 hours
for training and eight for evaluation, not runtime predictions.
Optional environment overrides: `PILOT_UPDATES`, `PILOT_BATCH_SIZE`,
`TAIL_RAMP_UPDATES`, `MIN_POST_RAMP_UPDATES`, `D_MIN_ESS`, `AE_CACHE_DIR`.
The launcher rejects a budget shorter than ramp + minimum post-ramp period.

Outputs:

- `$BASE/outputs/${TAG}_double_nurd/`: source checkpoint/AE copies, provenance,
  role indices, class physical statistics, per-arm training histories and
  `global/checkpoint_final.pth.tar`, `double/checkpoint_final.pth.tar`.
  `training_complete.json` appears only after BOTH final models exist.
- `$BASE/outputs/${TAG}_double_nurd_eval/`: `double_nurd_comparison.json`,
  `comparison.csv`, frozen calibration cut plan, per-event scores/IDs/weights,
  reusable grid NPZs, and `plots/` with full + top-10% weighted class KDEs,
  QCD planes, ABCD closure grids/curves, yields, region ESS, acceptance transfer
  and classification confusion matrices.
- `$BASE/logs/${TAG}_{doubletrain,doubleeval}-JOB.{out,err}`: job logs.

The evaluator fits one physics-weighted full-training-QCD MD reference per
model and reuses it for BOTH selections. It freezes all calibration-derived
cuts before opening checking events; checking never ranks cuts or epochs.
The primary A target is 1% of the ORIGINAL full QCD physical yield, with
0.5%/2% supporting targets and A/B/C ESS >=50, D ESS >=100. Candidate AE
positions cover 5--99% inside each selected calibration QCD population; absent
valid candidates remain absent. Historical ordinary MD scoring is preserved,
but the explicitly role-separated pilot cut protocol is not numerically the
same as the ordinary evaluator's threshold search. Use this standalone
evaluator, not the ordinary evaluator's legacy checkpoint split metadata.

Compare both closure AND per-class classification. A >5 percentage-point
balanced-accuracy loss against the matched control is flagged, not hidden by
checking-based model reselection. Adding the tail term also increases total
regularization: if successful, a global-only lambda=2 follow-up can distinguish
extra strength from tail specialization. Historical source-calibration
exposure and AE epoch selection are documented: this remains exploratory
previously inspected MC, not a blind test or a signal-sensitivity claim.

### Standard full/top-10% plots and decay8/parent4 views after double-NURD

Use the completed double/single comparison, not another training campaign:

```bash
cd /home/mb7126/nurd_hlt
git fetch origin
git switch double-nurd-experiment
git pull --ff-only origin double-nurd-experiment
module load anaconda3/2026.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
RUN=double_nurd_20261118_185256
TAG="double_nurd_standard_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_double_nurd_standard_eval.sh "$RUN" "$TAG" | tee "/tmp/${TAG}.launch"
```

The first argument is the original experiment tag, WITHOUT `_double_nurd` or
`_double_nurd_eval`. The second is a new unique output tag. The launcher checks
both original final models and completed score caches and archives committed
code. No original output is overwritten. It submits:

1. One generic-GPU checking inference job: only **one held-out encoder pass per
   model** to recover latent vectors and eight probabilities. No AE inference,
   training-manifest inference, MD refit or network updates. Original cached
   MD must agree with re-inference; archived AE/MD scores remain the axes.
2. A dependent CPU plotting job using those checksummed predictions, releasing
   the GPU before KDE/PCA rendering. The GPU request specifies no model or
   partition; the plotting job uses the CPU partition. Four CPUs and 32 GB
   host memory; four-hour inference and eight-hour plotting limits (not ETAs).

This uses the **ordinary evaluation threshold policy**, rather than the prior
pilot's matched-overall-A-efficiency policy: scan the 48 physics-weighted QCD
percentiles from 0.50 to 0.98 on the appropriate full/top-10% CALIBRATION
population, pick minimum absolute nonclosure, freeze the numerical AE/MD cuts
for all models/selections, then apply them on checking. Requirements remain
A/B/C ESS >=50, D >=100 (`D_MIN_ESS=500` restores the older D requirement).
Absent valid calibration candidates remain absent, never silently relaxed.

As in the ordinary evaluator, checking-quantile closure maps and diagonal
curves are descriptive, not a second working-point selection. The frozen
working point is marked at its ACTUAL checking percentiles, not at a hard-coded
median. Metadata explicitly distinguishes these checking scan cuts from the
calibration-selected working point. Raw numbers are never capped, although
the standard graphics cap color at 100% and ratio axes at 1.5.

**Parent4 is a reporting view of the SAME decay8 models**, summing probabilities
for Z `[0,6,7]`, QCD `[1]`, ttbar `[2]`, W `[3,4,5]` before parent argmax. It
is NOT relabeling the eight-class argmax and is NOT a new parent4 model. Other
parents' summed probabilities can beat QCD even when QCD was the largest single
decay probability, so QCD parent recall need not equal decay8 QCD recall.
Raw and physical-weighted confusion matrices/recalls, balanced accuracy,
physical accuracy and CE are all reported with their distinct definitions.

The same true QCD rows, physical weights, AE cut, MD transform and ABCD cuts
are retained for both taxonomy views. Therefore **QCD closure is identical
for decay8 and parent4**; only class grouping, probability decisions, KDEs,
class histograms/profiles and classification metrics change.

Results under `$BASE/outputs/${TAG}_standard_eval/`:

```text
standard_threshold_plan.json           # frozen before checking
standard_evaluation_summary.json        # all models, selections, taxonomies
classification_and_closure.csv
{global,double}/{f100,f010}/
  standard_closure_grid.npz
  {decay8,parent4}/
    evaluation_summary.json
    plots/
      closure_scan_2d.png
      cut_and_count_bkg_check.png
      hist2d_bkg.png
      hist2d_<class>.png
      hist2d_pca_md_kde.png
      hist2d_pca_md_scatter.png
      hist2d_by_class_combined.png
      pca_scatter_embeddings.png
      pca_corner.png
      profile_*.png
      confusion_matrix.png
      confusion_matrix_physics.png
```

Reusable predictions and `inference_complete.json` are in
`$BASE/outputs/${TAG}_standard_inference/`. Plotting uses original physical
weights once; class KDEs show separately normalized shapes, NOT relative
process rates. Scatter resampling is display-only. The original AE top-10%
threshold is inherited unchanged (about 9.50% of checking physical yield,
not 10% of raw MC rows). No signal benchmark is supplied; this is a background
closure/classification diagnostic, not an optimized signal analysis. The
standard minimum-nonclosure cut need not have the same QCD acceptance between
models, so use the original matched-rate comparison to judge that question.

### Historical Mequinna golden recipe versus additive double-NURD

`slurm/launch_legacy_double_nurd.sh UNIQUE_TAG` is a separate experiment on the
original Mequinna parent4 files. It does NOT run the train3M decay8 launcher on
old data, change the nominal physical AE, or overwrite old checkpoints.

The verified golden artifact is
`hlt_nurd_engineer_engineer_continuous_supcon030/checkpoint_main.pth.tar`,
selected epoch 24, training commit `b5b2272`. Its original AE has weighting
method `generator_weighted_uniform_class`. Its archived validation balanced
accuracy is 46.6129%, weighted accuracy 60.9339%, CE 0.932471. The old validation
function did not compute SupCon: its zero `contrastive_loss` is NOT evidence
that training omitted the configured 0.3 term.

The launcher archives the original `b5b2272` trainer/dataset/models/weighting
code separately from the current driver. SHA-256 checks pin the six operative
legacy source files; training runs in isolated Python processes. The original
recipe is four native parent labels, SupCon=0.3 (temperature .05), main batch
4096, global lambda=1 from epoch 1, three global critic updates per model
update, Adam LR 1e-4 / critic LR 1e-3, seed 42, 20 **unclipped weighted-quantile**
nuisance strata, original split/scaler/nuisance transform and archived effective
weight factors. It is NOT modern class-only/log-fixed/clipped weighting. The
old base weights apply generator weights to QCD and unit weights to non-QCD;
this is not the new sample's all-process cross-section mixture.

Five jobs implement the experiment:

1. Generic-GPU preparation reuses the original AE, verifies input signatures,
   generator checksum and original AE/classifier split, and reproduces the
   golden checkpoint's validation CE/accuracy to absolute tolerance 1e-3.
   Failure prevents dependent training from starting. It snapshots the golden
   model/AE and saves common fresh encoder/global-critic initialization + RNG.
2. Fresh **single** training uses the pinned original `train_epoch` unchanged.
3. Fresh **double** training uses the same full-population CE/SupCon and
   ORIGINAL all-parent global critic, adding only the separate tail-QCD critic.
   Both initialization states and the global data/dropout RNG are matched;
   auxiliary sampling/critic RNG never advances the original stream. The tail
   critic targets inference-mode latent versus AE loss inside training QCD
   above the fixed AE-unseen all-background legacy-weighted 90th percentile.
   Conditional physics sampling enters once; its loss weights are unity.
4. Generic-GPU scoring evaluates the **frozen golden reference and both new
   selected models**: full-training-QCD MD references, separate calibration
   cut selection, then independent held-out-file reporting. One MD transform
   per model is reused for full and tail. AE/MD definitions are unchanged.
5. A dependent CPU job renders all SIX standard plot suites, plus classification
   matrices, paired event bootstraps and a numerical golden-held-out reproduction
   check at the original archived numeric cuts. A mismatch is flagged, not
   silently corrected by retuning on checking.

Only steps 2 and 3 are model training. The AE and golden encoder are NEVER
retrained. Both new arms run a fixed 40 epochs, no CE early stopping. The new
tail critic trains for five epochs at zero encoder pressure, ramps linearly
to lambda=1 over epochs 6--15, then has 25 COMPLETE post-ramp epochs 16--40.
Both arms select best validation CE only from epoch 16 onward. A separate
`checkpoint_historical_best.pth.tar` records the old all-epoch CE rule up to
its simulated original patience stop; the actual trainings still finish the
common budget. `checkpoint_final.pth.tar` and five-epoch snapshots are retained.

The old 40-epoch cosine LR schedule is retained. `LEGACY_EPOCHS=60` can extend
both arms, holding LR at the old minimum after epoch 40; defaults remain 40.
Original main batch 4096 is never silently reduced, since SupCon depends on
batch size. RNG-preserving Transformer activation checkpointing saves GPU
memory without changing state keys, batch/loss or BN updates, at extra compute
cost. No specific GPU model or GPU partition is requested; preparation and
scoring limits are four/eight hours and each separate training has a 20-hour
limit. These limits are not runtime estimates. Cross-version/hardware bitwise
reproduction of a historical training trajectory is not promised.
GPU stages use four CPU cores and 64 GB host memory for the large legacy
tensors; the cache-only CPU plotting job uses two cores and 16 GB. At most
the two fresh model trainings run concurrently.

```bash
cd /home/mb7126/nurd_hlt
git fetch origin
git switch double-nurd-experiment
git pull --ff-only origin double-nurd-experiment
module load anaconda3/2026.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
TAG="mequinna_double_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_legacy_double_nurd.sh "$TAG" | tee "/tmp/${TAG}.launch"
```

The launcher uses `MEQUINNA_DIR` explicitly, ignoring any train3M `TRAIN_PT`
environment left from other campaigns. It also snapshots the old statistically
valid evaluation summary (override `GOLDEN_SUMMARY` only to an equivalent saved
report for the same golden model). Original output tags cannot be reused.

Outputs:

- `$BASE/outputs/${TAG}_legacy_double_nurd/`: preparation/reproduction report,
  frozen source copies, original recipe arrays, matched initialization, and
  `single/`, `double/` histories/checkpoints/completion reports.
- `$BASE/outputs/${TAG}_legacy_scores/`: frozen calibration plans, reference
  transforms, AE scores, latents, predictions and a checksummed cache manifest.
- `$BASE/outputs/${TAG}_legacy_eval/`: `legacy_comparison.json`,
  `legacy_comparison.csv`, and `{golden,single,double}/{f100,f010}/plots/` with
  the ordinary `closure_scan_2d.png`, `cut_and_count_bkg_check.png`, weighted
  class KDEs/histograms/scatters, PCA/corner, profiles and both raw/weighted
  confusion matrices. Each suite has its own `evaluation_summary.json`.

Full-sample evaluation restores historical A/B/C ESS >=50, D >=500. Tail
evaluation explicitly uses A/B/C >=50, D >=100, as in the newer tail studies;
invalid cells/candidates remain invalid, with no automatic lowering. The fixed
tail means 10% of the CALIBRATION legacy weighted mixture, not exactly 10% of
raw held-out rows. Historical files lack composite event IDs: signatures and
saved disjoint indices cannot certify absence of cross-file duplicate events.
This is an exploratory legacy-domain debugging benchmark, not a reason to
replace production physics weighting or claim signal sensitivity from closure.

### Evaluate interrupted Mequinna trainings without resuming

If a legacy training hits its wall-time limit, keep the original checkpoints
and histories. The normal evaluator still requires completed training. The
separate evaluation-only launcher explicitly permits the existing
`checkpoint_main.pth.tar` best post-ramp validation-CE checkpoints:

```bash
cd /home/mb7126/nurd_hlt
git pull --ff-only origin double-nurd-experiment
module load anaconda3/2026.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate disco
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
RUN=mequinna_double_20261119_122050
EVAL_TAG="${RUN}_best_so_far_$(date +%Y%m%d_%H%M%S)"
bash slurm/launch_legacy_partial_eval.sh "$RUN" "$EVAL_TAG"
```

This submits only one generic-GPU scoring job and one dependent CPU plotting
job, with no dependency on the timed-out training jobs and no GPU-model or GPU
partition constraint. It never trains, resumes, alters training files, or
manufactures completion markers. The saved main checkpoint must agree with
the best eligible validation CE in its recorded history. A corrupt completion
marker is still an error, not something partial mode bypasses. The original
preparation's train/weight paths and archived golden evaluation are reused.

Results: `$BASE/outputs/${EVAL_TAG}_legacy_partial_scores/` (including frozen
selected checkpoint copies and provenance) and
`$BASE/outputs/${EVAL_TAG}_legacy_partial_eval/`. The latter contains
`legacy_comparison.csv`, `legacy_comparison.json`, and all SIX ordinary
parent4 suites at `{golden,single,double}/{f100,f010}/plots/`: weighted KDEs,
ABCD grids/curves, PCA/corner, profiles and confusion matrices. The evaluator
keeps the original AE and legacy weights, fits MD on training QCD, and freezes
calibration cuts before checking; this is the standard threshold policy, NOT
the newer pilot's matched-1%-acceptance policy.

Reports label partial training and distinguish selected checkpoint epoch from
last completed epoch (e.g. single 35 versus double 33 completed of planned 40).
This is preliminary: the training budgets are not equal, neither arm has the
full planned post-ramp budget, and held-out MC has been inspected previously.
No held-out result is used to reselect the evaluated checkpoint. Original
stuck scoring/plot jobs can be cancelled separately; cancellation is never
automatic in this launcher.

## Repository map

- `train_ae.py`, `train_hlt.py`: main training entry points; standalone fixed-budget
  pilots reuse their models/losses and are documented above.
- `eval_abcd_nurd.py`: canonical ABCD evaluation; standalone audits reuse region
  counting utilities with explicit role/cut protocols.
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
