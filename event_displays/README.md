# PF-level QCD and TpTp event displays

This folder generates three reproducibly random, kinematically matched displays
for each of:

- usual QCD: held-out QCD in ABCD region D (below both score thresholds);
- false-positive QCD: held-out QCD in region A (above both thresholds); and
- TpTp signal.

Matching uses both total PF candidate pT and active PF multiplicity. TpTp
anchors are sampled randomly from the best-matched 20% of signal events, then
their nearest unused event is selected from each QCD pool. This prevents the
comparison from being driven only by event scale.

The displays use PF candidate pT, eta, phi, displacement, and particle ID, plus
the AE reconstruction residual and model scores. They are deliberately labelled
as PF pT flow: the current tensors do not contain Level-1 calorimeter towers.

## Launch on Della

From the repository root on `main`:

```bash
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
sbatch event_displays/submit_event_displays.sbatch
```

The default campaign is `engineer_continuous_v1`. Results are written under:

```text
event_displays/generated/engineer_continuous_v1_<job-id>/
├── usual_qcd/
├── false_positive_qcd/
├── tptp/
├── matched_triplets/
└── selection_manifest.json
```

The manifest records original tensor row indices, scores, generator weights,
matching distances, input paths, and held-out thresholds.

Useful overrides can be supplied at submission time:

```bash
RUN_TAG=engineer_continuous_v1 SEED=7 N_DISPLAYS=3 \
  sbatch --export=ALL event_displays/submit_event_displays.sbatch
```

`THRESHOLDS_JSON`, checkpoint paths, data paths, and `OUTDIR` can also be
overridden. The script refuses to overwrite an existing output directory.
