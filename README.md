# NURD HLT Anomaly Detection

An anomaly detection analysis using a two-axis ABCD background estimation method. Axis 1 is the AE reconstruction loss (how anomalous an event looks) and axis 2 is a contrastive model score (what type of event it is). NURD is used to decorrelate the two axes so the ABCD method is valid.

---

## How it works

### Axis 1 — Autoencoder (density estimation)

An MLP autoencoder trained on object-level features (pT, η, φ). Its reconstruction loss is the nuisance variable — events that reconstruct poorly are flagged as anomalous.

### Axis 2 — Contrastive encoder (clustering)

A Linformer-based Transformer encodes PF candidates into a low-dimensional
latent vector. The baseline uses class-balanced cross-entropy to learn all four
background classes; SupCon is optional and disabled for the first faithful
engineer-style run.

### NURD decorrelation

The nuisance passed to the critic is the **continuous AE reconstruction error**.
The training-only nuisance histogram is used only to estimate event weights; no
nuisance-bin index is passed to the network.

- **Unified weighting** — physics generator weights are retained within each
  class/nuisance stratum, while every occupied stratum and every background
  class receive equal total training mass. The same effective event weight is
  used for AE training, classification, critic training, and the encoder
  information loss. Full-split normalization avoids biased random-batch
  normalization for the broad Mequinna weights.
- **Density-ratio critic** — distinguishes real `(latent, nuisance, label)`
  tuples from tuples with a shuffled continuous nuisance. The encoder minimizes
  `log P(real) - log P(shuffled)` on real tuples, following the engineer
  reference implementation.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

Training is two steps — the AE must be trained first since its checkpoint is required by the main training script.

### Step 1 — Train the AE

```bash
python train_ae.py \
    --data /path/to/hlt_smcocktail_mequinna_train.pt \
    --gen_weight_path /path/to/weight_train.pt
```

The checkpoint is saved under
`checkpoints/hlt/<project_name>/<exp_name>/checkpoint_ae.pth`.

### Step 2 — Train the NURD contrastive model

```bash
python train_hlt.py \
    --data    /path/to/hlt_smcocktail_mequinna_train.pt \
    --gen_weight_path /path/to/weight_train.pt \
    --ae_ckpt checkpoints/hlt/hlt/ae_run/checkpoint_ae.pth \
    --balance_strata 20 \
    --critic_steps 3 \
    --lambda_info 1.0 \
    --contrast_weight 0.0
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--balance_strata` | 20 | Training-only strata for estimating unified weights; not a critic input |
| `--critic_steps` | 1 | Independent critic batches per encoder batch |
| `--lambda_info` | 1.0 | Weight on the engineer log-density-ratio penalty |
| `--info_warmup_epochs` | 0 | Epochs with no encoder information penalty; the critic still trains |
| `--info_ramp_epochs` | 0 | Cosine-ramp epochs from zero to `lambda_info`; zero restores immediate application |
| `--contrast_weight` | 0.0 | Optional SupCon weight; disabled in the faithful engineer baseline |
| `--lr_schedule_epochs` | 0 | Cosine LR horizon; zero follows `epochs`, otherwise LR stays at its minimum after this horizon |

Checkpoints are saved to `checkpoints/hlt/<project_name>/<exp_name>/`.

### Della campaign

From the repository root on `main`:

```bash
export BASE=/scratch/gpfs/IOJALVO/mb7126/nurd_hlt
bash slurm/launch_engineer_campaign.sh <run_tag> <supcon_weight>
```

The scheduled SupCon-0.3 comparison (five warm-up epochs, cosine ramp over
epochs 6--15, full information weight thereafter) is launched with:

```bash
INFO_WARMUP_EPOCHS=5 INFO_RAMP_EPOCHS=10 \
NURD_EPOCHS=80 LR_SCHEDULE_EPOCHS=40 \
  bash slurm/launch_engineer_campaign.sh engineer_continuous_supcon030_ramp 0.3
```

This keeps the original 40-epoch cosine learning-rate trajectory, holds its
minimum afterward, and leaves a high 80-epoch ceiling so patience-based early
stopping determines the endpoint. Omitting those environment variables exactly
restores the previous immediate information penalty and 40-epoch limit.
Checkpoint selection and early stopping are disabled until the ramp reaches its
target; the critic itself continues to train throughout warm-up.

The launcher submits fresh AE and NURD training followed by one dependent dual
evaluation. Results are written to
`$BASE/outputs/<run_tag>_eval/{held-out,legacy}`. An optional explicit run tag
may be passed as the first argument. The held-out protocol fits the MD reference
on saved training indices, selects thresholds on saved validation indices, and
only then reports closure on the independent Mequinna test file. The legacy
folder intentionally preserves main's same-sample comparison. A compact result
is printed and saved as `<run_tag>_eval/evaluation_summary.json`.

If training completed but its dependent evaluation failed, reuse the existing
checkpoints without resubmitting training:

```bash
bash slurm/launch_engineer_eval.sh <existing_run_tag>
```

---

## Evaluation

```bash
python eval_abcd_nurd.py \
    --ckpt       checkpoints/hlt/hlt/hlt_nurd_run/checkpoint_main.pth.tar \
    --ae_ckpt    checkpoints/hlt/hlt/ae_run/checkpoint_ae.pth \
    --test_pt    /path/to/hlt_smcocktail_mequinna_test.pt \
    --gen_weight_path /path/to/weight_test.pt \
    --reference_pt /path/to/hlt_smcocktail_mequinna_train.pt \
    --reference_weight_path /path/to/weight_train.pt \
    --outdir     outputs_abcd/<run> \
    --n_pca      6
```

This saves `<outdir>/abcd_thresholds.json` and `<outdir>/diagnostics.json`.
Omitting both reference arguments intentionally invokes the historical
same-sample threshold scan and should only be used for the legacy comparison.

---

## Datacard for CMS Combine

```bash
python make_datacard_ttbar.py \
    --ckpt       checkpoints/hlt/hlt/hlt_nurd_run/checkpoint_main.pth.tar \
    --ae_ckpt    checkpoints/hlt/ae/checkpoint_ae.pth \
    --test_pt    /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    --outdir     outputs_datacard/<run> \
    --n_pca      6
```

To skip the threshold scan and reuse thresholds from a previous eval run:

```bash
python make_datacard_ttbar.py \
    --ckpt      checkpoints/hlt/hlt/hlt_nurd_run/checkpoint_main.pth.tar \
    --ae_ckpt   checkpoints/hlt/ae/checkpoint_ae.pth \
    --test_pt   /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    --eval_json outputs_abcd/<run>/abcd_thresholds.json \
    --outdir    outputs_datacard/<run>
```

To run the datacard with Combine:

```bash
combine -M Significance datacard_ttbar.txt -t -1 --expectSignal 1
combine -M AsymptoticLimits datacard_ttbar.txt -t -1
```
