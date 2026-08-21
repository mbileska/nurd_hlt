"""
Pre-train the HLT Autoencoder on object-level features.

The AE defines axis 1 of the ABCD method (reco loss = nuisance z for NURD).
Saves a checkpoint that train_hlt.py loads with --ae_ckpt.

Usage
-----
python train_ae.py \
    --data /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_train.pt \
    [--epochs 100] [--batch_size 2048] [--lr 1e-3] \
    [--latent_dim 16] [--enc_nodes 512 256] [--dec_nodes 256 512]
"""
import argparse
import os
import math
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.hlt_autoencoder import HLTAutoencoder
from utils.hlt_weights import (
    apply_class_balance_factors,
    effective_mass_by_class,
    fit_class_balance_factors,
    load_generator_weights,
    sample_signature,
    stratified_split_indices,
    weighted_mean_and_std,
)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Pre-train HLT AE")
parser.add_argument("--data",       required=True,  type=str)
parser.add_argument("--gen_weight_path", "--gen_weights", dest="gen_weight_path",
                    default=None, type=str)
parser.add_argument("--qcd_label", default=1, type=int)
parser.add_argument("--epochs",     default=100,    type=int)
parser.add_argument("-b","--batch_size", default=2048, type=int)
parser.add_argument("--lr",         default=1e-3,   type=float)
parser.add_argument("--weight_decay",default=0.0,   type=float)
parser.add_argument("--cosine",     default=1,      type=int)
parser.add_argument("--val_split",  default=0.1,    type=float)
parser.add_argument("--latent_dim", default=16,     type=int)
parser.add_argument("--enc_nodes",  default=[512, 256], nargs="+", type=int)
parser.add_argument("--dec_nodes",  default=[256, 512], nargs="+", type=int)
parser.add_argument("--patience",   default=20,     type=int)
parser.add_argument("--max_events", default=-1,     type=int)
parser.add_argument("--exp_name",   default="ae",   type=str)
parser.add_argument("--project_name", default="hlt", type=str)
parser.add_argument("--local_testing", default=0,   type=int)
parser.add_argument("--manualSeed", default=42,     type=int)
args = parser.parse_args()

if not args.local_testing:
    import wandb
    wandb.init(id=args.exp_name, resume="allow",
               project="nurd-ood-" + args.project_name, reinit=True)
    wandb.config.update(args, allow_val_change=True)

# ── Setup ─────────────────────────────────────────────────────────────────────

directory = f"checkpoints/hlt/{args.project_name}/{args.exp_name}/"
os.makedirs(directory, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s : %(message)s")
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(directory, "ae_train.log"), mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s : %(message)s"))
log.addHandler(fh)

# ── Load & normalise obj features ─────────────────────────────────────────────

log.debug(f"Loading {args.data}")
raw = torch.load(args.data, map_location="cpu", weights_only=False)
data_signature = sample_signature(raw, max_events=args.max_events)
obj = raw["obj"]
labels = raw["label"].long().reshape(-1)
if args.max_events > 0:
    obj = obj[:args.max_events]
    labels = labels[:args.max_events]
physics_weights, generator_metadata = load_generator_weights(
    args.gen_weight_path,
    labels,
    qcd_label=args.qcd_label,
    max_events=args.max_events,
    sample=raw,
)
del raw

# flatten first 4 features per candidate: [N, n_cands, 4] → [N, n_cands*4]
obj_flat = torch.nan_to_num(
    obj[:, :, :4].reshape(obj.shape[0], -1).float(),
    nan=0.0, posinf=0.0, neginf=0.0)
del obj

# ── Train / val split ─────────────────────────────────────────────────────────

idx_tr, idx_val = stratified_split_indices(
    labels, val_fraction=args.val_split, seed=args.manualSeed,
    weights=physics_weights)
class_factors = fit_class_balance_factors(
    labels[idx_tr], physics_weights[idx_tr])
weights_tr = apply_class_balance_factors(
    labels[idx_tr], physics_weights[idx_tr], class_factors)
weights_val = apply_class_balance_factors(
    labels[idx_val], physics_weights[idx_val], class_factors)

mu, std = weighted_mean_and_std(obj_flat[idx_tr], weights_tr)
std = torch.where(std < 1e-8, torch.ones_like(std), std)
obj_norm = (obj_flat - mu.view(1, -1)) / std.view(1, -1)
del obj_flat
n_features = obj_norm.shape[1]
log.debug(f"AE input: {obj_norm.shape}  ({n_features} features)")
log.debug(
    "AE class-balanced train mass=%s validation mass=%s generator=%s",
    effective_mass_by_class(labels[idx_tr], weights_tr),
    effective_mass_by_class(labels[idx_val], weights_val),
    generator_metadata,
)

obj_tr = obj_norm[idx_tr]
obj_val = obj_norm[idx_val]
labels_tr = labels[idx_tr]
labels_val = labels[idx_val]

dl_tr = DataLoader(
    TensorDataset(obj_tr, weights_tr, labels_tr),
    batch_size=args.batch_size, shuffle=True, drop_last=False)
dl_val = DataLoader(
    TensorDataset(obj_val, weights_val, labels_val),
    batch_size=args.batch_size, shuffle=False, drop_last=False)

# ── Build AE ──────────────────────────────────────────────────────────────────

dec_nodes = args.dec_nodes + [n_features]   # output layer matches input dim
ae_config = {
    "features":       n_features,
    "latent_dim":     args.latent_dim,
    "encoder_config": {"nodes": args.enc_nodes},
    "decoder_config": {"nodes": dec_nodes},
    "alpha":          1.0,
}
ae = HLTAutoencoder(ae_config).to(device)
log.debug(f"AE: latent={args.latent_dim}  enc={args.enc_nodes}  dec={dec_nodes}")

optim  = torch.optim.Adam(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
n_steps = args.epochs * len(dl_tr)

def get_lr(step):
    if not args.cosine:
        return 1.0
    progress = step / n_steps
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optim, get_lr)

# ── Training loop ─────────────────────────────────────────────────────────────

best_val  = float("inf")
bad_epochs = 0
ckpt_path = os.path.join(directory, "checkpoint_ae.pth")

for epoch in range(args.epochs):
    ae.train()
    tr_numerator = 0.0
    tr_denominator = 0.0
    for batch, weights, _labels in dl_tr:
        batch = batch.to(device)
        weights = weights.to(device)
        optim.zero_grad()
        recon, _ = ae(batch)
        per_event = (recon - batch).square().mean(dim=1)
        # Full-split weights are normalized to mean one. A fixed denominator
        # avoids the stochastic bias caused by random per-batch weight sums.
        loss = (per_event * weights).mean()
        loss.backward()
        optim.step()
        scheduler.step()
        tr_numerator += float((per_event.detach() * weights).sum())
        tr_denominator += float(weights.sum())
    tr_loss = tr_numerator / max(tr_denominator, 1e-12)

    ae.eval()
    val_numerator = 0.0
    val_denominator = 0.0
    with torch.no_grad():
        for batch, weights, _labels in dl_val:
            batch = batch.to(device)
            weights = weights.to(device)
            recon, _ = ae(batch)
            per_event = (recon - batch).square().mean(dim=1)
            val_numerator += float((per_event * weights).sum())
            val_denominator += float(weights.sum())
    val_loss = val_numerator / max(val_denominator, 1e-12)

    log.debug(f"Epoch {epoch+1}/{args.epochs}  train={tr_loss:.6f}  val={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
    if not args.local_testing:
        wandb.log({"AE train loss": tr_loss, "AE val loss": val_loss}, step=epoch)

    if val_loss < best_val:
        best_val = val_loss
        bad_epochs = 0
        torch.save({
            "ae":        ae.state_dict(),
            "ae_config": ae_config,
            "ae_scaler": {
                "mu":  mu.cpu(),
                "std": std.cpu(),
            },
            "weighting": {
                "method": "generator_weighted_uniform_class",
                "class_factors": class_factors,
                "generator": generator_metadata,
                "split_seed": args.manualSeed,
                "validation_fraction": args.val_split,
            },
            "data_signature": data_signature,
            "epoch": epoch + 1,
        }, ckpt_path)
        log.debug(f"  → saved best checkpoint ({ckpt_path})")
        if not args.local_testing:
            wandb.run.summary["best_ae_val_loss"] = best_val
    else:
        bad_epochs += 1
        if bad_epochs >= args.patience:
            log.debug("Early stopping.")
            break

log.debug(f"Done. Best val loss: {best_val:.6f}  checkpoint: {ckpt_path}")
if not args.local_testing:
    wandb.finish()
