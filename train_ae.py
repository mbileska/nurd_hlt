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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.hlt_autoencoder import HLTAutoencoder

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Pre-train HLT AE")
parser.add_argument("--data",       required=True,  type=str)
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
    wandb_timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))
    wandb.init(id=args.exp_name, resume="allow",
               project="nurd-ood-" + args.project_name, reinit=True,
               settings=wandb.Settings(init_timeout=wandb_timeout))
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
raw = torch.load(args.data, map_location="cpu")
obj = raw["obj"]
if args.max_events > 0:
    obj = obj[:args.max_events]
del raw

# flatten first 4 features per candidate: [N, n_cands, 4] → [N, n_cands*4]
obj_flat = obj[:, :, :4].reshape(obj.shape[0], -1).float().numpy()
del obj

mu  = obj_flat.mean(axis=0).astype(np.float32)
std = obj_flat.std(axis=0).astype(np.float32)
std = np.where(std < 1e-8, 1.0, std)
obj_norm = torch.from_numpy((obj_flat - mu) / std)
del obj_flat
n_features = obj_norm.shape[1]
log.debug(f"AE input: {obj_norm.shape}  ({n_features} features)")

# ── Train / val split ─────────────────────────────────────────────────────────

idx = np.arange(len(obj_norm))
idx_tr, idx_val = train_test_split(idx, test_size=args.val_split, random_state=args.manualSeed)
obj_tr  = obj_norm[torch.tensor(idx_tr)]
obj_val = obj_norm[torch.tensor(idx_val)]

dl_tr  = DataLoader(TensorDataset(obj_tr),  batch_size=args.batch_size, shuffle=True,  drop_last=False)
dl_val = DataLoader(TensorDataset(obj_val), batch_size=args.batch_size, shuffle=False, drop_last=False)

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

mse    = nn.MSELoss()
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
    tr_loss = 0.0
    for (batch,) in dl_tr:
        batch = batch.to(device)
        optim.zero_grad()
        recon, _ = ae(batch)
        loss = mse(recon, batch)
        loss.backward()
        optim.step()
        scheduler.step()
        tr_loss += loss.item() * batch.size(0)
    tr_loss /= len(dl_tr.dataset)

    ae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (batch,) in dl_val:
            batch = batch.to(device)
            recon, _ = ae(batch)
            val_loss += mse(recon, batch).item() * batch.size(0)
    val_loss /= len(dl_val.dataset)

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
                "mu":  torch.from_numpy(mu),
                "std": torch.from_numpy(std),
            },
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
