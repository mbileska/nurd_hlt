"""
HLT NURD contrastive training.

Built on top of gabhijith's train_exact.py; the NURD reweighting and
joint-independence critic logic is kept verbatim.  What's added:

  * HLTSmCocktailDataset  — PF candidate data; AE reco loss as nuisance z
  * HLTContrastiveModel   — Roy's Linformer encoder + projector + classifier
  * HLTCritic             — predicts nuisance bin from (latent, y)
  * Contrastive loss      — SupCon / InfoNCE on top of NURD-weighted CE

Usage
-----
python train_hlt.py \\
    --data   /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_train.pt \\
    --ae_ckpt <path/to/ae_checkpoint.pth> \\
    [--reweight 1] [--joint_indep 1] [--critic_epochs 2] \\
    [--epochs 100] [--batch_size 2048] [--lr 1e-4]
"""
import argparse
import os
import time
import random
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.hlt_autoencoder import HLTAutoencoder
from models.hlt_con import HLTContrastiveModel, HLTCritic
from dataset.hlt_smcocktail_dataset import build_hlt_datasets
from utils.common import AverageMeter, save_checkpoint, accuracy

# ── Contrastive losses ────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.05, base_temperature=0.05):
        super().__init__()
        self.T  = temperature
        self.Tb = base_temperature

    def forward(self, features, labels):
        features = features.float()
        if features.dim() < 3:
            features = features.unsqueeze(1)
        B = features.shape[0]
        device = features.device

        labels = labels.contiguous().view(-1, 1)
        mask   = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_dot       = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.T)
        logits_max, _    = anchor_dot.max(dim=1, keepdim=True)
        logits           = anchor_dot - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(B).view(-1,1).to(device), 0)
        mask = mask * logits_mask

        exp_logits  = torch.exp(logits) * logits_mask
        log_prob    = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-8))
        n_pos       = mask.sum(1).clamp(min=1e-6)
        mean_lp_pos = (mask * log_prob).sum(1) / n_pos
        loss        = -(self.T / self.Tb) * mean_lp_pos
        return loss.view(1, B).mean()


# ── ABCD closure loss helpers (ported from double DisCo) ─────────────────────

def _sigmoid_counts(var1, var2, cut1, cut2, weights, scale=50.0):
    s1_high = torch.sigmoid(scale * (var1 - cut1))
    s1_low  = torch.sigmoid(scale * (cut1 - var1))
    s2_high = torch.sigmoid(scale * (var2 - cut2))
    s2_low  = torch.sigmoid(scale * (cut2 - var2))
    NA = torch.sum(s1_high * s2_high * weights)
    NB = torch.sum(s1_high * s2_low  * weights)
    NC = torch.sum(s1_low  * s2_high * weights)
    ND = torch.sum(s1_low  * s2_low  * weights)
    return NA, NB, NC, ND


def closure_loss_batch(var1, var2, weights, n_cuts=15, n_events_min=10, max_tries=20, scale=50.0):
    """ABCD closure |NA*ND - NB*NC| / (NA*ND + NB*NC) averaged over random cuts."""
    v1 = var1.view(-1).float()
    v2 = var2.view(-1).float()
    w  = weights.view(-1).float()
    with torch.no_grad():
        x_min, x_max = torch.quantile(v1, 0.01).item(), torch.quantile(v1, 0.99).item()
        y_min, y_max = torch.quantile(v2, 0.01).item(), torch.quantile(v2, 0.99).item()
    v1_n = (v1 - x_min) / (x_max - x_min + 1e-8)
    v2_n = (v2 - y_min) / (y_max - y_min + 1e-8)
    losses = []
    for _ in range(n_cuts):
        for _ in range(max_tries):
            with torch.no_grad():
                c1 = np.random.uniform(0.0, 1.0)
                c2 = np.random.uniform(0.0, 1.0)
            NA, NB, NC, ND = _sigmoid_counts(v1_n, v2_n, c1, c2, w, scale=scale)
            if all(v.item() > n_events_min for v in [NA, NB, NC, ND]):
                break
        else:
            continue
        losses.append(torch.abs(NA * ND - NB * NC) / (NA * ND + NB * NC + 1e-8))
    if not losses:
        return torch.tensor(0.0, device=var1.device)
    return torch.stack(losses).mean()


def _proxy_md(latent, qcd_mask):
    """Batch-level squared Mahalanobis distance from QCD centroid (whitened PCA)."""
    if qcd_mask.sum() < 2:
        return torch.zeros(latent.size(0), device=latent.device)
    with torch.no_grad():
        bkg = latent[qcd_mask].detach().float()
        mu  = bkg.mean(0)
        centered = bkg - mu
        cov = (centered.T @ centered) / bkg.shape[0]
        L, V = torch.linalg.eigh(cov)
        W = V / L.clamp(min=1e-6).sqrt()
    z = (latent.float() - mu) @ W
    return (z * z).sum(dim=1).to(latent.dtype)


# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="HLT NURD contrastive training")
# Data
parser.add_argument("--data",       required=True,  type=str, help="Path to .pt training file")
parser.add_argument("--ae_ckpt",    required=True,  type=str, help="Path to pre-trained AE checkpoint (.pth)")
parser.add_argument("--val_split",  default=0.1,    type=float)
parser.add_argument("--n_bins",     default=10,     type=int, help="Nuisance bins for AE reco loss")
# Training
parser.add_argument("--epochs",         default=100,    type=int)
parser.add_argument("--reweight_epochs",default=0,      type=int)
parser.add_argument("--critic_epochs",  default=2,      type=int)
parser.add_argument("-b","--batch_size",default=2048,   type=int)
parser.add_argument("--lr",             default=1e-4,   type=float)
parser.add_argument("--weight_decay",   default=5e-3,   type=float)
parser.add_argument("--cosine",         default=1,      type=int)
parser.add_argument("--optimizer",      default="adam", type=str)
parser.add_argument("--momentum",       default=0.9,    type=float)
# NURD flags
parser.add_argument("--reweight",       default=1,      type=int)
parser.add_argument("--joint_indep",    default=1,      type=int)
parser.add_argument("--_lambda",        default=0.01,   type=float)
parser.add_argument("--marginal_indep", default=0,      type=int)
parser.add_argument("--critic_restart", default=0,      type=int)
parser.add_argument("--exact",          default=1,      type=int)
# Contrastive loss
parser.add_argument("--contrast_weight",default=0.05,   type=float)
parser.add_argument("--contrast_temp",  default=0.05,   type=float)
# Model architecture
parser.add_argument("--embed_size",     default=128,    type=int)
parser.add_argument("--latent_dim",     default=6,      type=int)
parser.add_argument("--proj_dim",       default=6,      type=int)
parser.add_argument("--num_heads",      default=8,      type=int)
parser.add_argument("--num_layers",     default=4,      type=int)
parser.add_argument("--dim_ff",         default=512,    type=int)
parser.add_argument("--linear_dim",     default=16,     type=int)
# Logging
parser.add_argument("--exp_name",       default="hlt_nurd_run", type=str)
parser.add_argument("--project_name",   default="hlt",          type=str)
parser.add_argument("--log_name",       default="info.log",     type=str)
parser.add_argument("--gpu_ids",        default="0",            type=str)
parser.add_argument("--local_rank",     default=-1,             type=int)
parser.add_argument("--manualSeed",     default=None,           type=int)
parser.add_argument("--local_testing",  default=0,              type=int)
parser.add_argument("--max_events",     default=-1,             type=int)
parser.add_argument("--critic_schedule", default="per_batch",   type=str,
                    help="'per_batch' (exact, trains critic on full dataset each batch) "
                         "or 'per_epoch' (trains critic once per epoch)")
parser.add_argument("--critic_type",    default="bin_pred",     type=str,
                    help="'bin_pred' (predict nuisance bin, our default) "
                         "or 'density_ratio' (gabhijith's shuffled-z binary classification)")
# Warmup critic schedule parameters (used when --critic_schedule warmup)
parser.add_argument("--critic_warmup_epochs", default=7,        type=int,
                    help="Epochs to train critic without applying penalty (let contrastive converge first)")
parser.add_argument("--critic_ramp_epochs",   default=10,       type=int,
                    help="Epochs to cosine-ramp lambda from 0 to target after warmup")
parser.add_argument("--critic_train_frac",        default=0.2,   type=float,
                    help="Fraction of batches per epoch on which to do a critic gradient step")
parser.add_argument("--critic_lr_multiplier",     default=10.0,  type=float,
                    help="LR multiplier for the critic optimizer relative to main model LR")
parser.add_argument("--n_critic_steps_per_batch", default=3,     type=int,
                    help="Number of gradient steps to take on the critic per selected batch")
parser.add_argument("--closure_weight",       default=0.0,   type=float,
                    help="Weight on ABCD closure loss (0 = disabled). Uses batch-level MD as axis 2.")
parser.add_argument("--qcd_label",            default=1,     type=int,
                    help="Label index for QCD (used as reference class for MD and closure)")
parser.add_argument("--no_mi_norm",           action="store_true",
                    help="Skip per-batch normalization of MI penalty (divide by batch mean). "
                         "Without this, lambda is effectively rescaled by ~1/raw_mi, making "
                         "different lambda values produce near-identical gradients.")
args, unknown = parser.parse_known_args()
print(f"Unknown args: {unknown}")

if not args.local_testing:
    import wandb
    wandb.init(name=args.exp_name,
               project="nurd-ood-" + args.project_name, reinit=True)
    wandb.config.update(args, allow_val_change=True)

# ── Setup ─────────────────────────────────────────────────────────────────────

directory = f"checkpoints/hlt/{args.project_name}/{args.exp_name}/"
os.makedirs(directory, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
set_random_seed(args.manualSeed)


# ── Metric helpers (same as train_exact.py) ───────────────────────────────────

def freeze_model(m):
    for p in m.parameters(): p.requires_grad_(False)
    return m

def unfreeze_model(m):
    for p in m.parameters(): p.requires_grad_(True)
    return m

def record_metrics(acc, loss, top1, inputs, outputs, targets, losses):
    prec1 = accuracy(outputs.data, targets, topk=(1,))[0]
    acc.update((torch.max(outputs,1)[1].data == targets).sum().data / len(outputs), inputs.size(0))
    loss.update(losses.mean().data, inputs.size(0))
    top1.update(prec1, inputs.size(0))
    return acc, loss, top1

def record_rw_metrics(acc, loss, inputs, outputs, targets, losses, weights):
    num_correct = torch.max(outputs,1)[1].data == targets
    acc.update((num_correct * weights).sum().data / weights.sum().data, inputs.size(0))
    loss.update((losses * weights).sum().data / weights.sum().data, inputs.size(0))
    return acc, loss

def log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss=None, rw_acc=None, split=None):
    log.debug(f"{split} Epoch [{epoch}] Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Acc {acc.avg:.3f}"
              + (f" RwLoss {rw_loss.avg:.4f} RwAcc {rw_acc.avg:.3f}" if rw_loss else ""))

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (0.1 ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def get_effective_lambda(epoch):
    """Cosine ramp: lambda=0 during warmup, then 0→target over critic_ramp_epochs."""
    if epoch < args.critic_warmup_epochs:
        return 0.0
    ramp_progress = min(1.0, (epoch - args.critic_warmup_epochs) / max(args.critic_ramp_epochs, 1))
    return args._lambda * (1 - math.cos(math.pi * ramp_progress)) / 2


def compute_critic_loss(inputs, labels, nuisances, model, critic_model,
                        critic_criterion, reweight_args, joint_indep_args, split="train"):
    activations, _ = model(inputs)
    y_in = (torch.zeros_like(labels.unsqueeze(1)).float().to(device)
            if joint_indep_args["marginal_indep"]
            else labels.unsqueeze(1).float().to(device))

    if joint_indep_args.get("critic_type") == "density_ratio":
        # gabhijith's density-ratio trick: classify real vs shuffled-z
        pos_out = critic_model(activations, y_in, nuisances)
        pos_losses = critic_criterion(pos_out, torch.ones_like(labels))
        shuffled_z = nuisances[torch.randperm(nuisances.size(0))]
        neg_out = critic_model(activations, y_in, shuffled_z)
        neg_losses = critic_criterion(neg_out, torch.zeros_like(labels))
        outputs = torch.cat([pos_out, neg_out], dim=0)
        targets = torch.cat([torch.ones_like(labels), torch.zeros_like(labels)])
        losses  = torch.cat([pos_losses, neg_losses])
        return outputs, targets, losses
    else:
        # bin-prediction approach (WHAT WE are doing rn)
        outputs = critic_model(activations, y_in)
        losses = critic_criterion(outputs, nuisances.long())
        nuisance_marginals = torch.tensor(
            [joint_indep_args["nuisance_prior"][int(z.item())] for z in nuisances]
        ).to(device)
        losses = torch.div(losses, nuisance_marginals + 1e-8)
        return outputs, nuisances, losses


def train_critic(critic_model, model, train_loader, critic_criterion, critic_optimizer,
                 epoch, log, reweight_args, joint_indep_args):
    critic_model.train()
    model.eval()
    batch_time = AverageMeter()
    rw_loss = AverageMeter()
    rw_acc = AverageMeter()
    end = time.time()
    for inputs, targets, nuisances, _ae_reco in train_loader:
        exact_weights = torch.tensor([
            reweight_args["train_dataset"].weights[(int(y.item()), int(z.item()))]
            for y, z in zip(targets, nuisances)
        ]).to(device)
        inputs, targets, nuisances = inputs.to(device), targets.long().to(device), nuisances.to(device)
        outputs, tgts, losses = compute_critic_loss(
            inputs, targets, nuisances, model, critic_model,
            critic_criterion, reweight_args, joint_indep_args, "train")
        weights = exact_weights if reweight_args["reweight"] else torch.ones_like(exact_weights)
        tensor_loss = (losses * weights).sum() / weights.sum()
        critic_optimizer.zero_grad()
        tensor_loss.backward(); critic_optimizer.step()
        batch_time.update(time.time() - end); end = time.time()
    log.debug(f"Train Critic Epoch [{epoch}]")
    return critic_model


def validate_critic(val_loader, critic_model, model, critic_criterion, epoch, log,
                    reweight_args, joint_indep_args):
    critic_model.eval(); model.eval()
    loss_m = AverageMeter(); rw_acc_m = AverageMeter(); acc_m = AverageMeter()
    with torch.no_grad():
        for inputs, targets, nuisances, _ae_reco in val_loader:
            exact_weights = torch.tensor([
                reweight_args["val_dataset"].weights.get((int(y.item()), int(z.item())), 1.0)
                for y, z in zip(targets, nuisances)
            ]).to(device)
            inputs, targets, nuisances = inputs.to(device), targets.long().to(device), nuisances.to(device)
            outputs, tgts, losses = compute_critic_loss(
                inputs, targets, nuisances, model, critic_model,
                critic_criterion, reweight_args, joint_indep_args, "val")
            weights = exact_weights if reweight_args["reweight"] else torch.ones_like(exact_weights)
            loss_m.update((losses * weights).sum().item() / weights.sum().item(), inputs.size(0))
    return loss_m.avg, acc_m.avg, rw_acc_m.avg


#training functions

#get contrastive loss
contrastive_loss_fn = SupConLoss(temperature=args.contrast_temp)

def train(model, train_loader, val_loader, criterion, optimizer, epoch, log,
          reweight_args, joint_indep_args, effective_lambda=None):
    batch_time = AverageMeter()
    acc = AverageMeter()
    loss = AverageMeter()
    top1 = AverageMeter()
    rw_acc = AverageMeter()
    rw_loss = AverageMeter()
    total_m = AverageMeter()   # total loss
    nurd_m = AverageMeter()   # NURD-weighted CE
    con_m = AverageMeter()   # contrastive
    closure_m  = AverageMeter()   # ABCD closure
    mi_m  = AverageMeter()   # MI / independence penalty (normalized, always ~1)
    raw_mi_m  = AverageMeter()   # raw critic CE before normalization
    weight_cv_m = AverageMeter()  # coeff. of variation of NURD weights (std/mean); 0 = uniform, >1 = heavy tails
    weight_ess_m = AverageMeter() # effective sample size fraction: ESS/N; 1.0 = no reweighting cost

    model.train()
    end = time.time()
    #look up the Nurd weight for every sample
    for inputs, targets, nuisances, ae_reco in train_loader:
        exact_weights = torch.tensor([
            reweight_args["train_dataset"].weights[(int(y.item()), int(z.item()))]
            for y, z in zip(targets, nuisances)
        ]).to(device)
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        nuisances = nuisances.to(device)
        ae_reco = ae_reco.to(device)

        #critic gets updated first before main model see this batch. 
        if joint_indep_args["joint_indep"] and joint_indep_args.get("critic_schedule") == "warmup":
            #take our critic train frac for each batch
            if random.random() < joint_indep_args["critic_train_frac"]:
                #critic unfreezes and goes into train mode (eval main model)
                joint_indep_args["critic_model"] = unfreeze_model(joint_indep_args["critic_model"])
                joint_indep_args["critic_model"].train()
                model.eval()
                #run main model to get activations 
                with torch.no_grad(): #activations are detached (critic's gradient update can't affect main models weights)
                    act_detached, _ = model(inputs)
                #critic takes latent representation and class label as input
                y_in = (torch.zeros_like(targets.unsqueeze(1)).float().to(device)
                        if joint_indep_args["marginal_indep"]
                        else targets.unsqueeze(1).float().to(device))
                #looks up P(z=k) for each sample (probabliity of that nuisance bin - used to normalize the critic)
                nu_marg = torch.tensor(
                    [joint_indep_args["nuisance_prior"][int(z.item())] for z in nuisances]
                ).to(device)
                # multiple steps on the same batch to help critic converge faster
                # critic is trained unweighted so it sees the natural latent-z correlation
                n_steps = joint_indep_args.get("n_critic_steps_per_batch", 1)
                for _ in range(n_steps): #in this case we are oding 3 steps
                    c_out = joint_indep_args["critic_model"](act_detached, y_in)
                    c_losses = joint_indep_args["critic_criterion"](c_out, nuisances.long())
                    c_losses = torch.div(c_losses, nu_marg + 1e-8)
                    c_loss = c_losses.mean()
                    joint_indep_args["critic_optimizer"].zero_grad()
                    c_loss.backward()
                    joint_indep_args["critic_optimizer"].step()
                #refreeze critic and put main model in train mode again
                joint_indep_args["critic_model"] = freeze_model(joint_indep_args["critic_model"])
                model.train()

        # ── joint independence: one critic gradient step on this batch (interleaved) ─
        # elif joint_indep_args["joint_indep"] and joint_indep_args.get("critic_schedule") == "interleaved":
        #     joint_indep_args["critic_model"] = unfreeze_model(joint_indep_args["critic_model"])
        #     joint_indep_args["critic_model"].train()
        #     model.eval()
        #     with torch.no_grad():
        #         act_detached, _ = model(inputs)
        #     y_in = (torch.zeros_like(targets.unsqueeze(1)).float().to(device)
        #             if joint_indep_args["marginal_indep"]
        #             else targets.unsqueeze(1).float().to(device))
        #     c_out    = joint_indep_args["critic_model"](act_detached, y_in)
        #     c_losses = joint_indep_args["critic_criterion"](c_out, nuisances.long())
        #     nu_marg  = torch.tensor(
        #         [joint_indep_args["nuisance_prior"][int(z.item())] for z in nuisances]
        #     ).to(device)
        #     c_losses = torch.div(c_losses, nu_marg + 1e-8)
        #     w = exact_weights if reweight_args["reweight"] else torch.ones_like(exact_weights)
        #     c_loss = (c_losses * w).sum() / w.sum()
        #     joint_indep_args["critic_optimizer"].zero_grad()
        #     c_loss.backward()
        #     joint_indep_args["critic_optimizer"].step()
        #     joint_indep_args["critic_model"] = freeze_model(joint_indep_args["critic_model"])
        #     model.train()

        # ── joint independence: train critic inner loop (per_batch schedule) ───
        # elif joint_indep_args["joint_indep"] and joint_indep_args.get("critic_schedule") == "per_batch":
        #     best_loss = None
        #     joint_indep_args["critic_model"] = unfreeze_model(joint_indep_args["critic_model"])
        #     model = freeze_model(model)
        #     critic_optimizer = torch.optim.Adam(
        #         joint_indep_args["critic_model"].parameters(),
        #         lr=joint_indep_args["lr"], weight_decay=joint_indep_args["weight_decay"])
        #     for ce in range(joint_indep_args["critic_epochs"]):
        #         joint_indep_args["critic_model"] = train_critic(
        #             joint_indep_args["critic_model"], model, train_loader,
        #             joint_indep_args["critic_criterion"], critic_optimizer, ce, log,
        #             reweight_args, joint_indep_args)
        #         c_loss, c_acc, c_rw_acc = validate_critic(
        #             val_loader, joint_indep_args["critic_model"], model,
        #             joint_indep_args["critic_criterion"], ce, log,
        #             reweight_args, joint_indep_args)
        #         if best_loss is None or c_loss < best_loss:
        #             best_loss = c_loss
        #             save_checkpoint(args, {
        #                 "epoch": ce+1,
        #                 "state_dict_model": joint_indep_args["critic_model"].state_dict()
        #             }, ce+1, name="critic")
        #     ckpt_file = f"checkpoints/hlt/{args.project_name}/{args.exp_name}/checkpoint_critic.pth.tar"
        #     joint_indep_args["critic_model"].load_state_dict(
        #         torch.load(ckpt_file)["state_dict_model"])
        #     joint_indep_args["critic_model"] = freeze_model(joint_indep_args["critic_model"])
        #     model = unfreeze_model(model)

        #forward pass
        activations, outputs = model(inputs)
        losses_ce = criterion(outputs, targets)         # [B] CE loss

        acc, loss, top1 = record_metrics(acc, loss, top1, inputs, outputs, targets, losses_ce)

        # ── NURD joint independence penalty ───────────────────────────────────
        info_loss_val = 0.0 #normalized MI penalty
        raw_mi_val = 0.0 #raw MI penalty
     
        if joint_indep_args["joint_indep"]:
            #use ramped lambda during warmup, else fix lambda
            lam = effective_lambda if effective_lambda is not None else joint_indep_args["lambda"]
            
            with (torch.no_grad() if lam == 0.0 else torch.enable_grad()):
                # run frozen critic on current activations → per-sample losses [B]
                # low loss = critic can predict nuisance bin = encoder is leaking nuisance info
                # high loss = critic can't predict nuisance bin = encoder is already independent
                _, _, info_losses = compute_critic_loss(
                    inputs, targets, nuisances, model,
                    joint_indep_args["critic_model"], joint_indep_args["critic_criterion"],
                    reweight_args, joint_indep_args, "train")
            #not doing this right now
            if joint_indep_args.get("critic_type") == "density_ratio":
                half = len(info_losses) // 2
                penalty = info_losses[half:] - info_losses[:half]
                raw_mi_val = penalty.mean().item()
                if lam > 0.0:
                    losses_ce = losses_ce + lam * penalty
            #bin pred critic
            else:
                raw_mi_val = info_losses.mean().item()
                if lam > 0.0:
                    if not args.no_mi_norm:
                        info_losses = info_losses / (info_losses.detach().mean() + 1e-8)
                    # subtract because high critic loss = encoder already independent = good
                    # gradient rewards encoder for confusing the critic
                    losses_ce = losses_ce - lam * info_losses
                    info_loss_val = info_losses.mean().item()

        #nurd reweighting
        weights = exact_weights.to(device) if reweight_args["reweight"] else torch.ones_like(exact_weights).to(device)
        rw_acc, rw_loss = record_rw_metrics(rw_acc, rw_loss, inputs, outputs, targets, losses_ce, weights)
        loss_nurd = (losses_ce * weights).sum() / weights.sum()

        #contrastive loss
        embeddings  = model.get_embeddings(activations)
        loss_con = contrastive_loss_fn(embeddings, targets)
        tensor_loss = (1 - args.contrast_weight) * loss_nurd + args.contrast_weight * loss_con

        #ABCD closure loss (batch level MD)
        loss_closure = torch.tensor(0.0, device=device)
        if args.closure_weight > 0.0:
            qcd_mask = targets == args.qcd_label
            if qcd_mask.sum() > 10:
                proxy_md = _proxy_md(activations, qcd_mask)
                nw = torch.ones(qcd_mask.sum(), device=device)
                loss_closure = closure_loss_batch(
                    ae_reco[qcd_mask].float(),
                    proxy_md[qcd_mask].float(),
                    nw,
                )
                tensor_loss = tensor_loss + args.closure_weight * loss_closure

        optimizer.zero_grad()
        tensor_loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        batch_time.update(time.time() - end); end = time.time()
        w_mean = weights.mean()
        w_std  = weights.std()
        ess = (w_mean ** 2 / (weights ** 2).mean()).item()   # ESS / batch_size
        total_m.update(tensor_loss.item(),      bs)
        nurd_m.update(loss_nurd.item(),         bs)
        con_m.update(loss_con.item(),           bs)
        closure_m.update(loss_closure.item(),   bs)
        mi_m.update(info_loss_val,              bs)
        raw_mi_m.update(raw_mi_val,             bs)
        weight_cv_m.update((w_std / (w_mean + 1e-8)).item(), bs)
        weight_ess_m.update(ess,                bs)

    #logging
    log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss, rw_acc, split="Train")
    current_lr = optimizer.param_groups[0]["lr"]
    log.debug(f"  total={total_m.avg:.5f}  nurd={nurd_m.avg:.5f}  "
              f"con={con_m.avg:.5f}  closure={closure_m.avg:.5f}  "
              f"mi={mi_m.avg:.5f}  raw_mi={raw_mi_m.avg:.5f}  "
              f"w_cv={weight_cv_m.avg:.3f}  w_ess={weight_ess_m.avg:.3f}  lr={current_lr:.2e}")
    if not args.local_testing:
        wandb.log({
            "Train/total_loss":       total_m.avg,
            "Train/nurd_weighted_ce": nurd_m.avg,
            "Train/contrastive":      con_m.avg,
            "Train/closure":          closure_m.avg,
            "Train/mi_penalty":       mi_m.avg,
            "Train/raw_mi_penalty":   raw_mi_m.avg,
            "Train/nurd_weight_cv":   weight_cv_m.avg,
            "Train/nurd_weight_ess":  weight_ess_m.avg,
            "Train/rw_loss":          rw_loss.avg,
            "Train/rw_acc":           rw_acc.avg,
            "Train/acc":              acc.avg,
            "Train/prec1":            top1.avg,
            "LR":                     current_lr,
            "Train/effective_lambda": effective_lambda if effective_lambda is not None else args._lambda,
        }, step=epoch)


def validate(val_loader, model, criterion, epoch, log, reweight_args):
    batch_time = AverageMeter()
    acc = AverageMeter(); loss = AverageMeter(); top1 = AverageMeter()
    rw_acc = AverageMeter(); rw_loss = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for inputs, targets, nuisances, _ae_reco in val_loader:
            exact_weights = torch.tensor([
                reweight_args["val_dataset"].weights.get((int(y.item()), int(z.item())), 1.0)
                for y, z in zip(targets, nuisances)
            ]).to(device)
            inputs    = inputs.to(device)
            targets   = targets.long().to(device)
            _, outputs = model(inputs)
            losses    = criterion(outputs, targets)

            acc, loss, top1 = record_metrics(acc, loss, top1, inputs, outputs, targets, losses)
            if reweight_args["reweight"]:
                rw_acc, rw_loss = record_rw_metrics(
                    rw_acc, rw_loss, inputs, outputs, targets, losses,
                    exact_weights.to(device))
            batch_time.update(time.time() - end); end = time.time()

    log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss, rw_acc, split="Val")
    if not args.local_testing:
        wandb.log({
            "Val/loss":    loss.avg,
            "Val/prec1":   top1.avg,
            "Val/acc":     acc.avg,
            "Val/rw_loss": rw_loss.avg,
            "Val/rw_acc":  rw_acc.avg,
        }, step=epoch)
    return_loss = rw_loss.avg if reweight_args["reweight"] else loss.avg
    return return_loss, acc.avg, rw_acc.avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    #logging
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(directory, args.log_name), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s : %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s : %(message)s"))
    log.addHandler(fh); log.addHandler(sh)

    args.in_dataset = "hlt"   # required by save_checkpoint path construction

    #load the frozen AE (pretrained)
    ae_ckpt = torch.load(args.ae_ckpt, map_location=device)
    ae_cfg  = ae_ckpt.get("ae_config", {
        "features": None, "latent_dim": 16,
        "encoder_config": {"nodes": [512,256]},
        "decoder_config": {"nodes": [256,512, None]},
        "alpha": 1.0
    })
    
    if ae_cfg["features"] is None:
        first_w = ae_ckpt["ae"][next(iter(ae_ckpt["ae"]))]
        ae_cfg["features"] = first_w.shape[1]

    ae = HLTAutoencoder(ae_cfg).to(device)
    ae.load_state_dict(ae_ckpt["ae"])
    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)
    log.debug(f"Loaded frozen AE from {args.ae_ckpt}")

    #loads data and also AE reco losses to then bin into nuisance categories inside dataset builder (norm saved for inference)
    log.debug("Loading data and computing nuisance bins (AE reco)...")
    train_dataset, val_dataset, obj_scaler = build_hlt_datasets(
        args.data, ae, n_bins=args.n_bins,
        val_split=args.val_split, max_events=args.max_events
    )
    log.debug(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    num_classes = int(train_dataset.labels.max().item()) + 1
    num_tokens  = train_dataset.features.size(1)

    kwargs = {"pin_memory": False, "num_workers": 0, "drop_last": True}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, **kwargs)

    #nuisance prior is marginal probability of each bin (used to normalize critic loss)
    label_prior    = train_dataset.get_label_prior()
    nuisance_prior = train_dataset.get_nuisance_prior() if args.joint_indep else None

    # load HLT model
    model = HLTContrastiveModel(
        num_classes=num_classes,
        embed_size=args.embed_size,
        latent_dim=args.latent_dim,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        linear_dim=args.linear_dim,
        num_tokens=num_tokens,
    ).to(device)

    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=args.momentum)

    #load critic model
    critic_model = HLTCritic(args.latent_dim, num_classes, args.n_bins,
                             critic_type=args.critic_type).to(device) if args.joint_indep else None

    reweight_args = {
        "reweight":      args.reweight,
        "label_prior":   label_prior,
        "train_dataset": train_dataset,
        "val_dataset":   val_dataset,
    }
    joint_indep_args = {
        "joint_indep":      args.joint_indep,
        "critic_model":     critic_model,
        "lr":               args.lr,
        "weight_decay":     args.weight_decay,
        "critic_epochs":    args.critic_epochs,
        "marginal_indep":   args.marginal_indep,
        "lambda":           args._lambda,
        "nuisance_prior":   nuisance_prior,
        "critic_criterion": nn.CrossEntropyLoss(reduction="none").to(device),
        "critic_schedule":  args.critic_schedule,
        "critic_type":      args.critic_type,
        "critic_train_frac": args.critic_train_frac,
        "critic_optimizer": (torch.optim.Adam(critic_model.parameters(),
                                              lr=args.lr * args.critic_lr_multiplier,
                                              weight_decay=args.weight_decay)
                             if args.joint_indep and args.critic_schedule in ("interleaved", "warmup")
                             else None),
        "n_critic_steps_per_batch": args.n_critic_steps_per_batch,
    }

    cudnn.benchmark = True
    best_loss = None
    for epoch in range(args.epochs):
        log.debug(f"Epoch {epoch}")
        adjust_learning_rate(optimizer, epoch)

        #per epoch (train critic once per epoch) THIS IS NOT USED rn (Skipped)
        if args.joint_indep and args.critic_schedule == "per_epoch":
            joint_indep_args["critic_model"] = unfreeze_model(joint_indep_args["critic_model"])
            model = freeze_model(model)
            critic_optimizer = torch.optim.Adam(
                joint_indep_args["critic_model"].parameters(),
                lr=args.lr, weight_decay=args.weight_decay)
            best_critic_loss = None
            for ce in range(args.critic_epochs):
                joint_indep_args["critic_model"] = train_critic(
                    joint_indep_args["critic_model"], model, train_loader,
                    joint_indep_args["critic_criterion"], critic_optimizer, ce, log,
                    reweight_args, joint_indep_args)
                c_loss, _, _ = validate_critic(
                    val_loader, joint_indep_args["critic_model"], model,
                    joint_indep_args["critic_criterion"], ce, log,
                    reweight_args, joint_indep_args)
                if best_critic_loss is None or c_loss < best_critic_loss:
                    best_critic_loss = c_loss
                    save_checkpoint(args, {
                        "epoch": ce + 1,
                        "state_dict_model": joint_indep_args["critic_model"].state_dict()
                    }, ce + 1, name="critic")
            ckpt_file = f"checkpoints/hlt/{args.project_name}/{args.exp_name}/checkpoint_critic.pth.tar"
            joint_indep_args["critic_model"].load_state_dict(
                torch.load(ckpt_file)["state_dict_model"])
            joint_indep_args["critic_model"] = freeze_model(joint_indep_args["critic_model"])
            model = unfreeze_model(model)

        #ramp lambda
        effective_lambda = get_effective_lambda(epoch) if args.critic_schedule == "warmup" else None

        #all the critic logic here (loss computation, weight updates)
        train(model, train_loader, val_loader, criterion, optimizer,
              epoch + args.reweight_epochs, log, reweight_args, joint_indep_args, effective_lambda)
        #runs model on validation set with no gradient updates (just forward passes)
        val_loss, val_acc, val_rw_acc = validate(
            val_loader, model, criterion, epoch + args.reweight_epochs, log, reweight_args)

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            log.debug("Saving checkpoint")
            save_checkpoint(args, {
                "epoch": epoch + 1,
                "state_dict_model": model.state_dict(),
                "ae_scaler": obj_scaler,
                "config": vars(args),
            }, epoch + 1, name="main")
            if not args.local_testing:
                wandb.run.summary["best_val_rw_acc"] = val_rw_acc
                wandb.run.summary["best_val_acc"]    = val_acc

    log.debug(f"Done. Best val loss: {best_loss:.5f}")
    if not args.local_testing:
        wandb.finish()


if __name__ == "__main__":
    main()
