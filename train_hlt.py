"""
HLT NURD contrastive training.

Built on top of gabhijith's train_exact.py, with an HLT-specific QCD closure
objective and a memory-safe interleaved density-ratio critic:

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
from utils.hlt_training_stats import RunningQCDMDProxy

# ── Contrastive losses ────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.05, base_temperature=0.05):
        super().__init__()
        self.T  = temperature
        self.Tb = base_temperature

    def forward(self, features, labels, weights=None):
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
        loss = loss.view(B)
        if weights is not None:
            weights = weights.float().view(-1).to(device)
            return (loss * weights).sum() / weights.sum().clamp(min=1e-8)
        return loss.mean()


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


def compute_proxy_md(latent, ref_mask, md_proxy=None, proxy_type="batch",
                     update=True):
    if proxy_type == "ema" and md_proxy is not None:
        return md_proxy.md(latent, ref_mask, update=update)
    return _proxy_md(latent, ref_mask)


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v) for v in str(value).replace(",", " ").split()]


def parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).replace(",", " ").split() if str(v).strip()]


def distance_corr_loss(x, y, max_samples=512, eps=1e-8):
    """Differentiable squared distance correlation for one-dimensional tensors."""
    x = x.float().view(-1)
    y = y.float().view(-1)
    n = x.numel()
    if n < 4:
        zero = (x.sum() + y.sum()) * 0.0
        return zero, None
    if max_samples > 0 and n > max_samples:
        idx = torch.randperm(n, device=x.device)[:max_samples]
        x = x[idx]
        y = y[idx]
    x_dist = torch.cdist(x.view(-1, 1), x.view(-1, 1), p=1)
    y_dist = torch.cdist(y.view(-1, 1), y.view(-1, 1), p=1)
    x_centered = x_dist - x_dist.mean(dim=0, keepdim=True) - x_dist.mean(dim=1, keepdim=True) + x_dist.mean()
    y_centered = y_dist - y_dist.mean(dim=0, keepdim=True) - y_dist.mean(dim=1, keepdim=True) + y_dist.mean()
    dcov = (x_centered * y_centered).mean()
    dvar_x = (x_centered * x_centered).mean()
    dvar_y = (y_centered * y_centered).mean()
    dcor = dcov / torch.sqrt(dvar_x * dvar_y + eps)
    dcor = dcor.clamp(min=0.0)
    return dcor, dcor.detach().item()


def profile_flatness_loss(x, y, n_bins=8, tail_weight=2.0, eps=1e-8):
    """Penalize trends in the mean MD profile versus AE loss quantile."""
    x = x.float().view(-1)
    y = y.float().view(-1)
    if x.numel() < max(4, n_bins * 2):
        zero = (x.sum() + y.sum()) * 0.0
        return zero, None
    with torch.no_grad():
        edges = torch.quantile(x.detach(), torch.linspace(0, 1, n_bins + 1, device=x.device))
        bin_ids = torch.bucketize(x.detach(), edges[1:-1])
    global_mean = y.mean()
    global_scale = y.std(unbiased=False).detach().clamp(min=eps)
    losses = []
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() < 3:
            continue
        rel_tail = i / max(n_bins - 1, 1)
        weight = 1.0 + tail_weight * rel_tail * rel_tail
        bin_mean = y[mask].mean()
        losses.append(weight * ((bin_mean - global_mean) / global_scale).pow(2))
    if not losses:
        zero = (x.sum() + y.sum()) * 0.0
        return zero, None
    loss = torch.stack(losses).mean()
    return loss, loss.detach().item()


def soft_abcd_tail_loss(x, y, quantiles, min_events=5, scale=12.0,
                        tail_focus_weight=2.0, eps=1e-6):
    """Deterministic soft ABCD loss on high QCD quantile cuts."""
    x = x.float().view(-1)
    y = y.float().view(-1)
    quantiles = [q for q in quantiles if 0.0 < q < 1.0]
    if x.numel() < max(20, 4 * min_events) or not quantiles:
        zero = (x.sum() + y.sum()) * 0.0
        return zero, None

    with torch.no_grad():
        x_scale = (torch.quantile(x.detach(), 0.84) - torch.quantile(x.detach(), 0.16)).clamp(min=eps)
        y_scale = (torch.quantile(y.detach(), 0.84) - torch.quantile(y.detach(), 0.16)).clamp(min=eps)
        cuts_x = {q: torch.quantile(x.detach(), q) for q in quantiles}
        cuts_y = {q: torch.quantile(y.detach(), q) for q in quantiles}

    losses = []
    abs_log_ratios = []
    reliabilities = []
    for qx in quantiles:
        cut_x = cuts_x[qx]
        hard_x_high = x.detach() > cut_x
        sx_high = torch.sigmoid(scale * (x - cut_x) / x_scale)
        sx_low = 1.0 - sx_high
        for qy in quantiles:
            cut_y = cuts_y[qy]
            hard_y_high = y.detach() > cut_y
            hard_counts = (
                (hard_x_high & hard_y_high).sum(),
                (hard_x_high & ~hard_y_high).sum(),
                (~hard_x_high & hard_y_high).sum(),
                (~hard_x_high & ~hard_y_high).sum(),
            )
            hard_min = torch.stack(hard_counts).float().min()
            reliability = torch.sigmoid((hard_min - float(min_events)) / max(float(min_events) * 0.25, 1.0)).detach()

            sy_high = torch.sigmoid(scale * (y - cut_y) / y_scale)
            sy_low = 1.0 - sy_high
            A = (sx_high * sy_high).sum()
            B = (sx_high * sy_low).sum()
            C = (sx_low * sy_high).sum()
            D = (sx_low * sy_low).sum()
            log_ratio = torch.log(B + eps) + torch.log(C + eps) - torch.log(D + eps) - torch.log(A + eps)
            rel_tail = max(0.0, ((qx + qy) * 0.5 - 0.5) / 0.5)
            weight = 1.0 + tail_focus_weight * rel_tail * rel_tail
            losses.append(reliability * weight * log_ratio.pow(2))
            abs_log_ratios.append(torch.abs(log_ratio).detach())
            reliabilities.append(reliability)

    if not losses:
        zero = (x.sum() + y.sum()) * 0.0
        return zero, None
    loss = torch.stack(losses).mean()
    rel = torch.stack(reliabilities)
    metric = (torch.stack(abs_log_ratios) * rel).sum() / rel.sum().clamp(min=eps)
    return loss, metric.item()


def closure_dependency_loss(ae_reco, proxy_md_values, args, eps=1e-8):
    """Direct QCD dependence loss for the two ABCD axes."""
    x = torch.log(ae_reco.float().clamp(min=eps))
    y = torch.log1p(proxy_md_values.float().clamp(min=0.0))
    corr_loss, corr_val = closure_corr_loss(ae_reco, proxy_md_values, eps=eps)
    if args.closure_loss_type == "corr":
        return corr_loss, corr_val, {"corr": corr_val}
    if args.closure_loss_type in {"dcorr_profile", "hybrid", "tail_abcd"}:
        total = args.closure_corr_weight * corr_loss
        diag = {"corr": corr_val}
        if args.closure_loss_type in {"dcorr_profile", "hybrid"}:
            dcorr, dcorr_val = distance_corr_loss(
                x, y, max_samples=args.closure_dcorr_max_samples, eps=eps)
            profile, profile_val = profile_flatness_loss(
                x, y, n_bins=args.closure_profile_bins,
                tail_weight=args.closure_profile_tail_weight, eps=eps)
            total = total + args.closure_dcorr_weight * dcorr + args.closure_profile_weight * profile
            diag.update({"dcorr": dcorr_val, "profile": profile_val})
            if args.closure_reverse_profile_weight > 0.0:
                profile_rev, profile_rev_val = profile_flatness_loss(
                    y, x, n_bins=args.closure_profile_bins,
                    tail_weight=args.closure_profile_tail_weight, eps=eps)
                total = total + args.closure_reverse_profile_weight * profile_rev
                diag["profile_reverse"] = profile_rev_val
        if args.closure_loss_type in {"hybrid", "tail_abcd"} and args.closure_tail_abcd_weight > 0.0:
            tail_loss, tail_val = soft_abcd_tail_loss(
                x, y, args.closure_tail_quantiles_values,
                min_events=args.closure_tail_min_events,
                scale=args.closure_tail_scale,
                tail_focus_weight=args.closure_tail_focus_weight,
                eps=eps)
            total = total + args.closure_tail_abcd_weight * tail_loss
            diag["tail_abcd"] = tail_val
        return total, corr_val, diag
    raise ValueError(f"Unsupported closure_loss_type={args.closure_loss_type!r}")


def compute_qcd_closure_loss(activations, labels, ae_reco, args, md_proxy=None,
                             update=True):
    """Apply closure only to QCD while leaving CE/SupCon all-background."""
    qcd_mask = labels.long() == int(args.qcd_label)
    n_qcd = int(qcd_mask.sum().item())
    if n_qcd < args.closure_class_min_events:
        zero = activations.sum() * 0.0
        return zero, {}, n_qcd

    proxy_md = compute_proxy_md(
        activations, qcd_mask, md_proxy=md_proxy,
        proxy_type=args.md_proxy_type, update=update)
    if args.closure_loss_type == "abcd":
        weights = torch.ones(n_qcd, device=activations.device)
        loss = closure_loss_batch(
            ae_reco[qcd_mask].float(), proxy_md[qcd_mask].float(), weights)
        corr = _safe_pearson_torch(ae_reco[qcd_mask], proxy_md[qcd_mask])
        return loss, {"corr": corr}, n_qcd

    loss, corr, diagnostics = closure_dependency_loss(
        ae_reco[qcd_mask], proxy_md[qcd_mask], args)
    diagnostics["corr"] = corr
    return loss, diagnostics, n_qcd


def abcd_grid_metrics_np(x, y, quantiles, min_count=20,
                         min_count_fraction=0.0, tail_min_quantile=0.8):
    """Hard-count validation proxy for ABCD stability on QCD only."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    quantiles = [q for q in quantiles if 0.0 < q < 1.0]
    effective_min_count = max(
        int(min_count), int(math.ceil(float(min_count_fraction) * x.size)))
    if x.size < max(20, 4 * effective_min_count) or not quantiles:
        return {"score": float("nan"), "n_points": 0}

    x_cuts = {q: np.quantile(x, q) for q in quantiles}
    y_cuts = {q: np.quantile(y, q) for q in quantiles}
    abs_logs = []
    tail_abs_logs = []
    eps = 1e-9
    for qx in quantiles:
        x_high = x > x_cuts[qx]
        for qy in quantiles:
            y_high = y > y_cuts[qy]
            A = int(np.count_nonzero(x_high & y_high))
            B = int(np.count_nonzero(x_high & ~y_high))
            C = int(np.count_nonzero(~x_high & y_high))
            D = int(np.count_nonzero(~x_high & ~y_high))
            if min(A, B, C, D) < effective_min_count:
                continue
            a_hat = B * C / max(D, eps)
            abs_log = abs(math.log((a_hat + eps) / (A + eps)))
            abs_logs.append(abs_log)
            if qx >= tail_min_quantile or qy >= tail_min_quantile:
                tail_abs_logs.append(abs_log)

    if not abs_logs:
        return {"score": float("nan"), "n_points": 0}

    arr = np.asarray(abs_logs, dtype=np.float64)
    tail_arr = np.asarray(tail_abs_logs if tail_abs_logs else abs_logs, dtype=np.float64)
    median = float(np.median(arr))
    p90 = float(np.quantile(arr, 0.90))
    tail_mean = float(np.mean(tail_arr))
    score = p90 + 0.5 * tail_mean + 0.25 * median
    return {
        "score": float(score),
        "n_points": int(arr.size),
        "min_region_count": int(effective_min_count),
        "mean_abs_log_nonclosure": float(np.mean(arr)),
        "median_abs_log_nonclosure": median,
        "p90_abs_log_nonclosure": p90,
        "tail_mean_abs_log_nonclosure": tail_mean,
    }


# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="HLT NURD contrastive training")
# Data
parser.add_argument("--data",       required=True,  type=str, help="Path to .pt training file")
parser.add_argument("--ae_ckpt",    required=True,  type=str, help="Path to pre-trained AE checkpoint (.pth)")
parser.add_argument("--val_split",  default=0.1,    type=float)
parser.add_argument("--n_bins",     default=20,     type=int, help="Nuisance bins for AE reco loss")
parser.add_argument("--nuisance_bin_scope", default="qcd",
                    choices=["all", "qcd", "per_class", "per_label", "baseline_per_class"],
                    help="Events used to define AE-loss quantile bins. 'per_class' defines bins "
                         "separately inside each baseline class.")
parser.add_argument("--baseline_labels", default="0,1,2,3", type=str,
                    help="Comma/space-separated background labels to target for all-baseline training/eval.")
# Training
parser.add_argument("--epochs",         default=100,    type=int)
parser.add_argument("--reweight_epochs",default=0,      type=int)
parser.add_argument("--critic_epochs",  default=2,      type=int)
parser.add_argument("-b","--batch_size",default=2048,   type=int)
parser.add_argument("--num_workers",    default=0,      type=int,
                    help="DataLoader workers. Keep 0 on shared clusters unless profiling says otherwise.")
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
parser.add_argument("--max_weight_ratio", default=10.0, type=float,
                    help="Clip NURD exact weights at this value after mean-normalization.")
# Contrastive loss
parser.add_argument("--contrast_weight",default=0.02,   type=float)
parser.add_argument("--contrast_weight_start", default=0.15, type=float,
                    help="Optional starting contrastive weight; cosine-annealed to --contrast_weight.")
parser.add_argument("--contrast_ramp_epochs", default=40, type=int,
                    help="Epochs over which contrastive weight moves from start to final.")
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
parser.add_argument("--critic_schedule", default="warmup",
                    choices=["warmup", "per_epoch", "per_batch"],
                    help="'warmup' trains the critic on selected batches and ramps lambda; "
                         "'per_epoch' trains it before each main epoch. 'per_batch' is kept "
                         "only to fail fast because that old block is disabled.")
parser.add_argument("--critic_type",    default="density_ratio",
                    choices=["bin_pred", "density_ratio"],
                    help="'density_ratio' uses the NURD shuffled-z binary critic; "
                         "'bin_pred' predicts nuisance bins directly.")
parser.add_argument("--critic_penalty_type", default="ratio_to_one",
                    choices=["ratio_to_one", "confusion", "logit_ratio", "ce_gap"],
                    help="Encoder-side density-ratio penalty. 'ratio_to_one' minimizes uniform-target "
                         "cross entropy, making the learned density ratio approach one; "
                         "'confusion' pushes real/shuffled logits to 0; "
                         "'logit_ratio' preserves the previous HLT proxy; 'ce_gap' matches the older script.")
parser.add_argument("--critic_scope",   default="qcd",
                    choices=["all", "qcd", "baselines", "all_baselines"],
                    help="Which events train/apply the nuisance critic. 'all' preserves old behavior; "
                         "'qcd' targets the ABCD closure background directly; 'baselines' targets "
                         "--baseline_labels.")
parser.add_argument("--critic_shuffle", default="global",
                    choices=["within_label", "global"],
                    help="For density-ratio critics, shuffle nuisance bins within labels to test "
                         "r ⟂ z conditional on class instead of learning class/nuisance priors.")
# Warmup critic schedule parameters (used when --critic_schedule warmup)
parser.add_argument("--critic_warmup_epochs", default=7,        type=int,
                    help="Epochs to train critic without applying penalty (let contrastive converge first)")
parser.add_argument("--critic_ramp_epochs",   default=10,       type=int,
                    help="Epochs to cosine-ramp lambda from 0 to target after warmup")
parser.add_argument("--critic_train_frac",        default=1.0,   type=float,
                    help="Fraction of batches per epoch on which to do a critic gradient step")
parser.add_argument("--critic_lr_multiplier",     default=10.0,  type=float,
                    help="LR multiplier for the critic optimizer relative to main model LR")
parser.add_argument("--n_critic_steps_per_batch", default=1,     type=int,
                    help="Number of gradient steps to take on the critic per selected batch")
parser.add_argument("--closure_weight",       default=1.0,   type=float,
                    help="Weight on closure regularization (0 = disabled).")
parser.add_argument("--closure_scope", default="qcd",
                    choices=["qcd", "baselines", "all", "all_baselines"],
                    help="Classes included in the direct closure loss. 'baselines' averages the "
                         "closure loss over --baseline_labels.")
parser.add_argument("--closure_class_min_events", default=20, type=int,
                    help="Minimum events from a class in a batch before adding its closure term.")
parser.add_argument("--closure_class_weighting", default="equal", choices=["equal", "count"],
                    help="Average per-class closure terms equally or proportional to class count.")
parser.add_argument("--closure_score_mode", default="own_class",
                    choices=["own_class", "union", "hybrid"],
                    help="Apply direct closure to each class's own MD, to a smooth all-class "
                         "union score, or to both. The union term aligns training with the "
                         "deployable all-background anomaly score.")
parser.add_argument("--closure_union_scope", default="qcd",
                    choices=["qcd", "baselines", "all_baselines", "all"],
                    help="Event population used by the smooth all-class union closure term.")
parser.add_argument("--closure_union_temperature", default=1.0, type=float,
                    help="Temperature of the differentiable min over per-class EMA MD scores.")
parser.add_argument("--closure_own_weight", default=0.5, type=float,
                    help="Relative weight of auxiliary own-class closure terms.")
parser.add_argument("--closure_union_weight", default=1.0, type=float,
                    help="Relative weight of the deployed-score union closure term.")
parser.add_argument("--closure_weight_start", default=0.0, type=float,
                    help="Starting closure weight; cosine-ramped to --closure_weight.")
parser.add_argument("--closure_ramp_epochs", default=15, type=int,
                    help="Epochs over which closure weight ramps from start to final.")
parser.add_argument("--closure_loss_type", default="hybrid",
                    choices=["corr", "dcorr_profile", "hybrid", "tail_abcd", "abcd", "none"],
                    help="'corr' penalizes QCD log(AE)-log(MD) correlation cheaply; "
                         "'dcorr_profile' adds nonlinear distance-correlation and profile flatness; "
                         "'hybrid' also adds reverse-profile and soft high-quantile ABCD terms; "
                         "'tail_abcd' uses only the high-quantile soft ABCD term; "
                         "'abcd' uses the older random-cut batch ABCD proxy; "
                         "'none' disables the term regardless of closure_weight.")
parser.add_argument("--closure_corr_weight", default=0.25, type=float,
                    help="Internal weight for Pearson-correlation component of dcorr_profile closure loss.")
parser.add_argument("--closure_dcorr_weight", default=1.0, type=float,
                    help="Internal weight for distance-correlation component of dcorr_profile closure loss.")
parser.add_argument("--closure_profile_weight", default=0.5, type=float,
                    help="Internal weight for profile-flatness component of dcorr_profile closure loss.")
parser.add_argument("--closure_reverse_profile_weight", default=0.5, type=float,
                    help="Internal weight for reverse profile-flatness, AE mean versus MD quantile bins.")
parser.add_argument("--closure_profile_bins", default=8, type=int,
                    help="Number of AE quantile bins for profile-flatness closure loss.")
parser.add_argument("--closure_profile_tail_weight", default=2.0, type=float,
                    help="Extra weight applied to high-AE bins in the profile-flatness loss.")
parser.add_argument("--closure_dcorr_max_samples", default=512, type=int,
                    help="Maximum selected-class events per batch used by distance correlation; <=0 uses all.")
parser.add_argument("--closure_tail_abcd_weight", default=1.0, type=float,
                    help="Internal weight for soft high-quantile ABCD closure loss.")
parser.add_argument("--closure_tail_quantiles", default="0.50,0.65,0.80,0.90", type=str,
                    help="Comma-separated QCD quantiles used by soft tail ABCD training loss.")
parser.add_argument("--closure_tail_scale", default=12.0, type=float,
                    help="Sigmoid sharpness for soft high-quantile ABCD counts.")
parser.add_argument("--closure_tail_min_events", default=5, type=int,
                    help="Minimum hard selected-class events per ABCD region before a batch tail cut contributes.")
parser.add_argument("--closure_tail_focus_weight", default=2.0, type=float,
                    help="Extra loss weight applied to higher quantile tail ABCD cuts.")
parser.add_argument("--closure_ckpt_loss_tol", default=1.10, type=float,
                    help="Legacy compatibility option; ABCD checkpointing uses --abcd_ckpt_loss_tol.")
parser.add_argument("--abcd_ckpt_loss_tol", default=1.10, type=float,
                    help="Save checkpoint_abcd only when val loss is within this factor of best val loss.")
parser.add_argument("--val_abcd_quantiles",
                    default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.92",
                    type=str,
                    help="Comma-separated per-class quantiles used by validation ABCD checkpoint score.")
parser.add_argument("--val_abcd_min_events", default=20, type=int,
                    help="Minimum hard events per class and validation ABCD region.")
parser.add_argument("--val_abcd_min_region_frac", default=0.005, type=float,
                    help="Minimum fraction of validation QCD required in every ABCD region.")
parser.add_argument("--val_abcd_tail_min_quantile", default=0.80, type=float,
                    help="Quantiles at or above this value are treated as tail cuts in the validation score.")
parser.add_argument("--qcd_label",            default=1,     type=int,
                    help="Label index for QCD compatibility modes.")
parser.add_argument("--md_proxy_type", default="ema", choices=["batch", "ema"],
                    help="Batch-local or EMA class-whitening proxy for training closure loss.")
parser.add_argument("--md_ema_momentum", default=0.05, type=float,
                    help="EMA update fraction for class-whitening statistics.")
parser.add_argument("--md_ema_eps", default=1e-5, type=float,
                    help="Diagonal regularization for EMA covariance whitening.")
parser.add_argument("--no_mi_norm",           action="store_true",
                    help="Skip per-batch normalization of MI penalty (divide by batch mean). "
                         "Without this, lambda is effectively rescaled by ~1/raw_mi, making "
                         "different lambda values produce near-identical gradients.")
parser.add_argument("--offload_critic_graph", action="store_true",
                    help="Compatibility flag from older scripts; the critic now reuses the "
                         "main forward activations, so no extra critic graph is offloaded.")
args, unknown = parser.parse_known_args()
print(f"Unknown args: {unknown}")
args.baseline_labels_values = parse_int_list(args.baseline_labels)
args.closure_tail_quantiles_values = parse_float_list(args.closure_tail_quantiles)
args.val_abcd_quantiles_values = parse_float_list(args.val_abcd_quantiles)
if not args.baseline_labels_values:
    raise ValueError("--baseline_labels must contain at least one integer label.")
if (
    args.closure_scope != "qcd"
    or args.closure_score_mode != "own_class"
    or args.critic_scope != "qcd"
    or args.nuisance_bin_scope != "qcd"
):
    raise ValueError(
        "The experimental QCD-closure campaign requires "
        "--closure_scope qcd --closure_score_mode own_class "
        "--critic_scope qcd --nuisance_bin_scope qcd. "
        "All-background classification and SupCon remain enabled."
    )
if args.critic_schedule == "per_batch":
    raise ValueError(
        "--critic_schedule per_batch is disabled in this HLT implementation. "
        "Use --critic_schedule warmup for the current memory-safe interleaved critic, "
        "or --critic_schedule per_epoch for the slower epoch-level critic."
    )

if not args.local_testing:
    import wandb
    wandb_timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))
    wandb.init(name=args.exp_name,
               project="nurd-ood-" + args.project_name, reinit=True,
               settings=wandb.Settings(init_timeout=wandb_timeout))
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


def cosine_schedule(epoch, start, end, ramp_epochs):
    if start is None or ramp_epochs <= 0:
        return end
    progress = min(1.0, max(0.0, epoch / max(ramp_epochs, 1)))
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


def get_effective_contrast_weight(epoch):
    return cosine_schedule(
        epoch, args.contrast_weight_start, args.contrast_weight, args.contrast_ramp_epochs)


def get_effective_closure_weight(epoch):
    return cosine_schedule(
        epoch, args.closure_weight_start, args.closure_weight, args.closure_ramp_epochs)


def label_membership_mask(labels, label_values):
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label in label_values:
        mask |= labels.long() == int(label)
    return mask


def critic_scope_mask(labels, joint_indep_args):
    """Return a boolean mask selecting events used by the nuisance critic."""
    scope = joint_indep_args.get("critic_scope", "all")
    if scope == "qcd":
        mask = labels.long() == int(joint_indep_args["qcd_label"])
    elif scope in {"baselines", "all_baselines"}:
        mask = label_membership_mask(labels, joint_indep_args["baseline_labels"])
    else:
        mask = torch.ones_like(labels, dtype=torch.bool)
    return mask


def select_critic_scope(inputs, labels, nuisances, joint_indep_args):
    """Return the batch subset used by the nuisance critic."""
    mask = critic_scope_mask(labels, joint_indep_args)
    return inputs[mask], labels[mask], nuisances[mask], mask


def _safe_pearson_torch(x, y):
    corr = _pearson_corr_tensor(x, y)
    return None if corr is None else corr.item()


def _pearson_corr_tensor(x, y):
    x = x.float().view(-1)
    y = y.float().view(-1)
    if x.numel() < 3:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).mean() * (y * y).mean()).clamp(min=1e-12)
    return (x * y).mean() / denom


def closure_corr_loss(ae_reco, proxy_md, eps=1e-8):
    """Differentiable QCD correlation penalty matching the eval axes more closely."""
    corr = _pearson_corr_tensor(
        torch.log(ae_reco.float().clamp(min=eps)),
        torch.log1p(proxy_md.float().clamp(min=0.0)),
    )
    if corr is None:
        zero = proxy_md.sum() * 0.0
        return zero, None
    return corr * corr, corr.item()


def _critic_accuracy(outputs, targets):
    return (outputs.argmax(dim=1) == targets.long()).float().mean().item()


def _critic_label_input(labels, joint_indep_args):
    y_in = (torch.zeros_like(labels.unsqueeze(1)).float().to(device)
            if joint_indep_args["marginal_indep"]
            else labels.unsqueeze(1).float().to(device))
    return y_in


def _nuisance_marginals(nuisances, joint_indep_args):
    return torch.tensor(
        [joint_indep_args["nuisance_prior"].get(int(z.item()), 1e-8) for z in nuisances],
        device=nuisances.device,
        dtype=torch.float32,
    )


def shuffle_nuisance_bins(z, labels, joint_indep_args):
    if joint_indep_args.get("critic_shuffle", "within_label") == "global":
        return z[torch.randperm(z.size(0), device=z.device)]

    shuffled = z.clone()
    n_shuffled = 0
    for label in labels.long().unique():
        idx = torch.nonzero(labels.long() == label, as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue
        shuffled[idx] = z[idx[torch.randperm(idx.numel(), device=z.device)]]
        n_shuffled += idx.numel()
    if n_shuffled < 2 and z.size(0) > 1:
        return z[torch.randperm(z.size(0), device=z.device)]
    return shuffled


def compute_critic_loss_from_activations(activations, labels, nuisances,
                                         critic_model, critic_criterion,
                                         joint_indep_args):
    y_in = _critic_label_input(labels, joint_indep_args)
    if joint_indep_args.get("critic_type") == "density_ratio":
        # NURD density-ratio trick: classify real (r,y,z) vs shuffled-z triples.
        z = nuisances.long()
        pos_targets = torch.ones_like(labels, dtype=torch.long)
        neg_targets = torch.zeros_like(labels, dtype=torch.long)
        pos_out = critic_model(activations, y_in, z)
        shuffled_z = shuffle_nuisance_bins(z, labels, joint_indep_args)
        neg_out = critic_model(activations, y_in, shuffled_z)
        pos_losses = critic_criterion(pos_out, pos_targets)
        neg_losses = critic_criterion(neg_out, neg_targets)
        outputs = torch.cat([pos_out, neg_out], dim=0)
        targets = torch.cat([pos_targets, neg_targets], dim=0)
        losses = torch.cat([pos_losses, neg_losses], dim=0)
        pos_log_ratio = pos_out[:, 1] - pos_out[:, 0]
        neg_log_ratio = neg_out[:, 1] - neg_out[:, 0]
        penalty_type = joint_indep_args.get("critic_penalty_type", "confusion")
        if penalty_type == "ratio_to_one":
            # The optimal density ratio for independence is one. Uniform-target
            # CE is non-negative after subtracting log(2), shift-invariant, and
            # avoids the unbounded encoder objective of raw logit minimization.
            pos_log_prob = F.log_softmax(pos_out, dim=1)
            neg_log_prob = F.log_softmax(neg_out, dim=1)
            pos_uniform_ce = -0.5 * pos_log_prob.sum(dim=1) - math.log(2.0)
            neg_uniform_ce = -0.5 * neg_log_prob.sum(dim=1) - math.log(2.0)
            penalty = 0.5 * (pos_uniform_ce + neg_uniform_ce)
        elif penalty_type == "logit_ratio":
            # Previous HLT behavior: minimize the real-sample log-density ratio.
            penalty = pos_log_ratio
        elif penalty_type == "ce_gap":
            # Older NURD script behavior: CE gap between shuffled and real pairs.
            penalty = neg_losses - pos_losses
        elif penalty_type == "confusion":
            # Bounded-below adversarial objective: make real/shuffled logits equal.
            penalty = 0.5 * (pos_log_ratio.pow(2) + neg_log_ratio.pow(2))
        else:
            raise ValueError(f"Unsupported critic_penalty_type={penalty_type!r}")
        return outputs, targets, losses, penalty

    # Diagnostic bin-prediction critic: predicts nuisance bin from (latent, y).
    outputs = critic_model(activations, y_in)
    targets = nuisances.long()
    losses = critic_criterion(outputs, targets)
    nuisance_marginals = _nuisance_marginals(nuisances, joint_indep_args)
    losses = torch.div(losses, nuisance_marginals + 1e-8)
    return outputs, targets, losses, losses


def _apply_critic_weights(losses, weights, joint_indep_args):
    if joint_indep_args.get("critic_type") == "density_ratio":
        weights = weights.repeat(2)
    return (losses * weights).sum() / weights.sum().clamp(min=1e-8)


def train_critic(critic_model, model, train_loader, critic_criterion, critic_optimizer,
                 epoch, log, reweight_args, joint_indep_args):
    critic_model.train()
    model.eval()
    batch_time = AverageMeter()
    rw_loss = AverageMeter()
    rw_acc = AverageMeter()
    end = time.time()
    for inputs, targets, nuisances, _ae_reco, exact_weights in train_loader:
        inputs, targets, nuisances = inputs.to(device), targets.long().to(device), nuisances.to(device)
        exact_weights = exact_weights.to(device)
        inputs_c, targets_c, nuisances_c, scope_mask = select_critic_scope(
            inputs, targets, nuisances, joint_indep_args)
        if inputs_c.size(0) == 0:
            continue
        with torch.no_grad():
            activations_c, _ = model(inputs_c)
        outputs, tgts, losses, _mi_proxy = compute_critic_loss_from_activations(
            activations_c, targets_c, nuisances_c, critic_model,
            critic_criterion, joint_indep_args)
        exact_weights = exact_weights[scope_mask]
        weights = exact_weights if reweight_args["reweight"] else torch.ones_like(exact_weights)
        tensor_loss = _apply_critic_weights(losses, weights, joint_indep_args)
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
        for inputs, targets, nuisances, _ae_reco, exact_weights in val_loader:
            inputs, targets, nuisances = inputs.to(device), targets.long().to(device), nuisances.to(device)
            exact_weights = exact_weights.to(device)
            inputs_c, targets_c, nuisances_c, scope_mask = select_critic_scope(
                inputs, targets, nuisances, joint_indep_args)
            if inputs_c.size(0) == 0:
                continue
            activations_c, _ = model(inputs_c)
            outputs, tgts, losses, _mi_proxy = compute_critic_loss_from_activations(
                activations_c, targets_c, nuisances_c, critic_model,
                critic_criterion, joint_indep_args)
            exact_weights = exact_weights[scope_mask]
            weights = exact_weights if reweight_args["reweight"] else torch.ones_like(exact_weights)
            weighted_loss = _apply_critic_weights(losses, weights, joint_indep_args)
            loss_m.update(weighted_loss.item(), inputs_c.size(0))
            acc_m.update(_critic_accuracy(outputs, tgts), inputs_c.size(0))
            if joint_indep_args.get("critic_type") == "bin_pred":
                num_correct = outputs.argmax(dim=1) == tgts.long()
                rw_acc_m.update((num_correct.float() * weights).sum().item() / weights.sum().item(),
                                inputs_c.size(0))
            else:
                num_correct = outputs.argmax(dim=1) == tgts.long()
                rw_weights = weights.repeat(2)
                rw_acc_m.update((num_correct.float() * rw_weights).sum().item() / rw_weights.sum().item(),
                                inputs_c.size(0))
    return loss_m.avg, acc_m.avg, rw_acc_m.avg


#training functions

#get contrastive loss
contrastive_loss_fn = SupConLoss(temperature=args.contrast_temp)

def train(model, train_loader, val_loader, criterion, optimizer, epoch, log,
          reweight_args, joint_indep_args, effective_lambda=None,
          effective_contrast_weight=None, effective_closure_weight=None,
          md_proxy=None):
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
    closure_dcorr_m = AverageMeter()
    closure_profile_m = AverageMeter()
    closure_profile_reverse_m = AverageMeter()
    closure_tail_abcd_m = AverageMeter()
    mi_m  = AverageMeter()   # MI / independence penalty (normalized, always ~1)
    raw_mi_m  = AverageMeter()   # raw critic CE before normalization
    critic_ce_m = AverageMeter()  # ordinary nuisance-bin CE
    critic_ce_ratio_m = AverageMeter()  # ordinary CE divided by log(n_bins)
    critic_norm_ratio_m = AverageMeter()  # prior-normalized CE divided by its chance baseline
    critic_acc_m = AverageMeter()  # nuisance-bin accuracy of the current critic
    critic_qcd_acc_m = AverageMeter()  # same, restricted to QCD when available
    critic_scope_frac_m = AverageMeter()
    closure_proxy_corr_m = AverageMeter()
    closure_classes_m = AverageMeter()
    weight_cv_m = AverageMeter()  # coeff. of variation of NURD weights (std/mean); 0 = uniform, >1 = heavy tails
    weight_ess_m = AverageMeter() # effective sample size fraction: ESS/N; 1.0 = no reweighting cost

    model.train()
    end = time.time()
    for inputs, targets, nuisances, ae_reco, exact_weights in train_loader:
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        nuisances = nuisances.to(device)
        ae_reco = ae_reco.to(device)
        exact_weights = exact_weights.to(device)

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

        # Forward pass. The same activations are reused for the critic update
        # below, avoiding a second Transformer pass on every selected batch.
        activations, outputs = model(inputs)
        losses_ce = criterion(outputs, targets)         # [B] CE loss

        if joint_indep_args["joint_indep"] and joint_indep_args.get("critic_schedule") == "warmup":
            if random.random() < joint_indep_args["critic_train_frac"]:
                joint_indep_args["critic_model"] = unfreeze_model(joint_indep_args["critic_model"])
                joint_indep_args["critic_model"].train()
                scope_mask = critic_scope_mask(targets, joint_indep_args)
                targets_c = targets[scope_mask]
                nuisances_c = nuisances[scope_mask]
                if targets_c.size(0) > 1:
                    act_detached = activations.detach()[scope_mask]
                    weights_c = exact_weights[scope_mask]
                    if not reweight_args["reweight"]:
                        weights_c = torch.ones_like(weights_c)
                    n_steps = joint_indep_args.get("n_critic_steps_per_batch", 1)
                    for _ in range(n_steps):
                        c_out, c_targets, c_losses, _mi_proxy = compute_critic_loss_from_activations(
                            act_detached, targets_c, nuisances_c,
                            joint_indep_args["critic_model"],
                            joint_indep_args["critic_criterion"],
                            joint_indep_args)
                        c_loss = _apply_critic_weights(c_losses, weights_c, joint_indep_args)
                        joint_indep_args["critic_optimizer"].zero_grad()
                        c_loss.backward()
                        joint_indep_args["critic_optimizer"].step()
                joint_indep_args["critic_model"] = freeze_model(joint_indep_args["critic_model"])
                joint_indep_args["critic_model"].eval()

        acc, loss, top1 = record_metrics(acc, loss, top1, inputs, outputs, targets, losses_ce)

        # ── NURD joint independence penalty ───────────────────────────────────
        info_loss_val = 0.0 #normalized MI penalty
        raw_mi_val = 0.0 #raw MI penalty
        critic_count = inputs.size(0)
     
        if joint_indep_args["joint_indep"]:
            #use ramped lambda during warmup, else fix lambda
            lam = effective_lambda if effective_lambda is not None else joint_indep_args["lambda"]
            scope_mask = critic_scope_mask(targets, joint_indep_args)
            targets_c = targets[scope_mask]
            nuisances_c = nuisances[scope_mask]
            critic_scope_frac_m.update(float(targets_c.size(0)) / max(inputs.size(0), 1), inputs.size(0))
            critic_count = max(targets_c.size(0), 1)
            
            if targets_c.size(0) > 1:
                act_c = activations[scope_mask]
                with (torch.no_grad() if lam == 0.0 else torch.enable_grad()):
                    critic_outputs, critic_targets, info_losses, mi_proxy = (
                        compute_critic_loss_from_activations(
                            act_c if lam > 0.0 else act_c.detach(),
                            targets_c, nuisances_c,
                            joint_indep_args["critic_model"],
                            joint_indep_args["critic_criterion"],
                            joint_indep_args))

                if joint_indep_args.get("critic_type") == "density_ratio":
                    penalty = mi_proxy
                    raw_mi_val = penalty.mean().item()
                    critic_ce = info_losses.mean().item()
                    critic_ce_m.update(critic_ce, targets_c.size(0))
                    critic_ce_ratio_m.update(
                        critic_ce / joint_indep_args["critic_chance_ce"],
                        targets_c.size(0))
                    critic_acc_m.update(_critic_accuracy(critic_outputs, critic_targets), targets_c.size(0))
                    qcd_metric_mask = targets_c.long() == args.qcd_label
                    if qcd_metric_mask.any():
                        # Real-vs-shuffled accuracy is paired, so duplicate the QCD mask.
                        qcd_pair_mask = torch.cat([qcd_metric_mask, qcd_metric_mask], dim=0)
                        critic_qcd_acc_m.update(
                            _critic_accuracy(critic_outputs[qcd_pair_mask], critic_targets[qcd_pair_mask]),
                            int(qcd_metric_mask.sum().item()))
                    if lam > 0.0:
                        losses_ce = losses_ce.clone()
                        losses_ce[scope_mask] = losses_ce[scope_mask] + lam * penalty
                    info_loss_val = raw_mi_val
                #bin pred critic
                else:
                    raw_mi_val = info_losses.mean().item()
                    critic_ce = F.cross_entropy(
                        critic_outputs, nuisances_c.long(), reduction="none")
                    critic_ce_m.update(critic_ce.mean().item(), targets_c.size(0))
                    critic_ce_ratio_m.update(
                        critic_ce.mean().item() / joint_indep_args["critic_chance_ce"],
                        targets_c.size(0))
                    critic_norm_ratio_m.update(
                        raw_mi_val / joint_indep_args["critic_chance_norm_ce"],
                        targets_c.size(0))
                    critic_acc_m.update(_critic_accuracy(critic_outputs, nuisances_c), targets_c.size(0))
                    qcd_metric_mask = targets_c.long() == args.qcd_label
                    if qcd_metric_mask.any():
                        critic_qcd_acc_m.update(
                            _critic_accuracy(critic_outputs[qcd_metric_mask], nuisances_c[qcd_metric_mask]),
                            int(qcd_metric_mask.sum().item()))
                    if lam > 0.0:
                        penalty_losses = info_losses
                        if not args.no_mi_norm:
                            penalty_losses = penalty_losses / (penalty_losses.detach().mean() + 1e-8)
                        # subtract because high critic loss = encoder already independent = good
                        # gradient rewards encoder for confusing the critic
                        losses_ce = losses_ce.clone()
                        losses_ce[scope_mask] = losses_ce[scope_mask] - lam * penalty_losses
                        info_loss_val = penalty_losses.mean().item()

        #nurd reweighting
        weights = exact_weights.to(device) if reweight_args["reweight"] else torch.ones_like(exact_weights).to(device)
        rw_acc, rw_loss = record_rw_metrics(rw_acc, rw_loss, inputs, outputs, targets, losses_ce, weights)
        loss_nurd = (losses_ce * weights).sum() / weights.sum()

        #contrastive loss
        embeddings  = model.get_embeddings(activations)
        loss_con = contrastive_loss_fn(
            embeddings, targets, weights if reweight_args["reweight"] else None)
        contrast_w = args.contrast_weight if effective_contrast_weight is None else effective_contrast_weight
        tensor_loss = (1 - contrast_w) * loss_nurd + contrast_w * loss_con

        # The model still learns every background through CE and SupCon. Closure
        # is targeted specifically on the QCD population used by the primary
        # ABCD evaluation.
        loss_closure = torch.tensor(0.0, device=device)
        closure_w = args.closure_weight if effective_closure_weight is None else effective_closure_weight
        closure_diag = {}
        if closure_w > 0.0 and args.closure_loss_type != "none":
            loss_closure, closure_diag, n_closure = compute_qcd_closure_loss(
                activations, targets, ae_reco, args, md_proxy=md_proxy,
                update=True)
            if n_closure > 0:
                closure_classes_m.update(1, inputs.size(0))
                if closure_diag.get("corr") is not None:
                    closure_proxy_corr_m.update(closure_diag["corr"], n_closure)
                if closure_diag.get("dcorr") is not None:
                    closure_dcorr_m.update(closure_diag["dcorr"], n_closure)
                if closure_diag.get("profile") is not None:
                    closure_profile_m.update(closure_diag["profile"], n_closure)
                if closure_diag.get("profile_reverse") is not None:
                    closure_profile_reverse_m.update(
                        closure_diag["profile_reverse"], n_closure)
                if closure_diag.get("tail_abcd") is not None:
                    closure_tail_abcd_m.update(closure_diag["tail_abcd"], n_closure)
                tensor_loss = tensor_loss + closure_w * loss_closure
        elif md_proxy is not None:
            # Warm the lagged reference while the closure coefficient ramps up.
            qcd_mask = targets.long() == int(args.qcd_label)
            if qcd_mask.sum() >= args.closure_class_min_events:
                md_proxy.update(activations[qcd_mask])

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
        mi_m.update(info_loss_val,              critic_count)
        raw_mi_m.update(raw_mi_val,             critic_count)
        weight_cv_m.update((w_std / (w_mean + 1e-8)).item(), bs)
        weight_ess_m.update(ess,                bs)

    #logging
    log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss, rw_acc, split="Train")
    current_lr = optimizer.param_groups[0]["lr"]
    log.debug(f"  total={total_m.avg:.5f}  nurd={nurd_m.avg:.5f}  "
              f"con={con_m.avg:.5f}  closure={closure_m.avg:.5f}  "
              f"tail_abcd={closure_tail_abcd_m.avg:.5f}  "
              f"mi={mi_m.avg:.5f}  raw_mi={raw_mi_m.avg:.5f}  "
              f"crit_acc={critic_acc_m.avg:.3f}  qcd_crit_acc={critic_qcd_acc_m.avg:.3f}  "
              f"closure_proxy_r={closure_proxy_corr_m.avg:.3f}  "
              f"closure_classes={closure_classes_m.avg:.1f}  "
              f"w_cv={weight_cv_m.avg:.3f}  w_ess={weight_ess_m.avg:.3f}  "
              f"cw={contrast_w:.3f}  clw={closure_w:.3f}  lr={current_lr:.2e}")
    if not args.local_testing:
        wandb.log({
            "Train/total_loss":       total_m.avg,
            "Train/nurd_weighted_ce": nurd_m.avg,
            "Train/contrastive":      con_m.avg,
            "Train/closure":          closure_m.avg,
            "Train/closure_dcorr":    closure_dcorr_m.avg,
            "Train/closure_profile":  closure_profile_m.avg,
            "Train/closure_profile_reverse": closure_profile_reverse_m.avg,
            "Train/closure_tail_abcd": closure_tail_abcd_m.avg,
            "Train/mi_penalty":       mi_m.avg,
            "Train/raw_mi_penalty":   raw_mi_m.avg,
            "Train/critic_ce":        critic_ce_m.avg,
            "Train/critic_ce_over_chance": critic_ce_ratio_m.avg,
            "Train/critic_norm_ce_over_chance": critic_norm_ratio_m.avg,
            "Train/critic_acc":       critic_acc_m.avg,
            "Train/critic_qcd_acc":   critic_qcd_acc_m.avg,
            "Train/critic_scope_frac": critic_scope_frac_m.avg,
            "Train/closure_proxy_pearson": closure_proxy_corr_m.avg,
            "Train/closure_classes":   closure_classes_m.avg,
            "Train/nurd_weight_cv":   weight_cv_m.avg,
            "Train/nurd_weight_ess":  weight_ess_m.avg,
            "Train/rw_loss":          rw_loss.avg,
            "Train/rw_acc":           rw_acc.avg,
            "Train/acc":              acc.avg,
            "Train/prec1":            top1.avg,
            "LR":                     current_lr,
            "Train/effective_lambda": effective_lambda if effective_lambda is not None else args._lambda,
            "Train/effective_contrast_weight": contrast_w,
            "Train/effective_closure_weight": closure_w,
        }, step=epoch)


def validate(val_loader, model, criterion, epoch, log, reweight_args,
             joint_indep_args=None, md_proxy=None):
    batch_time = AverageMeter()
    acc = AverageMeter(); loss = AverageMeter(); top1 = AverageMeter()
    rw_acc = AverageMeter(); rw_loss = AverageMeter()
    qcd_proxy_corr_m = AverageMeter()
    critic_acc_m = AverageMeter()
    critic_ce_m = AverageMeter()
    qcd_x_chunks = []
    qcd_y_chunks = []

    model.eval()
    with torch.no_grad():
        end = time.time()
        for inputs, targets, nuisances, ae_reco, exact_weights in val_loader:
            inputs    = inputs.to(device)
            targets   = targets.long().to(device)
            ae_reco   = ae_reco.to(device)
            exact_weights = exact_weights.to(device)
            activations, outputs = model(inputs)
            losses    = criterion(outputs, targets)

            acc, loss, top1 = record_metrics(acc, loss, top1, inputs, outputs, targets, losses)
            if reweight_args["reweight"]:
                rw_acc, rw_loss = record_rw_metrics(
                    rw_acc, rw_loss, inputs, outputs, targets, losses,
                    exact_weights)
            qcd_mask = targets.long() == int(args.qcd_label)
            if qcd_mask.sum() >= args.closure_class_min_events:
                if joint_indep_args and joint_indep_args["joint_indep"]:
                    critic_outputs, critic_targets, critic_losses, _ = (
                        compute_critic_loss_from_activations(
                            activations[qcd_mask], targets[qcd_mask],
                            nuisances.to(device)[qcd_mask],
                            joint_indep_args["critic_model"],
                            joint_indep_args["critic_criterion"],
                            joint_indep_args))
                    critic_acc_m.update(
                        _critic_accuracy(critic_outputs, critic_targets),
                        int(qcd_mask.sum().item()))
                    critic_ce_m.update(
                        critic_losses.mean().item(),
                        int(qcd_mask.sum().item()))
                proxy_md = compute_proxy_md(
                    activations, qcd_mask, md_proxy=md_proxy,
                    proxy_type=args.md_proxy_type, update=False)
                x_log = torch.log(ae_reco[qcd_mask].float().clamp(min=1e-8))
                y_log = torch.log1p(proxy_md[qcd_mask].float().clamp(min=0.0))
                corr_val = _safe_pearson_torch(x_log, y_log)
                if corr_val is not None:
                    qcd_proxy_corr_m.update(
                        corr_val, int(qcd_mask.sum().item()))
                qcd_x_chunks.append(x_log.cpu().numpy())
                qcd_y_chunks.append(y_log.cpu().numpy())
            batch_time.update(time.time() - end); end = time.time()

    val_qcd_corr = qcd_proxy_corr_m.avg if qcd_proxy_corr_m.count > 0 else float("nan")
    val_abcd = {"score": float("nan"), "n_points": 0}
    if qcd_x_chunks:
        val_abcd = abcd_grid_metrics_np(
            np.concatenate(qcd_x_chunks),
            np.concatenate(qcd_y_chunks),
            args.val_abcd_quantiles_values,
            min_count=args.val_abcd_min_events,
            min_count_fraction=args.val_abcd_min_region_frac,
            tail_min_quantile=args.val_abcd_tail_min_quantile)
    val_abcd["selection_mode"] = "qcd_md_grid"
    log_metrics(log, epoch, batch_time, loss, top1, acc, rw_loss, rw_acc, split="Val")
    log.debug(f"  qcd_proxy_r={val_qcd_corr:.3f}  "
              f"proxy_abcd_score={val_abcd.get('score', float('nan')):.5f}  "
              f"proxy_abcd_median={val_abcd.get('median_abs_log_nonclosure', float('nan')):.5f}  "
              f"proxy_abcd_p90={val_abcd.get('p90_abs_log_nonclosure', float('nan')):.5f}  "
              f"proxy_abcd_tail={val_abcd.get('tail_mean_abs_log_nonclosure', float('nan')):.5f}  "
              f"proxy_abcd_min_region={val_abcd.get('min_region_count', 0)}  "
              f"critic_acc={critic_acc_m.avg:.3f}  critic_ce={critic_ce_m.avg:.5f}")
    if not args.local_testing:
        wandb.log({
            "Val/loss":    loss.avg,
            "Val/prec1":   top1.avg,
            "Val/acc":     acc.avg,
            "Val/rw_loss": rw_loss.avg,
            "Val/rw_acc":  rw_acc.avg,
            "Val/qcd_proxy_pearson": val_qcd_corr,
            "Val/proxy_abcd_score": val_abcd.get("score", float("nan")),
            "Val/proxy_abcd_points": val_abcd.get("n_points", 0),
            "Val/proxy_abcd_median_abs_log_nonclosure": val_abcd.get(
                "median_abs_log_nonclosure", float("nan")),
            "Val/proxy_abcd_p90_abs_log_nonclosure": val_abcd.get(
                "p90_abs_log_nonclosure", float("nan")),
            "Val/proxy_abcd_tail_mean_abs_log_nonclosure": val_abcd.get(
                "tail_mean_abs_log_nonclosure", float("nan")),
            "Val/critic_acc": critic_acc_m.avg,
            "Val/critic_ce": critic_ce_m.avg,
        }, step=epoch)
    return_loss = rw_loss.avg if reweight_args["reweight"] else loss.avg
    return return_loss, acc.avg, rw_acc.avg, val_qcd_corr, val_abcd


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
    ae_scaler = ae_ckpt.get("ae_scaler")
    if ae_scaler is not None:
        log.debug("Using AE normalization scaler saved in the AE checkpoint.")
    else:
        log.debug("AE checkpoint has no scaler; recomputing object normalization from --data.")
    train_dataset, val_dataset, obj_scaler = build_hlt_datasets(
        args.data, ae, n_bins=args.n_bins,
        val_split=args.val_split, seed=args.manualSeed,
        max_events=args.max_events,
        ae_scaler=ae_scaler,
        max_weight_ratio=args.max_weight_ratio,
        nuisance_bin_scope=args.nuisance_bin_scope,
        qcd_label=args.qcd_label,
        baseline_labels=args.baseline_labels_values,
    )
    log.debug(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    num_classes = int(train_dataset.labels.max().item()) + 1
    num_tokens  = train_dataset.features.size(1)

    kwargs = {"pin_memory": False, "num_workers": args.num_workers}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              drop_last=False, **kwargs)

    #nuisance prior is marginal probability of each bin (used to normalize critic loss)
    label_prior    = train_dataset.get_label_prior()
    nuisance_prior = None
    if args.joint_indep:
        if args.critic_scope == "qcd":
            prior_label = args.qcd_label
        elif args.critic_scope in {"baselines", "all_baselines"}:
            prior_label = args.baseline_labels_values
        else:
            prior_label = None
        nuisance_prior = train_dataset.get_nuisance_prior(label=prior_label)
        if not nuisance_prior:
            raise RuntimeError(
                f"No events found for critic_scope={args.critic_scope}; cannot build nuisance prior."
            )
        log.debug(f"Critic scope: {args.critic_scope}; nuisance prior bins: {sorted(nuisance_prior)}")

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
    if critic_model is not None:
        critic_model = freeze_model(critic_model)
        critic_model.eval()

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
        "critic_penalty_type": args.critic_penalty_type,
        "critic_scope":     args.critic_scope,
        "critic_shuffle":   args.critic_shuffle,
        "qcd_label":        args.qcd_label,
        "baseline_labels":  args.baseline_labels_values,
        "critic_chance_ce": (math.log(2) if args.critic_type == "density_ratio"
                             else math.log(max(args.n_bins, 2))),
        "critic_chance_norm_ce": max(len(nuisance_prior or {}), 1) * math.log(max(args.n_bins, 2)),
        "critic_train_frac": args.critic_train_frac,
        "critic_optimizer": (torch.optim.Adam(critic_model.parameters(),
                                              lr=args.lr * args.critic_lr_multiplier,
                                              weight_decay=args.weight_decay)
                             if args.joint_indep and args.critic_schedule == "warmup"
                             else None),
        "n_critic_steps_per_batch": args.n_critic_steps_per_batch,
    }

    cudnn.benchmark = True
    md_proxy = RunningQCDMDProxy(
        momentum=args.md_ema_momentum, eps=args.md_ema_eps) if args.md_proxy_type == "ema" else None

    def checkpoint_state(epoch):
        return {
            "epoch": epoch + 1,
            "state_dict_model": model.state_dict(),
            "state_dict_critic": (
                None if critic_model is None else critic_model.state_dict()),
            "ae_scaler": obj_scaler,
            "config": vars(args),
            "nuisance_bin_edges": train_dataset.bin_edges,
            "nurd_weight_table": train_dataset.weights,
            "md_proxy_state": (
                None if md_proxy is None else md_proxy.state_dict()),
        }

    best_loss = None
    best_abcd_score = None
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
        effective_contrast_weight = get_effective_contrast_weight(epoch)
        effective_closure_weight = get_effective_closure_weight(epoch)

        #all the critic logic here (loss computation, weight updates)
        train(model, train_loader, val_loader, criterion, optimizer,
              epoch + args.reweight_epochs, log, reweight_args, joint_indep_args,
              effective_lambda, effective_contrast_weight, effective_closure_weight,
              md_proxy)
        #runs model on validation set with no gradient updates (just forward passes)
        val_loss, val_acc, val_rw_acc, val_qcd_corr, val_abcd = validate(
            val_loader, model, criterion, epoch + args.reweight_epochs, log,
            reweight_args, joint_indep_args=joint_indep_args,
            md_proxy=md_proxy)

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            log.debug("Saving checkpoint")
            save_checkpoint(
                args, checkpoint_state(epoch), epoch + 1, name="main")
            if not args.local_testing:
                wandb.run.summary["best_val_rw_acc"] = val_rw_acc
                wandb.run.summary["best_val_acc"]    = val_acc
                wandb.run.summary["best_val_loss"]   = val_loss

        val_abcd_score = val_abcd.get("score", float("nan"))
        if np.isfinite(val_abcd_score):
            loss_ok = best_loss is None or val_loss <= args.abcd_ckpt_loss_tol * best_loss
            if loss_ok and (best_abcd_score is None or val_abcd_score < best_abcd_score):
                best_abcd_score = val_abcd_score
                state = checkpoint_state(epoch)
                state.update({
                    "selection_metric": "val_proxy_abcd_qcd_md_grid_score",
                    "selection_value": float(val_abcd_score),
                    "selection_val_loss": float(val_loss),
                    "selection_val_qcd_proxy_corr": float(val_qcd_corr),
                    "selection_val_proxy_abcd": val_abcd,
                })
                log.debug("Saving ABCD closure checkpoint")
                save_checkpoint(args, state, epoch + 1, name="abcd")
                save_checkpoint(args, state, epoch + 1, name="closure")
                if not args.local_testing:
                    wandb.run.summary["best_val_proxy_abcd_score"] = val_abcd_score
                    wandb.run.summary["best_val_proxy_abcd_p90"] = val_abcd.get(
                        "p90_abs_log_nonclosure", float("nan"))
                    wandb.run.summary["best_val_proxy_abcd_tail"] = val_abcd.get(
                        "tail_mean_abs_log_nonclosure", float("nan"))

    log.debug(f"Done. Best val loss: {best_loss:.5f}")
    if not args.local_testing:
        wandb.finish()


if __name__ == "__main__":
    main()
