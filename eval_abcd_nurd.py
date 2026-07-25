"""Evaluate AE loss against a frozen, class-calibrated NURD anomaly score.

The training file fits class-conditional latent references, calibrates their
tail probabilities, and supplies a disjoint validation subset for ABCD
threshold selection. The test file is used only for the final closure report.
"""
import os
import gc
import json
import argparse
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic, gaussian_kde, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D

from models.hlt_con import HLTContrastiveModel
from models.hlt_autoencoder import HLTAutoencoder
from utils.hlt_score_calibration import fit_class_references, score_latents

CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}


def parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).replace(",", " ").split() if str(v).strip()]


def label_membership_mask(labels, label_values):
    labels = np.asarray(labels)
    if not label_values:
        return np.ones(labels.shape, dtype=bool)
    return np.isin(labels, np.asarray([int(v) for v in label_values]))


# ── ABCD helpers (identical to eval_abcd.py) ─────────────────────────────────

def abcd_counts(loss_1, loss_2, percent_1, percent_2):
    thresh_1 = np.quantile(loss_1, percent_1)
    thresh_2 = np.quantile(loss_2, percent_2)
    A, B, C, D = abcd_counts_at_thresholds(loss_1, loss_2, thresh_1, thresh_2)
    return thresh_1, thresh_2, A, B, C, D


def abcd_counts_at_thresholds(loss_1, loss_2, thresh_1, thresh_2):
    A = int(((loss_1 > thresh_1) & (loss_2 > thresh_2)).sum())
    B = int(((loss_1 > thresh_1) & (loss_2 <= thresh_2)).sum())
    C = int(((loss_1 <= thresh_1) & (loss_2 > thresh_2)).sum())
    D = int(((loss_1 <= thresh_1) & (loss_2 <= thresh_2)).sum())
    return A, B, C, D


def closure_metrics(A, B, C, D, eps=1e-8):
    A_hat = (B * C) / max(D, eps)
    if A <= 0 or A_hat <= 0:
        return {
            "A_hat": float(A_hat),
            "ratio": np.inf,
            "nonclosure": np.inf,
            "legacy_nonclosure": np.inf,
            "log_nonclosure": np.inf,
        }
    ratio = A_hat / max(A, eps)
    return {
        "A_hat": float(A_hat),
        "ratio": float(ratio),
        "nonclosure": float(ratio - 1.0),
        "legacy_nonclosure": float((A - A_hat) / max(A_hat, eps)),
        "log_nonclosure": float(np.log(max(ratio, eps))),
    }


def nonclosure_A(A, B, C, D, eps=1e-8):
    metrics = closure_metrics(A, B, C, D, eps=eps)
    return metrics["nonclosure"], metrics["A_hat"]


def abcd_record_at_thresholds(loss_1, loss_2, thresh_1, thresh_2):
    A, B, C, D = abcd_counts_at_thresholds(loss_1, loss_2, thresh_1, thresh_2)
    metrics = closure_metrics(A, B, C, D)
    ratio = metrics["ratio"]
    invA = 0.0 if A == 0 else 1.0 / A
    invB = 0.0 if B == 0 else 1.0 / B
    invC = 0.0 if C == 0 else 1.0 / C
    invD = 0.0 if D == 0 else 1.0 / D
    rel_var = invA + invB + invC + invD
    ratio_unc = abs(ratio) * np.sqrt(rel_var) if rel_var > 0 else 0.0
    return {
        "t1": float(thresh_1),
        "t2": float(thresh_2),
        "A": int(A),
        "B": int(B),
        "C": int(C),
        "D": int(D),
        "A_hat": metrics["A_hat"],
        "nonclosure": metrics["nonclosure"],
        "legacy_nonclosure": metrics["legacy_nonclosure"],
        "log_nonclosure": metrics["log_nonclosure"],
        "ratio": float(ratio),
        "ratio_unc": float(ratio_unc),
    }


def _grid_summary(abs_nonclosure):
    values = np.asarray(abs_nonclosure, dtype=np.float64)
    return {
        "n_points": int(values.size),
        "mean_abs_nonclosure": float(np.mean(values)) if values.size else np.nan,
        "median_abs_nonclosure": float(np.median(values)) if values.size else np.nan,
        "p90_abs_nonclosure": float(np.quantile(values, 0.90)) if values.size else np.nan,
    }


def scan_abcd_grid(loss_1, loss_2, percent, min_A=50, min_D=500,
                   min_A_frac=0.0, selection_stat_weight=0.0):
    best = {"selection_score": np.inf, "log_nonclosure": np.inf, "nonclosure": np.inf}
    scan_abs_nonclosure = []
    min_A_effective = max(int(min_A), int(np.ceil(float(min_A_frac) * len(loss_1))))
    for p1 in percent:
        for p2 in percent:
            t1, t2, A, B, C, D = abcd_counts(loss_1, loss_2, p1, p2)
            if A < min_A_effective or D < min_D:
                continue
            metrics = closure_metrics(A, B, C, D)
            nc = metrics["nonclosure"]
            score = abs(metrics["log_nonclosure"])
            record = abcd_record_at_thresholds(loss_1, loss_2, t1, t2)
            selection_score = score + float(selection_stat_weight) * record["ratio_unc"]
            if np.isfinite(nc):
                scan_abs_nonclosure.append(abs(nc))
            if np.isfinite(selection_score) and selection_score < best["selection_score"]:
                best.update(dict(p1=float(p1), p2=float(p2), t1=float(t1), t2=float(t2),
                                 A=int(A), B=int(B), C=int(C), D=int(D),
                                 A_hat=metrics["A_hat"],
                                 nonclosure=float(nc),
                                 legacy_nonclosure=metrics["legacy_nonclosure"],
                                 log_nonclosure=metrics["log_nonclosure"],
                                 ratio=metrics["ratio"],
                                 ratio_unc=record["ratio_unc"],
                                 selection_score=float(selection_score),
                                 min_A_effective=int(min_A_effective)))
    return best, _grid_summary(scan_abs_nonclosure)


def split_for_threshold_report(n_events, holdout_frac=0.5, seed=42,
                               axis1=None, axis2=None, n_strata=8,
                               strata_labels=None):
    idx = np.arange(n_events)
    if holdout_frac <= 0.0 or holdout_frac >= 1.0 or n_events < 4:
        return idx, idx, "same_sample"
    rng = np.random.default_rng(seed)
    if axis1 is None or axis2 is None:
        perm = rng.permutation(idx)
        n_report = int(round(holdout_frac * n_events))
        n_report = min(max(n_report, 1), n_events - 1)
        report_idx = perm[:n_report]
        tune_idx = perm[n_report:]
        return tune_idx, report_idx, "holdout_random"

    axis1 = np.asarray(axis1)
    axis2 = np.asarray(axis2)
    valid = np.isfinite(axis1) & np.isfinite(axis2)
    if valid.sum() != n_events:
        return split_for_threshold_report(n_events, holdout_frac, seed)

    q1 = np.quantile(axis1, np.linspace(0.0, 1.0, n_strata + 1))
    q2 = np.quantile(axis2, np.linspace(0.0, 1.0, n_strata + 1))
    b1 = np.searchsorted(q1[1:-1], axis1, side="right")
    b2 = np.searchsorted(q2[1:-1], axis2, side="right")
    strata = b1 * n_strata + b2
    if strata_labels is not None:
        strata_labels = np.asarray(strata_labels)
        if strata_labels.shape[0] == n_events:
            _, label_codes = np.unique(strata_labels, return_inverse=True)
            strata = label_codes * (n_strata * n_strata) + strata

    tune_parts, report_parts = [], []
    for s in np.unique(strata):
        members = idx[strata == s]
        members = rng.permutation(members)
        if members.size < 2:
            tune_parts.append(members)
            continue
        n_report = int(round(holdout_frac * members.size))
        n_report = min(max(n_report, 1), members.size - 1)
        report_parts.append(members[:n_report])
        tune_parts.append(members[n_report:])

    tune_idx = np.concatenate(tune_parts) if tune_parts else np.array([], dtype=int)
    report_idx = np.concatenate(report_parts) if report_parts else np.array([], dtype=int)
    if tune_idx.size == 0 or report_idx.size == 0:
        return split_for_threshold_report(n_events, holdout_frac, seed)
    return rng.permutation(tune_idx), rng.permutation(report_idx), "holdout_stratified"


def profile_plot(ax, x, y, nbins=30, logx=False, min_per_bin=20, label="mean ± SE"):
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if logx:
        m &= (x > 0)
    x, y = x[m], y[m]
    xu = np.log10(x) if logx else x
    lo, hi = float(xu.min()), float(xu.max())
    if lo == hi:
        hi = np.nextafter(hi, np.inf)
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean, _, _ = binned_statistic(xu, y, statistic="mean", bins=edges)
    std,  _, _ = binned_statistic(xu, y, statistic="std",  bins=edges)
    cnt,  _, _ = binned_statistic(xu, y, statistic="count",bins=edges)
    sem = std / np.sqrt(np.maximum(cnt, 1))
    good = cnt >= min_per_bin
    xc = centers[good]
    xplot = (10.0 ** xc) if logx else xc
    if logx:
        ax.set_xscale("log")
    ax.errorbar(xplot, mean[good], yerr=sem[good],
                fmt="o", ms=3, lw=1, capsize=2, label=label)
    ax.grid(alpha=0.3)
    return {"x": xplot, "mean": mean[good], "sem": sem[good], "count": cnt[good]}


def _sample_pair(x, y, max_points, seed=42):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if max_points and max_points > 0 and x.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], max_points, replace=False)
        x, y = x[idx], y[idx]
    return x.astype(np.float64), y.astype(np.float64)


def _safe_correlations(x, y, max_points=50_000, dcor_points=2_000, seed=42):
    x_s, y_s = _sample_pair(x, y, max_points, seed=seed)
    out = {"n": int(x_s.shape[0]), "pearson": np.nan, "spearman": np.nan,
           "distance_corr": np.nan}
    if x_s.shape[0] < 3 or np.std(x_s) == 0 or np.std(y_s) == 0:
        return out
    out["pearson"] = float(pearsonr(x_s, y_s).statistic)
    out["spearman"] = float(spearmanr(x_s, y_s).statistic)
    if dcor_points and dcor_points > 0:
        x_d, y_d = _sample_pair(x, y, dcor_points, seed=seed + 1)
        if x_d.shape[0] >= 3 and np.std(x_d) > 0 and np.std(y_d) > 0:
            ax = np.abs(x_d[:, None] - x_d[None, :])
            ay = np.abs(y_d[:, None] - y_d[None, :])
            ax = ax - ax.mean(axis=0, keepdims=True) - ax.mean(axis=1, keepdims=True) + ax.mean()
            ay = ay - ay.mean(axis=0, keepdims=True) - ay.mean(axis=1, keepdims=True) + ay.mean()
            dcov2 = np.mean(ax * ay)
            dvarx = np.mean(ax * ax)
            dvary = np.mean(ay * ay)
            denom = np.sqrt(max(dvarx * dvary, 0.0))
            if denom > 0:
                out["distance_corr"] = float(np.sqrt(max(dcov2, 0.0) / denom))
    return out


def per_class_correlations(axis1, axis2, labels, class_labels,
                           corr_sample_size=50_000, dcor_sample_size=2_000):
    out = {}
    for cls in class_labels:
        mask = labels == int(cls)
        out[str(int(cls))] = _safe_correlations(
            axis1[mask], axis2[mask],
            max_points=corr_sample_size,
            dcor_points=dcor_sample_size,
            seed=42 + int(cls))
    return out


def per_class_abcd_at_thresholds(axis1, axis2, labels, class_labels, t1, t2):
    out = {}
    for cls in class_labels:
        mask = labels == int(cls)
        if mask.sum() == 0:
            continue
        out[str(int(cls))] = abcd_record_at_thresholds(axis1[mask], axis2[mask], t1, t2)
    return out


# ── Model loading ─────────────────────────────────────────────────────────────

def load_nurd_model(ckpt_path, device):
    """Load HLTContrastiveModel from NURD main checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]

    # num_tokens: infer from state dict (linear attn projection shape)
    sd = ckpt["state_dict_model"]
    e_proj_key = next((k for k in sd if k.endswith(".attn.e.weight")), None)
    if e_proj_key is not None:
        num_tokens = sd[e_proj_key].shape[1] - 1   # -1 for CLS token
    else:
        num_tokens = cfg.get("linear_dim", 100)     # fallback

    # num_classes: infer from classifier weight
    num_classes = sd["classifier.weight"].shape[0]

    model = HLTContrastiveModel(
        num_classes=num_classes,
        embed_size=cfg["embed_size"],
        latent_dim=cfg["latent_dim"],
        proj_dim=cfg["proj_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        linear_dim=cfg["linear_dim"],
        num_tokens=num_tokens,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded HLTContrastiveModel: latent_dim={cfg['latent_dim']}, "
          f"num_layers={cfg['num_layers']}, num_tokens={num_tokens}", flush=True)
    return model, ckpt


def load_ae(ae_ckpt_path, ae_scaler, device):
    """Load HLTAutoencoder from its own checkpoint."""
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_cfg  = ae_ckpt.get("ae_config", {
        "features": None, "latent_dim": 16,
        "encoder_config": {"nodes": [512, 256]},
        "decoder_config": {"nodes": [256, 512, None]},
        "alpha": 1.0,
    })
    if ae_cfg["features"] is None:
        first_w = ae_ckpt["ae"][next(iter(ae_ckpt["ae"]))]
        ae_cfg["features"] = first_w.shape[1]
    ae = HLTAutoencoder(ae_cfg).to(device)
    ae.load_state_dict(ae_ckpt["ae"])
    ae.eval()
    print(f"Loaded HLTAutoencoder: features={ae_cfg['features']}, "
          f"latent={ae_cfg['latent_dim']}", flush=True)
    return ae


# ── Inference ─────────────────────────────────────────────────────────────────

def compute_ae_scores(ae, ae_scaler, pt_path, device, batch_size=4096):
    """AE reco loss (MSE) per event using obj features from pt_path."""
    mu = ae_scaler["mu"].detach().cpu().float()
    std = ae_scaler["std"].detach().cpu().float().clamp(min=1e-8)

    raw = torch.load(pt_path, map_location="cpu")
    obj = raw["obj"]
    N = obj.shape[0]
    print(f"  AE inference on {N} events...", flush=True)

    scores = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            obj_batch = obj[i0:i0 + batch_size, :, :4]
            xb = obj_batch.reshape(obj_batch.shape[0], -1).float()
            xb = (xb - mu.view(1, -1)) / std.view(1, -1)
            xb = xb.to(device)
            recon, _ = ae(xb)
            mse = ((recon - xb) ** 2).mean(dim=1)
            scores.append(mse.cpu())
    del raw
    return torch.cat(scores).numpy().astype(np.float32)


def embed_pf(model, pt_path, device, batch_size=512):
    """Run NURD encoder; return latents, classifier logits, and labels."""
    raw    = torch.load(pt_path, map_location="cpu")
    pf     = torch.nan_to_num(raw["pf"], nan=0.0, posinf=0.0, neginf=0.0)
    labels = raw["label"].numpy()
    N = pf.shape[0]
    print(f"  Encoder inference on {N} events from {pt_path}...", flush=True)

    latents = []
    logits = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = pf[i0:i0 + batch_size].to(device)
            latent, output = model(xb)
            latents.append(latent.cpu())
            logits.append(output.cpu())
    del raw, pf
    return (
        torch.cat(latents, dim=0).numpy(),
        torch.cat(logits, dim=0).numpy(),
        labels,
    )


def make_reference_splits(labels, val_fraction=0.1, calibration_fraction=0.5,
                          seed=42):
    """Reproduce training split, then divide untouched validation into calibration/tuning."""
    labels = np.asarray(labels)
    indices = np.arange(labels.shape[0])
    train_indices, tune_indices = train_test_split(
        indices, test_size=val_fraction, random_state=seed, stratify=labels)
    calibration_indices, tune_indices = train_test_split(
        tune_indices, train_size=calibration_fraction, random_state=seed + 1,
        stratify=labels[tune_indices])
    return train_indices, calibration_indices, tune_indices


def class_assignment_diagnostics(true_labels, score_products):
    reference_labels = score_products["reference_labels"]
    true_labels = np.asarray(true_labels)
    result = {}
    for name, key in (
        ("classifier", "classifier_route_index"),
        ("gaussian", "gaussian_route_index"),
        ("typicality", "typicality_route_index"),
    ):
        predicted = reference_labels[score_products[key]]
        valid = np.isin(true_labels, reference_labels)
        matrix = np.zeros((len(reference_labels), len(reference_labels)), dtype=np.int64)
        for row, true_label in enumerate(reference_labels):
            for column, predicted_label in enumerate(reference_labels):
                matrix[row, column] = np.count_nonzero(
                    (true_labels == true_label) & (predicted == predicted_label))
        result[name] = {
            "accuracy": float(np.mean(predicted[valid] == true_labels[valid])) if valid.any() else np.nan,
            "confusion_matrix": matrix.tolist(),
        }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def ABCD(config):
    print("Logging in to wandb...", flush=True)
    wandb.login()
    resume_id = config.get("resume_run_id", None)
    wandb.init(project=config.get("wandb_project", "AE vs. Contrastive ABCD"),
               name=config.get("wandb_run_name", None),
               id=resume_id,
               resume="allow" if resume_id else None,
               settings=wandb.Settings(_disable_stats=True),
               config=config)
    run_name = wandb.run.name
    print(f"Run name: {run_name}", flush=True)

    outdir   = config.get("outdir", "outputs_abcd")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_labels = parse_int_list(config.get("baseline_labels", "0,1,2,3"))
    qcd_label = int(config.get("qcd_label", 1))
    abcd_scope = config.get("abcd_scope", "qcd")
    score_mode = config.get("score_mode", "calibrated_union")
    reference_pt = config.get("reference_pt")
    if not baseline_labels:
        raise ValueError("baseline_labels must contain at least one label")
    if bool(config.get("min_md")):
        print("WARNING: --min_md is deprecated; using score_mode=min_md.", flush=True)
        score_mode = "min_md"
    score_label = {
        "calibrated_union": "Calibrated all-background anomaly score",
        "mixture_nll": "Background-mixture negative log likelihood",
        "qcd_md": "QCD Mahalanobis distance",
        "min_md": "Minimum class Mahalanobis distance",
        "classifier_routed": "Classifier-routed calibrated anomaly score",
        "gaussian_routed": "Gaussian-routed calibrated anomaly score",
    }[score_mode]

    # ── load models ───────────────────────────────────────────────────────────
    model, main_ckpt = load_nurd_model(config["ckpt"], device)
    ae_scaler = main_ckpt["ae_scaler"]
    ae = load_ae(config["ae_ckpt"], ae_scaler, device)

    # The reference file is the original training sample. Its model-validation
    # split selects thresholds; the independent test file is report-only.
    ae_reference = None
    if reference_pt:
        print("Computing AE scores (reference)...", flush=True)
        ae_reference = compute_ae_scores(
            ae, ae_scaler, reference_pt, device,
            batch_size=config.get("ae_batch_size", 4096))
    print("Computing AE scores (test)...", flush=True)
    ae_bkg = compute_ae_scores(
        ae, ae_scaler, config["test_pt"], device,
        batch_size=config.get("ae_batch_size", 4096))

    # free AE GPU memory before running encoder
    del ae
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Axis 2 score mode: {score_mode}", flush=True)
    print(f"ABCD event scope: {abcd_scope}", flush=True)
    reference_details = {}
    if reference_pt:
        print("Computing encoder outputs (reference)...", flush=True)
        reference_latents, reference_logits, reference_labels = embed_pf(
            model, reference_pt, device,
            batch_size=config.get("batch_size", 512))
        fit_idx, calibration_idx, reference_tune_idx = make_reference_splits(
            reference_labels,
            val_fraction=float(config.get("reference_val_fraction", 0.1)),
            calibration_fraction=float(config.get("reference_calibration_fraction", 0.5)),
            seed=int(config.get("reference_split_seed", 42)))
        references = fit_class_references(
            reference_latents, reference_labels, fit_idx, calibration_idx,
            baseline_labels, n_components=config.get("n_pca"))
        reference_details = {
            "fit_n": int(len(fit_idx)),
            "calibration_n": int(len(calibration_idx)),
            "threshold_tune_n": int(len(reference_tune_idx)),
            "per_class_fit": {
                str(label): int(np.count_nonzero(reference_labels[fit_idx] == label))
                for label in baseline_labels
            },
            "per_class_calibration": {
                str(label): int(np.count_nonzero(reference_labels[calibration_idx] == label))
                for label in baseline_labels
            },
        }
        reference_axis2, reference_products = score_latents(
            reference_latents, reference_logits, references,
            score_mode=score_mode, qcd_label=qcd_label)
        print(
            f"Reference splits: fit={len(fit_idx)} calibration={len(calibration_idx)} "
            f"threshold_tune={len(reference_tune_idx)}",
            flush=True)
    else:
        print(
            "WARNING: no --reference_pt supplied. Latent references are fit on the "
            "test sample, so this compatibility mode must not be quoted as final.",
            flush=True)
        reference_latents = reference_logits = reference_labels = None
        reference_axis2 = reference_products = reference_tune_idx = None

    print("Computing encoder outputs (test)...", flush=True)
    latents_all, logits_all, labels = embed_pf(
        model, config["test_pt"], device,
        batch_size=config.get("batch_size", 512))
    if not reference_pt:
        fit_idx, calibration_idx, _ = make_reference_splits(
            labels, val_fraction=0.2, calibration_fraction=0.25,
            seed=int(config.get("reference_split_seed", 42)))
        references = fit_class_references(
            latents_all, labels, fit_idx, calibration_idx, baseline_labels,
            n_components=config.get("n_pca"))
    con_bkg, score_products = score_latents(
        latents_all, logits_all, references,
        score_mode=score_mode, qcd_label=qcd_label)
    class_transforms = [
        (ref.label, ref.mean, ref.whitening) for ref in references
    ]
    qcd_reference = next(ref for ref in references if ref.label == qcd_label)
    md_mu, md_W = qcd_reference.mean, qcd_reference.whitening

    if len(con_bkg) != len(ae_bkg):
        raise ValueError(f"Length mismatch: contrastive {len(con_bkg)} vs AE {len(ae_bkg)}")

    # ── mask ──────────────────────────────────────────────────────────────────
    mask = np.isfinite(ae_bkg) & np.isfinite(con_bkg) & (ae_bkg > 0)
    axis1_bkg = ae_bkg[mask]
    axis2_bkg = con_bkg[mask]
    labels_masked  = labels[mask]
    latents_masked = latents_all[mask]
    masked_score_products = {
        key: (value[mask] if isinstance(value, np.ndarray)
              and value.shape[:1] == (len(mask),) else value)
        for key, value in score_products.items()
    }
    print(f"Events after masking: {mask.sum()}", flush=True)

    emb_pca  = (latents_masked - md_mu) @ md_W
    n_pca    = emb_pca.shape[1]
    axis2_pca = axis2_bkg

    qcd_only  = labels_masked == qcd_label
    baseline_only = label_membership_mask(labels_masked, baseline_labels)
    axis1_qcd = axis1_bkg[qcd_only]
    axis2_qcd = axis2_pca[qcd_only]
    axis1_baselines = axis1_bkg[baseline_only]
    axis2_baselines = axis2_pca[baseline_only]
    labels_baselines = labels_masked[baseline_only]
    print(f"QCD events for ABCD: {qcd_only.sum()}", flush=True)
    print(f"Baseline events for ABCD: {baseline_only.sum()} labels={baseline_labels}", flush=True)

    corr_sample_size = int(config.get("corr_sample_size", 50_000))
    dcor_sample_size = int(config.get("dcor_sample_size", 2_000))
    diagnostics = {
        "counts": {
            "all_background": int(axis1_bkg.shape[0]),
            "qcd": int(qcd_only.sum()),
            "all_baselines": int(baseline_only.sum()),
            "per_class": {
                str(int(cls)): int(np.count_nonzero(labels_masked == int(cls)))
                for cls in baseline_labels
            },
        },
        "correlations": {
            "all_background": _safe_correlations(
                axis1_bkg, axis2_bkg,
                max_points=corr_sample_size,
                dcor_points=dcor_sample_size),
            "all_baselines": _safe_correlations(
                axis1_baselines, axis2_baselines,
                max_points=corr_sample_size,
                dcor_points=dcor_sample_size),
            "qcd": _safe_correlations(
                axis1_qcd, axis2_qcd,
                max_points=corr_sample_size,
                dcor_points=dcor_sample_size),
        },
        "per_class_correlations": per_class_correlations(
            axis1_bkg, axis2_bkg, labels_masked, baseline_labels,
            corr_sample_size=corr_sample_size,
            dcor_sample_size=dcor_sample_size),
        "score_definition": {
            "mode": score_mode,
            "abcd_scope": abcd_scope,
            "reference_source": reference_pt,
            "reference_labels": baseline_labels,
            "reference_is_independent_of_test": bool(reference_pt),
            "reference_splits": reference_details,
        },
        "class_assignment": class_assignment_diagnostics(
            labels_masked, masked_score_products),
    }
    print("Correlation diagnostics:", flush=True)
    for scope, vals in diagnostics["correlations"].items():
        print(
            f"  {scope}: pearson={vals['pearson']:.4f} "
            f"spearman={vals['spearman']:.4f} "
            f"distance_corr={vals['distance_corr']:.4f} n={vals['n']}",
            flush=True,
        )
    wandb.log({
        "Corr/all_pearson": diagnostics["correlations"]["all_background"]["pearson"],
        "Corr/all_spearman": diagnostics["correlations"]["all_background"]["spearman"],
        "Corr/all_distance": diagnostics["correlations"]["all_background"]["distance_corr"],
        "Corr/all_baselines_pearson": diagnostics["correlations"]["all_baselines"]["pearson"],
        "Corr/all_baselines_spearman": diagnostics["correlations"]["all_baselines"]["spearman"],
        "Corr/all_baselines_distance": diagnostics["correlations"]["all_baselines"]["distance_corr"],
        "Corr/qcd_pearson": diagnostics["correlations"]["qcd"]["pearson"],
        "Corr/qcd_spearman": diagnostics["correlations"]["qcd"]["spearman"],
        "Corr/qcd_distance": diagnostics["correlations"]["qcd"]["distance_corr"],
    })

    # ── signal (optional) ─────────────────────────────────────────────────────
    sig_axis1 = sig_axis2 = sig_axis2_pca = None
    sig_latents_masked = sig_emb_pca = None
    if config.get("signal_pt"):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print("Running signal inference...", flush=True)
        sig_latents, sig_logits, _ = embed_pf(
            model, config["signal_pt"], device,
            batch_size=config.get("batch_size", 512))
        sig_con, sig_score_products = score_latents(
            sig_latents, sig_logits, references,
            score_mode=score_mode, qcd_label=qcd_label)

        ae_sig = load_ae(config["ae_ckpt"], ae_scaler, device)
        sig_ae = compute_ae_scores(
            ae_sig, ae_scaler, config["signal_pt"], device,
            batch_size=config.get("ae_batch_size", 4096))
        del ae_sig

        sig_mask = np.isfinite(sig_ae) & np.isfinite(sig_con) & (sig_ae > 0)
        sig_axis1         = sig_ae[sig_mask]
        sig_axis2         = sig_con[sig_mask]
        sig_latents_masked = sig_latents[sig_mask]
        sig_emb_pca       = (sig_latents_masked - md_mu) @ md_W
        # Every main AE-vs-score plot must use the same deployed axis for signal
        # and background. QCD-coordinate PCA plots use sig_emb_pca explicitly.
        sig_axis2_pca     = sig_axis2
        print(f"Signal events after masking: {sig_mask.sum()}", flush=True)

    # ── ABCD scan ─────────────────────────────────────────────────────────────
    percent = np.linspace(
        float(config.get("scan_percent_min", 0.50)),
        float(config.get("scan_percent_max", 0.98)),
        int(config.get("scan_percent_steps", 48)))
    min_A   = int(config.get("min_A", 50))
    min_D   = int(config.get("min_D", 500))
    min_A_frac = float(config.get("min_A_frac", 0.0))
    selection_stat_weight = float(config.get("selection_stat_weight", 0.0))
    holdout_frac = float(config.get("closure_holdout_frac", 0.5))
    split_seed = int(config.get("closure_split_seed", 42))
    if abcd_scope in {"all_baselines", "baselines"}:
        axis1_report = axis1_baselines
        axis2_report = axis2_baselines
        labels_report = labels_baselines
        scope_name = "all_baselines"
    elif abcd_scope == "qcd":
        axis1_report = axis1_qcd
        axis2_report = axis2_qcd
        labels_report = labels_masked[qcd_only]
        scope_name = "qcd"
    else:
        raise ValueError(f"Unsupported abcd_scope={abcd_scope!r}")

    if reference_pt:
        reference_valid = (
            np.isfinite(ae_reference)
            & np.isfinite(reference_axis2)
            & (ae_reference > 0)
        )
        reference_candidates = reference_tune_idx[reference_valid[reference_tune_idx]]
        if scope_name == "qcd":
            reference_candidates = reference_candidates[
                reference_labels[reference_candidates] == qcd_label]
        else:
            reference_candidates = reference_candidates[
                label_membership_mask(
                    reference_labels[reference_candidates], baseline_labels)]
        axis1_tune = ae_reference[reference_candidates]
        axis2_tune = reference_axis2[reference_candidates]
        labels_tune = reference_labels[reference_candidates]
        closure_mode = "train_validation_to_independent_test"
    else:
        tune_idx, report_idx, closure_mode = split_for_threshold_report(
            len(axis1_report), holdout_frac=holdout_frac, seed=split_seed,
            axis1=axis1_report, axis2=axis2_report,
            strata_labels=labels_report if scope_name == "all_baselines" else None)
        axis1_tune, axis2_tune = axis1_report[tune_idx], axis2_report[tune_idx]
        labels_tune = labels_report[tune_idx]
        axis1_report, axis2_report = axis1_report[report_idx], axis2_report[report_idx]
        labels_report = labels_report[report_idx]
    print(
        f"ABCD threshold scope: {scope_name}; mode: {closure_mode}; "
        f"tune={len(axis1_tune)} report={len(axis1_report)}",
        flush=True,
    )

    best_tune, tune_grid_summary = scan_abcd_grid(
        axis1_tune, axis2_tune, percent, min_A=min_A, min_D=min_D,
        min_A_frac=min_A_frac, selection_stat_weight=selection_stat_weight)
    if "t1" not in best_tune:
        raise RuntimeError("No ABCD working point found on threshold-tuning split. "
                           "Try lowering min_A/min_D or closure_holdout_frac.")

    t1_opt, t2_opt = best_tune["t1"], best_tune["t2"]
    report_at_selected = abcd_record_at_thresholds(axis1_report, axis2_report, t1_opt, t2_opt)
    report_at_selected.update({
        "p1": float(best_tune["p1"]),
        "p2": float(best_tune["p2"]),
        "selection_nonclosure": float(best_tune["nonclosure"]),
        "selection_log_nonclosure": float(best_tune["log_nonclosure"]),
        "selection_score": float(best_tune.get("selection_score", np.nan)),
    })
    best_report_scan, grid_summary = scan_abcd_grid(
        axis1_report, axis2_report, percent, min_A=min_A, min_D=min_D,
        min_A_frac=min_A_frac, selection_stat_weight=selection_stat_weight)
    selected_per_class = per_class_abcd_at_thresholds(
        axis1_report, axis2_report, labels_report, baseline_labels, t1_opt, t2_opt)

    print(f"Optimized on tune split: p1={best_tune['p1']:.3f}, p2={best_tune['p2']:.3f}", flush=True)
    print(f"Thresholds: t1={t1_opt:.4g}, t2={t2_opt:.4g}", flush=True)
    print(f"Tune nonclosure: {100.0*best_tune['nonclosure']:.2f}%", flush=True)
    print(
        f"Reported nonclosure: {100.0*report_at_selected['nonclosure']:.2f}% "
        f"(ABCD/true ratio={report_at_selected['ratio']:.4f})",
        flush=True,
    )

    diagnostics["abcd_selection"] = {
        "scope": scope_name,
        "score_mode": score_mode,
        "mode": closure_mode,
        "holdout_frac": holdout_frac,
        "split_seed": split_seed,
        "min_A": min_A,
        "min_A_frac": min_A_frac,
        "min_D": min_D,
        "selection_stat_weight": selection_stat_weight,
        "tune_n": int(len(axis1_tune)),
        "report_n": int(len(axis1_report)),
        "tune_best": best_tune,
        "report_at_selected": report_at_selected,
        "report_at_selected_per_class": selected_per_class,
        "report_best_for_reference": best_report_scan,
    }
    diagnostics["abcd_grid"] = grid_summary
    diagnostics["abcd_tune_grid"] = tune_grid_summary
    print(
        "ABCD report grid: "
        f"mean |nonclosure|={grid_summary['mean_abs_nonclosure']:.4f}, "
        f"median={grid_summary['median_abs_nonclosure']:.4f}, "
        f"p90={grid_summary['p90_abs_nonclosure']:.4f}, "
        f"points={grid_summary['n_points']}",
        flush=True,
    )

    wandb.log({
        "ABCD/opt_p1":     report_at_selected["p1"],
        "ABCD/opt_p2":     report_at_selected["p2"],
        "ABCD/opt_t1":     float(t1_opt),
        "ABCD/opt_t2":     float(t2_opt),
        "ABCD/nonclosure": float(report_at_selected["nonclosure"]),
        "ABCD/legacy_nonclosure": float(report_at_selected["legacy_nonclosure"]),
        "ABCD/log_nonclosure": float(report_at_selected["log_nonclosure"]),
        "ABCD/ratio_pred_over_true": float(report_at_selected["ratio"]),
        "ABCD/selection_score": float(report_at_selected["selection_score"]),
        "ABCD/tune_nonclosure": float(best_tune["nonclosure"]),
        "ABCD/tune_log_nonclosure": float(best_tune["log_nonclosure"]),
        "ABCD/report_best_nonclosure": float(best_report_scan.get("nonclosure", np.nan)),
        "ABCD/report_best_log_nonclosure": float(best_report_scan.get("log_nonclosure", np.nan)),
        "ABCD/scope_all_baselines": int(scope_name == "all_baselines"),
        "ABCD/closure_mode_holdout": int(closure_mode.startswith("holdout")),
        "ABCD/closure_mode_independent_test": int(
            closure_mode == "train_validation_to_independent_test"),
        "ABCD/grid_mean_abs_nonclosure": grid_summary["mean_abs_nonclosure"],
        "ABCD/grid_median_abs_nonclosure": grid_summary["median_abs_nonclosure"],
        "ABCD/grid_p90_abs_nonclosure": grid_summary["p90_abs_nonclosure"],
        "ABCD/grid_points": grid_summary["n_points"],
        "ABCD/A": int(report_at_selected["A"]), "ABCD/B": int(report_at_selected["B"]),
        "ABCD/C": int(report_at_selected["C"]), "ABCD/D": int(report_at_selected["D"]),
    })

    if sig_axis1 is not None and sig_axis2 is not None:
        sig_A, sig_B, sig_C, sig_D = abcd_counts_at_thresholds(sig_axis1, sig_axis2, t1_opt, t2_opt)
        sig_total = max(len(sig_axis1), 1)
        signal_metrics = {
            "N": int(len(sig_axis1)),
            "A": int(sig_A),
            "B": int(sig_B),
            "C": int(sig_C),
            "D": int(sig_D),
            "eff_A": float(sig_A / sig_total),
            "eff_B": float(sig_B / sig_total),
            "eff_C": float(sig_C / sig_total),
            "eff_D": float(sig_D / sig_total),
            "s_over_sqrt_Ahat": float(sig_A / np.sqrt(max(report_at_selected["A_hat"], 1e-8))),
            "s_over_sqrt_A": float(sig_A / np.sqrt(max(report_at_selected["A"], 1e-8))),
        }
        diagnostics["signal_at_selected"] = signal_metrics
        wandb.log({
            "Signal/A": signal_metrics["A"],
            "Signal/eff_A": signal_metrics["eff_A"],
            "Signal/eff_B": signal_metrics["eff_B"],
            "Signal/eff_C": signal_metrics["eff_C"],
            "Signal/eff_D": signal_metrics["eff_D"],
            "Signal/S_over_sqrt_Ahat": signal_metrics["s_over_sqrt_Ahat"],
            "Signal/S_over_sqrt_A": signal_metrics["s_over_sqrt_A"],
        })

    # ── Plots ─────────────────────────────────────────────────────────────────
    fs, fs_leg, fs_legend = 28, 24, 16
    fig_size = (8, 6)

    class_names  = CLASS_NAMES
    class_colors = CLASS_COLORS

    # 2D histogram (all bkg)
    fig = plt.figure(figsize=(6, 5))
    xbins = np.geomspace(axis1_bkg[axis1_bkg > 0].min(), axis1_bkg.max(), 201)
    ybins = np.geomspace(axis2_bkg[axis2_bkg > 0].min(), axis2_bkg.max(), 201)
    plt.hist2d(axis1_bkg, axis2_bkg, bins=[xbins, ybins], norm=LogNorm(vmin=1), cmin=1)
    plt.xscale("log"); plt.yscale("log")
    plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
    plt.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("AE reco loss"); plt.ylabel(score_label)
    plt.title("AE vs NURD score (bkg only)"); plt.colorbar(label="Counts")
    out = os.path.join(plot_dir, "hist2d_bkg.png")
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    wandb.log({"Hists2D/bkg": wandb.Image(out)})

    # combined scatter by class
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() == 0:
            continue
        ax.scatter(axis1_bkg[m], axis2_bkg[m], s=0.3, alpha=0.15,
                   color=class_colors[cls], label=name, rasterized=True)
    if sig_axis1 is not None:
        ax.scatter(sig_axis1, sig_axis2, s=0.5, alpha=0.4,
                   color="tab:purple", label="TpTp", rasterized=True)
    ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AE reco loss", fontsize=fs)
    ax.set_ylabel(score_label, fontsize=fs)
    ax.set_title("AE vs NURD score - all classes")
    ax.legend(markerscale=10, fontsize=fs_legend)
    out_combined = os.path.join(plot_dir, "hist2d_by_class_combined.png")
    fig.savefig(out_combined, dpi=200, bbox_inches="tight"); plt.close(fig)
    wandb.log({"Hists2D/by_class_combined": wandb.Image(out_combined)})

    # signal hist2d
    if sig_axis1 is not None:
        fig = plt.figure(figsize=(6, 5))
        xbins_s = np.geomspace(sig_axis1[sig_axis1 > 0].min(), sig_axis1.max(), 101)
        ybins_s = np.geomspace(sig_axis2[sig_axis2 > 0].min(), sig_axis2.max(), 101)
        plt.hist2d(sig_axis1, sig_axis2, bins=[xbins_s, ybins_s], norm=LogNorm(vmin=1), cmin=1)
        plt.xscale("log"); plt.yscale("log")
        plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        plt.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
        plt.xlabel("AE reco loss", fontsize=fs)
        plt.ylabel(score_label, fontsize=fs)
        plt.title("AE vs NURD score - TpTp (signal)"); plt.colorbar(label="Counts")
        out_sig = os.path.join(plot_dir, "hist2d_TpTp.png")
        plt.savefig(out_sig, dpi=200, bbox_inches="tight"); plt.close()
        wandb.log({"Hists2D/TpTp": wandb.Image(out_sig)})

    # individual hist2d per class
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() < 2:
            continue
        x_cls, y_cls = axis1_bkg[m], axis2_bkg[m]
        fig = plt.figure(figsize=(6, 5))
        xbins_c = np.geomspace(x_cls[x_cls > 0].min(), x_cls.max(), 101)
        ybins_c = np.geomspace(y_cls[y_cls > 0].min(), y_cls.max(), 101)
        plt.hist2d(x_cls, y_cls, bins=[xbins_c, ybins_c], norm=LogNorm(vmin=1), cmin=1)
        plt.xscale("log"); plt.yscale("log")
        plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        plt.xlabel("AE reco loss", fontsize=fs)
        plt.ylabel(score_label, fontsize=fs)
        plt.title(f"AE vs NURD score - {name}"); plt.colorbar(label="Counts")
        out_cls = os.path.join(plot_dir, f"hist2d_{name}.png")
        plt.savefig(out_cls, dpi=200, bbox_inches="tight"); plt.close()
        wandb.log({f"Hists2D/{name}": wandb.Image(out_cls)})

    # Selected-score scatter + KDE. Filenames retain the historical names.
    if not config.get("skip_pca_md_plots"):
        fig, ax = plt.subplots(figsize=fig_size)
        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() == 0:
                continue
            ax.scatter(axis1_bkg[m], axis2_pca[m],
                       s=0.3, alpha=0.15, color=class_colors[cls], label=name, rasterized=True)
        if sig_axis1 is not None and sig_axis2_pca is not None:
            ax.scatter(sig_axis1, sig_axis2_pca, s=0.5, alpha=0.4,
                       color="tab:purple", label="TpTp", rasterized=True)
        ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("AE reco loss", fontsize=fs)
        ax.set_ylabel(score_label, fontsize=fs)
        ax.set_title("AE vs selected NURD score - all classes")
        ax.legend(markerscale=10, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_pca_scatter = os.path.join(plot_dir, "hist2d_pca_md_scatter.png")
        fig.savefig(out_pca_scatter, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"Hists2D/pca_md_scatter": wandb.Image(out_pca_scatter)})

        # KDE contours
        rng_pca = np.random.default_rng(42)
        fig, ax = plt.subplots(figsize=fig_size)
        kde_legend_handles = []
        all_classes_for_kde = list(class_names.items())
        if sig_axis1 is not None and sig_axis2_pca is not None:
            all_classes_for_kde.append((-1, "TpTp"))

        all_lx, all_ly = [], []
        for cls, name in all_classes_for_kde:
            x_raw = sig_axis1 if cls == -1 else axis1_bkg[labels_masked == cls]
            y_raw = sig_axis2_pca if cls == -1 else axis2_pca[labels_masked == cls]
            valid = (x_raw > 0) & (y_raw > 0) & np.isfinite(x_raw) & np.isfinite(y_raw)
            if valid.sum() >= 50:
                all_lx.append(np.log10(x_raw[valid]))
                all_ly.append(np.log10(y_raw[valid]))
        glx_min, glx_max = np.concatenate(all_lx).min(), np.concatenate(all_lx).max()
        gly_min, gly_max = np.concatenate(all_ly).min(), np.concatenate(all_ly).max()
        xi_global, yi_global = np.mgrid[glx_min:glx_max:200j, gly_min:gly_max:200j]

        for cls, name in all_classes_for_kde:
            if cls == -1:
                x_raw, y_raw = sig_axis1, sig_axis2_pca
                color = "tab:purple"
            else:
                m = labels_masked == cls
                if m.sum() < 50:
                    continue
                x_raw, y_raw = axis1_bkg[m], axis2_pca[m]
                color = class_colors[cls]
            valid = (x_raw > 0) & (y_raw > 0) & np.isfinite(x_raw) & np.isfinite(y_raw)
            lx = np.log10(x_raw[valid]); ly = np.log10(y_raw[valid])
            if lx.shape[0] > 20_000:
                idx = rng_pca.choice(lx.shape[0], 20_000, replace=False)
                lx, ly = lx[idx], ly[idx]
            kde = gaussian_kde(np.vstack([lx, ly]))
            zi  = kde(np.vstack([xi_global.flatten(), yi_global.flatten()]))
            zi_grid = zi.reshape(xi_global.shape)
            # only draw contours in the bulk; suppress far-tail lines
            levels = zi_grid.max() * np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.88])
            ax.contour(10**xi_global, 10**yi_global, zi_grid,
                       levels=levels, colors=color, alpha=0.7, linewidths=1.5)
            kde_legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=name))

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("AE reco loss", fontsize=fs)
        ax.set_ylabel(score_label, fontsize=fs)
        ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("AE vs selected NURD score - KDE contours")
        ax.legend(handles=kde_legend_handles, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        ax.grid(alpha=0.3)
        out_pca_kde = os.path.join(plot_dir, "hist2d_pca_md_kde.png")
        fig.savefig(out_pca_kde, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"Hists2D/pca_md_kde": wandb.Image(out_pca_kde)})

    # PCA embedding scatter + KDE
    if not config.get("skip_embedding_pca"):
        pca2 = PCA(n_components=2)
        pca2.fit(latents_masked[labels_masked == qcd_label])
        emb_2d = pca2.transform(latents_masked)
        sig_emb_2d = pca2.transform(sig_latents_masked) if sig_latents_masked is not None else None

        fig, ax = plt.subplots(figsize=fig_size)
        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() == 0:
                continue
            ax.scatter(emb_2d[m, 0], emb_2d[m, 1],
                       s=0.5, alpha=0.12, color=class_colors[cls],
                       label=name, rasterized=True)
        if sig_emb_2d is not None:
            ax.scatter(sig_emb_2d[:, 0], sig_emb_2d[:, 1],
                       s=0.5, alpha=0.4, color="tab:purple", label="TpTp", rasterized=True)
        ax.set_xlabel("PCA Component 1", fontsize=fs)
        ax.set_ylabel("PCA Component 2", fontsize=fs)
        ax.set_title(f"NURD latent — PCA scatter (fit on {CLASS_NAMES.get(qcd_label, qcd_label)})")
        ax.legend(markerscale=10, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_pca_scatter2 = os.path.join(plot_dir, "pca_scatter_embeddings.png")
        fig.savefig(out_pca_scatter2, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"PCA/scatter": wandb.Image(out_pca_scatter2)})

    # Corner plot: pairwise PCA-MD components
    if n_pca >= 2:
        pairs  = [(i, j) for i in range(n_pca) for j in range(i + 1, n_pca)]
        n_pairs = len(pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]
        for ax, (ci, cj) in zip(axes, pairs):
            for cls, name in class_names.items():
                m = labels_masked == cls
                if m.sum() == 0:
                    continue
                ax.scatter(emb_pca[m, ci], emb_pca[m, cj],
                           s=0.3, alpha=0.12, color=class_colors[cls],
                           label=name, rasterized=True)
            if sig_emb_pca is not None:
                ax.scatter(sig_emb_pca[:, ci], sig_emb_pca[:, cj],
                           s=0.5, alpha=0.4, color="tab:purple", label="TpTp", rasterized=True)
            ax.set_xlabel(f"PCA Component {ci + 1}", fontsize=fs)
            ax.set_ylabel(f"PCA Component {cj + 1}", fontsize=fs)
            ax.legend(markerscale=10, fontsize=fs_legend)
            ax.tick_params(axis="both", labelsize=fs_leg)
        fig.suptitle(
            f"PCA-MD space — pairwise components (NURD latent, fit on {CLASS_NAMES.get(qcd_label, qcd_label)})",
            fontsize=fs)
        plt.tight_layout()
        out_corner = os.path.join(plot_dir, "pca_corner.png")
        fig.savefig(out_corner, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"PCA/corner": wandb.Image(out_corner)})

    # Profile plots
    for (x_arr, y_arr, xlabel, ylabel, title, key) in [
        (axis2_bkg, axis1_bkg, score_label, "Mean AE reco loss",
         "⟨AE loss⟩ vs NURD MD", "AE_vs_contrastive"),
        (axis1_bkg, axis2_bkg, "AE reco loss", f"Mean {score_label}",
         "⟨NURD MD⟩ vs AE loss", "contrastive_vs_AE"),
    ]:
        fig, ax = plt.subplots(figsize=fig_size)
        profile_plot(ax, x_arr, y_arr, nbins=60, logx=True)
        ax.set_xlabel(xlabel, fontsize=fs); ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_p = os.path.join(plot_dir, f"profile_{key}.png")
        fig.savefig(out_p, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({f"Profiles/{key}": wandb.Image(out_p)})

    for (x_arr, y_arr, xlabel, ylabel, title, key) in [
        (axis2_bkg, axis1_bkg, score_label, "Mean AE reco loss",
         "⟨AE loss⟩ vs NURD MD (by class)", "AE_vs_contrastive_by_class"),
        (axis1_bkg, axis2_bkg, "AE reco loss", f"Mean {score_label}",
         "⟨NURD MD⟩ vs AE loss (by class)", "contrastive_vs_AE_by_class"),
    ]:
        fig, ax = plt.subplots(figsize=fig_size)
        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() < 20:
                continue
            profile_plot(ax, x_arr[m], y_arr[m], nbins=40, logx=True, label=name)
        ax.set_xlabel(xlabel, fontsize=fs); ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title); ax.legend(fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_p = os.path.join(plot_dir, f"profile_{key}.png")
        fig.savefig(out_p, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({f"Profiles/{key}": wandb.Image(out_p)})

    # 1D closure scan
    effs, closure_ratio, closure_unc = [], [], []
    curve_axis1 = axis1_report
    curve_axis2 = axis2_report
    Ntot_bkg = float(len(curve_axis1))

    for p in percent:
        t1, t2, A, B, C, D = abcd_counts(curve_axis1, curve_axis2, p, p)
        A_hat  = (B * C) / max(D, 1e-8)
        ratio  = A_hat / max(A, 1e-8)
        invA   = 0.0 if A == 0 else 1.0 / A
        invB   = 0.0 if B == 0 else 1.0 / B
        invC   = 0.0 if C == 0 else 1.0 / C
        invD   = 0.0 if D == 0 else 1.0 / D
        rel_var = invA + invB + invC + invD
        sigma  = abs(ratio) * np.sqrt(rel_var) if rel_var > 0 else 0.0
        effs.append(A / max(Ntot_bkg, 1.0))
        closure_ratio.append(ratio)
        closure_unc.append(sigma)

    effs          = np.array(effs)
    closure_ratio = np.array(closure_ratio)
    closure_unc   = np.array(closure_unc)
    order         = np.argsort(effs)
    effs          = effs[order]
    closure_ratio = closure_ratio[order]
    closure_unc   = closure_unc[order]

    eff_opt   = report_at_selected["A"] / max(Ntot_bkg, 1.0)
    ratio_opt = report_at_selected["ratio"]
    curve_abs = np.abs(closure_ratio - 1.0)
    curve_summary = {
        "median_abs_ratio_minus1": float(np.median(curve_abs)) if curve_abs.size else np.nan,
        "p90_abs_ratio_minus1": float(np.quantile(curve_abs, 0.90)) if curve_abs.size else np.nan,
        "min_ratio": float(np.min(closure_ratio)) if closure_ratio.size else np.nan,
        "max_ratio": float(np.max(closure_ratio)) if closure_ratio.size else np.nan,
    }
    tail = effs <= 0.02
    if tail.any():
        curve_summary["tail_le_2pct_mean_ratio"] = float(np.mean(closure_ratio[tail]))
        curve_summary["tail_le_2pct_median_abs_ratio_minus1"] = float(np.median(curve_abs[tail]))
    else:
        curve_summary["tail_le_2pct_mean_ratio"] = np.nan
        curve_summary["tail_le_2pct_median_abs_ratio_minus1"] = np.nan
    diagnostics["closure_curve"] = curve_summary
    wandb.log({
        "Closure/median_abs_ratio_minus1": curve_summary["median_abs_ratio_minus1"],
        "Closure/p90_abs_ratio_minus1": curve_summary["p90_abs_ratio_minus1"],
        "Closure/min_ratio": curve_summary["min_ratio"],
        "Closure/max_ratio": curve_summary["max_ratio"],
        "Closure/tail_le_2pct_mean_ratio": curve_summary["tail_le_2pct_mean_ratio"],
        "Closure/tail_le_2pct_median_abs_ratio_minus1": curve_summary["tail_le_2pct_median_abs_ratio_minus1"],
    })

    fig, ax = plt.subplots(figsize=fig_size)
    curve_label = f"AE + {score_label}"
    if closure_mode == "train_validation_to_independent_test":
        curve_label += " independent test"
    elif closure_mode.startswith("holdout"):
        curve_label += " held-out"
    ax.plot(effs, closure_ratio, c="g", label=curve_label)
    ax.fill_between(effs, closure_ratio - closure_unc, closure_ratio + closure_unc,
                    facecolor="g", alpha=0.5, interpolate=True)
    ax.plot(effs, np.ones_like(effs),       linestyle="-",  color="black")
    ax.plot(effs, np.full_like(effs, 0.95), linestyle="--", color="black")
    ax.plot(effs, np.full_like(effs, 1.05), linestyle="--", color="black")
    ax.plot([eff_opt], [ratio_opt], marker="o", c="red", label="Selected threshold")
    ax.set_xlabel("Selection Efficiency (bkg A/Ntot)", fontsize=fs)
    ax.set_ylabel("Predicted Bkg. / True Bkg.",        fontsize=fs)
    ax.set_ylim([0.0, 1.5]); ax.set_xscale("log")
    plt.tick_params(axis="x", labelsize=fs_leg)
    plt.tick_params(axis="y", labelsize=fs_leg)
    plt.legend(loc="lower right", fontsize=fs_legend)
    closure_path = os.path.join(plot_dir, "cut_and_count_bkg_check.png")
    plt.savefig(closure_path, dpi=200, bbox_inches="tight"); plt.close()
    wandb.log({"Closure/plot": wandb.Image(closure_path)})

    # Save thresholds JSON so make_datacard_ttbar.py can skip the scan
    thresholds_path = os.path.join(outdir, "abcd_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump({
            "t1":    float(t1_opt),
            "t2":    float(t2_opt),
            "p1":    float(report_at_selected["p1"]),
            "p2":    float(report_at_selected["p2"]),
            "n_pca": config.get("n_pca", None),
            "nonclosure": float(report_at_selected["nonclosure"]),
            "legacy_nonclosure": float(report_at_selected["legacy_nonclosure"]),
            "log_nonclosure": float(report_at_selected["log_nonclosure"]),
            "ratio": float(report_at_selected["ratio"]),
            "tune_nonclosure": float(best_tune["nonclosure"]),
            "tune_log_nonclosure": float(best_tune["log_nonclosure"]),
            "report_best_nonclosure": float(best_report_scan.get("nonclosure", np.nan)),
            "report_best_log_nonclosure": float(best_report_scan.get("log_nonclosure", np.nan)),
            "abcd_scope": scope_name,
            "score_mode": score_mode,
            "reference_pt": reference_pt,
            "closure_mode": closure_mode,
            "min_A": min_A,
            "min_A_frac": min_A_frac,
            "min_D": min_D,
            "selection_stat_weight": selection_stat_weight,
            "report_A": int(report_at_selected["A"]),
            "report_B": int(report_at_selected["B"]),
            "report_C": int(report_at_selected["C"]),
            "report_D": int(report_at_selected["D"]),
            "report_per_class": selected_per_class,
            "signal_at_selected": diagnostics.get("signal_at_selected"),
        }, f, indent=2)
    print(f"Thresholds saved to: {thresholds_path}", flush=True)

    score_path = os.path.join(outdir, "event_scores.npz")
    score_payload = {
        "ae_score": axis1_bkg.astype(np.float32),
        "selected_score": axis2_bkg.astype(np.float32),
        "true_label": labels_masked.astype(np.int16),
        "score_mode": np.asarray(score_mode),
    }
    score_payload.update({
        key: value for key, value in masked_score_products.items()
        if isinstance(value, np.ndarray)
    })
    np.savez_compressed(score_path, **score_payload)
    print(f"Class-resolved event scores saved to: {score_path}", flush=True)

    diagnostics_path = os.path.join(outdir, "diagnostics.json")
    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Diagnostics saved to: {diagnostics_path}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True,
                        help="Path to NURD main checkpoint (checkpoint_main.pth.tar)")
    parser.add_argument("--ae_ckpt",      required=True,
                        help="Path to AE checkpoint (checkpoint_ae.pth)")
    parser.add_argument("--test_pt",      required=True,
                        help="Path to test .pt file (SM cocktail)")
    parser.add_argument("--reference_pt", default=None,
                        help="Training .pt file used to fit class references and select "
                             "thresholds on its model-validation split. Required for final results.")
    parser.add_argument("--signal_pt",    default=None,
                        help="Optional signal .pt file")
    parser.add_argument("--outdir",       default="outputs_abcd")
    parser.add_argument("--min_A",        type=int, default=50)
    parser.add_argument("--min_D",        type=int, default=500)
    parser.add_argument("--min_A_frac",   type=float, default=0.0,
                        help="Minimum A-region fraction on the threshold-tuning sample. "
                             "Useful to avoid low-stat working points.")
    parser.add_argument("--selection_stat_weight", type=float, default=0.0,
                        help="Add this times the tune-split ABCD ratio uncertainty to the "
                             "threshold-selection score.")
    parser.add_argument("--scan_percent_min", type=float, default=0.50)
    parser.add_argument("--scan_percent_max", type=float, default=0.98)
    parser.add_argument("--scan_percent_steps", type=int, default=48)
    parser.add_argument("--abcd_scope", default="qcd", choices=["qcd", "all_baselines", "baselines"],
                        help="Event population used for ABCD threshold selection and reporting.")
    parser.add_argument(
        "--score_mode", default="calibrated_union",
        choices=[
            "calibrated_union", "mixture_nll", "qcd_md", "min_md",
            "classifier_routed", "gaussian_routed",
        ],
        help="Definition of ABCD axis 2, independent of --abcd_scope. "
             "calibrated_union is high only when an event is atypical for every baseline.")
    parser.add_argument("--baseline_labels", default="0,1,2,3",
                        help="Comma/space-separated baseline background labels.")
    parser.add_argument("--qcd_label", type=int, default=1)
    parser.add_argument("--n_pca",        type=int, default=None,
                        help="Number of PCA components for MD (default: keep all latent dims)")
    parser.add_argument("--batch_size",   type=int, default=512,
                        help="Batch size for NURD encoder inference")
    parser.add_argument("--ae_batch_size", type=int, default=4096,
                        help="Batch size for AE inference")
    parser.add_argument("--corr_sample_size", type=int, default=50_000,
                        help="Max events sampled for Pearson/Spearman correlation diagnostics")
    parser.add_argument("--dcor_sample_size", type=int, default=2_000,
                        help="Max events sampled for distance-correlation diagnostics; 0 disables it")
    parser.add_argument("--closure_holdout_frac", type=float, default=0.5,
                        help="Compatibility mode used only without --reference_pt: fraction "
                             "held out for reporting after threshold selection.")
    parser.add_argument("--closure_split_seed", type=int, default=42,
                        help="Random seed for QCD tune/report split")
    parser.add_argument("--reference_val_fraction", type=float, default=0.1,
                        help="Must match the validation fraction used in train_hlt.py.")
    parser.add_argument("--reference_calibration_fraction", type=float, default=0.5,
                        help="Fraction of the untouched model-validation split used to "
                             "calibrate class MD tails; the rest selects ABCD thresholds.")
    parser.add_argument("--reference_split_seed", type=int, default=42,
                        help="Seed used to reproduce the model train/validation split.")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_project",  default="AE vs. Contrastive ABCD",
                        help="W&B project to log to")
    parser.add_argument("--resume_run_id",  default=None,
                        help="Resume an existing W&B run (e.g. the training run from a sweep)")
    parser.add_argument("--skip_pca_md_plots",   action="store_true")
    parser.add_argument("--skip_embedding_pca",  action="store_true")
    parser.add_argument("--min_md",              action="store_true",
                        help="Deprecated alias for --score_mode min_md")
    args = parser.parse_args()
    ABCD(vars(args))
