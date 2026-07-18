"""
ABCD eval for the NURD contrastive checkpoint (hlt_nurd_con).

Axis 1: AE reco loss (HLTAutoencoder, loaded from separate ae_ckpt)
Axis 2: Mahalanobis distance in PCA-whitened NURD latent space

Usage
-----
python eval_abcd_nurd.py \
    --ckpt      /eos/user/e/escheull/ssl_checkpoints/hlt/hlt/hlt_nurd_run_epoch_critic/checkpoint_main.pth.tar \
    --ae_ckpt   /eos/user/e/escheull/ssl_checkpoints/hlt/hlt/ae_pretrain/checkpoint_ae.pth \
    --test_pt   /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    [--signal_pt /eos/user/e/escheull/signal_pt/hlt_signal_TpTp.pt] \
    [--n_pca 6] \
    [--outdir /eos/user/e/escheull/abcd_outputs] \
    [--wandb_run_name nurd_abcd_v1]
"""
import os
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic, gaussian_kde
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

from models.hlt_con import HLTContrastiveModel
from models.hlt_autoencoder import HLTAutoencoder


# ── ABCD helpers (identical to eval_abcd.py) ─────────────────────────────────

def abcd_counts(loss_1, loss_2, percent_1, percent_2):
    thresh_1 = np.quantile(loss_1, percent_1)
    thresh_2 = np.quantile(loss_2, percent_2)
    A = int(((loss_1 > thresh_1) & (loss_2 > thresh_2)).sum())
    B = int(((loss_1 > thresh_1) & (loss_2 <= thresh_2)).sum())
    C = int(((loss_1 <= thresh_1) & (loss_2 > thresh_2)).sum())
    D = int(((loss_1 <= thresh_1) & (loss_2 <= thresh_2)).sum())
    return thresh_1, thresh_2, A, B, C, D


def nonclosure_A(A, B, C, D, eps=1e-8):
    A_hat = (B * C) / max(D, eps)
    if A_hat <= 0:
        return np.inf, A_hat
    return (A - A_hat) / A_hat, A_hat


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
    mu  = ae_scaler["mu"].cpu().numpy()
    std = ae_scaler["std"].cpu().numpy()

    raw = torch.load(pt_path, map_location="cpu")
    obj = raw["obj"][:, :, :4].reshape(raw["obj"].shape[0], -1).float().numpy()
    obj_norm = torch.from_numpy(((obj - mu) / (std + 1e-8)).astype(np.float32))
    N = obj_norm.shape[0]
    print(f"  AE inference on {N} events...", flush=True)

    scores = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = obj_norm[i0:i0 + batch_size].to(device)
            recon, _ = ae(xb)
            mse = ((recon - xb) ** 2).mean(dim=1)
            scores.append(mse.cpu())
    return torch.cat(scores).numpy().astype(np.float32)


def embed_pf(model, pt_path, device, batch_size=512):
    """Run NURD encoder on PF candidates; return (latents [N,D], labels [N])."""
    raw    = torch.load(pt_path, map_location="cpu")
    pf     = torch.nan_to_num(raw["pf"], nan=0.0, posinf=0.0, neginf=0.0)
    labels = raw["label"].numpy()
    N = pf.shape[0]
    print(f"  Encoder inference on {N} events from {pt_path}...", flush=True)

    latents = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = pf[i0:i0 + batch_size].to(device)
            latent, _ = model(xb)
            latents.append(latent.cpu())
    return torch.cat(latents, dim=0).numpy(), labels


def _fit_class_transform(embeddings, mask, n_pca, class_name):
    """Fit PCA whitening on embeddings[mask]. Returns (mu, W)."""
    ref = embeddings[mask]
    print(f"  Fitting PCA whitening on {mask.sum()} {class_name} events (dim={ref.shape[1]})...", flush=True)
    mu = ref.mean(axis=0)
    centered = ref - mu
    cov = (centered.T @ centered) / ref.shape[0]
    L, V = np.linalg.eigh(cov)
    if n_pca is not None:
        V = V[:, -n_pca:]
        L = L[-n_pca:]
        print(f"    Using top {n_pca} PCA components", flush=True)
    L = np.clip(L, 1e-6, None)
    W = V / np.sqrt(L)
    return mu, W


def compute_md_scores(model, pt_path, device, batch_size=512, n_pca=None, bkg_labels=None):
    """
    Embed all events, fit PCA whitening per background class, return MD scores.

    bkg_labels: list of class labels to use as reference.
      [1]       (default) → QCD-only MD.
      [0, 1, 3] → min-MD across DY, QCD, WJets (element-wise minimum).

    Returns (md [N], labels [N], mu_qcd, W_qcd, latents [N,D], class_transforms).
    class_transforms is a list of (label, mu, W) — reuse for signal inference.
    """
    _CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
    if bkg_labels is None:
        bkg_labels = [1]

    latents, labels = embed_pf(model, pt_path, device, batch_size)

    class_transforms = []
    for cls in bkg_labels:
        mask = (labels == cls)
        if mask.sum() < 10:
            print(f"  WARNING: class {cls} has only {mask.sum()} events — skipping", flush=True)
            continue
        mu, W = _fit_class_transform(latents, mask, n_pca, _CLASS_NAMES.get(cls, str(cls)))
        class_transforms.append((cls, mu, W))

    if not class_transforms:
        raise RuntimeError("No background classes with enough events.")

    md_per_class = []
    for cls, mu_c, W_c in class_transforms:
        z_c = (latents - mu_c) @ W_c
        md_per_class.append((z_c * z_c).sum(axis=1))
    md = np.stack(md_per_class, axis=0).min(axis=0).astype(np.float32)

    if len(class_transforms) > 1:
        print(f"  Min-MD across classes {[c for c,_,_ in class_transforms]}", flush=True)

    qcd_entry = next((t for t in class_transforms if t[0] == 1), class_transforms[0])
    mu_qcd, W_qcd = qcd_entry[1], qcd_entry[2]

    return md, labels, mu_qcd, W_qcd, latents, class_transforms


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

    # ── load models ───────────────────────────────────────────────────────────
    model, main_ckpt = load_nurd_model(config["ckpt"], device)
    ae_scaler = main_ckpt["ae_scaler"]
    ae = load_ae(config["ae_ckpt"], ae_scaler, device)

    # ── AE scores ─────────────────────────────────────────────────────────────
    print("Computing AE scores (bkg)...", flush=True)
    ae_bkg = compute_ae_scores(
        ae, ae_scaler, config["test_pt"], device,
        batch_size=config.get("ae_batch_size", 4096))

    # free AE GPU memory before running encoder
    del ae
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── contrastive MD scores ─────────────────────────────────────────────────
    bkg_labels = [0, 1, 3] if config.get("min_md") else [1]
    if config.get("min_md"):
        print("Min-MD mode: axis 2 = min(MD_DY, MD_QCD, MD_WJets)", flush=True)
    print("Computing contrastive MD scores (bkg)...", flush=True)
    con_bkg, labels, md_mu, md_W, latents_all, class_transforms = compute_md_scores(
        model, config["test_pt"], device,
        batch_size=config.get("batch_size", 512),
        n_pca=config.get("n_pca"),
        bkg_labels=bkg_labels,
    )

    if len(con_bkg) != len(ae_bkg):
        raise ValueError(f"Length mismatch: contrastive {len(con_bkg)} vs AE {len(ae_bkg)}")

    # ── mask ──────────────────────────────────────────────────────────────────
    mask = np.isfinite(ae_bkg) & np.isfinite(con_bkg) & (ae_bkg > 0)
    axis1_bkg = ae_bkg[mask]
    axis2_bkg = con_bkg[mask]
    labels_masked  = labels[mask]
    latents_masked = latents_all[mask]
    print(f"Events after masking: {mask.sum()}", flush=True)

    emb_pca  = (latents_masked - md_mu) @ md_W
    n_pca    = emb_pca.shape[1]
    axis2_pca = axis2_bkg

    qcd_only  = labels_masked == 1
    axis1_qcd = axis1_bkg[qcd_only]
    axis2_qcd = axis2_pca[qcd_only]
    print(f"QCD events for ABCD: {qcd_only.sum()}", flush=True)

    # ── signal (optional) ─────────────────────────────────────────────────────
    sig_axis1 = sig_axis2 = sig_axis2_pca = None
    sig_latents_masked = sig_emb_pca = None
    if config.get("signal_pt"):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print("Running signal inference...", flush=True)
        sig_latents, _ = embed_pf(
            model, config["signal_pt"], device,
            batch_size=config.get("batch_size", 512))
        sig_mds = []
        for cls, mu_c, W_c in class_transforms:
            z_c = (sig_latents - mu_c) @ W_c
            sig_mds.append((z_c * z_c).sum(axis=1))
        sig_con = np.stack(sig_mds, axis=0).min(axis=0).astype(np.float32)

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
        sig_axis2_pca     = (sig_emb_pca * sig_emb_pca).sum(axis=1).astype(np.float32)
        print(f"Signal events after masking: {sig_mask.sum()}", flush=True)

    # ── ABCD scan ─────────────────────────────────────────────────────────────
    percent = np.linspace(0.50, 0.98, 48)
    best    = {"nonclosure": np.inf}
    min_A   = int(config.get("min_A", 50))
    min_D   = int(config.get("min_D", 500))

    for p1 in percent:
        for p2 in percent:
            t1, t2, A, B, C, D = abcd_counts(axis1_qcd, axis2_qcd, p1, p2)
            if A < min_A or D < min_D:
                continue
            nc, A_hat = nonclosure_A(A, B, C, D)
            if np.isfinite(nc) and abs(nc) < abs(best["nonclosure"]):
                best.update(dict(p1=p1, p2=p2, t1=t1, t2=t2,
                                 A=A, B=B, C=C, D=D, A_hat=A_hat, nonclosure=nc))

    if "t1" not in best:
        raise RuntimeError("No ABCD working point found. Try lowering min_A/min_D.")

    t1_opt, t2_opt = best["t1"], best["t2"]
    print(f"Optimized: p1={best['p1']:.3f}, p2={best['p2']:.3f}", flush=True)
    print(f"Thresholds: t1={t1_opt:.4g}, t2={t2_opt:.4g}", flush=True)
    print(f"Nonclosure: {100.0*best['nonclosure']:.2f}%", flush=True)

    wandb.log({
        "ABCD/opt_p1":     best["p1"],
        "ABCD/opt_p2":     best["p2"],
        "ABCD/opt_t1":     float(t1_opt),
        "ABCD/opt_t2":     float(t2_opt),
        "ABCD/nonclosure": float(best["nonclosure"]),
        "ABCD/A": int(best["A"]), "ABCD/B": int(best["B"]),
        "ABCD/C": int(best["C"]), "ABCD/D": int(best["D"]),
    })

    # ── Plots ─────────────────────────────────────────────────────────────────
    fs, fs_leg, fs_legend = 28, 24, 16
    fig_size = (8, 6)

    class_names  = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
    class_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}

    # 2D histogram (all bkg)
    fig = plt.figure(figsize=(6, 5))
    xbins = np.geomspace(axis1_bkg[axis1_bkg > 0].min(), axis1_bkg.max(), 201)
    ybins = np.geomspace(axis2_bkg[axis2_bkg > 0].min(), axis2_bkg.max(), 201)
    plt.hist2d(axis1_bkg, axis2_bkg, bins=[xbins, ybins], norm=LogNorm(vmin=1), cmin=1)
    plt.xscale("log"); plt.yscale("log")
    plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
    plt.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("AE reco loss"); plt.ylabel("NURD Contrastive score (MD)")
    plt.title("AE vs NURD Contrastive (bkg only)"); plt.colorbar(label="Counts")
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
    ax.set_ylabel("NURD Contrastive score (MD)", fontsize=fs)
    ax.set_title("AE vs NURD Contrastive — all classes")
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
        plt.xlabel("AE reco loss", fontsize=fs)
        plt.ylabel("NURD Contrastive score (MD)", fontsize=fs)
        plt.title("AE vs NURD Contrastive — TpTp (signal)"); plt.colorbar(label="Counts")
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
        plt.ylabel("NURD Contrastive score (MD)", fontsize=fs)
        plt.title(f"AE vs NURD Contrastive — {name}"); plt.colorbar(label="Counts")
        out_cls = os.path.join(plot_dir, f"hist2d_{name}.png")
        plt.savefig(out_cls, dpi=200, bbox_inches="tight"); plt.close()
        wandb.log({f"Hists2D/{name}": wandb.Image(out_cls)})

    # PCA-MD scatter + KDE
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
        ax.set_ylabel("NURD Contrastive score (PCA-MD)", fontsize=fs)
        ax.set_title("AE vs PCA-MD — all classes (scatter)")
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
        ax.set_ylabel("NURD Contrastive score (PCA-MD)", fontsize=fs)
        ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("AE vs PCA-MD — KDE contours")
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
        pca2.fit(latents_masked[labels_masked == 1])
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
        ax.set_title("NURD latent — PCA scatter (fit on QCD)")
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
        fig.suptitle("PCA-MD space — pairwise components (NURD latent, fit on QCD)", fontsize=fs)
        plt.tight_layout()
        out_corner = os.path.join(plot_dir, "pca_corner.png")
        fig.savefig(out_corner, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"PCA/corner": wandb.Image(out_corner)})

    # Profile plots
    for (x_arr, y_arr, xlabel, ylabel, title, key) in [
        (axis2_bkg, axis1_bkg, "NURD Contrastive score (MD)", "Mean AE reco loss",
         "⟨AE loss⟩ vs NURD MD", "AE_vs_contrastive"),
        (axis1_bkg, axis2_bkg, "AE reco loss", "Mean NURD Contrastive score (MD)",
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
        (axis2_bkg, axis1_bkg, "NURD Contrastive score (MD)", "Mean AE reco loss",
         "⟨AE loss⟩ vs NURD MD (by class)", "AE_vs_contrastive_by_class"),
        (axis1_bkg, axis2_bkg, "AE reco loss", "Mean NURD Contrastive score (MD)",
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
    Ntot_bkg = float(len(axis1_qcd))

    for p in percent:
        t1, t2, A, B, C, D = abcd_counts(axis1_qcd, axis2_qcd, p, p)
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

    eff_opt   = best["A"] / max(Ntot_bkg, 1.0)
    ratio_opt = best["A_hat"] / max(best["A"], 1e-8)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(effs, closure_ratio, c="g", label="AE + NURD Contrastive (MD)")
    ax.fill_between(effs, closure_ratio - closure_unc, closure_ratio + closure_unc,
                    facecolor="g", alpha=0.5, interpolate=True)
    ax.plot(effs, np.ones_like(effs),       linestyle="-",  color="black")
    ax.plot(effs, np.full_like(effs, 0.95), linestyle="--", color="black")
    ax.plot(effs, np.full_like(effs, 1.05), linestyle="--", color="black")
    ax.plot([eff_opt], [ratio_opt], marker="o", c="red", label="Optimized")
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
            "p1":    float(best["p1"]),
            "p2":    float(best["p2"]),
            "n_pca": config.get("n_pca", None),
            "nonclosure": float(best["nonclosure"]),
        }, f, indent=2)
    print(f"Thresholds saved to: {thresholds_path}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True,
                        help="Path to NURD main checkpoint (checkpoint_main.pth.tar)")
    parser.add_argument("--ae_ckpt",      required=True,
                        help="Path to AE checkpoint (checkpoint_ae.pth)")
    parser.add_argument("--test_pt",      required=True,
                        help="Path to test .pt file (SM cocktail)")
    parser.add_argument("--signal_pt",    default=None,
                        help="Optional signal .pt file")
    parser.add_argument("--outdir",       default="outputs_abcd")
    parser.add_argument("--min_A",        type=int, default=50)
    parser.add_argument("--min_D",        type=int, default=500)
    parser.add_argument("--n_pca",        type=int, default=None,
                        help="Number of PCA components for MD (default: keep all latent dims)")
    parser.add_argument("--batch_size",   type=int, default=512,
                        help="Batch size for NURD encoder inference")
    parser.add_argument("--ae_batch_size", type=int, default=4096,
                        help="Batch size for AE inference")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_project",  default="AE vs. Contrastive ABCD",
                        help="W&B project to log to")
    parser.add_argument("--resume_run_id",  default=None,
                        help="Resume an existing W&B run (e.g. the training run from a sweep)")
    parser.add_argument("--skip_pca_md_plots",   action="store_true")
    parser.add_argument("--skip_embedding_pca",  action="store_true")
    parser.add_argument("--min_md",              action="store_true",
                        help="Use min-MD across DY+QCD+WJets (labels 0,1,3) instead of QCD-only MD")
    args = parser.parse_args()
    ABCD(vars(args))
