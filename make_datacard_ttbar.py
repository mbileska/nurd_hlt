"""
Produces a CMS Combine ABCD datacard for a ttbar signal search.
Axis 1: AE reco loss. Axis 2: Mahalanobis distance in NURD latent space.
Signal: TT (label=2). Background: DY (0) + QCD (1) + WJets (3).
Writes <outdir>/datacard_ttbar.txt (ABCD), <outdir>/datacard_ttbar_cnt.txt (cut-and-count), and <outdir>/abcd_summary.json.
"""

import os
import gc
import json
import argparse
import numpy as np
import torch

from models.hlt_con import HLTContrastiveModel
from models.hlt_autoencoder import HLTAutoencoder


# ---------------------------------------------------------------------------
# Model loading  (identical to eval_abcd_nurd.py)
# ---------------------------------------------------------------------------

def load_nurd_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    sd   = ckpt["state_dict_model"]

    e_proj_key = next((k for k in sd if k.endswith(".attn.e.weight")), None)
    num_tokens  = sd[e_proj_key].shape[1] - 1 if e_proj_key else cfg.get("linear_dim", 100)
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
    print(f"Loaded NURD model: latent_dim={cfg['latent_dim']}, n_layers={cfg['num_layers']}", flush=True)
    return model, ckpt


def load_ae(ae_ckpt_path, ae_scaler, device):
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
    print(f"Loaded AE: features={ae_cfg['features']}, latent={ae_cfg['latent_dim']}", flush=True)
    return ae


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def compute_ae_scores(ae, ae_scaler, pt_path_or_dict, device, batch_size=4096):
    """Returns (ae_scores [N], labels [N])."""
    mu = ae_scaler["mu"].detach().cpu().float()
    std = ae_scaler["std"].detach().cpu().float().clamp(min=1e-8)

    raw = torch.load(pt_path_or_dict, map_location="cpu") if isinstance(pt_path_or_dict, str) else pt_path_or_dict
    labels = raw["label"].numpy()
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
    if isinstance(pt_path_or_dict, str):
        del raw
    return torch.cat(scores).numpy().astype(np.float32), labels


def embed_pf(model, pt_path_or_dict, device, batch_size=512):
    """Returns (latents [N, D], labels [N])."""
    raw = torch.load(pt_path_or_dict, map_location="cpu") if isinstance(pt_path_or_dict, str) else pt_path_or_dict
    pf     = torch.nan_to_num(raw["pf"], nan=0.0, posinf=0.0, neginf=0.0)
    labels = raw["label"].numpy()
    N = pf.shape[0]
    print(f"  Encoder inference on {N} events...", flush=True)

    latents = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = pf[i0:i0 + batch_size].to(device)
            latent, _ = model(xb)
            latents.append(latent.cpu())
    return torch.cat(latents, dim=0).numpy(), labels


def compute_md_scores(latents, labels, n_pca=None, bkg_labels=None):
    """
    Fit PCA-whitened Gaussian to bkg_labels events, compute MD for all events.
    Returns md [N].
    """
    if bkg_labels is None:
        bkg_labels = [1]  # QCD by default

    _CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
    class_transforms = []
    for cls in bkg_labels:
        mask = labels == cls
        if mask.sum() < 10:
            print(f"  WARNING: class {cls} has only {mask.sum()} events — skipping", flush=True)
            continue
        ref = latents[mask]
        mu  = ref.mean(axis=0)
        centered = ref - mu
        cov = (centered.T @ centered) / ref.shape[0]
        L, V = np.linalg.eigh(cov)
        if n_pca is not None:
            V = V[:, -n_pca:]
            L = L[-n_pca:]
        L = np.clip(L, 1e-6, None)
        W = V / np.sqrt(L)
        class_transforms.append((cls, mu, W))
        print(f"  Fit MD transform on {mask.sum()} {_CLASS_NAMES.get(cls, str(cls))} events", flush=True)

    md_per_class = []
    for cls, mu_c, W_c in class_transforms:
        z_c = (latents - mu_c) @ W_c
        md_per_class.append((z_c * z_c).sum(axis=1))
    return np.stack(md_per_class, axis=0).min(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# ABCD helpers
# ---------------------------------------------------------------------------

def abcd_counts(ax1, ax2, t1, t2):
    """Returns dict {A, B, C, D} with float event counts."""
    return {
        "A": float(((ax1 > t1) & (ax2 > t2)).sum()),
        "B": float(((ax1 > t1) & (ax2 <= t2)).sum()),
        "C": float(((ax1 <= t1) & (ax2 > t2)).sum()),
        "D": float(((ax1 <= t1) & (ax2 <= t2)).sum()),
    }


def find_best_abcd_wp(ae_qcd, md_qcd, min_A=10, min_D=100, n_scan=48):
    """Scan percentile pairs and return (t1, t2, nonclosure) with minimum |nonclosure|."""
    percent = np.linspace(0.50, 0.98, n_scan)
    best = {"nonclosure": np.inf}
    for p1 in percent:
        for p2 in percent:
            t1_ = float(np.quantile(ae_qcd, p1))
            t2_ = float(np.quantile(md_qcd, p2))
            A = int(((ae_qcd > t1_) & (md_qcd > t2_)).sum())
            B = int(((ae_qcd > t1_) & (md_qcd <= t2_)).sum())
            C = int(((ae_qcd <= t1_) & (md_qcd > t2_)).sum())
            D = int(((ae_qcd <= t1_) & (md_qcd <= t2_)).sum())
            if A < min_A or D < min_D:
                continue
            A_hat = (B * C) / max(D, 1e-8)
            nc    = A_hat / max(A, 1e-8) - 1.0
            if np.isfinite(nc) and abs(nc) < abs(best["nonclosure"]):
                best.update({"nonclosure": nc, "t1": t1_, "t2": t2_,
                             "p1": p1, "p2": p2, "A": A, "B": B, "C": C, "D": D})
    if "t1" not in best:
        raise RuntimeError("No ABCD working point found. Try lowering --min_A / --min_D.")
    return best


# ---------------------------------------------------------------------------
# Datacard writer
# ---------------------------------------------------------------------------

def write_datacard(
    path,
    bins_sig,
    bins_bkg,
    bins_obs,
    nonclosure,
    signal_name="tt",
    bkg_name="bkg",
):
    N_bkg_B = max(bins_bkg["B"], 1e-3)
    N_bkg_C = max(bins_bkg["C"], 1e-3)
    N_bkg_D = max(bins_bkg["D"], 1e-3)

    sig_A = bins_sig["A"]
    sig_B = bins_sig["B"]
    sig_C = bins_sig["C"]
    sig_D = bins_sig["D"]

    obs_A = int(round(bins_obs["A"]))
    obs_B = int(round(bins_obs["B"]))
    obs_C = int(round(bins_obs["C"]))
    obs_D = int(round(bins_obs["D"]))

    def fmt(x): return f"{x:.4f}"

    lines = [
        "imax 4",
        "jmax 1",
        "kmax 0",
        "",
        60 * "-",
        "",
        f"bin          A        B        C        D",
        f"observation  {obs_A:<8d} {obs_B:<8d} {obs_C:<8d} {obs_D:<8d}",
        "",
        60 * "-",
        "",
        f"bin      {'A':<10} {'B':<10} {'C':<10} {'D':<10} {'A':<10} {'B':<10} {'C':<10} {'D':<10}",
        f"process  {signal_name:<10} {signal_name:<10} {signal_name:<10} {signal_name:<10} "
        f"{bkg_name:<10} {bkg_name:<10} {bkg_name:<10} {bkg_name:<10}",
        f"process  {'0':<10} {'0':<10} {'0':<10} {'0':<10} "
        f"{'1':<10} {'1':<10} {'1':<10} {'1':<10}",
        f"rate     {fmt(sig_A):<10} {fmt(sig_B):<10} {fmt(sig_C):<10} {fmt(sig_D):<10} "
        f"{'1':<10} {'1':<10} {'1':<10} {'1':<10}",
        "",
        60 * "-",
        "",
        f"r_B   rateParam  B  {bkg_name}  {fmt(N_bkg_B)}",
        f"r_C   rateParam  C  {bkg_name}  {fmt(N_bkg_C)}",
        f"r_D   rateParam  D  {bkg_name}  {fmt(N_bkg_D)}",
        f"r_A   rateParam  A  {bkg_name}  @0*@1/@2  r_B,r_C,r_D",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Datacard written to: {path}")


def write_datacard_cnt(path, sig_A, bkg_A, obs_A, signal_name="tt", bkg_name="bkg"):
    """Single-bin cut-and-count datacard for signal region A. Background from MC directly."""
    def fmt(x): return f"{x:.4f}"

    lines = [
        "imax 1",
        "jmax 1",
        "kmax 0",
        "",
        60 * "-",
        "",
        f"bin          A",
        f"observation  {int(round(obs_A))}",
        "",
        60 * "-",
        "",
        f"bin      {'A':<10} {'A':<10}",
        f"process  {signal_name:<10} {bkg_name:<10}",
        f"process  {'0':<10} {'1':<10}",
        f"rate     {fmt(sig_A):<10} {fmt(bkg_A):<10}",
        "",
        60 * "-",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Cut-and-count datacard written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Make CMS Combine ABCD datacard for ttbar search")
    parser.add_argument("--ckpt",       required=True, help="NURD main checkpoint (.pth.tar)")
    parser.add_argument("--ae_ckpt",    required=True, help="AE checkpoint (.pth)")
    parser.add_argument("--test_pt",    required=True, help="SM cocktail test file (.pt)")
    parser.add_argument("--eval_json",  default=None,
                        help="Path to abcd_thresholds.json from eval_abcd_nurd.py. "
                             "If provided, t1/t2/n_pca are read from this file and the scan is skipped.")
    parser.add_argument("--allow_internal_scan", action="store_true",
                        help="Allow this script to scan thresholds internally. Prefer --eval_json so the "
                             "datacard uses the exact held-out thresholds from eval_abcd_nurd.py.")
    parser.add_argument("--n_pca",      type=int,   default=None,  help="PCA dims for MD (default: all, overridden by --eval_json)")
    parser.add_argument("--p1",         type=float, default=None,  help="AE-loss percentile threshold (skips scan, ignored if --eval_json set)")
    parser.add_argument("--p2",         type=float, default=None,  help="MD percentile threshold (skips scan, ignored if --eval_json set)")
    parser.add_argument("--tt_label",          type=int,   default=2,     help="Class label for TT (default: 2)")
    parser.add_argument("--bkg_process_labels", type=int, nargs="+", default=[1],
                        help="Which labels count as background in ABCD (default: 1=QCD only). "
                             "Events with other labels are excluded from the analysis entirely. "
                             "Example: --bkg_process_labels 0 1 3  includes DY+QCD+WJets.")
    parser.add_argument("--bkg_labels", type=int, nargs="+", default=[1],
                        help="Labels to use when fitting the MD Gaussian (default: 1=QCD). "
                             "Should be a subset of --bkg_process_labels.")
    parser.add_argument("--min_md",     action="store_true",
                        help="Use min-MD across bkg_labels instead of QCD-only")
    parser.add_argument("--min_A",      type=int,   default=10,    help="Min QCD events in A for WP scan")
    parser.add_argument("--min_D",      type=int,   default=100,   help="Min QCD events in D for WP scan")
    parser.add_argument("--outdir",     default="outputs_datacard", help="Output directory")
    parser.add_argument("--datacard_name", default="datacard_ttbar.txt")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── load thresholds from eval JSON if provided ────────────────────────────
    if args.eval_json is not None:
        with open(args.eval_json) as f:
            eval_thresholds = json.load(f)
        args.p1   = None  # not used — we have absolute thresholds
        args.p2   = None
        _t1_fixed = eval_thresholds["t1"]
        _t2_fixed = eval_thresholds["t2"]
        if args.n_pca is None:
            args.n_pca = eval_thresholds.get("n_pca", None)
        print(f"Loaded thresholds from {args.eval_json}: t1={_t1_fixed:.4g}, t2={_t2_fixed:.4g}, n_pca={args.n_pca}")
    else:
        _t1_fixed = _t2_fixed = None
        if (args.p1 is None) != (args.p2 is None):
            raise RuntimeError("Set both --p1 and --p2, or neither.")
        if args.p1 is None and args.p2 is None and not args.allow_internal_scan:
            raise RuntimeError(
                "No --eval_json was provided. Pass "
                "$OUTDIR/abcd_thresholds.json from eval_abcd_nurd.py, or set "
                "--allow_internal_scan to reproduce the older internal scan behavior."
            )

    # ── load models ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading models...")
    model, ckpt = load_nurd_model(args.ckpt, device)
    ae_scaler   = ckpt["ae_scaler"]
    ae          = load_ae(args.ae_ckpt, ae_scaler, device)

    # ── AE scores ─────────────────────────────────────────────────────────────
    print("\n[2/4] Computing AE scores...")
    ae_scores, labels = compute_ae_scores(ae, ae_scaler, args.test_pt, device)
    del ae; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

    # ── MD scores ─────────────────────────────────────────────────────────────
    print("\n[3/4] Computing NURD MD scores...")
    latents, _  = embed_pf(model, args.test_pt, device)
    bkg_labels  = ([0, 1, 3] if args.min_md else args.bkg_labels)
    md_scores   = compute_md_scores(latents, labels, n_pca=args.n_pca, bkg_labels=bkg_labels)

    # ── valid event mask ──────────────────────────────────────────────────────
    CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
    keep_labels = set(args.bkg_process_labels) | {args.tt_label}
    process_mask = np.isin(labels, list(keep_labels))
    mask = np.isfinite(ae_scores) & np.isfinite(md_scores) & (ae_scores > 0) & process_mask
    ae_scores = ae_scores[mask]
    md_scores = md_scores[mask]
    labels    = labels[mask]
    print(f"Valid events after masking (keeping labels {sorted(keep_labels)}): {mask.sum()}")

    # ── split by process ──────────────────────────────────────────────────────
    sig_mask = labels == args.tt_label
    bkg_mask = np.isin(labels, args.bkg_process_labels)
    bkg_process_desc = " + ".join(
        f"{CLASS_NAMES.get(l, str(l))} (label={l})"
        for l in sorted(args.bkg_process_labels)
    )
    print(f"\nProcess counts used in analysis:")
    for cls in sorted(keep_labels):
        name = CLASS_NAMES.get(cls, str(cls))
        role = "SIGNAL" if cls == args.tt_label else "background"
        print(f"  {name} (label={cls}): {(labels==cls).sum()}  [{role}]")

    ae_sig, md_sig = ae_scores[sig_mask], md_scores[sig_mask]
    ae_bkg, md_bkg = ae_scores[bkg_mask], md_scores[bkg_mask]

    # ── find ABCD thresholds ──────────────────────────────────────────────────
    print("\n[4/4] Finding ABCD working point...")
    qcd_mask_all = labels == 1
    if qcd_mask_all.sum() == 0:
        raise RuntimeError("No QCD events found after filtering. Check --bkg_process_labels.")

    if _t1_fixed is not None:
        # Use thresholds loaded from eval_abcd_nurd.py JSON — skip scan entirely
        t1, t2 = _t1_fixed, _t2_fixed
        nonclosure_qcd = None
        print(f"Using thresholds from eval JSON: t1={t1:.4g}, t2={t2:.4g}")
    elif args.p1 is not None and args.p2 is not None:
        t1 = float(np.quantile(ae_scores[qcd_mask_all], args.p1))
        t2 = float(np.quantile(md_scores[qcd_mask_all], args.p2))
        nonclosure_qcd = None
        print(f"Using specified percentiles: p1={args.p1}, p2={args.p2}")
        print(f"Thresholds: t1={t1:.4g}, t2={t2:.4g}")
    else:
        # Scan on QCD-only (same as eval_abcd_nurd.py)
        ae_qcd = ae_scores[qcd_mask_all]
        md_qcd = md_scores[qcd_mask_all]
        wp = find_best_abcd_wp(ae_qcd, md_qcd, min_A=args.min_A, min_D=args.min_D)
        t1, t2 = wp["t1"], wp["t2"]
        nonclosure_qcd = wp["nonclosure"]
        print(f"Best WP: p1={wp['p1']:.3f}, p2={wp['p2']:.3f}")
        print(f"Thresholds: t1={t1:.4g}, t2={t2:.4g}")
        print(f"QCD ABCD: A={wp['A']}  B={wp['B']}  C={wp['C']}  D={wp['D']}")
        print(f"QCD nonclosure: {100*nonclosure_qcd:.2f}%")

    # ── ABCD counts ──────────────────────────────────────────────────────────
    bins_sig = abcd_counts(ae_sig, md_sig, t1, t2)
    bins_bkg = abcd_counts(ae_bkg, md_bkg, t1, t2)
    bins_obs = abcd_counts(ae_scores, md_scores, t1, t2)  # all events (what you'd see in data)

    bkg_A_hat    = bins_bkg["B"] * bins_bkg["C"] / max(bins_bkg["D"], 1e-8)
    nonclosure_bkg = bkg_A_hat / max(bins_bkg["A"], 1e-8) - 1.0
    legacy_nonclosure_bkg = (bins_bkg["A"] - bkg_A_hat) / max(bkg_A_hat, 1e-8)

    print(f"\n{'='*55}")
    print(f"{'Region':<10} {'Signal(TT)':<14} {'Background':<14} {'Observed':<12}")
    print(f"{'-'*55}")
    for reg in ["A", "B", "C", "D"]:
        print(f"  {reg:<8} {bins_sig[reg]:<14.1f} {bins_bkg[reg]:<14.1f} {bins_obs[reg]:<12.0f}")
    print(f"{'='*55}")
    print(f"Background A predicted (B*C/D) : {bkg_A_hat:.2f}")
    print(f"Background A true (MC)         : {bins_bkg['A']:.2f}")
    print(f"Nonclosure (all bkg)           : {100*nonclosure_bkg:.2f}%")
    if nonclosure_qcd is not None:
        print(f"Nonclosure (QCD-only, at WP)   : {100*nonclosure_qcd:.2f}%")
    print(f"Signal (TT) in A               : {bins_sig['A']:.1f}")
    print(f"Signal / background in A       : {bins_sig['A']/max(bkg_A_hat,1e-8):.3f}")
    print()

    # Use the larger of the two nonclosure estimates as the systematic
    nc_for_card = max(abs(nonclosure_bkg), abs(nonclosure_qcd) if nonclosure_qcd else 0)

    # ── write datacards ───────────────────────────────────────────────────────
    card_path = os.path.join(args.outdir, args.datacard_name)
    write_datacard(
        card_path,
        bins_sig=bins_sig,
        bins_bkg=bins_bkg,
        bins_obs=bins_obs,
        nonclosure=nc_for_card,
    )

    cnt_name = args.datacard_name.replace(".txt", "_cnt.txt")
    write_datacard_cnt(
        os.path.join(args.outdir, cnt_name),
        sig_A=bins_sig["A"],
        bkg_A=bins_bkg["A"],
        obs_A=bins_obs["A"],
    )

    # ── save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "thresholds": {"t1": float(t1), "t2": float(t2)},
        "n_pca": args.n_pca,
        "bkg_process_labels": args.bkg_process_labels,
        "bkg_labels_for_md": bkg_labels,
        "tt_label": args.tt_label,
        "bins_signal": bins_sig,
        "bins_background": bins_bkg,
        "bins_observed": bins_obs,
        "bkg_A_predicted": float(bkg_A_hat),
        "nonclosure_bkg": float(nonclosure_bkg),
        "legacy_nonclosure_bkg": float(legacy_nonclosure_bkg),
        "nonclosure_qcd": float(nonclosure_qcd) if nonclosure_qcd is not None else None,
        "S_over_B_in_A": float(bins_sig["A"] / max(bkg_A_hat, 1e-8)),
    }
    json_path = os.path.join(args.outdir, "abcd_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
