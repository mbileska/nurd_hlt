"""Held-out-style plots for a fixed physics-all AE-selected VALIDATION sample.

Preserves the source selection, ABCD cuts, models, and full-training-QCD MD
definition. Frozen PF inference is necessary for all-class KDE/PCA plots; no
networks are trained and no held-out background sample is opened.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde
from threadpoolctl import threadpool_limits
import torch

from scripts.audit_qcd_abcd_composition import infer_ae_scores, infer_latents
from scripts.evaluate_ae_tail_validation import select_tail
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


# Match eval_abcd_nurd.py's ordinary evaluation figures, not the two-panel audit.
FIG_SIZE = (8, 6)
HIST_SIZE = (6, 5)
SCAN_SIZE = (7.5, 6.5)
FS, TICK_FS, LEGEND_FS, DPI = 28, 24, 16, 200
PALETTE = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
           "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan")
MD_LABEL = "NURD Contrastive score (MD)"


def weight_ess(weights):
    total = float(weights.sum())
    sum2 = float(np.square(weights).sum())
    return total * total / sum2 if sum2 else 0.


def log_edges(values, count=201):
    values = np.asarray(values)
    good = values[np.isfinite(values) & (values > 0)]
    if not len(good):
        return np.geomspace(1e-12, 1, count)
    low, high = float(good.min()), float(good.max())
    if high <= low:
        low, high = low / 1.01, high * 1.01
    return np.geomspace(low, high, count)


def display_indices(labels, weights, seed=42, cap_per_class=5000):
    """Display-only weighted draws; never used for fitting or ABCD statistics."""
    rng = np.random.default_rng(seed)
    result = []
    for label in np.unique(labels):
        pool = np.flatnonzero((labels == label) & (weights > 0))
        if len(pool):
            result.append(rng.choice(pool, size=min(cap_per_class, len(pool)),
                                     replace=True, p=weights[pool] / weights[pool].sum()))
    return np.concatenate(result) if result else np.empty(0, dtype=np.int64)


def weighted_profile(x, y, weights, bins=60, logx=True, minimum_ess=20):
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if logx:
        good &= x > 0
    x, y, weights = x[good], y[good], weights[good]
    if not len(x):
        return np.array([]), np.array([]), np.array([])
    coordinates = np.log10(x) if logx else x
    low, high = float(coordinates.min()), float(coordinates.max())
    if high <= low:
        high = low + 1e-6
    edges = np.linspace(low, high, bins + 1)
    assigned = np.clip(np.searchsorted(edges, coordinates, side="right") - 1, 0, bins - 1)
    mass = np.bincount(assigned, weights=weights, minlength=bins)
    square_mass = np.bincount(assigned, weights=weights ** 2, minlength=bins)
    neff = np.divide(mass ** 2, square_mass, out=np.zeros(bins), where=square_mass > 0)
    means = np.divide(np.bincount(assigned, weights=weights * y, minlength=bins), mass,
                      out=np.zeros(bins), where=mass > 0)
    variance = np.divide(np.bincount(assigned, weights=weights * (y - means[assigned]) ** 2,
                                    minlength=bins), mass, out=np.zeros(bins), where=mass > 0)
    sem = np.sqrt(np.divide(variance, neff - 1, out=np.zeros(bins), where=neff > 1))
    centers = .5 * (edges[:-1] + edges[1:])
    if logx:
        centers = 10 ** centers
    keep = (neff >= minimum_ess) & np.isfinite(sem)
    return centers[keep], means[keep], sem[keep]


def weighted_kde(x, y, weights, grid_x, grid_y):
    """Exact original event weights: no weighted resample followed by weighting."""
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (weights > 0)
    if good.sum() < 3 or weight_ess(weights[good]) <= 2:
        return None
    try:
        estimator = gaussian_kde(np.vstack((np.log10(x[good]), np.log10(y[good]))),
                                 weights=weights[good])
        return estimator(np.vstack((grid_x.ravel(), grid_y.ravel()))).reshape(grid_x.shape)
    except (np.linalg.LinAlgError, ValueError):
        return None


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def decorate_score_axes(ax, point, title, large=True, large_labels=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    label_size = FS if (large if large_labels is None else large_labels) else None
    ax.set_xlabel("AE reco loss", fontsize=label_size)
    ax.set_ylabel(MD_LABEL, fontsize=label_size)
    ax.set_title(title)
    if point is not None:
        ax.axvline(point["ae_cut"], color="black", ls="--", lw=1)
        if point["md_cut"] > 0:
            ax.axhline(point["md_cut"], color="black", ls="--", lw=1)
    if large:
        ax.tick_params(axis="both", labelsize=TICK_FS)


def render_suite(data, names, report, grid, reference, output, signal=None, kde_grid=200):
    context = report.get("plot_context", {})
    population = context.get("population", "AE-selected validation")
    point_label = context.get("point_label", "Fixed conditional threshold")
    output.mkdir(parents=True, exist_ok=False)
    ae, md, labels, weights, latent = (data[k] for k in ("ae", "md", "label", "weight", "latent"))
    original_plot_rows = len(ae)
    good = np.isfinite(ae) & np.isfinite(md) & (ae > 0) & (md > 0) & (weights > 0)
    ae, md, labels, weights, latent = (x[good] for x in (ae, md, labels, weights, latent))
    if not len(ae):
        raise ValueError("No finite positive selected validation scores to plot")
    colors = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(sorted(names))}
    point = report["scans"]["tail"]["working_point"]
    scatter_rows = display_indices(labels, weights)
    signal_rows = None
    if signal is not None and len(signal["ae"]):
        signal_rows = display_indices(np.zeros(len(signal["ae"]), int), signal["weight"])
    else:
        signal = None

    # Standalone closure grid, exactly the ordinary evaluator's canvas/scale.
    percentiles = np.asarray(grid["percentiles"])
    fig, ax = plt.subplots(figsize=SCAN_SIZE)
    values = np.clip(grid["tail_valid_grid"] * 100, 0, 100)
    finite = values[np.isfinite(values)]
    vmax = max(float(np.percentile(finite, 95)), 1e-6) if finite.size else 100.
    mesh = ax.pcolormesh(percentiles, percentiles, values.T, cmap="viridis_r",
                         vmin=0, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax).set_label("|Non-closure| (%)", fontsize=TICK_FS)
    if point is not None:
        nc = point["absolute_nonclosure"]
        value = "undefined" if nc is None else f"{100 * nc:.2f}%"
        position = point.get("plot_percentiles", [.5, .5])
        ax.scatter([position[0]], [position[1]], marker="*" if point["statistically_valid"] else "X", s=400,
                   color="red", edgecolor="black", zorder=5,
                   label=f"{context.get('scan_point_label', 'Fixed conditional median/median')}\n|nonclosure|={value}\n"
                         + ("valid statistics" if point["statistically_valid"] else "insufficient statistics"))
        ax.legend(loc="lower left", fontsize=12)
    if not finite.size:
        ax.text(.5, .65, "No statistically valid grid cells", ha="center", transform=ax.transAxes)
    ax.set_xlabel(context.get("scan_xlabel", "Conditional percentile, axis 1 (AE reco loss)"), fontsize=TICK_FS)
    ax.set_ylabel(context.get("scan_ylabel", "Conditional percentile, axis 2 (NURD MD)"), fontsize=TICK_FS)
    ax.set_title(context.get("scan_title", "ABCD closure scan — AE-selected validation"), fontsize=TICK_FS)
    fig.tight_layout()
    save(fig, output / "closure_scan_2d.png")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    curve = report["scans"]["tail"]["curve"]
    for valid, color, legend in ((True, "g", "AE + NURD — valid statistics"),
                                 (False, "0.65", "Insufficient statistics (diagnostic only)")):
        points = sorted([p for p in curve if p["valid"] == valid and p["ratio"] is not None
                         and p["conditional_A_efficiency"] > 0], key=lambda p: p["conditional_A_efficiency"])
        if points:
            x = np.array([p["conditional_A_efficiency"] for p in points])
            y = np.array([p["ratio"] for p in points])
            error = np.array([p["uncertainty"] for p in points])
            ax.plot(x, y, color=color, ls="-" if valid else "none", marker=None if valid else ".", label=legend)
            if valid:
                ax.fill_between(x, y - error, y + error, color="g", alpha=.5)
    for value, style in ((1., "-"), (.95, "--"), (1.05, "--")):
        ax.axhline(value, color="black", ls=style)
    if point is not None and point["predicted_over_observed"] is not None:
        mass = sum(p["yield"] for p in point["regions"].values())
        efficiency = point["regions"]["A"]["yield"] / mass
        if efficiency > 0:
            ax.plot([efficiency], [point["predicted_over_observed"]], c="red",
                    marker="o" if point["statistically_valid"] else "X", label=point_label)
    efficiencies = [p["conditional_A_efficiency"] for p in curve if p["conditional_A_efficiency"] > 0]
    ax.set(xscale="log", xlim=(max(min(efficiencies) * .8, 1e-12) if efficiencies else 1e-5, 1), ylim=(0, 1.5))
    ax.set_xlabel(context.get("efficiency_xlabel", "Selection Efficiency (bkg A/Ntot in tail)"), fontsize=FS)
    ax.set_ylabel("Predicted Bkg. / True Bkg.", fontsize=FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(fontsize=12)
    save(fig, output / "cut_and_count_bkg_check.png")

    def histogram(x, y, w, title, path, unit_signal=False, large_labels=False):
        good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (w > 0)
        fig, ax = plt.subplots(figsize=HIST_SIZE)
        if good.any():
            values, edges1, edges2 = np.histogram2d(x[good], y[good],
                bins=(log_edges(x[good]), log_edges(y[good])), weights=w[good])
            positive = values[values > 0]
            lo, hi = float(positive.min()), float(positive.max())
            mesh = ax.pcolormesh(edges1, edges2, np.ma.masked_less_equal(values.T, 0),
                                 norm=LogNorm(vmin=lo, vmax=max(hi, lo * 1.001)), shading="auto")
            fig.colorbar(mesh, ax=ax, label="Unit-weight signal rows" if unit_signal else "Physics-weighted yield (log color)")
        else:
            ax.text(.5, .5, "No surviving positive-score events", ha="center", transform=ax.transAxes)
        decorate_score_axes(ax, point, title, large=False, large_labels=large_labels)
        save(fig, path)

    histogram(ae, md, weights, f"AE vs NURD — {population} backgrounds", output / "hist2d_bkg.png")
    for label, name in names.items():
        mask = labels == label
        histogram(ae[mask], md[mask], weights[mask], f"AE vs NURD — {name}", output / f"hist2d_{name}.png",
                  large_labels=True)
    if signal is not None:
        histogram(signal["ae"], signal["md"], signal["weight"], "AE vs NURD — selected TpTp benchmark",
                  output / "hist2d_TpTp.png", unit_signal=True, large_labels=True)

    for size, filename, large in ((HIST_SIZE, "hist2d_by_class_combined.png", False),
                                   (FIG_SIZE, "hist2d_pca_md_scatter.png", True)):
        fig, ax = plt.subplots(figsize=size)
        for label, name in names.items():
            chosen = scatter_rows[labels[scatter_rows] == label]
            ax.scatter(ae[chosen], md[chosen], s=.3, alpha=.15, color=colors[label], label=name, rasterized=True)
        if signal is not None:
            ax.scatter(signal["ae"][signal_rows], signal["md"][signal_rows],
                       s=.5, alpha=.4, color="tab:purple", label="TpTp", rasterized=True)
        decorate_score_axes(ax, point, f"AE vs PCA-MD — {population} (weighted display)",
                            large, large_labels=True)
        ax.legend(markerscale=10, fontsize=LEGEND_FS)
        save(fig, output / filename)

    # Same log-score grid and contour heights as the standard evaluation; the
    # KDE itself is fitted using original physics weights on ALL surviving rows.
    extra_x = signal["ae"] if signal is not None else np.array([])
    extra_y = signal["md"] if signal is not None else np.array([])
    xedges, yedges = log_edges(np.r_[ae, extra_x], kde_grid), log_edges(np.r_[md, extra_y], kde_grid)
    gx, gy = np.meshgrid(np.log10(xedges), np.log10(yedges))
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    handles, skipped = [], []
    for label, name in [*names.items(), *(([(-1, "TpTp")]) if signal is not None else [])]:
        if label == -1:
            x, y, w, color = signal["ae"], signal["md"], signal["weight"], "tab:purple"
        else:
            mask = labels == label
            x, y, w, color = ae[mask], md[mask], weights[mask], colors[label]
        density = weighted_kde(x, y, w, gx, gy)
        if density is None or not np.isfinite(density).all() or density.max() <= 0:
            skipped.append(name)
            continue
        ax.contour(10 ** gx, 10 ** gy, density,
                   levels=density.max() * np.array([.05, .15, .3, .5, .7, .88]),
                   colors=color, alpha=.7, linewidths=1.5)
        handles.append(Line2D([0], [0], color=color, lw=1.5, label=name))
    decorate_score_axes(ax, point, "AE vs PCA-MD — physics-weighted KDE shapes")
    if handles:
        ax.legend(handles=handles, fontsize=LEGEND_FS)
    ax.grid(alpha=.3)
    save(fig, output / "hist2d_pca_md_kde.png")

    centered = latent - reference["mean"]
    basis = reference["whitening"] / np.linalg.norm(reference["whitening"], axis=0)
    pca2 = centered @ basis[:, -2:][:, ::-1]
    white = centered @ reference["whitening"]
    sig_pca = sig_white = None
    if signal is not None:
        sig_centered = signal["latent"] - reference["mean"]
        sig_pca = sig_centered @ basis[:, -2:][:, ::-1]
        sig_white = sig_centered @ reference["whitening"]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for label, name in names.items():
        chosen = scatter_rows[labels[scatter_rows] == label]
        ax.scatter(pca2[chosen, 0], pca2[chosen, 1], s=.5, alpha=.12,
                   color=colors[label], label=name, rasterized=True)
    if signal is not None:
        ax.scatter(sig_pca[signal_rows, 0], sig_pca[signal_rows, 1], s=.5, alpha=.4,
                   color="tab:purple", label="TpTp", rasterized=True)
    ax.set_xlabel("PCA Component 1", fontsize=FS)
    ax.set_ylabel("PCA Component 2", fontsize=FS)
    ax.set_title("NURD latent — PCA (physics-weighted training QCD fit)")
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(markerscale=10, fontsize=LEGEND_FS)
    save(fig, output / "pca_scatter_embeddings.png")

    pairs = list(itertools.combinations(range(white.shape[1]), 2))
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)
        for ax, (first, second) in zip(axes.flat, pairs):
            for label, name in names.items():
                chosen = scatter_rows[labels[scatter_rows] == label]
                ax.scatter(white[chosen, first], white[chosen, second], s=.3, alpha=.12,
                           color=colors[label], label=name, rasterized=True)
            if signal is not None:
                ax.scatter(sig_white[signal_rows, first], sig_white[signal_rows, second],
                           s=.5, alpha=.4, color="tab:purple", label="TpTp", rasterized=True)
            ax.set_xlabel(f"PCA Component {first + 1}", fontsize=FS)
            ax.set_ylabel(f"PCA Component {second + 1}", fontsize=FS)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.legend(markerscale=10, fontsize=LEGEND_FS)
        fig.suptitle(f"PCA-MD space — {population}; fit on physics-weighted training QCD", fontsize=FS)
        fig.tight_layout()
        save(fig, output / "pca_corner.png")

    for x, y, xlabel, ylabel, key in (
        (md, ae, MD_LABEL, "Mean AE reco loss", "AE_vs_contrastive"),
        (ae, md, "AE reco loss", f"Mean {MD_LABEL}", "contrastive_vs_AE"),
    ):
        for by_class in (False, True):
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            count = 0
            for label, name in names.items() if by_class else [(None, "Physics-weighted mean ± SE")]:
                mask = labels == label if by_class else np.ones(len(labels), bool)
                centers, means, errors = weighted_profile(x[mask], y[mask], weights[mask], bins=40 if by_class else 60)
                if len(centers):
                    ax.errorbar(centers, means, yerr=errors, fmt="o", ms=3, lw=1, capsize=2,
                                color=colors[label] if by_class else None, label=name)
                    count += 1
            ax.set_xscale("log")
            ax.set_xlim(log_edges(x, 2))
            ax.set_xlabel(xlabel, fontsize=FS)
            ax.set_ylabel(ylabel, fontsize=FS)
            ax.set_title(f"Physics-weighted {population} profile" + (" by class" if by_class else ""))
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.grid(alpha=.3)
            if count:
                ax.legend(fontsize=LEGEND_FS)
            else:
                ax.text(.5, .5, "No bins meet effective-count requirement", ha="center", transform=ax.transAxes)
            save(fig, output / f"profile_{key}{'_by_class' if by_class else ''}.png")
    return {"kde_skipped_classes": skipped,
            "positive_score_plot_rows": len(ae),
            "nonpositive_or_zero_weight_rows_omitted_from_log_plots": original_plot_rows - len(ae),
            "plot_style": {"score_figure_inches": FIG_SIZE, "hist_figure_inches": HIST_SIZE,
                           "closure_grid_inches": SCAN_SIZE, "font_size": FS,
                           "tick_font_size": TICK_FS, "legend_font_size": LEGEND_FS, "dpi": DPI},
            "scatter_protocol": "fixed_seed_physics_weighted_within_class_draws_for_display_only",
            "kde_protocol": "all_selected_rows_original_physics_weighted_per_class_log_score_shapes",
            "profile_protocol": "physics_weighted_mean_and_SE_minimum_bin_ESS_20",
            "PCA_reference": "unchanged_physics_weighted_full_training_QCD_reference",
            "closure_display_note": "standard axis limits: color capped at 100% and ratio y-range 0..1.5; source numbers are uncapped"}


def load_verified_selection(tail_dir, raw, physics, validation, labels, report):
    with np.load(tail_dir / "validation_ae_selection.npz", allow_pickle=False) as archive:
        result = {key: np.array(archive[key]) for key in archive.files}
    if (not np.array_equal(result["source_row_index"], validation)
            or not np.array_equal(result["label"], labels[validation])
            or not np.array_equal(result["event_id"], raw["event_id"][validation].numpy())
            or not np.allclose(result["physics_weight"], physics[validation], rtol=1e-12, atol=0)):
        raise ValueError("Saved AE selection is not aligned with the checkpoint's validation rows/weights")
    if report["selection"]["measure"] != "physics-all":
        raise ValueError("This plot suite requires the physics-all mixture tail, not a raw-MC tail")
    expected, selection = select_tail(result["ae_score"], result["label"], result["physics_weight"], validation,
                                      report["selection"]["requested_tail_fraction"], "physics-all")
    if (not np.array_equal(result["selected"], expected)
            or not np.isclose(selection["ae_boundary"], report["selection"]["ae_boundary"], rtol=1e-12)):
        raise ValueError("Saved selected rows differ from the fixed physics-mixture AE selection")
    result["selected"] = result["selected"].astype(bool)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--signal", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and thread count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose cpu or request a GPU job")
    from eval_abcd_nurd import _fit_class_transform, checkpoint_reference_indices, load_ae, load_nurd_model
    report = json.loads((args.tail_dir / "tail_summary.json").read_text())
    if report.get("selection", {}).get("measure") != "physics-all":
        raise ValueError("Choose a completed physics-all tail evaluation, not a raw-MC or QCD-only selection")
    checkpoint_path = Path(report["checkpoint"])
    if file_sha256(checkpoint_path) != report["checkpoint_sha256"]:
        raise ValueError("Model checkpoint changed since the tail selection")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, metadata = load_nurd_model(str(checkpoint_path), device)
    # AE path comes from the verified original cache report, never a new fit.
    cache_report = json.loads(Path(report["cache_report"]).read_text())
    ae_path = Path(cache_report["ae_checkpoint"])
    if (file_sha256(ae_path) != report["ae_checkpoint_sha256"]
            or metadata.get("ae_checkpoint_sha256") != report["ae_checkpoint_sha256"]):
        raise ValueError("AE checkpoint differs from the original model and tail selection")
    raw = load_hlt_sample(report["training_manifest"])
    physics, weight_metadata = load_generator_weights(report["training_weights"], raw["label"], qcd_label=1, sample=raw)
    if (weight_metadata["weight_scope"] != "all_events"
            or weight_metadata["effective_physics_weight_sha256"] != report["evaluation_weighting"]["physics_weight_sha256"]):
        raise ValueError("The all-event physics measure changed since tail selection")
    physics = physics.numpy().astype(np.float64)
    train, validation = checkpoint_reference_indices(metadata, sample_signature(raw), weight_metadata)
    labels = checkpoint_target_labels(raw, metadata["preprocessing"]).numpy()
    selected = load_verified_selection(args.tail_dir, raw, physics, validation, labels, report)
    fit_qcd = train[labels[train] == 1]
    selected_rows = validation[selected["selected"]]
    print(f"Frozen inference: full training QCD reference={len(fit_qcd):,}, selected validation={len(selected_rows):,}; device={device}", flush=True)
    with threadpool_limits(limits=args.threads):
        train_latent = infer_latents(model, raw, fit_qcd, device, args.batch_size)
        mean, whitening = _fit_class_transform(train_latent, np.ones(len(fit_qcd), bool), 6,
                                               "unchanged full training QCD", weights=physics[fit_qcd])
        del train_latent
        latent = infer_latents(model, raw, selected_rows, device, args.batch_size)
    md = np.square((latent - mean) @ whitening).sum(axis=1)
    with np.load(args.tail_dir / "validation_qcd_scores.npz", allow_pickle=False) as archive:
        cached_rows, cached_md = archive["source_row_index"], archive["raw_md"]
        order = np.argsort(cached_rows)
        qcd_selected = labels[selected_rows] == 1
        positions = np.searchsorted(cached_rows[order], selected_rows[qcd_selected])
        if (np.any(positions >= len(order))
                or not np.array_equal(cached_rows[order][positions], selected_rows[qcd_selected])):
            raise ValueError("Selected QCD rows are absent from the original score cache")
        expected = cached_md[order][positions]
    if not np.allclose(md[qcd_selected], expected, rtol=2e-3, atol=1e-5):
        raise ValueError("Rebuilt training-QCD MD does not match original cached QCD scores")
    # Keep QCD scores bit-for-bit consistent with the source closure masks.
    md[qcd_selected] = expected
    data = {"ae": selected["ae_score"][selected["selected"]], "md": md,
            "label": labels[selected_rows], "weight": physics[selected_rows], "latent": latent}
    signal = None
    signal_retention = None
    if args.signal is not None:
        raw_signal = load_hlt_sample(args.signal)
        ae = load_ae(str(ae_path), metadata["ae_scaler"], device)
        source = np.arange(len(raw_signal["pf"]))
        signal_ae = infer_ae_scores(ae, metadata["ae_scaler"], raw_signal, source, device, 4096)
        keep = np.isfinite(signal_ae) & (signal_ae >= report["selection"]["ae_boundary"])
        signal_retention = {"rows_before": len(source), "rows_after": int(keep.sum()),
                            "raw_efficiency": float(keep.mean()), "normalization": "unit-weight benchmark, not part of the SM mixture"}
        if keep.any():
            signal_latent = infer_latents(model, raw_signal, source[keep], device, args.batch_size)
            signal = {"ae": signal_ae[keep], "md": np.square((signal_latent - mean) @ whitening).sum(axis=1),
                      "latent": signal_latent, "weight": np.ones(int(keep.sum()))}
    args.output_dir.mkdir(parents=True)
    with np.load(args.tail_dir / "closure_scan_arrays.npz", allow_pickle=False) as archive:
        grid = {key: np.array(archive[key]) for key in archive.files}
    names = {int(k): v for k, v in metadata["preprocessing"]["target_class_names"].items()}
    style = render_suite(data, names, report, grid, {"mean": mean, "whitening": whitening},
                          args.output_dir / "plots", signal=signal)
    np.savez_compressed(args.output_dir / "selected_validation_scores.npz", source_row_index=selected_rows,
                        ae_score=data["ae"], md_score=md, latent=latent, label=data["label"], physics_weight=data["weight"],
                        event_id=raw["event_id"][selected_rows].numpy(),
                        background_category=raw["background_category"][selected_rows].numpy())
    np.savez_compressed(args.output_dir / "training_qcd_reference.npz", mean=mean, whitening=whitening)
    if signal is not None:
        np.savez_compressed(args.output_dir / "selected_signal_scores.npz", **signal)
    result = {"protocol": "heldout_style_plots_on_fixed_physics_all_AE_tail_validation",
              "heldout_background_accessed": False, "networks_updated": False,
              "source_tail_dir": str(args.tail_dir), "checkpoint_sha256": report["checkpoint_sha256"],
              "AE_checkpoint_sha256": report["ae_checkpoint_sha256"],
              "physics_weight_sha256": weight_metadata["effective_physics_weight_sha256"],
              "selected_validation_rows": len(selected_rows), "device": args.device,
              "class_names": names, "source_tail_report_sha256": file_sha256(args.tail_dir / "tail_summary.json"),
              "MD_cache_agreement": True, "selection": report["selection"],
              "working_point": report["scans"]["tail"]["working_point"],
              "signal_retention": signal_retention, **style}
    (args.output_dir / "plot_suite_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"STANDARD-STYLE VALIDATION TAIL PLOTS DONE: {args.output_dir / 'plots'}", flush=True)


if __name__ == "__main__":
    main()
