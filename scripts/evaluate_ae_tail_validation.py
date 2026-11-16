"""CPU-only AE-tail ABCD diagnostic, reusing uncalibrated validation QCD MD.

The full training-QCD MD reference and both trained networks stay unchanged.
The default selects the highest-AE 10% of the PHYSICS-WEIGHTED mixture of ALL
backgrounds, using original all-event generator/cross-section weights before
the percentile calculation. Thresholds are rescanned inside the retained QCD.
This is exploratory validation, not a held-out result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from threadpoolctl import threadpool_limits
import torch

from models.hlt_autoencoder import HLTAutoencoder
from scripts.audit_qcd_abcd_composition import CATEGORY_NAMES, infer_ae_scores, region_codes, region_statistics
from scripts.audit_qcd_validation_hardness import weighted_quantile
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


def effective_count(weights):
    weights = np.asarray(weights, dtype=np.float64)
    square_sum = np.square(weights).sum()
    return float(weights.sum() ** 2 / square_sum) if square_sum > 0 else 0.


def select_tail(scores, labels, weights, source_rows, fraction=.1, measure="physics-all"):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    source_rows = np.asarray(source_rows)
    if (scores.ndim != 1 or not len(scores)
            or any(np.asarray(value).shape != scores.shape for value in (labels, weights, source_rows))):
        raise ValueError("AE scores, labels, weights, and source rows must be nonempty and aligned")
    if not 0 < fraction < 1 or not np.isfinite(scores).all():
        raise ValueError("Tail fraction must be in (0,1) and AE scores finite")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Original physics weights must be finite, nonnegative, and have positive mass")
    reference = np.ones(len(scores), dtype=bool)
    if measure == "raw-all":
        count = max(1, int(np.ceil(len(scores) * fraction)))
        # Rank ties by source row, never by MD or class label.
        order = np.lexsort((source_rows, -scores))
        selected = np.zeros(len(scores), dtype=bool)
        selected[order[:count]] = True
        cut = float(scores[order[count - 1]])
        definition = "highest_AE_validation_rows_all_classes_exact_count_source_row_tiebreak"
    elif measure in ("physics-all", "physics-qcd"):
        reference = (labels == 1) if measure == "physics-qcd" else reference
        if not reference.any() or weights[reference].sum() <= 0:
            raise ValueError("No positive reference mass for weighted AE selection")
        cut = float(weighted_quantile(scores[reference], [1 - fraction], weights[reference])[0])
        selected = scores >= cut
        definition = ("validation_all_background_generator_cross_section_weighted_AE_quantile_include_boundary_ties"
                      if measure == "physics-all" else
                      "validation_physics_weighted_QCD_AE_quantile_applied_to_all_classes_include_boundary_ties")
    else:
        raise ValueError(f"Unknown tail measure: {measure}")
    return selected, {"measure": measure, "definition": definition,
                      "requested_tail_fraction": fraction, "ae_boundary": cut,
                      "boundary_tie_rows": int((scores == cut).sum()),
                      "selected_rows": int(selected.sum()),
                      "selected_raw_fraction": float(selected.mean()),
                      "selected_physics_fraction_all_backgrounds": float(weights[selected].sum() / weights.sum()),
                      "reference_population": "QCD" if measure == "physics-qcd" else "all_backgrounds",
                      "reference_selected_physics_fraction": float(
                          weights[selected & reference].sum() / weights[reference].sum()),
                      "reference_boundary_physics_fraction": float(
                          weights[reference & (scores == cut)].sum() / weights[reference].sum()),
                      "event_features_or_AE_scores_reweighted": False}


def selection_histograms(scores, labels, weights, bins=80):
    """Keep physical mixture normalization separate from per-class shapes."""
    scores = np.maximum(np.asarray(scores, dtype=np.float64), 1e-12)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    if (not len(scores) or scores.shape != labels.shape or scores.shape != weights.shape
            or not np.isfinite(scores).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Histogram inputs must be aligned, finite, and have positive physical mass")
    # Cover ALL events; raw quantile clipping could hide physically important
    # low-score rows when only a few MC rows carry substantial rate weight.
    edges = np.geomspace(scores.min(), max(scores.max(), scores.min() * 1.01), bins + 1)
    classes = np.unique(labels)
    raw_shapes, physics_shapes, mixture_parts = [], [], []
    for label in classes:
        mask = labels == label
        counts = np.histogram(scores[mask], bins=edges)[0]
        mass = np.histogram(scores[mask], bins=edges, weights=weights[mask])[0]
        raw_shapes.append(counts / mask.sum())
        physics_shapes.append(mass / weights[mask].sum() if weights[mask].sum() else mass)
        mixture_parts.append(mass / weights.sum())
    contributions = np.asarray(mixture_parts)
    return {"bin_edges": edges, "class_labels": classes,
            "raw_per_class_fraction": np.asarray(raw_shapes),
            "physics_per_class_fraction": np.asarray(physics_shapes),
            "physics_mixture_class_contributions": contributions,
            "physics_mixture_total": contributions.sum(axis=0)}


def plot_ae_selection(scores, labels, weights, names, cut, path):
    histogram = selection_histograms(scores, labels, weights)
    edges = histogram["bin_edges"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharex=True)
    colors = plt.get_cmap("tab10")
    for index, label in enumerate(histogram["class_labels"]):
        color = colors(int(label) % 10)
        name = names[int(label)]
        for ax, field in zip(axes, ("physics_mixture_class_contributions",
                                    "physics_per_class_fraction", "raw_per_class_fraction")):
            ax.stairs(histogram[field][index], edges, color=color, label=name)
    axes[0].stairs(histogram["physics_mixture_total"], edges, color="black", lw=2, label="Total physical mixture")
    for ax in axes:
        ax.axvline(cut, ls="--", color="black", label=f"Selection AE cut = {cut:.4g}")
        ax.set(xscale="log", xlabel="AE reconstruction MSE")
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    axes[0].set(title="PHYSICS-WEIGHTED all-background mixture",
                ylabel="Fraction of TOTAL physical mixture / bin")
    axes[1].set(title="Physics-weighted per-class SHAPES",
                ylabel="Fraction of each class's physical weight / bin")
    axes[2].set(title="RAW MC coverage — NOT the physical mixture",
                ylabel="Raw per-class fraction / bin")
    fig.suptitle("Same validation events and AE scores; physical panels use generator × cross-section weights exactly once\n"
                 "Full distributions before preselection; dashed line is the selection boundary, not the later ABCD split")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return histogram


def read_raw_md_cache(score_path, report_path, checkpoint, ae_hash,
                      expected_qcd_rows, physics, checkpoint_sha, fit_qcd_events):
    report = json.loads(report_path.read_text())
    required = {
        "protocol": "train_QCD_fitted_conditional_MD_CDF_to_disjoint_validation_QCD",
        "heldout_sample_accessed": False, "checkpoint_sha256": checkpoint_sha,
        "ae_checkpoint_sha256": ae_hash, "qcd_label": 1, "n_pca": 6,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "fit_qcd_events": int(fit_qcd_events),
        "validation_qcd_events": len(expected_qcd_rows),
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("QCD cache provenance differs from the selected checkpoint/AE/protocol")
    with np.load(score_path, allow_pickle=False) as cache:
        for field in ("source_row_index", "ae_score", "raw_md", "physics_weight"):
            if field not in cache:
                raise ValueError(f"Missing uncalibrated score-cache field: {field}")
        result = {field: np.array(cache[field]) for field in
                  ("source_row_index", "ae_score", "raw_md", "physics_weight")}
    rows = result["source_row_index"]
    if (rows.ndim != 1 or rows.dtype.kind not in "iu"
            or not np.array_equal(np.sort(rows), np.sort(expected_qcd_rows))):
        raise ValueError("Cached rows do not equal the checkpoint's complete saved validation QCD")
    if any(value.shape != rows.shape for value in result.values()):
        raise ValueError("Cached score fields are not aligned")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite cached QCD scores")
    if np.any(result["raw_md"] < 0) or not np.allclose(
            result["physics_weight"], physics[rows], rtol=1e-7, atol=0):
        raise ValueError("Cached weights/scores differ from the current physics payload")
    return result


def scan_closure(ae, md, weights, percentiles, minimums):
    """All cut percentiles are conditional on the population passed here."""
    # Import only for the actual audit; --help and selection/cache helpers are
    # usable without the evaluator's W&B dependency.
    from eval_abcd_nurd import (
        abcd_region_statistics_at_thresholds, abcd_yields,
        closure_ratio_and_uncertainty, nonclosure_A, statistically_valid_regions,
    )
    shape = (len(percentiles), len(percentiles))
    central, ratio, error = (np.full(shape, np.nan) for _ in range(3))
    valid = np.zeros(shape, dtype=bool)
    ess = np.zeros((*shape, 4))
    if len(ae) == 0 or weights.sum() <= 0:
        return {"central_nonclosure": central, "valid_grid": central.copy(),
                "ratio_grid": ratio, "uncertainty_grid": error, "valid": valid,
                "region_ess": ess, "ae_thresholds": [], "md_thresholds": [],
                "curve": [], "working_point": None, "events": len(ae), "ess": 0.,
                "valid_points": 0, "rejected_points": int(valid.size),
                "grid_median_absolute_nonclosure": None,
                "grid_p90_absolute_nonclosure": None}
    t1 = weighted_quantile(ae, percentiles, weights)
    t2 = weighted_quantile(md, percentiles, weights)
    curve, point = [], None
    for i, cut1 in enumerate(t1):
        for j, cut2 in enumerate(t2):
            stats = abcd_region_statistics_at_thresholds(ae, md, cut1, cut2, weights=weights)
            values = abcd_yields(stats)
            nc, _ = nonclosure_A(*values)
            r, u = closure_ratio_and_uncertainty(stats)
            ess[i, j] = [stats[region]["effective_count"] for region in "ABCD"]
            if np.isfinite(nc):
                central[i, j] = abs(nc)
            ratio[i, j], error[i, j] = r, u
            valid[i, j] = statistically_valid_regions(stats, minimums) and np.isfinite(nc)
            if i == j:
                curve.append({"percentile": float(percentiles[i]),
                              "conditional_A_efficiency": float(values[0] / weights.sum()),
                              "ratio": float(r) if np.isfinite(r) else None,
                              "uncertainty": float(u) if np.isfinite(u) else None,
                              "valid": bool(valid[i, j])})
            if i == 0 and j == 0:
                point = {"definition": "fixed_conditional_median_median_not_optimized",
                         "ae_cut": float(cut1), "md_cut": float(cut2),
                         "absolute_nonclosure": float(abs(nc)) if np.isfinite(nc) else None,
                         "predicted_over_observed": float(r) if np.isfinite(r) else None,
                         "uncertainty": float(u) if np.isfinite(u) else None,
                         "statistically_valid": bool(valid[i, j]), "regions": stats}
    grid = np.where(valid, central, np.nan)
    return {"central_nonclosure": central, "valid_grid": grid,
            "ratio_grid": ratio, "uncertainty_grid": error, "valid": valid,
            "region_ess": ess, "ae_thresholds": t1.tolist(), "md_thresholds": t2.tolist(),
            "curve": curve, "working_point": point, "events": len(ae),
            "ess": effective_count(weights), "valid_points": int(valid.sum()),
            "rejected_points": int(valid.size - valid.sum()),
            "grid_median_absolute_nonclosure": float(np.median(grid[valid])) if valid.any() else None,
            "grid_p90_absolute_nonclosure": float(np.percentile(grid[valid], 90)) if valid.any() else None}


def compact_scan(scan):
    arrays = {"central_nonclosure", "valid_grid", "ratio_grid", "uncertainty_grid", "valid", "region_ess"}
    return {key: value for key, value in scan.items() if key not in arrays}


def retention_rows(labels, weights, selected, names):
    rows = []
    total_before, total_after = float(weights.sum()), float(weights[selected].sum())
    for label in sorted(names):
        mask = labels == label
        keep = mask & selected
        mass = float(weights[mask].sum())
        rows.append({"label": int(label), "name": names[label],
                     "events_before": int(mask.sum()), "events_after": int(keep.sum()),
                     "raw_retention": float(keep.sum() / mask.sum()) if mask.any() else 0.,
                     "physics_mass_before": mass, "physics_mass_after": float(weights[keep].sum()),
                     "fraction_of_physical_mixture_before": mass / total_before if total_before else 0.,
                     "fraction_of_physical_mixture_after": float(weights[keep].sum() / total_after) if total_after else 0.,
                     "physics_retention": float(weights[keep].sum() / mass) if mass else 0.,
                     "ess_before": effective_count(weights[mask]),
                     "ess_after": effective_count(weights[keep])})
    return rows


def plots(scans, ae, md, weights, selected, percentiles, minimums, class_rows,
          category_rows, all_ae, all_labels, all_physics, cut, outdir):
    titles = ("Full validation QCD", "AE-tail validation QCD")
    retained = scans["tail"]
    finite = np.concatenate([scans[key]["valid_grid"].ravel() for key in scans])
    finite = finite[np.isfinite(finite)]
    vmax = max(float(np.percentile(finite, 95) * 100), 1.) if finite.size else 100.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, key in zip(axes, titles, ("full", "tail")):
        result = scans[key]
        mesh = ax.pcolormesh(percentiles, percentiles, result["valid_grid"].T * 100,
                             vmin=0, vmax=vmax, cmap="viridis_r", shading="auto")
        if not result["valid_points"]:
            ax.text(.5, .5, "No statistically valid cells", transform=ax.transAxes, ha="center")
        point = result["working_point"]
        cuts = (f"\nMedian split: AE={point['ae_cut']:.4g}, MD={point['md_cut']:.4g}"
                if point is not None else "")
        ax.set(title=f"{title}\nN={result['events']:,}, ESS={result['ess']:.1f}{cuts}",
               xlabel="AE percentile WITHIN this QCD population", ylabel="MD percentile WITHIN this QCD population")
        fig.colorbar(mesh, ax=ax, label="|Nonclosure| (%)")
    fig.tight_layout()
    fig.savefig(outdir / "closure_scan_2d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, key in zip(axes, titles, ("full", "tail")):
        scale = np.asarray([minimums[r] for r in "ABCD"])
        sufficiency = np.min(scans[key]["region_ess"] / scale, axis=2)
        mesh = ax.pcolormesh(percentiles, percentiles, sufficiency.T, vmin=0, vmax=1,
                             cmap="RdYlGn", shading="auto")
        ax.set(title=title, xlabel="Conditional AE percentile", ylabel="Conditional MD percentile")
        fig.colorbar(mesh, ax=ax, label="Minimum region ESS / required ESS (1 = sufficient)")
    fig.tight_layout()
    fig.savefig(outdir / "closure_effective_statistics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, key in zip(axes, titles, ("full", "tail")):
        for valid, color, label in ((False, "0.7", "Insufficient ESS"), (True, "tab:blue", "Valid")):
            points = [p for p in scans[key]["curve"] if p["valid"] == valid
                      and p["ratio"] is not None and p["conditional_A_efficiency"] > 0]
            if points:
                ax.errorbar([p["conditional_A_efficiency"] for p in points], [p["ratio"] for p in points],
                            yerr=[p["uncertainty"] for p in points], fmt="o", ms=3, color=color, label=label)
        ax.axhline(1, color="black", ls="--")
        efficiencies = [p["conditional_A_efficiency"] for p in scans[key]["curve"]
                        if p["ratio"] is not None and p["conditional_A_efficiency"] > 0]
        low = min(efficiencies) * .8 if efficiencies else 1e-4
        ax.set(title=title, xscale="log", xlim=(max(low, np.finfo(float).tiny), 1),
               xlabel="Weighted QCD A efficiency WITHIN population",
               ylabel="ABCD predicted / observed")
        ax.grid(alpha=.2)
        if efficiencies:
            ax.legend()
        else:
            ax.text(.5, .5, "No finite four-region ratios", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(outdir / "cut_and_count_bkg_check.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, keep, result in zip(axes, titles, (np.ones(len(ae), bool), selected),
                                      (scans["full"], retained)):
        if keep.any():
            hist = ax.hist2d(np.log10(np.maximum(ae[keep], 1e-12)),
                             np.log10(np.maximum(md[keep], 1e-12)), bins=65,
                             weights=weights[keep], norm=LogNorm())
            fig.colorbar(hist[3], ax=ax, label="Physics-weighted QCD yield (log color)")
            point = result["working_point"]
            ax.axvline(np.log10(point["ae_cut"]), ls="--", color="red")
            ax.axhline(np.log10(max(point["md_cut"], 1e-12)), ls="--", color="red")
            ax.axvline(np.log10(max(cut, 1e-12)), ls=":", color="tab:orange",
                       label=f"All-class preselection AE={cut:.4g}")
            ax.plot([], [], ls="--", color="red",
                    label=f"ABCD splits: AE={point['ae_cut']:.4g}, MD={point['md_cut']:.4g}")
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "No QCD rows retained", transform=ax.transAxes, ha="center")
        ax.set(title=title, xlabel="log10 AE reconstruction MSE", ylabel="log10 unchanged QCD MD²")
    fig.suptitle("Physics-weighted yields; logarithmic colors keep even very rare weighted populations visible")
    fig.tight_layout()
    fig.savefig(outdir / "qcd_ae_md.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rows, title in zip(axes, (class_rows, category_rows),
                               ("All validation classes", "QCD hard-pT categories")):
        positions = np.arange(len(rows))
        ax.bar(positions - .18, [r["raw_retention"] for r in rows], .36, label="Raw row retention")
        ax.bar(positions + .18, [r["physics_retention"] for r in rows], .36, label="Physics-mass retention")
        ax.set_xticks(positions, [r["name"] for r in rows], rotation=40, ha="right")
        ax.set(title=title, ylabel="Fraction retained", ylim=(0, 1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "selection_retention.png", dpi=180)
    plt.close(fig)

    histograms = plot_ae_selection(all_ae, all_labels, all_physics,
                                    {row["label"]: row["name"] for row in class_rows},
                                    cut, outdir / "ae_selection.png")
    np.savez_compressed(outdir.parent / "ae_selection_histograms.npz", **histograms)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ae-checkpoint", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--training-weights", required=True, type=Path)
    parser.add_argument("--qcd-score-cache", required=True, type=Path)
    parser.add_argument("--qcd-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tail-fraction", type=float, default=.1)
    parser.add_argument("--selection-measure", choices=("physics-all", "raw-all", "physics-qcd"), default="physics-all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    if not 0 < args.tail_fraction < 1 or args.threads < 1 or args.ae_batch_size < 1:
        parser.error("Invalid tail fraction, thread count, or AE batch size")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.checkpoint, args.ae_checkpoint, args.training_manifest, args.training_weights,
                 args.qcd_score_cache, args.qcd_cache_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    from eval_abcd_nurd import checkpoint_reference_indices
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ae_hash, checkpoint_sha = file_sha256(args.ae_checkpoint), file_sha256(args.checkpoint)
    if checkpoint.get("ae_checkpoint_sha256") != ae_hash:
        raise ValueError("AE differs from the one used to train the model")
    raw = load_hlt_sample(args.training_manifest)
    physics, metadata = load_generator_weights(
        str(args.training_weights), raw["label"], qcd_label=1, sample=raw)
    if args.selection_measure == "physics-all" and metadata.get("weight_scope") != "all_events":
        raise ValueError("physics-all requires original all-event production weights, not legacy QCD-only weights")
    physics = physics.numpy().astype(np.float64)
    train, validation = checkpoint_reference_indices(checkpoint, sample_signature(raw), metadata)
    preprocessing = checkpoint["preprocessing"]
    if preprocessing.get("target_scheme") != "decay8" or preprocessing.get("qcd_label") != 1:
        raise ValueError("Expected the full decay8 baseline with QCD label 1")
    labels = checkpoint_target_labels(raw, preprocessing).numpy()
    print("Evaluation measure: original generator × cross-section weights; no class balancing, "
          "no nuisance balancing, no resampling, and no feature/score multiplication.", flush=True)
    qcd_rows = validation[labels[validation] == 1]
    cache = read_raw_md_cache(args.qcd_score_cache, args.qcd_cache_report, checkpoint,
                              ae_hash, qcd_rows, physics, checkpoint_sha,
                              int((labels[train] == 1).sum()))
    ae_payload = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    ae = HLTAutoencoder(ae_payload["ae_config"])
    ae.load_state_dict(ae_payload["ae"])
    ae.eval()
    print(f"CPU only: AE scoring {len(validation):,} validation rows; reusing {len(qcd_rows):,} raw QCD MD scores.", flush=True)
    with threadpool_limits(limits=args.threads):
        ae_values = infer_ae_scores(ae, checkpoint["ae_scaler"], raw, validation, "cpu", args.ae_batch_size)
    if not np.isfinite(ae_values).all() or np.any(ae_values < 0):
        raise ValueError("Invalid validation AE score")
    full_ae = np.full(len(labels), np.nan)
    full_ae[validation] = ae_values
    if not np.allclose(full_ae[cache["source_row_index"]], cache["ae_score"], rtol=2e-4, atol=1e-7):
        raise ValueError("Recomputed CPU AE scores do not match the cached checkpoint scores")
    selected, selection = select_tail(ae_values, labels[validation], physics[validation],
                                       validation, args.tail_fraction, args.selection_measure)
    selected_full = np.zeros(len(labels), dtype=bool)
    selected_full[validation] = selected
    qcd_rows = cache["source_row_index"]
    qcd_ae, qcd_md, qcd_weights = full_ae[qcd_rows], cache["raw_md"], physics[qcd_rows]
    tail = selected_full[qcd_rows]
    percentiles = np.linspace(.5, .98, 48)
    minimums = {"A": 50, "B": 50, "C": 50, "D": 500}
    scans = {
        "full": scan_closure(qcd_ae, qcd_md, qcd_weights, percentiles, minimums),
        "tail": scan_closure(qcd_ae[tail], qcd_md[tail], qcd_weights[tail], percentiles, minimums),
    }
    names = {int(k): v for k, v in preprocessing["target_class_names"].items()}
    class_rows = retention_rows(labels[validation], physics[validation], selected, names)
    print("Physical mixture before selection: " + str({
        row["name"]: row["fraction_of_physical_mixture_before"] for row in class_rows}), flush=True)
    categories = raw["background_category"][qcd_rows].numpy()
    category_rows = retention_rows(categories, qcd_weights, tail, CATEGORY_NAMES)
    args.output_dir.mkdir(parents=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir()
    with (args.output_dir / "mixture_composition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    plots(scans, qcd_ae, qcd_md, qcd_weights, tail, percentiles, minimums,
          class_rows, category_rows, ae_values, labels[validation], physics[validation],
          selection["ae_boundary"], plot_dir)
    np.savez_compressed(args.output_dir / "validation_ae_selection.npz",
                        source_row_index=validation, ae_score=ae_values,
                        label=labels[validation], physics_weight=physics[validation],
                        selected=selected, event_id=raw["event_id"][validation].numpy())
    np.savez_compressed(args.output_dir / "validation_qcd_scores.npz", source_row_index=qcd_rows,
                        ae_score=qcd_ae, raw_md=qcd_md, physics_weight=qcd_weights, selected=tail,
                        event_id=raw["event_id"][qcd_rows].numpy(), background_category=categories)
    np.savez_compressed(args.output_dir / "closure_scan_arrays.npz",
                        percentiles=percentiles,
                        **{f"{role}_{field}": result[field] for role, result in scans.items()
                           for field in ("central_nonclosure", "valid_grid", "ratio_grid", "valid", "region_ess")})
    with (args.output_dir / "abcd_cut_coordinates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("population", "conditional_percentile", "ae_cut", "md_cut"))
        writer.writeheader()
        for population, scan in scans.items():
            for p, ae_cut, md_cut in zip(percentiles, scan["ae_thresholds"], scan["md_thresholds"]):
                writer.writerow({"population": population, "conditional_percentile": p,
                                 "ae_cut": ae_cut, "md_cut": md_cut})
    point = scans["tail"]["working_point"]
    if point is not None:
        codes = region_codes(qcd_ae[tail], qcd_md[tail], point["ae_cut"], point["md_cut"])
        composition = region_statistics(codes, qcd_weights[tail], categories[tail])
        with (args.output_dir / "tail_region_composition.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(composition[0]))
            writer.writeheader()
            writer.writerows(composition)
    report = {
        "protocol": "validation_AE_tail_selection_then_conditional_ABCD_scan_fixed_training_QCD_MD",
        "heldout_sample_accessed": False, "network_parameters_updated": False,
        "device": "cpu", "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "effective_lambda_info": checkpoint.get("effective_lambda_info"),
        "ae_checkpoint_sha256": ae_hash, "qcd_score_cache": str(args.qcd_score_cache),
        "training_manifest": str(args.training_manifest), "training_weights": str(args.training_weights),
        "cache_report": str(args.qcd_cache_report),
        "qcd_score_cache_sha256": file_sha256(args.qcd_score_cache),
        "md_reference": "unchanged_full_training_QCD_physics_weighted_six_component_encoder_MD",
        "md_calibration_applied": False, "selection": selection,
        "evaluation_weighting": {
            "source": "original_all_event_generator_cross_section_physics_payload",
            "weight_scope": metadata["weight_scope"],
            "physics_weight_sha256": metadata["effective_physics_weight_sha256"],
            "class_balancing_applied": False, "nuisance_balancing_applied": False,
            "weights_applied_before_percentile_selection": args.selection_measure != "raw-all",
            "AE_inputs_or_scores_multiplied_by_weights": False,
        },
        "validation_events": len(validation), "training_events": len(train),
        "qcd_raw_retention": float(tail.mean()),
        "qcd_physics_retention": float(qcd_weights[tail].sum() / qcd_weights.sum()),
        "minimum_region_ess": minimums, "class_retention": class_rows,
        "qcd_category_retention": category_rows,
        "scans": {key: compact_scan(value) for key, value in scans.items()},
        "warning": ("All selection/cut scans are exploratory on saved validation; not independent held-out evidence. "
                    "Tail selection changes the target population and need not improve independence. "
                    "Cuts are conditional on the retained population; insufficient-ESS cells remain masked."),
    }
    (args.output_dir / "tail_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selection": selection,
                      "qcd_raw_retention": report["qcd_raw_retention"],
                      "qcd_physics_retention": report["qcd_physics_retention"],
                      "tail_QCD_ESS": scans["tail"]["ess"],
                      "tail_valid_grid_points": scans["tail"]["valid_points"],
                      "tail_grid_median": scans["tail"]["grid_median_absolute_nonclosure"]}, indent=2), flush=True)
    for population, scan in scans.items():
        point = scan["working_point"]
        if point is not None:
            print(f"{population} median/median ABCD cuts: AE={point['ae_cut']:.8g}, "
                  f"MD={point['md_cut']:.8g}; valid={point['statistically_valid']}; "
                  f"region ESS=" + str({r: round(point['regions'][r]['effective_count'], 2) for r in "ABCD"}),
                  flush=True)
    print(f"AE TAIL VALIDATION DONE: {args.output_dir / 'tail_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
