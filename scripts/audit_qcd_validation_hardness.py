"""Validation-only QCD AE/MD diagnostic, with reconstructed jet hardness.

The QCD MD reference is fitted on the checkpoint's *training* split; plots and
statistics use only its disjoint validation split. Fixed ABCD thresholds are
read from the existing evaluation summary for annotation/checking, never fit
or selected here. The held-out test manifest is not opened.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_qcd_abcd_composition import (  # noqa: E402
    infer_ae_scores,
    infer_latents,
    region_codes,
)
from utils.hlt_data import load_hlt_sample  # noqa: E402
from utils.hlt_targets import checkpoint_target_labels  # noqa: E402
from utils.hlt_weights import file_sha256  # noqa: E402


HT_LABEL = r"$H_T^{(10)}$: leading 10 scouting PF jets, $p_T\geq30$ GeV"


def ht10_from_objects(obj, indices, *, jet_threshold=30.0):
    """Jet HT lower bound available from the converted AE object tensor."""
    jets = obj[indices, :10, 0].numpy().astype(np.float64)
    return np.where(jets >= jet_threshold, jets, 0.0).sum(axis=1)


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("Values and weights must be aligned one-dimensional arrays")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("Values and weights must be finite")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be nonnegative with positive total mass")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    positions = np.searchsorted(cumulative, np.asarray(quantiles) * cumulative[-1])
    return values[order[np.clip(positions, 0, len(order) - 1)]]


def weighted_pearson(x, y, weights):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (x.shape == y.shape == weights.shape):
        raise ValueError("Correlation inputs must have matching shapes")
    mass = weights.sum()
    xmean = np.dot(weights, x) / mass
    ymean = np.dot(weights, y) / mass
    dx, dy = x - xmean, y - ymean
    covariance = np.dot(weights, dx * dy)
    variance_x = np.dot(weights, dx * dx)
    variance_y = np.dot(weights, dy * dy)
    return float(covariance / np.sqrt(variance_x * variance_y)) \
        if variance_x > 0 and variance_y > 0 else float("nan")


def ht_slice_edges(ht, weights):
    """Prefer equal-weight thirds; handle a pileup at HT=0 without splitting ties."""
    inner = weighted_quantile(ht, [1/3, 2/3], weights)
    edges = np.array([np.min(ht), *inner, np.max(ht)])
    method = "physics_weighted_thirds"
    if np.any(np.diff(edges) <= 0):
        edges = np.array([np.min(ht), *np.quantile(ht, [1/3, 2/3]), np.max(ht)])
        method = "raw_count_thirds_due_to_ties"
    if np.any(np.diff(edges) <= 0):
        distinct = np.unique(ht)
        if len(distinct) < 4:
            raise ValueError("Fewer than four distinct HT10 values")
        edges = distinct[[0, len(distinct)//3, 2*len(distinct)//3, -1]]
        method = "distinct_value_thirds_due_to_ties"
    return edges, method


def region_rows(codes, weights, group):
    rows = []
    for category in sorted(np.unique(group)):
        mask = group == category
        values = {r: float(weights[mask & (codes == i)].sum())
                  for i, r in enumerate("ABCD")}
        sums2 = {r: float(np.square(weights[mask & (codes == i)]).sum())
                 for i, r in enumerate("ABCD")}
        a, b, c, d = (values[r] for r in "ABCD")
        rows.append({
            "group": int(category),
            "events": int(mask.sum()),
            "weighted_mass": float(weights[mask].sum()),
            **{f"yield_{r}": values[r] for r in "ABCD"},
            **{f"ess_{r}": values[r]**2 / sums2[r] if sums2[r] else 0.0
               for r in "ABCD"},
            "predicted_over_observed": b * c / (d * a) if d * a else float("nan"),
        })
    return rows


def positive_log_edges(values, *, count=65):
    positive = np.asarray(values)[np.asarray(values) > 0]
    if len(positive) < 2:
        raise ValueError("Too few positive values for log-scale plot")
    low, high = np.quantile(positive, [0.001, 0.999])
    if low <= 0 or high <= low:
        raise ValueError("Degenerate log-scale plot range")
    return np.geomspace(low, high, count)


def plot_weighted_hist(ax, x, y, weights, xedges, yedges):
    histogram, _, _ = np.histogram2d(
        x, y, bins=[xedges, yedges], weights=weights)
    positive = histogram[histogram > 0]
    minimum = positive.min() if len(positive) else 1.0
    maximum = positive.max() if len(positive) else 10.0
    mesh = ax.pcolormesh(xedges, yedges, histogram.T,
                         norm=LogNorm(vmin=minimum, vmax=max(maximum, minimum * 1.001)),
                         cmap="viridis", shading="auto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    return mesh


def plot_ae_md_slices(ae_score, md_score, ht, weights, ht_edges,
                      ae_threshold, md_threshold, path):
    xedges = positive_log_edges(ae_score)
    yedges = positive_log_edges(md_score)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    for index, ax in enumerate(axes):
        mask = ((ht >= ht_edges[index]) &
                ((ht < ht_edges[index + 1]) if index < 2 else
                 (ht <= ht_edges[index + 1])))
        mesh = plot_weighted_hist(
            ax, ae_score[mask], md_score[mask], weights[mask], xedges, yedges)
        ax.axvline(ae_threshold, color="red", linestyle="--", linewidth=1)
        ax.axhline(md_threshold, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{ht_edges[index]:.1f}–{ht_edges[index+1]:.1f} GeV\n"
                     f"{mask.sum():,} rows; {weights[mask].sum()/weights.sum():.1%} weight")
        ax.set_xlabel("AE reconstruction MSE")
        figure.colorbar(mesh, ax=ax, label="Physics-weighted yield / cell")
    axes[0].set_ylabel("QCD-reference latent MD$^2$")
    figure.suptitle("Validation QCD, generator hard-$p_T$ 15–50 GeV; " + HT_LABEL)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_ht_ae(ht, ae_score, weights, ae_threshold, path):
    # The +1 shift keeps genuine zero-jet events visible on the log x axis.
    xedges = positive_log_edges(ht + 1.0)
    yedges = positive_log_edges(ae_score)
    figure, ax = plt.subplots(figsize=(7.4, 5.3))
    mesh = plot_weighted_hist(ax, ht + 1.0, ae_score, weights, xedges, yedges)
    ax.axhline(ae_threshold, color="red", linestyle="--", linewidth=1,
               label="Saved AE threshold")
    ax.set_xlabel(HT_LABEL + r" [GeV]; plotted as $H_T^{(10)}+1$")
    ax.set_ylabel("AE reconstruction MSE")
    ax.set_title("Validation QCD, generator hard-$p_T$ 15–50 GeV")
    ax.legend()
    figure.colorbar(mesh, ax=ax, label="Physics-weighted yield / cell")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--training-weights", required=True)
    parser.add_argument("--evaluation-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-batch-size", type=int, default=512)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output_dir}")

    # Import evaluator after argparse so the pure analysis helpers remain
    # usable in lightweight local environments without W&B.
    from eval_abcd_nurd import (
        _fit_class_transform,
        checkpoint_reference_indices,
        load_ae,
        load_nurd_model,
        load_physics_weights,
    )

    saved = json.loads(Path(args.evaluation_summary).read_text())[
        "evaluations"]["held-out"]
    thresholds = saved["thresholds"]
    if thresholds["selection_source"] != "saved_training_validation_split":
        raise ValueError("Expected thresholds selected on saved validation split")
    if int(thresholds["n_pca"]) != 6:
        raise ValueError("Expected six-component QCD-only MD reference")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_nurd_model(args.checkpoint, device)
    if file_sha256(args.ae_checkpoint) != checkpoint.get("ae_checkpoint_sha256"):
        raise ValueError("AE checkpoint differs from the classifier's saved AE")
    preprocessing = checkpoint["preprocessing"]
    if preprocessing.get("target_scheme") != "decay8":
        raise ValueError("This diagnostic expects the decay8 campaign")
    if int(preprocessing.get("qcd_label", 1)) != 1:
        raise ValueError("Expected QCD target label 1")

    sample = load_hlt_sample(args.training_manifest)
    weights, metadata, signature = load_physics_weights(
        args.training_weights, args.training_manifest,
        int(preprocessing.get("source_parent_qcd_label", 1)))
    fit_indices, val_indices = checkpoint_reference_indices(
        checkpoint, signature, metadata)
    labels = checkpoint_target_labels(sample, preprocessing).numpy()
    fit_qcd = np.sort(fit_indices[labels[fit_indices] == 1])
    val_qcd = np.sort(val_indices[labels[val_indices] == 1])
    print(f"MD fit QCD={len(fit_qcd):,}; validation QCD={len(val_qcd):,}", flush=True)

    reference_latents = infer_latents(
        model, sample, fit_qcd, device, args.latent_batch_size)
    mu, whitening = _fit_class_transform(
        reference_latents, np.ones(len(fit_qcd), dtype=bool), 6, "QCD",
        weights=weights[fit_qcd])
    del reference_latents

    ae = load_ae(args.ae_checkpoint, checkpoint["ae_scaler"], device)
    ae_score = infer_ae_scores(
        ae, checkpoint["ae_scaler"], sample, val_qcd,
        device, args.ae_batch_size)
    del ae
    if device == "cuda":
        torch.cuda.empty_cache()
    latents = infer_latents(model, sample, val_qcd, device, args.latent_batch_size)
    whitened = (latents - mu) @ whitening
    md_score = (whitened * whitened).sum(axis=1).astype(np.float32)

    valid = np.isfinite(ae_score) & np.isfinite(md_score) & (ae_score > 0)
    val_qcd = val_qcd[valid]
    ae_score, md_score = ae_score[valid], md_score[valid]
    categories = sample["background_category"][val_qcd].numpy()
    event_ids = sample["event_id"][val_qcd].numpy()
    physics_weights = weights[val_qcd]
    ht = ht10_from_objects(sample["obj"], val_qcd)
    if np.unique(event_ids, axis=0).shape[0] != len(event_ids):
        raise ValueError("Validation QCD event IDs are not unique")
    codes = region_codes(ae_score, md_score,
                         thresholds["t1"], thresholds["t2"])
    all_yields = {region: float(physics_weights[codes == index].sum())
                  for index, region in enumerate("ABCD")}
    predicted_over_observed = (all_yields["B"] * all_yields["C"] /
                               (all_yields["D"] * all_yields["A"]))
    selection_nonclosure = abs(1.0 / predicted_over_observed - 1.0)
    if not np.isclose(selection_nonclosure,
                      abs(thresholds["selection_nonclosure"]), rtol=0.02, atol=0.01):
        raise ValueError("Validation closure differs from saved threshold-selection report")

    soft = categories == 0
    soft_ht, soft_ae, soft_md = ht[soft], ae_score[soft], md_score[soft]
    soft_weights = physics_weights[soft]
    ht_edges, ht_slice_method = ht_slice_edges(soft_ht, soft_weights)
    ht_group = np.digitize(soft_ht, ht_edges[1:3], right=False)
    soft_codes = codes[soft]
    ht_rows = region_rows(soft_codes, soft_weights, ht_group)

    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / "validation_qcd_scores.npz",
                        event_id=event_ids, category=categories,
                        physics_weight=physics_weights, ae_score=ae_score,
                        md_score=md_score, ht10_ge30=ht,
                        region=np.asarray(tuple("ABCD"))[codes])
    with (output_dir / "ht10_thirds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ht_rows[0]))
        writer.writeheader()
        writer.writerows(ht_rows)
    plot_ae_md_slices(soft_ae, soft_md, soft_ht, soft_weights,
                      ht_edges, thresholds["t1"], thresholds["t2"],
                      output_dir / "ae_vs_md_by_ht10_qcd15_50.png")
    plot_ht_ae(soft_ht, soft_ae, soft_weights, thresholds["t1"],
               output_dir / "ht10_vs_ae_qcd15_50.png")
    report = {
        "sample_role": "checkpoint_saved_training_validation_split_only",
        "heldout_sample_accessed": False,
        "checkpoint": args.checkpoint,
        "ae_checkpoint": args.ae_checkpoint,
        "evaluation_summary": args.evaluation_summary,
        "ht_definition": "sum of pT >= 30 GeV for first 10 pT-sorted ScoutingPFJet rows in obj; lower bound on full jet HT",
        "qcd_validation_events": int(len(val_qcd)),
        "qcd_15_50_validation_events": int(soft.sum()),
        "thresholds": {"ae": thresholds["t1"], "md": thresholds["t2"]},
        "validation_predicted_over_observed": predicted_over_observed,
        "validation_selection_absolute_nonclosure": selection_nonclosure,
        "ht10_slice_edges_gev": ht_edges.tolist(),
        "ht10_slice_method": ht_slice_method,
        "qcd_15_50_log10_ht10_vs_log10_ae_weighted_pearson": weighted_pearson(
            np.log10(np.maximum(soft_ht, 1)), np.log10(soft_ae), soft_weights),
        "qcd_15_50_log10_ae_vs_log10_md_weighted_pearson": weighted_pearson(
            np.log10(soft_ae), np.log10(np.maximum(soft_md, 1e-12)), soft_weights),
        "ht10_thirds": ht_rows,
    }
    (output_dir / "validation_hardness_summary.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("ht10_thirds",)}, indent=2), flush=True)
    print(f"VALIDATION HARDNESS AUDIT DONE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
