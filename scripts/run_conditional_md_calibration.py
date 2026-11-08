"""Fit QCD MD|AE calibration on training rows; report validation-only closure.

The outer test sample is never opened. This is a score-level diagnostic for a
fixed AE and classifier checkpoint, not a new model-training or held-out result.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_abcd_nurd import (
    _fit_class_transform,
    checkpoint_reference_indices,
    load_ae,
    load_nurd_model,
)
from scripts.scan_validation_checkpoints import (
    _compute_ae_scores,
    _infer_latents,
    validation_closure_metrics,
)
from utils.hlt_conditional_md import ConditionalMDCDF
from utils.hlt_data import load_hlt_sample
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


def _ess(weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    squared = np.square(weights).sum()
    return float(total * total / squared) if squared > 0 else 0.0


def _weighted_uniform_ks(values, weights):
    order = np.argsort(values, kind="stable")
    values = np.asarray(values, dtype=np.float64)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(weights) / weights.sum()
    previous = cumulative - weights / weights.sum()
    return float(max(np.max(np.abs(cumulative - values)),
                     np.max(np.abs(previous - values))))


def _compact_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "grid"}


def _plot_comparison(raw, calibrated, percentiles, plot_dir):
    finite = np.concatenate([
        grid[np.isfinite(grid)] for grid in (raw["grid"], calibrated["grid"])
    ])
    vmax = max(float(np.percentile(finite, 95) * 100), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, title, metrics in zip(
        axes, ("Raw QCD MD", "Training-fitted conditional MD percentile"),
        (raw, calibrated),
    ):
        mesh = ax.pcolormesh(
            percentiles, percentiles, metrics["grid"].T * 100,
            cmap="viridis_r", vmin=0, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("AE percentile threshold")
    axes[0].set_ylabel("Axis-2 percentile threshold")
    fig.colorbar(mesh, ax=axes, label="|Nonclosure| (%)")
    fig.savefig(plot_dir / "validation_closure_grid_comparison.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for title, metrics in (("Raw MD", raw), ("Conditional percentile", calibrated)):
        curve = sorted(metrics["curve"], key=lambda point: point["efficiency"])
        ax.plot([point["efficiency"] for point in curve],
                [point["ratio"] for point in curve], "o-", markersize=2,
                linewidth=1.2, label=title)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Weighted QCD efficiency in region A")
    ax.set_ylabel("ABCD predicted / observed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "validation_closure_curves_comparison.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--data", required=True,
                        help="Outer-training sample; never pass the held-out test.")
    parser.add_argument("--gen-weight-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--qcd-label", type=int, default=1)
    parser.add_argument("--n-pca", type=int, default=6)
    parser.add_argument("--ae-bins", type=int, default=20)
    parser.add_argument("--min-fit-bin-ess", type=float, default=200.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).resolve()
    ae_path = Path(args.ae_ckpt).resolve()
    data_path = Path(args.data).resolve()
    weight_path = Path(args.gen_weight_path).resolve()
    outdir = Path(args.outdir).resolve()
    if data_path.name != "sample.json" or data_path.parent.name != "train":
        raise ValueError("This experiment requires the outer train/sample.json, not test data.")
    for path in (checkpoint_path, ae_path, data_path, weight_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if outdir.exists():
        raise FileExistsError(outdir)
    outdir.mkdir(parents=True)
    plot_dir = outdir / "plots"
    plot_dir.mkdir()

    raw = load_hlt_sample(data_path)
    labels = raw["label"].long().reshape(-1)
    signature = sample_signature(raw)
    physics_weights, weight_metadata = load_generator_weights(
        str(weight_path), labels, qcd_label=args.qcd_label, sample=raw)
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_indices, validation_indices = checkpoint_reference_indices(
        metadata, signature, weight_metadata)
    expected_ae = metadata.get("ae_checkpoint_sha256")
    if not expected_ae or file_sha256(ae_path) != expected_ae:
        raise ValueError("AE checkpoint differs from the one used by the classifier.")

    labels_array = labels.numpy()
    train_qcd = train_indices[labels_array[train_indices] == args.qcd_label]
    validation_qcd = validation_indices[
        labels_array[validation_indices] == args.qcd_label]
    if not len(train_qcd) or not len(validation_qcd):
        raise ValueError("Training and validation splits must both contain QCD.")
    weight_array = physics_weights.numpy().astype(np.float64)
    train_weights = weight_array[train_qcd]
    validation_weights = weight_array[validation_qcd]
    print(f"QCD: fit={len(train_qcd)}, validation={len(validation_qcd)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, _ = load_ae(str(ae_path), metadata["ae_scaler"], device)
    train_ae = _compute_ae_scores(
        ae, metadata["ae_scaler"], raw["obj"], train_qcd,
        device, args.ae_batch_size)
    validation_ae = _compute_ae_scores(
        ae, metadata["ae_scaler"], raw["obj"], validation_qcd,
        device, args.ae_batch_size)
    del ae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, _ = load_nurd_model(str(checkpoint_path), device)
    train_latents = _infer_latents(
        model, raw["pf"], train_qcd, device, args.batch_size)
    validation_latents = _infer_latents(
        model, raw["pf"], validation_qcd, device, args.batch_size)
    del model
    mu, transform = _fit_class_transform(
        train_latents, np.ones(len(train_latents), dtype=bool),
        args.n_pca, "QCD training split", weights=train_weights)
    train_md = np.square((train_latents - mu) @ transform).sum(axis=1)
    validation_md = np.square(
        (validation_latents - mu) @ transform).sum(axis=1)
    del train_latents, validation_latents
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    calibrator = ConditionalMDCDF.fit(
        train_ae, train_md, train_weights,
        n_ae_bins=args.ae_bins, min_ess=args.min_fit_bin_ess)
    calibrated_md = calibrator.transform(validation_ae, validation_md)
    calibrator.save(outdir / "conditional_md_fit_train_qcd.npz")
    np.savez_compressed(
        outdir / "validation_qcd_scores.npz",
        source_row_index=validation_qcd,
        ae_score=validation_ae,
        raw_md=validation_md,
        calibrated_md_percentile=calibrated_md,
        physics_weight=validation_weights,
    )

    percentiles = np.linspace(0.50, 0.98, 48)
    minimums = {"A": 50, "B": 50, "C": 50, "D": 500}
    raw_metrics = validation_closure_metrics(
        validation_ae, validation_md, validation_weights,
        percentiles, minimums)
    calibrated_metrics = validation_closure_metrics(
        validation_ae, calibrated_md, validation_weights,
        percentiles, minimums)
    common = np.isfinite(raw_metrics["grid"]) & np.isfinite(calibrated_metrics["grid"])
    if not common.any():
        raise RuntimeError("Raw and calibrated scans have no common valid grid cells.")
    _plot_comparison(raw_metrics, calibrated_metrics, percentiles, plot_dir)

    validation_bins = calibrator.bin_indices(validation_ae)
    bin_report = []
    for index in range(len(calibrator.fit_counts)):
        mask = validation_bins == index
        mass = validation_weights[mask]
        values = calibrated_md[mask]
        bin_report.append({
            "bin": index,
            "log10_ae_lower": float(calibrator.log_ae_edges[index]),
            "log10_ae_upper": float(calibrator.log_ae_edges[index + 1]),
            "fit_events": int(calibrator.fit_counts[index]),
            "fit_ess": float(calibrator.fit_effective_counts[index]),
            "validation_events": int(mask.sum()),
            "validation_ess": _ess(mass),
            "validation_weight_fraction": float(mass.sum() / validation_weights.sum()),
            "validation_u_weighted_mean": float(np.average(values, weights=mass))
            if mass.sum() else None,
            "validation_u_weighted_ks": _weighted_uniform_ks(values, mass)
            if mass.sum() else None,
        })

    report = {
        "protocol": "train_QCD_fitted_conditional_MD_CDF_to_disjoint_validation_QCD",
        "heldout_sample_accessed": False,
        "warning": ("Background closure diagnostic only; conditional calibration "
                    "uses AE loss and may alter signal ranking. Do not tune bin count "
                    "on outer held-out data."),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(metadata["epoch"]),
        "ae_checkpoint": str(ae_path),
        "ae_checkpoint_sha256": expected_ae,
        "data": str(data_path),
        "generator_weights": str(weight_path),
        "qcd_label": args.qcd_label,
        "n_pca": args.n_pca,
        "fit_qcd_events": int(len(train_qcd)),
        "validation_qcd_events": int(len(validation_qcd)),
        "fit_qcd_ess": _ess(train_weights),
        "validation_qcd_ess": _ess(validation_weights),
        "requested_ae_bins": args.ae_bins,
        "effective_ae_bins": len(calibrator.fit_counts),
        "min_fit_bin_ess": args.min_fit_bin_ess,
        "minimum_region_ess": minimums,
        "raw": _compact_metrics(raw_metrics),
        "calibrated": _compact_metrics(calibrated_metrics),
        "common_valid_grid": {
            "points": int(common.sum()),
            "raw_median_absolute_nonclosure": float(np.median(raw_metrics["grid"][common])),
            "calibrated_median_absolute_nonclosure": float(
                np.median(calibrated_metrics["grid"][common])),
            "raw_p90_absolute_nonclosure": float(np.percentile(raw_metrics["grid"][common], 90)),
            "calibrated_p90_absolute_nonclosure": float(
                np.percentile(calibrated_metrics["grid"][common], 90)),
        },
        "ae_bins": bin_report,
    }
    with (outdir / "conditional_md_report.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print("Raw grid median:", report["raw"]["grid_median_absolute_nonclosure"])
    print("Calibrated grid median:", report["calibrated"]["grid_median_absolute_nonclosure"])
    print("Common-cell raw/calibrated medians:",
          report["common_valid_grid"]["raw_median_absolute_nonclosure"],
          report["common_valid_grid"]["calibrated_median_absolute_nonclosure"])
    print("Report:", outdir / "conditional_md_report.json")
    print("HELD-OUT TEST NOT ACCESSED", flush=True)


if __name__ == "__main__":
    main()
