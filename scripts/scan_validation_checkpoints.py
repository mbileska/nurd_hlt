"""Compare NURD checkpoints using only the saved training validation split.

This is a model-selection diagnostic, not a held-out evaluation.  For every
checkpoint it fits the QCD Mahalanobis reference on the saved training rows
and measures weighted QCD ABCD closure on the disjoint saved validation rows.
The held-out test file is never opened.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_abcd_nurd import (
    _fit_class_transform,
    abcd_region_statistics_at_thresholds,
    abcd_yields,
    checkpoint_reference_indices,
    closure_ratio_and_uncertainty,
    load_ae,
    load_nurd_model,
    nonclosure_A,
    statistically_valid_regions,
    weighted_quantile,
)
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


_PERIODIC_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pth\.tar$")


def discover_checkpoints(
    checkpoint_dir: Path,
    requested_epochs: Sequence[int] | None = None,
) -> List[Tuple[int, Path, str]]:
    """Return one checkpoint per epoch, preferring the selected checkpoint."""
    paths = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth.tar"))
    for name in ("checkpoint_final.pth.tar", "checkpoint_main.pth.tar"):
        path = checkpoint_dir / name
        if path.is_file():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No loadable checkpoints found under {checkpoint_dir}.")

    requested = None if not requested_epochs else set(map(int, requested_epochs))
    priority = {"periodic": 1, "final": 2, "selected": 3}
    by_epoch: Dict[int, Tuple[int, Path, str]] = {}
    for path in paths:
        match = _PERIODIC_PATTERN.match(path.name)
        if match:
            epoch = int(match.group(1))
            role = "periodic"
        else:
            metadata = torch.load(path, map_location="cpu", weights_only=False)
            epoch = int(metadata["epoch"])
            role = "selected" if path.name == "checkpoint_main.pth.tar" else "final"
        if requested is not None and epoch not in requested:
            continue
        previous = by_epoch.get(epoch)
        if previous is None or priority[role] > previous[0]:
            by_epoch[epoch] = (priority[role], path, role)

    missing = [] if requested is None else sorted(requested - set(by_epoch))
    if missing:
        raise FileNotFoundError(
            "No saved checkpoint for requested epoch(s): "
            + ", ".join(map(str, missing)))
    return [
        (epoch, by_epoch[epoch][1], by_epoch[epoch][2])
        for epoch in sorted(by_epoch)
    ]


def _infer_latents(model, pf, indices, device, batch_size):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start:start + int(batch_size)]
            batch = torch.nan_to_num(
                pf[batch_indices].float(), nan=0.0, posinf=0.0, neginf=0.0
            ).to(device)
            latent, _ = model(batch)
            outputs.append(latent.cpu())
    if not outputs:
        raise ValueError("Cannot infer latents for an empty index collection.")
    return torch.cat(outputs).numpy()


def _compute_ae_scores(ae, scaler, obj, indices, device, batch_size):
    mu = torch.as_tensor(scaler["mu"], dtype=torch.float32).cpu().reshape(1, -1)
    std = torch.as_tensor(scaler["std"], dtype=torch.float32).cpu().reshape(1, -1)
    scores = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start:start + int(batch_size)]
            flat = torch.nan_to_num(
                obj[batch_indices, :, :4].reshape(len(batch_indices), -1).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            normalized = ((flat - mu) / (std + 1e-8)).to(device)
            reconstruction, _ = ae(normalized)
            scores.append((reconstruction - normalized).square().mean(dim=1).cpu())
    if not scores:
        raise ValueError("Cannot calculate AE scores for an empty index collection.")
    return torch.cat(scores).numpy().astype(np.float32)


def validation_closure_metrics(
    axis_1: np.ndarray,
    axis_2: np.ndarray,
    weights: np.ndarray,
    percentiles: Iterable[float],
    minimums: Dict[str, int],
):
    """Return statistically filtered full-grid and diagonal closure metrics."""
    axis_1 = np.asarray(axis_1, dtype=np.float64)
    axis_2 = np.asarray(axis_2, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    percentiles = np.asarray(list(percentiles), dtype=np.float64)
    if not (axis_1.shape == axis_2.shape == weights.shape):
        raise ValueError("Scores and weights must have identical shapes.")
    if not len(axis_1) or np.any(weights < 0.0) or not np.isfinite(weights).all():
        raise ValueError("Validation weights must be non-negative, finite, and non-empty.")

    thresholds_1 = [weighted_quantile(axis_1, p, weights) for p in percentiles]
    thresholds_2 = [weighted_quantile(axis_2, p, weights) for p in percentiles]
    grid = np.full((len(percentiles), len(percentiles)), np.nan, dtype=np.float64)
    curve = []
    total_yield = float(weights.sum())

    for i, threshold_1 in enumerate(thresholds_1):
        for j, threshold_2 in enumerate(thresholds_2):
            statistics = abcd_region_statistics_at_thresholds(
                axis_1, axis_2, threshold_1, threshold_2, weights=weights)
            if not statistically_valid_regions(statistics, minimums):
                continue
            A, B, C, D = abcd_yields(statistics)
            nonclosure, _ = nonclosure_A(A, B, C, D)
            if np.isfinite(nonclosure):
                grid[i, j] = abs(nonclosure)
            if i == j:
                ratio, uncertainty = closure_ratio_and_uncertainty(statistics)
                if np.isfinite(ratio):
                    curve.append({
                        "percentile": float(percentiles[i]),
                        "efficiency": float(A / max(total_yield, 1e-12)),
                        "ratio": float(ratio),
                        "uncertainty": float(uncertainty),
                        "absolute_nonclosure": float(abs(nonclosure)),
                    })

    finite_grid = grid[np.isfinite(grid)]
    finite_curve = np.asarray(
        [point["absolute_nonclosure"] for point in curve], dtype=np.float64)
    if not finite_grid.size or not finite_curve.size:
        raise RuntimeError(
            "No statistically valid validation closure points were found.")
    return {
        "grid": grid,
        "grid_points": int(finite_grid.size),
        "grid_rejected_points": int(grid.size - finite_grid.size),
        "grid_median_absolute_nonclosure": float(np.median(finite_grid)),
        "grid_p90_absolute_nonclosure": float(np.percentile(finite_grid, 90)),
        "curve": curve,
        "curve_points": int(finite_curve.size),
        "curve_rejected_points": int(len(percentiles) - finite_curve.size),
        "curve_median_absolute_nonclosure": float(np.median(finite_curve)),
        "curve_p90_absolute_nonclosure": float(np.percentile(finite_curve, 90)),
    }


def _save_grid_plot(grid, percentiles, epoch, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    values = np.asarray(grid) * 100.0
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 100.0
    mesh = ax.pcolormesh(
        percentiles, percentiles, values.T, cmap="viridis_r",
        vmin=0.0, vmax=max(vmax, 1e-6), shading="auto")
    fig.colorbar(mesh, ax=ax, label="|Non-closure| (%)")
    ax.set_xlabel("AE percentile threshold")
    ax.set_ylabel("NURD MD percentile threshold")
    ax.set_title(f"Validation-only QCD closure: epoch {epoch}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_summary_plots(results, outdir):
    epochs = np.asarray([row["epoch"] for row in results])
    fig, (closure_ax, accuracy_ax) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for key, label, style in (
        ("grid_median_absolute_nonclosure", "Grid median", "o-"),
        ("curve_median_absolute_nonclosure", "Curve median", "s-"),
        ("grid_p90_absolute_nonclosure", "Grid p90", "o--"),
        ("curve_p90_absolute_nonclosure", "Curve p90", "s--"),
    ):
        closure_ax.plot(
            epochs, 100.0 * np.asarray([row[key] for row in results]),
            style, label=label)
    closure_ax.set_ylabel("Absolute non-closure (%)")
    closure_ax.grid(alpha=0.3)
    closure_ax.legend(ncol=2)

    accuracy_ax.plot(
        epochs,
        100.0 * np.asarray([row["balanced_accuracy"] for row in results]),
        "o-", label="Balanced accuracy")
    accuracy_ax.plot(
        epochs,
        100.0 * np.asarray([row["weighted_accuracy"] for row in results]),
        "s-", label="Weighted accuracy")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Validation accuracy (%)")
    accuracy_ax.grid(alpha=0.3)
    accuracy_ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "validation_metrics_vs_epoch.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for row in results:
        curve = sorted(row["curve"], key=lambda point: point["efficiency"])
        ax.plot(
            [point["efficiency"] for point in curve],
            [point["ratio"] for point in curve],
            marker="o", markersize=2, linewidth=1, label=f"epoch {row['epoch']}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Weighted QCD efficiency in region A")
    ax.set_ylabel("ABCD predicted / observed")
    ax.set_title("Validation-only closure curves")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "validation_closure_curves.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan saved NURD checkpoints on training-validation QCD only.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--gen-weight-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=None)
    parser.add_argument("--qcd-label", type=int, default=1)
    parser.add_argument("--n-pca", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    parser.add_argument("--min-A", type=int, default=50)
    parser.add_argument("--min-B", type=int, default=50)
    parser.add_argument("--min-C", type=int, default=50)
    parser.add_argument("--min-D", type=int, default=500)
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    ae_path = Path(args.ae_ckpt).resolve()
    data_path = Path(args.data).resolve()
    weight_path = Path(args.gen_weight_path).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    plot_dir = outdir / "plots"
    plot_dir.mkdir()

    candidates = discover_checkpoints(checkpoint_dir, args.epochs)
    print("Checkpoints:", flush=True)
    for epoch, path, role in candidates:
        print(f"  epoch {epoch:3d} ({role}): {path}", flush=True)

    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    for key in ("pf", "obj", "label"):
        if key not in raw:
            raise KeyError(f"Training sample is missing {key!r}.")
    labels = raw["label"].long().reshape(-1)
    signature = sample_signature(raw)
    physics_weights, weight_metadata = load_generator_weights(
        str(weight_path), labels, qcd_label=args.qcd_label, sample=raw)

    first_metadata = torch.load(
        candidates[0][1], map_location="cpu", weights_only=False)
    train_indices, validation_indices = checkpoint_reference_indices(
        first_metadata, signature, weight_metadata)
    expected_ae = first_metadata.get("ae_checkpoint_sha256")
    if not expected_ae or file_sha256(ae_path) != expected_ae:
        raise ValueError("The supplied AE is not the AE used by these checkpoints.")

    train_qcd_indices = train_indices[
        labels[torch.as_tensor(train_indices)].numpy() == args.qcd_label]
    validation_qcd_indices = validation_indices[
        labels[torch.as_tensor(validation_indices)].numpy() == args.qcd_label]
    if len(train_qcd_indices) < 10 or len(validation_qcd_indices) < 10:
        raise ValueError("Saved split has too few QCD events for a closure scan.")
    print(
        f"QCD rows: MD fit={len(train_qcd_indices)}, "
        f"validation report={len(validation_qcd_indices)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_ae(str(ae_path), first_metadata["ae_scaler"], device)
    validation_ae = _compute_ae_scores(
        ae, first_metadata["ae_scaler"], raw["obj"],
        validation_qcd_indices, device, args.ae_batch_size)
    del ae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validation_weights = physics_weights[
        torch.as_tensor(validation_qcd_indices)].numpy().astype(np.float64)
    percentiles = np.linspace(0.50, 0.98, 48)
    minimums = {
        "A": args.min_A, "B": args.min_B,
        "C": args.min_C, "D": args.min_D,
    }
    results = []
    reference_split = (train_indices, validation_indices)
    training_commit = None

    for epoch, checkpoint_path, role in candidates:
        print(f"\n=== Epoch {epoch} ({role}) ===", flush=True)
        model, metadata = load_nurd_model(str(checkpoint_path), device)
        candidate_split = checkpoint_reference_indices(
            metadata, signature, weight_metadata)
        if not all(np.array_equal(left, right) for left, right in zip(
                reference_split, candidate_split)):
            raise ValueError(f"Checkpoint {checkpoint_path} uses a different split.")
        if metadata.get("ae_checkpoint_sha256") != expected_ae:
            raise ValueError(f"Checkpoint {checkpoint_path} uses a different AE.")
        candidate_commit = metadata.get("config", {}).get("code_commit", "")
        if training_commit is None:
            training_commit = candidate_commit
        elif candidate_commit != training_commit:
            raise ValueError("Checkpoint candidates were produced by different code commits.")

        training_latents = _infer_latents(
            model, raw["pf"], train_qcd_indices, device, args.batch_size)
        validation_latents = _infer_latents(
            model, raw["pf"], validation_qcd_indices, device, args.batch_size)
        mu, transform = _fit_class_transform(
            training_latents,
            np.ones(len(training_latents), dtype=bool),
            args.n_pca,
            "QCD training split",
            weights=physics_weights[torch.as_tensor(train_qcd_indices)].numpy(),
        )
        transformed = (validation_latents - mu) @ transform
        validation_md = np.square(transformed).sum(axis=1).astype(np.float32)
        finite = (
            np.isfinite(validation_ae)
            & np.isfinite(validation_md)
            & np.isfinite(validation_weights)
            & (validation_ae > 0.0)
        )
        metrics = validation_closure_metrics(
            validation_ae[finite], validation_md[finite],
            validation_weights[finite], percentiles, minimums)
        validation_metrics = metadata.get("validation_metrics", {})
        result = {
            "epoch": epoch,
            "checkpoint_role": role,
            "checkpoint": str(checkpoint_path),
            "balanced_accuracy": float(
                validation_metrics.get("balanced_accuracy", np.nan)),
            "weighted_accuracy": float(
                validation_metrics.get("weighted_accuracy", np.nan)),
            "weighted_ce": float(validation_metrics.get("weighted_ce", np.nan)),
            "critic_accuracy": float(
                validation_metrics.get("critic_accuracy", np.nan)),
            "critic_context_only_accuracy": float(
                validation_metrics.get("critic_context_only_accuracy", np.nan)),
            **{key: value for key, value in metrics.items() if key != "grid"},
        }
        results.append(result)
        _save_grid_plot(
            metrics["grid"], percentiles, epoch,
            plot_dir / f"validation_closure_grid_epoch_{epoch:03d}.png")
        print(
            f"grid median/p90={100*result['grid_median_absolute_nonclosure']:.2f}%/"
            f"{100*result['grid_p90_absolute_nonclosure']:.2f}%  "
            f"curve median/p90={100*result['curve_median_absolute_nonclosure']:.2f}%/"
            f"{100*result['curve_p90_absolute_nonclosure']:.2f}%  "
            f"balanced/weighted={100*result['balanced_accuracy']:.2f}%/"
            f"{100*result['weighted_accuracy']:.2f}%", flush=True)

        del model, training_latents, validation_latents
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _save_summary_plots(results, plot_dir)
    central_best = min(
        results,
        key=lambda row: 0.5 * (
            row["grid_median_absolute_nonclosure"]
            + row["curve_median_absolute_nonclosure"]),
    )
    summary = {
        "protocol": "saved_training_split_MD_fit_to_disjoint_validation_QCD",
        "heldout_sample_accessed": False,
        "training_commit": training_commit,
        "data": str(data_path),
        "generator_weights": str(weight_path),
        "ae_checkpoint": str(ae_path),
        "qcd_label": args.qcd_label,
        "n_pca": args.n_pca,
        "percentiles": percentiles.tolist(),
        "statistical_minimums": minimums,
        "diagnostic_lowest_central_nonclosure_epoch": central_best["epoch"],
        "selection_warning": (
            "This diagnostic ranking is not sufficient by itself: require acceptable "
            "classification and confirm exactly once on held-out data."),
        "results": results,
    }
    json_path = outdir / "validation_checkpoint_scan.json"
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    csv_fields = [
        "epoch", "checkpoint_role", "balanced_accuracy", "weighted_accuracy",
        "weighted_ce", "critic_accuracy", "critic_context_only_accuracy",
        "grid_points", "grid_rejected_points",
        "grid_median_absolute_nonclosure", "grid_p90_absolute_nonclosure",
        "curve_points", "curve_rejected_points",
        "curve_median_absolute_nonclosure", "curve_p90_absolute_nonclosure",
        "checkpoint",
    ]
    with (outdir / "validation_checkpoint_scan.csv").open(
            "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Validation checkpoint comparison ===")
    print(" epoch  role       grid med/p90   curve med/p90  bal acc  wgt acc")
    for row in results:
        print(
            f" {row['epoch']:5d}  {row['checkpoint_role']:<9s} "
            f"{100*row['grid_median_absolute_nonclosure']:6.2f}/"
            f"{100*row['grid_p90_absolute_nonclosure']:6.2f}%  "
            f"{100*row['curve_median_absolute_nonclosure']:6.2f}/"
            f"{100*row['curve_p90_absolute_nonclosure']:6.2f}%  "
            f"{100*row['balanced_accuracy']:6.2f}% "
            f"{100*row['weighted_accuracy']:6.2f}%")
    print(
        "Diagnostic lowest central nonclosure: epoch "
        f"{central_best['epoch']} (do not inspect held-out to tune this choice).")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
