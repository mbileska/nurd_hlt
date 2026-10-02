#!/usr/bin/env python3
"""Dissect the short HLT training chain and produce failure-oriented plots.

This is deliberately an analysis driver, not an alternative training
implementation.  It reloads checkpoints made by the normal AE/NURD programs,
rebuilds their saved preprocessing contract, and compares the last
classification/SupCon-only snapshot with a short NURD snapshot.  A separate
critic is also trained on frozen baseline latents to distinguish an
underpowered critic from ineffective encoder pressure.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde, rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset.hlt_smcocktail_dataset import build_hlt_datasets
from models.hlt_autoencoder import HLTAutoencoder
from models.hlt_con import HLTCritic, HLTContrastiveModel
from utils.hlt_density_ratio import (
    critic_context_only_accuracy,
    density_ratio_critic_loss,
    make_density_ratio_examples,
)
from utils.hlt_weights import (
    _assign_strata,
    _transform_nuisance_for_balance,
    apply_joint_balance,
    effective_mass_by_class,
    effective_sample_size_fraction,
    load_generator_weights,
)


CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
COLORS = ("#3B82F6", "#F59E0B", "#EF4444", "#10B981", "#8B5CF6")


def _name(label: int) -> str:
    return CLASS_NAMES.get(int(label), f"class {int(label)}")


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    return float(np.sum(values * weights) / total) if total > 0 else math.nan


def weighted_corr(x, y, weights):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mx, my = weighted_mean(x, weights), weighted_mean(y, weights)
    covariance = weighted_mean((x - mx) * (y - my), weights)
    vx = weighted_mean((x - mx) ** 2, weights)
    vy = weighted_mean((y - my) ** 2, weights)
    denominator = math.sqrt(max(vx * vy, 0.0))
    return float(covariance / denominator) if denominator > 0 else math.nan


def weighted_quantile(values, quantile, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    position = np.searchsorted(cumulative, float(quantile) * cumulative[-1])
    return float(values[order[min(position, len(values) - 1)]])


def validation_closure_metrics(axis_1, axis_2, weights, percentiles, minimums):
    """Small-sample, validation-only ABCD grid with effective-count guards."""
    axis_1 = np.asarray(axis_1, dtype=np.float64)
    axis_2 = np.asarray(axis_2, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    percentiles = np.asarray(percentiles, dtype=np.float64)
    threshold_1 = [weighted_quantile(axis_1, q, weights) for q in percentiles]
    threshold_2 = [weighted_quantile(axis_2, q, weights) for q in percentiles]
    grid = np.full((len(percentiles), len(percentiles)), np.nan)
    curve = []
    for row, cut_1 in enumerate(threshold_1):
        for column, cut_2 in enumerate(threshold_2):
            high_1, high_2 = axis_1 > cut_1, axis_2 > cut_2
            masks = {
                "A": high_1 & high_2,
                "B": high_1 & ~high_2,
                "C": ~high_1 & high_2,
                "D": ~high_1 & ~high_2,
            }
            statistics = {}
            for region, mask in masks.items():
                selected = weights[mask]
                event_yield = float(selected.sum())
                sumw2 = float(np.square(selected).sum())
                statistics[region] = {
                    "yield": event_yield,
                    "sumw2": sumw2,
                    "effective": event_yield ** 2 / sumw2 if sumw2 > 0 else 0.0,
                }
            if not all(statistics[key]["effective"] >= minimums[key]
                       for key in statistics):
                continue
            A, B, C, D = (statistics[key]["yield"] for key in "ABCD")
            if min(A, B, C, D) <= 0:
                continue
            prediction = B * C / D
            nonclosure = abs((A - prediction) / prediction)
            grid[row, column] = nonclosure
            if row == column:
                ratio = prediction / A
                relative_variance = sum(
                    statistics[key]["sumw2"] / statistics[key]["yield"] ** 2
                    for key in "ABCD")
                curve.append({
                    "percentile": float(percentiles[row]),
                    "efficiency": float(A / weights.sum()),
                    "ratio": float(ratio),
                    "uncertainty": float(abs(ratio) * math.sqrt(relative_variance)),
                    "absolute_nonclosure": float(nonclosure),
                })
    finite_grid = grid[np.isfinite(grid)]
    finite_curve = np.asarray(
        [point["absolute_nonclosure"] for point in curve], dtype=np.float64)
    if not finite_grid.size or not finite_curve.size:
        raise RuntimeError("No statistically valid validation closure points were found.")
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


def distance_correlation(x, y, maximum=1200, seed=42):
    """Unweighted sample distance correlation, bounded for diagnostic memory."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) > maximum:
        selected = np.random.default_rng(seed).choice(
            len(x), size=maximum, replace=False)
        x, y = x[selected], y[selected]
    if len(x) < 3:
        return math.nan
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a -= a.mean(axis=0)[None, :] + a.mean(axis=1)[:, None] - a.mean()
    b -= b.mean(axis=0)[None, :] + b.mean(axis=1)[:, None] - b.mean()
    dcov2 = np.mean(a * b)
    dvar_x = np.mean(a * a)
    dvar_y = np.mean(b * b)
    denominator = math.sqrt(max(dvar_x * dvar_y, 0.0))
    return float(math.sqrt(max(dcov2, 0.0) / denominator)) if denominator > 0 else 0.0


def load_ae(path: Path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HLTAutoencoder(payload["ae_config"]).to(device)
    model.load_state_dict(payload["ae"])
    model.eval()
    return model, payload


def load_nurd_model(path: Path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    state = payload["state_dict_model"]
    projection_key = next(
        (key for key in state if key.endswith(".attn.e.weight")), None)
    num_tokens = (
        int(state[projection_key].shape[1] - 1)
        if projection_key is not None else int(config.get("linear_dim", 100)))
    num_classes = int(state["classifier.weight"].shape[0])
    model = HLTContrastiveModel(
        num_classes=num_classes,
        embed_size=config["embed_size"],
        latent_dim=config["latent_dim"],
        proj_dim=config["proj_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        linear_dim=config["linear_dim"],
        num_tokens=num_tokens,
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload


def fit_class_transform(embeddings, mask, n_pca, weights):
    reference = np.asarray(embeddings)[np.asarray(mask)]
    reference_weights = np.asarray(weights, dtype=np.float64)[np.asarray(mask)]
    reference_weights /= reference_weights.sum()
    mean = np.sum(reference * reference_weights[:, None], axis=0)
    centered = reference - mean
    covariance = (centered * reference_weights[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = eigenvectors[:, -int(n_pca):]
    eigenvalues = np.clip(eigenvalues[-int(n_pca):], 1e-6, None)
    return mean, eigenvectors / np.sqrt(eigenvalues)


@torch.no_grad()
def infer(model, features, indices, device, batch_size):
    latents, logits = [], []
    model.eval()
    indices = torch.as_tensor(indices).long()
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start:start + batch_size]
        batch = torch.nan_to_num(
            features[selected].float(), nan=0.0, posinf=0.0, neginf=0.0
        ).to(device)
        latent, logit = model(batch)
        latents.append(latent.cpu())
        logits.append(logit.cpu())
    return torch.cat(latents).numpy(), torch.cat(logits).numpy()


def parse_ae_history(log_path: Path):
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+train=([0-9.eE+-]+)\s+"
        r"val=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)")
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            rows.append({
                "epoch": int(match.group(1)),
                "train": float(match.group(3)),
                "validation": float(match.group(4)),
                "lr": float(match.group(5)),
            })
    return rows


def plot_ae_history(rows, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([row["epoch"] for row in rows], [row["train"] for row in rows],
            "o-", label="train")
    ax.plot([row["epoch"] for row in rows],
            [row["validation"] for row in rows], "s-", label="validation")
    ax.set(xlabel="Epoch", ylabel="Generator-weighted reconstruction MSE",
           title="AE short-run learning curve")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_class_composition(labels, physics_weights, effective_weights, path):
    classes = sorted(np.unique(labels).astype(int).tolist())
    raw = np.asarray([(labels == label).sum() for label in classes], dtype=float)
    physics = np.asarray([
        physics_weights[labels == label].sum() for label in classes])
    effective = np.asarray([
        effective_weights[labels == label].sum() for label in classes])
    raw /= raw.sum()
    physics /= physics.sum()
    effective /= effective.sum()
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, raw, width, label="raw event fraction")
    ax.bar(x, physics, width, label="generator-weighted mass")
    ax.bar(x + width, effective, width, label="NURD effective mass")
    ax.set_xticks(x, [_name(value) for value in classes])
    ax.set(ylabel="Fraction", title="What distribution each training measure sees")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_split_weight_transfer(train_dataset, val_dataset, path):
    classes = sorted(set(train_dataset.labels.tolist()))
    train_mass = effective_mass_by_class(
        train_dataset.labels, train_dataset.effective_weights)
    val_mass = effective_mass_by_class(
        val_dataset.labels, val_dataset.effective_weights)
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x - 0.18, [train_mass[label] for label in classes], 0.36,
           label="training fit")
    ax.bar(x + 0.18, [val_mass[label] for label in classes], 0.36,
           label="validation using training fit")
    ax.axhline(1.0 / len(classes), color="black", linestyle="--",
               label="equal class target")
    ax.set_xticks(x, [_name(label) for label in classes])
    ax.set(ylabel="Effective class-mass fraction",
           title="Does the training-fitted weighting transfer to validation?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_ae_scores(scores, labels, physics_weights, path):
    positive = scores[np.isfinite(scores) & (scores > 0)]
    edges = np.geomspace(max(np.quantile(positive, 0.001), 1e-10),
                         np.quantile(positive, 0.9995), 80)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for index, label in enumerate(sorted(np.unique(labels).astype(int))):
        mask = labels == label
        axes[0].hist(scores[mask], bins=edges, density=True, histtype="step",
                     linewidth=1.6, color=COLORS[index], label=_name(label))
        axes[1].hist(scores[mask], bins=edges, weights=physics_weights[mask],
                     density=True, histtype="step", linewidth=1.6,
                     color=COLORS[index], label=_name(label))
    for ax, title in zip(axes, ("Raw events", "Generator-weighted physics measure")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlabel="AE reconstruction MSE", ylabel="Density", title=title)
        ax.grid(alpha=0.2)
        ax.legend()
    _save(fig, path)


def plot_weighted_ae_cdfs(scores, labels, physics_weights, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for index, label in enumerate(sorted(np.unique(labels).astype(int))):
        mask = (labels == label) & np.isfinite(scores) & (scores > 0)
        order = np.argsort(scores[mask])
        values = scores[mask][order]
        weights = physics_weights[mask][order]
        cumulative = np.cumsum(weights) / weights.sum()
        ax.plot(values, cumulative, color=COLORS[index], linewidth=1.6,
                label=_name(label))
    ax.set_xscale("log")
    ax.set(xlabel="AE reconstruction MSE", ylabel="Generator-weighted CDF",
           title="AE score CDF by physics class")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_nuisance_transform(scores, spec, path):
    transformed = _transform_nuisance_for_balance(scores, spec["transform"]).numpy()
    edges = torch.as_tensor(spec["edges"]).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(scores, bins=80, histtype="stepfilled", alpha=0.65)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Raw AE reconstruction MSE", ylabel="Events",
                title="Continuous nuisance seen by the critic")
    axes[1].hist(transformed, bins=80, histtype="stepfilled", alpha=0.65)
    for edge in edges:
        axes[1].axvline(edge, color="black", alpha=0.22, linewidth=0.8)
    axes[1].set(xlabel="Training-fitted log1p coordinate", ylabel="Events",
                title="Coordinate used only to calculate weights")
    for ax in axes:
        ax.grid(alpha=0.2)
    _save(fig, path)


def unclipped_spec(spec):
    output = copy.deepcopy(spec)
    output["weight_clipping"] = {"enabled": False}
    return output


def plot_weight_diagnostics(
    labels, scores, physics, effective, spec, outdir,
):
    before = apply_joint_balance(
        torch.as_tensor(labels), torch.as_tensor(scores),
        torch.as_tensor(physics), unclipped_spec(spec)).numpy()
    transformed = _transform_nuisance_for_balance(
        torch.as_tensor(scores), spec["transform"])
    strata = _assign_strata(transformed, spec["edges"]).numpy()
    classes = sorted(np.unique(labels).astype(int).tolist())

    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 4.8),
                             sharey=True)
    for index, (ax, label) in enumerate(zip(np.atleast_1d(axes), classes)):
        mask = labels == label
        values = np.concatenate([before[mask], effective[mask]])
        positive = values[values > 0]
        edges = np.geomspace(max(positive.min(), 1e-12), positive.max(), 65)
        ax.hist(before[mask], bins=edges, density=True, histtype="step",
                linewidth=1.5, label="before clipping")
        ax.hist(effective[mask], bins=edges, density=True, histtype="step",
                linewidth=1.5, label="after clipping")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(title=_name(label), xlabel="Effective event weight")
        ax.grid(alpha=0.2)
        if index == 0:
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)
    _save(fig, outdir / "effective_weights_before_after_clipping.png")

    bins = int(torch.as_tensor(spec["edges"]).numel() - 1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for index, label in enumerate(classes):
        raw, gen, post = [], [], []
        for stratum in range(bins):
            mask = (labels == label) & (strata == stratum)
            raw.append(mask.sum())
            gen.append(physics[mask].sum())
            post.append(effective[mask].sum())
        for ax, values, title in zip(
            axes, (raw, gen, post),
            ("Raw rows", "Generator-weighted mass", "Post-clipping effective mass"),
        ):
            ax.step(np.arange(bins), values, where="mid", linewidth=1.4,
                    color=COLORS[index], label=_name(label))
            ax.set_ylabel(title)
            ax.grid(alpha=0.2)
    axes[0].legend(ncol=len(classes))
    axes[-1].set_xlabel("Log-space nuisance stratum (weighting only)")
    _save(fig, outdir / "nuisance_stratum_occupancy_and_mass.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sample = np.arange(len(scores))
    if len(sample) > 30000:
        sample = np.random.default_rng(42).choice(sample, 30000, replace=False)
    scatter = ax.scatter(scores[sample], effective[sample], c=labels[sample],
                         s=4, alpha=0.2, cmap="tab10", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="AE reconstruction MSE", ylabel="Effective NURD weight",
           title="Weight leverage and possible tail domination")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Class label")
    _save(fig, outdir / "ae_score_vs_effective_weight.png")
    return before


@torch.no_grad()
def reconstruction_residuals(ae, scaler, obj, indices, weights, device, maximum=20000):
    indices = torch.as_tensor(indices).long()
    if indices.numel() > maximum:
        selected = torch.randperm(indices.numel(), generator=torch.Generator().manual_seed(42))[:maximum]
        indices = indices[selected]
        weights = torch.as_tensor(weights)[selected]
    else:
        weights = torch.as_tensor(weights)
    flat = torch.nan_to_num(
        obj[indices, :, :4].reshape(indices.numel(), -1).float(),
        nan=0.0, posinf=0.0, neginf=0.0)
    mu = torch.as_tensor(scaler["mu"]).float().reshape(1, -1)
    std = torch.as_tensor(scaler["std"]).float().reshape(1, -1)
    normalized = (flat - mu) / std.clamp(min=1e-8)
    residual_sum = torch.zeros(normalized.shape[1])
    weight_sum = weights.double().sum().clamp(min=1e-12)
    for start in range(0, len(normalized), 4096):
        batch = normalized[start:start + 4096].to(device)
        reconstruction, _ = ae(batch)
        squared = (reconstruction - batch).square().cpu()
        batch_weights = weights[start:start + 4096].double().reshape(-1, 1)
        residual_sum += (squared.double() * batch_weights).sum(dim=0).float()
    return (residual_sum.double() / weight_sum).numpy()


def plot_feature_residuals(residuals, path):
    order = np.argsort(residuals)[::-1][:30]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(np.arange(len(order)), residuals[order])
    ax.set_xticks(np.arange(len(order)), [str(value) for value in order], rotation=90)
    ax.set(xlabel="Flattened object-feature index", ylabel="Weighted squared residual",
           title="Largest AE reconstruction residuals (top 30)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def model_metrics(logits, labels, weights):
    predictions = logits.argmax(axis=1)
    classes = sorted(np.unique(labels).astype(int).tolist())
    per_class = {
        str(label): float(np.mean(predictions[labels == label] == label))
        for label in classes
    }
    return {
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "weighted_accuracy": weighted_mean(predictions == labels, weights),
        "per_class_accuracy": per_class,
    }


def plot_confusion(logits, labels, weights, title, path):
    classes = sorted(np.unique(labels).astype(int).tolist())
    matrix = confusion_matrix(
        labels, logits.argmax(axis=1), labels=classes, sample_weight=weights,
        normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(classes)):
        for column in range(len(classes)):
            ax.text(column, row, f"{100*matrix[row, column]:.1f}%",
                    ha="center", va="center",
                    color="white" if matrix[row, column] > 0.5 else "black")
    ax.set_xticks(range(len(classes)), [_name(value) for value in classes], rotation=30)
    ax.set_yticks(range(len(classes)), [_name(value) for value in classes])
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.colorbar(image, ax=ax, label="Row-normalized weighted fraction")
    _save(fig, path)


def plot_latent_space(latents, labels, weights, title, outdir):
    sample = np.arange(len(latents))
    if len(sample) > 15000:
        sample = np.random.default_rng(42).choice(sample, 15000, replace=False)
    reducer = PCA(n_components=2, random_state=42)
    reduced = reducer.fit_transform(latents[sample])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for index, label in enumerate(sorted(np.unique(labels).astype(int))):
        mask = labels[sample] == label
        ax.scatter(reduced[mask, 0], reduced[mask, 1], s=4, alpha=0.22,
                   color=COLORS[index], label=_name(label), rasterized=True)
    ax.set(xlabel="Latent PCA 1", ylabel="Latent PCA 2", title=title)
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)
    _save(fig, outdir / "latent_pca_by_class.png")

    norms = np.linalg.norm(latents, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for index, label in enumerate(sorted(np.unique(labels).astype(int))):
        mask = labels == label
        ax.hist(norms[mask], bins=70, weights=weights[mask], density=True,
                histtype="step", color=COLORS[index], linewidth=1.5,
                label=_name(label))
    ax.set(xlabel="Latent L2 norm", ylabel="Weighted density",
           title="Collapse / scale diagnostic")
    ax.grid(alpha=0.2)
    ax.legend()
    _save(fig, outdir / "latent_norm_by_class.png")

    classes = sorted(np.unique(labels).astype(int).tolist())
    centroids = []
    for label in classes:
        mask = labels == label
        centroids.append(np.average(latents[mask], axis=0, weights=weights[mask]))
    centroids = np.asarray(centroids)
    unit = centroids / np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-12)
    distances = 1.0 - unit @ unit.T
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(distances, cmap="magma", vmin=0)
    ax.set_xticks(range(len(classes)), [_name(value) for value in classes], rotation=30)
    ax.set_yticks(range(len(classes)), [_name(value) for value in classes])
    ax.set_title("Cosine distance between weighted class centroids")
    fig.colorbar(image, ax=ax)
    _save(fig, outdir / "class_centroid_distance.png")


def md_from_reference(train_latents, train_labels, train_weights,
                      val_latents, qcd_label, n_pca):
    mask = train_labels == qcd_label
    mean, transform = fit_class_transform(
        train_latents, mask, n_pca, train_weights)
    return np.square((val_latents - mean) @ transform).sum(axis=1)


def weighted_profile(x, y, weights, bins=20):
    x = np.asarray(x)
    y = np.asarray(y)
    weights = np.asarray(weights)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (x > 0) & (weights > 0)
    x, y, weights = x[finite], y[finite], weights[finite]
    edges = np.geomspace(np.quantile(x, 0.005), np.quantile(x, 0.995), bins + 1)
    centers, means, errors = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (x >= low) & (x < high)
        if mask.sum() < 10 or weights[mask].sum() <= 0:
            continue
        mean = weighted_mean(y[mask], weights[mask])
        variance = weighted_mean((y[mask] - mean) ** 2, weights[mask])
        ess = weights[mask].sum() ** 2 / np.square(weights[mask]).sum()
        centers.append(math.sqrt(low * high))
        means.append(mean)
        errors.append(math.sqrt(max(variance, 0.0) / max(ess, 1.0)))
    return np.asarray(centers), np.asarray(means), np.asarray(errors)


def independence_plots(ae_scores, md, weights, title, outdir):
    finite = np.isfinite(ae_scores) & np.isfinite(md) & np.isfinite(weights) \
        & (ae_scores > 0) & (weights > 0)
    ae_scores, md, weights = ae_scores[finite], md[finite], weights[finite]
    x_edges = np.geomspace(np.quantile(ae_scores, 0.005),
                           np.quantile(ae_scores, 0.995), 26)
    y_edges = np.quantile(md, np.linspace(0.005, 0.995, 26))
    y_edges = np.unique(y_edges)
    observed, _, _ = np.histogram2d(ae_scores, md, bins=(x_edges, y_edges),
                                     weights=weights)
    raw_counts, _, _ = np.histogram2d(ae_scores, md, bins=(x_edges, y_edges))
    total = observed.sum()
    expected = observed.sum(axis=1)[:, None] * observed.sum(axis=0)[None, :] / max(total, 1e-12)
    expected_counts = (
        raw_counts.sum(axis=1)[:, None] * raw_counts.sum(axis=0)[None, :]
        / max(raw_counts.sum(), 1.0))
    residual = np.log2((observed + 1e-12) / (expected + 1e-12))
    residual[(raw_counts < 3) | (expected_counts < 3)] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mesh = ax.pcolormesh(x_edges, y_edges, observed.T,
                         norm=LogNorm(vmin=max(observed[observed > 0].min(), 1e-12),
                                      vmax=max(observed.max(), 1e-11)),
                         cmap="viridis", shading="auto")
    ax.set_xscale("log")
    ax.set(xlabel="AE reconstruction MSE", ylabel="QCD-reference MD",
           title=f"{title}: weighted density")
    fig.colorbar(mesh, ax=ax, label="Generator-weighted QCD yield")
    _save(fig, outdir / "qcd_ae_vs_md_density.png")

    # KDE is evaluated in log(AE)-MD space on a bounded sample.  It complements
    # the exact weighted histogram by making narrow ridges/hotspots visible.
    kde_indices = np.arange(len(ae_scores))
    if len(kde_indices) > 4000:
        kde_indices = np.random.default_rng(42).choice(
            kde_indices, 4000, replace=False)
    kde_x = np.log10(ae_scores[kde_indices])
    kde_y = md[kde_indices]
    kde_weights = weights[kde_indices]
    try:
        estimator = gaussian_kde(
            np.vstack([kde_x, kde_y]), weights=kde_weights)
        grid_x = np.linspace(np.quantile(kde_x, 0.005), np.quantile(kde_x, 0.995), 90)
        grid_y = np.linspace(np.quantile(kde_y, 0.005), np.quantile(kde_y, 0.995), 90)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        density = estimator(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        contour = ax.contourf(10.0 ** mesh_x, mesh_y, density, levels=30, cmap="magma")
        ax.set_xscale("log")
        ax.set(xlabel="AE reconstruction MSE", ylabel="QCD-reference MD",
               title=f"{title}: weighted KDE / ridge diagnostic")
        fig.colorbar(contour, ax=ax, label="KDE density")
        _save(fig, outdir / "qcd_ae_vs_md_kde.png")
    except np.linalg.LinAlgError:
        pass

    finite_residual = residual[np.isfinite(residual) & (expected > 0)]
    limit = (
        max(float(np.nanpercentile(np.abs(finite_residual), 95)), 0.25)
        if finite_residual.size else 1.0)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mesh = ax.pcolormesh(
        x_edges, y_edges, residual.T, cmap="coolwarm", shading="auto",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    ax.set_xscale("log")
    ax.set(xlabel="AE reconstruction MSE", ylabel="QCD-reference MD",
           title=f"{title}: independence hotspot map")
    fig.colorbar(mesh, ax=ax, label="log2(observed / factorized expectation)")
    _save(fig, outdir / "qcd_independence_hotspots.png")

    centers, means, errors = weighted_profile(ae_scores, md, weights)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(centers, means, yerr=errors, fmt="o-", markersize=3, capsize=2)
    ax.set_xscale("log")
    ax.set(xlabel="AE reconstruction MSE", ylabel="Weighted mean MD",
           title=f"{title}: QCD profile")
    ax.grid(alpha=0.25)
    _save(fig, outdir / "qcd_profile_md_vs_ae.png")

    metrics = {
        "weighted_pearson": weighted_corr(np.log1p(ae_scores), md, weights),
        "weighted_spearman": weighted_corr(
            rankdata(ae_scores), rankdata(md), weights),
        "distance_correlation_sample": distance_correlation(
            np.log1p(ae_scores), md),
    }
    return metrics


def closure_plots(ae_scores, md, weights, title, outdir):
    percentiles = np.linspace(0.50, 0.95, 10)
    try:
        result = validation_closure_metrics(
            ae_scores, md, weights, percentiles,
            minimums={"A": 5, "B": 5, "C": 5, "D": 20})
    except RuntimeError as error:
        return {"available": False, "reason": str(error)}
    grid = result.pop("grid")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    values = 100.0 * grid
    finite = values[np.isfinite(values)]
    vmax = max(float(np.percentile(finite, 95)), 1.0)
    mesh = ax.pcolormesh(percentiles, percentiles, values.T, cmap="viridis_r",
                         vmin=0, vmax=vmax, shading="auto")
    ax.set(xlabel="AE percentile", ylabel="MD percentile",
           title=f"{title}: validation-only QCD non-closure")
    fig.colorbar(mesh, ax=ax, label="Absolute non-closure (%)")
    _save(fig, outdir / "qcd_closure_grid.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    curve = sorted(result["curve"], key=lambda point: point["efficiency"])
    ax.errorbar([point["efficiency"] for point in curve],
                [point["ratio"] for point in curve],
                yerr=[point["uncertainty"] for point in curve], fmt="o-")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set(xlabel="Weighted QCD efficiency in A",
           ylabel="ABCD predicted / observed", title=f"{title}: closure curve")
    ax.grid(alpha=0.25)
    _save(fig, outdir / "qcd_closure_curve.png")
    return {"available": True, **result}


def plot_training_history(history_path, switch_epoch, path):
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = np.asarray([row["epoch"] for row in history])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panels = (
        ("weighted_ce", "Weighted CE"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("contrastive_loss", "SupCon loss"),
        ("critic_accuracy", "Critic accuracy"),
    )
    for ax, (key, ylabel) in zip(axes.flat, panels):
        if key == "balanced_accuracy":
            for split, style in (("train", "o-"), ("validation", "s-")):
                ax.plot(epochs, [row[split][key] for row in history], style,
                        markersize=3, label=f"{split} balanced")
            ax.plot(epochs, [row["validation"]["weighted_accuracy"] for row in history],
                    "^-", markersize=3, label="validation weighted")
        else:
            for split, style in (("train", "o-"), ("validation", "s-")):
                ax.plot(epochs, [row[split][key] for row in history], style,
                        markersize=3, label=split)
        ax.axvline(switch_epoch, color="black", linestyle="--", linewidth=1,
                   label="NURD enabled" if key == "weighted_ce" else None)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    axes[0, 0].legend()
    _save(fig, path)
    return history


def critic_predictions(critic, latent, labels, nuisance, weights, device,
                       batch_size, seed=42):
    critic.eval()
    scores, targets, output_weights = [], [], []
    output_labels, output_nuisance = [], []
    context_numerator, context_denominator = 0.0, 0.0
    generator = torch.Generator().manual_seed(seed)
    order = torch.arange(len(latent))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            z = torch.as_tensor(latent[selected]).float().to(device)
            y = torch.as_tensor(labels[selected]).long().to(device)
            n = torch.as_tensor(nuisance[selected]).float().to(device)
            w = torch.as_tensor(weights[selected]).float().to(device)
            permutation = torch.randperm(len(selected), generator=generator).to(device)
            examples = make_density_ratio_examples(
                z, y, n, w, permutation=permutation, shuffle_mode="global")
            ex_z, ex_y, ex_n, ex_w, ex_target = examples
            logits = critic(ex_z, ex_y, ex_n)
            scores.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            targets.append(ex_target.cpu().numpy())
            output_weights.append(ex_w.cpu().numpy())
            output_labels.append(ex_y.cpu().numpy())
            output_nuisance.append(ex_n.cpu().numpy())
            context = critic_context_only_accuracy(
                critic, z, y, n, w, permutation=permutation,
                shuffle_mode="global")
            context_numerator += float(context) * float(ex_w.sum())
            context_denominator += float(ex_w.sum())
    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    output_weights = np.concatenate(output_weights)
    output_labels = np.concatenate(output_labels)
    output_nuisance = np.concatenate(output_nuisance)
    auc = roc_auc_score(targets, scores, sample_weight=output_weights)
    accuracy = weighted_mean((scores >= 0.5) == targets, output_weights)
    return {
        "scores": scores,
        "targets": targets,
        "weights": output_weights,
        "labels": output_labels,
        "nuisance": output_nuisance,
        "auc": float(auc),
        "accuracy": float(accuracy),
        "context_only_accuracy": context_numerator / max(context_denominator, 1e-12),
    }


def train_frozen_critic(
    train_latent, train_labels, train_nuisance, train_weights,
    val_latent, val_labels, val_nuisance, val_weights,
    latent_dim, num_classes, epochs, batch_size, learning_rate, device,
):
    torch.manual_seed(42)
    critic = HLTCritic(latent_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    dataset = TensorDataset(
        torch.as_tensor(train_latent).float(),
        torch.as_tensor(train_labels).long(),
        torch.as_tensor(train_nuisance).float(),
        torch.as_tensor(train_weights).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False, generator=torch.Generator().manual_seed(42))
    history = []
    for epoch in range(1, epochs + 1):
        critic.train()
        losses = []
        for latent, labels, nuisance, weights in loader:
            latent, labels = latent.to(device), labels.to(device)
            nuisance, weights = nuisance.to(device), weights.to(device)
            optimizer.zero_grad()
            loss, _accuracy, _ = density_ratio_critic_loss(
                critic, latent, labels, nuisance, weights,
                shuffle_mode="global")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = critic_predictions(
            critic, val_latent, val_labels, val_nuisance, val_weights,
            device, batch_size)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_auc": validation["auc"],
            "validation_accuracy": validation["accuracy"],
            "context_only_accuracy": validation["context_only_accuracy"],
        })
    return critic, history, validation


def plot_critic_diagnostics(history, predictions, title, outdir):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(epochs, [row["train_loss"] for row in history], "o-")
    axes[0].axhline(math.log(2), color="black", linestyle="--", label="random CE")
    axes[0].set(xlabel="Probe epoch", ylabel="Critic loss", title="Frozen-latent probe loss")
    axes[0].legend()
    for key, label, marker in (
        ("validation_auc", "validation AUC", "o-"),
        ("validation_accuracy", "validation accuracy", "s-"),
        ("context_only_accuracy", "context-only accuracy", "^-"),
    ):
        axes[1].plot(epochs, [row[key] for row in history], marker, label=label)
    axes[1].axhline(0.5, color="black", linestyle="--")
    axes[1].set(xlabel="Probe epoch", ylabel="Score", ylim=(0.4, 1.01),
                title="Can a fresh critic find nuisance information?")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, outdir / "critic_learning_curves.png")

    scores = predictions["scores"]
    targets = predictions["targets"]
    weights = predictions["weights"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].hist(scores[targets == 1], bins=60, weights=weights[targets == 1],
                 density=True, histtype="step", linewidth=1.6, label="real tuples")
    axes[0].hist(scores[targets == 0], bins=60, weights=weights[targets == 0],
                 density=True, histtype="step", linewidth=1.6, label="shuffled nuisance")
    axes[0].set(xlabel="P(real)", ylabel="Weighted density", title=title)
    axes[0].legend()
    fpr, tpr, _ = roc_curve(targets, scores, sample_weight=weights)
    axes[1].plot(fpr, tpr, label=f"AUC = {predictions['auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate",
                title="Real-vs-shuffled ROC")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, outdir / "critic_scores_and_roc.png")

    labels = predictions["labels"]
    nuisance = predictions["nuisance"]
    classes = sorted(np.unique(labels).astype(int).tolist())
    per_class_auc, per_class_accuracy = [], []
    for label in classes:
        mask = labels == label
        per_class_auc.append(roc_auc_score(
            targets[mask], scores[mask], sample_weight=weights[mask]))
        per_class_accuracy.append(weighted_mean(
            (scores[mask] >= 0.5) == targets[mask], weights[mask]))
    quantile_edges = np.unique(np.quantile(nuisance, np.linspace(0, 1, 11)))
    stratum = np.clip(np.digitize(nuisance, quantile_edges[1:-1]), 0,
                      max(len(quantile_edges) - 2, 0))
    by_stratum = []
    for value in range(max(len(quantile_edges) - 1, 1)):
        mask = stratum == value
        by_stratum.append(weighted_mean(
            (scores[mask] >= 0.5) == targets[mask], weights[mask])
            if mask.any() else math.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(classes))
    axes[0].bar(x - 0.18, per_class_auc, 0.36, label="AUC")
    axes[0].bar(x + 0.18, per_class_accuracy, 0.36, label="accuracy")
    axes[0].axhline(0.5, color="black", linestyle="--")
    axes[0].set_xticks(x, [_name(value) for value in classes])
    axes[0].set(ylabel="Critic performance", title="Shortcuts or failures by class",
                ylim=(0.35, 1.0))
    axes[0].legend()
    axes[1].plot(np.arange(len(by_stratum)), by_stratum, "o-")
    axes[1].axhline(0.5, color="black", linestyle="--")
    axes[1].set(xlabel="Validation nuisance decile", ylabel="Critic accuracy",
                title="Tail-dependent critic performance", ylim=(0.35, 1.0))
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, outdir / "critic_performance_by_class_and_nuisance.png")


def compare_stages(baseline, after, labels, weights, outdir):
    drift = np.linalg.norm(after["latents"] - baseline["latents"], axis=1)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for index, label in enumerate(sorted(np.unique(labels).astype(int))):
        mask = labels == label
        ax.hist(drift[mask], bins=70, weights=weights[mask], density=True,
                histtype="step", linewidth=1.5, color=COLORS[index],
                label=_name(label))
    ax.set(xlabel="Per-event latent L2 change", ylabel="Weighted density",
           title="How strongly the short NURD phase moved the representation")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, outdir / "latent_drift_after_nurd.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].scatter(baseline["md"], after["md"], s=3, alpha=0.15, rasterized=True)
    limit = np.quantile(np.concatenate([baseline["md"], after["md"]]), 0.995)
    axes[0].plot([0, limit], [0, limit], "k--")
    axes[0].set(xlim=(0, limit), ylim=(0, limit), xlabel="Baseline MD",
                ylabel="After-NURD MD", title="Event-level anomaly-score movement")
    metric_names = ["weighted_pearson", "weighted_spearman",
                    "distance_correlation_sample"]
    x = np.arange(len(metric_names))
    axes[1].bar(x - 0.18, [baseline["dependence"][key] for key in metric_names],
                0.36, label="SupCon only")
    axes[1].bar(x + 0.18, [after["dependence"][key] for key in metric_names],
                0.36, label="After NURD")
    axes[1].set_xticks(x, ["Pearson", "Spearman", "dCorr"])
    axes[1].set(ylabel="Dependence (lower is better)", title="QCD dependence change")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, outdir / "before_after_nurd_summary.png")
    return {
        "weighted_mean_latent_drift": weighted_mean(drift, weights),
        "weighted_p95_latent_drift": float(np.quantile(drift, 0.95)),
    }


def checkpoint_stage(
    name, checkpoint_path, raw, train_dataset, val_dataset, preprocessing,
    device, batch_size, n_pca, outdir,
):
    outdir.mkdir(parents=True, exist_ok=False)
    model, metadata = load_nurd_model(str(checkpoint_path), device)
    train_latents, _ = infer(
        model, raw["pf"], train_dataset.indices, device, batch_size)
    val_latents, val_logits = infer(
        model, raw["pf"], val_dataset.indices, device, batch_size)
    train_labels = train_dataset.labels.numpy()
    val_labels = val_dataset.labels.numpy()
    train_physics = train_dataset.physics_weights.numpy().astype(np.float64)
    val_physics = val_dataset.physics_weights.numpy().astype(np.float64)
    val_effective = val_dataset.effective_weights.numpy().astype(np.float64)
    qcd_label = int(preprocessing["qcd_label"])
    md = md_from_reference(
        train_latents, train_labels, train_physics,
        val_latents, qcd_label, n_pca)
    ae_scores = val_dataset.ae_reco.numpy()
    qcd = val_labels == qcd_label

    metrics = model_metrics(val_logits, val_labels, val_effective)
    plot_confusion(val_logits, val_labels, val_effective,
                   f"{name}: effective-weight validation confusion",
                   outdir / "classification_confusion.png")
    plot_latent_space(val_latents, val_labels, val_effective, name, outdir)
    dependence = independence_plots(
        ae_scores[qcd], md[qcd], val_physics[qcd], name, outdir)
    closure = closure_plots(
        ae_scores[qcd], md[qcd], val_physics[qcd], name, outdir)
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(metadata["epoch"]),
        "classification": metrics,
        "dependence": dependence,
        "closure": closure,
        "latents": val_latents,
        "logits": val_logits,
        "md": md,
        "metadata": metadata,
        "train_latents": train_latents,
    }


def serializable_stage(stage):
    return {
        key: value for key, value in stage.items()
        if key not in {"latents", "logits", "md", "metadata", "train_latents"}
    }


def plot_dashboard(summary, path):
    baseline = summary["stages"]["supcon_only"]
    after = summary["stages"]["after_nurd"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].axis("off")
    lines = [
        "DIAGNOSTIC SUMMARY",
        f"Events: {summary['data']['events']:,}",
        f"Train ESS/N: {summary['weights']['train_ess_fraction']:.4f}",
        f"Weights clipped: {summary['weights']['fraction_clipped_overall']:.3%}",
        "",
        f"Baseline bal. acc: {baseline['classification']['balanced_accuracy']:.3f}",
        f"After NURD bal. acc: {after['classification']['balanced_accuracy']:.3f}",
        f"Fresh probe AUC before: {summary['critic_probe_before_nurd']['auc']:.3f}",
        f"Fresh probe AUC after:  {summary['critic_probe_after_nurd']['auc']:.3f}",
        f"Coupled critic AUC:     {summary['coupled_critic']['auc']:.3f}",
    ]
    axes[0, 0].text(0.03, 0.97, "\n".join(lines), va="top", family="monospace")

    stages = (baseline, after)
    axes[0, 1].bar([0, 1], [row["classification"]["balanced_accuracy"] for row in stages])
    axes[0, 1].set_xticks([0, 1], ["SupCon only", "After NURD"])
    axes[0, 1].set(ylabel="Balanced accuracy", ylim=(0, 1))
    axes[1, 0].bar([0, 1], [row["dependence"]["distance_correlation_sample"] for row in stages])
    axes[1, 0].set_xticks([0, 1], ["SupCon only", "After NURD"])
    axes[1, 0].set(ylabel="QCD dCorr", title="Lower is better")
    axes[1, 1].bar(
        [0, 1, 2],
        [summary["critic_probe_before_nurd"]["auc"],
         summary["critic_probe_after_nurd"]["auc"],
         summary["coupled_critic"]["auc"]])
    axes[1, 1].axhline(0.5, color="black", linestyle="--")
    axes[1, 1].set_xticks(
        [0, 1, 2], ["Fresh probe\nbefore", "Fresh probe\nafter", "Coupled\ncritic"])
    axes[1, 1].set(ylabel="Real-vs-shuffled AUC", ylim=(0.45, 1.0))
    for ax in axes.flat[1:]:
        ax.grid(alpha=0.25)
    _save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a staged diagnostic run.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--gen-weight-path", required=True)
    parser.add_argument("--ae-ckpt", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--baseline-epoch", required=True, type=int)
    parser.add_argument("--nurd-epoch", required=True, type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--qcd-label", type=int, default=1)
    parser.add_argument("--balance-strata", type=int, default=20)
    parser.add_argument("--balance-binning", default="log_fixed")
    parser.add_argument("--balance-clip-quantile", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    parser.add_argument("--n-pca", type=int, default=6)
    parser.add_argument("--critic-probe-epochs", type=int, default=3)
    parser.add_argument("--critic-probe-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir).resolve()
    if (outdir / "plots").exists() or (outdir / "diagnostic_summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    stages_dir = outdir / "plots"
    ae_dir = stages_dir / "01_data_and_ae"
    baseline_dir = stages_dir / "02_supcon_only"
    critic_dir = stages_dir / "03_frozen_critic_probe"
    after_dir = stages_dir / "04_after_nurd"
    comparison_dir = stages_dir / "05_comparison"
    for directory in (ae_dir, critic_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=False)

    data_path = Path(args.data).resolve()
    weight_path = Path(args.gen_weight_path).resolve()
    ae_path = Path(args.ae_ckpt).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    baseline_path = checkpoint_dir / f"checkpoint_epoch_{args.baseline_epoch:03d}.pth.tar"
    after_path = checkpoint_dir / f"checkpoint_epoch_{args.nurd_epoch:03d}.pth.tar"
    for path in (data_path, weight_path, ae_path, baseline_path, after_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    labels_original = raw["label"].long().reshape(-1)
    _physics_all, generator_metadata = load_generator_weights(
        str(weight_path), labels_original, qcd_label=args.qcd_label, sample=raw)
    ae, ae_payload = load_ae(ae_path, device)
    train_dataset, val_dataset, preprocessing = build_hlt_datasets(
        str(data_path), ae, seed=args.seed,
        gen_weight_path=str(weight_path), qcd_label=args.qcd_label,
        ae_scaler=ae_payload["ae_scaler"],
        balance_strata=args.balance_strata,
        balance_binning=args.balance_binning,
        balance_clip_quantile=args.balance_clip_quantile,
        ae_batch_size=args.ae_batch_size,
    )

    train_labels = train_dataset.labels.numpy()
    train_scores = train_dataset.ae_reco.numpy()
    train_physics = train_dataset.physics_weights.numpy().astype(np.float64)
    train_effective = train_dataset.effective_weights.numpy().astype(np.float64)
    balance_spec = preprocessing["weighting"]["balance_spec"]
    plot_class_composition(
        train_labels, train_physics, train_effective,
        ae_dir / "class_composition_by_measure.png")
    plot_split_weight_transfer(
        train_dataset, val_dataset,
        ae_dir / "effective_class_mass_train_vs_validation.png")
    plot_ae_scores(
        train_scores, train_labels, train_physics,
        ae_dir / "ae_score_distributions.png")
    plot_weighted_ae_cdfs(
        train_scores, train_labels, train_physics,
        ae_dir / "ae_score_weighted_cdfs.png")
    plot_nuisance_transform(
        train_scores, balance_spec,
        ae_dir / "ae_score_raw_vs_weighting_coordinate.png")
    preclip = plot_weight_diagnostics(
        train_labels, train_scores, train_physics, train_effective,
        balance_spec, ae_dir)
    residuals = reconstruction_residuals(
        ae, ae_payload["ae_scaler"], raw["obj"], train_dataset.indices,
        train_physics, device)
    plot_feature_residuals(residuals, ae_dir / "ae_feature_residuals.png")
    ae_history = parse_ae_history(ae_path.parent / "ae_train.log")
    if ae_history:
        plot_ae_history(ae_history, ae_dir / "ae_learning_curve.png")

    history = plot_training_history(
        checkpoint_dir / "training_history.json", args.nurd_epoch,
        comparison_dir / "staged_training_curves.png")
    baseline = checkpoint_stage(
        "SupCon/classifier only", baseline_path, raw, train_dataset,
        val_dataset, preprocessing, device, args.batch_size, args.n_pca,
        baseline_dir)
    after = checkpoint_stage(
        "After short NURD phase", after_path, raw, train_dataset,
        val_dataset, preprocessing, device, args.batch_size, args.n_pca,
        after_dir)

    val_labels = val_dataset.labels.numpy()
    val_nuisance = val_dataset.nuisance.numpy()
    val_effective = val_dataset.effective_weights.numpy()
    _probe, probe_history, probe_predictions = train_frozen_critic(
        baseline["train_latents"], train_labels, train_dataset.nuisance.numpy(),
        train_effective, baseline["latents"], val_labels, val_nuisance,
        val_effective, baseline["latents"].shape[1],
        len(np.unique(train_labels)), args.critic_probe_epochs,
        args.batch_size, args.critic_probe_lr, device)
    plot_critic_diagnostics(
        probe_history, probe_predictions,
        "Fresh critic on frozen SupCon-only latents", critic_dir)

    after_probe_dir = after_dir / "fresh_frozen_critic_probe"
    after_probe_dir.mkdir()
    _after_probe, after_probe_history, after_probe_predictions = train_frozen_critic(
        after["train_latents"], train_labels, train_dataset.nuisance.numpy(),
        train_effective, after["latents"], val_labels, val_nuisance,
        val_effective, after["latents"].shape[1],
        len(np.unique(train_labels)), args.critic_probe_epochs,
        args.batch_size, args.critic_probe_lr, device)
    plot_critic_diagnostics(
        after_probe_history, after_probe_predictions,
        "Fresh critic on frozen post-NURD latents", after_probe_dir)

    coupled_dir = after_dir / "coupled_training_critic"
    coupled_dir.mkdir()
    final_critic = HLTCritic(
        after["latents"].shape[1], len(np.unique(train_labels))).to(device)
    final_critic.load_state_dict(after["metadata"]["state_dict_critic"])
    coupled_predictions = critic_predictions(
        final_critic, after["latents"], val_labels, val_nuisance,
        val_effective, device, args.batch_size)
    plot_critic_diagnostics(
        [{"epoch": args.nurd_epoch, "train_loss": float(
            after["metadata"].get("train_metrics", {}).get("critic_loss", math.nan)),
          "validation_auc": coupled_predictions["auc"],
          "validation_accuracy": coupled_predictions["accuracy"],
          "context_only_accuracy": coupled_predictions["context_only_accuracy"]}],
        coupled_predictions, "Coupled critic after short NURD phase",
        coupled_dir)

    comparison = compare_stages(
        baseline, after, val_labels, val_effective, comparison_dir)
    clipped_by_class = balance_spec["weight_clipping"].get(
        "training_fraction_clipped", {})
    fraction_clipped = float(sum(
        float(clipped_by_class.get(int(label), 0.0)) * int((train_labels == label).sum())
        for label in np.unique(train_labels)
    ) / max(len(train_labels), 1))
    summary = {
        "protocol": "short_staged_training_diagnostic_no_heldout_access",
        "heldout_sample_accessed": False,
        "data": {
            "path": str(data_path),
            "events": int(labels_original.numel()),
            "class_counts": {
                str(label): int((labels_original == label).sum())
                for label in labels_original.unique().tolist()
            },
            "generator_weights": generator_metadata,
        },
        "ae": {
            "checkpoint": str(ae_path),
            "checkpoint_epoch": int(ae_payload.get("epoch", -1)),
            "history": ae_history,
            "score_quantiles": np.quantile(
                train_scores, [0, 0.01, 0.5, 0.99, 1]).tolist(),
        },
        "weights": {
            "method": preprocessing["weighting"]["method"],
            "train_class_mass": effective_mass_by_class(
                torch.as_tensor(train_labels), torch.as_tensor(train_effective)),
            "train_ess_fraction": effective_sample_size_fraction(
                torch.as_tensor(train_effective)),
            "fraction_clipped_overall": fraction_clipped,
            "clipping": balance_spec["weight_clipping"],
        },
        "schedule": {
            "baseline_epoch": args.baseline_epoch,
            "nurd_epoch": args.nurd_epoch,
            "history": history,
        },
        "stages": {
            "supcon_only": serializable_stage(baseline),
            "after_nurd": serializable_stage(after),
        },
        "critic_probe_before_nurd": {
            "purpose": "fresh critic trained with SupCon-only encoder latents frozen",
            "epochs": args.critic_probe_epochs,
            "history": probe_history,
            "auc": probe_predictions["auc"],
            "accuracy": probe_predictions["accuracy"],
            "context_only_accuracy": probe_predictions["context_only_accuracy"],
        },
        "critic_probe_after_nurd": {
            "purpose": "same fresh critic protocol on frozen post-NURD latents",
            "epochs": args.critic_probe_epochs,
            "history": after_probe_history,
            "auc": after_probe_predictions["auc"],
            "accuracy": after_probe_predictions["accuracy"],
            "context_only_accuracy": after_probe_predictions["context_only_accuracy"],
        },
        "coupled_critic": {
            "auc": coupled_predictions["auc"],
            "accuracy": coupled_predictions["accuracy"],
            "context_only_accuracy": coupled_predictions["context_only_accuracy"],
        },
        "comparison": comparison,
        "interpretation": {
            "healthy_ae": "finite smooth scores; decreasing train/validation MSE; no single feature dominates unexpectedly",
            "healthy_weights": "equal NURD class mass, occupied strata, little clipping, and non-negligible ESS/N",
            "healthy_supcon": "falling CE/SupCon losses, separated non-collapsed latent classes, usable balanced accuracy",
            "healthy_critic": "a frozen-latent probe above 0.5 AUC demonstrates learnable nuisance leakage; context-only should remain near 0.5",
            "healthy_nurd": "the matched fresh-probe AUC and QCD AE-MD dependence move toward 0.5/0, hotspot residuals and non-closure shrink without classifier collapse",
        },
    }
    summary_path = outdir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        outdir / "diagnostic_arrays.npz",
        validation_labels=val_labels,
        validation_ae_score=val_dataset.ae_reco.numpy(),
        validation_physics_weight=val_dataset.physics_weights.numpy(),
        validation_effective_weight=val_effective,
        baseline_md=baseline["md"],
        after_nurd_md=after["md"],
        baseline_logits=baseline["logits"],
        after_nurd_logits=after["logits"],
        latent_l2_drift=np.linalg.norm(
            after["latents"] - baseline["latents"], axis=1),
    )
    plot_dashboard(summary, comparison_dir / "diagnostic_dashboard.png")
    print(f"Diagnostic summary: {summary_path}", flush=True)
    print(f"Plots: {stages_dir}", flush=True)


if __name__ == "__main__":
    main()
