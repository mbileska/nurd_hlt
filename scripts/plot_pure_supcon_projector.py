#!/usr/bin/env python3
"""Descriptive held-out KDE of the representation optimized by pure SupCon.

The 2D PCA is fit on a fixed, class-balanced held-out display subset only.
It is never used to choose a checkpoint, threshold, or anomaly score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.hlt_con import HLTContrastiveModel  # noqa: E402
from utils.hlt_data import load_hlt_sample  # noqa: E402
from utils.hlt_targets import checkpoint_target_labels  # noqa: E402


def selected_indices(labels, per_class: int, seed: int) -> tuple[np.ndarray, dict]:
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    indices = []
    counts = {}
    for label in sorted(np.unique(labels).tolist()):
        pool = np.flatnonzero(labels == label)
        chosen = rng.choice(pool, min(per_class, len(pool)), replace=False)
        indices.extend(chosen.tolist())
        counts[str(int(label))] = len(chosen)
    return np.asarray(indices, dtype=np.int64), counts


def load_encoder(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("classification_head_trained") is not False:
        raise ValueError("Expected a checkpoint from a pure-SupCon run.")
    config = checkpoint["config"]
    state = checkpoint["state_dict_model"]
    e_key = next(key for key in state if key.endswith(".attn.e.weight"))
    model = HLTContrastiveModel(
        num_classes=state["classifier.weight"].shape[0],
        embed_size=config["embed_size"],
        latent_dim=config["latent_dim"],
        proj_dim=config["proj_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dim_ff=config["dim_ff"],
        linear_dim=config["linear_dim"],
        num_tokens=state[e_key].shape[1] - 1,
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def projector_embeddings(model, pf, indices, device, batch_size: int):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = torch.as_tensor(indices[start:start + batch_size])
            batch = pf[selected].to(device)
            latent, _ = model(batch)
            chunks.append(model.get_embeddings(latent).cpu().numpy())
    return np.concatenate(chunks)


def plot_kde(projected, labels, names, output: Path):
    classes = sorted(np.unique(labels).tolist())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    x_range = np.quantile(projected[:, 0], [.005, .995])
    y_range = np.quantile(projected[:, 1], [.005, .995])
    xx, yy = np.meshgrid(np.linspace(*x_range, 90),
                         np.linspace(*y_range, 90))
    grid = np.vstack((xx.ravel(), yy.ravel()))
    colors = plt.get_cmap("tab10")
    for ax, label in zip(axes.flat, classes):
        mask = labels == label
        color = colors(label % 10)
        ax.scatter(projected[~mask, 0], projected[~mask, 1],
                   color="0.75", s=1, alpha=.04, rasterized=True)
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   color=color, s=2, alpha=.15, rasterized=True)
        if mask.sum() >= 5:
            try:
                kde = gaussian_kde(projected[mask].T)
                density = kde(grid).reshape(xx.shape)
                if density.max() > 0:
                    ax.contour(xx, yy, density,
                               levels=density.max() * np.array([.15, .4, .7]),
                               colors=[color], linewidths=1.3)
            except np.linalg.LinAlgError:
                ax.text(.03, .03, "KDE singular", transform=ax.transAxes)
        ax.set_title(
            f"{label}: {names.get(label, names.get(str(label), str(label)))} "
            f"(n={mask.sum()})")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.grid(alpha=.2)
        ax.set_xlabel("projector PCA 1")
        ax.set_ylabel("projector PCA 2")
    for ax in list(axes.flat)[len(classes):]:
        ax.set_visible(False)
    fig.suptitle("Pure-SupCon projector embeddings — held-out display subset\n"
                 "Colored class vs gray other classes; PCA is visualization-only")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_class < 5 or args.batch_size < 1:
        parser.error("Need --per-class >= 5 and --batch-size >= 1.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_encoder(args.checkpoint, device)
    sample = load_hlt_sample(args.sample)
    labels_all = checkpoint_target_labels(sample, checkpoint["preprocessing"])
    labels_all = torch.as_tensor(labels_all).cpu().numpy()
    indices, counts = selected_indices(labels_all, args.per_class, args.seed)
    labels = labels_all[indices]
    embeddings = projector_embeddings(
        model, sample["pf"], indices, device, args.batch_size)
    pca = PCA(n_components=2, random_state=args.seed)
    projected = pca.fit_transform(embeddings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "projector_pca_kde_by_class.png"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    names = checkpoint["preprocessing"].get("target_class_names", {})
    plot_kde(projected, labels, names, output)
    metadata = args.output_dir / "projector_pca_kde_metadata.json"
    metadata.write_text(json.dumps({
        "protocol": "descriptive_heldout_projector_pca_kde",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "selection_metric": checkpoint["selection_metric"],
        "sample": str(args.sample),
        "class_counts_displayed": counts,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "selection_or_threshold_use": False,
    }, indent=2) + "\n")
    print(f"Projector KDE: {output}", flush=True)


if __name__ == "__main__":
    main()
