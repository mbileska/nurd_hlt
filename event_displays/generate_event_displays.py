#!/usr/bin/env python3
"""Generate matched PF-level event displays for QCD and TpTp events.

The three displayed categories are tied to the held-out ABCD working point:

* ``usual_qcd``: QCD in region D (below both AE and MD thresholds);
* ``false_positive_qcd``: QCD in region A (above both thresholds); and
* ``tptp``: events from the independent TpTp signal file.

Three reproducibly random, well-matched triplets are selected by total PF pT
and active PF multiplicity.  The plots deliberately say "PF pT flow" rather
than "L1 calorimeter": the input tensors do not contain calorimeter towers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hlt_autoencoder import HLTAutoencoder  # noqa: E402
from models.hlt_con import HLTContrastiveModel  # noqa: E402
from utils.hlt_weights import (  # noqa: E402
    load_generator_weights,
    sample_signature,
)


CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "W+jets"}
CATEGORY_TITLES = {
    "usual_qcd": "Usual QCD",
    "false_positive_qcd": "False-positive QCD",
    "tptp": "TpTp",
}
PDG_STYLE = {
    211: ("charged hadron", "#e66101"),
    130: ("neutral hadron", "#5e3c99"),
    22: ("photon", "#fdb863"),
    11: ("electron", "#1b9e77"),
    13: ("muon", "#377eb8"),
    1: ("ID 1", "#66c2a5"),
    2: ("ID 2", "#fc8d62"),
}
UNKNOWN_STYLE = ("unknown/other", "#777777")
ETA_LIMIT = 5.0
PHI_EDGES = np.linspace(-math.pi, math.pi, 65)
ETA_EDGES = np.linspace(-ETA_LIMIT, ETA_LIMIT, 51)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.linewidth": 1.3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.facecolor": "white",
})


@dataclass
class Scores:
    ae: np.ndarray
    latent: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    sum_pt: np.ndarray
    multiplicity: np.ndarray
    md: np.ndarray | None = None


@dataclass
class DisplayEvent:
    category: str
    source_index: int
    pf: np.ndarray
    ae_residual: np.ndarray
    ae_score: float
    md_score: float
    probabilities: np.ndarray
    sum_pt: float
    multiplicity: int
    region: str
    physics_weight: float | None
    match_distance: float


def load_torch(path: Path) -> Any:
    """Memory-map large samples when supported by the installed PyTorch."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def require_sample(payload: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a mapping, got {type(payload).__name__}.")
    missing = [key for key in ("pf", "obj") if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing required tensors {missing}.")
    return payload


def load_nurd_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict_model"]
    config = checkpoint["config"]
    projection_key = next(
        (key for key in state if key.endswith(".attn.e.weight")), None)
    num_tokens = (
        int(state[projection_key].shape[1]) - 1
        if projection_key is not None else int(config.get("linear_dim", 100))
    )
    num_classes = int(state["classifier.weight"].shape[0])
    model = HLTContrastiveModel(
        num_classes=num_classes,
        embed_size=int(config["embed_size"]),
        latent_dim=int(config["latent_dim"]),
        proj_dim=int(config["proj_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        dim_ff=int(config["dim_ff"]),
        linear_dim=int(config["linear_dim"]),
        num_tokens=num_tokens,
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def load_autoencoder(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("ae_config")
    if config is None:
        first_weight = checkpoint["ae"][next(iter(checkpoint["ae"]))]
        features = int(first_weight.shape[1])
        config = {
            "features": features,
            "latent_dim": 16,
            "encoder_config": {"nodes": [512, 256]},
            "decoder_config": {"nodes": [256, 512, features]},
            "alpha": 1.0,
        }
    model = HLTAutoencoder(config).to(device)
    model.load_state_dict(checkpoint["ae"])
    model.eval()
    return model


def infer_sample(
    path: Path,
    nurd: HLTContrastiveModel,
    ae: HLTAutoencoder,
    scaler: Mapping[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> Scores:
    payload = require_sample(load_torch(path), path)
    pf = torch.as_tensor(payload["pf"])
    obj = torch.as_tensor(payload["obj"])
    if pf.ndim != 3 or pf.shape[2] < 7:
        raise ValueError(f"Expected PF shape [N, candidates, >=7], got {tuple(pf.shape)}.")
    if obj.ndim != 3 or obj.shape[2] < 4:
        raise ValueError(f"Expected object shape [N, objects, >=4], got {tuple(obj.shape)}.")
    if pf.shape[0] != obj.shape[0]:
        raise ValueError("PF and object tensors have different event counts.")

    if "label" in payload:
        labels = torch.as_tensor(payload["label"]).long().reshape(-1).numpy()
    else:
        labels = np.full(pf.shape[0], -1, dtype=np.int64)
    mean = torch.as_tensor(scaler["mu"]).float().reshape(1, -1)
    std = torch.as_tensor(scaler["std"]).float().reshape(1, -1).clamp_min(1e-8)

    ae_scores: List[torch.Tensor] = []
    latents: List[torch.Tensor] = []
    probabilities: List[torch.Tensor] = []
    sum_pt: List[torch.Tensor] = []
    multiplicity: List[torch.Tensor] = []
    print(f"Scoring {pf.shape[0]:,} events from {path}", flush=True)
    with torch.inference_mode():
        for start in range(0, pf.shape[0], batch_size):
            stop = min(start + batch_size, pf.shape[0])
            pf_batch = torch.nan_to_num(
                pf[start:stop].float(), nan=0.0, posinf=0.0, neginf=0.0)
            active = pf_batch[..., 0] > 0
            sum_pt.append(torch.where(
                active, pf_batch[..., 0], torch.zeros_like(pf_batch[..., 0])
            ).sum(dim=1).cpu())
            multiplicity.append(active.sum(dim=1).cpu())

            obj_batch = torch.nan_to_num(
                obj[start:stop, :, :4].reshape(stop - start, -1).float(),
                nan=0.0, posinf=0.0, neginf=0.0)
            obj_batch = ((obj_batch - mean) / std).to(device)
            reconstruction, _ = ae(obj_batch)
            ae_scores.append(
                (reconstruction - obj_batch).square().mean(dim=1).cpu())

            latent, logits = nurd(pf_batch.to(device))
            latents.append(latent.cpu())
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu())

    return Scores(
        ae=torch.cat(ae_scores).numpy().astype(np.float32),
        latent=torch.cat(latents).numpy().astype(np.float32),
        probabilities=torch.cat(probabilities).numpy().astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        sum_pt=torch.cat(sum_pt).numpy().astype(np.float32),
        multiplicity=torch.cat(multiplicity).numpy().astype(np.int32),
    )


def infer_reference_latents(
    payload: Mapping[str, Any],
    nurd: HLTContrastiveModel,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pf = torch.as_tensor(payload["pf"])
    labels = torch.as_tensor(payload["label"]).long().reshape(-1).numpy()
    parts: List[torch.Tensor] = []
    print(f"Embedding {pf.shape[0]:,} MD-reference events", flush=True)
    with torch.inference_mode():
        for start in range(0, pf.shape[0], batch_size):
            batch = torch.nan_to_num(
                pf[start:start + batch_size].float(),
                nan=0.0, posinf=0.0, neginf=0.0).to(device)
            latent, _ = nurd(batch)
            parts.append(latent.cpu())
    return torch.cat(parts).numpy().astype(np.float32), labels


def fit_weighted_qcd_md(
    reference_latent: np.ndarray,
    reference_labels: np.ndarray,
    reference_weights: np.ndarray,
    fit_indices: np.ndarray,
    qcd_label: int,
    n_pca: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    selected = fit_indices[reference_labels[fit_indices] == qcd_label]
    if selected.size < 10:
        raise ValueError("The saved training split has fewer than ten QCD events.")
    values = reference_latent[selected].astype(np.float64)
    weights = reference_weights[selected].astype(np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Reference QCD weights must be finite, nonnegative, and nonzero.")
    weights /= weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if n_pca is not None:
        count = min(int(n_pca), eigenvalues.size)
        eigenvalues = eigenvalues[-count:]
        eigenvectors = eigenvectors[:, -count:]
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    whitening = eigenvectors / np.sqrt(eigenvalues)
    return mean.astype(np.float64), whitening.astype(np.float64)


def md_scores(latent: np.ndarray, mean: np.ndarray, whitening: np.ndarray):
    transformed = (latent.astype(np.float64) - mean) @ whitening
    return np.square(transformed).sum(axis=1).astype(np.float32)


def validate_reference_contract(
    checkpoint: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_weight_path: Path,
    qcd_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    preprocessing = checkpoint.get("preprocessing", {})
    expected_signature = preprocessing.get("data_signature")
    actual_signature = sample_signature(reference)
    if expected_signature != actual_signature:
        raise ValueError(
            "The supplied reference sample does not match the NURD checkpoint.")
    reference_labels = torch.as_tensor(reference["label"]).long().reshape(-1)
    weights, metadata = load_generator_weights(
        str(reference_weight_path), reference_labels,
        qcd_label=qcd_label, sample=reference)
    expected_checksum = preprocessing.get("weighting", {}).get(
        "generator", {}).get("effective_physics_weight_sha256")
    if expected_checksum != metadata.get("effective_physics_weight_sha256"):
        raise ValueError(
            "The supplied reference weights do not match the NURD checkpoint.")
    fit_indices = torch.as_tensor(
        preprocessing.get("split", {}).get("train_indices", []),
        dtype=torch.long).cpu().numpy().astype(np.int64, copy=False)
    if fit_indices.size == 0:
        raise ValueError("The checkpoint does not contain saved MD-fit indices.")
    return weights.numpy().astype(np.float64), fit_indices


def standardized_match_features(
    sum_pt: np.ndarray,
    multiplicity: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    raw = np.column_stack([
        np.log1p(np.maximum(sum_pt, 0.0)),
        np.log1p(np.maximum(multiplicity, 0)),
    ])
    return (raw - center) / scale


def nearest_unused(
    tree: cKDTree,
    pool_indices: np.ndarray,
    target: np.ndarray,
    used: set[int],
) -> Tuple[int, float]:
    k = min(max(16, len(used) + 4), pool_indices.size)
    distances, positions = tree.query(target, k=k)
    for distance, position in zip(
            np.atleast_1d(distances), np.atleast_1d(positions)):
        event_index = int(pool_indices[int(position)])
        if event_index not in used:
            return event_index, float(distance)
    raise RuntimeError("Could not find an unused matched event.")


def select_matched_triplets(
    test: Scores,
    signal: Scores,
    usual_pool: np.ndarray,
    false_positive_pool: np.ndarray,
    qcd_mask: np.ndarray,
    count: int,
    seed: int,
    matchable_fraction: float,
) -> List[Dict[str, float | int]]:
    base_valid = qcd_mask & np.isfinite(test.sum_pt) & (test.sum_pt > 0)
    base_raw = np.column_stack([
        np.log1p(test.sum_pt[base_valid]),
        np.log1p(test.multiplicity[base_valid]),
    ])
    center = np.median(base_raw, axis=0)
    q25, q75 = np.quantile(base_raw, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(base_raw, axis=0)
    scale = np.where(scale > 1e-6, scale, np.maximum(fallback, 1e-6))

    test_features = standardized_match_features(
        test.sum_pt, test.multiplicity, center, scale)
    signal_features = standardized_match_features(
        signal.sum_pt, signal.multiplicity, center, scale)
    valid_signal = (
        np.isfinite(signal_features).all(axis=1)
        & np.isfinite(signal.ae) & np.isfinite(signal.md)
        & (signal.sum_pt > 0) & (signal.multiplicity > 0)
    )
    signal_indices = np.flatnonzero(valid_signal)
    if signal_indices.size < count:
        raise ValueError("Not enough finite TpTp events to make the displays.")

    usual_tree = cKDTree(test_features[usual_pool])
    false_positive_tree = cKDTree(test_features[false_positive_pool])
    usual_distance, _ = usual_tree.query(signal_features[signal_indices], k=1)
    false_distance, _ = false_positive_tree.query(
        signal_features[signal_indices], k=1)
    quality = np.maximum(usual_distance, false_distance)
    cutoff = float(np.quantile(quality, matchable_fraction))
    eligible = signal_indices[quality <= cutoff]
    if eligible.size < count:
        eligible = signal_indices[np.argsort(quality)[:max(count, 1)]]

    rng = np.random.default_rng(seed)
    eligible = rng.permutation(eligible)
    used_usual: set[int] = set()
    used_false: set[int] = set()
    triplets: List[Dict[str, float | int]] = []
    for signal_index in eligible:
        target = signal_features[int(signal_index)]
        usual_index, usual_match = nearest_unused(
            usual_tree, usual_pool, target, used_usual)
        false_index, false_match = nearest_unused(
            false_positive_tree, false_positive_pool, target, used_false)
        used_usual.add(usual_index)
        used_false.add(false_index)
        triplets.append({
            "usual_qcd": usual_index,
            "false_positive_qcd": false_index,
            "tptp": int(signal_index),
            "usual_match_distance": usual_match,
            "false_positive_match_distance": false_match,
        })
        if len(triplets) == count:
            break
    if len(triplets) != count:
        raise RuntimeError(f"Selected only {len(triplets)} of {count} triplets.")
    return triplets


def ae_residuals(
    path: Path,
    indices: Sequence[int],
    ae: HLTAutoencoder,
    scaler: Mapping[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    payload = require_sample(load_torch(path), path)
    obj = torch.as_tensor(payload["obj"])[torch.as_tensor(indices), :, :4]
    shape = obj.shape
    flat = torch.nan_to_num(
        obj.reshape(shape[0], -1).float(),
        nan=0.0, posinf=0.0, neginf=0.0)
    mean = torch.as_tensor(scaler["mu"]).float().reshape(1, -1)
    std = torch.as_tensor(scaler["std"]).float().reshape(1, -1).clamp_min(1e-8)
    normalized = ((flat - mean) / std).to(device)
    with torch.inference_mode():
        reconstruction, _ = ae(normalized)
    residual = (reconstruction - normalized).square().cpu().numpy()
    return residual.reshape(shape[0], shape[1], 4)


def fetch_pf(path: Path, indices: Sequence[int]) -> np.ndarray:
    payload = require_sample(load_torch(path), path)
    return torch.nan_to_num(
        torch.as_tensor(payload["pf"])[torch.as_tensor(indices), :, :7].float(),
        nan=0.0, posinf=0.0, neginf=0.0).numpy()


def active_pf(pf: np.ndarray) -> Dict[str, np.ndarray]:
    active = np.isfinite(pf).all(axis=1) & (pf[:, 0] > 0)
    values = pf[active]
    if not values.size:
        return {
            key: np.empty(0) for key in
            ("pt", "eta", "phi", "dxy", "dxysig", "flag", "pid")
        }
    phi = (values[:, 2] + math.pi) % (2.0 * math.pi) - math.pi
    return {
        "pt": values[:, 0], "eta": values[:, 1], "phi": phi,
        "dxy": values[:, 3], "dxysig": values[:, 4],
        "flag": values[:, 5], "pid": np.rint(values[:, 6]).astype(np.int64),
    }


def pdg_group(pid: np.ndarray) -> np.ndarray:
    absolute = np.abs(pid)
    return np.asarray([
        int(value) if int(value) in PDG_STYLE else -1 for value in absolute
    ], dtype=np.int64)


def marker_sizes(pt: np.ndarray, common_max: float | None = None) -> np.ndarray:
    if pt.size == 0:
        return np.empty(0)
    maximum = float(np.max(pt) if common_max is None else common_max)
    maximum = max(maximum, 1e-6)
    return 10.0 + 190.0 * np.sqrt(np.clip(pt / maximum, 0.0, 1.0))


def draw_eta_phi(
    ax: plt.Axes,
    candidates: Dict[str, np.ndarray],
    common_max_pt: float | None = None,
    show_legend: bool = True,
) -> None:
    within = np.abs(candidates["eta"]) <= ETA_LIMIT
    groups = pdg_group(candidates["pid"])
    for group in list(PDG_STYLE) + [-1]:
        selected = within & (groups == group)
        if not np.any(selected):
            continue
        label, color = PDG_STYLE.get(group, UNKNOWN_STYLE)
        ax.scatter(
            candidates["eta"][selected], candidates["phi"][selected],
            s=marker_sizes(candidates["pt"][selected], common_max_pt),
            color=color, alpha=0.72, edgecolors="black", linewidths=0.2,
            label=label, rasterized=True,
        )
    ax.set_xlim(-ETA_LIMIT, ETA_LIMIT)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel(r"PF candidate $\eta$")
    ax.set_ylabel(r"PF candidate $\phi$")
    ax.set_yticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
                  [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.grid(alpha=0.18, linestyle=":")
    outside = int(np.count_nonzero(~within))
    if outside:
        ax.text(0.02, 0.02, f"{outside} candidates outside |η|≤5",
                transform=ax.transAxes, fontsize=9)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)


def pt_histogram(candidates: Dict[str, np.ndarray]) -> np.ndarray:
    within = np.abs(candidates["eta"]) <= ETA_LIMIT
    histogram, _, _ = np.histogram2d(
        candidates["eta"][within], candidates["phi"][within],
        bins=[ETA_EDGES, PHI_EDGES], weights=candidates["pt"][within])
    return histogram.T


def draw_pt_flow(ax: plt.Axes, candidates: Dict[str, np.ndarray]):
    histogram = pt_histogram(candidates)
    positive = histogram[histogram > 0]
    if positive.size:
        vmax = float(positive.max())
        vmin = max(float(positive.min()), vmax * 1e-4)
        image = ax.pcolormesh(
            ETA_EDGES, PHI_EDGES, np.ma.masked_where(histogram <= 0, histogram),
            cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax), shading="auto")
    else:
        image = ax.pcolormesh(ETA_EDGES, PHI_EDGES, histogram,
                              cmap="magma", shading="auto")
    ax.set_xlabel(r"PF candidate $\eta$")
    ax.set_ylabel(r"PF candidate $\phi$")
    ax.set_yticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
                  [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_title(r"PF $p_T$ flow (not L1 calorimeter towers)", fontsize=11)
    return image


def score_text(event: DisplayEvent, thresholds: Mapping[str, float]) -> str:
    probability_text = "  ".join(
        f"{CLASS_NAMES.get(index, str(index))}: {100.0 * value:.1f}%"
        for index, value in enumerate(event.probabilities))
    weight_text = (
        "" if event.physics_weight is None
        else f"\nGenerator weight: {event.physics_weight:.4g}"
    )
    return (
        f"PF ΣpT: {event.sum_pt:.1f}    active PF: {event.multiplicity}\n"
        f"AE: {event.ae_score:.4g}  (cut {thresholds['t1']:.4g})    "
        f"MD: {event.md_score:.4g}  (cut {thresholds['t2']:.4g})\n"
        f"Classifier — {probability_text}\n"
        f"ABCD region: {event.region}    match distance: {event.match_distance:.3f}"
        f"{weight_text}"
    )


def plot_event(
    event: DisplayEvent,
    thresholds: Mapping[str, float],
    output_path: Path,
) -> None:
    candidates = active_pf(event.pf)
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.3))
    draw_eta_phi(axes[0, 0], candidates)
    axes[0, 0].set_title("PF candidate topology", fontsize=12)

    image = draw_pt_flow(axes[0, 1], candidates)
    fig.colorbar(image, ax=axes[0, 1], pad=0.02, label=r"Σ PF $p_T$ per bin")

    groups = pdg_group(candidates["pid"])
    colors = [PDG_STYLE.get(int(group), UNKNOWN_STYLE)[1] for group in groups]
    axes[1, 0].scatter(
        candidates["eta"], candidates["dxysig"],
        s=marker_sizes(candidates["pt"]), c=colors,
        alpha=0.72, edgecolors="black", linewidths=0.2, rasterized=True)
    axes[1, 0].set_xlabel(r"PF candidate $\eta$")
    axes[1, 0].set_ylabel(r"$d_{xy}/\sigma$")
    axes[1, 0].set_yscale("symlog", linthresh=1.0)
    axes[1, 0].set_title("Tracking/displacement information", fontsize=12)
    axes[1, 0].grid(alpha=0.18, linestyle=":")

    log_residual = np.log10(np.maximum(event.ae_residual.T, 1e-8))
    residual_image = axes[1, 1].imshow(
        log_residual, origin="lower", aspect="auto", interpolation="nearest",
        cmap="viridis")
    axes[1, 1].set_xlabel("AE object slot")
    axes[1, 1].set_ylabel("AE object feature")
    axes[1, 1].set_yticks(range(4), ["feature 0", "feature 1", "feature 2", "feature 3"])
    axes[1, 1].set_title("AE standardized squared residual", fontsize=12)
    fig.colorbar(residual_image, ax=axes[1, 1], pad=0.02,
                 label=r"$\log_{10}[(x-\hat{x})^2]$")

    fig.suptitle(
        f"CMS Simulation Preliminary   {CATEGORY_TITLES[event.category]}"
        f"   source row {event.source_index}",
        fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.015, score_text(event, thresholds),
             ha="center", va="bottom", fontsize=10.5,
             bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
                   "edgecolor": "0.7", "alpha": 0.95})
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_triplet(
    events: Sequence[DisplayEvent],
    output_path: Path,
) -> None:
    candidates = [active_pf(event.pf) for event in events]
    common_max = max(
        (float(values["pt"].max()) for values in candidates if values["pt"].size),
        default=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.8), sharex=True, sharey=True)
    for ax, event, values in zip(axes, events, candidates):
        draw_eta_phi(ax, values, common_max_pt=common_max, show_legend=False)
        ax.set_title(
            f"{CATEGORY_TITLES[event.category]}\n"
            f"ΣpT={event.sum_pt:.1f}, NPF={event.multiplicity}\n"
            f"AE={event.ae_score:.3g}, MD={event.md_score:.3g}",
            fontsize=11)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7,
               markerfacecolor=color, markeredgecolor="black", label=label)
        for label, color in list(PDG_STYLE.values()) + [UNKNOWN_STYLE]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        "CMS Simulation Preliminary — PF topology matched in total pT and multiplicity",
        fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def manifest_event(event: DisplayEvent) -> Dict[str, Any]:
    return {
        "category": event.category,
        "source_index": event.source_index,
        "sum_pf_pt": event.sum_pt,
        "active_pf_multiplicity": event.multiplicity,
        "ae_score": event.ae_score,
        "md_score": event.md_score,
        "abcd_region": event.region,
        "physics_weight": event.physics_weight,
        "match_distance": event.match_distance,
        "class_probabilities": {
            CLASS_NAMES.get(index, str(index)): float(value)
            for index, value in enumerate(event.probabilities)
        },
    }


def parse_args(argv: Sequence[str] | None = None):
    default_output = (
        Path(__file__).resolve().parent / "generated"
        / datetime.now(timezone.utc).strftime("event_displays_%Y%m%d_%H%M%S_utc")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="NURD checkpoint_main.pth.tar")
    parser.add_argument("--ae-ckpt", type=Path, required=True,
                        help="AE checkpoint_ae.pth")
    parser.add_argument("--test-pt", type=Path, required=True,
                        help="Held-out Mequinna background sample")
    parser.add_argument("--test-weight-path", type=Path, required=True,
                        help="Generator weights aligned with --test-pt")
    parser.add_argument("--reference-pt", type=Path, required=True,
                        help="Mequinna training sample used for the MD fit")
    parser.add_argument("--reference-weight-path", type=Path, required=True,
                        help="Generator weights aligned with --reference-pt")
    parser.add_argument("--signal-pt", type=Path, required=True,
                        help="Independent TpTp sample")
    parser.add_argument("--thresholds-json", type=Path, required=True,
                        help="Held-out eval abcd_thresholds.json")
    parser.add_argument("--outdir", type=Path, default=default_output)
    parser.add_argument("--n-displays", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--qcd-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--matchable-fraction", type=float, default=0.20,
        help="Randomly sample TpTp anchors from this best-matched fraction.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_displays < 1:
        raise ValueError("--n-displays must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not 0 < args.matchable_fraction <= 1:
        raise ValueError("--matchable-fraction must lie in (0, 1].")
    input_paths = [
        args.ckpt, args.ae_ckpt, args.test_pt, args.test_weight_path,
        args.reference_pt, args.reference_weight_path, args.signal_pt,
        args.thresholds_json,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    print(f"Using device: {device}", flush=True)

    thresholds = json.loads(args.thresholds_json.read_text())
    if "t1" not in thresholds or "t2" not in thresholds:
        raise KeyError("Threshold JSON must contain t1 and t2.")
    n_pca = thresholds.get("n_pca")
    n_pca = None if n_pca is None else int(n_pca)

    nurd, checkpoint = load_nurd_model(args.ckpt, device)
    preprocessing = checkpoint.get("preprocessing", {})
    checkpoint_qcd_label = int(preprocessing.get("qcd_label", args.qcd_label))
    if checkpoint_qcd_label != args.qcd_label:
        raise ValueError(
            f"Checkpoint QCD label is {checkpoint_qcd_label}, not {args.qcd_label}.")
    scaler = checkpoint.get("ae_scaler", preprocessing.get("ae_scaler"))
    if scaler is None:
        raise KeyError("NURD checkpoint does not contain the fitted AE scaler.")
    ae = load_autoencoder(args.ae_ckpt, device)

    reference = require_sample(load_torch(args.reference_pt), args.reference_pt)
    reference_weights, fit_indices = validate_reference_contract(
        checkpoint, reference, args.reference_weight_path, args.qcd_label)
    reference_latent, reference_labels = infer_reference_latents(
        reference, nurd, device, args.batch_size)
    md_mean, md_whitening = fit_weighted_qcd_md(
        reference_latent, reference_labels, reference_weights,
        fit_indices, args.qcd_label, n_pca)
    del reference_latent, reference_labels, reference

    test = infer_sample(
        args.test_pt, nurd, ae, scaler, device, args.batch_size)
    signal = infer_sample(
        args.signal_pt, nurd, ae, scaler, device, args.batch_size)
    test.md = md_scores(test.latent, md_mean, md_whitening)
    signal.md = md_scores(signal.latent, md_mean, md_whitening)

    test_payload = require_sample(load_torch(args.test_pt), args.test_pt)
    test_weights_tensor, test_weight_metadata = load_generator_weights(
        str(args.test_weight_path), torch.as_tensor(test_payload["label"]),
        qcd_label=args.qcd_label, sample=test_payload)
    test_weights = test_weights_tensor.numpy().astype(np.float64)
    del test_payload

    finite_test = (
        np.isfinite(test.ae) & np.isfinite(test.md)
        & np.isfinite(test.sum_pt) & (test.sum_pt > 0)
        & (test.multiplicity > 0)
    )
    qcd_mask = finite_test & (test.labels == args.qcd_label)
    usual_pool = np.flatnonzero(
        qcd_mask & (test.ae <= float(thresholds["t1"]))
        & (test.md <= float(thresholds["t2"])))
    false_positive_pool = np.flatnonzero(
        qcd_mask & (test.ae > float(thresholds["t1"]))
        & (test.md > float(thresholds["t2"])))
    print(
        f"Selection pools: usual QCD (D)={usual_pool.size:,}, "
        f"false-positive QCD (A)={false_positive_pool.size:,}, "
        f"TpTp={signal.ae.size:,}", flush=True)
    if min(usual_pool.size, false_positive_pool.size) < args.n_displays:
        raise ValueError("An ABCD QCD pool is too small for the requested displays.")

    triplets = select_matched_triplets(
        test, signal, usual_pool, false_positive_pool, qcd_mask,
        args.n_displays, args.seed, args.matchable_fraction)
    test_indices = [int(item["usual_qcd"]) for item in triplets]
    test_indices += [int(item["false_positive_qcd"]) for item in triplets]
    signal_indices = [int(item["tptp"]) for item in triplets]
    test_pf = fetch_pf(args.test_pt, test_indices)
    signal_pf = fetch_pf(args.signal_pt, signal_indices)
    test_residual = ae_residuals(
        args.test_pt, test_indices, ae, scaler, device)
    signal_residual = ae_residuals(
        args.signal_pt, signal_indices, ae, scaler, device)

    args.outdir.mkdir(parents=True, exist_ok=False)
    category_dirs = {
        category: args.outdir / category
        for category in ("usual_qcd", "false_positive_qcd", "tptp", "matched_triplets")
    }
    for directory in category_dirs.values():
        directory.mkdir(parents=True)

    all_manifest: List[Dict[str, Any]] = []
    for display_number, triplet in enumerate(triplets, start=1):
        usual_position = display_number - 1
        false_position = args.n_displays + display_number - 1
        signal_position = display_number - 1
        usual_index = int(triplet["usual_qcd"])
        false_index = int(triplet["false_positive_qcd"])
        signal_index = int(triplet["tptp"])
        events = [
            DisplayEvent(
                "usual_qcd", usual_index, test_pf[usual_position],
                test_residual[usual_position], float(test.ae[usual_index]),
                float(test.md[usual_index]), test.probabilities[usual_index],
                float(test.sum_pt[usual_index]), int(test.multiplicity[usual_index]),
                "D", float(test_weights[usual_index]),
                float(triplet["usual_match_distance"])),
            DisplayEvent(
                "false_positive_qcd", false_index, test_pf[false_position],
                test_residual[false_position], float(test.ae[false_index]),
                float(test.md[false_index]), test.probabilities[false_index],
                float(test.sum_pt[false_index]), int(test.multiplicity[false_index]),
                "A", float(test_weights[false_index]),
                float(triplet["false_positive_match_distance"])),
            DisplayEvent(
                "tptp", signal_index, signal_pf[signal_position],
                signal_residual[signal_position], float(signal.ae[signal_index]),
                float(signal.md[signal_index]), signal.probabilities[signal_index],
                float(signal.sum_pt[signal_index]), int(signal.multiplicity[signal_index]),
                "signal", None, 0.0),
        ]
        for event in events:
            output = category_dirs[event.category] / f"display_{display_number:02d}.png"
            plot_event(event, thresholds, output)
            all_manifest.append(manifest_event(event))
        plot_triplet(
            events,
            category_dirs["matched_triplets"] / f"triplet_{display_number:02d}.png")

    manifest = {
        "description": (
            "PF-level displays; these are not Level-1 calorimeter-tower displays."),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "matching": {
            "features": ["log1p(sum PF pT)", "log1p(active PF multiplicity)"],
            "standardization": "QCD median and IQR/1.349",
            "signal_anchor_pool": (
                f"random among best-matched {100 * args.matchable_fraction:.1f}% of TpTp"),
        },
        "categories": {
            "usual_qcd": "QCD in ABCD region D (below both held-out thresholds)",
            "false_positive_qcd": "QCD in ABCD region A (above both held-out thresholds)",
            "tptp": "independent TpTp signal sample",
        },
        "thresholds": thresholds,
        "inputs": {
            "nurd_checkpoint": str(args.ckpt.resolve()),
            "ae_checkpoint": str(args.ae_ckpt.resolve()),
            "test_sample": str(args.test_pt.resolve()),
            "test_weights": str(args.test_weight_path.resolve()),
            "reference_sample": str(args.reference_pt.resolve()),
            "reference_weights": str(args.reference_weight_path.resolve()),
            "signal_sample": str(args.signal_pt.resolve()),
            "thresholds_json": str(args.thresholds_json.resolve()),
            "test_weight_alignment": test_weight_metadata,
        },
        "events": all_manifest,
    }
    (args.outdir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"Event displays written to {args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
