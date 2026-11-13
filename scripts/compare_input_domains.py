#!/usr/bin/env python3
"""Compare legacy Mequinna and train3M held-out HLT input domains.

All figures compare *within-parent* shapes, not cross-section-normalized rates.
No threshold or model checkpoint is selected using either test sample.
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
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.hlt_autoencoder import HLTAutoencoder
from utils.hlt_data import load_hlt_sample
from utils.hlt_weights import load_generator_weights


PARENTS = {0: "Z+jets", 1: "QCD", 2: r"$t\bar t$", 3: "W+jets"}
VARIABLES = {
    "pf_count": ("Active PF rows / event", "integer"),
    "nonpf_count": ("Active rows with is_pf=0 / event", "integer"),
    "pf_sum_pt": (r"PF $\sum p_T$ [GeV]", "continuous"),
    "jet_count": ("Active jet slots / event (max 10)", "integer"),
    "ht10": (r"$H_T^{10}$: jets $p_T>30$, $|\eta|<2.4$ [GeV]", "continuous"),
    "jet_lead_pt": (r"Leading stored jet $p_T$ [GeV]", "continuous"),
    "met_pt": (r"Stored MET $p_T$ [GeV]", "continuous"),
    "ae_loss": ("Frozen physics-AE reconstruction MSE", "continuous"),
}
PDG_NAMES = ["charged hadron", "neutral hadron", "photon", "electron", "muon", "other"]
OBJECT_NAMES = ["muon", "electron", "photon"]
COLORS = {"Mequinna": "#b33a3a", "train3M": "#245b8a"}


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def load_ae(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = HLTAutoencoder(payload["ae_config"]).to(device)
    model.load_state_dict(payload["ae"])
    model.eval()
    scaler = payload["ae_scaler"]
    mu = torch.as_tensor(scaler["mu"], dtype=torch.float32, device=device)
    std = torch.as_tensor(scaler["std"], dtype=torch.float32, device=device)
    if mu.numel() != 92 or std.numel() != 92:
        raise ValueError("Expected 23 x 4 object-feature AE scaler.")
    return model, mu, std


def ae_scores(model, mu, std, obj: np.ndarray, device: torch.device) -> np.ndarray:
    flat = np.nan_to_num(obj[:, :, :4].reshape(len(obj), -1),
                         nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.as_tensor(flat, dtype=torch.float32, device=device)
    x = (x - mu) / (std + 1e-8)
    with torch.no_grad():
        reconstruction, _ = model(x)
        return ((reconstruction - x) ** 2).mean(dim=1).cpu().numpy()


def pdg_categories(ids: np.ndarray) -> np.ndarray:
    ids = np.abs(np.rint(ids)).astype(np.int32)
    category = np.full(ids.shape, 5, dtype=np.int8)
    category[np.isin(ids, (211, 321, 2212))] = 0
    category[np.isin(ids, (130, 310, 2112))] = 1
    category[ids == 22] = 2
    category[ids == 11] = 3
    category[ids == 13] = 4
    return category


def extract_chunk(pf: np.ndarray, obj: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    if pf.ndim != 3 or pf.shape[1:] != (400, 7):
        raise ValueError(f"PF tensor has shape {pf.shape}; expected (N,400,7).")
    if obj.ndim != 3 or obj.shape[1:] != (23, 4):
        raise ValueError(f"Object tensor has shape {obj.shape}; expected (N,23,4).")
    active = np.isfinite(pf[:, :, 0]) & (pf[:, :, 0] > 0)
    jet_pt = np.nan_to_num(obj[:, :10, 0], nan=0.0, posinf=0.0, neginf=0.0)
    jet_eta = np.nan_to_num(obj[:, :10, 1], nan=0.0, posinf=0.0, neginf=0.0)
    jet_selected = (jet_pt > 30) & (np.abs(jet_eta) < 2.4)
    values = {
        "pf_count": active.sum(axis=1),
        "nonpf_count": (active & (pf[:, :, 5] < 0.5)).sum(axis=1),
        "pf_sum_pt": np.where(active, pf[:, :, 0], 0).sum(axis=1),
        "jet_count": (jet_pt > 0).sum(axis=1),
        "ht10": np.where(jet_selected, jet_pt, 0).sum(axis=1),
        "jet_lead_pt": jet_pt.max(axis=1),
        "met_pt": np.nan_to_num(obj[:, 22, 0], nan=0.0, posinf=0.0, neginf=0.0),
    }
    object_present = np.stack([
        (obj[:, 10:14, 0] > 0).any(axis=1),
        (obj[:, 14:18, 0] > 0).any(axis=1),
        (obj[:, 18:22, 0] > 0).any(axis=1),
    ], axis=1)
    return values, active, object_present


def extract_sample(path: Path, weight_path: Path, model, mu, std,
                   device: torch.device, batch_size: int) -> dict:
    sample = load_hlt_sample(path)
    labels = torch.as_tensor(sample["label"]).long().reshape(-1)
    if set(labels.unique().tolist()) != set(PARENTS):
        raise ValueError(f"Expected all four parent labels; got {labels.unique().tolist()}.")
    weights, weight_meta = load_generator_weights(
        str(weight_path), labels, qcd_label=1, sample=sample)
    n = len(labels)
    output = {
        "path": str(path), "weight_path": str(weight_path),
        "weight_metadata": weight_meta, "events": n,
        "labels": labels.numpy().astype(np.int8, copy=True),
        "weights": weights.numpy().copy(),
        "values": {key: np.empty(n, dtype=np.float32) for key in VARIABLES},
        "pdg_raw": np.zeros((4, len(PDG_NAMES)), dtype=np.float64),
        "pdg_weighted": np.zeros((4, len(PDG_NAMES)), dtype=np.float64),
        "objects_raw": np.zeros((4, len(OBJECT_NAMES)), dtype=np.float64),
        "objects_weighted": np.zeros((4, len(OBJECT_NAMES)), dtype=np.float64),
        "contract": {"nonfinite_pf_active": 0, "nonfinite_obj": 0,
                     "nonbinary_is_pf_active": 0, "bad_object_type_slots": 0},
    }
    if "background_category" in sample:
        categories = _numpy(sample["background_category"]).reshape(-1)
        if len(categories) != n:
            raise ValueError("Background-category length differs from labels.")
        qcd = output["labels"] == 1
        output["qcd_category"] = {
            str(int(category)): {
                "events": int((qcd & (categories == category)).sum()),
                "within_qcd_raw_fraction": float(
                    (qcd & (categories == category)).sum() / qcd.sum()),
                "within_qcd_weighted_fraction": float(
                    output["weights"][qcd & (categories == category)].sum()
                    / output["weights"][qcd].sum()),
            }
            for category in np.unique(categories[qcd])
        }
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        pf = _numpy(sample["pf"][start:stop])
        obj = _numpy(sample["obj"][start:stop])
        parent = output["labels"][start:stop]
        w = output["weights"][start:stop]
        values, active, objects = extract_chunk(pf, obj)
        values["ae_loss"] = ae_scores(model, mu, std, obj, device)
        for key, value in values.items():
            output["values"][key][start:stop] = value
        valid_is_pf = np.isclose(pf[:, :, 5], 0) | np.isclose(pf[:, :, 5], 1)
        output["contract"]["nonbinary_is_pf_active"] += int((active & ~valid_is_pf).sum())
        output["contract"]["nonfinite_pf_active"] += int((active[:, :, None] & ~np.isfinite(pf)).sum())
        output["contract"]["nonfinite_obj"] += int((~np.isfinite(obj)).sum())
        expected_types = np.r_[np.repeat(0, 10), np.repeat(1, 4),
                               np.repeat(2, 4), np.repeat(3, 4), 4]
        output["contract"]["bad_object_type_slots"] += int(
            ((obj[:, :, 0] > 0) & (obj[:, :, 3] != expected_types[None, :])).sum())
        category = pdg_categories(pf[:, :, 6])
        active_parent = np.broadcast_to(parent[:, None], active.shape)[active]
        code = active_parent * len(PDG_NAMES) + category[active]
        output["pdg_raw"] += np.bincount(code, minlength=24).reshape(4, 6)
        candidate_weights = np.broadcast_to(w[:, None], active.shape)[active]
        output["pdg_weighted"] += np.bincount(
            code, weights=candidate_weights, minlength=24).reshape(4, 6)
        for c in PARENTS:
            mask = parent == c
            output["objects_raw"][c] += objects[mask].sum(axis=0)
            output["objects_weighted"][c] += (objects[mask] * w[mask, None]).sum(axis=0)
        if stop == n or stop % 100000 < batch_size:
            print(f"  {path.name}: {stop}/{n} events", flush=True)
    if any(not np.isfinite(v).all() for v in output["values"].values()):
        raise ValueError(f"Non-finite derived values in {path}.")
    return output


def _limits(old: dict, new: dict, key: str) -> tuple[float, float]:
    if VARIABLES[key][1] == "integer":
        return 0.0, max(float(old["values"][key].max()),
                        float(new["values"][key].max()))
    values = np.concatenate((old["values"][key], new["values"][key]))
    top = float(np.quantile(values, 0.995))
    return 0.0, max(top, 1e-6)


def plot_variable(old: dict, new: dict, key: str, output: Path) -> None:
    low, high = _limits(old, new, key)
    if VARIABLES[key][1] == "integer":
        high = min(high, 400)
        edges = np.arange(low - 0.5, high + 1.5)
    else:
        edges = np.linspace(low, high, 70)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for c, ax in enumerate(axes.flat):
        for name, data in (("Mequinna", old), ("train3M", new)):
            mask = data["labels"] == c
            vals = data["values"][key][mask]
            physics_w = data["weights"][mask]
            for weight, style, suffix in (
                (None, "-", "raw"), (physics_w, "--", "weighted")):
                counts, _ = np.histogram(vals, bins=edges, weights=weight)
                total = len(vals) if weight is None else physics_w.sum()
                if total > 0:
                    ax.stairs(counts / total, edges, color=COLORS[name],
                              linestyle=style, linewidth=1.6,
                              label=f"{name} {suffix}")
        ax.set_title(PARENTS[c])
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)
        ax.grid(alpha=0.2)
        ax.set_ylabel("fraction of parent events / bin")
        ax.set_xlabel(VARIABLES[key][0])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{VARIABLES[key][0]} — common range to pooled raw 99.5th percentile\n"
                 "Each parent/sample normalized separately; tails beyond range omitted")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_composition(old: dict, new: dict, key: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    names = PDG_NAMES if key == "pdg" else OBJECT_NAMES
    for c, ax in enumerate(axes.flat):
        x = np.arange(len(names))
        for j, (name, data) in enumerate((("Mequinna", old), ("train3M", new))):
            if key == "pdg":
                counts = data["pdg_raw"][c]
                title = "fraction of active PF rows"
            else:
                counts = data["objects_raw"][c]
                title = "fraction of parent events with object"
            denom = counts.sum() if key == "pdg" else np.count_nonzero(data["labels"] == c)
            ax.bar(x + (j - 0.5) * 0.38, counts / max(denom, 1),
                   width=0.38, color=COLORS[name], label=name)
        ax.set_title(PARENTS[c])
        ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Active PF PDG-ID composition" if key == "pdg"
                 else "Object occupancy in fixed 23-slot tensor")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _weighted_quantile(x, w, q):
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cumulative = np.cumsum(ws)
    return float(xs[min(np.searchsorted(cumulative, q * cumulative[-1]), len(xs) - 1)])


def write_metrics(old: dict, new: dict, output: Path) -> None:
    rows = []
    for name, data in (("Mequinna", old), ("train3M", new)):
        for c in PARENTS:
            mask = data["labels"] == c
            w = data["weights"][mask]
            for key in VARIABLES:
                x = data["values"][key][mask]
                rows.append({
                    "sample": name, "parent_label": c,
                    "parent_name": PARENTS[c].replace("$", ""),
                    "variable": key, "events": len(x),
                    "raw_median": float(np.median(x)),
                    "raw_p90": float(np.quantile(x, .90)),
                    "raw_p99": float(np.quantile(x, .99)),
                    "weighted_median": _weighted_quantile(x, w, .5),
                    "weighted_p90": _weighted_quantile(x, w, .9),
                    "zero_fraction": float(np.mean(x == 0)),
                    "weighted_zero_fraction": float(w[x == 0].sum() / w.sum()),
                    "slot_saturation_fraction": float(np.mean(x == 400))
                    if key == "pf_count" else (float(np.mean(x == 10))
                    if key == "jet_count" else ""),
                })
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "metadata.json").write_text(json.dumps({
        "protocol": "heldout_input_domain_shapes_no_threshold_selection",
        "caveat": "Mequinna bare weights are QCD-only; train3M weights cover all events. "
                  "Weighted curves are separately normalized within each parent/sample; "
                  "absolute rates must not be compared.",
        "HT_definition": "sum pT of first 10 stored jets with pT>30 GeV and |eta|<2.4; "
                         "lower bound if jet collection was truncated",
        "samples": {name: {
            "path": data["path"], "weights": data["weight_path"],
            "events": data["events"],
            "parent_counts": {str(c): int((data["labels"] == c).sum()) for c in PARENTS},
            "weight_scope": data["weight_metadata"]["weight_scope"],
            "alignment_verified": data["weight_metadata"]["alignment_verified"],
            "contract": data["contract"],
            "qcd_generator_pt_categories": data.get("qcd_category"),
        } for name, data in (("Mequinna", old), ("train3M", new))},
        "figures": [f"{key}.png" for key in VARIABLES] +
                   ["pf_pdg_composition.png", "object_occupancy.png"],
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--legacy-weights", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--new-weights", required=True, type=Path)
    parser.add_argument("--ae-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite {args.output_dir}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    model, mu, std = load_ae(args.ae_checkpoint, device)
    print(f"Device: {device}", flush=True)
    old = extract_sample(args.legacy, args.legacy_weights, model, mu, std,
                         device, args.batch_size)
    new = extract_sample(args.new, args.new_weights, model, mu, std,
                         device, args.batch_size)
    args.output_dir.mkdir(parents=True)
    for key in VARIABLES:
        plot_variable(old, new, key, args.output_dir / f"{key}.png")
    plot_composition(old, new, "pdg", args.output_dir / "pf_pdg_composition.png")
    plot_composition(old, new, "objects", args.output_dir / "object_occupancy.png")
    write_metrics(old, new, args.output_dir)
    print(f"Wrote 10 figures, summary.csv, metadata.json to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
