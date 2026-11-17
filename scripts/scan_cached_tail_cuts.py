"""Fast, exploratory cut placement using cached physics-all AE-tail scores only."""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def quantiles(x, w, probabilities):
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    return x[order[np.minimum(np.searchsorted(cumulative, probabilities * cumulative[-1]), len(x)-1)]]


def scan(x, y, w, tx, ty, minimums):
    # Bin numbers implement the evaluator's strict > threshold convention,
    # including tied values. One histogram replaces thousands of event loops.
    ix = np.searchsorted(tx, x, side="left")
    iy = np.searchsorted(ty, y, side="left")
    shape = (len(tx)+1, len(ty)+1)
    index = ix * shape[1] + iy
    arrays = []
    for values in (w, w*w, np.ones(len(w))):
        hist = np.bincount(index, weights=values, minlength=np.prod(shape)).reshape(shape)
        cumulative = hist.cumsum(0).cumsum(1)
        d = cumulative[:-1, :-1]
        b = cumulative[-1, :-1][None, :] - d
        c = cumulative[:-1, -1][:, None] - d
        a = cumulative[-1, -1] - b - c - d
        arrays.append(np.maximum(np.stack((a, b, c, d)), 0))
    yields, sumw2, raw = arrays
    ess = np.divide(yields*yields, sumw2, out=np.zeros_like(yields), where=sumw2 > 0)
    a, b, c, d = yields
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = b*c/(d*a)
        sigma = ratio*np.sqrt(np.sum(sumw2/yields**2, axis=0))
    valid = np.all(ess >= np.asarray(minimums)[:, None, None], axis=0) & np.isfinite(ratio)
    return dict(yields=yields, sumw2=sumw2, raw=raw, ess=ess, ratio=ratio,
                sigma=sigma, valid=valid, efficiency=a/w.sum())


def stable_choice(result, low, high):
    valid = result["valid"] & (result["efficiency"] >= low) & (result["efficiency"] <= high)
    objective = np.full(valid.shape, np.inf)
    # Require a populated 3x3 neighborhood, not an isolated lucky crossing.
    for i in range(1, valid.shape[0]-1):
        for j in range(1, valid.shape[1]-1):
            block = np.s_[i-1:i+2, j-1:j+2]
            if valid[block].all():
                objective[i, j] = np.median(np.abs(result["ratio"][block]-1)) + result["sigma"][i, j]
    if not np.isfinite(objective).any():
        return None
    return tuple(int(v) for v in np.unravel_index(np.argmin(objective), objective.shape))


def point(result, index, tx, ty):
    i, j = index
    values = result["yields"][:, i, j]
    return {"ae_cut": float(tx[i]), "md_cut": float(ty[j]),
            "predicted_over_observed": float(result["ratio"][i, j]),
            "ratio_uncertainty": float(result["sigma"][i, j]),
            "A_efficiency_within_tail": float(result["efficiency"][i, j]),
            "statistically_valid": bool(result["valid"][i, j]),
            "regions": {r: {"yield": float(values[k]),
                             "effective_count": float(result["ess"][k, i, j]),
                             "raw_count": int(round(result["raw"][k, i, j]))}
                        for k, r in enumerate("ABCD")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tail-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--minimums", type=int, nargs=4, default=[50, 50, 50, 500], metavar=("A", "B", "C", "D"))
    p.add_argument("--efficiency-range", type=float, nargs=2, default=[.02, .25])
    args = p.parse_args()
    if str(args.tail_dir) == "latest":
        root = Path(os.environ.get("BASE", "/scratch/gpfs/IOJALVO/mb7126/nurd_hlt"))/"outputs"
        reports = sorted(root.glob("*_ae_tail_validation/tail_summary.json"),
                         key=lambda f: f.stat().st_mtime, reverse=True)
        for path in reports:
            candidate = json.loads(path.read_text())
            if candidate.get("selection", {}).get("measure") == "physics-all":
                args.tail_dir = path.parent
                break
        else:
            raise FileNotFoundError("No completed physics-all cached tail report found")
    if any(v <= 0 for v in args.minimums) or not 0 < args.efficiency_range[0] < args.efficiency_range[1] < 1:
        p.error("Require positive statistics minimums and 0 < efficiency low < high < 1")
    source = json.loads((args.tail_dir/"tail_summary.json").read_text())
    if source["selection"]["measure"] != "physics-all" or source.get("md_calibration_applied"):
        raise ValueError("Use an existing physics-all, uncalibrated tail report")
    if not np.isclose(source["selection"]["requested_tail_fraction"], .1):
        raise ValueError("Expected the existing weighted top-10% selection")
    with np.load(args.tail_dir/"validation_qcd_scores.npz", allow_pickle=False) as data:
        selected = data["selected"].astype(bool)
        x, y, w = (np.asarray(data[k][selected], float) for k in ("ae_score", "raw_md", "physics_weight"))
        ids = data["event_id"][selected]
    if len(x) < 4 or not np.isfinite(np.stack((x, y, w))).all() or np.any(w < 0) or np.any(x <= 0) or np.any(y < 0) or w.sum() <= 0:
        raise ValueError("Insufficient or invalid cached tail scores")
    if np.unique(ids, axis=0).shape[0] != len(ids):
        raise ValueError("Duplicate cached event IDs")
    # Canonical ordering makes the seeded split independent of cache row order.
    order = np.lexsort(tuple(ids[:, c] for c in (3, 2, 1, 0)))
    permutation = np.random.default_rng(args.seed).permutation(order)
    tune, check = permutation[::2], permutation[1::2]
    if min(w[tune].sum(), w[check].sum()) <= 0:
        raise ValueError("One validation partition has zero physics mass")
    probabilities = np.linspace(.05, .95, 91)
    tx, ty = quantiles(x[tune], w[tune], probabilities), quantiles(y[tune], w[tune], probabilities)
    results = {name: scan(x[rows], y[rows], w[rows], tx, ty, args.minimums)
               for name, rows in (("tune", tune), ("check", check))}
    chosen = stable_choice(results["tune"], *args.efficiency_range)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"source_tail_dir": str(args.tail_dir.resolve()), "selection": source["selection"],
              "checkpoint_sha256": source["checkpoint_sha256"], "seed": args.seed,
              "minimum_region_ess": args.minimums, "A_efficiency_range": args.efficiency_range,
              "objective": "minimum tune 3x3 median abs(predicted/observed-1) plus central ratio uncertainty",
              "heldout_accessed": False, "network_inference": False, "source_tail_events": len(x),
              "warning": "Exploratory re-split of previously inspected validation, not a new independent test. "
                         "AE preselection is inherited from that validation sample. Check split never ranks cuts. "
                         "No signal sample: this is a closure/stability candidate, not a sensitivity optimum.",
              "candidate": None if chosen is None else {
                  name: point(result, chosen, tx, ty) for name, result in results.items()},
              "valid_cells": {name: int(result["valid"].sum()) for name, result in results.items()}}
    with (args.output_dir/"cut_scan.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ae_cut", "md_cut", "tune_ratio", "tune_valid", "check_ratio", "check_valid", "tune_A_efficiency"])
        for i, a in enumerate(tx):
            for j, b in enumerate(ty):
                writer.writerow([a, b, results["tune"]["ratio"][i,j], results["tune"]["valid"][i,j],
                                 results["check"]["ratio"][i,j], results["check"]["valid"][i,j],
                                 results["tune"]["efficiency"][i,j]])
    for name, result in results.items():
        fig, ax = plt.subplots(figsize=(8,6))
        z = np.where(result["valid"], np.abs(result["ratio"]-1)*100, np.nan)
        image = ax.pcolormesh(probabilities, probabilities, z.T, shading="auto", vmin=0, vmax=100)
        fig.colorbar(image, ax=ax, label="|Predicted/observed - 1| (%)")
        if chosen is not None:
            ax.plot(probabilities[chosen[0]], probabilities[chosen[1]], "r*", ms=15)
        ax.set(xlabel="AE percentile within tuning tail", ylabel="MD percentile within tuning tail", title=f"{name}: identical numerical cuts")
        fig.tight_layout(); fig.savefig(args.output_dir/f"{name}_closure_scan.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist2d(np.log10(x), np.log10(np.maximum(y, 1e-12)), weights=w, bins=100, cmap="viridis")
    if chosen is not None:
        ax.axvline(np.log10(tx[chosen[0]]), color="red", ls="--")
        ax.axhline(np.log10(max(ty[chosen[1]], 1e-12)), color="red", ls="--")
    ax.set(xlabel="log10 AE MSE", ylabel="log10 QCD MD²", title="Fixed selected tail: physics-weighted yield")
    fig.tight_layout(); fig.savefig(args.output_dir/"absolute_cut_positions.png", dpi=160); plt.close(fig)
    (args.output_dir/"cut_search_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if chosen is None:
        print("NO FEASIBLE STABLE CANDIDATE: do not interpret this as an optimum or silently lower ESS requirements.")


if __name__ == "__main__":
    main()
