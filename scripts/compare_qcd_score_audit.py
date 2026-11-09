"""Paired, report-only QCD ABCD and validation-to-test score audit.

Inputs are per-event NPZs produced by audit_qcd_abcd_composition.py and
audit_qcd_validation_hardness.py for two checkpoints. No checkpoint selection,
MD fitting, or held-out threshold optimization occurs here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def weighted_quantile(values, weights, probability):
    if len(values) == 0 or np.sum(weights) <= 0:
        raise ValueError("Weighted quantile requires positive non-empty mass")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    index = min(np.searchsorted(cumulative, probability * cumulative[-1]), len(order) - 1)
    return float(values[order[index]])


def effective_count(weights):
    total = float(np.sum(weights))
    sumw2 = float(np.dot(weights, weights))
    return total * total / sumw2 if sumw2 else 0.0


def weighted_ks(x, wx, y, wy):
    """Two-sample distance between weighted empirical CDFs (descriptive only)."""
    order_x = np.argsort(x)
    order_y = np.argsort(y)
    sx, sy = x[order_x], y[order_y]
    cx = np.cumsum(wx[order_x], dtype=np.float64) / np.sum(wx)
    cy = np.cumsum(wy[order_y], dtype=np.float64) / np.sum(wy)
    knots = np.unique(np.concatenate((sx, sy)))
    ix = np.searchsorted(sx, knots, side="right") - 1
    iy = np.searchsorted(sy, knots, side="right") - 1
    fx = np.where(ix >= 0, cx[np.maximum(ix, 0)], 0.0)
    fy = np.where(iy >= 0, cy[np.maximum(iy, 0)], 0.0)
    return float(np.max(np.abs(fx - fy)))


def load_scores(path, *, heldout):
    with np.load(path, allow_pickle=False) as archive:
        category_key = "background_category" if heldout else "category"
        required = ("event_id", category_key, "ae_score", "md_score",
                    "physics_weight", "ht10_ge30")
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}; rerun the updated audit")
        result = {key: np.asarray(archive[key]) for key in required}
    result["category"] = result.pop(category_key).astype(np.int16)
    result["ae"] = result.pop("ae_score").astype(np.float64)
    result["md"] = result.pop("md_score").astype(np.float64)
    result["weight"] = result.pop("physics_weight").astype(np.float64)
    result["ht"] = result.pop("ht10_ge30").astype(np.float64)
    ids = result["event_id"]
    if ids.ndim != 2 or ids.shape[1] != 4:
        raise ValueError("Composite event IDs must have shape [N, 4]")
    order = np.lexsort(tuple(ids[:, column] for column in (3, 2, 1, 0)))
    result = {key: value[order] for key, value in result.items()}
    if np.any(np.all(result["event_id"][1:] == result["event_id"][:-1], axis=1)):
        raise ValueError(f"Duplicate event IDs in {path}")
    if not all(np.isfinite(result[key]).all() for key in ("ae", "md", "weight", "ht")):
        raise ValueError(f"Non-finite scores or weights in {path}")
    if np.any(result["weight"] < 0) or np.any(result["ae"] <= 0):
        raise ValueError(f"Invalid weights or AE scores in {path}")
    return result


def verify_pair(left, right):
    """Fail closed unless the checkpoints scored the exact same weighted rows."""
    for name in ("event_id", "category"):
        if not np.array_equal(left[name], right[name]):
            raise ValueError(f"Paired {name} values differ")
    for name in ("weight", "ae", "ht"):
        if not np.allclose(left[name], right[name], rtol=1e-5, atol=1e-6):
            raise ValueError(f"Paired {name} values differ")


def region_yields(ae, md, weights, ae_cut, md_cut):
    high_ae, high_md = ae > ae_cut, md > md_cut
    codes = np.where(high_ae, np.where(high_md, 0, 1),
                     np.where(high_md, 2, 3))
    yields = np.bincount(codes, weights=weights, minlength=4).astype(np.float64)
    sumw2 = np.bincount(codes, weights=weights * weights, minlength=4).astype(np.float64)
    raw = np.bincount(codes, minlength=4)
    return codes, yields, sumw2, raw


def closure_from_yields(yields):
    a, b, c, d = yields
    if min(a, b, c, d) <= 0:
        return float("nan"), float("nan")
    predicted_over_observed = float(b * c / (d * a))
    return predicted_over_observed, float(abs(1.0 / predicted_over_observed - 1.0))


def verify_saved_report(scores, summary_path):
    saved = json.loads(Path(summary_path).read_text())["evaluations"]["held-out"]
    diagnostic = saved["diagnostics"]
    if diagnostic["evaluation_protocol"] != "saved_training_validation_split":
        raise ValueError("Expected independent held-out evaluation protocol")
    if not diagnostic["weighted_qcd"] or diagnostic["qcd_events"] != len(scores["ae"]):
        raise ValueError("Saved evaluation is not the expected weighted QCD sample")
    thresholds = saved["thresholds"]
    _, yields, sumw2, raw = region_yields(
        scores["ae"], scores["md"], scores["weight"],
        thresholds["t1"], thresholds["t2"])
    for index, region in enumerate("ABCD"):
        expected = diagnostic["working_point"]["regions"][region]
        if raw[index] != expected["raw_count"] or not np.isclose(
                yields[index], expected["yield"], rtol=1e-6):
            raise ValueError(f"{region} count/yield does not match {summary_path}")
        if not np.isclose(sumw2[index], expected["sumw2"], rtol=1e-6):
            raise ValueError(f"{region} sumw2 does not match {summary_path}")
    return thresholds


def fixed_point(heldout, validation, ae_percentile, target_A_efficiency):
    ae_cut = weighted_quantile(validation["ae"], validation["weight"], ae_percentile)
    high_ae = validation["ae"] > ae_cut
    high_ae_efficiency = float(validation["weight"][high_ae].sum()
                               / validation["weight"].sum())
    if not 0 < target_A_efficiency < high_ae_efficiency:
        raise ValueError("Target A efficiency exceeds validation high-AE mass")
    conditional_md_quantile = 1 - target_A_efficiency / high_ae_efficiency
    md_cut = weighted_quantile(validation["md"][high_ae],
                               validation["weight"][high_ae],
                               conditional_md_quantile)
    _, val_yields, _, _ = region_yields(
        validation["ae"], validation["md"], validation["weight"], ae_cut, md_cut)
    codes, yields, sumw2, raw = region_yields(
        heldout["ae"], heldout["md"], heldout["weight"], ae_cut, md_cut)
    ratio, nonclosure = closure_from_yields(yields)
    ess = np.divide(yields * yields, sumw2, out=np.zeros(4), where=sumw2 > 0)
    return {
        "ae_percentile": ae_percentile,
        "target_A_weighted_efficiency": target_A_efficiency,
        "validation_A_weighted_efficiency": float(val_yields[0] / val_yields.sum()),
        "ae_cut": ae_cut, "md_cut": md_cut,
        "predicted_over_observed": ratio, "absolute_nonclosure": nonclosure,
        "region_A_weighted_efficiency": float(yields[0] / yields.sum()),
        "region_yields": yields.tolist(), "region_raw_counts": raw.tolist(),
        "region_effective_counts": ess.tolist(),
        "statistically_valid": bool(np.all(ess >= np.array([50, 50, 50, 500]))),
    }, codes


def paired_bootstrap(left, right, left_codes, right_codes, *, repeats, seed):
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    rng = np.random.default_rng(seed)
    weights = left["weight"]
    deltas, left_nc, right_nc = [], [], []
    for _ in range(repeats):
        multiplicity = rng.poisson(1.0, size=len(weights))
        weighted = weights * multiplicity
        left_yields = np.bincount(left_codes, weights=weighted, minlength=4)
        right_yields = np.bincount(right_codes, weights=weighted, minlength=4)
        _, lnc = closure_from_yields(left_yields)
        _, rnc = closure_from_yields(right_yields)
        if np.isfinite(lnc) and np.isfinite(rnc):
            left_nc.append(lnc)
            right_nc.append(rnc)
            deltas.append(rnc - lnc)
    if len(deltas) < repeats * 0.95:
        raise ValueError("Too many invalid bootstrap replicates")
    return {
        "method": "paired_event_Poisson_1_fixed_validation_thresholds",
        "repeats": len(deltas), "seed": seed,
        "delta_definition": "candidate absolute nonclosure minus baseline absolute nonclosure",
        "delta_median": float(np.median(deltas)),
        "delta_95pct_interval": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "fraction_candidate_better": float(np.mean(np.asarray(deltas) < 0)),
        "baseline_nonclosure_95pct_interval": np.percentile(left_nc, [2.5, 97.5]).tolist(),
        "candidate_nonclosure_95pct_interval": np.percentile(right_nc, [2.5, 97.5]).tolist(),
        "warning": "Event bootstrap omits shared-file/systematic correlations; held-out comparison is exploratory after prior test inspection.",
    }


def conditional_shift(validation, heldout, *, model_name):
    """Compare MD|AE,H_T on the physics-dominant QCD 15--50 GeV bin."""
    val_soft = validation["category"] == 0
    test_soft = heldout["category"] == 0
    if not val_soft.any() or not test_soft.any():
        raise ValueError("Missing QCD 15--50 GeV category")
    va, vw, vh = (validation[key][val_soft] for key in ("ae", "weight", "ht"))
    ta, tw, th = (heldout[key][test_soft] for key in ("ae", "weight", "ht"))
    ht_edges = [weighted_quantile(vh, vw, p) for p in (1/3, 2/3)]
    if len(set(ht_edges)) < 2:
        ht_edges = list(np.quantile(vh, [1/3, 2/3]))
    if len(set(ht_edges)) < 2:
        distinct = np.unique(vh)
        if len(distinct) < 4:
            raise ValueError("Too few distinct reconstructed HT values")
        ht_edges = [distinct[len(distinct)//3], distinct[2*len(distinct)//3]]
    ae_edges = [weighted_quantile(va, vw, p) for p in np.linspace(0.1, 0.9, 9)]
    if len(set(ht_edges)) < 2 or len(set(ae_edges)) < 9:
        raise ValueError("Degenerate validation-fitted AE/HT slice boundaries")
    rows = []
    for ht_bin in range(3):
        vmh = np.digitize(vh, ht_edges) == ht_bin
        tmh = np.digitize(th, ht_edges) == ht_bin
        for ae_bin in range(10):
            vm = vmh & (np.digitize(va, ae_edges) == ae_bin)
            tm = tmh & (np.digitize(ta, ae_edges) == ae_bin)
            ve, te = effective_count(vw[vm]), effective_count(tw[tm])
            valid = ve >= 30 and te >= 30
            rows.append({
                "model": model_name, "ht_bin": ht_bin, "ae_bin": ae_bin,
                "validation_events": int(vm.sum()), "heldout_events": int(tm.sum()),
                "validation_ess": ve, "heldout_ess": te,
                "validation_md_median": weighted_quantile(validation["md"][val_soft][vm], vw[vm], .5) if valid else float("nan"),
                "heldout_md_median": weighted_quantile(heldout["md"][test_soft][tm], tw[tm], .5) if valid else float("nan"),
                "weighted_md_ks": weighted_ks(validation["md"][val_soft][vm], vw[vm],
                                                heldout["md"][test_soft][tm], tw[tm]) if valid else float("nan"),
            })
    return rows, {"ae_edges": ae_edges, "ht_edges_gev": ht_edges,
                  "minimum_cell_ess": 30}


def make_plots(grid_rows, shift_rows, output_dir):
    figure, ax = plt.subplots(figsize=(7, 4.2))
    for name in ("baseline", "candidate"):
        rows = [row for row in grid_rows if row["model"] == name and row["ae_percentile"] == .5]
        ax.plot([row["target_A_weighted_efficiency"] for row in rows],
                [row["absolute_nonclosure"] for row in rows], marker="o", label=name)
    ax.set(xlabel="Validation target weighted QCD A efficiency",
           ylabel="Held-out absolute nonclosure", title="Fixed AE percentile = 0.5")
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "matched_efficiency_closure.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, name in zip(axes, ("baseline", "candidate")):
        matrix = np.full((3, 10), np.nan)
        for row in shift_rows:
            if row["model"] == name:
                matrix[row["ht_bin"], row["ae_bin"]] = row["weighted_md_ks"]
        image = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=.3)
        ax.set(title=name, xlabel="Validation-fitted AE-loss decile",
               ylabel="Reconstructed HT third")
    figure.colorbar(image, ax=axes, label="Weighted validation-vs-held-out MD KS")
    figure.savefig(output_dir / "conditional_md_shift_qcd15_50.png", dpi=180,
                   bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ht_bin, ax in enumerate(axes):
        for name in ("baseline", "candidate"):
            rows = [row for row in shift_rows if row["model"] == name
                    and row["ht_bin"] == ht_bin]
            for role, marker in (("validation", "o"), ("heldout", "s")):
                ax.plot([row["ae_bin"] for row in rows],
                        [row[f"{role}_md_median"] for row in rows],
                        marker=marker, ms=3, label=f"{name} {role}")
        ax.set(title=f"HT third {ht_bin + 1}", xlabel="AE-loss decile")
    axes[0].set_ylabel("Physics-weighted median QCD MD")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "conditional_md_medians_qcd15_50.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-heldout", "candidate-heldout", "baseline-validation",
                 "candidate-validation", "baseline-evaluation", "candidate-evaluation"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite {args.output_dir}")

    heldout = {name: load_scores(getattr(args, name + "_heldout"), heldout=True)
               for name in ("baseline", "candidate")}
    validation = {name: load_scores(getattr(args, name + "_validation"), heldout=False)
                  for name in ("baseline", "candidate")}
    verify_pair(heldout["baseline"], heldout["candidate"])
    verify_pair(validation["baseline"], validation["candidate"])
    for name in ("baseline", "candidate"):
        verify_saved_report(heldout[name], getattr(args, name + "_evaluation"))

    grid_rows = []
    bootstrap_codes = {}
    bootstrap_points = {}
    for name in ("baseline", "candidate"):
        for p1 in (.5, .7):
            for target_efficiency in (.02, .05, .1, .2):
                point, codes = fixed_point(
                    heldout[name], validation[name], p1, target_efficiency)
                point["model"] = name
                grid_rows.append(point)
                if p1 == .5 and target_efficiency == .1:
                    bootstrap_codes[name] = codes
                    bootstrap_points[name] = point
    if not all(point["statistically_valid"] for point in bootstrap_points.values()):
        raise ValueError("Matched 10% A-efficiency point fails ABCD ESS minimums")
    bootstrap = paired_bootstrap(
        # The fixed point is required to satisfy the same per-region ESS
        # minimums used by statistically-valid production closure reports.
        heldout["baseline"], heldout["candidate"],
        bootstrap_codes["baseline"], bootstrap_codes["candidate"],
        repeats=args.bootstrap_repeats, seed=args.seed)
    shifts, boundaries = [], {}
    for name in ("baseline", "candidate"):
        rows, bounds = conditional_shift(validation[name], heldout[name], model_name=name)
        shifts.extend(rows)
        boundaries[name] = bounds

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "matched_efficiency_abcd.csv").open("w", newline="") as stream:
        fields = [key for key in grid_rows[0] if key not in
                  ("region_yields", "region_raw_counts", "region_effective_counts")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in grid_rows])
    with (args.output_dir / "conditional_md_shift_qcd15_50.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(shifts[0]))
        writer.writeheader()
        writer.writerows(shifts)
    make_plots(grid_rows, shifts, args.output_dir)
    report = {
        "protocol": "paired_same_heldout_events_validation_fitted_matched_marginal_efficiencies",
        "inputs": {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)},
        "paired_heldout_events": len(heldout["baseline"]["ae"]),
        "paired_validation_events": len(validation["baseline"]["ae"]),
        "heldout_weight_sum": float(heldout["baseline"]["weight"].sum()),
        "weighted_ae_ks_validation_vs_heldout": weighted_ks(
            validation["baseline"]["ae"], validation["baseline"]["weight"],
            heldout["baseline"]["ae"], heldout["baseline"]["weight"]),
        "matched_point_p1_050_A_eff_010": bootstrap_points,
        "paired_bootstrap": bootstrap,
        "conditional_slice_boundaries": boundaries,
        "warning": "Outer test has already been inspected in this study; treat these comparisons as diagnostic, not a new blind selection.",
    }
    (args.output_dir / "paired_qcd_audit.json").write_text(json.dumps(report, indent=2))
    print("PAIRED QCD AUDIT DONE", args.output_dir, flush=True)
    print("Matched-point bootstrap delta 95%:", bootstrap["delta_95pct_interval"], flush=True)


if __name__ == "__main__":
    main()
