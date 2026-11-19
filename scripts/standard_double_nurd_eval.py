"""Standard-style frozen double/single NURD evaluation, decay8 and parent4.

Reuse original AE/MD caches and training-QCD MD transforms. Only checking
encoder inference is needed for PCA and probabilities; no refitting/retraining.
The ordinary 50--98% calibration-QCD scan selects the working point. Checking
quantile maps/diagonal curves are descriptive only, as in the ordinary evaluator.
"""
import argparse
import csv
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.audit_qcd_abcd_composition import infer_latents
from scripts.audit_tail_pilot_split import quantile
from scripts.evaluate_double_nurd_pilot import load_weights, verify_model
from scripts.plot_ae_tail_eval_suite import render_suite
from scripts.prepare_tail_pilot import report_json
from scripts.run_double_nurd_pilot import ARMS, PROTOCOL
from scripts.run_tail_nurd_pilot import finite_report
from scripts.scan_cached_tail_cuts import scan
from scripts.study_tail_pilot_cuts import finish, load_nurd_model, qcd, recount
from utils.hlt_data import load_hlt_sample
from utils.hlt_weights import file_sha256, sample_signature

PERCENTILES = np.linspace(.5, .98, 48)
PARENT = np.array([0, 1, 2, 3, 3, 3, 0, 0])
PARENT_NAMES = {0: "Z+jets", 1: "QCD", 2: "ttbar", 3: "W+jets"}
DECAY_NAMES = {0: "Z->ee", 1: "QCD", 2: "ttbar", 3: "W->enu",
               4: "W->munu", 5: "W->taunu", 6: "Z->mumu", 7: "Z->tautau"}


def read_scores(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    count = len(data["ae"])
    if (not count or any(len(data[k]) != count for k in ("md", "label", "weight", "event_id"))
            or not np.isfinite(np.stack([data[k] for k in ("ae", "md", "weight")])).all()
            or np.any(data["weight"] < 0) or data["weight"].sum() <= 0
            or np.any(data["ae"] < 0) or np.any(data["md"] < 0)
            or len(np.unique(data["event_id"], axis=0)) != count):
        raise ValueError(f"Invalid score cache: {path}")
    return data


def selected(data, threshold):
    mask = np.ones(len(data["ae"]), bool) if threshold is None else data["ae"] >= threshold
    return {key: value[mask] for key, value in data.items()}


def scan_data(data, minimums):
    if not len(data["ae"]) or data["weight"].sum() <= 0:
        raise ValueError("Selected QCD has no positive physical mass")
    tx = quantile(data["ae"], data["weight"], PERCENTILES)
    ty = quantile(data["md"], data["weight"], PERCENTILES)
    result = scan(data["ae"], data["md"], data["weight"], tx, ty, minimums)
    return tx, ty, result


def standard_plan(calibration, minimums):
    """Accepts calibration ONLY. Match ordinary evaluator's minimum |1/R-1|."""
    tx, ty, result = scan_data(calibration, minimums)
    ratios = result["ratio"]
    with np.errstate(divide="ignore", invalid="ignore"):
        objective = np.where(result["valid"] & (ratios > 0), abs(1/ratios-1), np.inf)
    index = None if not np.isfinite(objective).any() else tuple(np.unravel_index(np.argmin(objective), objective.shape))
    candidate = None
    if index is not None:
        i, j = index
        candidate = recount(calibration, float(tx[i]), float(ty[j]), minimums)[0]
        candidate.update(calibration_p1=float(PERCENTILES[i]), calibration_p2=float(PERCENTILES[j]),
                         definition="ordinary_calibration_minimum_absolute_nonclosure_50_to_98_percentiles")
    return {"candidate": candidate, "valid_calibration_points": int(result["valid"].sum())}


def standard_check(data, plan, minimums, full_qcd_mass):
    tx, ty, grid = scan_data(data, minimums)
    ratio = grid["ratio"]
    with np.errstate(divide="ignore", invalid="ignore"):
        nc = abs(1/ratio-1)
    valid_nc = np.where(grid["valid"], nc, np.nan)
    point = None
    if plan["candidate"] is not None:
        c = plan["candidate"]
        point = recount(data, c["ae_cut"], c["md_cut"], minimums)[0]
        point["plot_percentiles"] = [float(data["weight"][data[k] <= c[f"{k}_cut"]].sum()/data["weight"].sum()) for k in ("ae", "md")]
        point["A_efficiency_original_full_QCD"] = point["regions"]["A"]["yield"] / full_qcd_mass
        point["selection_source"] = "frozen_calibration_QCD_cuts"
    curve = [{"percentile": float(p), "conditional_A_efficiency": float(grid["efficiency"][i, i]),
              "ratio": float(ratio[i, i]), "uncertainty": float(grid["sigma"][i, i]),
              "valid": bool(grid["valid"][i, i])} for i, p in enumerate(PERCENTILES)]
    values = valid_nc[np.isfinite(valid_nc)]
    result = {"working_point": point, "curve": curve,
        "valid_points": int(grid["valid"].sum()), "total_points": int(grid["valid"].size),
        "grid_median_absolute_nonclosure": float(np.median(values)) if len(values) else None,
        "grid_p90_absolute_nonclosure": float(np.percentile(values, 90)) if len(values) else None}
    return result, {"percentiles": PERCENTILES, "tail_valid_grid": valid_nc,
                    "ae_cuts": tx, "md_cuts": ty, **grid}


def probabilities_from_latent(latent, payload):
    state = payload["state_dict_model"]
    logits = latent @ state["classifier.weight"].numpy().T + state["classifier.bias"].numpy()
    exp = np.exp(logits-logits.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


def taxonomy_view(data, probabilities, taxonomy):
    if taxonomy == "decay8":
        return data, probabilities, DECAY_NAMES
    if taxonomy != "parent4": raise ValueError("Expected decay8 or parent4")
    # Sum the post-softmax probabilities, NOT hard predicted decay labels.
    parents = np.stack([probabilities[:, PARENT == k].sum(1) for k in range(4)], 1)
    return {**data, "label": PARENT[data["label"]]}, parents, PARENT_NAMES


def class_metrics(labels, probabilities, weights, class_count):
    predictions = probabilities.argmax(1)
    raw, physical = np.zeros((class_count, class_count)), np.zeros((class_count, class_count))
    np.add.at(raw, (labels, predictions), 1.)
    np.add.at(physical, (labels, predictions), weights)
    def recalls(matrix):
        return np.divide(matrix.diagonal(), matrix.sum(1), out=np.full(class_count, np.nan), where=matrix.sum(1)>0)
    present = raw.sum(1) > 0
    physical_present = physical.sum(1) > 0
    per_class, per_class_phys = recalls(raw), recalls(physical)
    return {"events": len(labels), "raw_accuracy": float((labels == predictions).mean()),
        "balanced_accuracy": float(per_class[present].mean()),
        "per_class_accuracy": per_class.tolist(), "classes_present": np.flatnonzero(present).tolist(),
        "physics_weighted_accuracy": float((weights*(labels == predictions)).sum()/weights.sum()),
        "physics_weighted_ce": float((weights * -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-30))).sum()/weights.sum()),
        "physics_balanced_accuracy": float(per_class_phys[physical_present].mean()),
        "per_class_physics_accuracy": per_class_phys.tolist(),
        "raw_confusion": raw.tolist(), "physics_confusion": physical.tolist()}


def confusion_plots(metrics, names, output):
    for key, filename in (("raw_confusion", "confusion_matrix.png"), ("physics_confusion", "confusion_matrix_physics.png")):
        values = np.asarray(metrics[key]); totals = values.sum(1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals>0)
        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(names)), names.values(), rotation=55, ha="right")
        ax.set_yticks(range(len(names)), names.values())
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", color="white" if values[i,j]>.55 else "black", fontsize=9)
        ax.set(xlabel="Predicted", ylabel="True", title="Row-normalized " + ("physics-weighted" if key.startswith("physics") else "raw") + " confusion")
        fig.colorbar(mesh, ax=ax, label="Per-class recall fraction")
        finish(fig, output / filename)


def run(args):
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    comparison_path = args.comparison_dir / "double_nurd_comparison.json"
    source = json.loads(comparison_path.read_text())
    manifest = json.loads((args.experiment_dir / "training_complete.json").read_text())
    if source["protocol"] != PROTOCOL or manifest["protocol"] != PROTOCOL:
        raise ValueError("Not a completed matched double-NURD experiment")
    if (set(source["AE_thresholds"]) != {"f100", "f010"}
            or source["AE_thresholds"]["f100"] is not None
            or not np.isfinite(source["AE_thresholds"]["f010"])):
        raise ValueError("Expected original frozen full/top-10% selection")
    paths = {arm: args.experiment_dir / arm / "checkpoint_final.pth.tar" for arm in ARMS}
    hashes = {arm: file_sha256(path) for arm, path in paths.items()}
    if hashes != source["checkpoint_sha256"] or hashes != manifest["final_checkpoint_sha256"]:
        raise ValueError("Final checkpoints differ from cached evaluation")
    source_hash = file_sha256(comparison_path)
    cal_paths = {arm: args.comparison_dir / f"{arm}_calibration_QCD.npz" for arm in ARMS}
    check_paths = {arm: args.comparison_dir / f"{arm}_checking_all_classes.npz" for arm in ARMS}
    cache_hashes = {str(path): file_sha256(path) for path in [*cal_paths.values(), *check_paths.values()]}
    plans = {}; common_calibration = None
    for arm, path in cal_paths.items():
        calibration = read_scores(path)
        if not np.all(calibration["label"] == 1): raise ValueError("Non-QCD calibration rows")
        if common_calibration is None:
            common_calibration = {key: calibration[key] for key in ("event_id", "ae", "weight")}
        elif any(not np.array_equal(calibration[key], value) for key, value in common_calibration.items()):
            raise ValueError("Models have different calibration IDs, AE scores or physics weights")
        plans[arm] = {key: standard_plan(selected(calibration, cut), args.minimums)
                      for key, cut in source["AE_thresholds"].items()}
    args.output_dir.mkdir(parents=True)
    frozen = {"protocol": "double_nurd_standard_percentile_evaluation_v1",
        "checkpoint_sha256": hashes, "source_comparison_sha256": source_hash,
        "AE_thresholds": source["AE_thresholds"], "percentiles": PERCENTILES.tolist(),
        "minimum_region_ESS": args.minimums, "plans": plans,
        "working_point_selection": "Minimum |1/R-1| on calibration QCD's ordinary 50--98% percentile grid; no A-efficiency target",
        "checking_used_to_choose_thresholds": False, "MD_refitted": False,
        "MD_references": source["MD_references"], "source_cache_sha256": cache_hashes,
        "checking_map_and_curve": "Checking-quantile descriptive scans, as in ordinary evaluator; never choose their best cell",
        "parent4_definition": {str(k): np.flatnonzero(PARENT == k).tolist() for k in range(4)},
        "parent4_changes_QCD_ABCD": False, "networks_retrained": False,
        "evaluation_weights": "Original generator-times-cross-section physics weights once, no class factors/clipping",
        "warning": "Exploratory previously inspected MC. Standard optimum is not a fixed-background-rate or signal-sensitivity comparison.",
        "code_commit": args.code_commit}
    plan_path = args.output_dir / "standard_threshold_plan.json"
    plan_path.write_text(report_json(finite_report(frozen))); plan_hash = file_sha256(plan_path)
    prediction_manifest = None
    if args.prediction_cache is not None:
        prediction_manifest = json.loads((args.prediction_cache / "inference_complete.json").read_text())
        if (prediction_manifest["checkpoint_sha256"] != hashes
                or prediction_manifest["source_comparison_sha256"] != source_hash
                or prediction_manifest["standard_threshold_plan_sha256"] != plan_hash):
            raise ValueError("Prediction cache belongs to different models, comparison or frozen cut plan")
    print("All standard calibration working points frozen; opening checking sample now.", flush=True)
    raw = load_hlt_sample(args.heldout)
    if sample_signature(raw) != source["checking_signature"]: raise ValueError("Checking manifest changed")
    role_ids = np.concatenate((raw["event_id"].numpy(), common_calibration["event_id"]))
    if len(np.unique(role_ids, axis=0)) != len(role_ids):
        raise ValueError("Calibration/checking IDs overlap or contain duplicates")
    physics, _ = load_weights(args.heldout_weights, raw, source["checking_physics_weights"]["effective_physics_weight_sha256"])
    all_results = {}; rows = []
    for arm in ARMS:
        data = read_scores(check_paths[arm])
        if (not np.array_equal(data["event_id"], raw["event_id"].numpy())
                or not np.array_equal(data["weight"], physics)):
            raise ValueError("Cached checking IDs/weights differ")
        if args.prediction_cache is None:
            model, payload = load_nurd_model(paths[arm], args.device)
        else:
            model = None
            payload = torch.load(paths[arm], map_location="cpu", weights_only=False)
        verify_model(payload, arm, manifest)
        from utils.hlt_targets import checkpoint_target_labels
        if not np.array_equal(data["label"], checkpoint_target_labels(raw, payload["preprocessing"]).numpy()):
            raise ValueError("Cached checking labels differ")
        if args.prediction_cache is None:
            print(f"{arm}: frozen checking inference for latent PCA and classifier probabilities only ({len(physics):,} rows)", flush=True)
            latent = infer_latents(model, raw, torch.arange(len(physics)), args.device, args.batch_size)
            probabilities = probabilities_from_latent(latent, payload)
        else:
            path = args.prediction_cache / f"{arm}_checking_predictions.npz"
            if file_sha256(path) != prediction_manifest["prediction_sha256"][arm]:
                raise ValueError("Prediction cache changed")
            with np.load(path, allow_pickle=False) as predictions:
                if not np.array_equal(predictions["event_id"], data["event_id"]):
                    raise ValueError("Cached prediction IDs differ")
                latent = predictions["latent"].copy()
                probabilities = predictions["decay8_probabilities"].copy()
            if (probabilities.shape != (len(physics), 8) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1., atol=1e-6)):
                raise ValueError("Invalid cached probabilities")
        reference = {k: np.asarray(source["MD_references"][arm][k]) for k in ("mean", "whitening")}
        md = np.square((latent-reference["mean"]) @ reference["whitening"]).sum(1)
        if not np.allclose(md, data["md"], rtol=2e-3, atol=1e-5): raise ValueError("Frozen MD inference parity failed")
        # Keep archived AE/MD scores exactly; never substitute re-inferred MD.
        data["latent"] = latent
        if args.prediction_cache is None:
            np.savez_compressed(args.output_dir / f"{arm}_checking_predictions.npz", event_id=data["event_id"],
                                latent=latent, decay8_probabilities=probabilities)
        if args.inference_only:
            del model, payload, latent, probabilities, data; gc.collect()
            if args.device == "cuda": torch.cuda.empty_cache()
            continue
        full_qcd_mass = float(data["weight"][data["label"] == 1].sum())
        all_results[arm] = {}
        for key, cut in source["AE_thresholds"].items():
            chosen = selected(data, cut)
            mask = np.ones(len(physics), bool) if cut is None else data["ae"] >= cut
            probs = probabilities[mask]
            closure, grid = standard_check(qcd(chosen), plans[arm][key], args.minimums, full_qcd_mass)
            closure = finite_report(closure)
            folder = args.output_dir / arm / key; folder.mkdir(parents=True)
            np.savez_compressed(folder / "standard_closure_grid.npz", **grid)
            context = {"population": f"{arm}, {'full' if cut is None else 'fixed top-10%'} held-out",
                "point_label": "Calibration-selected fixed cuts", "scan_point_label": "Calibration-selected fixed cuts",
                "scan_title": "Standard ABCD scan — held-out (descriptive)",
                "scan_xlabel": "Held-out QCD AE percentile", "scan_ylabel": "Held-out QCD MD percentile",
                "efficiency_xlabel": "Selection Efficiency (QCD A / selected QCD)"}
            all_results[arm][key] = {"closure": closure, "classification": {}}
            for taxonomy in ("decay8", "parent4"):
                view, view_probs, names = taxonomy_view(chosen, probs, taxonomy)
                metrics = class_metrics(view["label"], view_probs, view["weight"], len(names))
                plot_report = {"scans": {"tail": closure}, "plot_context": context}
                print(f"{arm}/{key}/{taxonomy}: render standard plots; balanced accuracy={metrics['balanced_accuracy']:.4f}", flush=True)
                style = render_suite(view, names, plot_report, grid, reference, folder / taxonomy / "plots", kde_grid=args.kde_grid)
                confusion_plots(metrics, names, folder / taxonomy / "plots")
                result = {"classification": metrics, "closure": closure, "class_names": names, **style}
                (folder / taxonomy / "evaluation_summary.json").write_text(report_json(finite_report(result)))
                all_results[arm][key]["classification"][taxonomy] = metrics
                point = closure["working_point"]
                rows.append({"model": arm, "selection": key, "taxonomy": taxonomy,
                    "balanced_accuracy": metrics["balanced_accuracy"], "physics_accuracy": metrics["physics_weighted_accuracy"],
                    "ratio": None if point is None else point["predicted_over_observed"],
                    "nonclosure": None if point is None else point["absolute_nonclosure"],
                    "valid": None if point is None else point["statistically_valid"],
                    "grid_median_nonclosure": closure["grid_median_absolute_nonclosure"]})
        del model, payload, latent, probabilities, data; gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
    if (file_sha256(plan_path) != plan_hash or file_sha256(comparison_path) != source_hash
            or any(file_sha256(path) != digest for path, digest in cache_hashes.items())
            or {arm: file_sha256(path) for arm, path in paths.items()} != hashes):
        raise ValueError("Source model/cache or frozen plan changed")
    if args.inference_only:
        report = {"checkpoint_sha256": hashes, "source_comparison_sha256": source_hash,
            "standard_threshold_plan_sha256": plan_hash,
            "prediction_sha256": {arm: file_sha256(args.output_dir / f"{arm}_checking_predictions.npz") for arm in ARMS},
            "original_inputs_unchanged": True}
        (args.output_dir / "inference_complete.json").write_text(report_json(report))
        print("Frozen inference complete. GPU can be released; render standard suites on CPU.", flush=True)
        return
    (args.output_dir / "standard_evaluation_summary.json").write_text(report_json(finite_report({**frozen,
        "results": all_results, "original_inputs_unchanged": True, "signal_sample_evaluated": False})))
    with (args.output_dir / "classification_and_closure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"STANDARD DOUBLE/SINGLE EVALUATION DONE: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison-dir", "experiment-dir", "heldout", "heldout-weights", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--minimums", nargs=4, type=int, default=[50, 50, 50, 100])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--kde-grid", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--inference-only", action="store_true", help="Write frozen predictions for both models; no plots")
    parser.add_argument("--prediction-cache", type=Path, default=None, help="Verified inference-only output; render on CPU without PF inference")
    args = parser.parse_args()
    if min(args.minimums) < 1 or args.batch_size < 1 or args.kde_grid < 8 or args.threads < 1:
        parser.error("Positive ESS minima, batch, threads; KDE grid >=8 required")
    if args.inference_only and args.prediction_cache is not None:
        parser.error("Choose inference-only OR an existing prediction cache")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads): run(args)


if __name__ == "__main__": main()
