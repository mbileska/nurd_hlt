"""Full + frozen top-10% checks for the matched double-NURD experiment.

One full-training-QCD MD reference per model; original physical evaluation
weights; all ABCD choices frozen on calibration BEFORE opening checking data.
No score calibration, retraining, checking-based epoch selection or cut tuning.
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
from scripts.compare_pilot_retentions import (
    GRID_PROBABILITIES, at_cuts, checking_curve, fraction_name, matched_curve,
    role_selection, subset, target_name,
)
from scripts.compare_qcd_score_audit import paired_bootstrap
from scripts.plot_pilot_retentions import render
from scripts.prepare_tail_pilot import report_json
from scripts.run_double_nurd_pilot import ARMS, PROTOCOL
from scripts.run_tail_nurd_pilot import finite_report, md_reference
from scripts.scan_cached_tail_cuts import scan
from scripts.study_tail_pilot_cuts import finish, load_nurd_model, qcd, save_scores
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import apply_joint_balance, file_sha256, load_generator_weights, sample_signature

FRACTIONS = (1., .1)
# Include high-AE candidates missing from the historical 5--95% search.
PROBABILITIES = np.linspace(.05, .99, 95)
MINIMUMS = (50, 50, 50, 100)
TARGETS = (.005, .01, .02)


def verify_model(payload, arm, manifest):
    if (payload.get("pilot_protocol") != PROTOCOL or payload.get("arm") != arm
            or payload.get("encoder_updates") != manifest["updates_per_arm"]
            or payload["ae_checkpoint_sha256"] != manifest["AE_sha256"]
            or payload["source_sha256"] != manifest["source_sha256"]):
        raise ValueError("Expected unchanged matched final checkpoints")


def classification(latent, payload, labels, physics, ae_scores):
    # The classifier is a linear layer on precisely this saved encoder latent.
    state = payload["state_dict_model"]
    logits = latent @ state["classifier.weight"].numpy().T + state["classifier.bias"].numpy()
    shifted = logits - logits.max(1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(1, keepdims=True)
    effective = apply_joint_balance(torch.as_tensor(labels), torch.as_tensor(ae_scores),
        torch.as_tensor(physics), payload["balance_spec"]).numpy()
    predictions = probabilities.argmax(1)
    raw = np.zeros((8, 8)); weighted = np.zeros((8, 8))
    np.add.at(raw, (labels, predictions), 1.)
    np.add.at(weighted, (labels, predictions), physics)
    per_class = np.divide(raw.diagonal(), raw.sum(1), out=np.full(8, np.nan), where=raw.sum(1)>0)
    # Sum eight probabilities into parents, rather than merely relabeling argmax.
    parents = np.array([0, 1, 2, 3, 3, 3, 0, 0])
    parent_probs = np.stack([probabilities[:, parents == k].sum(1) for k in range(4)], 1)
    parent_confusion = np.zeros((4, 4))
    np.add.at(parent_confusion, (parents[labels], parent_probs.argmax(1)), 1.)
    return {
        "balanced_accuracy": float(np.mean(per_class)),
        "per_class_accuracy": per_class.tolist(),
        "weighted_accuracy_common_training_factors": float((effective * (predictions == labels)).sum()/effective.sum()),
        "physics_weighted_accuracy": float((physics * (predictions == labels)).sum()/physics.sum()),
        "weighted_ce_common_training_factors": float((effective * -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-30))).sum()/effective.sum()),
        "raw_confusion": raw.tolist(), "physics_confusion": weighted.tolist(),
        "parent4_raw_confusion": parent_confusion.tolist(),
        "parent4_balanced_accuracy": float(np.mean(np.divide(parent_confusion.diagonal(), parent_confusion.sum(1),
            out=np.full(4, np.nan), where=parent_confusion.sum(1)>0))),
    }


def load_weights(path, sample, expected=None):
    tensor, metadata = load_generator_weights(str(path), sample["label"], sample=sample,
        qcd_label=1, weight_scope="all_events")
    if expected is not None and metadata["effective_physics_weight_sha256"] != expected:
        raise ValueError("Physics weights changed")
    return tensor.numpy(), metadata


def grid_summary(grid):
    valid = grid["valid"] & np.isfinite(grid["ratio"]) & (grid["ratio"] > 0)
    ratios = grid["ratio"][valid]
    result = {"valid_points": int(valid.sum()), "total_points": int(valid.size)}
    for name, values in (("absolute_ratio_bias", abs(ratios-1)),
                         ("legacy_absolute_nonclosure", abs(1/ratios-1))):
        result[name] = {"median": float(np.median(values)), "p90": float(np.percentile(values, 90))} if len(values) else None
    return result


def plot_classification(metrics, names, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for row, arm in enumerate(ARMS):
        for col, key in enumerate(("f100", "f010")):
            values = np.asarray(metrics[arm][key]["raw_confusion"])
            values = np.divide(values, values.sum(1, keepdims=True), out=np.zeros_like(values), where=values.sum(1, keepdims=True)>0)
            ax = axes[row, col]
            mesh = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(8), list(names.values()), rotation=60, ha="right")
            ax.set_yticks(range(8), list(names.values()))
            ax.set(xlabel="Predicted", ylabel="True", title=f"{arm}: {key[1:]}% retained; raw per-class recall")
            fig.colorbar(mesh, ax=ax)
    finish(fig, output / "classification_confusions.png")


def run(args):
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    manifest = json.loads((args.experiment_dir / "training_complete.json").read_text())
    if manifest["protocol"] != PROTOCOL: raise ValueError("Wrong experiment protocol")
    manifest_hash = file_sha256(args.experiment_dir / "training_complete.json")
    paths = {arm: args.experiment_dir / arm / "checkpoint_final.pth.tar" for arm in ARMS}
    hashes = {arm: file_sha256(path) for arm, path in paths.items()}
    if hashes != manifest["final_checkpoint_sha256"]: raise ValueError("Final checkpoints changed")
    cache_path = args.ae_cache_dir / "frozen_AE_scores.npz"
    if file_sha256(cache_path) != manifest["AE_cache_sha256"]: raise ValueError("AE cache changed")
    raw = load_hlt_sample(args.sample)
    if sample_signature(raw) != manifest["data_signature"]: raise ValueError("Training manifest changed")
    weights, _ = load_weights(args.weights, raw, manifest["physics_weight_sha256"])
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache["training_event_id"], raw["event_id"].numpy()): raise ValueError("Calibration AE IDs changed")
        scores = cache["baseline_training_manifest_scores"].copy()
    with np.load(args.experiment_dir / "training_roles.npz", allow_pickle=False) as roles:
        calibration = roles["calibration"].copy()
        reference_rows = roles["global_QCD"].copy()
        for name in ("full_train", "tail_train", "global_QCD", "tail_QCD", "ae_gradient_train"):
            if np.intersect1d(calibration, roles[name]).size: raise ValueError("Calibration/training overlap")
    source = torch.load(paths["global"], map_location="cpu", weights_only=False)
    pre = source["preprocessing"]
    labels = checkpoint_target_labels(raw, pre).numpy()
    if not np.all(labels[reference_rows] == 1): raise ValueError("Non-QCD in MD reference")
    cuts = {"f100": None, "f010": manifest["ae_threshold"]}
    if not np.isclose(cuts["f010"], quantile(scores[calibration], weights[calibration], .9), rtol=1e-7, atol=1e-9):
        raise ValueError("Frozen AE threshold changed")
    calibration_selection = {key: role_selection(scores[calibration], weights[calibration], labels[calibration], cut)
                              for key, cut in cuts.items()}
    qrows = calibration[labels[calibration] == 1]
    full_cal_mass = float(weights[qrows].sum())
    train_ids = raw["event_id"].numpy()
    args.output_dir.mkdir(parents=True)
    plans = {}; grids = {}; references = {}; cal_data = {}
    for arm, path in paths.items():
        print(f"{arm}: fit MD ONCE on {len(reference_rows):,} full-range training QCD rows", flush=True)
        model, payload = load_nurd_model(path, args.device); verify_model(payload, arm, manifest)
        mean, whitening = md_reference(model, raw, reference_rows, weights, args.device, args.batch_size)
        references[arm] = {"mean": mean.tolist(), "whitening": whitening.tolist(),
                           "training_QCD_events": len(reference_rows)}
        latent = infer_latents(model, raw, torch.as_tensor(qrows), args.device, args.batch_size)
        data = {"ae": scores[qrows], "md": np.square((latent - mean) @ whitening).sum(1),
                "weight": weights[qrows], "label": labels[qrows],
                "event_id": raw["event_id"][qrows].numpy()}
        if not np.isfinite(data["md"]).all(): raise ValueError("Nonfinite MD scores")
        save_scores(args.output_dir / f"{arm}_calibration_QCD.npz", data)
        cal_data[arm] = data; plans[arm] = {}; grids[arm] = {}
        for key, cut in cuts.items():
            selected = subset(data, cut)
            plans[arm][key] = {target_name(t): matched_curve(selected, full_cal_mass, t, args.minimums,
                probabilities=PROBABILITIES) for t in TARGETS}
            tx = quantile(selected["ae"], selected["weight"], GRID_PROBABILITIES)
            ty = quantile(selected["md"], selected["weight"], GRID_PROBABILITIES)
            grid = scan(selected["ae"], selected["md"], selected["weight"], tx, ty, args.minimums)
            grids[arm][key] = {"ae_cuts": tx, "md_cuts": ty, "calibration": grid}
            np.savez_compressed(args.output_dir / f"{arm}_{key}_calibration_grid.npz", **grids[arm][key]["calibration"], ae_cuts=tx, md_cuts=ty)
        del model, payload; gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
    plan = {"protocol": PROTOCOL, "AE_thresholds": cuts, "calibration_selection": calibration_selection,
        "plans": plans, "minimum_region_ESS": args.minimums, "overall_QCD_A_targets": TARGETS,
        "primary_overall_QCD_A_target": .01, "checkpoint_sha256": hashes, "MD_references": references,
        "MD_reference": "One original-physics-weighted full-training-QCD reference per model, identical for full and tail evaluation",
        "checking_opened_during_selection": False, "networks_updated": False,
        "AE_cut_candidate_percentiles": PROBABILITIES.tolist(),
        "AE_threshold_origin": "AE-gradient-unseen all-background physical calibration 90th percentile; fixed before pilot",
        "cut_objective": "On calibration only: fixed overall QCD A efficiency; three-point stable median |R-1| plus central uncertainty",
        "warning": "Exploratory, previously inspected MC; no signal sensitivity test. Checking grids are descriptive, never used to choose cuts.",
        "classification_constraint": "Flag double-model balanced-accuracy loss >5 percentage points versus matched global control; do not select another checking epoch",
        "code_commit": args.code_commit}
    frozen = args.output_dir / "frozen_comparison_plan.json"
    frozen.write_text(report_json(finite_report(plan))); plan_hash = file_sha256(frozen)
    print("Both final models and ALL full/tail cut plans frozen. Opening checking sample now.", flush=True)
    del raw, weights, labels, scores, source
    check = load_hlt_sample(args.heldout)
    weights, weight_metadata = load_weights(args.heldout_weights, check)
    labels = checkpoint_target_labels(check, pre).numpy()
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache["heldout_event_id"], check["event_id"].numpy()): raise ValueError("Checking AE IDs changed")
        scores = cache["baseline_heldout_scores"].copy()
    ids = np.concatenate((train_ids, check["event_id"].numpy()))
    if len(np.unique(ids, axis=0)) != len(ids): raise ValueError("Training/checking IDs overlap or have duplicates")
    checking_selection = {key: role_selection(scores, weights, labels, cut) for key, cut in cuts.items()}
    full_check_mass = float(weights[labels == 1].sum())
    checking = {}; results = {}; metrics = {}; paired = {}
    for arm, path in paths.items():
        print(f"{arm}: one inference over {len(labels):,} checking events, reused for both selections", flush=True)
        model, payload = load_nurd_model(path, args.device); verify_model(payload, arm, manifest)
        latent = infer_latents(model, check, torch.arange(len(labels)), args.device, args.batch_size)
        mean = np.asarray(references[arm]["mean"]); whitening = np.asarray(references[arm]["whitening"])
        data = {"ae": scores, "md": np.square((latent - mean) @ whitening).sum(1),
                "weight": weights, "label": labels, "event_id": check["event_id"].numpy()}
        if not np.isfinite(data["md"]).all(): raise ValueError("Nonfinite checking MD")
        save_scores(args.output_dir / f"{arm}_checking_all_classes.npz", data)
        checking[arm] = data; results[arm] = {}; metrics[arm] = {}
        for key, cut in cuts.items():
            mask = np.ones(len(labels), bool) if cut is None else scores >= cut
            metrics[arm][key] = classification(latent[mask], payload, labels[mask], weights[mask], scores[mask])
            selected = qcd(subset(data, cut)); grid = grids[arm][key]
            grid["checking"] = scan(selected["ae"], selected["md"], selected["weight"], grid["ae_cuts"], grid["md_cuts"], args.minimums)
            np.savez_compressed(args.output_dir / f"{arm}_{key}_checking_grid.npz", **grid["checking"])
            results[arm][key] = {}
            for target in TARGETS:
                name = target_name(target); fitted = plans[arm][key][name]; c = fitted["candidate"]
                point = None if c is None else at_cuts(selected, (c["ae_cut"], c["md_cut"]), full_check_mass, args.minimums)[0]
                results[arm][key][name] = {"calibration_candidate": c, "checking_point": point,
                    "checking_curve": checking_curve(selected, full_check_mass, fitted["curve"], args.minimums),
                    "status": "no_stable_calibration_candidate" if c is None else "frozen_candidate_checked"}
        del model, payload; gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
    primary = target_name(.01)
    for key, cut in cuts.items():
        arrays = [qcd(subset(checking[arm], cut)) for arm in ARMS]
        candidates = [plans[arm][key][primary]["candidate"] for arm in ARMS]
        if any(c is None for c in candidates): paired[key] = None; continue
        if (not np.array_equal(arrays[0]["event_id"], arrays[1]["event_id"])
                or not np.array_equal(arrays[0]["weight"], arrays[1]["weight"])):
            raise ValueError("Paired checking events/physics weights differ")
        codes = [at_cuts(a, (c["ae_cut"], c["md_cut"]), full_check_mass, args.minimums)[1] for a, c in zip(arrays, candidates)]
        try: paired[key] = paired_bootstrap(*arrays, *codes, repeats=500, seed=42)
        except ValueError as error: paired[key] = {"available": False, "reason": str(error)}
    names = {int(k): v for k, v in pre["target_class_names"].items()}
    style = render(plan, results, cal_data, checking, grids, checking_selection, names,
        args.output_dir / "plots", args.kde_grid, branches=ARMS, fractions=FRACTIONS, probabilities=PROBABILITIES)
    plot_classification(metrics, names, args.output_dir / "plots")
    if file_sha256(frozen) != plan_hash or file_sha256(args.experiment_dir / "training_complete.json") != manifest_hash:
        raise ValueError("Frozen plan or completed training manifest changed")
    if {arm: file_sha256(path) for arm, path in paths.items()} != hashes: raise ValueError("Final checkpoints changed")
    deltas = {key: metrics["double"][key]["balanced_accuracy"] - metrics["global"][key]["balanced_accuracy"] for key in cuts}
    report = {**plan, "results": results, "classification": metrics, "checking_selection": checking_selection,
        "metric_definitions": {"ratio": "R=BC/(DA)=predicted/observed QCD",
            "legacy_absolute_nonclosure": "|1/R-1|, matching historical tables",
            "absolute_ratio_bias": "|R-1|, used for cut ranking and closure-map colors"},
        "checking_grid_summary": {arm: {key: grid_summary(grids[arm][key]["checking"]) for key in cuts} for arm in ARMS},
        "checking_physics_weights": weight_metadata, "checking_signature": sample_signature(check),
        "double_minus_global_balanced_accuracy": deltas,
        "classification_guardrail_passed": {key: value >= -.05 for key, value in deltas.items()},
        "paired_double_minus_global_nonclosure": paired,
        "bootstrap_limitation": "Fixed-cut event statistics only; not calibration-selection, retraining or systematics",
        "original_checkpoints_unchanged": True, **style}
    (args.output_dir / "double_nurd_comparison.json").write_text(report_json(finite_report(report)))
    with (args.output_dir / "comparison.csv").open("w", newline="") as stream:
        fields = ["model", "retention", "overall_A_target", "ratio", "ratio_error", "valid", "balanced_accuracy", "A_ESS", "B_ESS", "C_ESS", "D_ESS"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for arm in ARMS:
            for fraction in FRACTIONS:
                key = fraction_name(fraction)
                for target in TARGETS:
                    value = results[arm][key][target_name(target)]["checking_point"]
                    row = {"model": arm, "retention": fraction, "overall_A_target": target,
                           "balanced_accuracy": metrics[arm][key]["balanced_accuracy"]}
                    if value is not None:
                        row.update(ratio=value["predicted_over_observed"], ratio_error=value["ratio_uncertainty"], valid=value["statistically_valid"])
                        row.update({r+"_ESS": value["regions"][r]["effective_count"] for r in "ABCD"})
                    writer.writerow(row)
    print(f"DOUBLE NURD EVALUATION DONE: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment-dir", "sample", "weights", "heldout", "heldout-weights", "ae-cache-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--kde-grid", type=int, default=64)
    parser.add_argument("--minimums", nargs=4, type=int, default=MINIMUMS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", default="")
    args = parser.parse_args()
    if args.batch_size < 1 or args.kde_grid < 8 or args.threads < 1 or min(args.minimums) < 1:
        parser.error("Positive batch, threads and ESS minima; KDE grid >=8 required")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads): run(args)


if __name__ == "__main__":
    main()
