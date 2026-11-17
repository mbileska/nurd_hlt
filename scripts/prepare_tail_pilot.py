"""Prepare a frozen training-defined AE tail; no PF inference or training."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.hlt_autoencoder import HLTAutoencoder
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import (
    apply_joint_balance, file_sha256, fit_joint_balance,
    load_generator_weights, sample_signature, tensor_sha256,
)


def validate_split(train, validation, size):
    train, validation = (np.asarray(v, dtype=np.int64) for v in (train, validation))
    combined = np.concatenate((train, validation))
    if (train.ndim != 1 or validation.ndim != 1 or not len(train) or not len(validation)
            or len(combined) != size or np.unique(combined).size != size
            or combined.min() != 0 or combined.max() != size-1):
        raise ValueError("Checkpoint splits must partition the complete training manifest")
    return train, validation


def freeze_selection(scores, weights, train, validation, fraction=.1):
    scores, weights = np.asarray(scores), np.asarray(weights, dtype=np.float64)
    train, validation = validate_split(train, validation, len(scores))
    if (scores.shape != weights.shape or not np.isfinite(scores).all()
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or np.any(scores < 0) or not 0 < fraction < 1):
        raise ValueError("Invalid scores, physics weights, or tail fraction")
    positive = train[weights[train] > 0]
    if not len(positive):
        raise ValueError("Training reference has zero physical mass")
    order = positive[np.argsort(scores[positive], kind="stable")]
    cumulative = np.cumsum(weights[order])
    cut = float(scores[order[min(np.searchsorted(cumulative, (1-fraction)*cumulative[-1]), len(order)-1)]])
    # Boundary ties are included, with no use of labels in selection.
    selected = scores >= cut
    return cut, selected, train[selected[train]], validation[selected[validation]]


def weight_stats(weights):
    weights = np.asarray(weights, dtype=np.float64)
    mass, sumw2 = float(weights.sum()), float(np.square(weights).sum())
    ess = mass*mass/sumw2 if sumw2 else 0.
    return {"events": len(weights), "sum_weights": mass, "sum_weights_squared": sumw2,
            "effective_events": ess, "ess_fraction": ess/len(weights) if len(weights) else 0.}


def summarize(labels, weights, train, validation, selected_train, selected_validation, names):
    rows = []
    for split, original, selected in (("train", train, selected_train),
                                       ("validation", validation, selected_validation)):
        total = weights[selected].sum()
        for label, name in sorted(names.items()):
            source_rows = original[labels[original] == label]
            selected_rows = selected[labels[selected] == label]
            stats = weight_stats(weights[selected_rows])
            source_mass = float(weights[source_rows].sum())
            rows.append({"split": split, "label": label, "name": name,
                         "source_events": len(source_rows), **stats,
                         "raw_retention": len(selected_rows)/len(source_rows) if len(source_rows) else 0.,
                         "physics_retention": stats["sum_weights"]/source_mass if source_mass else 0.,
                         "fraction_of_selected_physics": stats["sum_weights"]/total if total else 0.})
    return rows


def infer_scores(model, scaler, raw, batch_size):
    # Match build_hlt_datasets' training nuisance arithmetic exactly.
    mu, std = scaler["mu"].cpu().float(), scaler["std"].cpu().float()
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    scores = np.empty(len(raw["label"]), dtype=np.float32)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(scores), batch_size):
            obj = torch.nan_to_num(raw["obj"][start:start+batch_size, :, :4].reshape(-1, mu.numel()).float(),
                                   nan=0., posinf=0., neginf=0.)
            batch = (obj-mu)/std
            reco, _ = model(batch)
            scores[start:start+len(batch)] = (reco-batch).square().mean(1).numpy()
            if start == 0 or (start//batch_size) % 25 == 0 or start+len(batch) == len(scores):
                done = start+len(batch)
                elapsed = time.monotonic()-started
                print(f"AE preparation: {done:,}/{len(scores):,}; elapsed={elapsed:.1f}s; "
                      f"estimated remaining={elapsed*(len(scores)-done)/done:.1f}s", flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "ae-checkpoint", "sample", "weights", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("Positive threads and batch size required")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pre = checkpoint["preprocessing"]
    if pre["target_scheme"] != "decay8" or checkpoint.get("classification_head_trained") is False:
        raise ValueError("Expected a full decay8 classifier/NURD checkpoint")
    ae_hash = file_sha256(args.ae_checkpoint)
    if ae_hash != checkpoint["ae_checkpoint_sha256"]:
        raise ValueError("AE hash differs from the classifier's frozen AE")
    ae_payload = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    if ae_payload.get("weighting", {}).get("scheme") != "physics":
        raise ValueError("Expected the nominal physics-weighted AE, not parent balancing")
    raw = load_hlt_sample(args.sample)
    signature = sample_signature(raw)
    if signature != pre["data_signature"] or "event_id" not in raw:
        raise ValueError("Sample does not match the saved full training manifest")
    weights, metadata = load_generator_weights(str(args.weights), raw["label"], sample=raw,
                                               qcd_label=1, weight_scope="all_events")
    if metadata["effective_physics_weight_sha256"] != pre["weighting"]["generator"]["effective_physics_weight_sha256"]:
        raise ValueError("Original physics weights differ from checkpoint provenance")
    labels = checkpoint_target_labels(raw, pre).numpy()
    names = {int(k): str(v) for k, v in pre["target_class_names"].items()}
    if set(names) != set(range(8)):
        raise ValueError("Expected all eight target classes in checkpoint")
    train, validation = validate_split(
        checkpoint["preprocessing"]["split"]["train_indices"].cpu().numpy(),
        checkpoint["preprocessing"]["split"]["validation_indices"].cpu().numpy(), len(labels))
    model = HLTAutoencoder(ae_payload["ae_config"])
    model.load_state_dict(ae_payload["ae"])
    model.eval()
    scores = infer_scores(model, checkpoint["ae_scaler"], raw, args.batch_size)
    physical = weights.double().numpy()
    cut, selected, selected_train, selected_validation = freeze_selection(scores, physical, train, validation)
    rows = summarize(labels, physical, train, validation, selected_train, selected_validation, names)
    complete_train = all(r["events"] > 0 and r["sum_weights"] > 0 for r in rows if r["split"] == "train")
    complete_validation = all(r["events"] > 0 and r["sum_weights"] > 0 for r in rows if r["split"] == "validation")
    effective_train = effective_validation = np.array([], dtype=np.float32)
    balance_spec = None
    if complete_train:
        effective_train, balance_spec = fit_joint_balance(
            torch.from_numpy(labels[selected_train]), torch.from_numpy(scores[selected_train]),
            torch.from_numpy(physical[selected_train]), n_strata=1, binning="log_fixed", clip_quantile=.995)
        effective_validation = apply_joint_balance(
            torch.from_numpy(labels[selected_validation]), torch.from_numpy(scores[selected_validation]),
            torch.from_numpy(physical[selected_validation]), balance_spec) if len(selected_validation) else torch.empty(0)
        effective_train, effective_validation = effective_train.numpy(), effective_validation.numpy()
        for row in rows:
            indices, effective = ((selected_train, effective_train) if row["split"] == "train"
                                  else (selected_validation, effective_validation))
            stats = weight_stats(effective[labels[indices] == row["label"]])
            row["class_balanced_effective_events"] = stats["effective_events"]
            row["class_balanced_mass_fraction"] = stats["sum_weights"]/effective.sum() if effective.sum() else 0.
    report = {
        "protocol": "training_only_physics_all_AE_top10_pilot_preparation",
        "classifier_checkpoint": str(args.checkpoint.resolve()),
        "classifier_checkpoint_sha256": file_sha256(args.checkpoint), "ae_checkpoint_sha256": ae_hash,
        "sample": str(args.sample.resolve()), "data_signature": signature,
        "physics_weight_sha256": metadata["effective_physics_weight_sha256"],
        "ae_threshold": cut, "requested_retention": .1, "threshold_fit_split": "saved_classifier_training_only",
        "selection_rule": "AE_MSE >= frozen_threshold; include boundary ties; no labels used",
        "ae_score_arithmetic": "training_dataset: ((obj-mu)/clamped_std), reconstruction MSE; no epsilon added",
        "training_retained_physics_fraction": float(physical[selected_train].sum()/physical[train].sum()),
        "validation_retained_physics_fraction": float(physical[selected_validation].sum()/physical[validation].sum()),
        "training_boundary_physics_fraction": float(physical[train[scores[train] == cut]].sum()/physical[train].sum()),
        "original_train_indices_sha256": tensor_sha256(train), "original_validation_indices_sha256": tensor_sha256(validation),
        "all_eight_training_classes_present": complete_train,
        "all_eight_validation_classes_present": complete_validation,
        "class_coverage_ready": complete_train and complete_validation,
        "training_submitted": False, "heldout_opened": False, "networks_updated": False,
        "weighting_recipe": "physics weights for selection/eval; train-fitted class-only balancing with .995 clipping for pilot optimization",
        "classes": rows,
        "totals": {name: {"physics": weight_stats(physical[indices]), "class_balanced": weight_stats(effective)}
                   for name, indices, effective in (("train", selected_train, effective_train),
                                                    ("validation", selected_validation, effective_validation))},
        "warnings": [f"{r['split']} {r['name']}: physics ESS below 50" for r in rows if r["effective_events"] < 50],
        "note": "Class coverage is not a claim of sufficient ABCD region statistics. No test/held-out inference. "
                "Freeze this threshold for both pilot branches' validation comparisons.",
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir/"pilot_selection.npz", ae_score=scores, label=labels,
                        physics_weight=physical, event_id=raw["event_id"].numpy(), selected=selected,
                        train_indices=train, validation_indices=validation,
                        selected_train_indices=selected_train, selected_validation_indices=selected_validation,
                        selected_train_effective_weights=effective_train, selected_validation_effective_weights=effective_validation)
    torch.save({"balance_spec": balance_spec, "ae_threshold": cut, "provenance": report},
               args.output_dir/"pilot_preparation.pth")
    with (args.output_dir/"class_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir/"preparation_report.json").write_text(json.dumps(report, indent=2))
    print(f"Frozen TRAINING-derived AE threshold: {cut:.9g}", flush=True)
    for row in rows:
        print(f"{row['split']:10s} {row['name']:12s} N={row['events']:8d} "
              f"physics ESS={row['effective_events']:10.1f} ESS/N={row['ess_fraction']:.4f}", flush=True)
    print(f"All eight classes in both selected splits: {report['class_coverage_ready']}")
    print(f"Report: {args.output_dir/'preparation_report.json'}")
    print("PREPARATION COMPLETE; NO TRAINING SUBMITTED", flush=True)


if __name__ == "__main__":
    main()
