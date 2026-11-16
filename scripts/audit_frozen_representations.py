"""Frozen decay8 encoder/projector audit on the saved train/validation split.

No network training, held-out data, or hyperparameter search. Linear probes use
the checkpoint's training-fitted effective event measure. kNN reports both an
unweighted geometric control and weighted votes on the same support. QCD MD
references and percentile thresholds use training physics weights only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from eval_abcd_nurd import (
    _fit_class_transform, abcd_region_statistics_at_thresholds, abcd_yields,
    checkpoint_reference_indices, closure_ratio_and_uncertainty, load_ae,
    load_nurd_model, nonclosure_A, statistically_valid_regions, weighted_quantile,
)
from scripts.scan_validation_checkpoints import _compute_ae_scores
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import PARENT4_NAMES, checkpoint_target_labels
from utils.hlt_weights import (
    apply_joint_balance, file_sha256, load_generator_weights, sample_signature,
)


DECAY_TO_PARENT = np.array([0, 1, 2, 3, 3, 3, 0, 0], dtype=np.int64)
SPACES = ("encoder", "projector")


def parent_mapping(preprocessing):
    mapping = {int(k): int(v) for k, v in preprocessing["label_map"].items()}
    if (preprocessing.get("target_scheme") != "decay8"
            or set(mapping) != set(range(8)) or set(mapping.values()) != set(range(8))
            or mapping[1] != 1):
        raise ValueError("This audit requires a complete decay8 checkpoint.")
    result = np.empty(8, dtype=np.int64)
    for source, destination in mapping.items():
        result[destination] = DECAY_TO_PARENT[source]
    return result


def parent_probabilities(probabilities, mapping):
    return np.stack([probabilities[:, mapping == label].sum(axis=1)
                     for label in range(4)], axis=1)


def classification_metrics(labels, predictions, n_classes, weights):
    counts = confusion_matrix(labels, predictions, labels=np.arange(n_classes),
                              sample_weight=weights).astype(np.float64)
    mass = counts.sum(axis=1)
    recalls = np.divide(np.diag(counts), mass, out=np.zeros(n_classes), where=mass > 0)
    normalized = np.divide(counts, mass[:, None], out=np.zeros_like(counts),
                           where=mass[:, None] > 0)
    return {
        "accuracy": float(np.trace(counts) / counts.sum()),
        "balanced_accuracy": float(recalls.mean()),
        "per_class_recall": recalls.tolist(),
        "confusion_true_by_predicted": counts.tolist(),
        "row_normalized_confusion": normalized.tolist(),
    }


def report_predictions(probabilities, labels, mapping, measures):
    prediction = probabilities.argmax(axis=1)
    parent_truth = mapping[labels]
    parent_sum = parent_probabilities(probabilities, mapping).argmax(axis=1)
    parent_mapped = mapping[prediction]
    reports = {}
    for name, weights in measures.items():
        reports[name] = {
            "decay8": classification_metrics(labels, prediction, 8, weights),
            "parent4_probability_sum": classification_metrics(
                parent_truth, parent_sum, 4, weights),
            "parent4_mapped_argmax": classification_metrics(
                parent_truth, parent_mapped, 4, weights),
        }
    return reports


def training_support(labels, cap_per_class, seed):
    """Uniform within class, with known inclusion probability for vote weights."""
    rng = np.random.default_rng(seed)
    selected, inclusion = [], []
    for label in np.unique(labels):
        pool = np.flatnonzero(labels == label)
        count = min(len(pool), cap_per_class)
        selected.append(rng.choice(pool, count, replace=False))
        inclusion.append(np.full(count, count / len(pool)))
    return np.concatenate(selected), np.concatenate(inclusion)


def knn_probabilities(distances, neighbors, support_labels, vote_weights, n_classes=8):
    zero = distances <= 1e-12
    inverse = 1.0 / np.maximum(distances, 1e-12)
    inverse[zero.any(axis=1)] = zero[zero.any(axis=1)].astype(float)
    votes = inverse * vote_weights[neighbors]
    probabilities = np.zeros((len(neighbors), n_classes), dtype=np.float64)
    for label in range(n_classes):
        probabilities[:, label] = (votes * (support_labels[neighbors] == label)).sum(axis=1)
    denominator = probabilities.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError("kNN produced a zero-weight vote.")
    return (probabilities / denominator).astype(np.float32)


def fit_linear_probe(train, validation, train_labels, train_weights, support, max_iter):
    """Fixed C=1 linear probe. No validation-fitted quantities."""
    scaler = StandardScaler().fit(train, sample_weight=train_weights)
    standardized_train = scaler.transform(train)
    standardized_validation = scaler.transform(validation)
    linear = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter, tol=1e-5)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        linear.fit(standardized_train, train_labels, sample_weight=train_weights)
    converged = not any(isinstance(item.message, ConvergenceWarning) for item in recorded)
    if not converged:
        print("WARNING: linear probe reached max_iter; inspect linear_converged before drawing conclusions.",
              flush=True)
    prediction = {"linear": linear.predict_proba(standardized_validation).astype(np.float32)}
    if not np.array_equal(linear.classes_, np.arange(8)):
        raise ValueError("Linear probe did not fit all eight classes.")
    training_prediction = linear.predict(standardized_train)
    # Preserve the actual cosine-equivalent Euclidean geometry for normalized
    # projector outputs; standardize only the unnormalized encoder kNN features.
    fit = {
        "linear_mean": scaler.mean_, "linear_scale": scaler.scale_,
        "linear_coef": linear.coef_, "linear_intercept": linear.intercept_,
    }
    return prediction, fit, {
        "linear_converged": converged, "linear_iterations": linear.n_iter_.tolist(),
        "linear_C": 1.0, "linear_max_iter": max_iter,
        "linear_training_raw": classification_metrics(
            train_labels, training_prediction, 8, np.ones(len(train_labels))),
        "linear_training_effective": classification_metrics(
            train_labels, training_prediction, 8, train_weights),
    }, standardized_train[support], standardized_validation


def infer_representations(model, pf, indices, device, batch_size):
    latent_chunks, projector_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            inputs = torch.nan_to_num(pf[chosen].float(), nan=0., posinf=0., neginf=0.).to(device)
            latent, _ = model(inputs)
            latent_chunks.append(latent.cpu().numpy())
            projector_chunks.append(model.get_embeddings(latent).cpu().numpy())
    return {"encoder": np.concatenate(latent_chunks),
            "projector": np.concatenate(projector_chunks)}


def fit_md_reference(train, weights, n_pca):
    if not np.isfinite(train).all():
        raise ValueError("Nonfinite training embedding.")
    mu, transform = _fit_class_transform(
        train, np.ones(len(train), dtype=bool), n_pca,
        "physics-weighted training QCD", weights=weights)
    eigenvalues = np.linalg.eigvalsh(
        ((train - mu) * (weights / weights.sum())[:, None]).T @ (train - mu))
    return mu, transform, eigenvalues


def md_scores(values, mu, transform):
    return np.square((values.astype(np.float64) - mu) @ transform).sum(axis=1)


def closure_on_validation(train_ae, train_md, train_weights, val_ae, val_md,
                          val_weights, percentiles, minimums):
    thresholds_ae = [float(weighted_quantile(train_ae, p, train_weights)) for p in percentiles]
    thresholds_md = [float(weighted_quantile(train_md, p, train_weights)) for p in percentiles]
    grid = np.full((len(percentiles), len(percentiles)), np.nan)
    curve = []
    for i, t1 in enumerate(thresholds_ae):
        for j, t2 in enumerate(thresholds_md):
            regions = abcd_region_statistics_at_thresholds(
                val_ae, val_md, t1, t2, weights=val_weights)
            if not statistically_valid_regions(regions, minimums):
                continue
            A, B, C, D = abcd_yields(regions)
            nc, _ = nonclosure_A(A, B, C, D)
            if not np.isfinite(nc):
                continue
            grid[i, j] = abs(nc)
            if i == j:
                ratio, uncertainty = closure_ratio_and_uncertainty(regions)
                curve.append({"percentile": float(percentiles[i]),
                              "efficiency": float(A / val_weights.sum()),
                              "ratio": float(ratio), "uncertainty": float(uncertainty)})
    valid = grid[np.isfinite(grid)]
    return {
        "grid": grid, "grid_points": int(valid.size),
        "grid_rejected_points": int(grid.size - valid.size),
        "grid_median_absolute_nonclosure": float(np.median(valid)) if valid.size else None,
        "grid_p90_absolute_nonclosure": float(np.percentile(valid, 90)) if valid.size else None,
        "threshold_ae": thresholds_ae, "threshold_md": thresholds_md, "curve": curve,
    }


def score_discrimination(scores, labels, weights):
    reports = {}
    for label in range(8):
        if label == 1:
            continue
        selected = (labels == 1) | (labels == label)
        truth = labels[selected] == label
        reports[str(label)] = {
            "raw_auc": float(roc_auc_score(truth, scores[selected])),
            "physics_weighted_auc": float(roc_auc_score(
                truth, scores[selected], sample_weight=weights[selected])),
        }
    return reports


def plot_confusions(probes, measure, names, output, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for i, space in enumerate(SPACES):
        for j, taxonomy in enumerate(("decay8", "parent4_probability_sum")):
            result = probes[space][measure][taxonomy]
            matrix = np.asarray(result["row_normalized_confusion"])
            ax = axes[i, j]
            ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            labels = names if j == 0 else list(PARENT4_NAMES.values())
            ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
                   xlabel="Predicted", ylabel="True",
                   title=f"{space}: {taxonomy}\nBA={result['balanced_accuracy']:.1%}")
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            for row in range(len(labels)):
                for col in range(len(labels)):
                    ax.text(col, row, f"{matrix[row, col]:.0%}", ha="center", va="center",
                            fontsize=8, color="white" if matrix[row, col] > .55 else "black")
    fig.suptitle(f"{title} — validation, {measure} measure\n"
                 "Parent4 sums decay probabilities; no parent4 retraining")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_md_comparison(score_results, scores, val_ae, labels, weights, names,
                       percentiles, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    classes = [label for label in range(8) if label != 1]
    for ax, metric in zip(axes, ("raw_auc", "physics_weighted_auc")):
        for offset, space in ((-.18, "encoder"), (.18, "projector")):
            values = [score_results[space]["discrimination"][str(c)][metric] for c in classes]
            ax.bar(np.arange(7) + offset, values, .36, label=space)
        ax.set_xticks(np.arange(7), [names[c] for c in classes], rotation=40, ha="right")
        ax.axhline(.5, color="black", ls="--", lw=1)
        ax.set(ylim=(0, 1), ylabel="Class vs QCD ROC AUC", title=metric)
        ax.legend()
    fig.suptitle("Validation MD discrimination — higher score fixed as non-QCD")
    fig.tight_layout()
    fig.savefig(plot_dir / "md_discrimination.png", dpi=170)
    plt.close(fig)

    finite = np.concatenate([score_results[s]["closure"]["grid"].ravel() for s in SPACES])
    finite = finite[np.isfinite(finite)]
    vmax = max(float(np.percentile(finite, 95) * 100), 1.) if finite.size else 100.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for j, space in enumerate(SPACES):
        metrics = score_results[space]["closure"]
        mesh = axes[0, j].pcolormesh(
            percentiles, percentiles, metrics["grid"].T * 100,
            cmap="viridis_r", vmin=0, vmax=vmax, shading="auto")
        fig.colorbar(mesh, ax=axes[0, j], label="|Nonclosure| (%)")
        axes[0, j].set(title=f"{space} QCD MD", xlabel="Training-QCD AE percentile",
                       ylabel="Training-QCD MD percentile")
        curve = metrics["curve"]
        axes[1, j].errorbar([p["efficiency"] for p in curve], [p["ratio"] for p in curve],
                            yerr=[p["uncertainty"] for p in curve], fmt="o-", ms=3)
        axes[1, j].axhline(1, color="black", ls="--")
        axes[1, j].set(xscale="log", xlabel="Validation weighted QCD efficiency in A",
                       ylabel="ABCD predicted / observed")
        axes[1, j].grid(alpha=.2)
    fig.suptitle("Validation-only physics-weighted QCD closure\n"
                 "References AND thresholds fitted on training QCD; no conditional calibration")
    fig.tight_layout()
    fig.savefig(plot_dir / "md_closure_validation.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, space in zip(axes, SPACES):
        positive = np.maximum(scores[space], 1e-12)
        limits = np.quantile(positive, [.001, .999])
        bins = np.geomspace(max(limits[0], 1e-12), max(limits[1], limits[0] * 1.01), 65)
        for label in range(8):
            mask = labels == label
            mass = weights[mask]
            ax.hist(positive[mask], bins=bins, weights=mass / mass.sum(),
                    histtype="step", label=names[label])
        ax.set(xscale="log", title=space, xlabel="Training-QCD-fitted MD",
               ylabel="Per-class physics-weighted fraction / bin")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "md_distributions.png", dpi=170)
    plt.close(fig)

    qcd = labels == 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, space in zip(axes, SPACES):
        x = np.log10(np.maximum(val_ae[qcd], 1e-12))
        y = np.log10(np.maximum(scores[space][qcd], 1e-12))
        hist = ax.hist2d(x, y, bins=70, weights=weights[qcd], norm=LogNorm())
        fig.colorbar(hist[3], ax=ax, label="Physics-weighted QCD yield")
        ax.set(xlabel="log10 AE reconstruction loss", ylabel="log10 QCD MD", title=space)
    fig.suptitle("Identical saved validation QCD events; weighted, not raw counts")
    fig.tight_layout()
    fig.savefig(plot_dir / "md_qcd_ae_plane.png", dpi=170)
    plt.close(fig)


def audit_checkpoint(path, metadata, raw, physics, train_indices, val_indices,
                     train_ae, val_ae, output, args):
    preprocessing = metadata["preprocessing"]
    mapping = parent_mapping(preprocessing)
    labels = checkpoint_target_labels(raw, preprocessing).numpy()
    train_labels, val_labels = labels[train_indices], labels[val_indices]
    if set(np.unique(train_labels)) != set(range(8)) or set(np.unique(val_labels)) != set(range(8)):
        raise ValueError("All eight targets must be present in training AND validation.")
    spec = preprocessing["weighting"]["balance_spec"]
    effective = {}
    for role, indices, ae in (("train", train_indices, train_ae), ("validation", val_indices, val_ae)):
        effective[role] = apply_joint_balance(
            torch.from_numpy(labels[indices]), torch.from_numpy(ae),
            torch.from_numpy(physics[indices]), spec).numpy().astype(np.float64)
    support, inclusion = training_support(train_labels, args.knn_per_class, args.seed)
    vote_weights = effective["train"][support] / inclusion
    vote_weights /= vote_weights.mean()
    measures = {"raw": np.ones(len(val_indices)),
                "training_effective": effective["validation"], "physics": physics[val_indices]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_nurd_model(str(path), device)
    print(f"Inferring frozen training representations: {len(train_indices)} events", flush=True)
    train = infer_representations(model, raw["pf"], train_indices, device, args.batch_size)
    print(f"Inferring frozen validation representations: {len(val_indices)} events", flush=True)
    validation = infer_representations(model, raw["pf"], val_indices, device, args.batch_size)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not all(np.isfinite(v).all() for v in (*train.values(), *validation.values())):
        raise ValueError("Nonfinite frozen representation.")
    output.mkdir()
    plots = output / "plots"
    plots.mkdir()
    fits = {"knn_source_row_index": train_indices[support],
            "knn_support_labels": train_labels[support], "knn_vote_weight": vote_weights}
    predictions_to_save = {}
    probe_reports = {"linear": {}, "knn": {}, "knn_uniform_control": {}}
    probe_metadata, score_results, scores = {}, {}, {}
    train_qcd, val_qcd = train_labels == 1, val_labels == 1
    percentiles = np.linspace(.5, .98, args.grid_points)
    minimums = {"A": args.min_region_ess, "B": args.min_region_ess,
                "C": args.min_region_ess, "D": args.min_D_ess}
    for space in SPACES:
        print(f"Fitting {space} probes on training only", flush=True)
        predictions, fitted, probe_info, support_scaled, val_scaled = fit_linear_probe(
            train[space], validation[space], train_labels, effective["train"],
            support, args.max_iter)
        knn_support = train[space][support] if space == "projector" else support_scaled
        knn_val = validation[space] if space == "projector" else val_scaled
        nn = NearestNeighbors(n_neighbors=min(args.neighbors, len(support)),
                              algorithm="kd_tree", n_jobs=args.jobs).fit(knn_support)
        weighted_chunks, uniform_chunks = [], []
        for start in range(0, len(knn_val), 4096):
            distances, neighbors = nn.kneighbors(knn_val[start:start + 4096])
            weighted_chunks.append(knn_probabilities(
                distances, neighbors, train_labels[support], vote_weights))
            uniform_chunks.append(knn_probabilities(
                distances, neighbors, train_labels[support], np.ones(len(support))))
        predictions["knn"] = np.concatenate(weighted_chunks)
        predictions["knn_uniform_control"] = np.concatenate(uniform_chunks)
        probe_metadata[space] = probe_info
        mean = np.average(train[space], axis=0, weights=effective["train"])
        centered = train[space] - mean
        covariance = (centered * (effective["train"] / effective["train"].sum())[:, None]).T @ centered
        probe_metadata[space]["training_effective_covariance_eigenvalues"] = (
            np.linalg.eigvalsh(covariance).tolist())
        probe_metadata[space]["training_norm_quantiles_0_50_100"] = np.quantile(
            np.linalg.norm(train[space], axis=1), [0, .5, 1]).tolist()
        fits.update({f"{space}_{key}": value for key, value in fitted.items()})
        fits[f"{space}_knn_support"] = knn_support
        for probe, probabilities in predictions.items():
            probe_reports[probe][space] = report_predictions(
                probabilities, val_labels, mapping, measures)
            predictions_to_save[f"{space}_{probe}_probabilities"] = probabilities
        mu, transform, eigenvalues = fit_md_reference(
            train[space][train_qcd], physics[train_indices][train_qcd],
            min(6, train[space].shape[1]))
        scores[space] = md_scores(validation[space], mu, transform)
        train_md = md_scores(train[space][train_qcd], mu, transform)
        closure = closure_on_validation(
            train_ae[train_qcd], train_md, physics[train_indices][train_qcd],
            val_ae[val_qcd], scores[space][val_qcd], physics[val_indices][val_qcd],
            percentiles, minimums)
        score_results[space] = {
            "discrimination": score_discrimination(scores[space], val_labels, physics[val_indices]),
            "closure": closure,
            "reference_covariance_eigenvalues": eigenvalues.tolist(),
            "reference_eigenvalue_floor": 1e-6,
            "reference_eigenvalues_below_floor": int((eigenvalues < 1e-6).sum()),
            "reference_components": int(transform.shape[1]),
        }
        fits[f"{space}_md_mean"], fits[f"{space}_md_transform"] = mu, transform
    names_map = {int(k): v for k, v in preprocessing["target_class_names"].items()}
    names = [names_map[label] for label in range(8)]
    for probe in ("linear", "knn"):
        for measure in ("raw", "training_effective"):
            plot_confusions(probe_reports[probe], measure, names,
                            plots / f"{probe}_confusion_{measure}.png", probe)
    plot_md_comparison(score_results, scores, val_ae, val_labels, physics[val_indices],
                       names, percentiles, plots)
    np.savez_compressed(output / "probe_and_md_fits.npz", **fits)
    saved = {"source_row_index": val_indices, "decay8_label": val_labels,
             "parent4_label": mapping[val_labels], "ae_score": val_ae,
             "physics_weight": physics[val_indices],
             "training_effective_weight": effective["validation"],
             "encoder_latent": validation["encoder"],
             "projector_embedding": validation["projector"],
             "encoder_md": scores["encoder"], "projector_md": scores["projector"],
             **predictions_to_save}
    for field in ("event_id", "background_category"):
        if field in raw:
            saved[field] = raw[field][val_indices].numpy()
    np.savez_compressed(output / "validation_predictions_scores.npz", **saved)
    common = (np.isfinite(score_results["encoder"]["closure"]["grid"])
              & np.isfinite(score_results["projector"]["closure"]["grid"]))
    common_summary = {"points": int(common.sum())}
    for space in SPACES:
        grid = score_results[space]["closure"].pop("grid")
        np.save(output / f"{space}_validation_closure_grid.npy", grid)
        common_summary[f"{space}_median_absolute_nonclosure"] = (
            float(np.median(grid[common])) if common.any() else None)
    result = {
        "checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
        "selected_epoch": int(metadata["epoch"]),
        "training_objective": {key: metadata.get("config", {}).get(key)
                               for key in ("ce_weight", "contrast_weight", "nurd_enabled")},
        "classification_head_trained": metadata.get("classification_head_trained", True),
        "selection_metric": metadata.get("selection_metric"),
        "class_names": names_map, "probe_metadata": probe_metadata,
        "train_class_counts": {str(c): int((train_labels == c).sum()) for c in range(8)},
        "validation_class_counts": {str(c): int((val_labels == c).sum()) for c in range(8)},
        "training_effective_mass": {
            str(c): float(effective["train"][train_labels == c].sum() / effective["train"].sum())
            for c in range(8)},
        "probes": probe_reports, "scores": score_results,
        "common_valid_closure_grid": common_summary,
        "knn_support_counts": {str(c): int((train_labels[support] == c).sum()) for c in range(8)},
        "knn_protocol": {
            "k": args.neighbors, "cap_per_class": args.knn_per_class,
            "support_selection": "fixed_seed_uniform_without_replacement_within_training_class",
            "weighted_votes": "inverse_distance * training_effective_weight / support_inclusion_probability",
            "uniform_control": "same_neighbors_inverse_distance_only",
            "encoder_geometry": "training_effective_weight_fitted_standard_scaler",
            "projector_geometry": "Euclidean_on_normalized_projector_equivalent_to_cosine_ranking",
        },
        "closure_protocol": "physics_weighted_train_QCD_reference_and_thresholds_to_saved_validation_QCD",
        "closure_percentiles": percentiles.tolist(), "minimum_region_ess": minimums,
        "warning": "Validation diagnostic, not a new held-out result. No network updates or signal-sensitivity claim.",
    }
    (output / "representation_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--ae-ckpt", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--gen-weight-path", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    parser.add_argument("--knn-per-class", type=int, default=5000)
    parser.add_argument("--neighbors", type=int, default=25)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-points", type=int, default=48)
    parser.add_argument("--min-region-ess", type=int, default=50)
    parser.add_argument("--min-D-ess", type=int, default=500)
    args = parser.parse_args(argv)
    for key in ("batch_size", "ae_batch_size", "knn_per_class", "neighbors", "max_iter", "jobs"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.grid_points < 2:
        parser.error("grid-points must be at least two")
    if args.min_region_ess < 1 or args.min_D_ess < 1:
        parser.error("Region ESS minimums must be positive")
    if len({path.parent.name for path in args.checkpoints}) != len(args.checkpoints):
        parser.error("Checkpoint experiment names must be unique")
    for path in (*args.checkpoints, args.ae_ckpt, args.data, args.gen_weight_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    raw = load_hlt_sample(args.data)
    physics, weight_metadata = load_generator_weights(
        str(args.gen_weight_path), raw["label"], qcd_label=1, sample=raw)
    physics = physics.numpy().astype(np.float64)
    signature = sample_signature(raw)
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints]
    train_indices, val_indices = checkpoint_reference_indices(checkpoints[0], signature, weight_metadata)
    ae_hash = file_sha256(args.ae_ckpt)
    for checkpoint in checkpoints:
        train, val = checkpoint_reference_indices(checkpoint, signature, weight_metadata)
        if not (np.array_equal(train, train_indices) and np.array_equal(val, val_indices)):
            raise ValueError("Checkpoints do not have identical saved training/validation rows")
        if checkpoint.get("ae_checkpoint_sha256") != ae_hash:
            raise ValueError("Checkpoint was trained using a different AE")
        if any(not torch.equal(torch.as_tensor(checkpoint["ae_scaler"][key]),
                               torch.as_tensor(checkpoints[0]["ae_scaler"][key]))
               for key in ("mu", "std")):
            raise ValueError("Checkpoints do not use an identical AE scaler")
        parent_mapping(checkpoint["preprocessing"])
    print(f"Saved split: train={len(train_indices)}, validation={len(val_indices)}. Held-out unopened.", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_ae(str(args.ae_ckpt), checkpoints[0]["ae_scaler"], device)
    train_ae = _compute_ae_scores(ae, checkpoints[0]["ae_scaler"], raw["obj"], train_indices,
                                 device, args.ae_batch_size)
    val_ae = _compute_ae_scores(ae, checkpoints[0]["ae_scaler"], raw["obj"], val_indices,
                               device, args.ae_batch_size)
    del ae
    if not np.isfinite(train_ae).all() or not np.isfinite(val_ae).all():
        raise ValueError("Nonfinite AE reconstruction loss; refusing to fit probes or MD")
    args.outdir.mkdir(parents=True)
    reports = {}
    with threadpool_limits(limits=args.jobs):
        for path, metadata in zip(args.checkpoints, checkpoints):
            name = path.parent.name.removeprefix("hlt_nurd_engineer_")
            print(f"\n=== {name}, selected epoch {metadata['epoch']} ===", flush=True)
            reports[name] = audit_checkpoint(
                path, metadata, raw, physics, train_indices, val_indices,
                train_ae, val_ae, args.outdir / name, args)
            for space in SPACES:
                for probe in ("linear", "knn", "knn_uniform_control"):
                    values = reports[name]["probes"][probe][space]
                    print(
                        f"{space:9s} {probe:20s}: "
                        f"raw decay8 BA={values['raw']['decay8']['balanced_accuracy']:.2%}; "
                        f"effective decay8 BA={values['training_effective']['decay8']['balanced_accuracy']:.2%}; "
                        f"raw parent4 BA={values['raw']['parent4_probability_sum']['balanced_accuracy']:.2%}",
                        flush=True)
                print(f"{space}: validation closure grid median="
                      f"{reports[name]['scores'][space]['closure']['grid_median_absolute_nonclosure']}",
                      flush=True)
            gc.collect()
    summary = {
        "protocol": "frozen_network_train_fitted_probes_and_QCD_MD_to_saved_validation",
        "heldout_sample_accessed": False, "network_parameters_updated": False,
        "data": str(args.data), "generator_weights": str(args.gen_weight_path),
        "data_signature": signature,
        "physics_weight_sha256": weight_metadata["effective_physics_weight_sha256"],
        "ae_checkpoint": str(args.ae_ckpt), "ae_checkpoint_sha256": ae_hash,
        "train_events": len(train_indices), "validation_events": len(val_indices),
        "seed": args.seed, "runs": reports,
    }
    (args.outdir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["run", "space", "probe", "measure", "taxonomy", "balanced_accuracy", "accuracy"]
    with (args.outdir / "probe_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, report in reports.items():
            for probe, spaces in report["probes"].items():
                for space, measures in spaces.items():
                    for measure, taxonomies in measures.items():
                        for taxonomy, metrics in taxonomies.items():
                            writer.writerow(dict(zip(fields, [name, space, probe, measure, taxonomy,
                                                              metrics["balanced_accuracy"], metrics["accuracy"]])))
    print(f"FROZEN REPRESENTATION AUDIT DONE: {args.outdir / 'audit_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
