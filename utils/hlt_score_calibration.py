"""Frozen class-conditional latent references and deployable HLT anomaly scores."""

from dataclasses import dataclass
import json

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class ClassReference:
    label: int
    mean: np.ndarray
    whitening: np.ndarray
    eigenvalues: np.ndarray
    logdet: float
    prior: float
    calibration_md: np.ndarray
    calibration_weight: np.ndarray


def _softmax(values, axis=1):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=axis, keepdims=True), 1e-12)


def fit_class_references(latents, labels, fit_indices, calibration_indices,
                         class_labels, n_components=None, min_events=20,
                         sample_weights=None, shrinkage=0.05):
    """Fit shrinkage-covariance references on fit events and calibrate on disjoint events."""
    latents = np.asarray(latents, dtype=np.float64)
    labels = np.asarray(labels)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    calibration_indices = np.asarray(calibration_indices, dtype=np.int64)
    class_labels = [int(label) for label in class_labels]
    if sample_weights is None:
        sample_weights = np.ones(latents.shape[0], dtype=np.float64)
        use_weighted_covariance = False
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if sample_weights.shape[0] != latents.shape[0]:
            raise ValueError("sample_weights must align with latents.")
        use_weighted_covariance = True
    fit_total = sum(
        sample_weights[fit_indices][labels[fit_indices] == label].sum()
        for label in class_labels)
    references = []

    for label in class_labels:
        fit_mask = labels[fit_indices] == label
        calibration_mask = labels[calibration_indices] == label
        fit_values = latents[fit_indices[fit_mask]]
        calibration_values = latents[calibration_indices[calibration_mask]]
        fit_weights = sample_weights[fit_indices[fit_mask]]
        calibration_weights = sample_weights[
            calibration_indices[calibration_mask]]
        if fit_values.shape[0] < min_events or calibration_values.shape[0] < min_events:
            raise ValueError(
                f"Class {label} needs at least {min_events} fit and calibration events; "
                f"found {fit_values.shape[0]} and {calibration_values.shape[0]}."
            )

        if use_weighted_covariance:
            total_weight = max(float(fit_weights.sum()), 1e-12)
            mean = np.sum(
                fit_values * fit_weights[:, None], axis=0) / total_weight
            centered = fit_values - mean
            covariance_matrix = (
                (centered * fit_weights[:, None]).T @ centered / total_weight
            )
            covariance_matrix = 0.5 * (
                covariance_matrix + covariance_matrix.T)
            diagonal_mean = max(
                float(np.trace(covariance_matrix) / covariance_matrix.shape[0]),
                1e-8)
            covariance_matrix = (
                (1.0 - float(shrinkage)) * covariance_matrix
                + float(shrinkage) * np.eye(covariance_matrix.shape[0])
                * diagonal_mean
            )
        else:
            covariance = LedoitWolf(assume_centered=False).fit(fit_values)
            mean = covariance.location_.astype(np.float64)
            covariance_matrix = covariance.covariance_
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        order = np.argsort(eigenvalues)[::-1]
        if n_components is not None:
            order = order[:min(int(n_components), fit_values.shape[1])]
        eigenvalues = np.clip(eigenvalues[order], 1e-8, None)
        eigenvectors = eigenvectors[:, order]
        whitening = eigenvectors / np.sqrt(eigenvalues)
        calibrated = (calibration_values - mean) @ whitening
        calibration_md = np.sum(calibrated * calibrated, axis=1)
        calibration_order = np.argsort(calibration_md)
        calibration_md = calibration_md[calibration_order]
        calibration_weight = calibration_weights[calibration_order]
        prior = fit_weights.sum() / max(fit_total, 1e-12)
        references.append(ClassReference(
            label=label,
            mean=mean,
            whitening=whitening,
            eigenvalues=eigenvalues,
            logdet=float(np.log(eigenvalues).sum()),
            prior=float(prior),
            calibration_md=calibration_md,
            calibration_weight=calibration_weight,
        ))
    return references


def save_class_references(path, references, metadata=None):
    payload = {
        "labels": np.asarray([ref.label for ref in references], dtype=np.int64),
        "metadata": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    for ref in references:
        prefix = f"class_{ref.label}"
        payload[f"{prefix}_mean"] = ref.mean
        payload[f"{prefix}_whitening"] = ref.whitening
        payload[f"{prefix}_eigenvalues"] = ref.eigenvalues
        payload[f"{prefix}_logdet"] = np.asarray(ref.logdet)
        payload[f"{prefix}_prior"] = np.asarray(ref.prior)
        payload[f"{prefix}_calibration_md"] = ref.calibration_md
        payload[f"{prefix}_calibration_weight"] = ref.calibration_weight
    np.savez_compressed(path, **payload)


def load_class_references(path):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"].item()))
    references = []
    for label in data["labels"].astype(int).tolist():
        prefix = f"class_{label}"
        references.append(ClassReference(
            label=label,
            mean=data[f"{prefix}_mean"],
            whitening=data[f"{prefix}_whitening"],
            eigenvalues=data[f"{prefix}_eigenvalues"],
            logdet=float(data[f"{prefix}_logdet"]),
            prior=float(data[f"{prefix}_prior"]),
            calibration_md=data[f"{prefix}_calibration_md"],
            calibration_weight=(
                data[f"{prefix}_calibration_weight"]
                if f"{prefix}_calibration_weight" in data.files
                else np.ones_like(data[f"{prefix}_calibration_md"])
            ),
        ))
    return references, metadata


def validate_reference_metadata(actual, expected):
    return all(actual.get(key) == value for key, value in expected.items())


def score_latents(latents, logits, references, score_mode="calibrated_union",
                  qcd_label=1):
    """Return the selected anomaly axis and all class-resolved score products."""
    latents = np.asarray(latents, dtype=np.float64)
    labels = np.asarray([ref.label for ref in references], dtype=np.int64)
    md_columns = []
    tail_columns = []
    nll_columns = []

    for ref in references:
        whitened = (latents - ref.mean) @ ref.whitening
        md = np.sum(whitened * whitened, axis=1)
        position = np.searchsorted(ref.calibration_md, md, side="left")
        calibration_weight = getattr(
            ref, "calibration_weight", np.ones_like(ref.calibration_md))
        cumulative = np.concatenate([
            np.zeros(1, dtype=np.float64),
            np.cumsum(calibration_weight, dtype=np.float64),
        ])
        n_greater_equal = cumulative[-1] - cumulative[position]
        pseudo_weight = cumulative[-1] / max(ref.calibration_md.size, 1)
        tail_probability = (
            n_greater_equal + pseudo_weight
        ) / max(cumulative[-1] + pseudo_weight, 1e-12)
        dimension = ref.whitening.shape[1]
        nll = (
            0.5 * (md + ref.logdet + dimension * np.log(2.0 * np.pi))
            - np.log(max(ref.prior, 1e-12))
        )
        md_columns.append(md)
        tail_columns.append(tail_probability)
        nll_columns.append(nll)

    md_matrix = np.stack(md_columns, axis=1)
    tail_probability = np.stack(tail_columns, axis=1)
    gaussian_nll = np.stack(nll_columns, axis=1)
    gaussian_posterior = _softmax(-gaussian_nll)
    classifier_probability = (
        _softmax(np.asarray(logits, dtype=np.float64)[:, labels])
        if logits is not None else np.full_like(gaussian_posterior, np.nan)
    )
    classifier_route = (
        np.argmax(classifier_probability, axis=1)
        if logits is not None else np.argmin(gaussian_nll, axis=1)
    )
    gaussian_route = np.argmin(gaussian_nll, axis=1)
    typicality_route = np.argmax(tail_probability, axis=1)
    row = np.arange(latents.shape[0])

    max_tail = np.max(tail_probability, axis=1)
    calibrated_union = -np.log(np.clip(max_tail, 1e-12, 1.0))
    maximum_log_component = np.max(-gaussian_nll, axis=1)
    mixture_nll = -(
        maximum_log_component
        + np.log(np.exp(-gaussian_nll - maximum_log_component[:, None]).sum(axis=1))
    )
    qcd_matches = np.flatnonzero(labels == int(qcd_label))
    if qcd_matches.size == 0:
        raise ValueError(f"QCD label {qcd_label} is absent from class references {labels.tolist()}.")
    qcd_column = int(qcd_matches[0])

    score_map = {
        "qcd_md": md_matrix[:, qcd_column],
        "min_md": np.min(md_matrix, axis=1),
        "calibrated_union": calibrated_union,
        "mixture_nll": mixture_nll,
        "classifier_routed": -np.log(np.clip(
            tail_probability[row, classifier_route], 1e-12, 1.0)),
        "gaussian_routed": -np.log(np.clip(
            tail_probability[row, gaussian_route], 1e-12, 1.0)),
    }
    if score_mode not in score_map:
        raise ValueError(
            f"Unsupported score_mode={score_mode!r}; choose from {sorted(score_map)}."
        )

    products = {
        "reference_labels": labels,
        "md_per_class": md_matrix.astype(np.float32),
        "tail_probability_per_class": tail_probability.astype(np.float32),
        "gaussian_nll_per_class": gaussian_nll.astype(np.float32),
        "gaussian_posterior_per_class": gaussian_posterior.astype(np.float32),
        "classifier_probability_per_class": classifier_probability.astype(np.float32),
        "classifier_route_index": classifier_route.astype(np.int16),
        "gaussian_route_index": gaussian_route.astype(np.int16),
        "typicality_route_index": typicality_route.astype(np.int16),
        "calibrated_union": calibrated_union.astype(np.float32),
        "mixture_nll": mixture_nll.astype(np.float32),
    }
    return score_map[score_mode].astype(np.float32), products
