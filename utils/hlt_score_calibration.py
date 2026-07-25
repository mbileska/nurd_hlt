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


def _softmax(values, axis=1):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=axis, keepdims=True), 1e-12)


def fit_class_references(latents, labels, fit_indices, calibration_indices,
                         class_labels, n_components=None, min_events=20):
    """Fit shrinkage-covariance references on fit events and calibrate on disjoint events."""
    latents = np.asarray(latents, dtype=np.float64)
    labels = np.asarray(labels)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    calibration_indices = np.asarray(calibration_indices, dtype=np.int64)
    class_labels = [int(label) for label in class_labels]
    fit_total = sum(np.count_nonzero(labels[fit_indices] == label) for label in class_labels)
    references = []

    for label in class_labels:
        fit_mask = labels[fit_indices] == label
        calibration_mask = labels[calibration_indices] == label
        fit_values = latents[fit_indices[fit_mask]]
        calibration_values = latents[calibration_indices[calibration_mask]]
        if fit_values.shape[0] < min_events or calibration_values.shape[0] < min_events:
            raise ValueError(
                f"Class {label} needs at least {min_events} fit and calibration events; "
                f"found {fit_values.shape[0]} and {calibration_values.shape[0]}."
            )

        covariance = LedoitWolf(assume_centered=False).fit(fit_values)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.covariance_)
        order = np.argsort(eigenvalues)[::-1]
        if n_components is not None:
            order = order[:min(int(n_components), fit_values.shape[1])]
        eigenvalues = np.clip(eigenvalues[order], 1e-8, None)
        eigenvectors = eigenvectors[:, order]
        whitening = eigenvectors / np.sqrt(eigenvalues)
        mean = covariance.location_.astype(np.float64)

        calibrated = (calibration_values - mean) @ whitening
        calibration_md = np.sort(np.sum(calibrated * calibrated, axis=1))
        prior = fit_values.shape[0] / max(fit_total, 1)
        references.append(ClassReference(
            label=label,
            mean=mean,
            whitening=whitening,
            eigenvalues=eigenvalues,
            logdet=float(np.log(eigenvalues).sum()),
            prior=float(prior),
            calibration_md=calibration_md,
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
        n_greater_equal = ref.calibration_md.size - position
        tail_probability = (n_greater_equal + 1.0) / (ref.calibration_md.size + 1.0)
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
