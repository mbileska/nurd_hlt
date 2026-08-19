"""Validated event-weight loading and weighted statistics."""

import hashlib
import math
from pathlib import Path
from fractions import Fraction

import numpy as np
import torch


WEIGHT_KEYS = (
    "gen_weight",
    "gen_weights",
    "event_weight",
    "event_weights",
    "weight",
    "weights",
)
EVENT_ID_KEYS = ("eventid", "event_id", "eventids", "event_ids")


def tensor_sha256(value):
    """Stable content digest for a tensor/array after moving it to CPU."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    """Return the SHA256 of a file without loading the whole file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sample_signature(sample):
    """Small deterministic signature used to bind checkpoints to event rows."""
    if "label" not in sample:
        raise ValueError("Sample is missing the required 'label' tensor.")
    signature = {
        "n_events": int(torch.as_tensor(sample["label"]).shape[0]),
        "label_sha256": tensor_sha256(sample["label"]),
        "shapes": {
            key: list(torch.as_tensor(sample[key]).shape)
            for key in ("pf", "obj", "label") if key in sample
        },
    }
    event_ids, event_id_key = _find_value(sample, EVENT_ID_KEYS)
    if event_ids is not None:
        signature["event_id_key"] = event_id_key
        signature["event_id_sha256"] = tensor_sha256(event_ids)
    return signature


def _as_1d_float_tensor(value, name):
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 2 and 1 in tensor.shape:
        tensor = tensor.reshape(-1)
    elif tensor.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional (or have one singleton axis); "
            f"got shape {tuple(tensor.shape)}."
        )
    return tensor.float()


def _find_value(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key], key
    return None, None


def _event_ids_equal(left, right):
    left = torch.as_tensor(left).detach().cpu()
    right = torch.as_tensor(right).detach().cpu()
    return left.shape == right.shape and torch.equal(left, right)


def load_event_weights(path, sample, max_events=-1, require_nonnegative=True):
    """Load event-aligned generator weights.

    Supported files are a tensor/array, a dictionary with one of
    ``WEIGHT_KEYS``, or a one-entry dictionary containing an aligned tensor.
    If both files carry event IDs, exact alignment is required.
    """
    n_full = int(sample["label"].shape[0])
    n_used = n_full if max_events is None or int(max_events) <= 0 else min(
        n_full, int(max_events))

    if not path:
        weights = torch.ones(n_used, dtype=torch.float32)
        return weights, {
            "path": None,
            "source": "unit",
            "n_events": n_used,
            "sum_weights": float(n_used),
            "effective_events": float(n_used),
            "alignment_verified": True,
            "sha256": tensor_sha256(weights),
        }

    payload = torch.load(Path(path), map_location="cpu")
    source = type(payload).__name__
    payload_ids = None
    if isinstance(payload, dict):
        value, key = _find_value(payload, WEIGHT_KEYS)
        payload_ids, _ = _find_value(payload, EVENT_ID_KEYS)
        if value is None:
            tensor_values = [
                (name, value) for name, value in payload.items()
                if torch.is_tensor(value) or isinstance(value, np.ndarray)
            ]
            if len(tensor_values) != 1:
                raise ValueError(
                    f"Could not identify generator weights in {path}. Expected "
                    f"one of {WEIGHT_KEYS} or one tensor-like dictionary value."
                )
            key, value = tensor_values[0]
        source = f"dict:{key}"
    else:
        value = payload

    weights = _as_1d_float_tensor(value, f"generator weights in {path}")
    if weights.numel() != n_full:
        raise ValueError(
            f"Generator-weight length mismatch for {path}: got "
            f"{weights.numel()}, expected {n_full} events from the sample."
        )

    sample_ids, _ = _find_value(sample, EVENT_ID_KEYS)
    if payload_ids is not None:
        if sample_ids is None:
            raise ValueError(
                f"{path} contains event IDs but the sample does not; alignment "
                "cannot be verified."
            )
        if not _event_ids_equal(payload_ids, sample_ids):
            raise ValueError(
                f"Event IDs in {path} do not exactly match the sample ordering."
            )

    if not torch.isfinite(weights).all():
        raise ValueError(f"Generator weights in {path} contain NaN or infinity.")
    if require_nonnegative and (weights < 0).any():
        n_negative = int((weights < 0).sum().item())
        raise ValueError(
            f"Generator weights in {path} contain {n_negative} negative values. "
            "Signed weights need a dedicated statistical treatment and cannot "
            "be used safely by the current positive weighted losses."
        )
    if float(weights.sum().item()) <= 0.0:
        raise ValueError(f"Generator weights in {path} have non-positive total weight.")

    weights = weights[:n_used].contiguous()
    total = float(weights.sum().item())
    sumsq = float((weights.double() ** 2).sum().item())
    effective = total * total / max(sumsq, 1e-30)
    return weights, {
        "path": str(Path(path)),
        "source": source,
        "n_events": int(weights.numel()),
        "sum_weights": total,
        "min": float(weights.min().item()),
        "max": float(weights.max().item()),
        "mean": float(weights.mean().item()),
        "effective_events": float(effective),
        "alignment_verified": bool(payload_ids is not None),
        "sha256": tensor_sha256(weights),
    }


def weighted_balanced_folds(weights, n_splits=2, seed=42):
    """Assign rows to folds while balancing total generator weight."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        return np.empty(0, dtype=np.int16)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Fold weights must be finite and non-negative.")
    n_splits = min(max(int(n_splits), 2), weights.size)
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(weights.size)
    order = shuffled[np.argsort(-weights[shuffled], kind="stable")]
    fold_mass = np.zeros(n_splits, dtype=np.float64)
    fold_size = np.zeros(n_splits, dtype=np.int64)
    fold_ids = np.empty(weights.size, dtype=np.int16)
    for index in order:
        fold = min(range(n_splits), key=lambda value: (
            fold_mass[value], fold_size[value], value))
        fold_ids[index] = fold
        fold_mass[fold] += weights[index]
        fold_size[fold] += 1
    return fold_ids


def weighted_stratified_split(labels, weights, fractions, seed=42,
                              max_folds=100):
    """Split every class into generator-mass-balanced deterministic folds.

    Fractions must form a simple rational partition, e.g. ``(0.9, .05, .05)``.
    Rows are first balanced by physical generator mass within each class and
    then whole folds are assigned to the requested partitions. This prevents a
    few high-weight events from dominating one validation role while retaining
    disjoint event sets.
    """
    labels = np.asarray(torch.as_tensor(labels).detach().cpu()).reshape(-1)
    weights = np.asarray(
        torch.as_tensor(weights).detach().cpu(), dtype=np.float64).reshape(-1)
    fractions = [float(value) for value in fractions]
    if labels.shape[0] != weights.shape[0]:
        raise ValueError("labels and weights must have the same length.")
    if not fractions or any(value <= 0.0 for value in fractions):
        raise ValueError("split fractions must all be positive.")
    if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("split fractions must sum to one.")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("split weights must be finite and non-negative.")

    rational = [Fraction(str(value)).limit_denominator(max_folds)
                for value in fractions]
    n_folds = 1
    for value in rational:
        n_folds = math.lcm(n_folds, value.denominator)
    if n_folds > int(max_folds):
        raise ValueError(
            f"split fractions require {n_folds} folds; maximum is {max_folds}.")
    fold_counts = [int(value * n_folds) for value in rational]
    if sum(fold_counts) != n_folds or any(value < 1 for value in fold_counts):
        raise ValueError(
            "split fractions cannot be represented as non-empty whole folds.")

    split_parts = [[] for _ in fractions]
    split_ranges = []
    start = 0
    for count in fold_counts:
        split_ranges.append(range(start, start + count))
        start += count

    for label_index, label in enumerate(np.unique(labels)):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.size < n_folds:
            raise ValueError(
                f"Class {label} has {class_indices.size} rows but {n_folds} "
                "are required for the requested physical split.")
        local_folds = weighted_balanced_folds(
            weights[class_indices], n_splits=n_folds,
            seed=int(seed) + label_index * 1009)
        for split_index, fold_range in enumerate(split_ranges):
            mask = np.isin(local_folds, np.fromiter(
                fold_range, dtype=np.int16))
            split_parts[split_index].append(class_indices[mask])

    rng = np.random.default_rng(int(seed))
    result = []
    for parts in split_parts:
        values = np.concatenate(parts).astype(np.int64, copy=False)
        result.append(rng.permutation(values))
    combined = np.concatenate(result)
    if combined.size != labels.size or np.unique(combined).size != labels.size:
        raise RuntimeError("weighted stratified split lost or duplicated rows.")
    return result


def split_diagnostics(labels, weights, named_indices):
    """JSON-serializable row, sumw, and ESS diagnostics for data partitions."""
    labels = np.asarray(torch.as_tensor(labels).detach().cpu()).reshape(-1)
    weights = np.asarray(
        torch.as_tensor(weights).detach().cpu(), dtype=np.float64).reshape(-1)
    result = {}
    for name, indices in named_indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        subset = weights[indices]
        sumw = float(subset.sum())
        sumw2 = float(np.square(subset).sum())
        details = {
            "rows": int(indices.size),
            "sum_weights": sumw,
            "effective_events": sumw * sumw / max(sumw2, 1e-30),
            "per_class": {},
        }
        for label in np.unique(labels):
            class_weights = weights[indices[labels[indices] == label]]
            class_sumw = float(class_weights.sum())
            class_sumw2 = float(np.square(class_weights).sum())
            details["per_class"][str(int(label))] = {
                "rows": int(class_weights.size),
                "sum_weights": class_sumw,
                "effective_events": (
                    class_sumw * class_sumw / max(class_sumw2, 1e-30)),
            }
        result[str(name)] = details
    return result


def validate_shared_data_contract(checkpoint, provenance, split_indices,
                                  code_commit):
    """Require an AE checkpoint to match the NURD data, weights, and roles."""
    if checkpoint.get("data_signature") != provenance.get("sample"):
        raise ValueError("AE checkpoint data signature does not match --data.")
    expected_weights = checkpoint.get(
        "gen_weight_metadata", {}).get("sha256")
    actual_weights = provenance.get("generator_weights", {}).get("sha256")
    if not expected_weights or expected_weights != actual_weights:
        raise ValueError(
            "AE checkpoint generator weights do not match --gen_weights.")
    if not code_commit or checkpoint.get("code_commit") != code_commit:
        raise ValueError("AE checkpoint code commit does not match this NURD run.")
    expected_splits = checkpoint.get("data_split_indices")
    if not isinstance(expected_splits, dict):
        raise ValueError(
            "AE checkpoint lacks the shared 90/5/5 data split contract.")
    if set(expected_splits) != set(split_indices):
        raise ValueError("AE/NURD data split roles do not match.")
    for role, actual in split_indices.items():
        expected = torch.as_tensor(expected_splits[role]).long().cpu()
        actual = torch.as_tensor(actual).long().cpu()
        if not torch.equal(expected, actual):
            raise ValueError(
                f"AE/NURD data split mismatch for role {role!r}.")


def weighted_mean_and_std(values, weights, dim=0, eps=1e-12):
    values = torch.as_tensor(values).float()
    weights = torch.as_tensor(weights, device=values.device).float().reshape(-1)
    if values.shape[dim] != weights.numel():
        raise ValueError("Weight count does not match the requested value dimension.")
    shape = [1] * values.ndim
    shape[dim] = weights.numel()
    expanded = weights.reshape(shape)
    total = expanded.sum().clamp(min=eps)
    mean = (values * expanded).sum(dim=dim) / total
    centered = values - mean.unsqueeze(dim)
    variance = (centered.square() * expanded).sum(dim=dim) / total
    return mean, variance.clamp(min=0.0).sqrt()


def weighted_quantile(values, quantiles, weights=None):
    """Weighted order-statistic quantiles on the input tensor's device."""
    values = torch.as_tensor(values).reshape(-1)
    quantiles = torch.as_tensor(
        quantiles, device=values.device, dtype=values.dtype).reshape(-1)
    if weights is None:
        return torch.quantile(values, quantiles)
    weights = torch.as_tensor(
        weights, device=values.device, dtype=values.dtype).reshape(-1)
    if values.numel() != weights.numel():
        raise ValueError("Values and weights must have the same length.")
    valid = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError("Weighted quantiles require at least one positive finite weight.")
    values = values[valid]
    weights = weights[valid]
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = torch.cumsum(sorted_weights, dim=0)
    cdf = (cdf - 0.5 * sorted_weights) / sorted_weights.sum()
    cdf = torch.cat([
        torch.zeros(1, device=cdf.device, dtype=cdf.dtype),
        cdf,
        torch.ones(1, device=cdf.device, dtype=cdf.dtype),
    ])
    padded_values = torch.cat([
        sorted_values[:1], sorted_values, sorted_values[-1:],
    ])
    q = quantiles.clamp(0.0, 1.0)
    upper = torch.searchsorted(cdf, q, right=False).clamp(1, cdf.numel() - 1)
    lower = upper - 1
    fraction = (q - cdf[lower]) / (cdf[upper] - cdf[lower]).clamp(min=1e-12)
    return padded_values[lower] + fraction * (
        padded_values[upper] - padded_values[lower])


def weighted_mean_numpy(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values * weights) / max(np.sum(weights), 1e-30))


def weighted_quantile_numpy(values, quantiles, weights=None):
    values_t = torch.as_tensor(np.asarray(values), dtype=torch.float64)
    weights_t = None if weights is None else torch.as_tensor(
        np.asarray(weights), dtype=torch.float64)
    return weighted_quantile(values_t, quantiles, weights_t).cpu().numpy()
