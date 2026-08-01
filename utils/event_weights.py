"""Validated event-weight loading and weighted statistics."""

from pathlib import Path

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
    }


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
