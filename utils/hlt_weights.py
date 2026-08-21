"""Weighting and split utilities for the HLT AE/NURD workflow.

The physics generator weight defines the within-cell event measure.  Training
then targets equal total mass for every background class and for every occupied
continuous-nuisance stratum inside that class.  The strata are used only to
estimate weights; the critic always receives the continuous nuisance value.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split


WEIGHT_KEYS = (
    "weights", "weight", "gen_weights", "gen_weight",
    "event_weights", "event_weight",
)
EVENT_ID_KEYS = ("eventid", "event_id", "eventids", "event_ids")


def tensor_sha256(value) -> str:
    """Stable checksum for a tensor/array after moving it to CPU."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _find_mapping_value(mapping: Mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key], key
    return None, None


def sample_signature(sample: Mapping, max_events: int = -1) -> Dict[str, object]:
    """Compact signature binding an AE checkpoint to the event ordering.

    Labels are hashed in full. PF/object content is hashed on a deterministic
    set of up to 1024 rows, which makes accidental reuse with another sample
    fail without hashing several gigabytes on every launch.
    """
    if "label" not in sample:
        raise ValueError("Sample is missing the required label tensor.")
    n_full = int(torch.as_tensor(sample["label"]).shape[0])
    n_events = n_full if int(max_events) <= 0 else min(n_full, int(max_events))
    labels = torch.as_tensor(sample["label"])[:n_events]
    signature: Dict[str, object] = {
        "n_events": n_events,
        "label_sha256": tensor_sha256(labels),
        "shapes": {
            key: list(torch.as_tensor(sample[key])[:n_events].shape)
            for key in ("pf", "obj", "label") if key in sample
        },
    }
    if n_events:
        sample_count = min(n_events, 1024)
        row_indices = torch.linspace(
            0, n_events - 1, sample_count, dtype=torch.float64
        ).round().long().unique()
        for key in ("pf", "obj"):
            if key in sample:
                signature[f"{key}_sample_sha256"] = tensor_sha256(
                    torch.as_tensor(sample[key])[row_indices])
    event_ids, event_id_key = _find_mapping_value(sample, EVENT_ID_KEYS)
    if event_ids is not None:
        signature["event_id_key"] = event_id_key
        signature["event_id_sha256"] = tensor_sha256(
            torch.as_tensor(event_ids)[:n_events])
    return signature


def _as_float_vector(value) -> torch.Tensor:
    if isinstance(value, Mapping):
        for key in WEIGHT_KEYS:
            if key in value:
                value = value[key]
                break
        else:
            tensor_values = [
                item for key, item in value.items()
                if key not in EVENT_ID_KEYS
                and (torch.is_tensor(item) or isinstance(item, np.ndarray))
            ]
            if len(tensor_values) != 1:
                raise ValueError(
                    f"Weight mapping has none of the supported keys {WEIGHT_KEYS} "
                    "and does not contain exactly one unambiguous tensor.")
            value = tensor_values[0]
    return torch.as_tensor(value, dtype=torch.float64).detach().cpu().reshape(-1)


def load_generator_weights(
    path: Optional[str],
    labels: torch.Tensor,
    qcd_label: int = 1,
    max_events: int = -1,
    sample: Optional[Mapping] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Load and validate per-event physics weights.

    MEquiNNa weight files describe the QCD component.  Non-QCD events are
    intentionally assigned unit physics weight, irrespective of any unused
    values in the file.
    """
    labels = torch.as_tensor(labels).long().reshape(-1).cpu()
    n_events = labels.numel()
    if path is None:
        weights = torch.ones(n_events, dtype=torch.float64)
        source = "unit"
        raw_checksum = tensor_sha256(weights)
        alignment_verified = True
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_ids = None
        if isinstance(payload, Mapping):
            payload_ids, _ = _find_mapping_value(payload, EVENT_ID_KEYS)
        loaded_full = _as_float_vector(payload)
        expected_full = (
            int(torch.as_tensor(sample["label"]).shape[0])
            if sample is not None else n_events
        )
        if loaded_full.numel() != expected_full:
            raise ValueError(
                f"Generator-weight length {loaded_full.numel()} does not match "
                f"the data length {expected_full}.")
        loaded = loaded_full
        if max_events > 0:
            loaded = loaded[:max_events]
        if loaded.numel() != n_events:
            raise ValueError(
                f"Generator-weight length {loaded.numel()} does not match "
                f"the selected data length {n_events}.")
        if payload_ids is not None:
            sample_ids, _ = _find_mapping_value(sample or {}, EVENT_ID_KEYS)
            if sample_ids is None:
                raise ValueError(
                    "The weight file contains event IDs but the data file does not; "
                    "row alignment cannot be verified.")
            payload_ids = torch.as_tensor(payload_ids)
            sample_ids = torch.as_tensor(sample_ids)
            if max_events > 0:
                payload_ids = payload_ids[:max_events]
                sample_ids = sample_ids[:max_events]
            if payload_ids.shape != sample_ids.shape or not torch.equal(
                    payload_ids.cpu(), sample_ids.cpu()):
                raise ValueError(
                    "Generator-weight event IDs do not match the data ordering.")
            alignment_verified = True
        else:
            alignment_verified = False
        raw_checksum = tensor_sha256(loaded)
        weights = torch.ones(n_events, dtype=torch.float64)
        qcd_mask = labels == int(qcd_label)
        weights[qcd_mask] = loaded[qcd_mask]
        source = str(Path(path))

    if not torch.isfinite(weights).all():
        raise ValueError("Generator weights contain NaN or infinity.")
    if (weights < 0).any():
        raise ValueError("Generator weights must be non-negative.")
    if float(weights.sum()) <= 0.0:
        raise ValueError("Generator weights have non-positive total mass.")

    qcd_mask = labels == int(qcd_label)
    qcd = weights[qcd_mask]
    metadata = {
        "source": source,
        "n_events": int(n_events),
        "qcd_label": int(qcd_label),
        "qcd_events": int(qcd_mask.sum()),
        "qcd_sum": float(qcd.sum()) if qcd.numel() else 0.0,
        "qcd_min": float(qcd.min()) if qcd.numel() else None,
        "qcd_max": float(qcd.max()) if qcd.numel() else None,
        "raw_weight_sha256": raw_checksum,
        "effective_physics_weight_sha256": tensor_sha256(weights),
        "alignment_verified": alignment_verified,
        "alignment": (
            "event_ids" if alignment_verified and path is not None
            else "exact_length_only_no_event_ids"
        ),
    }
    return weights, metadata


def stratified_split_indices(
    labels: torch.Tensor,
    val_fraction: float,
    seed: int,
    weights: Optional[torch.Tensor] = None,
    max_folds: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split each class while balancing generator mass across folds."""
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must lie strictly between zero and one.")
    labels_np = torch.as_tensor(labels).long().cpu().numpy()
    if weights is None:
        weights_np = np.ones(labels_np.shape[0], dtype=np.float64)
    else:
        weights_np = np.asarray(
            torch.as_tensor(weights).double().cpu(), dtype=np.float64).reshape(-1)
    if weights_np.shape[0] != labels_np.shape[0]:
        raise ValueError("Split labels and weights must align.")
    if not np.isfinite(weights_np).all() or np.any(weights_np < 0.0):
        raise ValueError("Split weights must be finite and non-negative.")

    val_ratio = Fraction(str(float(val_fraction))).limit_denominator(max_folds)
    train_ratio = Fraction(1, 1) - val_ratio
    n_folds = math.lcm(train_ratio.denominator, val_ratio.denominator)
    train_folds = int(train_ratio * n_folds)
    val_folds = int(val_ratio * n_folds)
    if (n_folds > int(max_folds) or train_folds < 1 or val_folds < 1
            or train_folds + val_folds != n_folds):
        raise ValueError(
            "val_fraction cannot be represented by non-empty balanced folds.")

    split_parts = [[], []]
    rng = np.random.default_rng(int(seed))
    for label_index, label in enumerate(np.unique(labels_np)):
        class_indices = np.flatnonzero(labels_np == label)
        if class_indices.size < n_folds:
            # Retain a standard stratified split for tiny smoke/unit samples.
            indices = np.arange(labels_np.shape[0])
            train_idx, val_idx = train_test_split(
                indices, test_size=float(val_fraction), random_state=int(seed),
                stratify=labels_np)
            return (torch.as_tensor(train_idx, dtype=torch.long),
                    torch.as_tensor(val_idx, dtype=torch.long))
        class_weights = weights_np[class_indices]
        shuffled = rng.permutation(class_indices.size)
        order = shuffled[np.argsort(-class_weights[shuffled], kind="stable")]
        fold_mass = np.zeros(n_folds, dtype=np.float64)
        fold_size = np.zeros(n_folds, dtype=np.int64)
        fold_ids = np.empty(class_indices.size, dtype=np.int16)
        for local_index in order:
            fold = min(range(n_folds), key=lambda value: (
                fold_mass[value], fold_size[value], value))
            fold_ids[local_index] = fold
            fold_mass[fold] += class_weights[local_index]
            fold_size[fold] += 1
        split_parts[0].append(class_indices[fold_ids < train_folds])
        split_parts[1].append(class_indices[fold_ids >= train_folds])

    train_idx = rng.permutation(np.concatenate(split_parts[0]))
    val_idx = rng.permutation(np.concatenate(split_parts[1]))
    combined = np.concatenate([train_idx, val_idx])
    if combined.size != labels_np.size or np.unique(combined).size != labels_np.size:
        raise RuntimeError("Weighted stratified split lost or duplicated rows.")
    return (torch.as_tensor(train_idx, dtype=torch.long),
            torch.as_tensor(val_idx, dtype=torch.long))


def fit_class_balance_factors(
    labels: torch.Tensor, base_weights: torch.Tensor
) -> Dict[int, float]:
    """Fit factors giving every class equal total effective mass."""
    labels = torch.as_tensor(labels).long().reshape(-1)
    base_weights = torch.as_tensor(base_weights).double().reshape(-1)
    factors: Dict[int, float] = {}
    classes = sorted(int(value) for value in labels.unique().tolist())
    for label in classes:
        mass = float(base_weights[labels == label].sum())
        if mass <= 0.0:
            raise ValueError(f"Class {label} has non-positive physics mass.")
        factors[label] = 1.0 / (len(classes) * mass)
    return factors


def apply_class_balance_factors(
    labels: torch.Tensor,
    base_weights: torch.Tensor,
    factors: Mapping[int, float],
    normalize_mean: bool = True,
) -> torch.Tensor:
    labels = torch.as_tensor(labels).long().reshape(-1)
    base_weights = torch.as_tensor(base_weights).double().reshape(-1)
    output = torch.empty_like(base_weights)
    for label in labels.unique().tolist():
        label = int(label)
        if label not in factors:
            raise ValueError(f"No class-balance factor was fitted for label {label}.")
        mask = labels == label
        output[mask] = base_weights[mask] * float(factors[label])
    if normalize_mean:
        output = output / output.mean().clamp(min=torch.finfo(output.dtype).tiny)
    return output.float()


def weighted_mean_and_std(
    values: torch.Tensor, weights: torch.Tensor, dim: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(values).double()
    weights = torch.as_tensor(weights).double().reshape(-1)
    if values.shape[0] != weights.numel():
        raise ValueError("The first value dimension must align with weights.")
    shape = [weights.numel()] + [1] * (values.ndim - 1)
    expanded = weights.reshape(shape)
    total = expanded.sum(dim=dim).clamp(min=torch.finfo(values.dtype).tiny)
    mean = (values * expanded).sum(dim=dim) / total
    centered = values - mean.unsqueeze(dim)
    variance = (centered.square() * expanded).sum(dim=dim) / total
    return mean.float(), variance.clamp(min=0.0).sqrt().float()


def weighted_quantile(
    values: torch.Tensor, quantiles: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    values = torch.as_tensor(values).double().reshape(-1)
    quantiles = torch.as_tensor(quantiles).double().reshape(-1)
    weights = torch.as_tensor(weights).double().reshape(-1)
    if values.numel() != weights.numel():
        raise ValueError("values and weights must have equal lengths.")
    if not ((quantiles >= 0) & (quantiles <= 1)).all():
        raise ValueError("quantiles must lie in [0, 1].")
    valid = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError("Weighted quantiles require positive finite weight.")
    values = values[valid]
    weights = weights[valid]
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = torch.cumsum(sorted_weights, dim=0)
    if float(cumulative[-1]) <= 0.0:
        raise ValueError("Cannot compute weighted quantiles with zero total mass.")
    targets = quantiles * cumulative[-1]
    positions = torch.searchsorted(cumulative, targets, right=False)
    positions = positions.clamp(max=sorted_values.numel() - 1)
    return sorted_values[positions].float()


def _assign_strata(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values).float().reshape(-1)
    edges = torch.as_tensor(edges).float().reshape(-1)
    if edges.numel() < 2:
        raise ValueError("At least two distinct stratum edges are required.")
    return torch.bucketize(values, edges[1:-1]).long()


def fit_joint_balance(
    labels: torch.Tensor,
    nuisance: torch.Tensor,
    base_weights: torch.Tensor,
    n_strata: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Fit engineer-style class/nuisance balancing on the training split.

    The generator-weighted mass of every occupied nuisance stratum is equal
    within a class, and every class receives equal total mass.  Quantile strata
    are an internal density-estimation device and are never passed to the model.
    """
    if int(n_strata) < 2:
        raise ValueError("n_strata must be at least two.")
    labels = torch.as_tensor(labels).long().reshape(-1)
    nuisance = torch.as_tensor(nuisance).float().reshape(-1)
    base_weights = torch.as_tensor(base_weights).double().reshape(-1)
    if not (labels.numel() == nuisance.numel() == base_weights.numel()):
        raise ValueError("labels, nuisance, and base_weights must align.")

    class_factors = fit_class_balance_factors(labels, base_weights)
    quantile_weights = apply_class_balance_factors(
        labels, base_weights, class_factors, normalize_mean=False).double()
    edges = weighted_quantile(
        nuisance,
        torch.linspace(0.0, 1.0, int(n_strata) + 1),
        quantile_weights,
    )
    edges = torch.unique_consecutive(edges)
    if edges.numel() < 3:
        raise ValueError(
            "The continuous nuisance has too few distinct values for balancing.")
    strata = _assign_strata(nuisance, edges)

    classes = sorted(int(value) for value in labels.unique().tolist())
    factors: Dict[Tuple[int, int], float] = {}
    occupied: Dict[int, Iterable[int]] = {}
    for label in classes:
        label_mask = labels == label
        occupied_bins = sorted(int(v) for v in strata[label_mask].unique().tolist())
        occupied[label] = occupied_bins
        for stratum in occupied_bins:
            mask = label_mask & (strata == stratum)
            mass = float(base_weights[mask].sum())
            if mass <= 0.0:
                raise ValueError(
                    f"Class/stratum ({label}, {stratum}) has non-positive mass.")
            factors[(label, stratum)] = (
                1.0 / (len(classes) * len(occupied_bins) * mass))

    spec = {
        "version": 1,
        "method": "uniform_class_and_occupied_continuous_nuisance_strata",
        "edges": edges.cpu(),
        "factors": factors,
        "occupied": {key: list(value) for key, value in occupied.items()},
        "classes": classes,
        "requested_strata": int(n_strata),
        "effective_strata": int(edges.numel() - 1),
    }
    weights = apply_joint_balance(labels, nuisance, base_weights, spec)
    return weights, spec


def apply_joint_balance(
    labels: torch.Tensor,
    nuisance: torch.Tensor,
    base_weights: torch.Tensor,
    spec: Mapping[str, object],
    normalize_mean: bool = True,
) -> torch.Tensor:
    labels = torch.as_tensor(labels).long().reshape(-1)
    nuisance = torch.as_tensor(nuisance).float().reshape(-1)
    base_weights = torch.as_tensor(base_weights).double().reshape(-1)
    strata = _assign_strata(nuisance, spec["edges"])
    factors = spec["factors"]
    occupied = spec["occupied"]
    output = torch.empty_like(base_weights)
    fallback_counts = defaultdict(int)

    for label in labels.unique().tolist():
        label = int(label)
        if label not in occupied:
            raise ValueError(f"Validation contains unseen class {label}.")
        available = [int(value) for value in occupied[label]]
        for stratum in strata[labels == label].unique().tolist():
            stratum = int(stratum)
            key = (label, stratum)
            use_stratum = stratum
            if key not in factors:
                use_stratum = min(available, key=lambda value: abs(value - stratum))
                key = (label, use_stratum)
                fallback_counts[(label, stratum, use_stratum)] += 1
            mask = (labels == label) & (strata == stratum)
            output[mask] = base_weights[mask] * float(factors[key])

    if fallback_counts:
        print(
            "WARNING: applied nearest training-stratum factors for validation "
            f"cells {dict(fallback_counts)}",
            flush=True,
        )
    if normalize_mean:
        output = output / output.mean().clamp(min=torch.finfo(output.dtype).tiny)
    return output.float()


def effective_mass_by_class(
    labels: torch.Tensor, weights: torch.Tensor
) -> Dict[int, float]:
    labels = torch.as_tensor(labels).long().reshape(-1)
    weights = torch.as_tensor(weights).double().reshape(-1)
    total = float(weights.sum())
    return {
        int(label): float(weights[labels == label].sum()) / total
        for label in labels.unique().tolist()
    }


def effective_sample_size_fraction(weights: torch.Tensor) -> float:
    weights = torch.as_tensor(weights).double().reshape(-1)
    denominator = float(weights.square().sum())
    if denominator <= 0.0:
        return 0.0
    return float(weights.sum().square() / denominator / weights.numel())
