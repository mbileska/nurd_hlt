#!/usr/bin/env python3
"""Read-only, plot-heavy audit of the HLT SM-cocktail tensors.

The audit intentionally does not rewrite samples or prescribe physics cuts.  It
checks the exact assumptions made by ``train_ae.py``,
``dataset/hlt_smcocktail_dataset.py``, and ``models/hlt_con.py`` and exports row
indices that deserve inspection before a filtering decision is made.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
CLASS_COLORS = {0: "#3f90da", 1: "#ffa90e", 2: "#bd1f01", 3: "#94a4a2"}
PF_NAMES = [r"PF $p_{T}$", r"PF $\eta$", r"PF $\phi$", r"PF $d_{xy}$",
            r"PF $d_{xy}/\sigma$", "PF flag", "PF PDG ID"]
OBJ_NAMES = ["object feature 0", "object feature 1", "object feature 2",
             "object feature 3"]
WEIGHT_KEYS = ("gen_weight", "gen_weights", "event_weight", "event_weights",
               "weight", "weights")
EVENT_ID_KEYS = ("eventid", "event_id", "eventids", "event_ids")
ALLOWED_ABS_PDG_IDS = np.asarray([1, 2, 11, 13, 22, 130, 211], dtype=np.int64)
SEVERITY_ORDER = {"BLOCKER": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_torch(path: Path) -> Any:
    """Prefer memory mapping for the multi-GB samples, with old-Torch fallback."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def find_mapping_value(mapping: dict[str, Any], keys: Iterable[str]):
    for key in keys:
        if key in mapping:
            return mapping[key], key
    return None, None


def as_1d_numpy(value: Any) -> np.ndarray:
    array = torch.as_tensor(value).detach().cpu().numpy()
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"expected a one-dimensional array, got {array.shape}")
    return np.asarray(array)


def event_ids_from_payload(payload: Any):
    if not isinstance(payload, dict):
        return None, None
    value, key = find_mapping_value(payload, EVENT_ID_KEYS)
    if value is None:
        return None, None
    return as_1d_numpy(value), key


def load_weights_for_audit(path: Path | None, sample: dict[str, Any], n: int):
    """Load weights permissively so malformed weights appear in the report."""
    if path is None:
        weights = np.ones(n, dtype=np.float64)
        return weights, {
            "path": None, "source": "unit", "length_matches": True,
            "alignment_verified": True, "alignment_reason": "unit weights",
        }

    payload = load_torch(path)
    source = type(payload).__name__
    payload_ids, payload_id_key = event_ids_from_payload(payload)
    if isinstance(payload, dict):
        value, key = find_mapping_value(payload, WEIGHT_KEYS)
        if value is None:
            candidates = [(key, value) for key, value in payload.items()
                          if torch.is_tensor(value) or isinstance(value, np.ndarray)]
            candidates = [(key, value) for key, value in candidates
                          if key not in EVENT_ID_KEYS]
            if len(candidates) != 1:
                raise ValueError(
                    f"cannot identify weights in {path}; keys={list(payload)}")
            key, value = candidates[0]
        source = f"dict:{key}"
    else:
        value = payload

    weights = as_1d_numpy(value).astype(np.float64, copy=False)
    sample_ids, sample_id_key = event_ids_from_payload(sample)
    alignment_verified = False
    alignment_reason = "neither file supplies matching event IDs"
    if payload_ids is not None and sample_ids is not None:
        alignment_verified = (payload_ids.shape == sample_ids.shape and
                              np.array_equal(payload_ids, sample_ids))
        alignment_reason = ("exact event-ID order match" if alignment_verified
                            else "event IDs do not match exactly")
    elif payload_ids is not None:
        alignment_reason = "weight file has event IDs but sample does not"
    elif sample_ids is not None:
        alignment_reason = "sample has event IDs but weight file does not"
    return weights, {
        "path": str(path), "source": source,
        "length_matches": bool(weights.size == n),
        "alignment_verified": alignment_verified,
        "alignment_reason": alignment_reason,
        "weight_event_id_key": payload_id_key,
        "sample_event_id_key": sample_id_key,
    }


def safe_weights(weights: np.ndarray, n: int) -> np.ndarray:
    if weights.size != n:
        return np.ones(n, dtype=np.float64)
    result = np.asarray(weights, dtype=np.float64).copy()
    result[~np.isfinite(result) | (result < 0)] = 0.0
    if result.sum() <= 0:
        result[:] = 1.0
    return result


def effective_events(weights: np.ndarray) -> float:
    total = float(np.sum(weights, dtype=np.float64))
    square = float(np.dot(weights, weights))
    return total * total / square if square > 0 else 0.0


def weighted_quantile(values: np.ndarray, quantiles: Iterable[float],
                      weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    values = values[finite]
    q = np.asarray(list(quantiles), dtype=np.float64)
    if values.size == 0:
        return np.full(q.shape, np.nan)
    if weights is None:
        return np.quantile(values, q)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)[finite]
    keep = np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        return np.full(q.shape, np.nan)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= weights.sum()
    return np.interp(q, cdf, values, left=values[0], right=values[-1])


def weighted_ks(x: np.ndarray, y: np.ndarray, wx: np.ndarray,
                wy: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    wx, wy = np.asarray(wx, dtype=np.float64), np.asarray(wy, dtype=np.float64)
    mx = np.isfinite(x) & np.isfinite(wx) & (wx > 0)
    my = np.isfinite(y) & np.isfinite(wy) & (wy > 0)
    x, wx, y, wy = x[mx], wx[mx], y[my], wy[my]
    if x.size == 0 or y.size == 0:
        return float("nan")
    ox, oy = np.argsort(x), np.argsort(y)
    x, wx, y, wy = x[ox], wx[ox], y[oy], wy[oy]
    cwx, cwy = np.cumsum(wx) / wx.sum(), np.cumsum(wy) / wy.sum()
    grid = np.unique(np.concatenate([x, y]))
    ix = np.searchsorted(x, grid, side="right") - 1
    iy = np.searchsorted(y, grid, side="right") - 1
    fx = np.where(ix >= 0, cwx[np.maximum(ix, 0)], 0.0)
    fy = np.where(iy >= 0, cwy[np.maximum(iy, 0)], 0.0)
    return float(np.max(np.abs(fx - fy)))


def robust_outlier_score(metrics: dict[str, np.ndarray], seed: int,
                         max_reference: int = 250_000):
    """Maximum one-sided robust z over positive event-scale summaries."""
    names = ["pf_n_active", "pf_sum_pt", "pf_max_pt", "pf_max_abs_eta",
             "pf_max_abs_dxysig", "pf_sum_energy", "obj_n_active", "obj_max_abs", "obj_l2"]
    n = len(next(iter(metrics.values())))
    rng = np.random.default_rng(seed)
    reference = (np.arange(n) if n <= max_reference else
                 rng.choice(n, max_reference, replace=False))
    score = np.zeros(n, dtype=np.float32)
    calibration = {}
    for name in names:
        values = np.asarray(metrics[name], dtype=np.float64)
        transformed = np.log1p(np.maximum(values, 0.0))
        finite_ref = transformed[reference][np.isfinite(transformed[reference])]
        if finite_ref.size == 0:
            continue
        median = float(np.median(finite_ref))
        mad = float(np.median(np.abs(finite_ref - median)))
        scale = max(1.4826 * mad, 1e-6)
        z = np.abs(transformed - median) / scale
        z[~np.isfinite(z)] = np.inf
        score = np.maximum(score, np.minimum(z, np.finfo(np.float32).max))
        calibration[name] = {"log1p_median": median, "robust_scale": scale}
    return score, calibration


def stratified_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if labels.size <= maximum:
        return np.arange(labels.size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    parts = []
    unique, counts = np.unique(labels, return_counts=True)
    assigned = 0
    for position, (label, count) in enumerate(zip(unique, counts)):
        if position == len(unique) - 1:
            take = maximum - assigned
        else:
            take = max(1, int(round(maximum * count / labels.size)))
            take = min(take, maximum - assigned - (len(unique) - position - 1))
        choices = np.flatnonzero(labels == label)
        parts.append(rng.choice(choices, min(take, choices.size), replace=False))
        assigned += len(parts[-1])
    return np.sort(np.concatenate(parts).astype(np.int64))


def sampled_candidates(tensor: torch.Tensor, labels: np.ndarray,
                       weights: np.ndarray, maximum: int, seed: int,
                       active_from_pt: bool):
    n_events, n_candidates, n_features = tensor.shape
    total = n_events * n_candidates
    rng = np.random.default_rng(seed)
    count = min(maximum, total)
    flat = rng.choice(total, count, replace=False)
    event = flat // n_candidates
    slot = flat % n_candidates
    # Keep memory-mapped sample reads monotonic on GPFS rather than issuing
    # hundreds of thousands of random gathers across the multi-GB PF tensor.
    order = np.lexsort((slot, event))
    event, slot = event[order], slot[order]
    values = tensor[torch.from_numpy(event), torch.from_numpy(slot)].float().numpy()
    finite = np.isfinite(values).all(axis=1)
    active = ((values[:, 0] > 0) if active_from_pt
              else np.any(np.nan_to_num(values, nan=0.0) != 0, axis=1))
    keep = finite & active
    return {
        "values": values[keep], "labels": labels[event][keep],
        "weights": weights[event][keep], "event_indices": event[keep],
        "candidate_slots": slot[keep],
        "drawn": int(count), "retained": int(np.count_nonzero(keep)),
    }


@dataclass
class Finding:
    severity: str
    dataset: str
    check: str
    message: str
    action: str

    def as_dict(self):
        return self.__dict__.copy()


def add_finding(findings: list[Finding], severity: str, dataset: str,
                check: str, message: str, action: str):
    findings.append(Finding(severity, dataset, check, message, action))


def summarize_weights(raw_weights: np.ndarray, n: int) -> dict[str, Any]:
    valid_length = raw_weights.size == n
    if not valid_length:
        return {"n": int(raw_weights.size), "expected_n": int(n)}
    finite = np.isfinite(raw_weights)
    nonnegative = raw_weights >= 0
    usable = finite & nonnegative
    values = raw_weights[usable]
    positive = values[values > 0]
    result = {
        "n": int(n), "finite": int(finite.sum()),
        "nonfinite": int((~finite).sum()), "negative": int((raw_weights < 0).sum()),
        "zero": int((raw_weights == 0).sum()),
    }
    if positive.size:
        order = np.sort(positive)[::-1]
        csum = np.cumsum(order)
        total = float(csum[-1])
        result.update({
            "sum": total, "mean": float(np.mean(values)),
            "std": float(np.std(values)), "min_positive": float(order[-1]),
            "median": float(np.median(values)), "max": float(order[0]),
            "effective_events": effective_events(values),
            "effective_fraction": effective_events(values) / n,
            "max_over_median": (float(order[0] / np.median(positive))
                                if np.median(positive) > 0 else None),
            "top_0p1pct_mass_fraction": float(
                csum[min(order.size, max(1, math.ceil(.001 * n))) - 1] / total),
            "top_1pct_mass_fraction": float(
                csum[min(order.size, max(1, math.ceil(.01 * n))) - 1] / total),
            "max_mass_fraction": float(order[0] / total),
            "quantiles": weighted_quantile(values, [0, .001, .01, .1, .5, .9, .99, .999, 1]).tolist(),
        })
    return result


def audit_dataset(name: str, data_path: Path, weight_path: Path | None,
                  output: Path, args, findings: list[Finding]):
    print(f"[{name}] loading {data_path}", flush=True)
    payload = load_torch(data_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{data_path} is {type(payload).__name__}, expected a dict")
    missing = [key for key in ("pf", "obj", "label") if key not in payload]
    if missing:
        raise ValueError(f"{data_path} is missing required keys {missing}")
    pf, obj = torch.as_tensor(payload["pf"]), torch.as_tensor(payload["obj"])
    raw_labels = as_1d_numpy(payload["label"])
    numeric_labels = np.asarray(raw_labels, dtype=np.float64)
    invalid_labels = ~np.isfinite(numeric_labels) | (
        np.abs(numeric_labels - np.rint(numeric_labels)) > 1e-6)
    if invalid_labels.any():
        add_finding(findings, "BLOCKER", name, "invalid_labels",
                    f"{int(invalid_labels.sum()):,} labels are nonfinite or non-integral.",
                    "Fix the label tensor upstream; integer casting would silently change class identity.")
    labels = np.where(np.isfinite(numeric_labels), np.rint(numeric_labels), -999).astype(np.int64)
    n = labels.size
    if pf.ndim != 3 or obj.ndim != 3:
        raise ValueError(f"PF/obj must be rank 3; got {tuple(pf.shape)}, {tuple(obj.shape)}")
    if pf.shape[0] != n or obj.shape[0] != n:
        raise ValueError("PF, obj, and label event counts do not agree")
    if pf.shape[2] < 7:
        raise ValueError(f"PF has {pf.shape[2]} features; the model requires at least 7")
    if obj.shape[2] < 4:
        raise ValueError(f"obj has {obj.shape[2]} features; the AE requires at least 4")
    if pf.shape[2] > 7:
        add_finding(findings, "INFO", name, "unused_pf_columns",
                    f"PF contains {pf.shape[2]} columns but the model consumes only columns 0–6.",
                    "Confirm the ignored columns are intentional.")
    if obj.shape[2] > 4:
        add_finding(findings, "INFO", name, "unused_obj_columns",
                    f"Object records contain {obj.shape[2]} columns but the AE consumes only columns 0–3.",
                    "Confirm the ignored columns are intentional.")

    raw_weights, weight_meta = load_weights_for_audit(weight_path, payload, n)
    weights = safe_weights(raw_weights, n)
    weight_stats = summarize_weights(raw_weights, n)
    if raw_weights.size != n:
        add_finding(findings, "BLOCKER", name, "weight_length",
                    f"Weight file has {raw_weights.size:,} rows for {n:,} events.",
                    "Do not train; regenerate or match the correct weight file.")
    if weight_stats.get("nonfinite", 0) or weight_stats.get("negative", 0):
        add_finding(findings, "BLOCKER", name, "invalid_weights",
                    f"Weights contain {weight_stats.get('nonfinite', 0)} nonfinite and "
                    f"{weight_stats.get('negative', 0)} negative entries.",
                    "Current positive-weight losses cannot use these rows safely.")
    if weight_path and not weight_meta["alignment_verified"]:
        add_finding(findings, "HIGH", name, "weight_alignment",
                    f"Weight row order is not cryptographically/event-ID verifiable: "
                    f"{weight_meta['alignment_reason']}.",
                    "Regenerate data and weights with identical event IDs in both files; "
                    "marginal agreement alone cannot prove alignment.")
    ess_frac = weight_stats.get("effective_fraction")
    if ess_frac is not None and ess_frac < args.min_ess_fraction:
        add_finding(findings, "HIGH", name, "weight_ess",
                    f"Generator-weight ESS is {weight_stats['effective_events']:,.0f} "
                    f"({100 * ess_frac:.2f}% of rows).",
                    "Do not drop high-weight events blindly; inspect generator strata and "
                    "use weighted uncertainty/regularization or obtain more effective MC.")
    if weight_stats.get("top_1pct_mass_fraction", 0) > args.max_top1_mass:
        add_finding(findings, "WARNING", name, "weight_concentration",
                    f"Top 1% of rows carry {100 * weight_stats['top_1pct_mass_fraction']:.1f}% "
                    "of generator-weight mass.",
                    "Treat raw event count as misleading and propagate sumw2/ESS everywhere.")

    label_values, label_counts = np.unique(labels, return_counts=True)
    class_summary = {}
    for label, count in zip(label_values, label_counts):
        mask = labels == label
        w = weights[mask]
        class_summary[str(int(label))] = {
            "name": CLASS_NAMES.get(int(label), f"class {label}"),
            "rows": int(count), "row_fraction": float(count / n),
            "sum_weights": float(w.sum()), "weight_fraction": float(w.sum() / weights.sum()),
            "effective_events": effective_events(w),
        }
    unexpected = sorted(set(label_values.tolist()) - set(CLASS_NAMES))
    if unexpected:
        add_finding(findings, "WARNING", name, "labels",
                    f"Unexpected class labels are present: {unexpected}.",
                    "Confirm the label map before training or evaluating background groups.")

    metrics = {
        "pf_n_active": np.empty(n, dtype=np.uint16),
        "pf_sum_pt": np.empty(n, dtype=np.float32),
        "pf_max_pt": np.empty(n, dtype=np.float32),
        "pf_max_abs_eta": np.empty(n, dtype=np.float32),
        "pf_max_abs_dxysig": np.empty(n, dtype=np.float32),
        "pf_sum_energy": np.empty(n, dtype=np.float32),
        "obj_n_active": np.empty(n, dtype=np.uint8),
        "obj_max_abs": np.empty(n, dtype=np.float32),
        "obj_l2": np.empty(n, dtype=np.float32),
    }
    flags = {key: np.zeros(n, dtype=bool) for key in (
        "pf_nonfinite", "obj_nonfinite", "negative_or_nonfinite_pt",
        "all_padded_pf", "all_zero_obj", "nonzero_padded_pf",
        "pf_sequence_hole", "pf_pt_not_nonincreasing",
        "pf_candidate_capacity_saturated", "obj_candidate_capacity_saturated",
        "unsupported_pdgid", "invalid_pf_flag")}
    if raw_weights.size == n:
        flags["nonfinite_weight"] = ~np.isfinite(raw_weights)
        flags["negative_weight"] = np.isfinite(raw_weights) & (raw_weights < 0)
        flags["zero_weight"] = np.isfinite(raw_weights) & (raw_weights == 0)
    else:
        flags["nonfinite_weight"] = np.zeros(n, dtype=bool)
        flags["negative_weight"] = np.zeros(n, dtype=bool)
        flags["zero_weight"] = np.zeros(n, dtype=bool)
    flags["invalid_label"] = invalid_labels.astype(bool, copy=True)
    pf_feature_counts = np.zeros((pf.shape[2], 4), dtype=np.int64)
    obj_feature_counts = np.zeros((min(4, obj.shape[2]), 3), dtype=np.int64)
    obj_weighted_sum = np.zeros((obj.shape[1], 4), dtype=np.float64)
    obj_weighted_square_sum = np.zeros((obj.shape[1], 4), dtype=np.float64)

    print(f"[{name}] scanning {n:,} events in chunks", flush=True)
    for start in range(0, n, args.chunk_size):
        stop = min(start + args.chunk_size, n)
        p = pf[start:stop].float()
        o = obj[start:stop, :, :4].float()
        p_np = p.numpy()
        o_np = o.numpy()
        finite_p = np.isfinite(p_np)
        finite_o = np.isfinite(o_np)
        pt = p_np[:, :, 0]
        active = np.isfinite(pt) & (pt > 0)
        padded = np.isfinite(pt) & (pt == 0)
        sanitized_p = np.nan_to_num(p_np, nan=0.0, posinf=0.0, neginf=0.0)
        sanitized_o = np.nan_to_num(o_np, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_weights = weights[start:stop]

        flags["pf_nonfinite"][start:stop] = (~finite_p).any(axis=(1, 2))
        flags["obj_nonfinite"][start:stop] = (~finite_o).any(axis=(1, 2))
        flags["negative_or_nonfinite_pt"][start:stop] = (
            (~np.isfinite(pt)) | (pt < 0)).any(axis=1)
        flags["all_padded_pf"][start:stop] = ~active.any(axis=1)
        flags["all_zero_obj"][start:stop] = ~np.any(sanitized_o != 0, axis=(1, 2))
        flags["nonzero_padded_pf"][start:stop] = (
            padded & np.any(sanitized_p[:, :, 1:7] != 0, axis=2)).any(axis=1)
        seen_pad = np.maximum.accumulate(~active, axis=1)
        flags["pf_sequence_hole"][start:stop] = (active & seen_pad).any(axis=1)
        adjacent_active = active[:, 1:] & active[:, :-1]
        flags["pf_pt_not_nonincreasing"][start:stop] = (
            adjacent_active & (pt[:, 1:] > pt[:, :-1] + 1e-6)).any(axis=1)
        flags["pf_candidate_capacity_saturated"][start:stop] = (
            active.sum(axis=1) == pf.shape[1])
        pid = sanitized_p[:, :, 6].astype(np.int64, copy=False)
        flags["unsupported_pdgid"][start:stop] = (
            active & ~np.isin(np.abs(pid), ALLOWED_ABS_PDG_IDS)).any(axis=1)
        pf_flag = sanitized_p[:, :, 5]
        flags["invalid_pf_flag"][start:stop] = (
            active & ~np.isin(pf_flag, [0.0, 1.0])).any(axis=1)

        metrics["pf_n_active"][start:stop] = active.sum(axis=1)
        metrics["pf_sum_pt"][start:stop] = np.where(active, sanitized_p[:, :, 0], 0).sum(axis=1)
        metrics["pf_max_pt"][start:stop] = np.where(active, sanitized_p[:, :, 0], 0).max(axis=1)
        metrics["pf_max_abs_eta"][start:stop] = np.where(
            active, np.abs(sanitized_p[:, :, 1]), 0).max(axis=1)
        metrics["pf_max_abs_dxysig"][start:stop] = np.where(
            active, np.abs(sanitized_p[:, :, 4]), 0).max(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            energy = np.where(
                active, np.abs(sanitized_p[:, :, 0] * np.cosh(sanitized_p[:, :, 1])), 0.0)
        flags.setdefault("pf_preprocessor_nonfinite_energy", np.zeros(n, dtype=bool))
        flags["pf_preprocessor_nonfinite_energy"][start:stop] = ~np.isfinite(energy).all(axis=1)
        metrics["pf_sum_energy"][start:stop] = np.nan_to_num(
            energy, nan=0.0, posinf=np.finfo(np.float32).max,
            neginf=0.0).sum(axis=1, dtype=np.float64).clip(
                max=np.finfo(np.float32).max)
        obj_active = np.any(sanitized_o != 0, axis=2)
        flags["obj_candidate_capacity_saturated"][start:stop] = (
            obj_active.sum(axis=1) == obj.shape[1])
        metrics["obj_n_active"][start:stop] = obj_active.sum(axis=1)
        metrics["obj_max_abs"][start:stop] = np.abs(sanitized_o).max(axis=(1, 2))
        metrics["obj_l2"][start:stop] = np.sqrt(np.square(sanitized_o).sum(axis=(1, 2)))
        obj_weighted_sum += np.einsum(
            "n,ncf->cf", chunk_weights, sanitized_o, optimize=True)
        obj_weighted_square_sum += np.einsum(
            "n,ncf->cf", chunk_weights, np.square(sanitized_o), optimize=True)

        for feature in range(pf.shape[2]):
            feature_finite = finite_p[:, :, feature]
            pf_feature_counts[feature] += [
                int(feature_finite.sum()), int((~feature_finite).sum()),
                int((sanitized_p[:, :, feature] == 0).sum()),
                int((active & feature_finite).sum()),
            ]
        for feature in range(min(4, obj.shape[2])):
            feature_finite = finite_o[:, :, feature]
            obj_feature_counts[feature] += [
                int(feature_finite.sum()), int((~feature_finite).sum()),
                int((sanitized_o[:, :, feature] == 0).sum()),
            ]
        del p, o, p_np, o_np, finite_p, finite_o, sanitized_p, sanitized_o, energy

    obj_total_weight = max(float(weights.sum()), 1e-30)
    obj_scaler_mean = obj_weighted_sum / obj_total_weight
    obj_scaler_variance = np.maximum(
        obj_weighted_square_sum / obj_total_weight - np.square(obj_scaler_mean), 0.0)
    obj_scaler_std = np.sqrt(obj_scaler_variance)
    obj_scaler_safe_std = np.where(obj_scaler_std < 1e-8, 1.0, obj_scaler_std)
    obj_z10_rows = np.zeros(n, dtype=bool)
    obj_z20_rows = np.zeros(n, dtype=bool)
    obj_z10_count = np.zeros_like(obj_scaler_mean, dtype=np.int64)
    obj_z20_count = np.zeros_like(obj_scaler_mean, dtype=np.int64)
    obj_z10_weight = np.zeros_like(obj_scaler_mean, dtype=np.float64)
    obj_z20_weight = np.zeros_like(obj_scaler_mean, dtype=np.float64)
    for start in range(0, n, args.chunk_size):
        stop = min(start + args.chunk_size, n)
        o_np = obj[start:stop, :, :4].float().numpy()
        sanitized_o = np.nan_to_num(o_np, nan=0.0, posinf=0.0, neginf=0.0)
        standardized = ((sanitized_o - obj_scaler_mean[None, :, :]) /
                        obj_scaler_safe_std[None, :, :])
        beyond10 = np.abs(standardized) > 10
        beyond20 = np.abs(standardized) > 20
        obj_z10_rows[start:stop] = beyond10.any(axis=(1, 2))
        obj_z20_rows[start:stop] = beyond20.any(axis=(1, 2))
        obj_z10_count += beyond10.sum(axis=0)
        obj_z20_count += beyond20.sum(axis=0)
        chunk_weights = weights[start:stop]
        obj_z10_weight += np.einsum("n,ncf->cf", chunk_weights, beyond10, optimize=True)
        obj_z20_weight += np.einsum("n,ncf->cf", chunk_weights, beyond20, optimize=True)
        del o_np, sanitized_o, standardized, beyond10, beyond20

    flags["obj_standardized_beyond_10"] = obj_z10_rows
    flags["obj_standardized_beyond_20"] = obj_z20_rows
    obj_scaling = {
        "scope": "full-file generator-weighted diagnostic; training fits the same statistic on its reference split",
        "near_constant_coordinates_std_lt_1e-8": int((obj_scaler_std < 1e-8).sum()),
        "total_coordinates": int(obj_scaler_std.size),
        "mean": obj_scaler_mean.tolist(), "std": obj_scaler_std.tolist(),
        "rows_with_abs_z_gt_10": int(obj_z10_rows.sum()),
        "rows_with_abs_z_gt_20": int(obj_z20_rows.sum()),
        "weight_fraction_rows_abs_z_gt_10": float(weights[obj_z10_rows].sum() / weights.sum()),
        "weight_fraction_rows_abs_z_gt_20": float(weights[obj_z20_rows].sum() / weights.sum()),
        "max_coordinate_row_fraction_abs_z_gt_10": float((obj_z10_count / n).max()),
        "max_coordinate_row_fraction_abs_z_gt_20": float((obj_z20_count / n).max()),
        "max_coordinate_weight_fraction_abs_z_gt_10": float((obj_z10_weight / weights.sum()).max()),
        "max_coordinate_weight_fraction_abs_z_gt_20": float((obj_z20_weight / weights.sum()).max()),
    }
    if obj_scaling["near_constant_coordinates_std_lt_1e-8"]:
        add_finding(findings, "WARNING", name, "ae_scaler_constant_coordinates",
                    f"{obj_scaling['near_constant_coordinates_std_lt_1e-8']} of "
                    f"{obj_scaling['total_coordinates']} flattened AE coordinates have weighted std < 1e-8.",
                    "Confirm these are intentional padding coordinates; the pipeline replaces their scale by one.")
    if obj_scaling["rows_with_abs_z_gt_20"]:
        add_finding(findings, "WARNING", name, "ae_scaler_extreme_z",
                    f"{obj_scaling['rows_with_abs_z_gt_20']:,} rows contain an AE input coordinate "
                    "more than 20 weighted standard deviations from its full-file mean.",
                    "Inspect the exported rows and per-feature plots; use a robust transform only if physics semantics justify it.")

    outlier_score, outlier_calibration = robust_outlier_score(
        metrics, seed=args.seed + sum(map(ord, name)))
    flags["extreme_robust_outlier"] = outlier_score > args.outlier_z
    if raw_weights.size == n:
        finite_positive = raw_weights[np.isfinite(raw_weights) & (raw_weights >= 0)]
        threshold = (np.quantile(finite_positive, 0.999) if finite_positive.size
                     else np.inf)
        flags["top_weight_0p1pct"] = np.isfinite(raw_weights) & (raw_weights >= threshold)
    else:
        flags["top_weight_0p1pct"] = np.zeros(n, dtype=bool)

    hard_names = ["pf_nonfinite", "obj_nonfinite", "negative_or_nonfinite_pt",
                  "all_padded_pf", "nonfinite_weight", "negative_weight", "invalid_label"]
    review_names = ["all_zero_obj", "nonzero_padded_pf", "pf_sequence_hole",
                    "pf_pt_not_nonincreasing",
                    "pf_candidate_capacity_saturated", "obj_candidate_capacity_saturated",
                    "unsupported_pdgid",
                    "invalid_pf_flag", "pf_preprocessor_nonfinite_energy",
                    "obj_standardized_beyond_10",
                    "extreme_robust_outlier", "top_weight_0p1pct", "zero_weight"]
    hard = np.logical_or.reduce([flags[key] for key in hard_names])
    review = np.logical_or.reduce([flags[key] for key in review_names])
    flag_summary = {}
    for flag_name, values in flags.items():
        sumw = float(weights[values].sum())
        flag_summary[flag_name] = {
            "rows": int(values.sum()), "row_fraction": float(values.mean()),
            "sum_weights": sumw, "weight_fraction": float(sumw / weights.sum()),
        }

    for flag_name, severity, description, action in [
        ("pf_nonfinite", "HIGH", "PF events contain NaN/Inf values",
         "Inspect their origin; current training silently replaces them by zero."),
        ("obj_nonfinite", "HIGH", "AE object inputs contain NaN/Inf values",
         "Inspect or explicitly filter/impute them before AE scaling; current training silently uses zero."),
        ("negative_or_nonfinite_pt", "HIGH", "PF pT is negative/nonfinite in some events",
         "These violate the model's pT-based validity convention and should be fixed upstream or excluded."),
        ("all_padded_pf", "HIGH", "Some events have no positive-pT PF candidate",
         "Review and normally exclude these unless empty events are intentional."),
        ("all_zero_obj", "WARNING", "Some events have an entirely zero AE object record",
         "Confirm whether this is a legitimate trigger-level empty record."),
        ("nonzero_padded_pf", "WARNING", "pT=0 padded candidates have nonzero auxiliary fields",
         "Confirm producer padding; the model masks these tokens by pT only."),
        ("pf_sequence_hole", "WARNING", "Positive-pT candidates occur after a padding/nonpositive slot",
         "Confirm candidates are compacted and sorted as expected."),
        ("pf_pt_not_nonincreasing", "WARNING", "Active PF candidates are not ordered by descending pT",
         "Confirm the producer ordering; this transformer uses positional token projections, so order changes its input."),
        ("pf_candidate_capacity_saturated", "WARNING", "Some events fill every stored PF candidate slot",
         "Check the pre-truncation multiplicity to determine whether physically relevant candidates were dropped."),
        ("obj_candidate_capacity_saturated", "WARNING", "Some events fill every stored object candidate slot",
         "Check the producer for overflow beyond the AE tensor capacity."),
        ("unsupported_pdgid", "WARNING", "Active candidates contain PDG IDs outside the model vocabulary",
         "Extend the vocabulary or confirm mapping to an all-zero category is intentional."),
        ("invalid_pf_flag", "WARNING", "The PF flag is not binary for some active candidates",
         "Confirm feature definition and upstream encoding."),
        ("pf_preprocessor_nonfinite_energy", "HIGH", "pT*cosh(eta) overflows in the model's energy transform",
         "Fix or exclude malformed kinematics before training; this can create nonfinite model activations."),
        ("extreme_robust_outlier", "WARNING", f"Event summaries exceed {args.outlier_z:g} robust MAD units",
         "Inspect these rows individually; do not remove them solely because they are rare."),
    ]:
        count = flag_summary[flag_name]["rows"]
        if count:
            add_finding(findings, severity, name, flag_name,
                        f"{description}: {count:,} rows ({100 * count / n:.4f}%), "
                        f"{100 * flag_summary[flag_name]['weight_fraction']:.4f}% of weight mass.", action)

    sample_idx = stratified_indices(labels, args.event_plot_sample,
                                    args.seed + sum(map(ord, name)))
    event_sample = {key: values[sample_idx] for key, values in metrics.items()}
    event_sample["outlier_score"] = outlier_score[sample_idx]
    pf_sample = sampled_candidates(pf[:, :, :7], labels, weights,
                                   args.candidate_plot_sample,
                                   args.seed + 1000 + sum(map(ord, name)), True)
    pf_event = pf_sample["event_indices"]
    pf_values = pf_sample["values"]
    pf_sample["log_pt_fraction"] = np.log(
        np.maximum(pf_values[:, 0] /
                   (metrics["pf_sum_pt"][pf_event] + 1e-4), np.finfo(np.float32).tiny))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        candidate_energy = np.abs(pf_values[:, 0] * np.cosh(pf_values[:, 1]))
        pf_sample["log_energy_fraction"] = np.log(
            candidate_energy / (metrics["pf_sum_energy"][pf_event] + 1e-4))
    pf_sample["tanh_dxy"] = np.tanh(pf_values[:, 3])
    obj_sample = sampled_candidates(obj[:, :, :4], labels, weights,
                                    args.candidate_plot_sample,
                                    args.seed + 2000 + sum(map(ord, name)), False)

    weight_impact = {}
    for kind, sample in (("pf", pf_sample), ("obj", obj_sample)):
        qcd = sample["labels"] == 1
        for feature in range(sample["values"].shape[1]):
            values = sample["values"][qcd, feature]
            physical = sample["weights"][qcd]
            weight_impact[f"{kind}_{feature}_qcd"] = weighted_ks(
                values, values, np.ones(values.size), physical)
    qcd_pf = pf_sample["labels"] == 1
    for feature in ("log_pt_fraction", "log_energy_fraction", "tanh_dxy"):
        values = pf_sample[feature][qcd_pf]
        physical = pf_sample["weights"][qcd_pf]
        weight_impact[f"pf_preprocessed_{feature}_qcd"] = weighted_ks(
            values, values, np.ones(values.size), physical)
    qcd_events = labels[sample_idx] == 1
    for feature, values in event_sample.items():
        physical = weights[sample_idx][qcd_events]
        selected = values[qcd_events]
        weight_impact[f"event_{feature}_qcd"] = weighted_ks(
            selected, selected, np.ones(selected.size), physical)
    finite_impacts = {key: value for key, value in weight_impact.items()
                      if np.isfinite(value)}
    if finite_impacts:
        worst_key, worst_value = max(finite_impacts.items(), key=lambda item: item[1])
        if worst_value > .20:
            add_finding(findings, "WARNING", name, "large_generator_weight_impact",
                        f"Generator weights reshape sampled QCD distributions strongly; "
                        f"largest raw-vs-weighted KS is {worst_value:.3f} for {worst_key}.",
                        "This is not a reason to remove high-weight rows. Check the weight derivation; "
                        "note that AE scaling is weighted but PF BatchNorm running statistics are not.")

    ids, id_key = event_ids_from_payload(payload)
    ids_unique = None
    if ids is not None:
        ids_unique = bool(np.unique(ids).size == ids.size)
        if not ids_unique:
            add_finding(findings, "HIGH", name, "duplicate_event_ids",
                        "Event IDs are not unique within this sample.",
                        "Resolve duplicates or prove why repeated IDs represent distinct events.")

    flag_dir = output / "flags"
    flag_dir.mkdir(parents=True, exist_ok=True)
    export = {f"{key}_indices": np.flatnonzero(values).astype(np.int64)
              for key, values in flags.items()}
    export["hard_filter_candidate_indices"] = np.flatnonzero(hard).astype(np.int64)
    export["review_candidate_indices"] = np.flatnonzero(review).astype(np.int64)
    if ids is not None:
        export["event_ids"] = ids
    np.savez_compressed(flag_dir / f"{name}_event_flags.npz", **export)

    summary = {
        "name": name, "data_path": str(data_path), "weight_path": str(weight_path) if weight_path else None,
        "file_size_bytes": data_path.stat().st_size, "keys": sorted(payload.keys()),
        "file_mtime_utc": datetime.fromtimestamp(
            data_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "label_sha256": array_sha256(raw_labels),
        "weight_sha256": array_sha256(raw_weights),
        "n_events": int(n), "pf_shape": list(pf.shape), "pf_dtype": str(pf.dtype),
        "obj_shape": list(obj.shape), "obj_dtype": str(obj.dtype),
        "label_shape": list(torch.as_tensor(payload["label"]).shape),
        "label_dtype": str(torch.as_tensor(payload["label"]).dtype),
        "unused_pf_features": max(0, int(pf.shape[2]) - 7),
        "unused_obj_features": max(0, int(obj.shape[2]) - 4),
        "event_id_key": id_key, "event_ids_unique": ids_unique,
        "event_id_sha256": array_sha256(ids) if ids is not None else None,
        "weights": {**weight_meta, **weight_stats}, "classes": class_summary,
        "flags": flag_summary,
        "hard_filter_candidate_union": {
            "rows": int(hard.sum()), "row_fraction": float(hard.mean()),
            "weight_fraction": float(weights[hard].sum() / weights.sum()),
            "definition": hard_names,
        },
        "review_candidate_union": {
            "rows": int(review.sum()), "row_fraction": float(review.mean()),
            "weight_fraction": float(weights[review].sum() / weights.sum()),
            "definition": review_names,
        },
        "pf_feature_counts_columns": ["finite", "nonfinite", "zero_after_nan_to_num", "active_finite"],
        "pf_feature_counts": pf_feature_counts.tolist(),
        "obj_feature_counts_columns": ["finite", "nonfinite", "zero_after_nan_to_num"],
        "obj_feature_counts": obj_feature_counts.tolist(),
        "ae_object_scaling": obj_scaling,
        "outlier_calibration": outlier_calibration,
        "generator_weight_impact_qcd_ks": weight_impact,
        "plot_sampling": {
            "event_rows": int(sample_idx.size),
            "pf_candidates": {key: value for key, value in pf_sample.items()
                              if key in ("drawn", "retained")},
            "obj_candidates": {key: value for key, value in obj_sample.items()
                               if key in ("drawn", "retained")},
        },
    }
    runtime = {
        "labels": labels, "weights": weights, "raw_weights": raw_weights,
        "event_indices": sample_idx, "event_metrics": event_sample,
        "pf_sample": pf_sample, "obj_sample": obj_sample,
        "flags": {key: values[sample_idx] for key, values in flags.items()},
        "event_ids": ids,
    }
    del payload, pf, obj, metrics, flags, outlier_score
    gc.collect()
    return summary, runtime


def hep_style():
    plt.rcParams.update({
        "figure.figsize": (7.4, 6.0), "figure.dpi": 120,
        "savefig.dpi": 180, "font.size": 12, "axes.labelsize": 13,
        "axes.titlesize": 13, "legend.fontsize": 10,
        "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
        "ytick.minor.visible": True, "legend.frameon": False,
    })


def label_axis(ax, sample_text="HLT SM cocktail"):
    ax.text(0.0, 1.015, "CMS", transform=ax.transAxes, fontweight="bold",
            fontsize=16, ha="left", va="bottom")
    ax.text(0.13, 1.015, "Simulation Preliminary", transform=ax.transAxes,
            style="italic", fontsize=11, ha="left", va="bottom")
    ax.text(1.0, 1.015, sample_text, transform=ax.transAxes, fontsize=10,
            ha="right", va="bottom")


def savefig(fig, path: Path, pdf: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def finite_range(arrays: Iterable[np.ndarray], quantiles=(.005, .995)):
    parts = [np.asarray(value)[np.isfinite(value)] for value in arrays
             if np.asarray(value).size]
    parts = [value for value in parts if value.size]
    if not parts:
        return -1.0, 1.0
    values = np.concatenate(parts)
    if values.size == 0:
        return -1.0, 1.0
    low, high = np.quantile(values, quantiles)
    if not high > low:
        delta = max(abs(float(low)) * .1, 1.0)
        return float(low - delta), float(high + delta)
    margin = .04 * (high - low)
    return float(low - margin), float(high + margin)


def hist_values(values, weights, bins):
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    sumw, _ = np.histogram(values[mask], bins=bins, weights=weights[mask])
    sumw2, _ = np.histogram(values[mask], bins=bins, weights=np.square(weights[mask]))
    return sumw.astype(float), sumw2.astype(float)


def plot_weight_diagnostics(name, runtime, output, pdf):
    w = runtime["raw_weights"]
    valid = w[np.isfinite(w) & (w > 0)]
    if valid.size == 0:
        return
    fig, ax = plt.subplots()
    lo, hi = np.quantile(valid, [.0001, .9999])
    if lo > 0 and hi > lo:
        bins = np.geomspace(lo, hi, 80)
        ax.hist(valid, bins=bins, histtype="step", linewidth=1.8, color="#3f90da")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("generator/event weight")
    ax.set_ylabel("events")
    label_axis(ax, name)
    savefig(fig, output / "weights" / f"{name}_weight_distribution.png", pdf)

    order = np.sort(valid)
    mass = np.cumsum(order) / order.sum()
    rows = np.arange(1, order.size + 1) / order.size
    fig, ax = plt.subplots()
    ax.plot(rows, mass, color="#bd1f01", linewidth=2, label="weight Lorenz curve")
    ax.plot([0, 1], [0, 1], "--", color="0.45", label="equal weights")
    ax.set_xlabel("cumulative fraction of rows (lowest weight first)")
    ax.set_ylabel("cumulative generator-weight mass")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left")
    label_axis(ax, name)
    savefig(fig, output / "weights" / f"{name}_weight_lorenz.png", pdf)

    labels, weights = runtime["labels"], runtime["weights"]
    classes = np.unique(labels)
    fig, ax = plt.subplots()
    positions = np.arange(classes.size)
    row_frac = np.asarray([(labels == c).mean() for c in classes])
    mass_frac = np.asarray([weights[labels == c].sum() / weights.sum() for c in classes])
    ax.bar(positions - .18, row_frac, width=.36, label="row fraction", color="#3f90da")
    ax.bar(positions + .18, mass_frac, width=.36, label="weighted fraction", color="#ffa90e")
    ax.set_xticks(positions, [CLASS_NAMES.get(int(c), str(c)) for c in classes])
    ax.set_ylabel("fraction")
    ax.legend()
    label_axis(ax, name)
    savefig(fig, output / "weights" / f"{name}_class_composition.png", pdf)


def plot_quality(name, summary, runtime, output, pdf):
    flags = summary["flags"]
    keys = [key for key, value in flags.items() if value["rows"] > 0]
    if keys:
        rows = np.asarray([flags[key]["row_fraction"] for key in keys])
        mass = np.asarray([flags[key]["weight_fraction"] for key in keys])
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, .38 * len(keys))))
        y = np.arange(len(keys))
        ax.barh(y + .18, rows, height=.34, label="row fraction", color="#3f90da")
        ax.barh(y - .18, mass, height=.34, label="weight fraction", color="#ffa90e")
        ax.set_yticks(y, [key.replace("_", " ") for key in keys])
        ax.set_xscale("log")
        ax.set_xlabel("fraction (log scale)")
        ax.legend()
        label_axis(ax, name)
        savefig(fig, output / "quality" / f"{name}_quality_flags.png", pdf)

    metrics = runtime["event_metrics"]
    names = ["pf_n_active", "pf_sum_pt", "pf_max_pt", "pf_max_abs_eta",
             "pf_max_abs_dxysig", "obj_n_active", "obj_max_abs", "obj_l2",
             "outlier_score"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    labels = runtime["labels"][runtime["event_indices"]]
    weights = runtime["weights"][runtime["event_indices"]]
    for ax, metric_name in zip(axes.flat, names):
        values = metrics[metric_name]
        low, high = finite_range([values])
        bins = np.linspace(low, high, 55)
        for cls in np.unique(labels):
            mask = labels == cls
            ax.hist(values[mask], bins=bins, weights=weights[mask], density=True,
                    histtype="step", linewidth=1.25,
                    color=CLASS_COLORS.get(int(cls)),
                    label=CLASS_NAMES.get(int(cls), str(cls)))
        ax.set_xlabel(metric_name.replace("_", " "))
        ax.set_ylabel("weighted density")
        ax.set_yscale("log")
    axes.flat[0].legend(ncol=2)
    label_axis(axes.flat[0], name)
    fig.tight_layout()
    savefig(fig, output / "quality" / f"{name}_event_summaries.png", pdf)

    matrix_names = ["pf_n_active", "pf_sum_pt", "pf_max_pt", "pf_max_abs_eta",
                    "pf_max_abs_dxysig", "pf_sum_energy", "obj_n_active",
                    "obj_max_abs", "obj_l2"]
    matrix = np.column_stack([np.log1p(np.maximum(metrics[key], 0)) for key in matrix_names])
    corr = np.corrcoef(matrix, rowvar=False)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(matrix_names)), [key.replace("_", " ") for key in matrix_names],
                  rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix_names)), [key.replace("_", " ") for key in matrix_names])
    fig.colorbar(image, ax=ax, label="Pearson correlation of log1p summaries")
    label_axis(ax, name)
    savefig(fig, output / "quality" / f"{name}_event_summary_correlations.png", pdf)


def plot_features(name, runtime, output, pdf):
    for kind, sample, feature_names in [
            ("pf", runtime["pf_sample"], PF_NAMES),
            ("obj", runtime["obj_sample"], OBJ_NAMES)]:
        values, labels, weights = sample["values"], sample["labels"], sample["weights"]
        for feature in range(min(values.shape[1], len(feature_names))):
            x = values[:, feature]
            low, high = finite_range([x])
            bins = np.linspace(low, high, 65)
            fig, ax = plt.subplots()
            for cls in np.unique(labels):
                mask = labels == cls
                ax.hist(x[mask], bins=bins, weights=weights[mask], density=True,
                        histtype="step", linewidth=1.5,
                        color=CLASS_COLORS.get(int(cls)),
                        label=CLASS_NAMES.get(int(cls), str(cls)))
            ax.set_xlabel(feature_names[feature])
            ax.set_ylabel("weighted candidate density")
            ax.set_yscale("log")
            ax.legend(ncol=2)
            label_axis(ax, name)
            savefig(fig, output / "features" / name /
                    f"{kind}_feature_{feature}.png", pdf)

            qcd = labels == 1
            if np.any(qcd):
                fig, ax = plt.subplots()
                ax.hist(x[qcd], bins=bins, density=True, histtype="step",
                        linewidth=1.6, color="0.4", label="QCD unweighted")
                ax.hist(x[qcd], bins=bins, weights=weights[qcd], density=True,
                        histtype="step", linewidth=1.7, color="#ffa90e",
                        label="QCD generator weighted")
                ax.set_xlabel(feature_names[feature])
                ax.set_ylabel("candidate density")
                ax.set_yscale("log")
                ax.legend()
                label_axis(ax, name)
                savefig(fig, output / "features" / name /
                        f"{kind}_feature_{feature}_qcd_weight_effect.png", pdf)

    pf_sample = runtime["pf_sample"]
    for key, xlabel in [
            ("log_pt_fraction", r"model input $\log(p_T/\sum p_T)$"),
            ("log_energy_fraction", r"model input $\log(E/\sum E)$"),
            ("tanh_dxy", r"model input $\tanh(d_{xy})$")]:
        values = pf_sample[key]
        labels, weights = pf_sample["labels"], pf_sample["weights"]
        low, high = finite_range([values])
        bins = np.linspace(low, high, 65)
        fig, ax = plt.subplots()
        for cls in np.unique(labels):
            mask = labels == cls
            ax.hist(values[mask], bins=bins, weights=weights[mask], density=True,
                    histtype="step", linewidth=1.5,
                    color=CLASS_COLORS.get(int(cls)),
                    label=CLASS_NAMES.get(int(cls), str(cls)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("weighted active-candidate density")
        ax.set_yscale("log")
        ax.legend(ncol=2)
        label_axis(ax, name)
        savefig(fig, output / "features" / name / f"pf_preprocessed_{key}.png", pdf)
        qcd = labels == 1
        if np.any(qcd):
            fig, ax = plt.subplots()
            ax.hist(values[qcd], bins=bins, density=True, histtype="step",
                    linewidth=1.6, color="0.4", label="QCD unweighted")
            ax.hist(values[qcd], bins=bins, weights=weights[qcd], density=True,
                    histtype="step", linewidth=1.7, color="#ffa90e",
                    label="QCD generator weighted")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("active-candidate density")
            ax.set_yscale("log")
            ax.legend()
            label_axis(ax, name)
            savefig(fig, output / "features" / name /
                    f"pf_preprocessed_{key}_qcd_weight_effect.png", pdf)


def ratio_plot(first, second, xlabel, output_path, pdf, first_name="train",
               second_name="test", qcd_only=False):
    if qcd_only:
        m1, m2 = first["labels"] == 1, second["labels"] == 1
    else:
        m1 = np.ones(len(first["labels"]), bool)
        m2 = np.ones(len(second["labels"]), bool)
    v1, v2 = first["values"][m1], second["values"][m2]
    w1, w2 = first["weights"][m1], second["weights"][m2]
    low, high = finite_range([v1, v2])
    bins = np.linspace(low, high, 61)
    h1, s1 = hist_values(v1, w1, bins)
    h2, s2 = hist_values(v2, w2, bins)
    norm1, norm2 = max(h1.sum(), 1e-30), max(h2.sum(), 1e-30)
    h1, h2 = h1 / norm1, h2 / norm2
    s1, s2 = s1 / np.square(norm1), s2 / np.square(norm2)
    centers = .5 * (bins[:-1] + bins[1:])
    fig, (ax, ratio_ax) = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1], "hspace": .06})
    ax.stairs(h1, bins, color="#3f90da", linewidth=1.7, label=first_name)
    ax.errorbar(centers, h2, yerr=np.sqrt(s2), fmt="o", markersize=2.7,
                color="#bd1f01", label=second_name)
    ax.set_ylabel("normalized weighted yield")
    ax.set_yscale("log")
    ax.legend()
    label_axis(ax, "QCD only" if qcd_only else "all backgrounds")
    ratio = np.divide(h2, h1, out=np.full_like(h1, np.nan), where=h1 > 0)
    ratio_error = ratio * np.sqrt(
        np.divide(s1, np.square(h1), out=np.zeros_like(s1), where=h1 > 0) +
        np.divide(s2, np.square(h2), out=np.zeros_like(s2), where=h2 > 0))
    ratio_ax.axhline(1, color="0.4", linestyle="--")
    ratio_ax.errorbar(centers, ratio, yerr=ratio_error, fmt="o", markersize=2.5,
                      color="#bd1f01")
    ratio_ax.set_ylim(.45, 1.55)
    ratio_ax.set_ylabel(f"{second_name}/{first_name}")
    ratio_ax.set_xlabel(xlabel)
    savefig(fig, output_path, pdf)


def comparison_sample(runtime, kind, feature):
    if kind == "event":
        indices = runtime["event_indices"]
        return {"values": runtime["event_metrics"][feature],
                "labels": runtime["labels"][indices],
                "weights": runtime["weights"][indices]}
    sample = runtime[f"{kind}_sample"]
    return {"values": sample["values"][:, feature], "labels": sample["labels"],
            "weights": sample["weights"]}


def compare_datasets(name1, run1, name2, run2, output, pdf):
    shift = {}
    for kind, names in [("pf", PF_NAMES), ("obj", OBJ_NAMES)]:
        n_features = min(run1[f"{kind}_sample"]["values"].shape[1],
                         run2[f"{kind}_sample"]["values"].shape[1], len(names))
        for feature in range(n_features):
            left, right = comparison_sample(run1, kind, feature), comparison_sample(run2, kind, feature)
            for scope in ("all", "qcd"):
                qcd = scope == "qcd"
                m1 = left["labels"] == 1 if qcd else np.ones(len(left["labels"]), bool)
                m2 = right["labels"] == 1 if qcd else np.ones(len(right["labels"]), bool)
                metric_key = f"{kind}_{feature}_{scope}"
                shift[metric_key] = weighted_ks(left["values"][m1], right["values"][m2],
                                                left["weights"][m1], right["weights"][m2])
                ratio_plot(left, right, names[feature],
                           output / "comparisons" / f"{name1}_vs_{name2}_{kind}_{feature}_{scope}.png",
                           pdf, name1, name2, qcd_only=qcd)
    for feature, xlabel in [
            ("log_pt_fraction", r"model input $\log(p_T/\sum p_T)$"),
            ("log_energy_fraction", r"model input $\log(E/\sum E)$"),
            ("tanh_dxy", r"model input $\tanh(d_{xy})$")]:
        left = {"values": run1["pf_sample"][feature],
                "labels": run1["pf_sample"]["labels"],
                "weights": run1["pf_sample"]["weights"]}
        right = {"values": run2["pf_sample"][feature],
                 "labels": run2["pf_sample"]["labels"],
                 "weights": run2["pf_sample"]["weights"]}
        for scope in ("all", "qcd"):
            qcd = scope == "qcd"
            m1 = left["labels"] == 1 if qcd else np.ones(len(left["labels"]), bool)
            m2 = right["labels"] == 1 if qcd else np.ones(len(right["labels"]), bool)
            shift[f"pf_preprocessed_{feature}_{scope}"] = weighted_ks(
                left["values"][m1], right["values"][m2],
                left["weights"][m1], right["weights"][m2])
            ratio_plot(left, right, xlabel,
                       output / "comparisons" /
                       f"{name1}_vs_{name2}_pf_preprocessed_{feature}_{scope}.png",
                       pdf, name1, name2, qcd_only=qcd)
    for feature in ["pf_n_active", "pf_sum_pt", "pf_max_pt", "pf_max_abs_eta",
                    "pf_max_abs_dxysig", "pf_sum_energy", "obj_n_active",
                    "obj_max_abs", "obj_l2"]:
        left, right = comparison_sample(run1, "event", feature), comparison_sample(run2, "event", feature)
        for scope in ("all", "qcd"):
            qcd = scope == "qcd"
            m1 = left["labels"] == 1 if qcd else np.ones(len(left["labels"]), bool)
            m2 = right["labels"] == 1 if qcd else np.ones(len(right["labels"]), bool)
            shift[f"event_{feature}_{scope}"] = weighted_ks(
                left["values"][m1], right["values"][m2],
                left["weights"][m1], right["weights"][m2])
            ratio_plot(left, right, feature.replace("_", " "),
                       output / "comparisons" /
                       f"{name1}_vs_{name2}_event_{feature}_{scope}.png",
                       pdf, name1, name2, qcd_only=qcd)
    return shift


def check_cross_dataset_ids(summaries, runtimes, findings):
    names = list(runtimes)
    overlaps = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            left_ids, right_ids = runtimes[left]["event_ids"], runtimes[right]["event_ids"]
            key = f"{left}_vs_{right}"
            if left_ids is None or right_ids is None:
                overlaps[key] = {"verifiable": False, "overlap": None}
                continue
            overlap = int(np.intersect1d(left_ids, right_ids, assume_unique=False).size)
            overlaps[key] = {"verifiable": True, "overlap": overlap}
            if overlap and {left, right} == {"train", "test"}:
                add_finding(findings, "BLOCKER", key, "train_test_overlap",
                            f"Train and test share {overlap:,} event IDs.",
                            "Rebuild disjoint samples before claiming held-out performance.")
    return overlaps


def write_flag_csv(path: Path, summaries: dict[str, Any]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "flag", "rows", "row_fraction", "sum_weights",
                         "weight_fraction"])
        for name, summary in summaries.items():
            for flag, values in summary["flags"].items():
                writer.writerow([name, flag, values["rows"], values["row_fraction"],
                                 values["sum_weights"], values["weight_fraction"]])


def report_markdown(report: dict[str, Any]) -> str:
    findings = sorted(report["findings"],
                      key=lambda item: (SEVERITY_ORDER[item["severity"]], item["dataset"], item["check"]))
    blockers = sum(item["severity"] == "BLOCKER" for item in findings)
    high = sum(item["severity"] == "HIGH" for item in findings)
    verdict = ("DO NOT TRAIN until blockers are fixed" if blockers else
               "TRAIN WITH CAUTION; resolve high-severity provenance/data issues" if high else
               "No blocking data-quality problem was detected by these checks")
    lines = [
        "# HLT data-quality and filtering audit", "", f"**Verdict: {verdict}.**", "",
        "This is a read-only statistical audit. Flagged rows are candidates for inspection, "
        "not an automatically justified physics cut. Any filtering decision must be repeated "
        "on train and held-out samples and documented before retraining.", "",
        "## Dataset summary", "",
        "| sample | rows | PF shape | object shape | weight ESS | hard candidates | review candidates |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for name, summary in report["datasets"].items():
        ess = summary["weights"].get("effective_events")
        ess_text = f"{ess:,.0f}" if ess is not None else "n/a"
        hard = summary["hard_filter_candidate_union"]
        review = summary["review_candidate_union"]
        lines.append(
            f"| {name} | {summary['n_events']:,} | {summary['pf_shape']} | {summary['obj_shape']} "
            f"| {ess_text} | {hard['rows']:,} ({100 * hard['row_fraction']:.3g}%) "
            f"| {review['rows']:,} ({100 * review['row_fraction']:.3g}%) |")
    lines += ["", "## Findings and recommended actions", ""]
    if not findings:
        lines.append("No threshold-triggered findings.")
    for item in findings:
        lines += [f"- **{item['severity']} — {item['dataset']} / {item['check']}:** "
                  f"{item['message']}  Action: {item['action']}", ""]
    lines += [
        "## How this maps to current preprocessing", "",
        "- PF training replaces every NaN/Inf with zero. The model defines padding by raw "
        "`pT == 0` and valid candidates by `pT > 0`; negative pT therefore violates an internal assumption.",
        "- The PF model consumes columns 0–6 as pT, eta, phi, dxy, dxy significance, PF flag, "
        "and PDG ID. PDG IDs outside its fixed vocabulary map to an all-zero category.",
        "- Its six continuous PF channels pass through ordinary, unweighted BatchNorm over active "
        "candidates. Generator weights affect the losses but not those running moments; the report "
        "therefore measures how strongly weighting reshapes each QCD input distribution.",
        "- The AE flattens all candidates' first four object features, replaces NaN/Inf by zero, "
        "and applies generator-weighted train-split mean/std scaling. It does not mask zero object rows.",
        "- Generator weights are expected to remain event-aligned and are used in splitting, AE "
        "scaling/loss, nuisance quantiles, NURD losses, validation, and held-out evaluation.", "",
        "## Files", "",
        "- `report.json`: complete machine-readable statistics and shift metrics.",
        "- `filter_candidates.csv`: count and weighted mass for every flag.",
        "- `flags/<sample>_event_flags.npz`: original row indices for each reason.",
        "- `features/`, `weights/`, `quality/`, `comparisons/`: HEP-style PNG plots "
        "(and PDFs when requested).", "",
        "Important limitation: separate tensor-only weight files cannot prove row-order alignment. "
        "Good-looking weighted histograms are a consistency check, not proof. Event IDs should be "
        "stored in both data and weight payloads for exact verification.", "",
    ]
    return "\n".join(lines)


def parse_dataset(values: list[list[str]]):
    result = []
    seen = set()
    for name, data, weight in values:
        if name in seen:
            raise ValueError(f"duplicate dataset name {name!r}")
        if not name.replace("-", "_").isalnum():
            raise ValueError(f"unsafe dataset name {name!r}")
        seen.add(name)
        result.append((name, Path(data).expanduser().resolve(),
                       None if weight == "-" else Path(weight).expanduser().resolve()))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", nargs=3,
                        metavar=("NAME", "DATA_PT", "WEIGHT_PT"), required=True,
                        help="Repeat per sample; use '-' for unit/no weights.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--event-plot-sample", type=int, default=100_000)
    parser.add_argument("--candidate-plot-sample", type=int, default=300_000)
    parser.add_argument("--outlier-z", type=float, default=12.0)
    parser.add_argument("--min-ess-fraction", type=float, default=0.10)
    parser.add_argument("--max-top1-mass", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pdf", action="store_true", help="Also write PDF copies of plots.")
    parser.add_argument("--no-plots", action="store_true", help="Run checks only (useful for tests).")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.chunk_size, args.event_plot_sample, args.candidate_plot_sample) < 1:
        raise SystemExit("chunk and sample sizes must be positive")
    datasets = parse_dataset(args.dataset)
    for _, data, weights in datasets:
        if not data.is_file():
            raise SystemExit(f"missing data file: {data}")
        if weights is not None and not weights.is_file():
            raise SystemExit(f"missing weight file: {weights}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    hep_style()
    findings: list[Finding] = []
    summaries, runtimes = {}, {}
    for name, data, weights in datasets:
        summaries[name], runtimes[name] = audit_dataset(
            name, data, weights, output, args, findings)
        if not args.no_plots:
            plot_weight_diagnostics(name, runtimes[name], output, args.pdf)
            plot_quality(name, summaries[name], runtimes[name], output, args.pdf)
            plot_features(name, runtimes[name], output, args.pdf)

    id_overlap = check_cross_dataset_ids(summaries, runtimes, findings)
    comparisons = {}
    names = list(runtimes)
    preferred_pairs = [("train", "test"), ("test", "legacy"), ("train", "legacy")]
    pairs = [(left, right) for left, right in preferred_pairs
             if left in runtimes and right in runtimes]
    if not pairs and len(names) >= 2:
        pairs = [(names[0], names[1])]
    if not args.no_plots:
        for left, right in pairs:
            print(f"[plots] comparing {left} vs {right}", flush=True)
            comparisons[f"{left}_vs_{right}"] = compare_datasets(
                left, runtimes[left], right, runtimes[right], output, args.pdf)
    else:
        comparisons = {f"{left}_vs_{right}": {} for left, right in pairs}

    if "train_vs_test" in comparisons:
        qcd_shifts = {key: value for key, value in comparisons["train_vs_test"].items()
                      if key.endswith("_qcd") and value is not None}
        severe = {key: value for key, value in qcd_shifts.items()
                  if np.isfinite(value) and value > .20}
        moderate = {key: value for key, value in qcd_shifts.items()
                    if np.isfinite(value) and .10 < value <= .20}
        if severe:
            worst = sorted(severe.items(), key=lambda item: item[1], reverse=True)[:5]
            add_finding(findings, "HIGH", "train_vs_test", "weighted_qcd_shift",
                        "Large weighted QCD shifts (KS > 0.20): " +
                        ", ".join(f"{key}={value:.3f}" for key, value in worst) + ".",
                        "Investigate sample production/phase-space coverage before interpreting closure transfer.")
        elif moderate:
            worst = sorted(moderate.items(), key=lambda item: item[1], reverse=True)[:5]
            add_finding(findings, "WARNING", "train_vs_test", "weighted_qcd_shift",
                        "Moderate weighted QCD shifts (KS > 0.10): " +
                        ", ".join(f"{key}={value:.3f}" for key, value in worst) + ".",
                        "Inspect plots and confirm differences are compatible with finite weighted statistics.")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "hostname": platform.node(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "configuration": vars(args), "datasets": summaries,
        "event_id_overlap": id_overlap, "distribution_shifts_weighted_ks": comparisons,
        "findings": [finding.as_dict() for finding in sorted(
            findings, key=lambda item: (SEVERITY_ORDER[item.severity], item.dataset, item.check))],
    }
    with (output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(report), handle, indent=2, sort_keys=True)
    write_flag_csv(output / "filter_candidates.csv", summaries)
    (output / "report.md").write_text(report_markdown(_json_value(report)), encoding="utf-8")
    blockers = sum(item.severity == "BLOCKER" for item in findings)
    high = sum(item.severity == "HIGH" for item in findings)
    print(f"DATA AUDIT DONE: {output}", flush=True)
    print(f"Findings: {blockers} blocker(s), {high} high-severity", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
