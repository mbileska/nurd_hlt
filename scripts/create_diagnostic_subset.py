#!/usr/bin/env python3
"""Create a deterministic, class-stratified HLT sample for diagnostics.

The normal training files are never modified.  Every event-aligned tensor in
the input mapping is sliced with the same indices, and the generator-weight
vector is sliced identically so row alignment is preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.hlt_weights import EVENT_ID_KEYS, WEIGHT_KEYS, tensor_sha256


def extract_weight_vector(payload) -> torch.Tensor:
    """Return the unique event-weight vector from a supported payload."""
    if isinstance(payload, Mapping):
        for key in WEIGHT_KEYS:
            if key in payload:
                return torch.as_tensor(payload[key]).reshape(-1)
        candidates = [
            value for key, value in payload.items()
            if key not in EVENT_ID_KEYS
            and (torch.is_tensor(value) or isinstance(value, np.ndarray))
            and torch.as_tensor(value).ndim >= 1
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Weight mapping has no standard weight key and is ambiguous.")
        return torch.as_tensor(candidates[0]).reshape(-1)
    return torch.as_tensor(payload).reshape(-1)


def stratified_subset_indices(
    labels: torch.Tensor,
    maximum_events: int,
    seed: int,
) -> torch.Tensor:
    """Sample without replacement while retaining every class proportion."""
    labels = torch.as_tensor(labels).long().reshape(-1).cpu()
    n_events = labels.numel()
    if maximum_events <= 0 or maximum_events >= n_events:
        return torch.arange(n_events, dtype=torch.long)
    classes, counts = torch.unique(labels, sorted=True, return_counts=True)
    if maximum_events < classes.numel():
        raise ValueError(
            "maximum_events must be at least the number of classes.")

    exact = counts.double() * float(maximum_events) / float(n_events)
    allocation = torch.floor(exact).long().clamp(min=1)
    while int(allocation.sum()) > maximum_events:
        candidates = torch.nonzero(allocation > 1, as_tuple=False).reshape(-1)
        if not candidates.numel():
            raise RuntimeError("Could not form a non-empty stratified subset.")
        remove = candidates[torch.argmin(exact[candidates] - allocation[candidates])]
        allocation[remove] -= 1
    remainders = exact - torch.floor(exact)
    while int(allocation.sum()) < maximum_events:
        available = torch.nonzero(allocation < counts, as_tuple=False).reshape(-1)
        add = available[torch.argmax(remainders[available])]
        allocation[add] += 1
        remainders[add] = -1.0
        if (remainders[available] < 0).all():
            remainders = exact - allocation.double()

    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for class_value, class_count in zip(classes.tolist(), allocation.tolist()):
        positions = torch.nonzero(labels == int(class_value), as_tuple=False).reshape(-1)
        order = torch.randperm(positions.numel(), generator=generator)
        selected.append(positions[order[:int(class_count)]])
    # Sorting retains original row order; training performs its own seeded split
    # and shuffle.  Most importantly, data and weights use exactly this index.
    return torch.sort(torch.cat(selected))[0]


def subset_event_mapping(sample: Mapping, indices: torch.Tensor) -> dict:
    labels = torch.as_tensor(sample["label"])
    n_events = int(labels.shape[0])
    output = {}
    for key, value in sample.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == n_events:
            output[key] = value[indices]
        elif isinstance(value, np.ndarray) and value.ndim and value.shape[0] == n_events:
            output[key] = value[indices.numpy()]
        else:
            output[key] = value
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a deterministic class-stratified diagnostic sample.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--output-weights", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--max-events", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data_path = Path(args.data).resolve()
    weight_path = Path(args.weights).resolve()
    output_data = Path(args.output_data).resolve()
    output_weights = Path(args.output_weights).resolve()
    metadata_path = Path(args.metadata).resolve()
    for output in (output_data, output_weights, metadata_path):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite diagnostic input: {output}")

    sample = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(sample, Mapping):
        raise TypeError("The HLT training sample must be a mapping.")
    for key in ("pf", "obj", "label"):
        if key not in sample:
            raise KeyError(f"Training sample is missing required key {key!r}.")
    labels = torch.as_tensor(sample["label"]).long().reshape(-1)
    raw_weights = extract_weight_vector(torch.load(
        weight_path, map_location="cpu", weights_only=False))
    if raw_weights.numel() != labels.numel():
        raise ValueError(
            f"Weight length {raw_weights.numel()} does not match data length "
            f"{labels.numel()}.")

    indices = stratified_subset_indices(labels, args.max_events, args.seed)
    selected_sample = subset_event_mapping(sample, indices)
    selected_weights = raw_weights[indices]
    torch.save(selected_sample, output_data)
    torch.save(selected_weights, output_weights)

    classes = sorted(int(value) for value in labels.unique().tolist())
    metadata = {
        "protocol": "deterministic_class_stratified_without_replacement",
        "source_data": str(data_path),
        "source_weights": str(weight_path),
        "seed": int(args.seed),
        "source_events": int(labels.numel()),
        "selected_events": int(indices.numel()),
        "class_counts_source": {
            str(label): int((labels == label).sum()) for label in classes
        },
        "class_counts_selected": {
            str(label): int((labels[indices] == label).sum()) for label in classes
        },
        "selected_indices_sha256": tensor_sha256(indices),
        "selected_weights_sha256": tensor_sha256(selected_weights),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
