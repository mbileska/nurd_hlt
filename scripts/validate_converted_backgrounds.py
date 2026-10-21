#!/usr/bin/env python3
"""Validate converted train/test tensors and their aligned physics weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.hlt_data import load_hlt_sample
from utils.hlt_weights import EVENT_ID_KEYS, WEIGHT_KEYS, load_generator_weights


LABEL_BY_SAMPLE = torch.tensor([1, 3, 0, 2], dtype=torch.int64)


def parse_integer_ranges(text: str):
    output = set()
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            low, high = map(int, piece.split("-", 1))
            if high < low:
                raise ValueError(f"Invalid integer range {piece!r}.")
            output.update(range(low, high + 1))
        else:
            output.add(int(piece))
    return output


def _finite_in_chunks(tensor: torch.Tensor, rows: int = 20_000) -> bool:
    for start in range(0, tensor.shape[0], rows):
        if not torch.isfinite(tensor[start:start + rows]).all():
            return False
    return True


def _all_in_chunks(tensor: torch.Tensor, predicate, rows: int = 20_000) -> bool:
    for start in range(0, tensor.shape[0], rows):
        if not predicate(tensor[start:start + rows]):
            return False
    return True


def _sort_event_ids(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.int64)
    order = np.lexsort(tuple(value[:, column] for column in reversed(range(4))))
    return value[order]


def validate_split(name: str, sample_path: Path, weight_path: Path):
    sample = load_hlt_sample(sample_path)
    for key in ("pf", "obj", "label", "event_id", "background_sample",
                "background_category"):
        if key not in sample:
            raise KeyError(f"{name} sample is missing {key!r}.")
    n_events = int(sample["label"].numel())
    checks = {
        "pf shape": tuple(sample["pf"].shape) == (n_events, 400, 7),
        "obj shape": tuple(sample["obj"].shape) == (n_events, 23, 4),
        "event ID shape": tuple(sample["event_id"].shape) == (n_events, 4),
        "PF values finite": _finite_in_chunks(sample["pf"]),
        "object values finite": _finite_in_chunks(sample["obj"]),
        "labels follow sample map": torch.equal(
            sample["label"].long(),
            LABEL_BY_SAMPLE[sample["background_sample"].long()]),
        "PF is_pf is binary": _all_in_chunks(
            sample["pf"],
            lambda value: bool(torch.all(
                (value[:, :, 5] == 0) | (value[:, :, 5] == 1)))),
        "PF padding is zero": _all_in_chunks(
            sample["pf"],
            lambda value: bool(torch.all(
                value[value[:, :, 0] <= 0] == 0))),
        "object padding is zero": _all_in_chunks(
            sample["obj"],
            lambda value: bool(torch.all(
                value[value[:, :, 0] <= 0] == 0))),
    }
    weights, weight_metadata = load_generator_weights(
        str(weight_path), sample["label"], sample=sample,
        weight_scope="all_events")
    checks.update({
        "weights aligned by event ID": weight_metadata["alignment_verified"],
        "weights finite": bool(torch.isfinite(weights).all()),
        "weights nonnegative": bool((weights >= 0).all()),
        "weights have positive mass": float(weights.sum()) > 0.0,
    })
    return sample, weights, weight_metadata, checks


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate a converted NURD background data directory.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--full-weights", type=Path)
    parser.add_argument("--expected-files", type=int, default=-1)
    parser.add_argument("--expected-events", type=int, default=-1)
    parser.add_argument("--expected-categories", default="0-23,27-36")
    parser.add_argument(
        "--allow-partial-classes", action="store_true",
        help="Permit a smoke sample that does not contain all four backgrounds.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    root = args.input_dir.expanduser().resolve()
    report_path = root / "conversion_report.json"
    conversion = json.loads(report_path.read_text(encoding="utf-8"))
    train, train_weights, train_meta, train_checks = validate_split(
        "train", root / "train/sample.json", root / "weight_train.npz")
    test, test_weights, test_meta, test_checks = validate_split(
        "test", root / "test/sample.json", root / "weight_test.npz")

    combined_ids = np.concatenate([
        train["event_id"].numpy(), test["event_id"].numpy()], axis=0)
    sorted_combined_ids = _sort_event_ids(combined_ids)
    categories = torch.cat([
        train["background_category"], test["background_category"]]).numpy()
    expected_categories = parse_integer_ranges(args.expected_categories)
    observed_categories = set(map(int, np.unique(categories)))
    overall_checks = {
        "conversion reported OK": conversion.get("ok") is True,
        "train/test event IDs unique": bool(
            np.all(np.any(sorted_combined_ids[1:] != sorted_combined_ids[:-1], axis=1))),
        "all expected categories populated": expected_categories <= observed_categories,
        "no unknown categories": observed_categories <= expected_categories,
        "both splits contain expected classes": (
            set(train["label"].unique().tolist())
            == set(test["label"].unique().tolist())
            and (args.allow_partial_classes
                 or set(train["label"].unique().tolist()) == set(range(4)))),
    }
    total_events = int(train["label"].numel() + test["label"].numel())
    if args.expected_events > 0:
        overall_checks["expected event count"] = total_events == args.expected_events
    if args.expected_files > 0:
        overall_checks["expected file count"] = conversion.get("files") == args.expected_files

    full_weight_comparison = None
    if args.full_weights is not None:
        with np.load(args.full_weights.expanduser().resolve(), allow_pickle=False) as archive:
            event_key = next((key for key in EVENT_ID_KEYS if key in archive), None)
            weight_key = next((key for key in WEIGHT_KEYS if key in archive), None)
            if event_key is None or weight_key is None:
                raise KeyError("Full weight payload lacks event IDs or weights.")
            full_ids = np.asarray(archive[event_key], dtype=np.int64)
            full_weights = np.asarray(
                archive[weight_key], dtype=np.float64).reshape(-1)
        full_weight_comparison = {
            "events": int(full_weights.size),
            "sum": float(full_weights.sum()),
            "split_sum": float(train_weights.sum() + test_weights.sum()),
        }
        overall_checks.update({
            "split IDs equal full weight IDs": (
                full_ids.shape == combined_ids.shape
                and np.array_equal(_sort_event_ids(full_ids), sorted_combined_ids)),
            "split weight mass equals full payload": np.isclose(
                full_weight_comparison["sum"],
                full_weight_comparison["split_sum"], rtol=1e-12, atol=1e-12),
        })

    all_checks = {
        **{f"train: {key}": value for key, value in train_checks.items()},
        **{f"test: {key}": value for key, value in test_checks.items()},
        **overall_checks,
    }
    report = {
        "ok": all(all_checks.values()),
        "events": total_events,
        "train_events": int(train["label"].numel()),
        "test_events": int(test["label"].numel()),
        "observed_categories": sorted(observed_categories),
        "missing_categories": sorted(expected_categories - observed_categories),
        "checks": all_checks,
        "train_weight_metadata": train_meta,
        "test_weight_metadata": test_meta,
        "full_weight_comparison": full_weight_comparison,
    }
    output = args.output or (root / "validation_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, passed in all_checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"Wrote {output}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
