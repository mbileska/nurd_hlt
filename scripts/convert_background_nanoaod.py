#!/usr/bin/env python3
"""Convert the validated background NanoAOD campaign into NURD tensors.

The converter is intentionally streaming and writes NumPy ``open_memmap``
arrays described by a small JSON manifest.  It therefore never materializes
the roughly 30 GB production tensor set in RAM and does not create a second
temporary copy.  Event IDs from the ROOT files are checked row-for-row against
the all-background physics-weight payload before any row is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.hlt_data import MANIFEST_FORMAT


PF_SIZE = 400
OBJECT_SLOTS = {
    "ScoutingPFJet": (0, 10, 0),
    "ScoutingMuonVtx": (10, 4, 1),
    "ScoutingElectron": (14, 4, 2),
    "ScoutingPhoton": (18, 4, 3),
}
SAMPLE_TO_LABEL = np.asarray([1, 3, 0, 2], dtype=np.int64)
SAMPLE_NAMES = {0: "QCD", 1: "W+jets", 2: "Z+jets", 3: "ttbar"}
EVENT_ID_KEYS = ("event_id", "eventid", "event_ids", "eventids")
SAMPLE_KEYS = ("background_sample", "samples", "sample_id", "sample")
CATEGORY_KEYS = (
    "background_category", "categories", "category_id", "category")

ROOT_BRANCHES = [
    "run", "luminosityBlock", "event",
    "Background_sample", "Background_category",
    "ScoutingParticle_pt", "ScoutingParticle_eta", "ScoutingParticle_phi",
    "ScoutingParticle_dxy", "ScoutingParticle_dxysig",
    "ScoutingParticle_pdgId",
    "ScoutingPFJet_pt", "ScoutingPFJet_eta", "ScoutingPFJet_phi",
    "ScoutingMuonVtx_pt", "ScoutingMuonVtx_eta", "ScoutingMuonVtx_phi",
    "ScoutingElectron_pt", "ScoutingElectron_eta", "ScoutingElectron_phi",
    "ScoutingPhoton_pt", "ScoutingPhoton_eta", "ScoutingPhoton_phi",
    "ScoutingMET_pt", "ScoutingMET_phi",
]


def _find_key(payload: Mapping, candidates: Sequence[str], description: str) -> str:
    key = next((candidate for candidate in candidates if candidate in payload), None)
    if key is None:
        raise KeyError(
            f"Weight payload has no {description}; tried {list(candidates)}.")
    return key


def stratified_test_mask(
    categories: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """Return a deterministic, fine-category-stratified test mask."""
    categories = np.asarray(categories).reshape(-1)
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one.")
    rng = np.random.default_rng(int(seed))
    selected = np.zeros(categories.size, dtype=bool)
    for category in np.unique(categories):
        positions = np.flatnonzero(categories == category)
        if positions.size < 2:
            raise ValueError(
                f"Category {category} has only {positions.size} event; cannot put "
                "events in both train and test samples.")
        n_test = int(round(positions.size * float(test_fraction)))
        n_test = min(max(n_test, 1), positions.size - 1)
        selected[rng.permutation(positions)[:n_test]] = True
    return selected


def labels_from_samples(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.int64).reshape(-1)
    if samples.size and (samples.min() < 0 or samples.max() >= SAMPLE_TO_LABEL.size):
        raise ValueError(f"Unsupported Background_sample values: {np.unique(samples)}")
    return SAMPLE_TO_LABEL[samples]


def split_weight_payload(
    payload: Mapping[str, np.ndarray],
    mask: np.ndarray,
    n_events: int,
) -> Dict[str, np.ndarray]:
    """Slice every event-aligned NPZ field while retaining scalar metadata."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != int(n_events):
        raise ValueError("Weight split mask does not match the selected event count.")
    output: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if array.ndim and array.shape[0] >= int(n_events):
            if array.shape[0] != int(n_events):
                raise ValueError(
                    f"Event-aligned weight field {key!r} has {array.shape[0]} rows, "
                    f"expected {n_events}.")
            output[key] = array[mask]
        else:
            output[key] = array
    output["weight_scope"] = np.asarray("all_events")
    return output


def _event_id_sha256(event_id: np.ndarray) -> str:
    value = np.ascontiguousarray(event_id, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _open_split_arrays(directory: Path, n_events: int):
    directory.mkdir(parents=True, exist_ok=False)
    definitions = {
        "pf": ((n_events, PF_SIZE, 7), np.float32),
        "obj": ((n_events, 23, 4), np.float32),
        "label": ((n_events,), np.int64),
        "event_id": ((n_events, 4), np.int64),
        "background_sample": ((n_events,), np.int8),
        "background_category": ((n_events,), np.int16),
    }
    return {
        key: np.lib.format.open_memmap(
            directory / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
        for key, (shape, dtype) in definitions.items()
    }


def _write_manifest(directory: Path, arrays, metadata: Mapping[str, object]):
    manifest = {
        "format": MANIFEST_FORMAT,
        "arrays": {
            key: {
                "path": f"{key}.npy",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in arrays.items()
        },
        "metadata": dict(metadata),
    }
    path = directory / "sample.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _padded_sorted(ak, arrays, prefix: str, fields: Sequence[str], size: int):
    pt = arrays[f"{prefix}_pt"]
    order = ak.argsort(pt, axis=1, ascending=False, stable=True)
    columns = []
    for field in fields:
        values = arrays[f"{prefix}_{field}"][order]
        padded = ak.fill_none(ak.pad_none(values, size, axis=1, clip=True), 0)
        columns.append(ak.to_numpy(padded).astype(np.float32, copy=False))
    return columns, ak.to_numpy(ak.num(pt, axis=1)).astype(np.int64, copy=False)


def _convert_pf(ak, arrays):
    columns, counts = _padded_sorted(
        ak, arrays, "ScoutingParticle",
        ("pt", "eta", "phi", "dxy", "dxysig", "pdgId"), PF_SIZE)
    pf = np.zeros((counts.size, PF_SIZE, 7), dtype=np.float32)
    for column, values in enumerate(columns[:5]):
        pf[:, :, column] = values
    active = pf[:, :, 0] > 0.0
    pf[:, :, 5] = active.astype(np.float32)
    raw_pdg = np.rint(columns[5]).astype(np.int64, copy=False)
    allowed = np.isin(
        raw_pdg, np.asarray([-211, -13, -11, 0, 1, 2, 11, 13, 22, 130, 211]))
    invalid = active & ~allowed
    raw_pdg[invalid] = 0
    pf[:, :, 6] = raw_pdg.astype(np.float32)
    pf[~active] = 0.0
    np.nan_to_num(pf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return pf, counts, int(invalid.sum())


def _convert_objects(ak, arrays):
    n_events = len(arrays["event"])
    obj = np.zeros((n_events, 23, 4), dtype=np.float32)
    counts = {}
    for prefix, (start, size, object_type) in OBJECT_SLOTS.items():
        columns, collection_counts = _padded_sorted(
            ak, arrays, prefix, ("pt", "eta", "phi"), size)
        stop = start + size
        obj[:, start:stop, :3] = np.stack(columns, axis=-1)
        active = obj[:, start:stop, 0] > 0.0
        obj[:, start:stop, 3] = active.astype(np.float32) * object_type
        obj[:, start:stop][~active] = 0.0
        counts[prefix] = collection_counts

    for field_index, field in enumerate(("pt", "phi")):
        values = arrays[f"ScoutingMET_{field}"]
        if values.ndim > 1:
            values = ak.fill_none(ak.firsts(values), 0)
        values = ak.to_numpy(values).astype(np.float32, copy=False).reshape(-1)
        obj[:, 22, 0 if field_index == 0 else 2] = values
    met_active = obj[:, 22, 0] > 0.0
    obj[met_active, 22, 3] = 4.0
    obj[~met_active, 22] = 0.0
    np.nan_to_num(obj, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return obj, counts


def _root_event_ids(ak, arrays):
    return np.column_stack([
        ak.to_numpy(arrays["Background_sample"]).astype(np.int64),
        ak.to_numpy(arrays["run"]).astype(np.int64),
        ak.to_numpy(arrays["luminosityBlock"]).astype(np.int64),
        ak.to_numpy(arrays["event"]).astype(np.int64),
    ])


def _flush_arrays(arrays):
    for value in arrays.values():
        value.flush()


def _load_weights(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    event_key = _find_key(payload, EVENT_ID_KEYS, "composite event-ID array")
    sample_key = _find_key(payload, SAMPLE_KEYS, "background-sample array")
    category_key = _find_key(payload, CATEGORY_KEYS, "background-category array")
    event_ids = np.asarray(payload[event_key], dtype=np.int64)
    samples = np.asarray(payload[sample_key], dtype=np.int64).reshape(-1)
    categories = np.asarray(payload[category_key], dtype=np.int64).reshape(-1)
    if event_ids.ndim != 2 or event_ids.shape[1] != 4:
        raise ValueError(
            f"Composite event IDs must have shape [N,4], got {event_ids.shape}.")
    if not (event_ids.shape[0] == samples.size == categories.size):
        raise ValueError("Weight event IDs, samples, and categories do not align.")
    scope = payload.get("weight_scope")
    if scope is None or str(np.asarray(scope).reshape(()).item()) != "all_events":
        raise ValueError("Production weight payload must declare all_events scope.")
    return payload, event_ids, samples, categories


def _selected_event_count(uproot, inputs: Sequence[str]) -> int:
    total = 0
    for index, path in enumerate(inputs, 1):
        with uproot.open(path) as root_file:
            total += int(root_file["Events"].num_entries)
        if index % 100 == 0 or index == len(inputs):
            print(f"Counted entries in {index}/{len(inputs)} files", flush=True)
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        description="Stream background NanoAOD into aligned NURD input arrays.")
    parser.add_argument("inputs", nargs="+", help="ROOT files or @filelist")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-size", default="50 MB")
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument("--expected-files", type=int, default=-1)
    parser.add_argument("--expected-events", type=int, default=-1)
    args = parser.parse_args(argv)

    try:
        import awkward as ak
        import uproot
    except ImportError as error:
        raise RuntimeError(
            "Conversion requires awkward and uproot (available in the CMSSW "
            "runtime used on login.hep.wisc.edu).") from error

    inputs = [str(Path(item).expanduser()) for item in args.inputs]
    if args.max_files > 0:
        inputs = inputs[:args.max_files]
    if not inputs:
        raise ValueError("No input ROOT files were selected.")
    if args.expected_files > 0 and len(inputs) != args.expected_files:
        raise ValueError(
            f"Selected {len(inputs)} files, expected {args.expected_files}.")
    for path in inputs:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".conversion_incomplete").write_text("incomplete\n")

    payload, weight_ids_full, samples_full, categories_full = _load_weights(
        args.weights.expanduser().resolve())
    if args.max_files > 0:
        n_events = _selected_event_count(uproot, inputs)
    else:
        n_events = int(weight_ids_full.shape[0])
    if args.expected_events > 0 and n_events != args.expected_events:
        raise ValueError(f"Selected {n_events} events, expected {args.expected_events}.")
    if n_events > weight_ids_full.shape[0]:
        raise ValueError("ROOT selection contains more events than the weight payload.")

    payload = {
        key: (np.asarray(value)[:n_events] if np.asarray(value).ndim
              and np.asarray(value).shape[0] == weight_ids_full.shape[0]
              else np.asarray(value))
        for key, value in payload.items()
    }
    weight_ids = weight_ids_full[:n_events]
    samples = samples_full[:n_events]
    categories = categories_full[:n_events]
    test_mask = stratified_test_mask(categories, args.test_fraction, args.seed)
    split_masks = {"train": ~test_mask, "test": test_mask}
    split_arrays = {
        name: _open_split_arrays(output_dir / name, int(mask.sum()))
        for name, mask in split_masks.items()
    }
    positions = {"train": 0, "test": 0}

    stats = {
        "pf_events_truncated": 0,
        "pf_candidates_dropped": 0,
        "pf_max_candidates": 0,
        "invalid_active_pdg_ids_replaced": 0,
        "object_events_truncated": {key: 0 for key in OBJECT_SLOTS},
        "object_candidates_dropped": {key: 0 for key in OBJECT_SLOTS},
        "object_max_candidates": {key: 0 for key in OBJECT_SLOTS},
    }
    offset = 0
    tree_paths = [f"{path}:Events" for path in inputs]
    for arrays in uproot.iterate(
            tree_paths, expressions=ROOT_BRANCHES, step_size=args.step_size,
            library="ak", how=dict):
        chunk_size = len(arrays["event"])
        stop = offset + chunk_size
        if stop > n_events:
            raise ValueError("ROOT iteration exceeded the allocated event count.")
        root_ids = _root_event_ids(ak, arrays)
        if not np.array_equal(root_ids, weight_ids[offset:stop]):
            mismatch = np.flatnonzero(np.any(root_ids != weight_ids[offset:stop], axis=1))
            first = int(mismatch[0]) if mismatch.size else -1
            raise ValueError(
                "ROOT/weight event-ID mismatch at global row "
                f"{offset + first}: ROOT={root_ids[first].tolist()} "
                f"weights={weight_ids[offset + first].tolist()}.")
        root_samples = ak.to_numpy(arrays["Background_sample"]).astype(np.int64)
        root_categories = ak.to_numpy(arrays["Background_category"]).astype(np.int64)
        if not np.array_equal(root_samples, samples[offset:stop]):
            raise ValueError(f"Background_sample mismatch in rows {offset}:{stop}.")
        if not np.array_equal(root_categories, categories[offset:stop]):
            raise ValueError(f"Background_category mismatch in rows {offset}:{stop}.")

        pf, pf_counts, invalid_pdg = _convert_pf(ak, arrays)
        obj, object_counts = _convert_objects(ak, arrays)
        labels = labels_from_samples(root_samples)
        if not np.isfinite(pf).all() or not np.isfinite(obj).all():
            raise ValueError(f"Non-finite converted tensor in rows {offset}:{stop}.")

        stats["pf_events_truncated"] += int((pf_counts > PF_SIZE).sum())
        stats["pf_candidates_dropped"] += int(np.maximum(pf_counts - PF_SIZE, 0).sum())
        stats["pf_max_candidates"] = max(stats["pf_max_candidates"], int(pf_counts.max()))
        stats["invalid_active_pdg_ids_replaced"] += invalid_pdg
        for prefix, counts in object_counts.items():
            size = OBJECT_SLOTS[prefix][1]
            stats["object_events_truncated"][prefix] += int((counts > size).sum())
            stats["object_candidates_dropped"][prefix] += int(
                np.maximum(counts - size, 0).sum())
            stats["object_max_candidates"][prefix] = max(
                stats["object_max_candidates"][prefix], int(counts.max()))

        chunk_values = {
            "pf": pf,
            "obj": obj,
            "label": labels,
            "event_id": root_ids,
            "background_sample": root_samples.astype(np.int8),
            "background_category": root_categories.astype(np.int16),
        }
        for name, global_mask in split_masks.items():
            local_mask = global_mask[offset:stop]
            count = int(local_mask.sum())
            destination = slice(positions[name], positions[name] + count)
            for key, value in chunk_values.items():
                split_arrays[name][key][destination] = value[local_mask]
            positions[name] += count
        offset = stop
        if offset % 100_000 < chunk_size or offset == n_events:
            print(f"Converted {offset}/{n_events} events", flush=True)

    if offset != n_events:
        raise ValueError(f"ROOT iteration produced {offset} events, expected {n_events}.")
    for name, arrays in split_arrays.items():
        expected = int(split_masks[name].sum())
        if positions[name] != expected:
            raise RuntimeError(f"{name} writer filled {positions[name]}/{expected} rows.")
        _flush_arrays(arrays)

    manifests = {}
    split_reports = {}
    for name, mask in split_masks.items():
        ids = split_arrays[name]["event_id"]
        split_metadata = {
            "split": name,
            "events": int(mask.sum()),
            "seed": int(args.seed),
            "test_fraction": float(args.test_fraction),
            "source_files": len(inputs),
            "event_id_sha256": _event_id_sha256(ids),
            "pf_contract": (
                "400 highest-pT ScoutingParticle rows: "
                "[pt,eta,phi,dxy,dxysig,is_pf,pdgId]"),
            "object_contract": (
                "10 PF jets, 4 vertex muons, 4 electrons, 4 photons, 1 MET; "
                "each [pt,eta,phi,type]"),
        }
        manifests[name] = str(_write_manifest(
            output_dir / name, split_arrays[name], split_metadata))
        split_reports[name] = {
            "events": int(mask.sum()),
            "labels": {
                str(label): int((labels_from_samples(samples[mask]) == label).sum())
                for label in range(4)
            },
            "categories": {
                str(category): int((categories[mask] == category).sum())
                for category in np.unique(categories[mask])
            },
            "event_id_sha256": split_metadata["event_id_sha256"],
        }

    np.savez_compressed(
        output_dir / "weight_train.npz",
        **split_weight_payload(payload, split_masks["train"], n_events))
    np.savez_compressed(
        output_dir / "weight_test.npz",
        **split_weight_payload(payload, split_masks["test"], n_events))

    report = {
        "ok": True,
        "format": MANIFEST_FORMAT,
        "files": len(inputs),
        "events": n_events,
        "weight_source": str(args.weights.expanduser().resolve()),
        "split_protocol": "fine_category_stratified_without_replacement",
        "sample_to_nurd_label": {
            str(sample): {"name": SAMPLE_NAMES[sample], "label": int(SAMPLE_TO_LABEL[sample])}
            for sample in SAMPLE_NAMES
        },
        "manifests": manifests,
        "splits": split_reports,
        "truncation": stats,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / ".conversion_incomplete").unlink()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
