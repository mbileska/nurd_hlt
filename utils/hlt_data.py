"""Load HLT samples from legacy PyTorch files or NumPy manifests.

The manifest format keeps the large event tensors in memory-mappable ``.npy``
files.  This avoids a second, monolithic copy of a production sample while
preserving the mapping interface used by the training and evaluation code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


MANIFEST_FORMAT = "nurd_hlt_numpy_v1"
REQUIRED_SAMPLE_KEYS = ("pf", "obj", "label")


def is_hlt_manifest(path) -> bool:
    """Return whether *path* looks like a JSON HLT sample manifest."""
    return Path(path).suffix.lower() == ".json"


def load_hlt_sample(path, *, mmap_mode: str = "c") -> Mapping[str, Any]:
    """Load a legacy ``.pt`` mapping or a memory-mapped NumPy manifest.

    ``mmap_mode='c'`` uses copy-on-write mappings.  PyTorch can safely wrap
    those writable views without reading the complete arrays into memory, and
    accidental in-process writes never modify the source files.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not is_hlt_manifest(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"HLT sample {path} must contain a mapping, got "
                f"{type(payload).__name__}.")
        return payload

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"Unsupported HLT manifest format {manifest.get('format')!r} in "
            f"{path}; expected {MANIFEST_FORMAT!r}.")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError(f"HLT manifest {path} has no array mapping.")

    output: Dict[str, Any] = {}
    for key, specification in arrays.items():
        if isinstance(specification, str):
            relative_path = specification
            expected_shape = None
            expected_dtype = None
        elif isinstance(specification, Mapping):
            relative_path = specification.get("path")
            expected_shape = specification.get("shape")
            expected_dtype = specification.get("dtype")
        else:
            raise TypeError(f"Invalid array specification for {key!r} in {path}.")
        if not relative_path:
            raise ValueError(f"Array {key!r} in {path} has no path.")
        array_path = (path.parent / str(relative_path)).resolve()
        if not array_path.is_file():
            raise FileNotFoundError(array_path)
        array = np.load(array_path, mmap_mode=mmap_mode, allow_pickle=False)
        if expected_shape is not None and list(array.shape) != list(expected_shape):
            raise ValueError(
                f"Array {key!r} has shape {list(array.shape)}, expected "
                f"{list(expected_shape)}.")
        if expected_dtype is not None and str(array.dtype) != str(expected_dtype):
            raise ValueError(
                f"Array {key!r} has dtype {array.dtype}, expected "
                f"{expected_dtype}.")
        output[str(key)] = torch.from_numpy(array)

    missing = [key for key in REQUIRED_SAMPLE_KEYS if key not in output]
    if missing:
        raise KeyError(f"HLT manifest {path} is missing required arrays {missing}.")
    n_events = int(output["label"].shape[0])
    for key, value in output.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] != n_events:
            raise ValueError(
                f"Array {key!r} has {value.shape[0]} rows, expected {n_events}.")

    metadata = manifest.get("metadata", {})
    if isinstance(metadata, Mapping):
        output["manifest_metadata"] = dict(metadata)
    output["manifest_path"] = str(path)
    return output


def sample_path_exists(path) -> bool:
    """Manifest-aware equivalent of ``Path(path).is_file()``."""
    return Path(path).expanduser().is_file()
