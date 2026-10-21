import json

import numpy as np
import pytest
import torch

from utils.hlt_data import MANIFEST_FORMAT, load_hlt_sample


def test_numpy_manifest_loads_memory_mapped_tensors(tmp_path):
    arrays = {
        "pf": np.arange(3 * 4 * 7, dtype=np.float32).reshape(3, 4, 7),
        "obj": np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
        "label": np.asarray([1, 3, 0], dtype=np.int64),
        "event_id": np.arange(12, dtype=np.int64).reshape(3, 4),
    }
    specifications = {}
    for key, value in arrays.items():
        np.save(tmp_path / f"{key}.npy", value)
        specifications[key] = {
            "path": f"{key}.npy",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps({
        "format": MANIFEST_FORMAT,
        "arrays": specifications,
        "metadata": {"split": "test"},
    }))

    loaded = load_hlt_sample(manifest)

    assert torch.equal(loaded["label"], torch.tensor([1, 3, 0]))
    assert loaded["pf"].shape == (3, 4, 7)
    assert loaded["manifest_metadata"]["split"] == "test"
    loaded["pf"][0, 0, 0] = -99
    assert np.load(tmp_path / "pf.npy")[0, 0, 0] == 0


def test_numpy_manifest_rejects_shape_mismatch(tmp_path):
    for key, shape, dtype in (
        ("pf", (2, 4, 7), np.float32),
        ("obj", (2, 2, 4), np.float32),
        ("label", (2,), np.int64),
    ):
        np.save(tmp_path / f"{key}.npy", np.zeros(shape, dtype=dtype))
    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps({
        "format": MANIFEST_FORMAT,
        "arrays": {
            "pf": {"path": "pf.npy", "shape": [9, 4, 7], "dtype": "float32"},
            "obj": "obj.npy",
            "label": "label.npy",
        },
    }))
    with pytest.raises(ValueError, match="shape"):
        load_hlt_sample(manifest)
