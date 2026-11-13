"""Small end-to-end check for the held-out input-domain diagnostic."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from models.hlt_autoencoder import HLTAutoencoder
from scripts.compare_input_domains import VARIABLES, extract_chunk, pdg_categories


def test_extraction_contract():
    pf = np.zeros((2, 400, 7), dtype=np.float32)
    obj = np.zeros((2, 23, 4), dtype=np.float32)
    pf[0, 0, [0, 5, 6]] = [40, 1, 211]
    pf[0, 1, [0, 5, 6]] = [30, 0, 22]
    obj[0, 0, :4] = [50, 1, 0, 0]
    obj[0, 1, :4] = [80, 3, 0, 0]
    obj[0, 22, :4] = [45, 0, 0, 4]
    values, active, _ = extract_chunk(pf, obj)
    assert values["pf_count"].tolist() == [2, 0]
    assert values["nonpf_count"].tolist() == [1, 0]
    assert values["ht10"].tolist() == [50, 0]
    assert values["jet_lead_pt"].tolist() == [80, 0]
    assert values["met_pt"].tolist() == [45, 0]
    assert pdg_categories(pf[:, :, 6])[active].tolist() == [0, 2]


def test_cli_writes_exactly_ten_figures(tmp_path):
    n = 16
    labels = np.tile(np.arange(4, dtype=np.int64), 4)
    pf = np.zeros((n, 400, 7), dtype=np.float32)
    obj = np.zeros((n, 23, 4), dtype=np.float32)
    for i in range(n):
        pf[i, 0, [0, 5, 6]] = [10 + i, 1, 211]
        pf[i, 1, [0, 5, 6]] = [5 + i, i % 2, 22]
        obj[i, 0, :4] = [31 + i, 1.1, 0.2, 0]
        obj[i, 22, :4] = [20 + i, 0, 0, 4]
    legacy = tmp_path / "legacy.pt"
    torch.save({"pf": torch.from_numpy(pf), "obj": torch.from_numpy(obj),
                "label": torch.from_numpy(labels)}, legacy)
    legacy_weights = tmp_path / "legacy_weights.pt"
    torch.save(torch.linspace(1, 2, n), legacy_weights)
    sample_dir = tmp_path / "new"
    sample_dir.mkdir()
    arrays = {"pf": pf.copy(), "obj": obj.copy(), "label": labels,
              "event_id": np.arange(n, dtype=np.int64)}
    for key, value in arrays.items():
        np.save(sample_dir / f"{key}.npy", value)
    manifest = sample_dir / "sample.json"
    manifest.write_text(json.dumps({"format": "nurd_hlt_numpy_v1",
                                    "arrays": {k: f"{k}.npy" for k in arrays}}))
    new_weights = tmp_path / "new_weights.npz"
    np.savez(new_weights, weights=np.linspace(1, 2, n),
             weight_scope=np.array("all_events"), event_id=np.arange(n))
    cfg = {"features": 92, "latent_dim": 4,
           "encoder_config": {"nodes": [8]},
           "decoder_config": {"nodes": [8, 92]}}
    ae = HLTAutoencoder(cfg)
    checkpoint = tmp_path / "ae.pth"
    torch.save({"ae": ae.state_dict(), "ae_config": cfg,
                "ae_scaler": {"mu": torch.zeros(92), "std": torch.ones(92)}},
               checkpoint)
    output = tmp_path / "out"
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")}
    command = [sys.executable, "scripts/compare_input_domains.py",
               "--legacy", str(legacy), "--legacy-weights", str(legacy_weights),
               "--new", str(manifest), "--new-weights", str(new_weights),
               "--ae-checkpoint", str(checkpoint), "--output-dir", str(output),
               "--device", "cpu", "--batch-size", "4"]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                   env=env, check=True, capture_output=True, text=True)
    assert len(list(output.glob("*.png"))) == 10
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["figures"]) == 10
    assert metadata["samples"]["Mequinna"]["weight_scope"] == "single_label"
    assert metadata["samples"]["train3M"]["weight_scope"] == "all_events"
    with (output / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 4 * len(VARIABLES)
