import json

import numpy as np
import torch

from scripts.audit_hlt_data import main, weighted_ks


def _sample(n=24):
    pf = torch.zeros(n, 8, 7)
    obj = torch.zeros(n, 4, 4)
    labels = torch.arange(n) % 4
    for index in range(n):
        pf[index, :3, 0] = torch.tensor([30.0, 12.0, 5.0]) + index
        pf[index, :3, 1] = torch.tensor([0.2, -0.4, 1.1])
        pf[index, :3, 2] = torch.tensor([0.1, -1.0, 2.2])
        pf[index, :3, 3] = torch.tensor([0.01, -0.02, 0.03])
        pf[index, :3, 4] = torch.tensor([1.0, -2.0, 3.0])
        pf[index, :3, 5] = 1
        pf[index, :3, 6] = torch.tensor([211, -11, 22])
        obj[index, :2] = torch.arange(8).reshape(2, 4) + index + 1
    event_ids = torch.arange(10_000, 10_000 + n)
    return {"pf": pf, "obj": obj, "label": labels, "event_id": event_ids}


def test_weighted_ks_identical_and_shifted():
    x = np.linspace(0, 1, 100)
    weights = np.linspace(1, 2, 100)
    assert weighted_ks(x, x, weights, weights) == 0.0
    assert weighted_ks(x, x + 10, weights, weights) > 0.99


def test_audit_exports_original_flag_indices(tmp_path):
    sample = _sample()
    sample["pf"][0, 0, 1] = torch.nan
    sample["pf"][1, 0, 0] = -1
    sample["pf"][2] = 0
    sample["pf"][3, 3, 0] = 4
    sample["pf"][3, 3, 6] = 999
    sample["pf"][4, 4, 1] = 9
    sample["obj"][5, 0, 0] = torch.inf
    sample["obj"][6] = 0
    data = tmp_path / "data.pt"
    weight_file = tmp_path / "weights.pt"
    torch.save(sample, data)
    torch.save({"weights": torch.linspace(.5, 2, len(sample["label"])),
                "event_id": sample["event_id"]}, weight_file)

    output = tmp_path / "audit"
    result = main([
        "--dataset", "train", str(data), str(weight_file),
        "--output-dir", str(output), "--chunk-size", "7",
        "--event-plot-sample", "20", "--candidate-plot-sample", "80",
        "--no-plots",
    ])
    assert result == 0
    report = json.loads((output / "report.json").read_text())
    assert report["datasets"]["train"]["weights"]["alignment_verified"] is True
    assert report["datasets"]["train"]["flags"]["pf_nonfinite"]["rows"] == 1
    assert report["datasets"]["train"]["flags"]["all_padded_pf"]["rows"] == 1
    assert report["datasets"]["train"]["flags"]["obj_nonfinite"]["rows"] == 1
    flags = np.load(output / "flags" / "train_event_flags.npz")
    assert flags["pf_nonfinite_indices"].tolist() == [0]
    assert flags["negative_or_nonfinite_pt_indices"].tolist() == [1]
    assert 2 in flags["all_padded_pf_indices"]
    assert 3 in flags["unsupported_pdgid_indices"]
    assert (output / "report.md").is_file()
    assert (output / "filter_candidates.csv").is_file()
