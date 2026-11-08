import json
import sys
import types

import numpy as np
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from scripts import run_conditional_md_calibration as runner  # noqa: E402


def test_runner_uses_saved_training_split_and_writes_reusable_fit(
    tmp_path, monkeypatch,
):
    rng = np.random.default_rng(31)
    n_train, n_val = 400, 200
    n = n_train + n_val
    ae = np.exp(rng.normal(size=n)).astype(np.float32)
    latent = rng.normal(size=(n, 2)).astype(np.float32)
    pf = torch.from_numpy(latent[:, None, :])
    obj = torch.zeros((n, 1, 4))
    obj[:, 0, 0] = torch.from_numpy(ae)
    sample = {"pf": pf, "obj": obj, "label": torch.ones(n, dtype=torch.long)}
    weights = torch.ones(n, dtype=torch.float64)

    train_manifest = tmp_path / "train/sample.json"
    train_manifest.parent.mkdir()
    checkpoint = tmp_path / "checkpoint_main.pth.tar"
    ae_checkpoint = tmp_path / "checkpoint_ae.pth"
    weight_file = tmp_path / "weight_train.npz"
    for path in (train_manifest, checkpoint, ae_checkpoint, weight_file):
        path.write_bytes(b"input")
    metadata = {
        "epoch": 22,
        "ae_checkpoint_sha256": "sha",
        "ae_scaler": {"mu": torch.zeros(4), "std": torch.ones(4)},
    }

    monkeypatch.setattr(runner, "load_hlt_sample", lambda _path: sample)
    monkeypatch.setattr(runner, "sample_signature", lambda _sample: "signature")
    monkeypatch.setattr(runner, "load_generator_weights",
                        lambda *_args, **_kwargs: (weights, {}))
    monkeypatch.setattr(runner, "checkpoint_reference_indices",
                        lambda *_args: (np.arange(n_train), np.arange(n_train, n)))
    monkeypatch.setattr(runner, "file_sha256", lambda _path: "sha")
    monkeypatch.setattr(runner.torch, "load", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(runner, "load_ae", lambda *_args: (object(), {}))
    monkeypatch.setattr(runner, "_compute_ae_scores",
                        lambda _model, _scaler, obj_data, indices, *_rest:
                        obj_data[indices, 0, 0].numpy())
    monkeypatch.setattr(runner, "load_nurd_model",
                        lambda *_args: (object(), {}))
    monkeypatch.setattr(runner, "_infer_latents",
                        lambda _model, pf_data, indices, *_rest:
                        pf_data[indices, 0, :].numpy())
    monkeypatch.setattr(runner, "_fit_class_transform",
                        lambda *_args, **_kwargs: (np.zeros(2), np.eye(2)))
    monkeypatch.setattr(runner, "_plot_comparison", lambda *_args: None)

    def fake_closure(_ae, _md, _weights, _percentiles, _minimums):
        return {
            "grid": np.array([[0.2, np.nan], [0.1, 0.3]]),
            "grid_points": 3,
            "grid_rejected_points": 1,
            "grid_median_absolute_nonclosure": 0.2,
            "grid_p90_absolute_nonclosure": 0.28,
            "curve_points": 1,
            "curve_rejected_points": 0,
            "curve_median_absolute_nonclosure": 0.2,
            "curve_p90_absolute_nonclosure": 0.2,
            "curve": [{"percentile": 0.5, "efficiency": 0.25,
                       "ratio": 0.8, "uncertainty": 0.1,
                       "absolute_nonclosure": 0.2}],
        }

    monkeypatch.setattr(runner, "validation_closure_metrics", fake_closure)
    output = tmp_path / "result"
    runner.main([
        "--checkpoint", str(checkpoint), "--ae-ckpt", str(ae_checkpoint),
        "--data", str(train_manifest), "--gen-weight-path", str(weight_file),
        "--outdir", str(output), "--ae-bins", "4",
        "--min-fit-bin-ess", "20",
    ])

    report = json.loads((output / "conditional_md_report.json").read_text())
    assert report["heldout_sample_accessed"] is False
    assert report["fit_qcd_events"] == n_train
    assert report["validation_qcd_events"] == n_val
    assert report["effective_ae_bins"] == 4
    assert report["common_valid_grid"]["points"] == 3
    assert len(report["ae_bins"]) == 4
    with np.load(output / "validation_qcd_scores.npz") as saved:
        assert saved["source_row_index"].tolist() == list(range(n_train, n))
        assert np.all((saved["calibrated_md_percentile"] >= 0)
                      & (saved["calibrated_md_percentile"] <= 1))
    assert (output / "conditional_md_fit_train_qcd.npz").is_file()
