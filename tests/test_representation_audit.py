import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from dataset.hlt_smcocktail_dataset import build_hlt_datasets
from models.hlt_autoencoder import HLTAutoencoder
from models.hlt_con import HLTContrastiveModel
from scripts import audit_frozen_representations as audit
from utils.hlt_weights import file_sha256


def test_parent_probability_sum_is_not_mapped_argmax():
    probability = np.array([[.2, .3, .05, .16, .14, .1, .03, .02]])
    labels = np.array([3])
    report = audit.report_predictions(
        probability, labels, audit.DECAY_TO_PARENT, {"raw": np.ones(1)})["raw"]
    assert report["decay8"]["accuracy"] == 0
    assert report["parent4_mapped_argmax"]["accuracy"] == 0
    assert report["parent4_probability_sum"]["accuracy"] == 1
    assert np.allclose(audit.parent_probabilities(probability, audit.DECAY_TO_PARENT),
                       [[.25, .3, .05, .4]])


def test_support_selection_is_deterministic_and_records_inclusion():
    labels = np.repeat(np.arange(8), np.arange(8) + 10)
    first, probability = audit.training_support(labels, 5, 42)
    second, _ = audit.training_support(labels, 5, 42)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 40
    assert np.allclose(probability, np.repeat(5 / (np.arange(8) + 10), 5))


def test_knn_weights_and_zero_distance_behavior():
    distances = np.array([[1., 1.], [0., 1.]])
    neighbors = np.array([[0, 1], [0, 1]])
    labels = np.array([0, 1])
    result = audit.knn_probabilities(distances, neighbors, labels,
                                     np.array([1., 9.]), n_classes=2)
    assert np.allclose(result, [[.1, .9], [1., 0.]])


def test_linear_probe_scaler_uses_training_not_validation():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(8), 20)
    train = np.eye(8)[labels] * 5 + rng.normal(scale=.1, size=(160, 8))
    validation = train[:8] + 100
    weights = np.ones(160)
    predictions, fitted, metadata, _, _ = audit.fit_linear_probe(
        train, validation, labels, weights, np.arange(40), 200)
    assert np.allclose(fitted["linear_mean"], train.mean(axis=0))
    assert predictions["linear"].shape == (8, 8)
    assert metadata["linear_converged"]


def test_md_reference_uses_only_passed_training_rows():
    train = np.array([[0., 1.], [2., 3.], [1., 2.]])
    weights = np.array([1., 3., 1.])
    mean, transform, eigenvalues = audit.fit_md_reference(train, weights, 2)
    assert np.allclose(mean, np.average(train, weights=weights, axis=0))
    scores = audit.md_scores(np.array([[100., 100.]]), mean, transform)
    assert scores.shape == (1,)
    assert eigenvalues.shape == (2,)
    assert np.isfinite(scores).all()


def test_closure_thresholds_do_not_fit_validation_distribution():
    rng = np.random.default_rng(11)
    train_ae, train_md = rng.uniform(size=(2, 1000))
    val_ae, val_md = rng.uniform(size=(2, 1000))
    minimums = {"A": 1, "B": 1, "C": 1, "D": 1}
    args = (train_ae, train_md, np.ones(1000))
    first = audit.closure_on_validation(
        *args, val_ae, val_md, np.ones(1000), [.5, .7], minimums)
    second = audit.closure_on_validation(
        *args, val_ae * 100, val_md * 100, np.ones(1000), [.5, .7], minimums)
    assert first["threshold_ae"] == second["threshold_ae"]
    assert first["threshold_md"] == second["threshold_md"]
    assert first["grid_points"] == 4


def test_end_to_end_frozen_audit_saved_split_and_artifacts(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(17)
    categories = torch.tensor([15, 0, 27, 6, 9, 12, 18, 21]).repeat(64)
    parents = torch.tensor([0, 1, 2, 3, 3, 3, 0, 0]).repeat(64)
    n = len(parents)
    pf = torch.randn(n, 6, 7, generator=generator)
    pf[:, :, 0] = pf[:, :, 0].abs() + .1
    sample = {"pf": pf, "obj": torch.randn(n, 2, 4, generator=generator),
              "label": parents, "background_category": categories,
              "event_id": torch.arange(n)[:, None]}
    data, weights = tmp_path / "train.pt", tmp_path / "weights.pt"
    torch.save(sample, data)
    torch.save(torch.ones(n), weights)
    ae_config = {"features": 8, "latent_dim": 3,
                 "encoder_config": {"nodes": [12]}, "decoder_config": {"nodes": [12, 8]}}
    ae = HLTAutoencoder(ae_config)
    scaler = {"mu": torch.zeros(8), "std": torch.ones(8)}
    ae_path = tmp_path / "ae.pth"
    torch.save({"ae": ae.state_dict(), "ae_config": ae_config}, ae_path)
    _, _, preprocessing = build_hlt_datasets(
        str(data), ae, gen_weight_path=str(weights), target_scheme="decay8",
        val_split=.25, ae_scaler=scaler, balance_strata=1)
    config = dict(embed_size=8, latent_dim=3, proj_dim=3, num_heads=1,
                  num_layers=1, dim_ff=16, linear_dim=2, dropout=.1)
    model = HLTContrastiveModel(num_classes=8, num_tokens=6, **config)
    path = tmp_path / "hlt_nurd_engineer_test/checkpoint_main.pth.tar"
    path.parent.mkdir()
    torch.save({"config": config, "state_dict_model": model.state_dict(),
                "ae_scaler": scaler, "preprocessing": preprocessing,
                "ae_checkpoint_sha256": file_sha256(ae_path), "epoch": 3,
                "classification_head_trained": False}, path)
    before = file_sha256(path)
    opened = []
    real_loader = audit.load_hlt_sample
    monkeypatch.setattr(audit, "load_hlt_sample", lambda p: (opened.append(p), real_loader(p))[1])
    output = tmp_path / "audit"
    args = ["--checkpoints", str(path), "--ae-ckpt", str(ae_path),
            "--data", str(data), "--gen-weight-path", str(weights), "--outdir", str(output),
            "--knn-per-class", "10", "--neighbors", "3", "--grid-points", "3",
            "--min-region-ess", "1", "--min-D-ess", "1", "--jobs", "1"]
    audit.main(args)
    summary = json.loads((output / "audit_summary.json").read_text())
    report = summary["runs"]["test"]
    assert summary["heldout_sample_accessed"] is False
    assert summary["network_parameters_updated"] is False
    assert opened == [data]
    assert file_sha256(path) == before
    assert len(list((output / "test/plots").glob("*.png"))) == 8
    assert set(report["probes"]) == {"linear", "knn", "knn_uniform_control"}
    assert set(report["scores"]) == {"encoder", "projector"}
    with np.load(output / "test/validation_predictions_scores.npz") as saved:
        assert saved["source_row_index"].tolist() == preprocessing["split"]["validation_indices"].tolist()
        assert saved["event_id"].shape[0] == summary["validation_events"]
        assert np.allclose(saved["encoder_linear_probabilities"].sum(axis=1), 1)
    with np.load(output / "test/probe_and_md_fits.npz") as fits:
        assert set(fits["knn_source_row_index"]).issubset(set(preprocessing["split"]["train_indices"].tolist()))
    with pytest.raises(FileExistsError):
        audit.main(args)
    bad_metadata = torch.load(path, map_location="cpu", weights_only=False)
    bad_metadata["ae_checkpoint_sha256"] = "different_ae"
    bad_path = path.parent / "checkpoint_bad.pth.tar"
    torch.save(bad_metadata, bad_path)
    bad_args = [str(bad_path) if value == str(path) else value for value in args]
    bad_args = [str(tmp_path / "bad_audit") if value == str(output) else value for value in bad_args]
    with pytest.raises(ValueError, match="different AE"):
        audit.main(bad_args)
    assert not (tmp_path / "bad_audit").exists()


def test_launcher_submits_one_job_and_only_outer_training_paths(tmp_path):
    root = Path(__file__).parents[1]
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    for name in ("launch_representation_audit.sh", "train3m_environment.sh",
                 "submit_representation_audit.sbatch"):
        shutil.copy2(root / "slurm" / name, slurm / name)
    base = tmp_path / "base"
    converted = base / "data/train3M_v1/converted_v1"
    for relative in ("train/sample.json", "test/sample.json", "weight_train.npz",
                     "weight_test.npz", "FULL_VALIDATION_PASSED"):
        p = converted / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    prefix = "pure_fixture"
    for name in ("ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth",
                 *(f"hlt_nurd_engineer_{prefix}_sc{s}/checkpoint_main.pth.tar" for s in ("01", "03", "05"))):
        p = base / "checkpoints/hlt/hlt" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "git").write_text("#!/bin/bash\nif [[ $1 == rev-parse ]]; then echo commit; fi\n")
    capture = tmp_path / "capture"
    (binaries / "sbatch").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$TRAIN_PT" "$TRAIN_WEIGHTS" "$PURE_RUN_PREFIX" "$OUTDIR" > "$CAPTURE"\necho 12345\n')
    for p in binaries.iterdir():
        p.chmod(0o755)
    env = os.environ.copy()
    for key in ("TRAIN_PT", "TRAIN_WEIGHTS", "GEN_WEIGHT_TRAIN", "HELDOUT_PT", "HELDOUT_WEIGHTS",
                "TRAIN3M_DIR", "TRAIN3M_CONVERTED_DIR"):
        env.pop(key, None)
    env.update(BASE=str(base), PATH=f"{binaries}:{env['PATH']}", CAPTURE=str(capture))
    result = subprocess.run(["bash", str(slurm / "launch_representation_audit.sh"), prefix, "audit_fixture"],
                            env=env, check=True, capture_output=True, text=True)
    assert "Job: 12345" in result.stdout
    lines = capture.read_text().splitlines()
    assert lines[:2] == [str(converted / "train/sample.json"), str(converted / "weight_train.npz")]
    assert lines[2] == prefix
    assert lines[3] == str(base / "outputs/audit_fixture")
