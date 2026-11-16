import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
from scipy.stats import gaussian_kde
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))

import eval_abcd_nurd as evaluator
from scripts import plot_ae_tail_eval_suite as suite
from scripts.evaluate_ae_tail_validation import scan_closure, compact_scan, select_tail
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


def test_weighted_kde_uses_original_weights_once():
    rng = np.random.default_rng(6)
    x, y = np.exp(rng.normal(size=(2, 20)))
    weights = np.linspace(.1, 10, 20)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
    result = suite.weighted_kde(x, y, weights, xx, yy)
    expected = gaussian_kde(np.vstack((np.log10(x), np.log10(y))), weights=weights)(
        np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    assert np.allclose(result, expected)
    assert not np.allclose(result, suite.weighted_kde(x, y, np.ones(20), xx, yy))


def test_weighted_profile_mean_and_effective_statistics():
    x, y, weights = np.ones(3), np.array([0., 10., 20.]), np.array([1., 3., 2.])
    centers, means, errors = suite.weighted_profile(x, y, weights, bins=1, minimum_ess=1)
    mean = np.average(y, weights=weights)
    variance = np.average((y - mean) ** 2, weights=weights)
    neff = weights.sum() ** 2 / np.square(weights).sum()
    assert len(centers) == 1
    assert means[0] == pytest.approx(mean)
    assert errors[0] == pytest.approx(np.sqrt(variance / (neff - 1)))
    assert len(suite.weighted_profile(x, y, weights, bins=1, minimum_ess=20)[0]) == 0


def test_scatter_draws_are_deterministic_display_only_and_ignore_zero_mass():
    labels = np.array([0, 0, 1, 1])
    weights = np.array([0., 1., 1., 0.])
    rows = suite.display_indices(labels, weights)
    assert set(rows) == {1, 2}
    assert np.array_equal(rows, suite.display_indices(labels, weights))


@pytest.mark.parametrize("include_signal", [False, True])
def test_end_to_end_fixed_tail_has_standard_files_and_unchanged_selection(tmp_path, monkeypatch, include_signal):
    rng = np.random.default_rng(23)
    n_train, n_val = 800, 400
    n = n_train + n_val
    categories = torch.tensor([15, 0, 27, 6, 9, 12, 18, 21]).repeat(n // 8)
    parents = torch.tensor([0, 1, 2, 3, 3, 3, 0, 0]).repeat(n // 8)
    latent = rng.normal(size=(n, 2)).astype(np.float32)
    raw = {"pf": torch.from_numpy(latent[:, None]), "obj": torch.zeros(n, 1, 4),
           "label": parents, "background_category": categories,
           "event_id": torch.arange(n)[:, None]}
    data_path, weight_path = tmp_path / "train.pt", tmp_path / "weights.pt"
    torch.save(raw, data_path)
    torch.save({"weights": torch.ones(n), "weight_scope": "all_events",
                "event_id": raw["event_id"]}, weight_path)
    physics, weight_metadata = load_generator_weights(str(weight_path), parents, sample=raw)
    physics = physics.numpy()
    train, validation = np.arange(n_train), np.arange(n_train, n)
    names = {0: "Z->ee", 1: "QCD", 2: "ttbar", 3: "W->enu", 4: "W->munu",
             5: "W->taunu", 6: "Z->mumu", 7: "Z->tautau"}
    preprocessing = {
        "target_scheme": "decay8", "label_map": {i: i for i in range(8)},
        "target_class_names": names, "data_signature": sample_signature(raw),
        "split": {"train_indices": torch.from_numpy(train), "validation_indices": torch.from_numpy(validation)},
        "weighting": {"generator": weight_metadata},
    }
    labels = checkpoint_target_labels(raw, preprocessing).numpy()
    ae_scores = np.exp(rng.normal(size=n_val))
    selected, selection = select_tail(ae_scores, labels[validation], physics[validation], validation)
    qcd_train = train[labels[train] == 1]
    mean, whitening = evaluator._fit_class_transform(
        latent[qcd_train], np.ones(len(qcd_train), bool), 6, "test", weights=physics[qcd_train])
    qcd_val = validation[labels[validation] == 1]
    md = np.square((latent[qcd_val] - mean) @ whitening).sum(axis=1)
    qcd_keep = selected[labels[validation] == 1]
    percentiles = np.array([.5, .7])
    ae_qcd = ae_scores[labels[validation] == 1]
    scans = {
        "full": scan_closure(ae_qcd, md, physics[qcd_val], percentiles, {r: 1 for r in "ABCD"}),
        "tail": scan_closure(ae_qcd[qcd_keep], md[qcd_keep], physics[qcd_val][qcd_keep],
                              percentiles, {r: 1 for r in "ABCD"}),
    }
    source = tmp_path / "tail"
    source.mkdir()
    ae_path = tmp_path / "ae.pth"
    ae_path.write_bytes(b"unused AE checkpoint: cached background scores only")
    model_path = tmp_path / "model.pth"
    metadata = {"preprocessing": preprocessing, "ae_checkpoint_sha256": file_sha256(ae_path),
                "ae_scaler": {"mu": torch.zeros(4), "std": torch.ones(4)}}
    torch.save(metadata, model_path)
    cache_report = tmp_path / "original_cache.json"
    cache_report.write_text(json.dumps({"ae_checkpoint": str(ae_path)}))
    report = {
        "checkpoint": str(model_path), "checkpoint_sha256": file_sha256(model_path),
        "ae_checkpoint_sha256": file_sha256(ae_path), "cache_report": str(cache_report),
        "training_manifest": str(data_path), "training_weights": str(weight_path),
        "evaluation_weighting": {"physics_weight_sha256": weight_metadata["effective_physics_weight_sha256"]},
        "selection": selection, "scans": {k: compact_scan(v) for k, v in scans.items()},
    }
    (source / "tail_summary.json").write_text(json.dumps(report))
    report_hash = file_sha256(source / "tail_summary.json")
    np.savez(source / "validation_ae_selection.npz", source_row_index=validation,
             ae_score=ae_scores, label=labels[validation], physics_weight=physics[validation],
             event_id=raw["event_id"][validation].numpy(), selected=selected)
    np.savez(source / "validation_qcd_scores.npz", source_row_index=qcd_val, raw_md=md)
    np.savez(source / "closure_scan_arrays.npz", percentiles=percentiles, tail_valid_grid=scans["tail"]["valid_grid"])
    monkeypatch.setattr(evaluator, "load_nurd_model", lambda *_: (object(), metadata))
    called_rows = []
    def infer(_model, sample, indices, _device, _batch):
        called_rows.append(np.asarray(indices))
        return sample["pf"][indices, 0, :].numpy()
    monkeypatch.setattr(suite, "infer_latents", infer)
    extra_args = []
    if include_signal:
        signal_path = tmp_path / "signal.pt"
        raw_signal = {"pf": torch.from_numpy(rng.normal(size=(31, 1, 2)).astype(np.float32)),
                      "obj": torch.full((31, 1, 4), 50.), "label": torch.zeros(31, dtype=torch.long)}
        torch.save(raw_signal, signal_path)
        monkeypatch.setattr(evaluator, "load_ae", lambda *_: object())
        monkeypatch.setattr(suite, "infer_ae_scores", lambda _model, _scaler, sample, rows, *_:
                            sample["obj"][rows, 0, 0].numpy())
        extra_args = ["--signal", str(signal_path)]
    else:
        monkeypatch.setattr(evaluator, "load_ae", lambda *_: pytest.fail("No AE rescore needed without signal"))
    output = tmp_path / "plots_out"
    suite.main(["--tail-dir", str(source), "--output-dir", str(output), "--threads", "1", *extra_args])
    result = json.loads((output / "plot_suite_metadata.json").read_text())
    assert result["heldout_background_accessed"] is False
    assert result["networks_updated"] is False
    assert result["MD_cache_agreement"] is True
    assert result["selection"] == selection
    assert result["plot_style"]["score_figure_inches"] == [8, 6]
    assert result["plot_style"]["hist_figure_inches"] == [6, 5]
    assert result["plot_style"]["closure_grid_inches"] == [7.5, 6.5]
    assert result["plot_style"]["dpi"] == 200
    assert len(list((output / "plots").glob("*.png"))) == 20 + int(include_signal)
    if include_signal:
        assert result["signal_retention"]["rows_after"] == 31
        assert (output / "plots/hist2d_TpTp.png").is_file()
    for name in ("hist2d_pca_md_kde.png", "pca_scatter_embeddings.png", "pca_corner.png",
                 "profile_contrastive_vs_AE_by_class.png", "cut_and_count_bkg_check.png"):
        assert (output / "plots" / name).stat().st_size > 0
    assert np.array_equal(called_rows[0], qcd_train)
    assert np.array_equal(called_rows[1], validation[selected])
    with np.load(output / "selected_validation_scores.npz") as saved:
        assert np.array_equal(saved["source_row_index"], validation[selected])
    assert file_sha256(source / "tail_summary.json") == report_hash
    with pytest.raises(FileExistsError):
        suite.main(["--tail-dir", str(source), "--output-dir", str(output)])


@pytest.mark.parametrize("resource", ["cpu", "gpu"])
def test_launcher_resource_choice_does_not_require_a100(tmp_path, resource):
    root = Path(__file__).parents[1]
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    for name in ("launch_ae_tail_standard_plots.sh", "submit_ae_tail_standard_plots.sbatch"):
        shutil.copy2(root / "slurm" / name, slurm / name)
    source = tmp_path / "base/outputs/fixture_ae_tail_validation"
    source.mkdir(parents=True)
    for name in ("tail_summary.json", "validation_ae_selection.npz", "validation_qcd_scores.npz", "closure_scan_arrays.npz"):
        (source / name).write_text("fixture")
    (source / "tail_summary.json").write_text(json.dumps({"selection": {"measure": "physics-all"}}))
    newer_raw = tmp_path / "base/outputs/newer_raw_ae_tail_validation/tail_summary.json"
    newer_raw.parent.mkdir()
    newer_raw.write_text(json.dumps({"selection": {"measure": "raw-all"}}))
    os.utime(newer_raw, (2000000000, 2000000000))
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "git").write_text("#!/bin/bash\nif [[ $1 == rev-parse ]]; then echo commit; fi\n")
    (binary / "sbatch").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\necho 1234\n')
    for path in binary.iterdir():
        path.chmod(0o755)
    capture = tmp_path / "capture"
    env = os.environ.copy()
    env.update(BASE=str(tmp_path / "base"), PATH=f"{binary}:{env['PATH']}", CAPTURE=str(capture), INCLUDE_SIGNAL="0")
    result = subprocess.run(["bash", str(slurm / "launch_ae_tail_standard_plots.sh"),
                             "latest" if resource == "cpu" else str(source), "fixture", resource],
                            check=True, capture_output=True, text=True, env=env)
    args = capture.read_text()
    assert "Job: 1234" in result.stdout
    assert f"Source selection: {source}" in result.stdout
    assert f"--partition={resource}" in args
    assert "a100" not in args.lower()
    assert ("--gres=gpu:1" in args) == (resource == "gpu")
