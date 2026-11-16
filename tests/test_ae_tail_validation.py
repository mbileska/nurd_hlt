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
from scripts import evaluate_ae_tail_validation as tail
from utils.hlt_weights import file_sha256


def test_literal_top_ten_percent_across_all_classes_not_physics_measure():
    scores = np.arange(100, dtype=float)
    labels = np.ones(100, dtype=int)
    labels[-10:] = 2
    weights = np.ones(100)
    weights[0] = 1e9
    selected, report = tail.select_tail(scores, labels, weights, np.arange(100))
    assert np.array_equal(np.flatnonzero(selected), np.arange(90, 100))
    assert report["ae_boundary"] == 90
    assert report["selected_raw_fraction"] == .1


def test_tied_raw_scores_use_row_id_not_class():
    scores = np.ones(10)
    labels = np.arange(10) % 8
    rows = np.arange(10)[::-1]
    selected, _ = tail.select_tail(scores, labels, np.ones(10), rows, fraction=.2)
    assert set(rows[selected]) == {0, 1}
    second, _ = tail.select_tail(scores, labels[::-1], np.ones(10), rows, fraction=.2)
    assert np.array_equal(selected, second)


def test_physics_qcd_quantile_applies_same_cut_to_all_classes():
    scores = np.arange(10, dtype=float)
    labels = np.array([1] * 9 + [2])
    weights = np.ones(10)
    weights[0] = 1000
    selected, report = tail.select_tail(scores, labels, weights, np.arange(10), measure="physics-qcd")
    assert report["ae_boundary"] == 0
    assert selected.all()  # all boundary ties kept, not falsely reported as exact 10%


def test_tail_abcd_thresholds_are_within_tail_not_original_ae_cut():
    rng = np.random.default_rng(5)
    ae, md = rng.uniform(90, 100, 1000), rng.uniform(size=1000)
    scan = tail.scan_closure(ae, md, np.ones(1000), [.5, .7], {r: 1 for r in "ABCD"})
    assert scan["working_point"]["ae_cut"] > 90
    assert scan["working_point"]["regions"]["D"]["raw_count"] > 0
    assert scan["valid_points"] == 4
    invalid = tail.scan_closure(ae[:10], md[:10], np.ones(10), [.5, .7],
                                {"A": 50, "B": 50, "C": 50, "D": 500})
    assert invalid["valid_points"] == 0
    assert invalid["grid_median_absolute_nonclosure"] is None
    assert np.isnan(invalid["valid_grid"]).all()


def test_cache_raw_md_and_provenance(tmp_path):
    path, report_path = tmp_path / "cache.npz", tmp_path / "report.json"
    np.savez(path, source_row_index=np.array([3, 1]), ae_score=[4., 2.],
             raw_md=[10., 20.], physics_weight=[1., 1.], calibrated_md_percentile=[.1, .2])
    report = {"protocol": "train_QCD_fitted_conditional_MD_CDF_to_disjoint_validation_QCD",
              "heldout_sample_accessed": False, "checkpoint_sha256": "checkpoint",
              "ae_checkpoint_sha256": "ae", "qcd_label": 1, "n_pca": 6,
              "checkpoint_epoch": 22, "fit_qcd_events": 20, "validation_qcd_events": 2}
    report_path.write_text(json.dumps(report))
    args = (path, report_path, {"epoch": 22}, "ae", np.array([1, 3]), np.ones(4), "checkpoint", 20)
    cached = tail.read_raw_md_cache(*args)
    assert cached["raw_md"].tolist() == [10., 20.]
    report["checkpoint_sha256"] = "another_model"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="provenance"):
        tail.read_raw_md_cache(*args)


def test_cpu_end_to_end_cache_reuse_and_outputs(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(55)
    categories = torch.tensor([15, 0, 27, 6, 9, 12, 18, 21]).repeat(50)
    labels = torch.tensor([0, 1, 2, 3, 3, 3, 0, 0]).repeat(50)
    n = len(labels)
    sample = {"label": labels, "background_category": categories,
              "pf": torch.zeros(n, 3, 7), "obj": torch.randn(n, 2, 4, generator=generator),
              "event_id": torch.arange(n)[:, None]}
    data, weights = tmp_path / "train.pt", tmp_path / "weights.pt"
    torch.save(sample, data)
    torch.save(torch.ones(n), weights)
    ae_config = {"features": 8, "latent_dim": 3,
                 "encoder_config": {"nodes": [12]}, "decoder_config": {"nodes": [12, 8]}}
    ae = HLTAutoencoder(ae_config).eval()
    scaler = {"mu": torch.zeros(8), "std": torch.ones(8)}
    ae_path = tmp_path / "ae.pth"
    torch.save({"ae": ae.state_dict(), "ae_config": ae_config}, ae_path)
    _, _, preprocessing = build_hlt_datasets(
        str(data), ae, gen_weight_path=str(weights), target_scheme="decay8", val_split=.4,
        ae_scaler=scaler, balance_strata=1)
    checkpoint = {"epoch": 22, "effective_lambda_info": 1., "ae_scaler": scaler,
                  "preprocessing": preprocessing, "ae_checkpoint_sha256": file_sha256(ae_path)}
    path = tmp_path / "model.pth"
    torch.save(checkpoint, path)  # deliberately has NO encoder weights: no PF inference is needed
    validation = preprocessing["split"]["validation_indices"].numpy()
    all_labels = tail.checkpoint_target_labels(sample, preprocessing).numpy()
    qcd_rows = validation[all_labels[validation] == 1]
    ae_scores = tail.infer_ae_scores(ae, scaler, sample, qcd_rows, "cpu", 128)
    cache, cache_report = tmp_path / "cache.npz", tmp_path / "cache.json"
    np.savez(cache, source_row_index=qcd_rows, ae_score=ae_scores,
             raw_md=np.arange(len(qcd_rows)) + 1., physics_weight=np.ones(len(qcd_rows)))
    cache_report.write_text(json.dumps({
        "protocol": "train_QCD_fitted_conditional_MD_CDF_to_disjoint_validation_QCD",
        "heldout_sample_accessed": False, "checkpoint_sha256": file_sha256(path),
        "ae_checkpoint_sha256": file_sha256(ae_path), "qcd_label": 1, "n_pca": 6,
        "checkpoint_epoch": 22, "fit_qcd_events": 30, "validation_qcd_events": 20,
    }))
    opened = []
    real_load = tail.load_hlt_sample
    monkeypatch.setattr(tail, "load_hlt_sample", lambda p: (opened.append(p), real_load(p))[1])
    output = tmp_path / "output"
    args = ["--checkpoint", str(path), "--ae-checkpoint", str(ae_path),
            "--training-manifest", str(data), "--training-weights", str(weights),
            "--qcd-score-cache", str(cache), "--qcd-cache-report", str(cache_report),
            "--output-dir", str(output), "--threads", "1"]
    tail.main(args)
    result = json.loads((output / "tail_summary.json").read_text())
    assert opened == [data]
    assert result["heldout_sample_accessed"] is False
    assert result["device"] == "cpu"
    assert result["md_calibration_applied"] is False
    assert result["selection"]["selected_rows"] == 16
    assert result["scans"]["tail"]["valid_points"] == 0
    assert len(list((output / "plots").glob("*.png"))) == 6
    with np.load(output / "validation_qcd_scores.npz") as saved:
        assert saved["source_row_index"].tolist() == qcd_rows.tolist()
        assert np.array_equal(saved["raw_md"], np.arange(len(qcd_rows)) + 1.)
    with pytest.raises(FileExistsError):
        tail.main(args)


def test_launcher_requests_cpu_and_no_gpu(tmp_path):
    root = Path(__file__).parents[1]
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    for name in ("launch_ae_tail_validation.sh", "submit_ae_tail_validation.sbatch", "train3m_environment.sh"):
        shutil.copy2(root / "slurm" / name, slurm / name)
    base = tmp_path / "base"
    converted = base / "data/train3M_v1/converted_v1"
    for name in ("train/sample.json", "test/sample.json", "weight_train.npz",
                 "weight_test.npz", "FULL_VALIDATION_PASSED"):
        path = converted / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    source = "train3m_physics_decay8_lambda1_20261107_132516"
    for name in (f"checkpoints/hlt/hlt/hlt_nurd_engineer_{source}/checkpoint_main.pth.tar",
                 "checkpoints/hlt/hlt/ae_engineer_train3m_physics_decay8_full_20261104_142222/checkpoint_ae.pth",
                 f"outputs/{source}_conditional_md_20261108_154347/validation_qcd_scores.npz",
                 f"outputs/{source}_conditional_md_20261108_154347/conditional_md_report.json"):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "git").write_text("#!/bin/bash\nif [[ $1 == rev-parse ]]; then echo commit; fi\n")
    (binary / "sbatch").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\necho 1234\n')
    for path in binary.iterdir():
        path.chmod(0o755)
    env = os.environ.copy()
    for key in ("TRAIN_PT", "TRAIN_WEIGHTS", "GEN_WEIGHT_TRAIN", "HELDOUT_PT", "HELDOUT_WEIGHTS",
                "TRAIN3M_DIR", "TRAIN3M_CONVERTED_DIR", "AE_TAIL_CACHE_DIR"):
        env.pop(key, None)
    capture = tmp_path / "capture"
    env.update(BASE=str(base), PATH=f"{binary}:{env['PATH']}", CAPTURE=str(capture))
    result = subprocess.run(["bash", str(slurm / "launch_ae_tail_validation.sh"), "tail_fixture"],
                            env=env, check=True, capture_output=True, text=True)
    assert "Job: 1234" in result.stdout
    assert "--partition=cpu" in capture.read_text()
    assert "--gres" not in capture.read_text()
    batch = (slurm / "submit_ae_tail_validation.sbatch").read_text()
    assert "#SBATCH --partition=cpu" in batch
    assert "#SBATCH --gres" not in batch
    assert 'export CUDA_VISIBLE_DEVICES=""' in batch
