import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import types

import pytest

from scripts import compare_abcd_axes as comparison


def make_args(tmp_path):
    args = argparse.Namespace(output_dir=tmp_path / "comparison", signal=None,
                              code_commit="test-commit")
    for name in ("checkpoint", "ae_checkpoint", "reference", "reference_weights",
                 "heldout", "heldout_weights"):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        setattr(args, name, path)
    return args


def example_evaluations():
    diagnostic = {
        "axis2_score": "qcd_mahalanobis",
        "evaluation_protocol": "saved_training_validation_split",
        "reference_sample": "train/sample.json", "report_sample": "test/sample.json",
        "qcd_events": 400, "weighted_qcd": True,
        "source_target_scheme": "decay8", "report_label_scheme": "decay8",
        "population_audit": {
            "report_events_before_mask": 600, "report_events_after_mask": 600,
            "accepted_event_mask_sha256": "mask", "report_data_signature": {"event_id_sha256": "ids"},
            "report_physics_weight_sha256": "physical-weights", "report_weight_scope": "all_events",
            "reference_data_signature": {"event_id_sha256": "reference-ids"},
            "reference_physics_weight_sha256": "reference-weights",
            "reference_fit_indices_sha256": "train-indices",
            "threshold_selection_indices_sha256": "validation-indices",
            "threshold_selection_event_mask_sha256": "validation-mask",
        },
        "statistical_filter": {"enabled": True, "count_basis": "effective_count",
                               "minimums": {"A": 50, "B": 50, "C": 50, "D": 500}},
        "working_point": {"selection_nonclosure": .2, "absolute_nonclosure": .3,
                          "statistically_valid": True, "regions": {}},
        "heldout_grid": {"points": 100, "median_absolute_nonclosure": .2,
                         "p90_absolute_nonclosure": .4},
        "heldout_diagonal_curve": {"median_absolute_nonclosure": .25},
    }
    for region in "ABCD":
        diagnostic["working_point"][region] = 25.
        diagnostic["working_point"]["regions"][region] = {
            "yield": 25., "sumw2": 25., "raw_count": 100, "effective_count": 25.}
    first = {"diagnostics": diagnostic, "thresholds": {"t1": .1, "t2": 2.}}
    second = copy.deepcopy(first)
    second["diagnostics"]["axis2_score"] = "one_minus_qcd_probability"
    second["thresholds"]["t2"] = .7
    return {"qcd_md": first, "one_minus_pqcd": second}


def test_same_ordinary_evaluator_only_axis_flag_changes(tmp_path):
    args = make_args(tmp_path)
    repo = Path(__file__).parents[1]
    md = comparison.evaluator_command(args, "qcd_md", repo)
    pqcd = comparison.evaluator_command(args, "one_minus_pqcd", repo)
    assert "--axis2_logit" not in md
    assert "--axis2_logit" in pqcd
    for option in ("--ckpt", "--ae_ckpt", "--test_pt", "--gen_weight_path",
                   "--reference_pt", "--reference_weight_path", "--n_pca",
                   "--target_scheme", "--qcd_label"):
        assert md[md.index(option) + 1] == pqcd[pqcd.index(option) + 1]
    for command in (md, pqcd):
        assert "--statistically_valid_closure" in command
        assert "--min_md" not in command
        assert not any("tail" in argument or "calibrat" in argument for argument in command[2:])
    args.signal = tmp_path / "signal.pt"
    assert "--signal_pt" in comparison.evaluator_command(args, "qcd_md", repo)


def test_comparison_verifies_events_weights_splits_and_conservation():
    evaluations = example_evaluations()
    comparison.verify_same_population(evaluations)
    row = comparison.comparison_row("qcd_md", evaluations["qcd_md"])
    assert row["heldout_predicted_over_observed"] == 1
    assert row["heldout_ratio_uncertainty"] == pytest.approx(.4)
    assert row["heldout_A_physics_efficiency"] == .25
    for field, value in (("accepted_event_mask_sha256", "different-mask"),
                         ("report_physics_weight_sha256", "different-weights"),
                         ("threshold_selection_event_mask_sha256", "different-selection")):
        broken = copy.deepcopy(evaluations)
        broken["one_minus_pqcd"]["diagnostics"]["population_audit"][field] = value
        with pytest.raises(ValueError, match="population_audit"):
            comparison.verify_same_population(broken)
    broken = copy.deepcopy(evaluations)
    broken["one_minus_pqcd"]["diagnostics"]["working_point"]["regions"]["A"]["yield"] += 1
    with pytest.raises(ValueError, match="physics yields"):
        comparison.verify_same_population(broken)
    broken = copy.deepcopy(evaluations)
    broken["one_minus_pqcd"]["diagnostics"]["evaluation_protocol"] = "legacy_same_sample_oracle"
    with pytest.raises(ValueError, match="validation-selected"):
        comparison.verify_same_population(broken)


def test_runner_calls_both_evaluations_and_writes_verified_summary(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    calls = []
    examples = example_evaluations()

    def fake_evaluate(command, **kwargs):
        calls.append(command)
        mode = "one_minus_pqcd" if "--axis2_logit" in command else "qcd_md"
        output = Path(command[command.index("--outdir") + 1])
        output.mkdir(parents=True)
        for key, filename in (("diagnostics", "diagnostics.json"),
                              ("thresholds", "abcd_thresholds.json")):
            (output / filename).write_text(json.dumps(examples[mode][key]))
        assert kwargs["check"] is True
        assert kwargs["env"]["MPLBACKEND"] == "Agg"

    monkeypatch.setattr(comparison.subprocess, "run", fake_evaluate)
    comparison.run_comparison(args)
    assert len(calls) == 2
    summary = json.loads((args.output_dir / "axis_comparison.json").read_text())
    assert summary["identical_population_verified"]
    assert summary["training"] is False
    assert summary["ae_tail_selection"] is False
    assert summary["conditional_calibration"] is False
    assert len(summary["results"]) == 2
    assert (args.output_dir / "axis_comparison.csv").is_file()
    with pytest.raises(FileExistsError):
        comparison.run_comparison(args)


def test_launcher_resources_defaults_and_env_isolation(tmp_path):
    source = Path(__file__).parents[1] / "slurm"
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    for name in ("launch_abcd_axis_comparison.sh", "train3m_environment.sh"):
        shutil.copy2(source / name, slurm / name)
    base = tmp_path / "base"
    converted = base / "data/train3M_v1/converted_v1"
    for directory in (converted / "train", converted / "test"):
        directory.mkdir(parents=True)
        (directory / "sample.json").write_text("manifest")
    (converted / "FULL_VALIDATION_PASSED").touch()
    for name in ("weight_train.npz", "weight_test.npz"):
        (converted / name).write_text("weights")
    for directory, name in (
        ("ae_engineer_train3m_physics_decay8_full_20261104_142222", "checkpoint_ae.pth"),
        ("hlt_nurd_engineer_train3m_physics_decay8_lambda1_20261107_132516", "checkpoint_main.pth.tar"),
    ):
        checkpoint = base / "checkpoints/hlt/hlt" / directory / name
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("checkpoint")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text('#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo test-commit; fi\n')
    git.chmod(0o755)
    sbatch = binaries / "sbatch"
    sbatch.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n'
        'for name in REFERENCE_PT REFERENCE_WEIGHTS HELDOUT_PT HELDOUT_WEIGHTS MODEL_CKPT RESOURCE; do\n'
        'printf "%s=%s\\n" "$name" "${!name}" >> "$CAPTURE"\ndone\necho 12345\n')
    sbatch.chmod(0o755)
    environment = os.environ.copy()
    environment.update({"PATH": f"{binaries}:{environment['PATH']}", "BASE": str(base),
                        "CAPTURE": str(tmp_path / "capture"),
                        "REFERENCE_PT": "/wrong/legacy.pt", "REFERENCE_WEIGHTS": "/wrong/weights.pt",
                        "HELDOUT_PT": "/wrong/test.pt", "HELDOUT_WEIGHTS": "/wrong/test_weights.pt"})
    for name in ("AE_CKPT", "MODEL_CKPT", "TRAIN3M_CONVERTED_DIR", "TRAIN3M_DIR", "TRAIN_PT", "GEN_WEIGHT_TRAIN"):
        environment.pop(name, None)
    for resource in ("gpu", "cpu"):
        result = subprocess.run(["bash", str(slurm / "launch_abcd_axis_comparison.sh"),
                                 "test_axes", resource], env=environment,
                                text=True, capture_output=True, check=True)
        assert "Job: 12345" in result.stdout
        capture = (tmp_path / "capture").read_text()
        assert f"--partition={resource}" in capture
        assert ("--gres=gpu:1" in capture) == (resource == "gpu")
        assert "/wrong/" not in capture
        assert "lambda1_20261107_132516" in capture


def test_actual_evaluator_both_axes_emit_matching_population_audits(tmp_path, monkeypatch):
    """Exercise real score/threshold/count/JSON paths, mocking only inference and image I/O."""
    import numpy as np
    import eval_abcd_nurd as evaluator
    from matplotlib.figure import Figure

    rng = np.random.default_rng(7)
    paths = {name: str(tmp_path / name) for name in ("reference", "heldout")}
    latents = {paths[name]: rng.normal(size=(size, 6)).astype(np.float32)
               for name, size in (("reference", 800), ("heldout", 400))}
    ae_scores = {path: rng.lognormal(size=len(z)).astype(np.float32)
                 for path, z in latents.items()}
    signatures = {path: {"n_events": len(z), "identity": path}
                  for path, z in latents.items()}
    metadata = {"weight_scope": "all_events", "effective_physics_weight_sha256": "physical-weights"}
    ae_path = tmp_path / "ae.pth"
    ae_path.write_bytes(b"fixed AE")
    checkpoint = {
        "ae_checkpoint_sha256": comparison.file_hash(ae_path), "ae_scaler": {},
        "preprocessing": {
            "target_scheme": "decay8", "qcd_label": 1,
            "data_signature": signatures[paths["reference"]],
            "weighting": {"generator": metadata},
            "split": {"train_indices": list(range(400)),
                      "validation_indices": list(range(400, 800))},
        },
    }
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(evaluator, "wandb", types.SimpleNamespace(
        init=lambda **kwargs: None, Settings=lambda **kwargs: None,
        run=types.SimpleNamespace(name="test axes"), log=lambda *args: None,
        Image=lambda path: path, finish=lambda: None))
    monkeypatch.setattr(evaluator, "load_nurd_model", lambda *args: (object(), checkpoint))
    monkeypatch.setattr(evaluator, "load_ae", lambda *args: object())
    monkeypatch.setattr(evaluator, "compute_ae_scores", lambda model, scaler, path, device: ae_scores[path])
    monkeypatch.setattr(evaluator, "load_physics_weights", lambda weights, path, label:
                        (np.ones(len(latents[path])), metadata, signatures[path]))
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)

    def embed(model, path, device, batch_size=512, return_logits=False, **kwargs):
        z = latents[path]
        labels = np.ones(len(z), dtype=np.int64)
        if return_logits:
            logits = np.zeros((len(z), 8), dtype=np.float32)
            logits[:, 1] = z[:, 0]
            return z, logits, labels
        return z, labels

    monkeypatch.setattr(evaluator, "embed_pf", embed)
    evaluations = {}
    for mode in comparison.MODES:
        output = tmp_path / mode
        evaluator.ABCD({
            "ckpt": "classifier.pth", "ae_ckpt": str(ae_path),
            "reference_pt": paths["reference"], "reference_weight_path": "reference-weights.npz",
            "test_pt": paths["heldout"], "gen_weight_path": "heldout-weights.npz",
            "outdir": str(output), "target_scheme": "decay8", "qcd_label": 1,
            "axis2_logit": mode == "one_minus_pqcd", "n_pca": 2,
            "statistically_valid_closure": True,
            "min_A": 5, "min_B": 5, "min_C": 5, "min_D": 5,
            "skip_pca_md_plots": True, "skip_embedding_pca": True,
        })
        evaluations[mode] = {
            "diagnostics": json.loads((output / "diagnostics.json").read_text()),
            "thresholds": json.loads((output / "abcd_thresholds.json").read_text())}
    comparison.verify_same_population(evaluations)

