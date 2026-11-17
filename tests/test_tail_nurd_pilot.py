import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_tail_nurd_pilot import FixedBatches, finite_report, plan_populations


def test_fixed_budget_cycles_without_partial_batches():
    dataset = torch.utils.data.TensorDataset(torch.arange(10))
    batches = list(FixedBatches(dataset, 4, 7, 42))
    assert len(batches) == 7
    assert all(len(batch[0]) == 4 for batch in batches)
    assert len(FixedBatches(dataset, 4, 7, 42)) == 7
    with pytest.raises(ValueError, match="full batch"):
        FixedBatches(dataset, 11, 2, 42)


def test_threshold_uses_only_calibration_and_excludes_it_from_both_arms():
    scores = np.arange(100, dtype=float)
    weights = np.ones(100)
    calibration = np.arange(0, 100, 5)
    train = np.arange(90)
    cut, full, tail = plan_populations(scores, weights, train, calibration)
    assert cut == 85
    assert not np.intersect1d(full, calibration).size
    assert not np.intersect1d(tail, calibration).size
    assert set(tail) <= set(full)
    changed = scores.copy()
    changed[full] = 10000
    assert plan_populations(changed, weights, train, calibration)[0] == cut
    # Physical, not raw, quantiles. A large low-score weight moves the cut.
    weights[0] = 1000
    assert plan_populations(scores, weights, train, calibration)[0] == 0


def test_finite_report_does_not_mistake_undefined_closure_for_zero():
    result = finite_report({"closure": np.float64(np.inf), "values": [np.nan, 1.0]})
    assert result == {"closure": None, "values": [None, 1.0]}
    json.dumps(result, allow_nan=False)


def test_worker_requests_generic_gpu_and_no_retraining_ae():
    worker = (
        Path(__file__).parents[1] / "slurm/submit_tail_nurd_pilot.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:1" in worker
    for disallowed in (
        "--partition",
        "--constraint",
        "--nodelist",
        "a100",
        "train_ae.py",
    ):
        assert disallowed not in worker.lower()
    assert "scripts.run_tail_nurd_pilot" in worker


def test_tiny_paired_continuation_keeps_sources_and_checks_only_after_both_finish(
    tmp_path, monkeypatch
):
    import scripts.run_tail_nurd_pilot as pilot
    from models.hlt_autoencoder import HLTAutoencoder
    from models.hlt_con import HLTCritic
    from scripts.audit_qcd_abcd_composition import infer_ae_scores
    from train_hlt import build_parser
    from utils.hlt_targets import DECAY8_NAMES
    from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature

    torch.manual_seed(11)
    n = 1600
    parents = torch.tensor([0, 1, 2, 3, 3, 3, 0, 0]).repeat(200)
    categories = torch.tensor([15, 0, 27, 6, 9, 12, 18, 21]).repeat(200)

    def sample(offset):
        pf = torch.randn(n, 6, 7) * 0.1
        pf[:, :, 0] = torch.rand(n, 6) + 1
        pf[:, :, 5] = 1
        pf[:, :, 6] = 211
        obj = torch.arange(1, 201, dtype=torch.float32).repeat_interleave(8) / 50
        return {
            "pf": pf,
            "obj": obj[:, None, None].expand(n, 2, 4).clone(),
            "label": parents,
            "background_category": categories,
            "event_id": torch.stack(
                (parents, parents * 0, parents * 0, torch.arange(n) + offset), 1
            ),
        }

    raw = sample(0)
    check = sample(10000)
    data = tmp_path / "sample.pt"
    heldout = tmp_path / "check.pt"
    weights = tmp_path / "weights.pt"
    heldout_weights = tmp_path / "check_weights.pt"
    torch.save(raw, data)
    torch.save(check, heldout)
    for path, source in ((weights, raw), (heldout_weights, check)):
        torch.save(
            {
                "weights": torch.ones(n),
                "event_id": source["event_id"],
                "weight_scope": "all_events",
            },
            path,
        )
    _, metadata = load_generator_weights(
        str(weights), parents, sample=raw, weight_scope="all_events"
    )
    signature = sample_signature(raw)
    ae_config = {
        "features": 8,
        "latent_dim": 3,
        "encoder_config": {"nodes": [12]},
        "decoder_config": {"nodes": [12, 8]},
        "alpha": 1.0,
    }
    ae_model = HLTAutoencoder(ae_config)
    ae_model.eval()
    scaler = {"mu": torch.zeros(8), "std": torch.ones(8)}
    calibration = torch.nonzero(torch.arange(n) // 8 % 5 == 0).flatten()
    ae_train = torch.nonzero(torch.arange(n) // 8 % 5 != 0).flatten()
    ae = tmp_path / "ae.pth"
    torch.save(
        {
            "ae": ae_model.state_dict(),
            "ae_config": ae_config,
            "ae_scaler": scaler,
            "data_signature": signature,
            "split": {"train_indices": ae_train, "validation_indices": calibration},
            "weighting": {"scheme": "physics"},
        },
        ae,
    )
    ae_hash = file_sha256(ae)
    scores = infer_ae_scores(ae_model, scaler, raw, torch.arange(n), "cpu", 512)
    cache = tmp_path / "cache"
    cache.mkdir()
    from scripts.audit_tail_pilot_split import quantile

    expected_cut = float(quantile(scores[calibration], np.ones(len(calibration)), 0.9))
    (cache / "AE_control_summary.json").write_text(
        json.dumps(
            {
                "baseline_sha256": ae_hash,
                "results": {"baseline": {"ae_threshold": expected_cut}},
            }
        )
    )
    np.savez_compressed(
        cache / "frozen_AE_scores.npz",
        training_event_id=raw["event_id"].numpy(),
        heldout_event_id=check["event_id"].numpy(),
        baseline_training_manifest_scores=scores,
        baseline_heldout_scores=scores.copy(),
    )
    config = vars(
        build_parser().parse_args(["--data", str(data), "--ae_ckpt", str(ae)])
    )
    config.update(
        embed_size=8,
        latent_dim=3,
        proj_dim=3,
        num_heads=2,
        num_layers=1,
        dim_ff=16,
        linear_dim=3,
        dropout=0.0,
        contrast_weight=0.3,
        balance_strata=1,
        balance_clip_quantile=0.995,
        critic_steps=2,
        critic_shuffle_mode="weighted_within_class",
    )
    source_model = pilot.model_from_config(config, 6, "cpu")
    source_critic = HLTCritic(3, 8)
    original_val = torch.nonzero(torch.arange(n) // 8 % 7 == 0).flatten()
    original_train = torch.nonzero(torch.arange(n) // 8 % 7 != 0).flatten()
    pre = {
        "data_signature": signature,
        "target_scheme": "decay8",
        "target_class_names": DECAY8_NAMES,
        "label_map": {i: i for i in range(8)},
        "nuisance_transform": {"mean": 0.0, "std": 1.0},
        "split": {"train_indices": original_train, "validation_indices": original_val},
        "weighting": {"generator": metadata},
    }
    source = tmp_path / "source.pth"
    torch.save(
        {
            "state_dict_model": source_model.state_dict(),
            "state_dict_critic": source_critic.state_dict(),
            "config": config,
            "preprocessing": pre,
            "ae_scaler": scaler,
            "ae_checkpoint_sha256": ae_hash,
            "nurd_active": True,
            "effective_lambda_info": 1.0,
            "epoch": 22,
        },
        source,
    )
    source_hash = file_sha256(source)
    out = tmp_path / "pilot"
    args = argparse.Namespace(
        checkpoint=source,
        ae_checkpoint=ae,
        sample=data,
        weights=weights,
        ae_cache_dir=cache,
        heldout=heldout,
        heldout_weights=heldout_weights,
        output_dir=out,
        updates=2,
        batch_size=16,
        report_every=1,
        save_every=2,
        lr=3e-5,
        seed=42,
        code_commit="test",
    )
    actual_load = pilot.load_hlt_sample

    def guarded_load(path):
        if path == heldout:
            for arm in ("full", "tail"):
                assert (out / arm / "checkpoint_final.pth.tar").is_file()
        return actual_load(path)

    monkeypatch.setattr(pilot, "load_hlt_sample", guarded_load)
    actual_epoch = pilot.train_epoch
    calls = []

    def checked_epoch(model, critic, *rest):
        if len(calls) in (0, 2):
            for key, value in source_model.state_dict().items():
                assert torch.equal(value, model.state_dict()[key])
            for key, value in source_critic.state_dict().items():
                assert torch.equal(value, critic.state_dict()[key])
        calls.append(len(rest[0]))
        return actual_epoch(model, critic, *rest)

    monkeypatch.setattr(pilot, "train_epoch", checked_epoch)
    pilot.run(args)
    assert calls == [1, 1, 1, 1]
    assert file_sha256(source) == source_hash and file_sha256(ae) == ae_hash
    manifest = json.loads((out / "pilot_manifest.json").read_text())
    assert manifest["constant_lambda"] == 1 and manifest["updates_per_branch"] == 2
    roles = np.load(out / "pilot_roles.npz")
    for arm in ("full", "tail"):
        payload = torch.load(out / arm / "checkpoint_final.pth.tar", weights_only=False)
        assert payload["encoder_updates"] == 2
        assert payload["pilot_protocol"] == "fixed_budget_full_vs_tail"
        assert not np.intersect1d(roles[f"{arm}_train"], roles["calibration"]).size
        history = json.loads((out / arm / "training_history.json").read_text())
        assert history[-1]["critic_updates"] == 4
    report = json.loads((out / "pilot_comparison.json").read_text())
    assert report["source_checkpoints_unchanged"]
    assert set(report["results"]) == {"source", "full", "tail"}
    assert len(list((out / "plots").glob("*.png"))) == 8
    assert (out / "class_statistics.csv").is_file()
    with pytest.raises(FileExistsError):
        pilot.run(args)
