import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.run_double_nurd_pilot as pilot
from models.hlt_autoencoder import HLTAutoencoder
from models.hlt_con import HLTCritic
from scripts.audit_qcd_abcd_composition import infer_ae_scores
from scripts.audit_tail_pilot_split import quantile
from scripts.evaluate_double_nurd_pilot import classification, grid_summary
from train_hlt import SupConLoss, build_parser
from utils.hlt_targets import DECAY8_NAMES
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


def test_tail_schedule_and_budget_require_full_strength_period():
    assert [pilot.tail_lambda(i, 4, 1.) for i in (1, 4, 5)] == [.25, 1., 1.]
    assert pilot.tail_lambda(1, 0, 2.) == 2.
    with pytest.raises(ValueError): pilot.tail_lambda(0, 4, 1.)
    args = argparse.Namespace(updates=4, batch_size=8, report_every=1, save_every=2,
        warmup_updates=2, critic_steps=1, min_post_ramp_updates=3, ramp_updates=2,
        lr=3e-5, global_lambda=1., tail_lambda=1.)
    with pytest.raises(ValueError, match="complete ramp"): pilot.validate_options(args)
    args.updates = 5; pilot.validate_options(args)


def tiny_config():
    return dict(embed_size=8, latent_dim=3, proj_dim=3, num_heads=2, num_layers=1,
                dim_ff=16, linear_dim=3, dropout=.1)


def batch(qcd=False):
    x = torch.randn(16, 6, 7)*.1
    x[..., 0] = torch.rand(16, 6)+1.; x[..., 5] = 1.; x[..., 6] = 211
    y = torch.ones(16, dtype=torch.long) if qcd else torch.arange(8).repeat(2)
    return x, y, torch.arange(16).float()/10, torch.ones(16), torch.ones(16), torch.ones(16)


def test_critic_warmup_freezes_encoder_including_bn_and_information_only_moves_encoder():
    model = pilot.model_from_config(tiny_config(), 6, "cpu")
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    critic = HLTCritic(3, 8)
    optimizer = torch.optim.Adam(critic.parameters(), lr=.001)
    pilot.critic_step(model, critic, optimizer, batch(True), "cpu", 2)
    assert model.training
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    penalty = pilot.information(model, critic, batch(True), "cpu")
    assert penalty.requires_grad
    penalty.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert all(p.requires_grad for p in critic.parameters())
    assert model.training


def test_physical_sampling_uses_weights_once_and_never_leaves_tail():
    b = batch(True)
    rows = torch.arange(2)
    dataset = pilot.SafeDataset(b[0], b[1], b[2], b[3], torch.ones(2), b[5], rows, "tail")
    stream = pilot.PhysicalQCDStream(dataset, np.array([1., 9.]), 64, 42)
    samples = torch.cat([next(stream)[2] for _ in range(20)])
    assert set(samples.tolist()) <= set(b[2][:2].tolist())
    assert .85 < float((samples == b[2][1]).float().mean()) < .95
    assert torch.all(next(stream)[4] == 1)
    dataset.effective_weights[0] = 2
    with pytest.raises(ValueError, match="unit loss"): pilot.PhysicalQCDStream(dataset, [1, 9], 8, 42)


def test_double_step_has_independent_critics_and_tail_gradient():
    model = pilot.model_from_config(tiny_config(), 6, "cpu")
    critics = {name: HLTCritic(3, 8) for name in ("global", "tail")}
    assert not ({p.data_ptr() for p in critics['global'].parameters()} & {p.data_ptr() for p in critics['tail'].parameters()})
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    result = pilot.encoder_step(model, critics, optimizer, batch(),
        {name: batch(True) for name in critics}, "cpu", SupConLoss(.05), 1., .5)
    assert result["tail_lambda"] == .5 and set(result["information"]) == {"global", "tail"}
    assert all(p.grad is None for c in critics.values() for p in c.parameters())


def test_grid_summaries_keep_historical_and_ratio_bias_definitions_distinct():
    result = grid_summary({"valid": np.array([True, True, False]),
                           "ratio": np.array([.5, .25, np.nan])})
    assert result['valid_points'] == 2
    assert result['legacy_absolute_nonclosure']['median'] == 2.
    assert result['absolute_ratio_bias']['median'] == .625
    assert grid_summary({"valid": np.array([False]), 'ratio': np.array([np.nan])})['legacy_absolute_nonclosure'] is None


def test_parent_aggregation_sums_probabilities_instead_of_relabeling_argmax():
    # Z's three moderate channels can outrank the largest individual W channel.
    state = {'classifier.weight': torch.zeros(8, 3), 'classifier.bias': torch.tensor([1., -8., -8., 1.5, -8., -8., 1., 1.])}
    _, spec = pilot.fit_joint_balance(torch.arange(8), torch.ones(8), torch.ones(8, dtype=torch.float64), 1, 'log_fixed', .995)
    result = classification(np.zeros((8, 3)), {'state_dict_model': state, 'balance_spec': spec},
                            np.arange(8), np.ones(8), np.ones(8))
    assert np.asarray(result['raw_confusion'])[:, 3].sum() == 8
    assert np.asarray(result['parent4_raw_confusion'])[:, 0].sum() == 8


@pytest.fixture
def experiment(tmp_path):
    torch.manual_seed(11)
    n = 1600
    parents = torch.tensor([0, 1, 2, 3, 3, 3, 0, 0]).repeat(200)
    categories = torch.tensor([15, 0, 27, 6, 9, 12, 18, 21]).repeat(200)
    def sample(offset):
        pf = torch.randn(n, 6, 7)*.1
        pf[..., 0] = torch.rand(n, 6)+1; pf[..., 5] = 1; pf[..., 6] = 211
        obj = torch.arange(1, 201).float().repeat_interleave(8)/50
        return {"pf": pf, "obj": obj[:, None, None].expand(n, 2, 4).clone(),
            "label": parents, "background_category": categories,
            "event_id": torch.stack((parents, parents*0, parents*0, torch.arange(n)+offset), 1)}
    raw, check = sample(0), sample(10000)
    data, heldout = tmp_path/'sample.pt', tmp_path/'check.pt'
    weight_path, check_weight_path = tmp_path/'weights.pt', tmp_path/'check_weights.pt'
    torch.save(raw, data); torch.save(check, heldout)
    for path, value in ((weight_path, raw), (check_weight_path, check)):
        torch.save({"weights": torch.ones(n), "event_id": value["event_id"], "weight_scope": "all_events"}, path)
    _, metadata = load_generator_weights(str(weight_path), parents, sample=raw, weight_scope='all_events')
    signature = sample_signature(raw)
    config_ae = {"features": 8, "latent_dim": 3, "encoder_config": {"nodes": [12]},
                 "decoder_config": {"nodes": [12, 8]}, "alpha": 1.}
    ae_model = HLTAutoencoder(config_ae); ae_model.eval()
    scaler = {"mu": torch.zeros(8), "std": torch.ones(8)}
    calibration = torch.nonzero(torch.arange(n)//8 % 5 == 0).flatten()
    ae_train = torch.nonzero(torch.arange(n)//8 % 5 != 0).flatten()
    ae_path = tmp_path/'ae.pth'
    torch.save({"ae": ae_model.state_dict(), "ae_config": config_ae,
        "ae_scaler": scaler, "data_signature": signature, "weighting": {"scheme": "physics"},
        "split": {"train_indices": ae_train, "validation_indices": calibration}}, ae_path)
    ae_hash = file_sha256(ae_path)
    scores = infer_ae_scores(ae_model, scaler, raw, torch.arange(n), "cpu", 512)
    cache = tmp_path/'cache'; cache.mkdir()
    (cache/'AE_control_summary.json').write_text(json.dumps({"baseline_sha256": ae_hash,
        "results": {"baseline": {"ae_threshold": float(quantile(scores[calibration], np.ones(len(calibration)), .9))}}}))
    np.savez_compressed(cache/'frozen_AE_scores.npz', training_event_id=raw['event_id'],
        heldout_event_id=check['event_id'], baseline_training_manifest_scores=scores, baseline_heldout_scores=scores.copy())
    cfg = vars(build_parser().parse_args(['--data', str(data), '--ae_ckpt', str(ae_path)]))
    cfg.update(**tiny_config(), contrast_weight=.3, balance_strata=1, balance_clip_quantile=.995)
    original_train = torch.nonzero(torch.arange(n)//8 % 7 != 0).flatten()
    original_val = torch.nonzero(torch.arange(n)//8 % 7 == 0).flatten()
    pre = {"data_signature": signature, "target_scheme": "decay8", "target_class_names": DECAY8_NAMES,
        "label_map": {i: i for i in range(8)}, "nuisance_transform": {"mean": 0., "std": 1.},
        "split": {"train_indices": original_train, "validation_indices": original_val},
        "weighting": {"generator": metadata}}
    source_path = tmp_path/'source.pth'
    model = pilot.model_from_config(cfg, 6, "cpu")
    torch.save({"state_dict_model": model.state_dict(), "state_dict_critic": HLTCritic(3, 8).state_dict(),
        "config": cfg, "preprocessing": pre, "ae_scaler": scaler,
        "ae_checkpoint_sha256": ae_hash, "nurd_active": True, "effective_lambda_info": 1., "epoch": 22}, source_path)
    return argparse.Namespace(checkpoint=source_path, ae_checkpoint=ae_path, sample=data, weights=weight_path,
        ae_cache_dir=cache, output_dir=tmp_path/'experiment', updates=2, batch_size=16,
        warmup_updates=1, ramp_updates=1, min_post_ramp_updates=1, critic_steps=1,
        global_lambda=1., tail_lambda=1., lr=3e-5, report_every=1, save_every=2,
        seed=42, device='cpu', code_commit='test'), heldout, check_weight_path


def test_tiny_training_then_full_and_tail_evaluation_without_source_mutation(experiment, monkeypatch):
    args, heldout, check_weights = experiment
    source_hash, ae_hash = file_sha256(args.checkpoint), file_sha256(args.ae_checkpoint)
    warm_states = []
    original_encoder_step = pilot.encoder_step
    def guarded_step(model, critics, *rest):
        if len(warm_states) in (0, 2):
            warm_states.append({k: v.clone() for k, v in critics['global'].state_dict().items()})
        else: warm_states.append(None)
        return original_encoder_step(model, critics, *rest)
    monkeypatch.setattr(pilot, 'encoder_step', guarded_step)
    pilot.run(args)
    assert all(torch.equal(v, warm_states[2][k]) for k, v in warm_states[0].items())
    assert file_sha256(args.checkpoint) == source_hash and file_sha256(args.ae_checkpoint) == ae_hash
    roles = np.load(args.output_dir/'training_roles.npz')
    for key in ('full_train', 'tail_train', 'global_QCD', 'tail_QCD'):
        assert not np.intersect1d(roles[key], roles['calibration']).size
    assert set(roles['tail_QCD']) <= set(roles['global_QCD'])
    for arm in pilot.ARMS:
        payload = torch.load(args.output_dir/arm/'checkpoint_final.pth.tar', weights_only=False)
        assert payload['encoder_updates'] == 2
        assert (payload['state_dict_tail_critic'] is None) == (arm == 'global')
        assert payload['checkpoint_role'] == 'fixed_budget_final'
    import scripts.evaluate_double_nurd_pilot as evaluation
    original_load = evaluation.load_hlt_sample
    def guarded_load(path):
        if path == heldout:
            assert (args.output_dir/'training_complete.json').is_file()
            frozen = json.loads((args.output_dir/'evaluation/frozen_comparison_plan.json').read_text())
            assert set(frozen['plans']) == {'global', 'double'}
            assert not frozen['checking_opened_during_selection']
        return original_load(path)
    monkeypatch.setattr(evaluation, 'load_hlt_sample', guarded_load)
    eval_args = argparse.Namespace(experiment_dir=args.output_dir, sample=args.sample, weights=args.weights,
        heldout=heldout, heldout_weights=check_weights, ae_cache_dir=args.ae_cache_dir,
        output_dir=args.output_dir/'evaluation', device='cpu', batch_size=64,
        minimums=(1, 1, 1, 1), kde_grid=8, code_commit='test')
    evaluation.run(eval_args)
    report = json.loads((eval_args.output_dir/'double_nurd_comparison.json').read_text())
    assert set(report['results']) == {'global', 'double'}
    assert set(report['results']['double']) == {'f100', 'f010'}
    assert len(list((eval_args.output_dir/'plots').rglob('hist2d_pca_md_kde.png'))) == 4
    assert report['original_checkpoints_unchanged']
    assert 'MD_references' in report and not report['signal_sample_evaluated']
    with pytest.raises(FileExistsError): pilot.run(args)


def test_workers_request_only_generic_gpu_and_conda_before_nounset():
    root = Path(__file__).parents[1]
    for stage in ('train', 'eval'):
        text = (root/f'slurm/submit_double_nurd_{stage}.sbatch').read_text()
        assert '#SBATCH --gres=gpu:1' in text
        for word in ('--partition', '--constraint', '--nodelist', 'a100', 'train_ae.py'):
            assert word not in text.lower()
        assert text.index('conda activate disco') < text.index('set -u')
    launch = (root/'slurm/launch_double_nurd_experiment.sh').read_text()
    assert 'afterok:' in launch and 'git archive HEAD' in launch
    assert 'double-nurd-experiment' in launch
