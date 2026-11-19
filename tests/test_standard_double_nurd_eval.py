import argparse
import copy
import json
from pathlib import Path
import types
import sys

import numpy as np
import pytest
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))
import eval_abcd_nurd as canonical
import scripts.standard_double_nurd_eval as suite
from scripts.prepare_tail_pilot import report_json
from scripts.run_double_nurd_pilot import PROTOCOL
from scripts.run_tail_nurd_pilot import finite_report
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


def test_fast_standard_plan_and_check_match_canonical_strict_weighted_counts():
    rng = np.random.default_rng(55)
    data = {"ae": rng.integers(1, 20, 1200).astype(float), "md": rng.integers(1, 20, 1200).astype(float),
            "weight": rng.uniform(.1, 5, 1200)}
    minima = [1, 1, 1, 1]
    plan = suite.standard_plan(data, minima)
    best = np.inf
    for p in suite.PERCENTILES:
        for q in suite.PERCENTILES:
            tx, ty, counts = canonical.abcd_statistics(data['ae'], data['md'], p, q, data['weight'])
            if not canonical.statistically_valid_regions(counts, dict(zip('ABCD', minima))): continue
            value = abs(canonical.nonclosure_A(*canonical.abcd_yields(counts))[0])
            best = min(best, value)
    assert plan['candidate']['absolute_nonclosure'] == pytest.approx(best)
    # A held-out shift cannot alter the numerical chosen calibration cuts.
    shifted = {**data, 'md': data['md'] * 1.1}
    result, _ = suite.standard_check(shifted, plan, minima, data['weight'].sum())
    assert result['working_point']['md_cut'] == plan['candidate']['md_cut']
    point = result['working_point']
    stats = canonical.abcd_region_statistics_at_thresholds(shifted['ae'], shifted['md'], point['ae_cut'], point['md_cut'], shifted['weight'])
    assert point['predicted_over_observed'] == pytest.approx(canonical.closure_ratio_and_uncertainty(stats)[0])


def test_parent_view_aggregates_probabilities_without_changing_QCD_closure():
    rng = np.random.default_rng(8)
    n = 800
    data = {'ae': np.exp(rng.normal(size=n)), 'md': np.exp(rng.normal(size=n)),
            'weight': rng.uniform(1, 4, n), 'label': np.tile(np.arange(8), 100)}
    probs = np.tile(np.array([.2, .01, .01, .25, .01, .01, .2, .31]), (n, 1))
    view, parents, _ = suite.taxonomy_view(data, probs, 'parent4')
    assert np.array_equal(parents[:, 1], probs[:, 1])
    assert np.allclose(parents[:, 0], probs[:, [0, 6, 7]].sum(1))
    assert np.array_equal(view['label'] == 1, data['label'] == 1)
    plan = suite.standard_plan(suite.qcd(data), [1]*4)
    left, _ = suite.standard_check(suite.qcd(data), plan, [1]*4, 100.)
    right, _ = suite.standard_check(suite.qcd(view), plan, [1]*4, 100.)
    assert finite_report(left) == finite_report(right)
    # Original array labels have not been overwritten.
    assert set(data['label']) == set(range(8))


def test_absent_valid_cells_are_not_a_fabricated_working_point():
    data = {'ae': np.arange(20)+1., 'md': np.ones(20), 'weight': np.ones(20)}
    plan = suite.standard_plan(data, [50, 50, 50, 100])
    assert plan['candidate'] is None
    result, _ = suite.standard_check(data, plan, [50, 50, 50, 100], 20.)
    assert result['working_point'] is None
    assert result['grid_median_absolute_nonclosure'] is None


def test_taxonomy_metrics_distinguish_physical_and_raw_recall():
    probabilities = np.array([[.9, .1], [.1, .9], [.1, .9], [.1, .9]])
    result = suite.class_metrics(np.array([0, 0, 1, 1]), probabilities, np.array([9., 1., 1., 1.]), 2)
    assert result['per_class_accuracy'] == [.5, 1.]
    assert result['per_class_physics_accuracy'] == [.9, 1.]
    assert result['balanced_accuracy'] == .75
    assert result['physics_balanced_accuracy'] == .95


def test_complete_two_stage_suites_preserve_caches_and_freeze_before_checking(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    n = 1600
    labels = np.tile(np.arange(8), n//8)
    raw = {'label': torch.as_tensor(suite.PARENT[labels]), 'pf': torch.zeros(n, 1, 7),
           'obj': torch.zeros(n, 1, 4), 'event_id': torch.arange(n)[:, None]}
    sample, weights_path = tmp_path/'sample.pt', tmp_path/'weights.pt'
    torch.save(raw, sample)
    physics = rng.uniform(.5, 2, n)
    torch.save({'weights': torch.from_numpy(physics), 'event_id': raw['event_id'], 'weight_scope': 'all_events'}, weights_path)
    _, meta = load_generator_weights(str(weights_path), raw['label'], sample=raw, weight_scope='all_events')
    comparison, experiment = tmp_path/'comparison', tmp_path/'experiment'
    comparison.mkdir(); experiment.mkdir()
    config = dict(embed_size=8, latent_dim=3, proj_dim=3, num_heads=2, num_layers=1, dim_ff=16, linear_dim=3, dropout=0.)
    latent = rng.normal(size=(n, 3)).astype(np.float32)
    ae = np.exp(rng.normal(size=n))
    data = {'ae': ae, 'md': (latent.astype(float)**2).sum(1), 'weight': physics,
            'label': labels, 'event_id': raw['event_id'].numpy()}
    hashes = {}
    from scripts.run_tail_nurd_pilot import model_from_config
    for arm in ('global', 'double'):
        model = model_from_config(config, 1, 'cpu')
        folder = experiment/arm; folder.mkdir()
        payload = {'state_dict_model': model.state_dict(), 'config': config, 'arm': arm,
            'encoder_updates': 10, 'pilot_protocol': PROTOCOL, 'ae_checkpoint_sha256': 'AE',
            'source_sha256': 'source', 'preprocessing': {'target_scheme': 'decay8'}}
        torch.save(payload, folder/'checkpoint_final.pth.tar')
        hashes[arm] = file_sha256(folder/'checkpoint_final.pth.tar')
        np.savez_compressed(comparison/f'{arm}_checking_all_classes.npz', **data)
        cal = {k: v[labels == 1].copy() for k, v in data.items()}
        cal['event_id'] += 10000
        np.savez_compressed(comparison/f'{arm}_calibration_QCD.npz', **cal)
    source = {'protocol': PROTOCOL, 'checkpoint_sha256': hashes, 'checking_signature': sample_signature(raw),
        'checking_physics_weights': meta, 'AE_thresholds': {'f100': None, 'f010': float(np.quantile(ae, .9))},
        'MD_references': {arm: {'mean': [0., 0., 0.], 'whitening': np.eye(3).tolist()} for arm in ('global', 'double')}}
    (comparison/'double_nurd_comparison.json').write_text(report_json(source))
    (experiment/'training_complete.json').write_text(report_json({'protocol': PROTOCOL, 'final_checkpoint_sha256': hashes,
        'updates_per_arm': 10, 'AE_sha256': 'AE', 'source_sha256': 'source'}))
    monkeypatch.setattr('utils.hlt_targets.checkpoint_target_labels', lambda *args: torch.as_tensor(labels))
    original_load = suite.load_hlt_sample
    output_seen = []
    def guarded_load(path):
        out = output_seen[-1]
        plan = json.loads((out/'standard_threshold_plan.json').read_text())
        assert not plan['checking_used_to_choose_thresholds']
        assert set(plan['plans']) == {'global', 'double'}
        return original_load(path)
    monkeypatch.setattr(suite, 'load_hlt_sample', guarded_load)
    inference_calls = []
    def inference(*args):
        inference_calls.append(1)
        return latent
    monkeypatch.setattr(suite, 'infer_latents', inference)
    args = argparse.Namespace(comparison_dir=comparison, experiment_dir=experiment, heldout=sample,
        heldout_weights=weights_path, output_dir=tmp_path/'predictions', device='cpu', minimums=[1]*4,
        batch_size=128, kde_grid=8, inference_only=True, prediction_cache=None, code_commit='test')
    output_seen.append(args.output_dir)
    suite.run(args)
    assert len(inference_calls) == 2
    assert (args.output_dir/'inference_complete.json').is_file()
    assert not list(args.output_dir.rglob('*.png'))
    inference_dir = args.output_dir
    args = copy.copy(args); args.output_dir = tmp_path/'plots'; args.inference_only=False; args.prediction_cache=inference_dir
    output_seen.append(args.output_dir)
    # CPU stage must never invoke encoder inference.
    monkeypatch.setattr(suite, 'infer_latents', lambda *a: pytest.fail('CPU plot job ran PF inference'))
    rendered = []
    actual_render = suite.render_suite
    def render(*a, **kw):
        rendered.append(a[5])
        # Fully exercise one ordinary suite; validate paths/data for the other seven.
        if len(rendered) == 1: return actual_render(*a, **kw)
        a[5].mkdir(parents=True)
        assert np.array_equal(a[3]['percentiles'], suite.PERCENTILES)
        return {'signal_sample_evaluated': False}
    monkeypatch.setattr(suite, 'render_suite', render)
    suite.run(args)
    assert len(rendered) == 8
    report = json.loads((args.output_dir/'standard_evaluation_summary.json').read_text())
    assert report['original_inputs_unchanged'] and not report['parent4_changes_QCD_ABCD']
    for arm in ('global', 'double'):
        for fraction in ('f100', 'f010'):
            root = args.output_dir/arm/fraction
            eight = json.loads((root/'decay8/evaluation_summary.json').read_text())
            four = json.loads((root/'parent4/evaluation_summary.json').read_text())
            assert eight['closure'] == four['closure']
            assert len(four['classification']['raw_confusion']) == 4
            assert (root/'parent4/plots/confusion_matrix.png').is_file()
    assert (rendered[0]/'closure_scan_2d.png').is_file()
    assert (rendered[0]/'pca_corner.png').is_file()
    assert (rendered[0]/'hist2d_pca_md_kde.png').is_file()
    with pytest.raises(FileExistsError): suite.run(args)


def test_launcher_separates_gpu_inference_and_cpu_plots_without_training():
    root = Path(__file__).parents[1]
    launch = (root/'slurm/launch_double_nurd_standard_eval.sh').read_text()
    worker = (root/'slurm/submit_double_nurd_standard_eval.sbatch').read_text()
    assert '--gres=gpu:1' in launch and '--partition=cpu' in launch and 'afterok:' in launch
    assert '--partition=gpu' not in launch and 'a100' not in launch.lower()
    assert 'git archive HEAD' in launch
    assert '--inference-only' in worker and '--prediction-cache' in worker
    assert 'train_hlt' not in worker and 'train_ae' not in worker
    assert worker.index('conda activate disco') < worker.index('set -u')
