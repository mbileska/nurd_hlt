"""Frozen full/tail models at 100%, 20%, 10% AE retention; no training.

All preselection thresholds and iso-overall-QCD-efficiency cut choices are
frozen on AE-gradient-unseen calibration rows before checking data are opened.
"""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.audit_qcd_abcd_composition import infer_latents
from scripts.audit_tail_pilot_split import quantile, retention
from scripts.prepare_tail_pilot import report_json, weight_stats
from scripts.run_tail_nurd_pilot import finite_report
from scripts.scan_cached_tail_cuts import scan
from scripts.study_tail_pilot_cuts import load_nurd_model, recount, qcd, save_scores
from scripts.compare_qcd_score_audit import paired_bootstrap
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import file_sha256, sample_signature, load_generator_weights

FRACTIONS = (1., .2, .1)
MINIMUMS = (50, 50, 50, 100)
TARGETS = (.005, .01, .02)
PROBABILITIES = np.linspace(.05, .95, 91)
GRID_PROBABILITIES = np.linspace(.05, .99, 95)


def fraction_name(fraction):
    return f'f{int(round(fraction*100)):03d}'


def target_name(target):
    return f'A{target:.4f}'


def subset(data, cut):
    mask = np.ones(len(data['ae']), bool) if cut is None else data['ae'] >= cut
    return {key: value[mask] for key, value in data.items()}


def thresholds(scores, weights):
    """All-background physical calibration, not training or checking quantiles."""
    return {fraction_name(f): None if f == 1 else float(quantile(scores, weights, 1-f)) for f in FRACTIONS}


def at_cuts(data, cuts, full_qcd_mass, minimums):
    value, codes = recount(data, cuts[0], cuts[1], minimums)
    value['A_efficiency_overall_QCD'] = value['regions']['A']['yield'] / full_qcd_mass
    value['preselection_QCD_retention'] = float(data['weight'].sum() / full_qcd_mass)
    value['passes_legacy_D500'] = bool(all(value['regions'][r]['effective_count']>=m for r,m in zip('ABCD',(50,50,50,500))))
    return value, codes


def matched_curve(data, full_qcd_mass, target, minimums=MINIMUMS,
                  probabilities=PROBABILITIES, match_tolerance=.05):
    """Fit MD cuts conditional on each AE cut to match an OVERALL QCD A rate.

    Cut selection uses calibration alone. Each adjacent point has the same
    target rate, unlike unconstrained neighboring cells of a 2D grid.
    """
    curve = []
    if not len(data['ae']) or data['weight'].sum()<=0 or full_qcd_mass<=0:
        raise ValueError('Selected calibration QCD must have positive physical mass')
    for probability in probabilities:
        ae_cut = float(quantile(data['ae'], data['weight'], probability))
        high = data['ae'] > ae_cut
        high_mass = data['weight'][high].sum()
        desired = target * full_qcd_mass
        if not np.isfinite(ae_cut) or not 0 < desired < high_mass:
            curve.append(None); continue
        md_cut = float(quantile(data['md'][high], data['weight'][high], 1-desired/high_mass))
        value, _ = at_cuts(data, (ae_cut, md_cut), full_qcd_mass, minimums)
        value['AE_percentile_within_preselection'] = float(probability)
        value['target_overall_QCD_A_efficiency'] = target
        value['acceptance_match'] = bool(abs(value['A_efficiency_overall_QCD']/target - 1) <= match_tolerance)
        value['eligible'] = bool(value['statistically_valid'] and value['acceptance_match'])
        curve.append(value)
    objectives = np.full(len(curve), np.inf)
    for i in range(1, len(curve)-1):
        neighbors = curve[i-1:i+2]
        if all(p is not None and p['eligible'] for p in neighbors):
            objectives[i] = np.median([abs(p['predicted_over_observed']-1) for p in neighbors]) + curve[i]['ratio_uncertainty']
    chosen = int(np.argmin(objectives)) if np.isfinite(objectives).any() else None
    candidate = None
    if chosen is not None:
        candidate = dict(curve[chosen])
        candidate['curve_index'] = chosen
        candidate['neighborhood_median_abs_ratio_bias'] = float(np.median([
            abs(p['predicted_over_observed']-1) for p in curve[chosen-1:chosen+2]]))
        candidate['neighborhood_max_abs_ratio_bias'] = float(max(
            abs(p['predicted_over_observed']-1) for p in curve[chosen-1:chosen+2]))
    return {'target_overall_QCD_A_efficiency': target, 'curve': curve,
            'candidate': candidate, 'eligible_curve_points': sum(p is not None and p['eligible'] for p in curve)}


def checking_curve(data, full_qcd_mass, calibration_curve, minimums):
    # No checking quantile, cut refit, ranking or acceptance matching here.
    return [None if p is None else at_cuts(data, (p['ae_cut'], p['md_cut']), full_qcd_mass, minimums)[0]
            for p in calibration_curve]


def role_selection(scores, weights, labels, cut):
    effective_cut = -np.inf if cut is None else cut
    return {'all_backgrounds': retention(scores, weights, effective_cut),
            'QCD': retention(scores[labels==1], weights[labels==1], effective_cut),
            'classes': {i: weight_stats(weights[(labels==i) & (scores>=effective_cut)]) for i in range(8)}}


def verify_snapshot(payload, pilot, branch):
    if (payload.get('branch') != branch or payload.get('encoder_updates') != pilot['updates_per_branch']
            or payload['ae_checkpoint_sha256'] != pilot['AE_sha256']):
        raise ValueError('Expected the untouched fixed-budget full/tail final checkpoint')


def load_weights(path, sample, expected):
    weights, metadata = load_generator_weights(str(path), sample['label'], sample=sample,
                                               qcd_label=1, weight_scope='all_events')
    if metadata['effective_physics_weight_sha256'] != expected: raise ValueError('Physics weights changed')
    return weights.numpy()


def run(args):
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    pilot_path = args.pilot_dir/'pilot_comparison.json'
    pilot_hash = file_sha256(pilot_path)
    pilot = json.loads(pilot_path.read_text())
    cache_path = args.ae_cache_dir/'frozen_AE_scores.npz'
    cache_hash = file_sha256(cache_path)
    if cache_hash != pilot['AE_cache_sha256']: raise ValueError('AE cache changed')
    paths = {b: args.pilot_dir/b/'checkpoint_final.pth.tar' for b in ('full','tail')}
    hashes = {b: file_sha256(p) for b,p in paths.items()}
    raw = load_hlt_sample(args.sample)
    if sample_signature(raw) != pilot['data_signature']: raise ValueError('Calibration manifest changed')
    weights = load_weights(args.weights, raw, pilot['physics_weight_sha256'])
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache['training_event_id'], raw['event_id'].numpy()): raise ValueError('Calibration AE IDs changed')
        scores = cache['baseline_training_manifest_scores'].copy()
    with np.load(args.pilot_dir/'pilot_roles.npz', allow_pickle=False) as roles:
        calibration = roles['calibration'].copy()  # complete AE-unseen pool, not just old top 10%
        for role in ('full_train','tail_train','ae_gradient_train'):
            if np.intersect1d(calibration, roles[role]).size: raise ValueError(f'Calibration overlaps {role}')
    pre = torch.load(paths['full'], map_location='cpu', weights_only=False)['preprocessing']
    labels = checkpoint_target_labels(raw, pre).numpy()
    cuts = thresholds(scores[calibration], weights[calibration])
    if not np.isclose(cuts['f010'], pilot['ae_threshold'], rtol=1e-7, atol=1e-9): raise ValueError('Original 10% threshold changed')
    cal_selection = {k: role_selection(scores[calibration], weights[calibration], labels[calibration], cut)
                     for k,cut in cuts.items()}
    qrows = calibration[labels[calibration]==1]
    cal_ids = raw['event_id'][calibration].numpy()
    cal_mass = float(weights[qrows].sum())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration_data = {}; plans = {}; references = {}; grids = {}
    for branch,path in paths.items():
        print(f'{branch}: frozen inference on {len(qrows):,} full-range calibration QCD rows', flush=True)
        model, payload = load_nurd_model(str(path), args.device)
        verify_snapshot(payload, pilot, branch)
        if payload['preprocessing']['label_map'] != pre['label_map']: raise ValueError('Model taxonomies differ')
        reference = pilot['results'][branch]
        mean = np.asarray(reference['MD_reference_mean']); whitening = np.asarray(reference['MD_reference_whitening'])
        references[branch] = (mean, whitening)
        latent = infer_latents(model, raw, torch.as_tensor(qrows), args.device, args.batch_size)
        data = {'ae': scores[qrows], 'md': np.square((latent-mean)@whitening).sum(1), 'weight': weights[qrows],
                'label': labels[qrows], 'event_id': raw['event_id'][qrows].numpy(),
                'background_category': raw['background_category'][qrows].numpy()}
        save_scores(args.output_dir/f'{branch}_calibration_QCD.npz', data)
        calibration_data[branch] = data; plans[branch] = {}; grids[branch] = {}
        for key,cut in cuts.items():
            selected = subset(data, cut)
            plans[branch][key] = {target_name(t): matched_curve(selected, cal_mass, t, args.minimums,
                                                              match_tolerance=args.match_tolerance) for t in args.targets}
            tx = quantile(selected['ae'], selected['weight'], GRID_PROBABILITIES)
            ty = quantile(selected['md'], selected['weight'], GRID_PROBABILITIES)
            grid = scan(selected['ae'], selected['md'], selected['weight'], tx, ty, args.minimums)
            grids[branch][key] = {'ae_cuts': tx, 'md_cuts': ty, 'calibration': grid}
            np.savez_compressed(args.output_dir/f'{branch}_{key}_calibration_grid.npz', ae_cuts=tx, md_cuts=ty,
                                percentiles=GRID_PROBABILITIES, **grid)
        del model, payload; gc.collect()
        if args.device=='cuda': torch.cuda.empty_cache()
    plan = {'protocol': 'fixed_models_100_20_10_matched_overall_QCD_A_acceptance',
            'calibration_sample':str(args.sample),'checking_sample':str(args.heldout),
            'pilot_report_sha256': pilot_hash, 'checkpoint_sha256': hashes, 'AE_sha256': pilot['AE_sha256'],
            'AE_cache_sha256': cache_hash, 'AE_thresholds': cuts, 'calibration_selection': cal_selection,
            'minimum_region_ESS': args.minimums, 'overall_QCD_A_targets': args.targets,
            'primary_overall_QCD_A_target': args.primary_target, 'relative_acceptance_tolerance': args.match_tolerance,
            'calibration_QCD_mass': cal_mass, 'plans': plans, 'networks_updated': False,
            'checking_opened_during_selection': False,
            'objective': 'Along each overall-acceptance curve: neighboring three valid AE positions, then minimum median |R-1| plus central uncertainty.',
            'MD_reference': 'Each original pilot training-QCD transform unchanged; no checking refit.',
            'code_commit': args.code_commit,
            'warning': 'Previously inspected MC: exploratory, not a blind test. Tail model was adapted on top 10%, not retrained at 20%/100%. '
                       'No signal-sensitivity optimum; thresholds and candidate choices never use checking results.'}
    (args.output_dir/'frozen_comparison_plan.json').write_text(report_json(finite_report(plan)))
    plan_hash = file_sha256(args.output_dir/'frozen_comparison_plan.json')
    print('ALL six model/retention cut plans frozen. Opening checking sample now.', flush=True)
    del raw, weights, scores, labels

    check = load_hlt_sample(args.heldout)
    if sample_signature(check) != pilot['checking_signature']: raise ValueError('Checking manifest changed')
    weights = load_weights(args.heldout_weights, check, pilot['checking_physics_weight_sha256'])
    labels = checkpoint_target_labels(check, pre).numpy()
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache['heldout_event_id'], check['event_id'].numpy()): raise ValueError('Checking AE IDs changed')
        scores = cache['baseline_heldout_scores'].copy()
    ids = np.concatenate((cal_ids, check['event_id'].numpy()))
    if len(np.unique(ids, axis=0)) != len(ids): raise ValueError('Calibration/checking IDs overlap or contain duplicates')
    checking_selection = {k: role_selection(scores, weights, labels, cut) for k,cut in cuts.items()}
    check_mass = float(weights[labels==1].sum())
    checking_data = {}; results = {}; paired = {}
    for branch,path in paths.items():
        print(f'{branch}: frozen inference on all {len(labels):,} checking events (once, reused across fractions)', flush=True)
        model, payload = load_nurd_model(str(path), args.device); verify_snapshot(payload, pilot, branch)
        mean, whitening = references[branch]
        latent = infer_latents(model, check, torch.arange(len(labels)), args.device, args.batch_size)
        data = {'ae': scores, 'md': np.square((latent-mean)@whitening).sum(1), 'weight': weights,
                'latent': latent, 'label': labels, 'event_id': check['event_id'].numpy(),
                'background_category': check['background_category'].numpy()}
        old_mask = (labels==1) & (scores>=cuts['f010'])
        with np.load(args.pilot_dir/f'{branch}_checking_QCD.npz', allow_pickle=False) as old:
            if not np.array_equal(data['event_id'][old_mask], old['event_id']): raise ValueError('Original QCD IDs differ')
            if not np.array_equal(data['ae'][old_mask], old['ae_score']) or not np.allclose(weights[old_mask], old['physics_weight'], rtol=1e-10, atol=0):
                raise ValueError('Original QCD scores/weights differ')
            if not np.allclose(data['md'][old_mask], old['md_score'], rtol=2e-3, atol=1e-5): raise ValueError('MD inference parity failed')
            data['md'][old_mask] = old['md_score']
        save_scores(args.output_dir/f'{branch}_checking_all_classes.npz', data)
        checking_data[branch] = data; results[branch] = {}
        for key,cut in cuts.items():
            selected = qcd(subset(data, cut)); results[branch][key] = {}
            grid = grids[branch][key]
            grid['checking'] = scan(selected['ae'], selected['md'], selected['weight'], grid['ae_cuts'], grid['md_cuts'], args.minimums)
            np.savez_compressed(args.output_dir/f'{branch}_{key}_checking_grid.npz', **grid['checking'])
            for target in args.targets:
                name = target_name(target); fitted = plans[branch][key][name]; candidate = fitted['candidate']
                value = None if candidate is None else at_cuts(selected, (candidate['ae_cut'],candidate['md_cut']), check_mass, args.minimums)[0]
                results[branch][key][name] = {'calibration_candidate': candidate, 'checking_point': value,
                    'checking_curve': checking_curve(selected, check_mass, fitted['curve'], args.minimums),
                    'status': 'no_stable_calibration_candidate' if candidate is None else 'frozen_candidate_checked'}
        del model, payload; gc.collect()
        if args.device=='cuda': torch.cuda.empty_cache()
    primary = target_name(args.primary_target)
    for key,cut in cuts.items():
        candidates = [plans[b][key][primary]['candidate'] for b in paths]
        if any(c is None for c in candidates): paired[key] = None; continue
        arrays = [qcd(subset(checking_data[b], cut)) for b in paths]
        codes = [at_cuts(d, (c['ae_cut'],c['md_cut']), check_mass, args.minimums)[1] for d,c in zip(arrays,candidates)]
        if not np.array_equal(arrays[0]['event_id'], arrays[1]['event_id']): raise ValueError('Paired event IDs differ')
        try: paired[key] = paired_bootstrap(*arrays, *codes, repeats=500, seed=42)
        except ValueError as error: paired[key] = {'available': False, 'reason': str(error)}
    names = {int(k):v for k,v in pre['target_class_names'].items()}
    from scripts.plot_pilot_retentions import render
    print('Rendering retention comparison and weighted class KDEs; no cut selection remains.', flush=True)
    style = render(plan, results, calibration_data, checking_data, grids, checking_selection,
                   names, args.output_dir/'plots', args.kde_grid)
    for branch,path in paths.items():
        if file_sha256(path) != hashes[branch]: raise ValueError('Original model changed')
    if (file_sha256(pilot_path) != pilot_hash or file_sha256(cache_path) != cache_hash
            or file_sha256(args.output_dir/'frozen_comparison_plan.json') != plan_hash):
        raise ValueError('Original inputs or frozen cut plan changed')
    report = {**plan, 'results': results, 'checking_selection': checking_selection,
              'checking_QCD_mass': check_mass, 'primary_tail_minus_full_model_bootstrap': paired,
              'bootstrap_limitation': 'Fixed cuts and thresholds only; excludes calibration-selection, retraining and systematic uncertainty.',
              'original_checkpoints_unchanged': True, 'original_top10_QCD_cache_agreement': True, **style}
    (args.output_dir/'retention_comparison.json').write_text(report_json(finite_report(report)))
    with (args.output_dir/'comparison.csv').open('w', newline='') as stream:
        fields = ['model','retained_fraction','overall_A_target','role','ratio','ratio_error','overall_A_actual','valid','A_ESS','B_ESS','C_ESS','D_ESS']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for branch in paths:
            for f in FRACTIONS:
                for target in args.targets:
                    entry = results[branch][fraction_name(f)][target_name(target)]
                    for role,point_ in [('calibration',entry['calibration_candidate']),('checking',entry['checking_point'])]:
                        row = {'model':branch,'retained_fraction':f,'overall_A_target':target,'role':role}
                        if point_ is not None:
                            row.update(ratio=point_['predicted_over_observed'],ratio_error=point_['ratio_uncertainty'],
                                       overall_A_actual=point_['A_efficiency_overall_QCD'],valid=point_['statistically_valid'])
                            row.update({r+'_ESS':point_['regions'][r]['effective_count'] for r in 'ABCD'})
                        writer.writerow(row)
    print(f'FROZEN RETENTION COMPARISON COMPLETE: {args.output_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','ae-cache-dir','sample','weights','heldout','heldout-weights','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--minimums', nargs=4, type=int, default=list(MINIMUMS), metavar=('A','B','C','D'))
    parser.add_argument('--targets', nargs='+', type=float, default=list(TARGETS))
    parser.add_argument('--primary-target', type=float, default=.01)
    parser.add_argument('--match-tolerance', type=float, default=.05)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--kde-grid', type=int, default=64)
    parser.add_argument('--code-commit', default='unspecified')
    args = parser.parse_args()
    if (min(args.minimums)<1 or args.batch_size<1 or args.threads<1 or args.kde_grid<8
            or not all(np.isfinite(t) and 0<t<.05 for t in args.targets)
            or len(set(args.targets))!=len(args.targets) or args.primary_target not in args.targets
            or len({target_name(t) for t in args.targets})!=len(args.targets)
            or not 0<args.match_tolerance<.5): parser.error('Invalid statistics, rate, resource or acceptance settings')
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('Requested GPU is not visible')
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads): run(args)


if __name__=='__main__': main()
