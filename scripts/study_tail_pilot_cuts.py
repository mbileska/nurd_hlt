"""Calibration-only cut search and standard-style plots for a frozen tail pilot.

No training, AE refitting, MD-reference refitting, conditional score calibration,
or checking-based model/cut selection. Previously inspected MC remains exploratory.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.audit_qcd_abcd_composition import infer_latents
from scripts.audit_tail_pilot_split import quantile
from scripts.compare_qcd_score_audit import closure_from_yields, region_yields, paired_bootstrap
from scripts.plot_ae_tail_eval_suite import weighted_kde, display_indices, PALETTE
from scripts.prepare_tail_pilot import report_json
from scripts.run_tail_nurd_pilot import finite_report, model_from_config
from scripts.scan_cached_tail_cuts import scan, stable_choice, point
from utils.hlt_data import load_hlt_sample
from utils.hlt_targets import checkpoint_target_labels
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


MINIMUMS = [50, 50, 50, 500]
EFFICIENCY_RANGE = [.02, .25]
FIG_SIZE = (8, 6)
DPI = 200


def load_nurd_model(path, device):
    """Frozen pilot loader, without initializing the ordinary W&B evaluator."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['state_dict_model']
    projection = next(k for k in state if k.endswith('.attn.e.weight'))
    if state['classifier.weight'].shape[0] != 8: raise ValueError('Expected an eight-class pilot')
    model = model_from_config(payload['config'], state[projection].shape[1]-1, device)
    model.load_state_dict(state); model.eval()
    return model, payload


def choose(calibration, probabilities, minimums=MINIMUMS, efficiency_range=EFFICIENCY_RANGE):
    """This interface deliberately accepts no checking events or checking metrics."""
    tx = quantile(calibration['ae'], calibration['weight'], probabilities)
    ty = quantile(calibration['md'], calibration['weight'], probabilities)
    result = scan(calibration['ae'], calibration['md'], calibration['weight'], tx, ty, minimums)
    chosen = stable_choice(result, *efficiency_range)
    return tx, ty, result, chosen


def recount(data, ae_cut, md_cut, minimums=MINIMUMS):
    codes, yields, squares, raw = region_yields(data['ae'], data['md'], data['weight'], ae_cut, md_cut)
    ratio, nonclosure = closure_from_yields(yields)
    ess = np.divide(yields**2, squares, out=np.zeros(4), where=squares > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        uncertainty = ratio * np.sqrt(np.sum(squares / yields**2))
    return {'ae_cut': ae_cut, 'md_cut': md_cut, 'predicted_over_observed': float(ratio),
            'ratio_uncertainty': float(uncertainty), 'absolute_nonclosure': float(nonclosure),
            'A_efficiency_within_tail': float(yields[0] / yields.sum()),
            'statistically_valid': bool(np.all(ess >= minimums) and np.isfinite(ratio)),
            'regions': {r: {'yield': float(yields[i]), 'sumw2': float(squares[i]),
                            'effective_count': float(ess[i]), 'raw_count': int(raw[i])}
                        for i, r in enumerate('ABCD')}}, codes


def qcd(data):
    mask = data['label'] == 1
    return {key: value[mask] for key, value in data.items()}


def save_scores(path, data):
    np.savez_compressed(path, **data)


def visible_range(values, weights, cuts=()):
    """Display-only weighted limits. Every row still enters ABCD and KDE fitting."""
    keep = (values > 0) & (weights > 0) & np.isfinite(values)
    if not keep.any(): return (1e-12, 1.)
    low, high = quantile(values[keep], weights[keep], [.001, .999])
    candidates = [v for v in cuts if v is not None and np.isfinite(v) and v > 0]
    low, high = min([low, *candidates]), max([high, *candidates])
    if high <= low:
        low, high = low / 1.01, high * 1.01
    margin = .04 * np.log(high / low)
    return low * np.exp(-margin), high * np.exp(margin)


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def score_axes(ax, cuts, limits, title):
    ax.set(xscale='log', yscale='log', xlim=limits[0], ylim=limits[1],
           xlabel='AE reconstruction loss', ylabel='QCD-reference encoder MD²', title=title)
    if cuts is not None:
        ax.axvline(cuts[0], color='black', ls='--', lw=1)
        ax.axhline(max(cuts[1], 1e-12), color='black', ls='--', lw=1)


def density_hist(ax, data, limits):
    edges = [np.geomspace(*bound, 101) for bound in limits]
    hist, x, y = np.histogram2d(data['ae'], data['md'], bins=edges, weights=data['weight'])
    positive = hist[hist > 0]
    if not len(positive):
        return None
    return ax.pcolormesh(x, y, np.ma.masked_less_equal(hist.T, 0), shading='auto',
                         norm=LogNorm(vmin=positive.min(), vmax=max(positive.max(), positive.min()*1.001)))


def render(cal, check, tune, checking_grid, probabilities, chosen, tx, ty, old_cuts,
           names, reference, output, kde_grid=80):
    output.mkdir()
    cuts = old_cuts if chosen is None else (float(tx[chosen[0]]), float(ty[chosen[1]]))
    cut_label = 'original reference cuts; no stable candidate' if chosen is None else 'calibration-selected cuts'
    qc = qcd(check)
    qlimits = [visible_range(qc[key], qc['weight'], [cuts[i], old_cuts[i]]) for i, key in enumerate(('ae', 'md'))]
    for role, result in (('calibration', tune), ('checking', checking_grid)):
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        bias = np.where(result['valid'], abs(result['ratio']-1)*100, np.nan)
        mesh = ax.pcolormesh(probabilities, probabilities, bias.T, shading='auto', vmin=0, vmax=100, cmap='viridis_r')
        fig.colorbar(mesh, ax=ax, label='|Predicted/observed - 1| (%)', extend='max')
        if chosen is not None:
            ax.plot(probabilities[chosen[0]], probabilities[chosen[1]], 'r*', ms=16)
        if not result['valid'].any():
            ax.text(.5, .5, 'No cells pass nominal region ESS', ha='center', transform=ax.transAxes)
        ax.set(xlabel='Calibration-tail AE percentile', ylabel='Calibration-tail MD percentile',
               title=f'{role}: ' + ('cut selection' if role == 'calibration' else 'diagnostic only; never ranks cuts'))
        finish(fig, output/f'{role}_closure_scan.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (role, result) in zip(axes, (('calibration', tune), ('checking', checking_grid))):
        mesh = ax.pcolormesh(probabilities, probabilities, result['efficiency'].T, shading='auto', vmin=0, vmax=.5)
        fig.colorbar(mesh, ax=ax, label='QCD A efficiency within AE tail')
        if chosen is not None: ax.plot(probabilities[chosen[0]], probabilities[chosen[1]], 'r*', ms=14)
        ax.set(xlabel='Calibration AE percentile', ylabel='Calibration MD percentile', title=role)
    finish(fig, output/'A_efficiency_scans.png')

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for role, result, color in (('calibration', tune, 'tab:blue'), ('checking', checking_grid, 'tab:orange')):
        diagonal = np.arange(len(probabilities))
        good = result['valid'][diagonal, diagonal]
        x, y, error = (result[k][diagonal, diagonal] for k in ('efficiency', 'ratio', 'sigma'))
        order = np.argsort(x[good])
        ax.errorbar(x[good][order], y[good][order], yerr=error[good][order], marker='.', label=f'{role}: valid diagonal cuts', color=color)
        finite = ~good & np.isfinite(y) & (x > 0)
        ax.scatter(x[finite], y[finite], s=10, marker='x', color=color, alpha=.3)
        value, _ = recount(cal if role == 'calibration' else qc, *cuts)
        if value['A_efficiency_within_tail'] > 0 and np.isfinite(value['predicted_over_observed']):
            ax.plot(value['A_efficiency_within_tail'], value['predicted_over_observed'], '*', color=color, ms=15)
    original, _ = recount(qc, *old_cuts)
    if np.isfinite(original['predicted_over_observed']):
        ax.errorbar(original['A_efficiency_within_tail'], original['predicted_over_observed'],
                    yerr=original['ratio_uncertainty'], fmt='D', c='black', label='Original pilot checking point')
    ax.axhline(1, c='black'); ax.axhline(.95, c='black', ls='--'); ax.axhline(1.05, c='black', ls='--')
    ax.set(xscale='log', xlabel='QCD A efficiency within AE tail', ylabel='Predicted / observed QCD',
           title='Fixed calibration-derived cuts; crosses fail ESS')
    ax.legend(fontsize=10); finish(fig, output/'cut_and_count_bkg_check.png')

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    mesh = density_hist(ax, qc, qlimits)
    if mesh is not None: fig.colorbar(mesh, ax=ax, label='Physics-weighted QCD yield (log color)')
    score_axes(ax, cuts, qlimits, 'Checking QCD: '+cut_label)
    finish(fig, output/'hist2d_QCD.png')

    # Common display bounds include 99.8% of each class's physical shape.
    # No event is removed from KDE estimation or ABCD because of these bounds.
    class_limits = {label: [visible_range(check[key][check['label']==label], check['weight'][check['label']==label], [cuts[i]])
                           for i, key in enumerate(('ae', 'md'))]
                    for label in names if np.any((check['label']==label) & (check['weight']>0))}
    limits = [(min(v[i][0] for v in class_limits.values()), max(v[i][1] for v in class_limits.values())) for i in range(2)]
    gx, gy = np.meshgrid(np.linspace(*np.log10(limits[0]), kde_grid), np.linspace(*np.log10(limits[1]), kde_grid))
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    handles, skipped = [], []
    for i, (label, name) in enumerate(names.items()):
        mask = check['label'] == label
        print(f'Weighted KDE: {name}, {int(mask.sum()):,} checking rows', flush=True)
        density = weighted_kde(check['ae'][mask], check['md'][mask], check['weight'][mask], gx, gy)
        if density is None or not np.isfinite(density).all() or density.max() <= 0:
            skipped.append(name); continue
        color = PALETTE[i % len(PALETTE)]
        ax.contour(10**gx, 10**gy, density, levels=density.max()*np.array([.05,.15,.3,.5,.7,.88]), colors=color, alpha=.7)
        handles.append(Line2D([0], [0], color=color, label=name))
    score_axes(ax, cuts, limits, 'Checking backgrounds: physics-weighted class KDE shapes')
    if handles: ax.legend(handles=handles, fontsize=10)
    finish(fig, output/'hist2d_pca_md_kde.png')

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (label, name) in zip(axes.flat, names.items()):
        mask = check['label'] == label
        if label in class_limits:
            density_hist(ax, {key: value[mask] for key, value in check.items()}, limits)
        score_axes(ax, cuts, limits, name)
        ax.tick_params(labelsize=8)
    finish(fig, output/'hist2d_by_class.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, (key, ax) in enumerate(zip(('ae', 'md'), axes)):
        edges = np.geomspace(*limits[i], 81)
        for j, (label, name) in enumerate(names.items()):
            mask = check['label'] == label
            weight = check['weight'][mask]
            if weight.sum() <= 0: continue
            mass, _ = np.histogram(check[key][mask], bins=edges, weights=weight/weight.sum())
            ax.stairs(mass, edges, label=name, color=PALETTE[j % len(PALETTE)])
        ax.axvline(cuts[i], c='black', ls='--')
        ax.set(xscale='log', xlabel='AE loss' if key=='ae' else 'QCD-reference MD²', ylabel='Class-normalized physical mass per log bin')
    axes[1].legend(fontsize=9); finish(fig, output/'score_distributions_by_class.png')

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    mean, whitening = reference
    basis = whitening / np.linalg.norm(whitening, axis=0)
    latent = (check['latent']-mean) @ basis[:, -2:][:, ::-1]
    display = display_indices(check['label'], check['weight'], cap_per_class=2500)
    for i, (label, name) in enumerate(names.items()):
        rows = display[check['label'][display] == label]
        ax.scatter(latent[rows, 0], latent[rows, 1], s=2, alpha=.2, label=name, color=PALETTE[i % len(PALETTE)], rasterized=True)
    ax.set(xlabel='Training-QCD PC1', ylabel='Training-QCD PC2', title='Checking latent: physics-weighted display draws')
    ax.legend(fontsize=10, markerscale=3); finish(fig, output/'pca_scatter_embeddings.png')

    value, _ = recount(qc, *cuts)
    yields = np.array([value['regions'][r]['yield'] for r in 'ABCD'])
    errors = np.sqrt([value['regions'][r]['sumw2'] for r in 'ABCD'])
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(range(4), yields/yields.sum(), yerr=errors/yields.sum())
    ax.set_xticks(range(4), [f"{r}\nESS={value['regions'][r]['effective_count']:.0f}" for r in 'ABCD'])
    ax.set(ylabel='Fraction of selected QCD physical yield', title='Checking regions: '+cut_label)
    finish(fig, output/'ABCD_region_yields.png')
    return {'KDE_skipped_classes': skipped, 'QCD_display_limits': qlimits, 'class_display_limits': limits,
            'display_quantiles': [.001,.999], 'KDE_weights': 'original physics weights exactly once; all positive-score rows',
            'checking_figures_are_not_for_cut_or_model_selection': True}


def run(args):
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    pilot_path = args.pilot_dir/'pilot_comparison.json'
    pilot_hash = file_sha256(pilot_path)
    pilot = json.loads(pilot_path.read_text())
    checkpoint = (args.pilot_dir/'source_lambda1_checkpoint.pth.tar' if args.branch=='source'
                  else args.pilot_dir/args.branch/'checkpoint_final.pth.tar')
    checkpoint_hash = file_sha256(checkpoint)
    model, payload = load_nurd_model(str(checkpoint), args.device)
    pre = payload['preprocessing']
    if payload['ae_checkpoint_sha256'] != pilot['AE_sha256']:
        raise ValueError('Pilot AE lineage differs')
    if args.branch != 'source' and (payload.get('encoder_updates') != pilot['updates_per_branch'] or payload.get('branch') != args.branch):
        raise ValueError('Expected the fixed-budget final pilot checkpoint')
    raw = load_hlt_sample(args.sample)
    if sample_signature(raw) != pilot['data_signature']: raise ValueError('Training/calibration manifest changed')
    weights, metadata = load_generator_weights(str(args.weights), raw['label'], sample=raw, qcd_label=1, weight_scope='all_events')
    if metadata['effective_physics_weight_sha256'] != pilot['physics_weight_sha256']: raise ValueError('Training physical weights changed')
    weights = weights.numpy(); labels = checkpoint_target_labels(raw, pre).numpy()
    with np.load(args.pilot_dir/'pilot_roles.npz', allow_pickle=False) as roles:
        cal_rows = roles['selected_calibration'].copy()
        if np.intersect1d(cal_rows, roles['full_train']).size or np.intersect1d(cal_rows, roles['tail_train']).size:
            raise ValueError('Calibration overlaps new pilot optimization')
        if np.intersect1d(cal_rows, roles['ae_gradient_train']).size: raise ValueError('Calibration contains AE gradient-training rows')
    cache_path = args.ae_cache_dir/'frozen_AE_scores.npz'
    if file_sha256(cache_path) != pilot['AE_cache_sha256']: raise ValueError('AE score cache changed')
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache['training_event_id'], raw['event_id'].numpy()): raise ValueError('Calibration cache IDs changed')
        scores = cache['baseline_training_manifest_scores'].copy()
    if np.any(scores[cal_rows] < pilot['ae_threshold']): raise ValueError('Calibration tail threshold mismatch')
    cal_rows = cal_rows[labels[cal_rows] == 1]
    mean = np.asarray(pilot['results'][args.branch]['MD_reference_mean'])
    whitening = np.asarray(pilot['results'][args.branch]['MD_reference_whitening'])
    print(f'Frozen calibration inference: {len(cal_rows):,} QCD rows; no training or MD refit', flush=True)
    latent = infer_latents(model, raw, torch.as_tensor(cal_rows), args.device, args.batch_size)
    cal = {'ae': scores[cal_rows], 'md': np.square((latent-mean)@whitening).sum(1),
           'weight': weights[cal_rows], 'label': labels[cal_rows], 'event_id': raw['event_id'][cal_rows].numpy(),
           'background_category': raw['background_category'][cal_rows].numpy()}
    probabilities = np.linspace(.05, .95, 91)
    tx, ty, tune, chosen = choose(cal, probabilities)
    candidate = None if chosen is None else point(tune, chosen, tx, ty)
    if chosen is not None:
        i, j = chosen
        candidate['neighborhood_median_abs_ratio_bias'] = float(np.median(abs(tune['ratio'][i-1:i+2, j-1:j+2]-1)))
        candidate['neighborhood_max_abs_ratio_bias'] = float(np.max(abs(tune['ratio'][i-1:i+2, j-1:j+2]-1)))
        candidate['within_20pct_throughout_neighborhood'] = candidate['neighborhood_max_abs_ratio_bias'] <= .2
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = {'pilot_report_sha256': pilot_hash, 'branch': args.branch, 'checkpoint_sha256': checkpoint_hash,
            'frozen_AE_threshold': pilot['ae_threshold'], 'MD_reference': 'unchanged pilot training-QCD transform',
            'minimum_region_ESS': MINIMUMS, 'A_efficiency_range_within_tail': EFFICIENCY_RANGE,
            'objective': 'calibration 3x3 median |predicted/observed-1| plus central ratio uncertainty; valid and efficient throughout neighborhood',
            'candidate': candidate, 'chosen_index': chosen, 'calibration_valid_cells': int(tune['valid'].sum()),
            'networks_updated': False, 'checking_opened_during_selection': False, 'code_commit': args.code_commit,
            'warning': 'Exploratory cut tuning on previously inspected MC. Not a new blind test or signal-sensitivity optimum; no signal optimization.'}
    (args.output_dir/'frozen_cut_plan.json').write_text(report_json(finite_report(plan)))
    save_scores(args.output_dir/'calibration_QCD_scores.npz', cal)
    np.savez_compressed(args.output_dir/'calibration_grid.npz', ae_cuts=tx, md_cuts=ty, percentiles=probabilities, **tune)
    print('Cut plan frozen before opening checking events. Candidate:', candidate, flush=True)
    del raw, weights, labels, scores

    check = load_hlt_sample(args.heldout)
    if sample_signature(check) != pilot['checking_signature']: raise ValueError('Checking manifest changed')
    check_weights, check_meta = load_generator_weights(str(args.heldout_weights), check['label'], sample=check, qcd_label=1, weight_scope='all_events')
    if check_meta['effective_physics_weight_sha256'] != pilot['checking_physics_weight_sha256']: raise ValueError('Checking physical weights changed')
    check_weights = check_weights.numpy(); check_labels = checkpoint_target_labels(check, pre).numpy()
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache['heldout_event_id'], check['event_id'].numpy()): raise ValueError('Checking cache IDs changed')
        check_scores = cache['baseline_heldout_scores'].copy()
    rows = np.flatnonzero(check_scores >= pilot['ae_threshold'])
    combined_ids = np.concatenate((cal['event_id'], check['event_id'][rows].numpy()))
    if len(np.unique(combined_ids, axis=0)) != len(combined_ids):
        raise ValueError('Calibration/checking IDs overlap or contain duplicates')
    print(f'Frozen checking inference: {len(rows):,} all-class selected rows; device={args.device}', flush=True)
    latent = infer_latents(model, check, torch.as_tensor(rows), args.device, args.batch_size)
    data = {'ae': check_scores[rows], 'md': np.square((latent-mean)@whitening).sum(1),
            'latent': latent, 'label': check_labels[rows], 'weight': check_weights[rows],
            'event_id': check['event_id'][rows].numpy(), 'source_row_index': rows,
            'background_category': check['background_category'][rows].numpy()}
    mask = data['label'] == 1
    with np.load(args.pilot_dir/f'{args.branch}_checking_QCD.npz', allow_pickle=False) as old:
        if not np.array_equal(data['event_id'][mask], old['event_id']) or not np.array_equal(data['ae'][mask], old['ae_score']):
            raise ValueError('Checking QCD alignment differs from original pilot')
        if not np.allclose(data['weight'][mask], old['physics_weight'], rtol=1e-10, atol=0): raise ValueError('QCD weights differ')
        if not np.allclose(data['md'][mask], old['md_score'], rtol=2e-3, atol=1e-5): raise ValueError('Frozen MD scoring parity failed')
        data['md'][mask] = old['md_score']  # preserve exact original counting arithmetic
    qc = qcd(data)
    checking_grid = scan(qc['ae'], qc['md'], qc['weight'], tx, ty, MINIMUMS)
    save_scores(args.output_dir/'checking_all_class_scores.npz', data)
    np.savez_compressed(args.output_dir/'checking_grid.npz', **checking_grid)
    old = pilot['results'][args.branch]['matched_point']; old_cuts = (old['ae_cut'], old['md_cut'])
    reference_point, original_codes = recount(qc, *old_cuts)
    if not np.isclose(reference_point['predicted_over_observed'], old['predicted_over_observed'], rtol=1e-9):
        raise ValueError('Original pilot point does not recount')
    new_point = None if chosen is None else recount(qc, tx[chosen[0]], ty[chosen[1]])[0]
    bootstrap = None
    if chosen is not None:
        _, candidate_codes = recount(qc, tx[chosen[0]], ty[chosen[1]])
        try:
            bootstrap = paired_bootstrap(qc, qc, original_codes, candidate_codes, repeats=500, seed=42)
        except ValueError as error:
            bootstrap = {'available': False, 'reason': str(error)}
    with (args.output_dir/'cut_scan.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ae_cut','md_cut','calibration_ratio','calibration_valid','checking_ratio','checking_valid','calibration_A_efficiency'])
        for i, a in enumerate(tx):
            for j, b in enumerate(ty):
                writer.writerow([a,b,tune['ratio'][i,j],tune['valid'][i,j],checking_grid['ratio'][i,j],checking_grid['valid'][i,j],tune['efficiency'][i,j]])
    names = {int(k): v for k,v in pre['target_class_names'].items()}
    print('Scores saved and original pilot point reproduced. Rendering ten diagnostic figures.', flush=True)
    style = render(cal, data, tune, checking_grid, probabilities, chosen, tx, ty, old_cuts,
                   names, (mean,whitening), args.output_dir/'plots', args.kde_grid)
    if file_sha256(checkpoint) != checkpoint_hash or file_sha256(pilot_path) != pilot_hash: raise ValueError('Original pilot changed')
    report = {**plan, 'checking_candidate': new_point, 'original_reference_checking_point': reference_point,
              'candidate_minus_original_fixed_cut_bootstrap': bootstrap,
              'bootstrap_limitation': 'Cuts fixed; calibration-selection and training uncertainty are not included.',
              'checking_valid_cells': int(checking_grid['valid'].sum()), 'original_QCD_cache_agreement': True,
              'status': 'no_stable_candidate' if candidate is None else 'candidate_checked_not_certified',
              'checking_plots_cannot_choose_another_cut': True, **style}
    (args.output_dir/'grid_study_summary.json').write_text(report_json(finite_report(report)))
    print(report_json(finite_report(report)), flush=True)
    print(f'FROZEN GRID STUDY COMPLETE: {args.output_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-dir','ae-cache-dir','sample','weights','heldout','heldout-weights','output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--branch', choices=['tail','full','source'], default='tail')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--kde-grid', type=int, default=80)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--code-commit', default='unspecified')
    args = parser.parse_args()
    if args.batch_size < 1 or args.kde_grid < 8 or args.threads < 1: parser.error('Positive batch/threads; KDE grid >= 8 required')
    if args.device == 'cuda' and not torch.cuda.is_available(): parser.error('Requested GPU is not visible')
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads): run(args)


if __name__ == '__main__': main()
