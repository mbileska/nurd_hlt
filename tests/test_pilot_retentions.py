import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import compare_pilot_retentions as comparison


def test_preselection_is_weighted_all_background_not_raw_or_class_balanced():
    scores=np.arange(100,dtype=float)
    weights=np.ones(100)
    cuts=comparison.thresholds(scores,weights)
    assert cuts=={'f100':None,'f020':79.,'f010':89.}
    weights[0]=1000
    cuts=comparison.thresholds(scores,weights)
    assert cuts['f020']==0 and cuts['f010']==0


def test_iso_curves_match_overall_not_within_tail_and_checking_does_not_refit():
    rng=np.random.default_rng(42)
    n=60000
    full={'ae':np.exp(rng.normal(size=n)),'md':np.exp(rng.normal(size=n)),'weight':np.ones(n)}
    for fraction in (1.,.2,.1):
        cut=None if fraction==1 else np.quantile(full['ae'],1-fraction)
        selected=comparison.subset(full,cut)
        fitted=comparison.matched_curve(selected,float(n),.01)
        assert fitted['candidate'] is not None
        candidate=fitted['candidate']
        assert candidate['A_efficiency_overall_QCD']==pytest.approx(.01,abs=.0001)
        assert candidate['A_efficiency_within_tail']==pytest.approx(.01/fraction,abs=.001)
        # Keep the SAME numerical cuts on a shifted checking score. Its A rate
        # must be allowed to change; calibrating it back to 1% would hide shift.
        shifted={**selected,'md':selected['md']*100}
        curve=comparison.checking_curve(shifted,float(n),fitted['curve'],comparison.MINIMUMS)
        checked=curve[candidate['curve_index']]
        assert checked['A_efficiency_overall_QCD']>.02
        assert checked['md_cut']==candidate['md_cut']


def test_relaxed_D_applies_explicitly_without_changing_previous_defaults():
    from scripts.study_tail_pilot_cuts import MINIMUMS as old_minimums
    assert old_minimums==[50,50,50,500]
    rng=np.random.default_rng(3)
    data={'ae':np.exp(rng.normal(size=800)),'md':np.exp(rng.normal(size=800)),'weight':np.ones(800)}
    assert comparison.matched_curve(data,10000.,.01,(50,50,50,500))['candidate'] is None
    assert comparison.matched_curve(data,10000.,.01,(50,50,50,100))['candidate'] is not None


def test_ties_do_not_fake_acceptance_matching_or_eligible_empty_regions():
    data={'ae':np.arange(1000,dtype=float)+1,'md':np.ones(1000),'weight':np.ones(1000)}
    fit=comparison.matched_curve(data,1000.,.01)
    assert fit['candidate'] is None
    assert all(p is None or not p['eligible'] for p in fit['curve'])


def test_worker_is_generic_gpu_inference_not_training():
    path=Path(__file__).parents[1]/'slurm/submit_pilot_retention_comparison.sbatch'
    worker=path.read_text()
    assert '#SBATCH --gres=gpu:1' in worker
    assert '--minimums 50 50 50 "$D_MIN_ESS"' in worker
    for forbidden in ('--partition','--constraint','--nodelist','a100','train_ae.py','train_hlt.py'):
        assert forbidden not in worker.lower()


def test_render_comparison_and_all_six_weighted_kdes(tmp_path):
    from scripts.plot_pilot_retentions import render
    from scripts.scan_cached_tail_cuts import scan
    from utils.hlt_targets import DECAY8_NAMES
    rng=np.random.default_rng(64)
    n=20000
    cal={'ae':np.exp(rng.normal(size=n)),'md':np.exp(rng.normal(size=n)),
         'weight':np.ones(n),'label':np.ones(n,dtype=int)}
    check={'ae':np.exp(rng.normal(size=800)),'md':np.exp(rng.normal(size=800)),
           'weight':np.ones(800),'label':np.tile(np.arange(8),100)}
    cuts=comparison.thresholds(cal['ae'],cal['weight'])
    targets=[.01];name=comparison.target_name(.01)
    plan={'AE_thresholds':cuts,'minimum_region_ESS':[50,50,50,100],
          'primary_overall_QCD_A_target':.01,'overall_QCD_A_targets':targets,
          'plans':{},'calibration_selection':{}}
    for key,cut in cuts.items():
        plan['calibration_selection'][key]=comparison.role_selection(cal['ae'],cal['weight'],cal['label'],cut)
    checking_selection={k:comparison.role_selection(check['ae'],check['weight'],check['label'],cut) for k,cut in cuts.items()}
    results={};grids={}
    for branch in ('full','tail'):
        plan['plans'][branch]={};results[branch]={};grids[branch]={}
        for key,cut in cuts.items():
            c=comparison.subset(cal,cut);q=comparison.qcd(comparison.subset(check,cut))
            fit=comparison.matched_curve(c,float(n),.01)
            plan['plans'][branch][key]={name:fit}
            candidate=fit['candidate']
            point=None if candidate is None else comparison.at_cuts(q,(candidate['ae_cut'],candidate['md_cut']),100.,[50,50,50,100])[0]
            results[branch][key]={name:{'calibration_candidate':candidate,'checking_point':point,
                'checking_curve':comparison.checking_curve(q,100.,fit['curve'],[50,50,50,100])}}
            tx=comparison.quantile(c['ae'],c['weight'],comparison.GRID_PROBABILITIES)
            ty=comparison.quantile(c['md'],c['weight'],comparison.GRID_PROBABILITIES)
            grids[branch][key]={'calibration':scan(c['ae'],c['md'],c['weight'],tx,ty,[50,50,50,100]),
                               'checking':scan(q['ae'],q['md'],q['weight'],tx,ty,[50,50,50,100])}
    style=render(plan,results,{'full':cal,'tail':cal},{'full':check,'tail':check},
                 grids,checking_selection,DECAY8_NAMES,tmp_path/'plots',kde_grid=8)
    assert len(list((tmp_path/'plots').rglob('*.png')))==14
    assert len(style['KDE_skipped_classes'])==6
    assert style['plots_never_rank_checking_cuts_or_retention_fractions']


def test_end_to_end_all_six_plans_frozen_before_checking_no_source_changes(tmp_path,monkeypatch):
    from scripts import plot_pilot_retentions as plotting
    from utils.hlt_targets import DECAY8_NAMES
    from utils.hlt_weights import file_sha256,load_generator_weights,sample_signature
    rng=np.random.default_rng(12)
    n=160000;calibration=np.arange(n//2)
    parents=torch.tensor([0,1,2,3,3,3,0,0]).repeat(n//8)
    categories=torch.tensor([15,0,27,6,9,12,18,21]).repeat(n//8)
    def sample(offset):
        return {'pf':torch.from_numpy(rng.normal(size=(n,1,2)).astype(np.float32)),
                'obj':torch.ones(n,1,4),'label':parents,'background_category':categories,
                'event_id':torch.stack((parents,parents*0,parents*0,torch.arange(n)+offset),1)}
    raw=sample(0);check=sample(n+10000)
    data=tmp_path/'train.pt';test=tmp_path/'check.pt'
    weights=tmp_path/'weights.pt';tw=tmp_path/'check_weights.pt'
    torch.save(raw,data);torch.save(check,test)
    meta=[]
    for path,sample_ in ((weights,raw),(tw,check)):
        torch.save({'weights':torch.ones(n),'event_id':sample_['event_id'],'weight_scope':'all_events'},path)
        _,m=load_generator_weights(str(path),sample_['label'],sample=sample_,weight_scope='all_events');meta.append(m)
    train_ae=np.exp(rng.normal(size=n));test_ae=np.exp(rng.normal(size=n))
    cuts=comparison.thresholds(train_ae[calibration],np.ones(len(calibration)))
    cache=tmp_path/'cache';cache.mkdir()
    np.savez_compressed(cache/'frozen_AE_scores.npz',training_event_id=raw['event_id'].numpy(),
                        heldout_event_id=check['event_id'].numpy(),baseline_training_manifest_scores=train_ae,
                        baseline_heldout_scores=test_ae)
    pilot=tmp_path/'pilot';pilot.mkdir()
    roles=pilot/'pilot_roles.npz'
    np.savez_compressed(roles,calibration=calibration,full_train=np.arange(n//2,n),
                        tail_train=np.arange(n//2,n),ae_gradient_train=np.arange(n//2,n))
    labels=np.tile(np.arange(8),n//8)
    pre={'label_map':{i:i for i in range(8)},'target_scheme':'decay8','target_class_names':DECAY8_NAMES}
    report={'AE_sha256':'test-ae','updates_per_branch':4000,'ae_threshold':cuts['f010'],
            'AE_cache_sha256':file_sha256(cache/'frozen_AE_scores.npz'),'data_signature':sample_signature(raw),
            'checking_signature':sample_signature(check),'physics_weight_sha256':meta[0]['effective_physics_weight_sha256'],
            'checking_physics_weight_sha256':meta[1]['effective_physics_weight_sha256'],'results':{}}
    hashes={}
    for branch in ('full','tail'):
        (pilot/branch).mkdir()
        path=pilot/branch/'checkpoint_final.pth.tar'
        torch.save({'branch':branch,'encoder_updates':4000,'ae_checkpoint_sha256':'test-ae','preprocessing':pre},path)
        hashes[branch]=file_sha256(path)
        scale=1. if branch=='full' else 2.
        md=np.square(check['pf'][:,0,:].numpy()*scale).sum(1)
        selected=(labels==1)&(test_ae>=cuts['f010'])
        np.savez_compressed(pilot/f'{branch}_checking_QCD.npz',event_id=check['event_id'][selected].numpy(),
                            ae_score=test_ae[selected],md_score=md[selected],physics_weight=np.ones(selected.sum()))
        report['results'][branch]={'MD_reference_mean':[0.,0.],'MD_reference_whitening':[[1.,0.],[0.,1.]]}
    report_path=pilot/'pilot_comparison.json';report_path.write_text(json.dumps(report));report_hash=file_sha256(report_path)
    def load(path,_device):
        payload=torch.load(path,weights_only=False)
        return payload['branch'],payload
    monkeypatch.setattr(comparison,'load_nurd_model',load)
    calls=[]
    def infer(model,sample_,rows,*_):
        calls.append((model,len(rows)))
        return sample_['pf'][rows,0,:].numpy()*(1 if model=='full' else 2)
    monkeypatch.setattr(comparison,'infer_latents',infer)
    output=tmp_path/'comparison'
    loader=comparison.load_hlt_sample
    def checked_loader(path):
        if path==test:
            frozen=json.loads((output/'frozen_comparison_plan.json').read_text())
            assert not frozen['checking_opened_during_selection']
            assert set(frozen['plans'])=={'full','tail'}
            assert all(set(v)=={'f100','f020','f010'} for v in frozen['plans'].values())
            assert frozen['plans']['full']['f010']['A0.0100']['candidate'] is not None
        return loader(path)
    monkeypatch.setattr(comparison,'load_hlt_sample',checked_loader)
    def render(*args):
        plan,results,cal,checking,*_=args
        assert len(checking['full']['ae'])==n
        assert np.array_equal(checking['full']['event_id'],checking['tail']['event_id'])
        return {'test_render':True}
    monkeypatch.setattr(plotting,'render',render)
    args=argparse.Namespace(pilot_dir=pilot,ae_cache_dir=cache,sample=data,weights=weights,
                            heldout=test,heldout_weights=tw,output_dir=output,device='cpu',batch_size=32,
                            minimums=[50,50,50,100],targets=[.01],primary_target=.01,
                            match_tolerance=.05,kde_grid=8,code_commit='test')
    comparison.run(args)
    result=json.loads((output/'retention_comparison.json').read_text())
    assert result['original_checkpoints_unchanged'] and result['original_top10_QCD_cache_agreement']
    assert not result['networks_updated']
    assert [b for b,_ in calls]==['full','tail','full','tail']
    assert calls[-1][1]==n and calls[-2][1]==n
    for f in ('f100','f020','f010'):
        candidate=result['results']['tail'][f]['A0.0100']['calibration_candidate']
        assert candidate is not None
        assert candidate['A_efficiency_overall_QCD']==pytest.approx(.01,abs=.0005)
        assert result['primary_tail_minus_full_model_bootstrap'][f]['delta_median']==pytest.approx(0.)
    for branch in hashes:
        assert file_sha256(pilot/branch/'checkpoint_final.pth.tar')==hashes[branch]
        with np.load(output/f'{branch}_checking_all_classes.npz') as cached:
            assert len(cached['ae'])==n
    assert file_sha256(report_path)==report_hash
    assert (output/'comparison.csv').is_file()
    with pytest.raises(FileExistsError):comparison.run(args)
