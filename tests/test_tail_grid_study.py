import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import study_tail_pilot_cuts as study


def test_scan_only_accepts_calibration_and_freezes_efficiency_and_ess():
    rng = np.random.default_rng(10)
    cal = {'ae': np.exp(rng.normal(size=30000)), 'md': np.exp(rng.normal(size=30000)),
           'weight': np.ones(30000)}
    probabilities = np.linspace(.05, .95, 31)
    tx, ty, result, choice = study.choose(cal, probabilities)
    assert choice is not None
    i, j = choice
    assert result['valid'][i-1:i+2,j-1:j+2].all()
    assert (result['efficiency'][i-1:i+2,j-1:j+2] >= .02).all()
    assert (result['efficiency'][i-1:i+2,j-1:j+2] <= .25).all()
    # An independent, catastrophically shifted checking sample cannot change
    # the selected cut. It only changes its subsequent recount.
    frozen = (tx[i], ty[j])
    check = {**cal, 'md': cal['ae']**4}
    original, _ = study.recount(cal, *frozen)
    shifted, _ = study.recount(check, *frozen)
    assert not np.isclose(original['predicted_over_observed'], shifted['predicted_over_observed'], equal_nan=True)
    assert study.choose(cal, probabilities)[3] == choice


def test_no_candidate_does_not_silently_lower_statistical_requirements():
    rng = np.random.default_rng(5)
    cal = dict(ae=np.exp(rng.normal(size=60)), md=np.exp(rng.normal(size=60)), weight=np.ones(60))
    assert study.choose(cal, np.linspace(.05,.95,21))[3] is None


def test_display_bounds_do_not_change_event_recount():
    x = np.r_[np.linspace(1,2,1000),1e9]
    weights = np.r_[np.ones(1000),1e-11]
    limits = study.visible_range(x, weights)
    assert limits[1] < 10
    data = {'ae': x, 'md': np.linspace(1,3,len(x)), 'weight': weights}
    result, codes = study.recount(data,1.5,2.)
    assert len(codes) == len(x)
    assert sum(r['raw_count'] for r in result['regions'].values()) == len(x)
    assert sum(r['yield'] for r in result['regions'].values()) == pytest.approx(weights.sum())


def test_no_training_no_specific_gpu_worker():
    worker=(Path(__file__).parents[1]/'slurm/submit_tail_grid_study.sbatch').read_text()
    assert '#SBATCH --gres=gpu:1' in worker
    for text in ('--partition','--constraint','--nodelist','a100','train_hlt.py','train_ae.py'):
        assert text not in worker.lower()


@pytest.mark.parametrize('with_candidate', [False, True])
def test_end_to_end_freezes_plan_before_checking_and_preserves_pilot(tmp_path,monkeypatch,with_candidate):
    from utils.hlt_targets import DECAY8_NAMES
    from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature
    rng=np.random.default_rng(42)
    n=24000 if with_candidate else 800
    cal_size=n//2 if with_candidate else 160
    parents=torch.tensor([0,1,2,3,3,3,0,0]).repeat(n//8)
    categories=torch.tensor([15,0,27,6,9,12,18,21]).repeat(n//8)
    def sample(offset):
        return {'pf':torch.from_numpy(rng.normal(size=(n,1,2)).astype(np.float32)),
                'obj':torch.ones(n,1,4),'label':parents,'background_category':categories,
                'event_id':torch.stack((parents,parents*0,parents*0,torch.arange(n)+offset),1)}
    raw=sample(0);check=sample(n+10000)
    data=tmp_path/'train.pt';test=tmp_path/'check.pt'
    wpath=tmp_path/'weights.pt';twpath=tmp_path/'check_weights.pt'
    torch.save(raw,data);torch.save(check,test)
    meta=[]
    for path,sample_ in ((wpath,raw),(twpath,check)):
        torch.save({'weights':torch.ones(n),'event_id':sample_['event_id'],'weight_scope':'all_events'},path)
        _,m=load_generator_weights(str(path),sample_['label'],sample=sample_,weight_scope='all_events')
        meta.append(m)
    pilot=tmp_path/'pilot';(pilot/'tail').mkdir(parents=True)
    checkpoint=pilot/'tail/checkpoint_final.pth.tar';checkpoint.write_bytes(b'frozen test checkpoint')
    cache=tmp_path/'cache';cache.mkdir()
    train_ae=np.exp(rng.normal(size=n));test_ae=np.exp(rng.normal(size=n))
    threshold=.1
    np.savez_compressed(cache/'frozen_AE_scores.npz',training_event_id=raw['event_id'].numpy(),
                        heldout_event_id=check['event_id'].numpy(),baseline_training_manifest_scores=train_ae,
                        baseline_heldout_scores=test_ae)
    cal_rows=np.arange(cal_size)[train_ae[:cal_size]>=threshold]
    np.savez_compressed(pilot/'pilot_roles.npz',selected_calibration=cal_rows,
                        full_train=np.arange(cal_size,n),tail_train=np.arange(cal_size,n),ae_gradient_train=np.arange(cal_size,n))
    labels=np.tile(np.arange(8),n//8)
    selected=np.flatnonzero(test_ae>=threshold)
    qcd_rows=selected[labels[selected]==1]
    md=np.square(check['pf'][:,0,:].numpy()).sum(1)
    np.savez_compressed(pilot/'tail_checking_QCD.npz',event_id=check['event_id'][qcd_rows].numpy(),
                        ae_score=test_ae[qcd_rows],md_score=md[qcd_rows],physics_weight=np.ones(len(qcd_rows)))
    value,_=study.recount(dict(ae=test_ae[qcd_rows],md=md[qcd_rows],weight=np.ones(len(qcd_rows))),1.,2.)
    report={'AE_sha256':'test-ae','updates_per_branch':4000,'ae_threshold':threshold,
            'data_signature':sample_signature(raw),'checking_signature':sample_signature(check),
            'physics_weight_sha256':meta[0]['effective_physics_weight_sha256'],
            'checking_physics_weight_sha256':meta[1]['effective_physics_weight_sha256'],
            'AE_cache_sha256':file_sha256(cache/'frozen_AE_scores.npz'),
            'results':{'tail':{'MD_reference_mean':[0.,0.],'MD_reference_whitening':[[1.,0.],[0.,1.]],
                               'matched_point':value}}}
    (pilot/'pilot_comparison.json').write_text(json.dumps(report))
    old_hash=file_sha256(pilot/'pilot_comparison.json');ckpt_hash=file_sha256(checkpoint)
    pre={'target_scheme':'decay8','label_map':{i:i for i in range(8)},'target_class_names':DECAY8_NAMES}
    payload={'preprocessing':pre,'ae_checkpoint_sha256':'test-ae','encoder_updates':4000,'branch':'tail'}
    monkeypatch.setattr(study,'load_nurd_model',lambda *_:(object(),payload))
    inferred=[]
    def infer(_model,sample_,rows,*_):
        inferred.append(rows.numpy())
        return sample_['pf'][rows,0,:].numpy()
    monkeypatch.setattr(study,'infer_latents',infer)
    output=tmp_path/'study'
    loader=study.load_hlt_sample
    def checked_load(path):
        if path==test:
            plan=json.loads((output/'frozen_cut_plan.json').read_text())
            assert plan['checking_opened_during_selection'] is False
            assert (plan['candidate'] is not None)==with_candidate
        return loader(path)
    monkeypatch.setattr(study,'load_hlt_sample',checked_load)
    args=argparse.Namespace(pilot_dir=pilot,ae_cache_dir=cache,sample=data,weights=wpath,
                            heldout=test,heldout_weights=twpath,output_dir=output,
                            branch='tail',device='cpu',batch_size=32,kde_grid=12,code_commit='test')
    study.run(args)
    result=json.loads((output/'grid_study_summary.json').read_text())
    assert result['status']==('candidate_checked_not_certified' if with_candidate else 'no_stable_candidate')
    assert (result['candidate'] is not None)==with_candidate
    assert (result['checking_candidate'] is not None)==with_candidate
    if with_candidate:
        assert result['candidate_minus_original_fixed_cut_bootstrap'] is not None
    assert result['original_QCD_cache_agreement'] and not result['networks_updated']
    assert len(list((output/'plots').glob('*.png')))==10
    assert np.array_equal(inferred[0],cal_rows[labels[cal_rows]==1])
    assert np.array_equal(inferred[1],selected)
    with np.load(output/'checking_all_class_scores.npz') as saved:
        assert np.array_equal(saved['event_id'],check['event_id'][selected].numpy())
        assert np.allclose(saved['md'][saved['label']==1],md[qcd_rows])
    assert file_sha256(pilot/'pilot_comparison.json')==old_hash
    assert file_sha256(checkpoint)==ckpt_hash
    with pytest.raises(FileExistsError): study.run(args)
