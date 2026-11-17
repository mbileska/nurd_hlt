import argparse
import json
import sys

import numpy as np
import pytest
import torch

from scripts.audit_tail_pilot_split import (
    audit, bootstrap_gap, random_resplits, retention, standardized_validation, weighted_ks,
)


def test_weighted_cdf_handles_ties_and_retention_errors():
    x=np.array([0.,0.,1.,1.]); w=np.ones(4); val=np.array([False,True,False,True])
    assert weighted_ks(x,w,val)==0
    assert weighted_ks(np.array([0.,1.,0.,1.]),w,val)==1
    result=retention(x,w,.5)
    assert result['retention']==.5
    assert result['standard_error_fixed_cut']==pytest.approx(.25)
    assert retention(np.array([]),np.array([]),1)['retention'] is None


def test_common_composition_reports_missing_support():
    x=np.array([1.,0.,1.,0.]); w=np.ones(4); val=np.array([False,False,True,True])
    result=standardized_validation(x,w,val,np.array([0,1,0,1]),.5)
    assert result['validation_retention_at_train_group_composition']==.5
    result=standardized_validation(x,w,val,np.array([0,1,0,0]),.5)
    assert result['validation_retention_at_train_group_composition'] is None
    assert result['unsupported_train_groups']==[1]


def test_uncertainty_is_reproducible_and_refits_training_cut():
    rng=np.random.default_rng(3)
    scores=rng.uniform(size=800); weights=rng.uniform(.2,3,800)
    labels=np.tile(np.arange(8),100); train=np.arange(640); val=np.arange(640,800)
    a,summary=bootstrap_gap(scores,weights,labels,train,val,repeats=20,seed=4)
    b,_=bootstrap_gap(scores,weights,labels,train,val,repeats=20,seed=4)
    np.testing.assert_array_equal(a,b)
    assert np.ptp(a[:,0])>0
    assert summary['usable_replicates']==20
    values,reference=random_resplits(scores,weights,labels,labels,val,repeats=10)
    assert values.shape==(10,4)
    assert len(reference['delta_95pct_reference_interval'])==2


def test_full_audit_from_toy_checkpoint_and_preparation(tmp_path,monkeypatch):
    from models.hlt_autoencoder import HLTAutoencoder
    from scripts.prepare_tail_pilot import main as prepare
    from utils.hlt_targets import DECAY8_NAMES
    from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature, stratified_split_indices

    parent=torch.tensor([0,1,2,3,3,3,0,0]).repeat(12)
    target=torch.arange(8).repeat(12)
    categories=torch.tensor([15,0,27,6,9,12,18,21]).repeat(12)
    obj=torch.arange(1,13).repeat_interleave(8).float().view(-1,1,1).expand(-1,1,4).clone()
    raw={'pf':torch.zeros(96,3,7),'obj':obj,'label':parent,'background_category':categories,
         'event_id':torch.stack((parent,torch.zeros(96).long(),torch.zeros(96).long(),torch.arange(96)),1)}
    sample,weight_file,ae_file,model_file=[tmp_path/n for n in ('sample.pt','weight.pt','ae.pth','model.pth')]
    torch.save(raw,sample)
    torch.save({'weights':torch.ones(96),'event_id':raw['event_id'],'weight_scope':'all_events'},weight_file)
    weights,metadata=load_generator_weights(str(weight_file),parent,sample=raw,weight_scope='all_events')
    signature=sample_signature(raw)
    cfg={'features':4,'latent_dim':2,'encoder_config':{'nodes':[6]},'decoder_config':{'nodes':[6,4]}}
    ae=HLTAutoencoder(cfg)
    for parameter in ae.parameters():
        parameter.data.zero_()
    scaler={'mu':torch.zeros(4),'std':torch.ones(4)}
    at,av=stratified_split_indices(parent,.1,42,weights=weights)
    ae_meta={'scheme':'physics','generator':metadata,'split_seed':42,'validation_fraction':.1}
    for name,idx in (('training',at),('validation',av)):
        ae_meta[f'physics_{name}_class_mass']={int(c):float(weights[idx[parent[idx]==c]].sum()/weights[idx].sum()) for c in parent.unique()}
    torch.save({'ae':ae.state_dict(),'ae_config':cfg,'ae_scaler':scaler,'data_signature':signature,'weighting':ae_meta,
                'validation_losses':{'physics':float(obj[av].square().mean())}},ae_file)
    train,val=stratified_split_indices(target,.1,42,weights=weights)
    torch.save({'ae_checkpoint_sha256':file_sha256(ae_file),'ae_scaler':scaler,
                'preprocessing':{'target_scheme':'decay8','target_class_names':DECAY8_NAMES,'label_map':{i:i for i in range(8)},
                                 'data_signature':signature,'weighting':{'generator':metadata},
                                 'split':{'seed':42,'validation_fraction':.1,'train_indices':train,'validation_indices':val}}},model_file)
    prep=tmp_path/'prep'
    monkeypatch.setattr(sys,'argv',['prepare','--checkpoint',str(model_file),'--ae-checkpoint',str(ae_file),
                                   '--sample',str(sample),'--weights',str(weight_file),'--output-dir',str(prep)])
    prepare()
    output=tmp_path/'audit'
    args=argparse.Namespace(preparation_dir=prep,ae_checkpoint=ae_file,weights=weight_file,
                            output_dir=output,bootstrap=20,resplits=10,seed=42)
    audit(args)
    report=json.loads((output/'split_audit_summary.json').read_text())
    assert report['integrity_checks_passed']
    assert report['saved_classifier_split_reproduced_by_current_splitter']
    assert report['reconstructed_AE_mass_matches_checkpoint']
    assert report['AE_physics_weighted_loss_reconstruction']['validation_loss_reconstruction_consistent']
    assert report['score_parity']['training_cache_max_absolute_error']==0
    assert not report['heldout_opened'] and not report['training_submitted']
    assert len(list((output/'plots').glob('*.png')))==8
    assert (output/'input_distribution_comparison.csv').is_file()
    # Corrupted cache is a hard failure, not a distribution-shift interpretation.
    with np.load(prep/'pilot_selection.npz') as data:
        contents={key:data[key].copy() for key in data.files}
    contents['physics_weight'][0]+=1
    np.savez_compressed(prep/'pilot_selection.npz',**contents)
    args.output_dir=tmp_path/'corrupted'
    with pytest.raises(ValueError,match='misaligned'):
        audit(args)
