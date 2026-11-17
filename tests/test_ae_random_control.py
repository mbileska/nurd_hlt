import argparse
import json
import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from utils.hlt_ae_control import random_category_split, validate_roles


def test_random_split_disjoint_deterministic_and_seed_sensitive():
    categories=np.repeat(np.arange(8),1000)
    a=random_category_split(categories,42)
    b=random_category_split(categories,43)
    validate_roles(a,len(categories))
    assert [len(a[k]) for k in a]==[7200,400,200,200]
    for role in a:
        np.testing.assert_array_equal(a[role],random_category_split(categories,42)[role])
        assert not np.array_equal(a[role],b[role])
        assert set(categories[a[role]])==set(range(8))
    assert np.isin(a['validation'],b['validation']).mean()<.15
    with pytest.raises(ValueError):
        random_category_split([0,0,0])
    broken={k:v.copy() for k,v in a.items()}
    broken['check'][0]=broken['train'][0]
    with pytest.raises(ValueError,match='partition'):
        validate_roles(broken,len(categories))


def test_control_gpu_script_no_partition_or_A100_request():
    source=Path(__file__).parents[1]/'slurm'
    worker=(source/'submit_ae_random_control.sbatch').read_text()
    assert '#SBATCH --gres=gpu:1' in worker
    assert '--partition' not in worker
    assert 'a100' not in worker.lower()
    assert '--constraint' not in worker
    assert 'train_hlt.py' not in worker


def test_full_control_keeps_baseline_and_archives_exact_splits(tmp_path):
    from models.hlt_autoencoder import HLTAutoencoder
    from scripts.run_ae_random_control import run
    from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature
    generator=torch.Generator().manual_seed(7)
    categories=torch.tensor([15,0,27,6,9,12,18,21]).repeat(40)
    parents=torch.tensor([0,1,2,3,3,3,0,0]).repeat(40)
    n=len(parents)
    def sample(offset):
        return {'pf':torch.zeros(n,2,7),'obj':torch.randn(n,2,4,generator=generator),
                'label':parents,'background_category':categories,
                'event_id':torch.stack((parents,torch.zeros(n).long(),torch.zeros(n).long(),torch.arange(n)+offset),1)}
    raw=sample(0); heldout=sample(10000)
    data,hw=tmp_path/'sample.pt',tmp_path/'heldout.pt'
    weights,hweights=tmp_path/'weights.pt',tmp_path/'heldout_weights.pt'
    torch.save(raw,data); torch.save(heldout,hw)
    for path,source in ((weights,raw),(hweights,heldout)):
        torch.save({'weights':torch.ones(n),'event_id':source['event_id'],'weight_scope':'all_events'},path)
    _,metadata=load_generator_weights(str(weights),parents,sample=raw,weight_scope='all_events')
    cfg={'features':8,'latent_dim':3,'encoder_config':{'nodes':[12]},'decoder_config':{'nodes':[12,8]},'alpha':1.}
    baseline=tmp_path/'baseline.pth'
    torch.save({'ae':HLTAutoencoder(cfg).state_dict(),'ae_config':cfg,
                'ae_scaler':{'mu':torch.zeros(8),'std':torch.ones(8)},
                'data_signature':sample_signature(raw),'epoch':1,
                'weighting':{'scheme':'physics','generator':metadata,'split_seed':42,'validation_fraction':.1}},baseline)
    original=file_sha256(baseline)
    args=argparse.Namespace(sample=data,weights=weights,heldout=hw,heldout_weights=hweights,
                            baseline_ae=baseline,output_dir=tmp_path/'control',
                            epochs=1,patience=1,seed=42,code_commit='test')
    run(args)
    report=json.loads((args.output_dir/'AE_control_summary.json').read_text())
    assert report['original_baseline_unchanged'] and not report['NURD_trained']
    assert file_sha256(baseline)==original
    assert file_sha256(args.output_dir/'baseline_checkpoint_ae.pth')==original
    plan=torch.load(args.output_dir/'random_split_plan.pth',weights_only=False)
    new=torch.load(args.output_dir/'ae_random_category/checkpoint_ae.pth',weights_only=False)
    for role,key in (('train','train_indices'),('validation','validation_indices'),
                     ('calibration','calibration_indices'),('check','check_indices')):
        np.testing.assert_array_equal(plan['indices'][role],new['split'][key].numpy())
    assert new['ae_config']==cfg
    assert new['weighting']['class_balancing'] is False
    assert new['config']['split_plan'] is not None
    assert new['weighting']['validation_fraction']==.05
    assert set(report['results'])=={'baseline','random_control'}
    assert (args.output_dir/'rate_comparison.csv').is_file()
    with pytest.raises(FileExistsError):
        run(args)
    # Calibration and check features cannot affect scaler, optimization, or
    # checkpoint selection. Change only those features and repeat one epoch.
    from scripts.run_ae_random_control import training_command
    altered=copy.deepcopy(raw)
    untouched=np.r_[plan['indices']['calibration'],plan['indices']['check']]
    altered['obj'][untouched]+=1000
    alternate_sample=tmp_path/'altered.pt'; torch.save(altered,alternate_sample)
    alternate_plan=copy.deepcopy(plan); alternate_plan['data_signature']=sample_signature(altered)
    alternate_plan_path=tmp_path/'altered_plan.pth'; torch.save(alternate_plan,alternate_plan_path)
    alternate_args=copy.copy(args)
    alternate_args.sample=alternate_sample; alternate_args.output_dir=tmp_path/'altered_control'
    subprocess.run(training_command(alternate_args,alternate_plan_path,cfg),check=True)
    alternate=torch.load(alternate_args.output_dir/'ae_random_category/checkpoint_ae.pth',weights_only=False)
    assert torch.equal(new['ae_scaler']['mu'],alternate['ae_scaler']['mu'])
    assert torch.equal(new['ae_scaler']['std'],alternate['ae_scaler']['std'])
    for key,value in new['ae'].items():
        assert torch.equal(value,alternate['ae'][key])
