import argparse
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch

from models.hlt_con import HLTContrastiveModel
from scripts.run_legacy_double_nurd import checkpoint_blocks, tail_coefficient
from scripts.prepare_tail_pilot import report_json
from scripts.run_tail_nurd_pilot import finite_report
from scripts.study_tail_pilot_cuts import recount
from utils.hlt_weights import file_sha256

ROOT = Path(__file__).parents[1]
OLD = 'b5b2272939cfe8898c7001c5b3abfcfb8c94ef4e'


def test_tail_schedule_reaches_one_then_has_25_complete_epochs():
    assert [tail_coefficient(i,5,10,1.) for i in (1,5,6,15,16,40)] == [0.,0.,.1,1.,1.,1.]
    assert len(range(16,41)) == 25


def test_activation_checkpointing_preserves_weights_gradients_dropout_and_bn():
    torch.manual_seed(4)
    original = HLTContrastiveModel(4,embed_size=8,latent_dim=3,proj_dim=3,num_heads=2,
                                 num_layers=1,dim_ff=16,linear_dim=3,num_tokens=6,dropout=.1)
    memory = copy.deepcopy(original); checkpoint_blocks(memory)
    pf = torch.randn(8,6,7)*.1
    pf[...,0] = torch.rand(8,6)+1; pf[...,5]=1; pf[...,6]=211
    gradients=[]; states=[]
    for model in (original,memory):
        torch.manual_seed(80); model.train()
        latent,logits=model(pf); loss=logits.square().mean()+latent.square().mean()
        loss.backward()
        gradients.append({k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None})
        states.append({k:v.clone() for k,v in model.state_dict().items()})
    assert states[0].keys()==states[1].keys()
    for key in states[0]: assert torch.equal(states[0][key],states[1][key])
    for key in gradients[0]: assert torch.allclose(gradients[0][key],gradients[1][key],rtol=1e-5,atol=1e-6)


FIXTURE = r'''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import train_hlt as r
import torch
from models.hlt_autoencoder import HLTAutoencoder
from utils.hlt_weights import load_generator_weights,sample_signature,file_sha256
root=Path(sys.argv[2]);torch.manual_seed(1)
n=320
def sample(n):
    pf=torch.randn(n,6,7)*.1;pf[:,:,0]=torch.rand(n,6)+1;pf[:,:,5]=1;pf[:,:,6]=211
    return {'pf':pf,'obj':torch.rand(n,2,4)*2,'label':torch.arange(4).repeat(n//4)}
raw=sample(n);test=sample(384)
torch.save(raw,root/'train.pt');torch.save(test,root/'test.pt')
torch.save(torch.linspace(.5,2.5,n),root/'weights.pt')
torch.save(torch.linspace(.5,2.5,384),root/'test_weights.pt')
physics,meta=load_generator_weights(str(root/'weights.pt'),raw['label'],sample=raw)
ac={'features':8,'latent_dim':3,'encoder_config':{'nodes':[12]},'decoder_config':{'nodes':[12,8]},'alpha':1.}
ae=HLTAutoencoder(ac);ae.eval();scaler={'mu':torch.zeros(8),'std':torch.ones(8)}
torch.save({'ae':ae.state_dict(),'ae_config':ac,'ae_scaler':scaler,'data_signature':sample_signature(raw),
    'epoch':1,'weighting':{'method':'generator_weighted_uniform_class','generator':meta,'split_seed':42,'validation_fraction':.1}},root/'ae.pth')
cfg=vars(r.build_parser().parse_args(['--data',str(root/'train.pt'),'--ae_ckpt',str(root/'ae.pth')]))
cfg.update(gen_weight_path=str(root/'weights.pt'),batch_size=16,ae_batch_size=32,embed_size=8,latent_dim=3,proj_dim=3,
    num_heads=2,num_layers=1,dim_ff=16,linear_dim=3,contrast_weight=.3,critic_steps=3,
    code_commit='b5b2272939cfe8898c7001c5b3abfcfb8c94ef4e')
ds,val,pre=r.build_hlt_datasets(cfg['data'],ae,gen_weight_path=cfg['gen_weight_path'],ae_scaler=scaler,ae_batch_size=32)
model=r.HLTContrastiveModel(4,embed_size=8,latent_dim=3,proj_dim=3,num_heads=2,num_layers=1,dim_ff=16,linear_dim=3,num_tokens=6)
critic=r.HLTCritic(3,4)
metrics=r.validate(model,critic,torch.utils.data.DataLoader(val,batch_size=16),'cpu')
torch.save({'epoch':24,'config':cfg,'state_dict_model':model.state_dict(),'state_dict_critic':critic.state_dict(),
 'preprocessing':pre,'ae_scaler':scaler,'ae_checkpoint_sha256':file_sha256(root/'ae.pth'),'validation_metrics':metrics},root/'golden.pth.tar')
'''


@pytest.fixture
def legacy_fixture(tmp_path):
    legacy=tmp_path/'pinned';legacy.mkdir()
    archive=subprocess.run(['git','archive',OLD,'train_hlt.py','dataset','models','utils'],cwd=ROOT,capture_output=True,check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream: stream.extractall(legacy,filter='data')
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLCONFIGDIR':str(tmp_path/'mpl')}
    subprocess.run([sys.executable,'-c',FIXTURE,str(legacy),str(tmp_path)],env=env,check=True,capture_output=True,text=True,timeout=120)
    return tmp_path,legacy,env


def test_pinned_preparation_fresh_pair_and_all_six_standard_suites(legacy_fixture,monkeypatch):
    root,legacy,env=legacy_fixture
    driver=ROOT/'scripts/run_legacy_double_nurd.py';output=root/'experiment'
    common=['--legacy-code',str(legacy),'--sample',str(root/'train.pt'),'--weights',str(root/'weights.pt'),
            '--output-dir',str(output),'--device','cpu','--threads','1']
    source_hash=file_sha256(root/'golden.pth.tar');ae_hash=file_sha256(root/'ae.pth')
    prep=subprocess.run([sys.executable,str(driver),'prepare',*common,'--golden-checkpoint',str(root/'golden.pth.tar'),
        '--ae-checkpoint',str(root/'ae.pth')],env=env,capture_output=True,text=True,timeout=120)
    assert prep.returncode==0,prep.stdout+prep.stderr
    report=json.loads((output/'preparation_complete.json').read_text())
    assert report['legacy_commit']==OLD and report['golden_sha256']==source_hash
    assert report['golden_validation']['weighted_ce']==pytest.approx(report['archived_validation']['weighted_ce'],abs=1e-3)
    for arm in ('single','double'):
        run=subprocess.run([sys.executable,str(driver),'train',*common,'--arm',arm,'--epochs','2',
            '--tail-warmup-epochs','1','--tail-ramp-epochs','0','--min-post-ramp-epochs','1',
            '--tail-batch-size','8','--save-every','1'],env=env,capture_output=True,text=True,timeout=120)
        assert run.returncode==0,run.stdout+run.stderr
    single=torch.load(output/'single/checkpoint_epoch_001.pth.tar',map_location='cpu',weights_only=False)
    double=torch.load(output/'double/checkpoint_epoch_001.pth.tar',map_location='cpu',weights_only=False)
    # The entire first epoch is identical BEFORE any extra encoder pressure,
    # despite all the new tail-critic fitting and sampler operations.
    for key,value in single['state_dict_model'].items(): assert torch.equal(value,double['state_dict_model'][key]),key
    assert single['config']['balance_strata']==20 and single['config']['contrast_weight']==.3
    for arm in ('single','double'):
        completed=json.loads((output/arm/'training_complete.json').read_text())
        assert completed['epochs_trained']==2 and not completed['early_stopping']
        assert (output/arm/'checkpoint_main.pth.tar').is_file()
    assert file_sha256(root/'golden.pth.tar')==source_hash and file_sha256(root/'ae.pth')==ae_hash
    from scripts import evaluate_legacy_double_nurd as evaluation
    args=argparse.Namespace(experiment_dir=output,output_dir=root/'scores',sample=root/'train.pt',weights=root/'weights.pt',
        heldout=root/'test.pt',heldout_weights=root/'test_weights.pt',device='cpu',batch_size=64,full_d_min=1,tail_d_min=1,
        code_commit='test',kde_grid=8)
    original_load=torch.load
    def guarded_load(path,*a,**kw):
        if Path(path)==args.heldout:
            plan=json.loads((args.output_dir/'frozen_evaluation_plan.json').read_text())
            assert set(plan['plans'])=={'golden','single','double'}
            assert not plan['checking_used_for_cut_selection']
        return original_load(path,*a,**kw)
    monkeypatch.setattr(torch,'load',guarded_load)
    evaluation.score(args)
    with np.load(args.output_dir/'heldout_common_scores.npz') as common_scores:
        with np.load(args.output_dir/'golden_heldout_predictions.npz') as predictions:
            data={**{k:common_scores[k].copy() for k in common_scores.files},**{k:predictions[k].copy() for k in ('md',)}}
    qc=evaluation.qcd(data);tx,ty=np.median(qc['ae']),np.median(qc['md'])
    point=recount(qc,tx,ty,[1]*4)[0]
    old_summary=root/'old_summary.json'
    old_summary.write_text(report_json(finite_report({'evaluations':{'held-out':{'diagnostics':{
        'working_point':{'absolute_nonclosure':point['absolute_nonclosure']},'heldout_grid':{'median_absolute_nonclosure':.1}},
        'thresholds':{'t1':tx,'t2':ty}}}})))
    plotted=[];original_render=evaluation.render_suite
    def render(*a,**kw):
        plotted.append(a[5])
        if len(plotted)==1:return original_render(*a,**kw)
        a[5].mkdir(parents=True)
        return {'signal_sample_evaluated':False}
    monkeypatch.setattr(evaluation,'render_suite',render)
    plot_args=argparse.Namespace(score_dir=args.output_dir,experiment_dir=output,output_dir=root/'plots',
        golden_summary=old_summary,kde_grid=8)
    evaluation.plots(plot_args)
    assert len(plotted)==6
    result=json.loads((plot_args.output_dir/'legacy_comparison.json').read_text())
    assert set(result['results'])=={'golden','single','double'}
    assert result['golden_heldout_reproduction']['agrees_within_0p1_percentage_point']
    for arm in result['results']:
        for fraction in ('f100','f010'):
            assert (plot_args.output_dir/arm/fraction/'plots/confusion_matrix.png').is_file()
    for name in ('closure_scan_2d.png','hist2d_pca_md_kde.png','pca_corner.png','hist2d_QCD.png'):
        assert (plotted[0]/name).is_file()


def test_launcher_pins_old_recipe_and_never_retrains_ae_or_requests_specific_gpu():
    launch=(ROOT/'slurm/launch_legacy_double_nurd.sh').read_text()
    worker=(ROOT/'slurm/submit_legacy_double_nurd.sbatch').read_text()
    assert OLD in launch and 'git archive "$OLD_COMMIT"' in launch
    assert '--gres=gpu:1' in launch and 'a100' not in launch.lower()
    assert '--partition=gpu' not in launch and 'train_ae.py' not in worker
    assert 'afterok:' in launch and 'LEGACY_EPOCHS < 40' in launch
    assert 'run_legacy_double_nurd.py" train' in worker
    assert worker.index('conda activate disco')<worker.index('set -u')
