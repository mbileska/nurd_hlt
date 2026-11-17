"""One fresh random-split physical AE and a frozen, report-only baseline comparison."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.hlt_autoencoder import HLTAutoencoder
from scripts.audit_tail_pilot_split import quantile, retention, cdf_plot
from scripts.prepare_tail_pilot import report_json
from utils.hlt_ae_control import random_category_split, validate_roles
from utils.hlt_data import load_hlt_sample
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature, stratified_split_indices, tensor_sha256


def score(payload, raw, device, batch_size=4096):
    model=HLTAutoencoder(payload['ae_config']).to(device)
    model.load_state_dict(payload['ae']); model.eval()
    mu=payload['ae_scaler']['mu'].cpu().float()
    std=payload['ae_scaler']['std'].cpu().float()
    std=torch.where(std<1e-8,torch.ones_like(std),std)
    scores=np.empty(len(raw['label']),np.float32)
    start_time=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(scores),batch_size):
            obj=torch.nan_to_num(raw['obj'][start:start+batch_size,:,:4].reshape(-1,mu.numel()).float(),nan=0.,posinf=0.,neginf=0.)
            batch=((obj-mu)/std).to(device)
            reconstruction,_=model(batch)
            scores[start:start+len(batch)]=(reconstruction-batch).square().mean(1).cpu().numpy()
            if start==0 or start//batch_size%50==0 or start+len(batch)==len(scores):
                print(f"AE scoring {start+len(batch):,}/{len(scores):,}, elapsed {time.monotonic()-start_time:.1f}s",flush=True)
    del model; gc.collect()
    if device=='cuda':
        torch.cuda.empty_cache()
    if not np.isfinite(scores).all() or np.any(scores<0):
        raise ValueError('Invalid AE scores')
    return scores


def training_command(args, plan, config):
    if config['decoder_config']['nodes'][-1]!=config['features']:
        raise ValueError('AE decoder output shape mismatch')
    return [sys.executable,str(Path(__file__).resolve().parents[1]/'train_ae.py'),
            '--data',str(args.sample),'--gen_weight_path',str(args.weights),
            '--generator_weight_scope','all_events','--ae_weighting_scheme','physics',
            '--split_plan',str(plan),'--output_dir',str(args.output_dir/'ae_random_category'),
            '--exp_name',args.output_dir.name,'--epochs',str(args.epochs),'--patience',str(args.patience),
            '--batch_size','4096','--lr','1e-3','--weight_decay','0','--cosine','1',
            '--val_split','.05',
            '--latent_dim',str(config['latent_dim']),'--enc_nodes',*[str(v) for v in config['encoder_config']['nodes']],
            '--dec_nodes',*[str(v) for v in config['decoder_config']['nodes'][:-1]],
            '--manualSeed',str(args.seed),'--local_testing','1']


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    baseline_hash=file_sha256(args.baseline_ae)
    baseline=torch.load(args.baseline_ae,map_location='cpu',weights_only=False)
    if baseline['weighting']['scheme']!='physics':
        raise ValueError('Baseline must be the nominal physical-weighted AE')
    raw=load_hlt_sample(args.sample)
    weights,metadata=load_generator_weights(str(args.weights),raw['label'],sample=raw,qcd_label=1,weight_scope='all_events')
    signature=sample_signature(raw)
    if baseline['data_signature']!=signature or baseline['weighting']['generator']['effective_physics_weight_sha256']!=metadata['effective_physics_weight_sha256']:
        raise ValueError('Baseline AE does not match the complete training sample/weights')
    weights=weights.numpy()
    roles=validate_roles(random_category_split(raw['background_category'].numpy(),args.seed),len(weights))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    snapshot=args.output_dir/'baseline_checkpoint_ae.pth'
    shutil.copy2(args.baseline_ae,snapshot)
    if file_sha256(snapshot)!=baseline_hash:
        raise ValueError('Baseline snapshot checksum mismatch')
    plan=args.output_dir/'random_split_plan.pth'
    torch.save({'indices':roles,'seed':args.seed,'data_signature':signature,
                'physics_weight_sha256':metadata['effective_physics_weight_sha256'],
                'background_category_sha256':tensor_sha256(raw['background_category']),
                'fractions':[.9,.05,.025,.025]},plan)
    manifest={'baseline_path':str(args.baseline_ae.resolve()),'baseline_sha256':baseline_hash,
              'baseline_snapshot':str(snapshot),'split_plan_sha256':file_sha256(plan),
              'code_commit':args.code_commit,'seed':args.seed,
              'optimizer_settings_provenance':'Original full-campaign defaults: Adam lr=1e-3, batch=4096, cosine, no weight decay. '
                                              'Historical AE checkpoint lacks a complete optimizer/run config; new config is archived.',
              'roles':{role:{'events':len(rows),'indices_sha256':tensor_sha256(rows),
                             'physics_mass':float(weights[rows].sum())} for role,rows in roles.items()},
              'note':'New AE train/model-selection/calibration/check = 90/5/2.5/2.5% by event counts, approximately per fine category. '
                     'No physics-weight sorting. Old baseline used 90/10%; this control also separates model selection and rate calibration. '
                     'Calibration/check roles are never used for scaler fitting or epoch selection. Common outer held-out report is evaluated once after freezing.',
              'training_command':training_command(args,plan,baseline['ae_config'])}
    (args.output_dir/'control_manifest.json').write_text(report_json(manifest))
    print('Starting one fresh physical AE; original checkpoint is read-only and snapshotted.',flush=True)
    subprocess.run(manifest['training_command'],check=True,cwd=Path(__file__).resolve().parents[1])
    new_path=args.output_dir/'ae_random_category/checkpoint_ae.pth'
    new=torch.load(new_path,map_location='cpu',weights_only=False)
    if new['ae_config']!=baseline['ae_config'] or new['weighting']['scheme']!='physics':
        raise ValueError('Control architecture or physical weighting changed')
    for role,rows in roles.items():
        key={'train':'train_indices','validation':'validation_indices','calibration':'calibration_indices','check':'check_indices'}[role]
        if not np.array_equal(new['split'][key].cpu().numpy(),rows):
            raise ValueError(f'Control checkpoint role mismatch: {role}')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # No held-out access occurs until training and epoch selection have finished.
    heldout=load_hlt_sample(args.heldout)
    heldout_weights,heldout_meta=load_generator_weights(str(args.heldout_weights),heldout['label'],sample=heldout,qcd_label=1,weight_scope='all_events')
    heldout_weights=heldout_weights.numpy()
    ids=np.concatenate((raw['event_id'].numpy(),heldout['event_id'].numpy()))
    if len(np.unique(ids,axis=0))!=len(ids):
        raise ValueError('Training/held-out composite IDs overlap or contain duplicates')
    if 'split' in baseline:
        old_train=baseline['split']['train_indices'].cpu().numpy()
        old_val=baseline['split']['validation_indices'].cpu().numpy()
        old_membership='archived indices'
    else:
        m=baseline['weighting']
        a,b=stratified_split_indices(raw['label'],m['validation_fraction'],m['split_seed'],weights=torch.from_numpy(weights))
        old_train,old_val=a.numpy(),b.numpy()
        old_membership='reconstructed from saved seed/fraction and legacy splitter'
    results={}; rates=[]; score_cache={}
    plot_dir=args.output_dir/'plots'; plot_dir.mkdir()
    for name,payload,groups,calibration in (
        ('baseline',baseline,{'train':old_train,'model_selection_and_calibration':old_val},old_val),
        ('random_control',new,roles,roles['calibration'])):
        print(f'Frozen score comparison: {name}',flush=True)
        x=score(payload,raw,device); y=score(payload,heldout,device)
        cut=float(quantile(x[calibration],weights[calibration],.9))
        values={}
        for role,rows in groups.items():
            values[role]={'all_backgrounds':retention(x[rows],weights[rows],cut),
                          'QCD':retention(x[rows][raw['label'].numpy()[rows]==1],weights[rows][raw['label'].numpy()[rows]==1],cut),
                          'weighted_mean_MSE':float(np.dot(x[rows],weights[rows])/weights[rows].sum())}
        values['heldout']={'all_backgrounds':retention(y,heldout_weights,cut),
                           'QCD':retention(y[heldout['label'].numpy()==1],heldout_weights[heldout['label'].numpy()==1],cut),
                           'weighted_mean_MSE':float(np.dot(y,heldout_weights)/heldout_weights.sum())}
        results[name]={'selected_epoch':payload['epoch'],'ae_threshold':cut,'rates':values}
        for role,entry in values.items():
            rates.append({'model':name,'role':role,'ae_threshold':cut,
                          'all_background_retention':entry['all_backgrounds']['retention'],
                          'QCD_retention':entry['QCD']['retention'],
                          'QCD_retention_fixed_cut_error':entry['QCD']['standard_error_fixed_cut'],
                          'QCD_selected_ESS':entry['QCD']['selected_effective_events']})
        score_cache[f'{name}_training_manifest_scores']=x
        score_cache[f'{name}_heldout_scores']=y
        fig,ax=plt.subplots(figsize=(8,6))
        cdf_plot(ax,x[calibration],weights[calibration],'threshold reference (no AE gradient training)')
        if name=='random_control':
            cdf_plot(ax,x[roles['check']],weights[roles['check']],'disjoint check')
        cdf_plot(ax,y,heldout_weights,'common held-out report')
        lo,hi=quantile(x[calibration],weights[calibration],[.001,.999])
        ax.set(xlim=(lo,hi),xlabel='AE MSE (within-model units)',ylabel='Physics-weighted CDF',title=name)
        ax.axvline(cut,color='black',ls='--',label=f'frozen cut={cut:.5g}')
        ax.legend(); fig.tight_layout(); fig.savefig(plot_dir/f'{name}_rate_transfer.png',dpi=160); plt.close(fig)
    history=json.loads((args.output_dir/'ae_random_category/ae_training_history.json').read_text())
    fig,ax=plt.subplots(figsize=(8,5))
    for key in ('train_loss','validation_physics_loss'):
        ax.plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    ax.set(xlabel='Epoch',ylabel='Physical-weighted MSE',yscale='log'); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir/'new_AE_learning_curve.png',dpi=160); plt.close(fig)
    np.savez_compressed(args.output_dir/'frozen_AE_scores.npz',**score_cache,
                        training_event_id=raw['event_id'].numpy(),heldout_event_id=heldout['event_id'].numpy())
    with (args.output_dir/'rate_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rates[0])); writer.writeheader(); writer.writerows(rates)
    if file_sha256(args.baseline_ae)!=baseline_hash or file_sha256(snapshot)!=baseline_hash:
        raise ValueError('Original baseline checkpoint or snapshot changed')
    report={**manifest,'results':results,'baseline_membership_source':old_membership,
            'new_checkpoint':str(new_path),'new_checkpoint_sha256':file_sha256(new_path),
            'original_baseline_unchanged':True,'NURD_trained':False,
            'heldout_signature':sample_signature(heldout),'heldout_physics_weight_sha256':heldout_meta['effective_physics_weight_sha256'],
            'limitations':['Old AE validation also selected its epoch; its calibration is not independently reserved from model selection.',
                           'The new model has disjoint early-stop/calibration/check pools; both models share the same final held-out report.',
                           'Absolute standardized MSE values across AEs are not directly comparable because scalers are training-fitted.',
                           'Fixed-cut error bars omit finite calibration-threshold uncertainty. No signal sensitivity or ABCD/NURD closure claim.',
                           'Held-out data have been inspected in past campaigns: this is a frozen report-only control, not a new blind discovery test.']}
    (args.output_dir/'AE_control_summary.json').write_text(report_json(report))
    print(report_json({'results':results,'original_baseline_unchanged':True}),flush=True)
    print(f"AE CONTROL COMPLETE: {args.output_dir/'AE_control_summary.json'}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sample','weights','baseline-ae','heldout','heldout-weights','output-dir'):
        p.add_argument(f'--{name}',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=150); p.add_argument('--patience',type=int,default=25)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--code-commit',default='unspecified')
    args=p.parse_args()
    if args.epochs<1 or args.patience<1:
        p.error('Positive epochs and patience required')
    run(args)


if __name__=='__main__':
    main()
