"""Recover only a missing JSON report from an existing completed score preparation.

Checks stored provenance and selection against cached arrays and checkpoints.
No scoring, training, new threshold fit for use, or overwriting of artifacts.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.prepare_tail_pilot import freeze_selection, report_json, weight_stats
from utils.hlt_weights import file_sha256, tensor_sha256


def recover(preparation_dir, ae_checkpoint):
    destination=preparation_dir/'preparation_report.json'
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    payload=torch.load(preparation_dir/'pilot_preparation.pth',map_location='cpu',weights_only=False)
    report=payload['provenance']
    checkpoint_path=Path(report['classifier_checkpoint'])
    if file_sha256(checkpoint_path)!=report['classifier_checkpoint_sha256']:
        raise ValueError('Classifier checkpoint hash mismatch')
    if file_sha256(ae_checkpoint)!=report['ae_checkpoint_sha256']:
        raise ValueError('AE checkpoint hash mismatch')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint['preprocessing']['data_signature']!=report['data_signature']:
        raise ValueError('Checkpoint sample signature mismatch')
    if checkpoint['preprocessing']['weighting']['generator']['effective_physics_weight_sha256']!=report['physics_weight_sha256']:
        raise ValueError('Checkpoint physics-weight provenance mismatch')
    with np.load(preparation_dir/'pilot_selection.npz',allow_pickle=False) as cache:
        weights=cache['physics_weight']; labels=cache['label']; scores=cache['ae_score']
        train,validation=cache['train_indices'],cache['validation_indices']
        if tensor_sha256(weights)!=report['physics_weight_sha256']:
            raise ValueError('Cached original physics-weight checksum mismatch')
        if len(labels)!=report['data_signature']['n_events']:
            raise ValueError('Cached event count mismatch')
        if tensor_sha256(cache['event_id'])!=report['data_signature']['event_id_sha256']:
            raise ValueError('Cached composite event-ID checksum mismatch')
        for name,indices in (('train',train),('validation',validation)):
            if tensor_sha256(indices)!=report[f'original_{name}_indices_sha256']:
                raise ValueError('Cached split checksum mismatch')
            if not np.array_equal(indices,checkpoint['preprocessing']['split'][f'{name}_indices'].cpu().numpy()):
                raise ValueError('Cached split differs from classifier checkpoint')
        cut,selected,st,sv=freeze_selection(scores,weights,train,validation)
        if cut!=report['ae_threshold'] or cut!=payload['ae_threshold'] or not np.array_equal(selected,cache['selected']):
            raise ValueError('Cached frozen selection cannot be reproduced')
        if not np.array_equal(st,cache['selected_train_indices']) or not np.array_equal(sv,cache['selected_validation_indices']):
            raise ValueError('Cached selected rows mismatch')
        for row in report['classes']:
            indices=st if row['split']=='train' else sv
            stats=weight_stats(weights[indices[labels[indices]==row['label']]])
            for name in ('events','sum_weights','sum_weights_squared','effective_events'):
                if not np.isclose(stats[name],row[name],rtol=1e-10,atol=1e-8):
                    raise ValueError(f"Cached class statistics differ: {row['split']} {row['label']} {name}")
    serialized=report_json(report)
    # Exclusive creation: preserve existing NPZ/PTH/CSV and never replace a report.
    with destination.open('x') as stream:
        stream.write(serialized)
    print(f'Recovered verified report: {destination}')
    print('Existing scores, weights, splits, threshold and CSV were not changed. No inference or training.')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-dir',required=True,type=Path)
    parser.add_argument('--ae-checkpoint',required=True,type=Path)
    args=parser.parse_args()
    recover(args.preparation_dir,args.ae_checkpoint)


if __name__=='__main__':
    main()
