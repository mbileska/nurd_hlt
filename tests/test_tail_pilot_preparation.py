import numpy as np
import pytest
import json
import sys
import torch

from scripts.prepare_tail_pilot import freeze_selection, main, summarize, validate_split, weight_stats


def test_threshold_is_training_only_physics_weighted_not_raw():
    scores = np.array([1., 2., 3., 100., 200., 300.])
    weights = np.array([100., 1., 1., 1., 1., 1.])
    train, validation = np.array([0,1,2,3]), np.array([4,5])
    cut, selected, st, sv = freeze_selection(scores, weights, train, validation)
    assert cut == 1.
    changed_scores, changed_weights = scores.copy(), weights.copy()
    changed_scores[validation] = 0.
    changed_weights[validation] = 1e20
    new_cut, _, new_st, new_sv = freeze_selection(changed_scores, changed_weights, train, validation)
    assert new_cut == cut
    assert np.array_equal(st, new_st)
    assert len(sv) == 2 and len(new_sv) == 0
    assert selected.all()  # A large boundary weight can exceed nominal 10%; ties explicit.


def test_split_rejects_leakage_and_missing_rows():
    with pytest.raises(ValueError, match="partition"):
        validate_split([0,1], [1,2], 4)
    with pytest.raises(ValueError, match="partition"):
        validate_split([0,1], [2,4], 4)


def test_report_distinguishes_counts_effective_size_and_missing_classes():
    labels = np.tile(np.arange(8), 4)
    weights = np.ones(32)
    weights[0] = 10
    names = {i: str(i) for i in range(8)}
    rows = summarize(labels, weights, np.arange(16), np.arange(16,32),
                     np.arange(8), np.arange(16,24), names)
    assert len(rows) == 16
    assert all(r['events'] == 1 for r in rows)
    assert all(r['effective_events'] == 1 for r in rows)
    rows = summarize(labels, weights, np.arange(16), np.arange(16,32),
                     np.arange(7), np.arange(16,24), names)
    assert next(r for r in rows if r['split']=='train' and r['label']==7)['events'] == 0
    assert weight_stats([10,1])['effective_events'] == pytest.approx(121/101)


def test_end_to_end_preparation_saves_selection_not_training(tmp_path, monkeypatch):
    from models.hlt_autoencoder import HLTAutoencoder
    from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature
    from utils.hlt_targets import DECAY8_NAMES
    parents = torch.tensor([0,1,2,3,3,3,0,0]).repeat(12)
    categories = torch.tensor([15,0,27,6,9,12,18,21]).repeat(12)
    obj = torch.arange(1,13).repeat_interleave(8).float().view(-1,1,1).expand(-1,1,4).clone()
    raw = {'pf': torch.zeros(96,3,7), 'obj': obj, 'label': parents,
           'background_category': categories,
           'event_id': torch.stack((parents, torch.zeros(96).long(), torch.zeros(96).long(), torch.arange(96)), 1)}
    sample_path, weights_path, ae_path, ckpt_path = [tmp_path/name for name in ('sample.pt','weights.pt','ae.pth','main.pth')]
    torch.save(raw, sample_path)
    torch.save({'weights': torch.ones(96), 'event_id': raw['event_id'], 'weight_scope': 'all_events'}, weights_path)
    _, metadata = load_generator_weights(str(weights_path), parents, sample=raw, weight_scope='all_events')
    cfg = {'features':4,'latent_dim':2,'encoder_config':{'nodes':[6]},'decoder_config':{'nodes':[6,4]}}
    ae = HLTAutoencoder(cfg)
    for parameter in ae.parameters():
        parameter.data.zero_()
    scaler = {'mu':torch.zeros(4),'std':torch.ones(4)}
    torch.save({'ae':ae.state_dict(),'ae_config':cfg,'weighting':{'scheme':'physics'}}, ae_path)
    torch.save({'ae_checkpoint_sha256':file_sha256(ae_path),'ae_scaler':scaler,
                'preprocessing':{'target_scheme':'decay8','target_class_names':DECAY8_NAMES,
                                 'label_map':{i:i for i in range(8)},'data_signature':sample_signature(raw),
                                 'weighting':{'generator':metadata},
                                 'split':{'train_indices':torch.arange(80),'validation_indices':torch.arange(80,96)}}}, ckpt_path)
    output = tmp_path/'prepared'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--checkpoint',str(ckpt_path), '--ae-checkpoint',str(ae_path),
                                    '--sample',str(sample_path),'--weights',str(weights_path),'--output-dir',str(output)])
    main()
    report = json.loads((output/'preparation_report.json').read_text())
    assert report['class_coverage_ready']
    assert not report['training_submitted'] and not report['heldout_opened']
    assert report['ae_threshold'] == 81.
    assert len(report['classes']) == 16
    assert all(r['class_balanced_mass_fraction'] == pytest.approx(.125) for r in report['classes'])
    with np.load(output/'pilot_selection.npz') as data:
        assert np.all(data['ae_score'][data['selected_train_indices']] >= 81)
        assert not np.intersect1d(data['selected_train_indices'],data['selected_validation_indices']).size
