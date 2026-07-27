import torch

from dataset.hlt_smcocktail_dataset import _make_nurd_weights
from utils.hlt_training_stats import RunningQCDMDProxy


def test_capped_nurd_weights_preserve_sample_mean():
    labels = torch.tensor([0] * 8 + [1] * 4 + [2] * 2)
    nuisances = torch.tensor(
        [0] * 7 + [1] + [0] + [1] * 3 + [0, 1])
    table = _make_nurd_weights(labels, nuisances, max_weight_ratio=2.0)
    sample_weights = torch.tensor([
        table[(int(label), int(nuisance))]
        for label, nuisance in zip(labels, nuisances)
    ])

    assert torch.isclose(sample_weights.mean(), torch.tensor(1.0), atol=1e-6)
    assert sample_weights.max() <= 2.0
    assert sample_weights.min() > 0.0


def test_qcd_md_proxy_tracks_full_second_moment():
    proxy = RunningQCDMDProxy(momentum=0.5, eps=1e-6)
    reference = torch.tensor([
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [2.0, 2.0],
    ])
    proxy.update(reference)

    covariance = proxy.second_moment - torch.outer(proxy.mean, proxy.mean)
    assert torch.allclose(proxy.mean, torch.tensor([1.0, 1.0]))
    assert torch.allclose(covariance, torch.eye(2), atol=1e-6)


def test_qcd_md_proxy_scores_before_updating():
    proxy = RunningQCDMDProxy(momentum=0.5, eps=1e-6)
    initial = torch.tensor([
        [0.0, 0.0],
        [0.0, 2.0],
        [2.0, 0.0],
        [2.0, 2.0],
    ])
    shifted = initial + 4.0
    mask = torch.ones(shifted.size(0), dtype=torch.bool)
    proxy.update(initial)

    scores = proxy.md(shifted, mask, update=True)

    # Scores use the old mean [1, 1], while the stored reference moves only
    # after scoring to 0.5 * [1, 1] + 0.5 * [5, 5] = [3, 3].
    assert scores.mean() > 20.0
    assert torch.allclose(proxy.mean, torch.tensor([3.0, 3.0]))

    restored = RunningQCDMDProxy()
    restored.load_state_dict(proxy.state_dict())
    assert restored.updates == proxy.updates
    assert torch.allclose(restored.mean, proxy.mean.cpu())
    assert torch.allclose(restored.second_moment, proxy.second_moment.cpu())
