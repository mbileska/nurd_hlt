import torch
import numpy as np

from dataset.hlt_smcocktail_dataset import _make_nurd_weights
from utils.hlt_training_stats import (
    QCDRichBatchSampler,
    RunningQCDMDProxy,
    cross_fitted_mahalanobis,
    soft_conditioner_profile_loss,
    soft_copula_grid_loss,
)


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
    proxy.finalize_epoch()

    covariance = proxy.second_moment - torch.outer(proxy.mean, proxy.mean)
    assert torch.allclose(proxy.mean, torch.tensor([1.0, 1.0]))
    assert torch.allclose(covariance, torch.eye(2), atol=1e-6)


def test_qcd_md_proxy_reference_is_frozen_for_the_epoch():
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
    proxy.finalize_epoch()

    scores = proxy.md(shifted, mask, update=True)

    # Scoring accumulates shifted moments but leaves the active reference fixed.
    assert scores.mean() > 20.0
    assert torch.allclose(proxy.mean, torch.tensor([1.0, 1.0]))
    proxy.finalize_epoch()
    assert torch.allclose(proxy.mean, torch.tensor([5.0, 5.0]))

    restored = RunningQCDMDProxy()
    restored.load_state_dict(proxy.state_dict())
    assert restored.updates == proxy.updates
    assert torch.allclose(restored.mean, proxy.mean.cpu())
    assert torch.allclose(restored.second_moment, proxy.second_moment.cpu())


def test_reverse_profile_has_conditioner_gradient():
    conditioner = torch.linspace(-2.0, 2.0, 256, requires_grad=True)
    target = conditioner.detach().pow(2) + 0.1 * torch.sin(
        conditioner.detach() * 4.0)
    loss, _ = soft_conditioner_profile_loss(
        conditioner, target, n_bins=8)
    loss.backward()

    assert conditioner.grad is not None
    assert conditioner.grad.abs().sum() > 0


def test_copula_grid_has_gradients_for_both_axes():
    x = torch.linspace(-2.0, 2.0, 256, requires_grad=True)
    y = (0.7 * x.detach() + torch.sin(x.detach())).requires_grad_()
    loss, metric = soft_copula_grid_loss(
        x, y, [0.5, 0.7, 0.9])
    loss.backward()

    assert metric > 0
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert y.grad is not None and y.grad.abs().sum() > 0


def test_cross_fitted_mahalanobis_scores_every_event():
    rng = np.random.default_rng(3)
    latents = rng.normal(size=(200, 6))
    scores = cross_fitted_mahalanobis(latents, n_splits=2, seed=4)

    assert scores.shape == (200,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_qcd_rich_sampler_composition_and_correction():
    labels = torch.tensor([1] * 20 + [0] * 80)
    sampler = QCDRichBatchSampler(
        labels, batch_size=20, qcd_label=1, qcd_fraction=0.4,
        seed=9)
    batch = next(iter(sampler))
    batch_labels = labels[batch]
    correction = sampler.sampling_correction()

    assert int((batch_labels == 1).sum()) == 8
    assert np.isclose(correction["qcd"], 0.2 / 0.4)
    assert np.isclose(correction["other"], 0.8 / 0.6)
