import torch
import numpy as np

from dataset.hlt_smcocktail_dataset import (
    _make_nurd_weights,
    apply_class_balance,
    class_balance_factors,
    weighted_cdf_coordinate,
)
from models.hlt_con import HLTCritic
from utils.hlt_training_stats import (
    QCDRichBatchSampler,
    RunningQCDMDProxy,
    classifier_checkpoint_eligible,
    conditional_cdf_loss,
    cross_fitted_mahalanobis,
    distance_corr_loss,
    soft_conditioner_profile_loss,
    soft_copula_grid_loss,
    weighted_balanced_folds,
    weighted_resample_indices,
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


def test_capped_nurd_weights_preserve_generator_weighted_mean():
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    nuisances = torch.tensor([0, 0, 1, 0, 1, 1])
    generator_weights = torch.tensor([1.0, 3.0, 2.0, 4.0, 1.0, 5.0])
    table = _make_nurd_weights(
        labels, nuisances, max_weight_ratio=4.0,
        base_weights=generator_weights)
    nurd_weights = torch.tensor([
        table[(int(label), int(nuisance))]
        for label, nuisance in zip(labels, nuisances)
    ])

    physical_mean = (
        generator_weights * nurd_weights
    ).sum() / generator_weights.sum()
    assert torch.isclose(physical_mean, torch.tensor(1.0), atol=1e-6)


def test_class_balanced_physical_measure_preserves_within_class_weights():
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    generator_weights = torch.tensor([1.0, 3.0, 10.0, 20.0, 30.0, 7.0])
    factors = class_balance_factors(labels, generator_weights)
    balanced = apply_class_balance(labels, generator_weights, factors)
    masses = torch.stack([
        balanced[labels == label].sum() for label in labels.unique()
    ])

    assert torch.allclose(masses, masses[0].expand_as(masses), atol=1e-5)
    assert torch.isclose(
        balanced[1] / balanced[0],
        generator_weights[1] / generator_weights[0])


def test_weighted_cdf_coordinate_is_monotonic_and_uses_physical_mass():
    reference = torch.tensor([0.0, 1.0, 2.0])
    weights = torch.tensor([8.0, 1.0, 1.0])
    values = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    coordinate = weighted_cdf_coordinate(values, reference, weights)

    assert torch.all(coordinate[1:] >= coordinate[:-1])
    assert coordinate[0] == 0.0 and coordinate[-1] == 1.0
    assert coordinate[1] > 0.35


def test_continuous_density_ratio_critic_accepts_qcd_cdf():
    critic = HLTCritic(
        latent_dim=6, num_classes=4, n_bins=50,
        critic_type="continuous_density_ratio", bin_resolutions=[50])
    latent = torch.randn(32, 6, requires_grad=True)
    labels = torch.ones(32, 1)
    nuisance_cdf = torch.linspace(0.0, 1.0, 32)

    output = critic(latent, labels, nuisance_cdf)
    output.sum().backward()

    assert output.shape == (32, 2)
    assert latent.grad is not None and latent.grad.abs().sum() > 0


def test_vector_distance_correlation_penalizes_full_latent_dependence():
    nuisance = torch.linspace(-2.0, 2.0, 256)
    latent = torch.stack([
        nuisance,
        nuisance.square(),
        torch.sin(3.0 * nuisance),
    ], dim=1).requires_grad_()

    loss, metric = distance_corr_loss(nuisance, latent, max_samples=0)
    loss.backward()

    assert metric > 0.5
    assert latent.grad is not None and latent.grad.abs().sum() > 0


def test_checkpoint_gate_rejects_collapsed_classifier():
    assert not classifier_checkpoint_eligible(0.25, 0.45)
    assert not classifier_checkpoint_eligible(float("nan"), 0.45)
    assert classifier_checkpoint_eligible(0.71, 0.45)
    assert not classifier_checkpoint_eligible(
        0.71, 0.45, [0.9, 0.0, 0.95, 0.99], 0.25)
    assert classifier_checkpoint_eligible(
        0.71, 0.45, [0.4, 0.7, 0.8, 0.94], 0.25)


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


def test_qcd_md_proxy_can_replace_reference_from_frozen_training_subset():
    proxy = RunningQCDMDProxy(momentum=0.5, eps=1e-6, mode="epoch")
    reference = torch.tensor([
        [0.0, 0.0], [0.0, 4.0], [2.0, 0.0], [2.0, 4.0],
    ])
    weights = torch.tensor([1.0, 1.0, 3.0, 3.0])

    assert proxy.replace_reference(reference, weights)
    assert torch.allclose(proxy.mean, torch.tensor([1.5, 2.0]))
    before = proxy.mean.clone()
    proxy.md(reference + 10.0, torch.ones(4, dtype=torch.bool), update=False)
    assert torch.allclose(proxy.mean, before)


def test_qcd_md_proxy_public_score_uses_frozen_reference():
    proxy = RunningQCDMDProxy(eps=1e-6, mode="epoch")
    reference = torch.tensor([
        [0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0],
    ])
    assert proxy.replace_reference(reference)

    values = reference + 3.0
    direct = proxy.score(values)
    through_md = proxy.md(
        values, torch.ones(values.size(0), dtype=torch.bool), update=False)

    assert torch.allclose(direct, through_md)


def test_qcd_md_proxy_ema_scores_before_tracking_current_batch():
    proxy = RunningQCDMDProxy(momentum=0.5, eps=1e-6, mode="ema")
    initial = torch.tensor([
        [0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0],
    ])
    shifted = initial + 4.0
    mask = torch.ones(shifted.size(0), dtype=torch.bool)
    proxy.update(initial)

    scores = proxy.md(shifted, mask, update=True)

    assert scores.mean() > 20.0
    assert torch.allclose(proxy.mean, torch.tensor([3.0, 3.0]))


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


def test_conditional_cdf_loss_detects_distribution_shift_and_has_gradient():
    conditioner = torch.linspace(0.0, 1.0, 400)
    dependent = (
        2.5 * conditioner + 0.15 * torch.sin(31.0 * conditioner)
    ).requires_grad_()
    independent = dependent.detach()[torch.randperm(dependent.numel())]

    dependent_loss, dependent_metric = conditional_cdf_loss(
        conditioner, dependent, n_bins=10)
    independent_loss, _ = conditional_cdf_loss(
        conditioner, independent, n_bins=10)
    dependent_loss.backward()

    assert dependent_metric > 0.1
    assert dependent_loss > independent_loss
    assert dependent.grad is not None and dependent.grad.abs().sum() > 0


def test_cross_fitted_mahalanobis_scores_every_event():
    rng = np.random.default_rng(3)
    latents = rng.normal(size=(200, 6))
    scores = cross_fitted_mahalanobis(latents, n_splits=2, seed=4)

    assert scores.shape == (200,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_weighted_resampling_follows_physical_mass():
    torch.manual_seed(12)
    weights = torch.tensor([99.0, 1.0])
    indices = weighted_resample_indices(weights, 5000)

    assert (indices == 0).float().mean() > 0.98


def test_weighted_folds_balance_extreme_generator_weights():
    weights = np.array([100.0, 80.0, 60.0, 40.0] + [1.0] * 100)
    fold_ids = weighted_balanced_folds(weights, n_splits=4, seed=5)
    masses = np.array([weights[fold_ids == fold].sum() for fold in range(4)])

    assert np.all(np.bincount(fold_ids, minlength=4) > 0)
    assert masses.max() - masses.min() <= weights.max()


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
