import pytest
import torch
import torch.nn as nn

from models.hlt_con import HLTCritic
from utils.hlt_density_ratio import (
    critic_context_only_accuracy,
    density_ratio_critic_loss,
    engineer_information_penalty,
    make_density_ratio_examples,
    sample_nuisance_donor_indices,
)


def test_critic_accepts_continuous_nuisance_and_has_engineer_dimensions():
    critic = HLTCritic(latent_dim=6, num_classes=4)
    assert critic.net[0].in_features == 6 + 1 + 4
    assert critic.net[0].out_features == 256
    latent = torch.randn(12, 6)
    labels = torch.arange(12) % 4
    nuisance = torch.linspace(-2.0, 2.0, 12)
    assert critic(latent, labels, nuisance).shape == (12, 2)


def test_real_and_shuffled_examples_duplicate_only_event_context():
    latent = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    labels = torch.tensor([0, 1, 2, 3])
    nuisance = torch.tensor([0.1, 0.2, 0.3, 0.4])
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    permutation = torch.tensor([3, 2, 1, 0])
    rx, y, z, w, targets = make_density_ratio_examples(
        latent, labels, nuisance, weights, permutation,
        shuffle_mode="global")
    assert torch.equal(rx[:4], rx[4:])
    assert torch.equal(y[:4], y[4:])
    assert torch.equal(w[:4], w[4:])
    assert torch.equal(z[4:], nuisance[permutation])
    assert targets.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]


def test_weighted_within_class_donors_preserve_labels_and_weights():
    labels = torch.tensor([0, 0, 1, 1])
    # A zero-weight row must never donate; the positive row in each class is
    # therefore selected deterministically.
    weights = torch.tensor([0.0, 2.0, 0.0, 5.0])
    donors = sample_nuisance_donor_indices(labels, weights)
    assert donors.tolist() == [1, 1, 3, 3]
    assert torch.equal(labels, labels[donors])


def test_weighted_within_class_rejects_cross_class_explicit_donors():
    latent = torch.zeros(4, 1)
    labels = torch.tensor([0, 0, 1, 1])
    nuisance = torch.arange(4, dtype=torch.float32)
    weights = torch.ones(4)
    with pytest.raises(ValueError, match="must preserve class labels"):
        make_density_ratio_examples(
            latent, labels, nuisance, weights,
            permutation=torch.tensor([2, 3, 0, 1]))


def test_density_ratio_loss_has_no_two_b_vs_b_weight_mismatch():
    critic = HLTCritic(latent_dim=3, num_classes=2)
    latent = torch.randn(9, 3, requires_grad=True)
    labels = torch.arange(9) % 2
    nuisance = torch.randn(9)
    weights = torch.linspace(0.1, 2.0, 9)
    loss, accuracy, logits = density_ratio_critic_loss(
        critic, latent, labels, nuisance, weights)
    assert loss.ndim == 0
    assert 0.0 <= float(accuracy) <= 1.0
    assert logits.shape == (18, 2)
    loss.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


class ZeroCritic(nn.Module):
    def forward(self, latent, labels, nuisance):
        return torch.zeros(latent.shape[0], 2, device=latent.device)


def test_critic_uses_fixed_full_split_weight_normalization():
    latent = torch.zeros(2, 1)
    labels = torch.zeros(2, dtype=torch.long)
    nuisance = torch.zeros(2)
    weights = torch.tensor([1.0, 3.0])
    loss, _, _ = density_ratio_critic_loss(
        ZeroCritic(), latent, labels, nuisance, weights,
        permutation=torch.tensor([1, 0]))
    assert float(loss) == pytest.approx(2.0 * torch.log(torch.tensor(2.0)).item())


def test_context_only_accuracy_is_half_for_uninformative_critic():
    latent = torch.randn(4, 2)
    labels = torch.tensor([0, 0, 1, 1])
    nuisance = torch.arange(4, dtype=torch.float32)
    weights = torch.ones(4)
    value = critic_context_only_accuracy(
        ZeroCritic(), latent, labels, nuisance, weights,
        permutation=torch.tensor([1, 0, 3, 2]))
    assert float(value) == pytest.approx(0.5)


class ProductCritic(nn.Module):
    def forward(self, latent, labels, nuisance):
        score = latent[:, 0] * nuisance
        return torch.stack([torch.zeros_like(score), score], dim=1)


def test_engineer_penalty_is_real_tuple_log_odds_and_backpropagates():
    latent = torch.tensor([[2.0], [-1.0]], requires_grad=True)
    nuisance = torch.tensor([3.0, 4.0])
    labels = torch.tensor([0, 0])
    penalty = engineer_information_penalty(
        ProductCritic(), latent, labels, nuisance)
    assert penalty.tolist() == pytest.approx([6.0, -4.0])
    penalty.mean().backward()
    assert latent.grad[:, 0].tolist() == pytest.approx([1.5, 2.0])
