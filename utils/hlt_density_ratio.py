"""Engineer-style continuous density-ratio objectives for HLT NURD."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1)
    weights = weights.reshape(-1).to(values.dtype)
    if values.numel() != weights.numel():
        raise ValueError("values and weights must have the same number of entries.")
    return (values * weights).sum() / weights.sum().clamp(min=1e-12)


def sample_nuisance_donor_indices(
    labels: torch.Tensor,
    weights: torch.Tensor,
    shuffle_mode: str = "weighted_within_class",
) -> torch.Tensor:
    """Sample nuisance donors for the shuffled density-ratio population.

    ``weighted_within_class`` samples an independent nuisance value from the
    batch's weighted conditional distribution p_w(z | y).  ``global`` keeps
    the engineer-reference permutation available for exact comparisons.
    """
    labels = labels.reshape(-1)
    weights = weights.reshape(-1).to(dtype=torch.float32)
    if labels.numel() != weights.numel():
        raise ValueError("labels and weights must contain the same number of events.")
    if shuffle_mode == "global":
        return torch.randperm(labels.numel(), device=labels.device)
    if shuffle_mode != "weighted_within_class":
        raise ValueError(
            "shuffle_mode must be 'global' or 'weighted_within_class'.")

    donors = torch.empty(labels.numel(), dtype=torch.long, device=labels.device)
    for label in labels.unique():
        positions = torch.nonzero(labels == label, as_tuple=False).reshape(-1)
        probabilities = weights[positions].clamp(min=0.0)
        if not torch.isfinite(probabilities).all() or float(
                probabilities.sum()) <= 0.0:
            raise ValueError(
                f"Class {int(label)} has invalid or zero nuisance-donor weight.")
        sampled = torch.multinomial(
            probabilities,
            num_samples=positions.numel(),
            replacement=True,
        )
        donors[positions] = positions[sampled]
    return donors


def make_density_ratio_examples(
    latent: torch.Tensor,
    labels: torch.Tensor,
    nuisance: torch.Tensor,
    weights: torch.Tensor,
    permutation: torch.Tensor | None = None,
    shuffle_mode: str = "weighted_within_class",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return real and conditionally resampled-nuisance critic examples.

    The fake example retains the anchor event's latent, label, and weight; only
    its nuisance is replaced. By default the donor is sampled from the same
    class according to the effective training measure. The old global shuffle
    remains selectable for controlled comparisons.
    """
    batch_size = latent.shape[0]
    if permutation is None:
        permutation = sample_nuisance_donor_indices(
            labels, weights, shuffle_mode=shuffle_mode)
    if permutation.numel() != batch_size:
        raise ValueError("permutation must contain one index per event.")
    permutation = permutation.long().to(nuisance.device)
    if (permutation.min() < 0 or permutation.max() >= batch_size):
        raise ValueError("permutation contains an out-of-range donor index.")
    if (shuffle_mode == "weighted_within_class"
            and not torch.equal(labels, labels[permutation])):
        raise ValueError(
            "weighted_within_class nuisance donors must preserve class labels.")
    combined_latent = torch.cat([latent, latent], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)
    combined_nuisance = torch.cat([nuisance, nuisance[permutation]], dim=0)
    combined_weights = torch.cat([weights, weights], dim=0)
    critic_targets = torch.cat([
        torch.ones(batch_size, dtype=torch.long, device=latent.device),
        torch.zeros(batch_size, dtype=torch.long, device=latent.device),
    ])
    return (
        combined_latent,
        combined_labels,
        combined_nuisance,
        combined_weights,
        critic_targets,
    )


def density_ratio_critic_loss(
    critic,
    latent: torch.Tensor,
    labels: torch.Tensor,
    nuisance: torch.Tensor,
    weights: torch.Tensor,
    permutation: torch.Tensor | None = None,
    shuffle_mode: str = "weighted_within_class",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = make_density_ratio_examples(
        latent, labels, nuisance, weights, permutation=permutation,
        shuffle_mode=shuffle_mode)
    combined_latent, combined_labels, combined_nuisance, combined_weights, targets = examples
    logits = critic(combined_latent, combined_labels, combined_nuisance)
    per_example = F.cross_entropy(logits, targets, reduction="none")
    # Dataset weights are normalized to unit mean over the full split. Keeping
    # that fixed normalization here gives an unbiased stochastic estimate of
    # the global weighted objective. Dividing by each random batch's weight sum
    # is biased when generator weights span many orders of magnitude.
    loss = (per_example * combined_weights.to(per_example.dtype)).mean()
    predictions = logits.argmax(dim=1)
    accuracy = weighted_mean((predictions == targets).float(), combined_weights)
    return loss, accuracy, logits


@torch.no_grad()
def critic_context_only_accuracy(
    critic,
    latent: torch.Tensor,
    labels: torch.Tensor,
    nuisance: torch.Tensor,
    weights: torch.Tensor,
    permutation: torch.Tensor | None = None,
    shuffle_mode: str = "weighted_within_class",
) -> torch.Tensor:
    """Ablate event-level latent information and measure critic shortcuts.

    Every event latent is replaced by its weighted class mean before applying
    the trained critic. Accuracy appreciably above 0.5 then indicates that the
    critic can distinguish real/fake examples from class and nuisance context
    without event-specific representation information.
    """
    examples = make_density_ratio_examples(
        latent, labels, nuisance, weights, permutation=permutation,
        shuffle_mode=shuffle_mode)
    _, combined_labels, combined_nuisance, combined_weights, targets = examples

    context_latent = torch.empty_like(latent)
    for label in labels.unique():
        mask = labels == label
        label_weights = weights[mask].to(latent.dtype).reshape(-1, 1)
        class_mean = (
            latent[mask] * label_weights
        ).sum(dim=0) / label_weights.sum().clamp(min=1e-12)
        context_latent[mask] = class_mean
    context_latent = torch.cat([context_latent, context_latent], dim=0)
    logits = critic(context_latent, combined_labels, combined_nuisance)
    predictions = logits.argmax(dim=1)
    return weighted_mean(
        (predictions == targets).float(), combined_weights)


def engineer_information_penalty(
    critic,
    latent: torch.Tensor,
    labels: torch.Tensor,
    nuisance: torch.Tensor,
) -> torch.Tensor:
    """Per-event log density-ratio penalty evaluated on real tuples.

    Minimizing ``log P(real) - log P(shuffled)`` makes real tuples
    indistinguishable from the shuffled product distribution.
    """
    logits = critic(latent, labels, nuisance)
    log_probabilities = F.log_softmax(logits, dim=1)
    return log_probabilities[:, 1] - log_probabilities[:, 0]


@contextmanager
def frozen_parameters(module):
    states = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield module
    finally:
        for parameter, state in zip(module.parameters(), states):
            parameter.requires_grad_(state)


def per_class_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, Dict[int, float]]:
    predictions = logits.argmax(dim=1)
    values: Dict[int, float] = {}
    for label in sorted(int(value) for value in labels.unique().tolist()):
        mask = labels == label
        values[label] = float((predictions[mask] == labels[mask]).float().mean())
    balanced = sum(values.values()) / max(len(values), 1)
    return balanced, values
