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


def make_density_ratio_examples(
    latent: torch.Tensor,
    labels: torch.Tensor,
    nuisance: torch.Tensor,
    weights: torch.Tensor,
    permutation: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return real and shuffled-nuisance critic examples.

    As in the engineer reference, the shuffled example retains the original
    event label and event weight; only the nuisance value is permuted.
    """
    batch_size = latent.shape[0]
    if permutation is None:
        permutation = torch.randperm(batch_size, device=nuisance.device)
    if permutation.numel() != batch_size:
        raise ValueError("permutation must contain one index per event.")
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = make_density_ratio_examples(
        latent, labels, nuisance, weights, permutation=permutation)
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
