import math

import pytest
import torch

from utils.hlt_weights import (
    _assign_strata,
    apply_joint_balance,
    effective_mass_by_class,
    fit_joint_balance,
    load_generator_weights,
)


def test_joint_balance_equalizes_classes_and_occupied_strata():
    labels = torch.tensor([0] * 80 + [1] * 240 + [2] * 40 + [3] * 120)
    nuisance = torch.cat([
        torch.linspace(0.0, 1.0, 80),
        torch.linspace(0.1, 1.2, 240),
        torch.linspace(0.2, 0.9, 40),
        torch.linspace(0.0, 1.1, 120),
    ])
    base = torch.ones(labels.numel(), dtype=torch.float64)
    base[labels == 1] = torch.linspace(
        1.0, 100.0, int((labels == 1).sum()), dtype=torch.float64)

    weights, spec = fit_joint_balance(labels, nuisance, base, n_strata=8)
    class_mass = effective_mass_by_class(labels, weights)
    assert set(class_mass) == {0, 1, 2, 3}
    assert all(value == pytest.approx(0.25, abs=1e-6)
               for value in class_mass.values())

    strata = _assign_strata(nuisance, spec["edges"])
    for label in labels.unique().tolist():
        cell_masses = []
        for stratum in strata[labels == label].unique().tolist():
            mask = (labels == label) & (strata == stratum)
            cell_masses.append(float(weights[mask].sum()))
        assert max(cell_masses) == pytest.approx(min(cell_masses), rel=1e-5)


def test_validation_uses_training_joint_factors_without_refitting():
    train_labels = torch.tensor([0] * 100 + [1] * 100)
    train_nuisance = torch.cat([
        torch.linspace(0.0, 1.0, 100),
        torch.linspace(0.0, 1.0, 100),
    ])
    train_base = torch.ones(200)
    _, spec = fit_joint_balance(
        train_labels, train_nuisance, train_base, n_strata=5)

    val_labels = torch.tensor([0] * 20 + [1] * 20)
    val_nuisance = torch.cat([
        torch.linspace(-0.1, 1.1, 20),
        torch.linspace(-0.1, 1.1, 20),
    ])
    val_weights = apply_joint_balance(
        val_labels, val_nuisance, torch.ones(40), spec)
    assert torch.isfinite(val_weights).all()
    assert effective_mass_by_class(val_labels, val_weights) == pytest.approx(
        {0: 0.5, 1: 0.5}, abs=1e-6)


def test_generator_weights_apply_only_to_qcd_and_validate_length(tmp_path):
    labels = torch.tensor([0, 1, 1, 2, 3])
    path = tmp_path / "weights.pt"
    torch.save(torch.tensor([9.0, 2.0, 3.0, 8.0, 7.0]), path)
    weights, metadata = load_generator_weights(str(path), labels, qcd_label=1)
    assert weights.tolist() == [1.0, 2.0, 3.0, 1.0, 1.0]
    assert metadata["qcd_sum"] == pytest.approx(5.0)

    short_path = tmp_path / "short.pt"
    torch.save(torch.ones(4), short_path)
    with pytest.raises(ValueError, match="does not match"):
        load_generator_weights(str(short_path), labels, qcd_label=1)


def test_generator_weight_event_ids_are_verified_when_available(tmp_path):
    labels = torch.tensor([0, 1, 1])
    sample = {
        "label": labels,
        "event_id": torch.tensor([101, 102, 103]),
    }
    path = tmp_path / "identified_weights.pt"
    torch.save({
        "weights": torch.tensor([1.0, 2.0, 3.0]),
        "event_id": torch.tensor([101, 102, 103]),
    }, path)
    _, metadata = load_generator_weights(
        str(path), labels, qcd_label=1, sample=sample)
    assert metadata["alignment_verified"] is True

    torch.save({
        "weights": torch.tensor([1.0, 2.0, 3.0]),
        "event_id": torch.tensor([101, 103, 102]),
    }, path)
    with pytest.raises(ValueError, match="event IDs"):
        load_generator_weights(str(path), labels, qcd_label=1, sample=sample)


def test_large_qcd_generator_normalization_cannot_dominate_class_mass():
    labels = torch.tensor([0] * 100 + [1] * 300 + [2] * 50 + [3] * 100)
    nuisance = torch.linspace(0.0, 1.0, labels.numel())
    base = torch.ones(labels.numel())
    base[labels == 1] = 1000.0
    weights, _ = fit_joint_balance(labels, nuisance, base, n_strata=5)
    masses = effective_mass_by_class(labels, weights)
    assert all(math.isclose(value, 0.25, abs_tol=1e-6)
               for value in masses.values())
