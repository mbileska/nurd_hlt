import numpy as np
import pytest
import torch

from eval_abcd_nurd import _weighted_midrank, abcd_record_at_thresholds
from utils.event_weights import (
    load_event_weights,
    validate_shared_data_contract,
    weighted_stratified_split,
)


def test_event_weight_loader_validates_alignment_and_reports_ess(tmp_path):
    sample = {
        "label": torch.tensor([0, 1, 1]),
        "eventid": torch.tensor([10, 11, 12]),
    }
    path = tmp_path / "weights.pt"
    torch.save({
        "gen_weight": torch.tensor([1.0, 2.0, 3.0]),
        "eventid": sample["eventid"].clone(),
    }, path)

    weights, metadata = load_event_weights(path, sample)

    assert weights.tolist() == [1.0, 2.0, 3.0]
    assert np.isclose(metadata["effective_events"], 36.0 / 14.0)
    assert metadata["alignment_verified"] is True
    assert len(metadata["sha256"]) == 64


def test_event_weight_loader_rejects_negative_weights(tmp_path):
    sample = {"label": torch.tensor([0, 1])}
    path = tmp_path / "weights.pt"
    torch.save(torch.tensor([1.0, -1.0]), path)

    with pytest.raises(ValueError, match="negative"):
        load_event_weights(path, sample)


def test_weighted_abcd_uses_sumw2_uncertainty():
    axis1 = np.asarray([1.0, 1.0, 0.0, 0.0])
    axis2 = np.asarray([1.0, 0.0, 1.0, 0.0])
    weights = np.asarray([2.0, 3.0, 4.0, 5.0])

    record = abcd_record_at_thresholds(
        axis1, axis2, 0.5, 0.5, weights=weights)

    assert record["A"] == 2.0
    assert record["B"] == 3.0
    assert record["C"] == 4.0
    assert record["D"] == 5.0
    assert record["A_sumw2"] == 4.0
    assert record["A_n"] == 1


def test_physical_stratified_roles_are_disjoint_and_mass_balanced():
    labels = torch.tensor([0] * 200 + [1] * 200)
    weights = torch.ones(400)
    weights[[0, 20, 200, 220]] = torch.tensor([80.0, 60.0, 90.0, 70.0])

    fit, checkpoint, tune = weighted_stratified_split(
        labels, weights, (0.9, 0.05, 0.05), seed=11)

    combined = np.concatenate([fit, checkpoint, tune])
    assert len(np.unique(combined)) == len(labels)
    assert set(fit).isdisjoint(checkpoint)
    assert set(fit).isdisjoint(tune)
    assert set(checkpoint).isdisjoint(tune)
    for label in (0, 1):
        total = float(weights[labels == label].sum())
        checkpoint_mass = float(weights[checkpoint][labels[checkpoint] == label].sum())
        tune_mass = float(weights[tune][labels[tune] == label].sum())
        # A single event is indivisible; each role is balanced to within the
        # largest weight in that class.
        largest = float(weights[labels == label].max())
        assert abs(checkpoint_mass - 0.05 * total) <= largest
        assert abs(tune_mass - 0.05 * total) <= largest


def test_weighted_midrank_uses_physical_mass_and_handles_ties():
    values = np.asarray([0.0, 0.0, 1.0])
    weights = np.asarray([2.0, 4.0, 4.0])

    ranks = _weighted_midrank(values, weights)

    assert np.allclose(ranks, [0.3, 0.3, 0.8])


def test_shared_ae_nurd_contract_rejects_any_split_mismatch():
    provenance = {
        "sample": {"n_events": 4, "label_sha256": "sample"},
        "generator_weights": {"sha256": "weights"},
    }
    splits = {
        "reference_fit": torch.tensor([0, 1]),
        "checkpoint_validation": torch.tensor([2]),
        "threshold_tune": torch.tensor([3]),
    }
    checkpoint = {
        "data_signature": provenance["sample"],
        "gen_weight_metadata": {"sha256": "weights"},
        "data_split_indices": {key: value.clone() for key, value in splits.items()},
        "code_commit": "abc123",
    }

    validate_shared_data_contract(checkpoint, provenance, splits, "abc123")
    checkpoint["data_split_indices"]["threshold_tune"] = torch.tensor([2])
    with pytest.raises(ValueError, match="threshold_tune"):
        validate_shared_data_contract(checkpoint, provenance, splits, "abc123")
