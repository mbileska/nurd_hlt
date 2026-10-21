import numpy as np
import pytest
import torch

from scripts.create_diagnostic_subset import (
    extract_weight_vector,
    save_weight_payload,
    stratified_subset_indices,
    subset_event_mapping,
    subset_weight_payload,
)


def test_stratified_diagnostic_subset_is_deterministic_and_keeps_classes():
    labels = torch.tensor([0] * 50 + [1] * 30 + [2] * 15 + [3] * 5)

    first = stratified_subset_indices(labels, maximum_events=40, seed=17)
    second = stratified_subset_indices(labels, maximum_events=40, seed=17)

    assert torch.equal(first, second)
    assert first.numel() == 40
    assert torch.bincount(labels[first], minlength=4).tolist() == [20, 12, 6, 2]


def test_diagnostic_subset_preserves_event_and_weight_row_alignment():
    labels = torch.arange(60) % 4
    event_ids = torch.arange(60) + 1000
    sample = {
        "pf": event_ids[:, None, None].expand(-1, 2, 7).clone(),
        "obj": event_ids[:, None, None].expand(-1, 2, 4).clone(),
        "label": labels,
        "event_id": event_ids,
        "metadata": "untouched",
    }
    weights = event_ids.float() / 10.0
    selected = stratified_subset_indices(labels, maximum_events=24, seed=5)

    subset = subset_event_mapping(sample, selected)

    assert torch.equal(subset["event_id"], event_ids[selected])
    assert torch.equal(subset["pf"][:, 0, 0], event_ids[selected])
    assert torch.equal(weights[selected], event_ids[selected].float() / 10.0)
    assert subset["metadata"] == "untouched"


def test_extract_weight_vector_accepts_standard_mapping_and_rejects_ambiguity():
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(
        extract_weight_vector({"weights": expected, "event_id": torch.arange(3)}),
        expected)
    assert torch.equal(extract_weight_vector(np.asarray([1.0, 2.0])),
                       torch.tensor([1.0, 2.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="ambiguous"):
        extract_weight_vector({"first": expected, "second": expected})


def test_npz_diagnostic_weights_preserve_scope_and_event_ids(tmp_path):
    event_ids = np.arange(32, dtype=np.int64).reshape(8, 4)
    payload = {
        "weights": np.linspace(1.0, 2.0, 8),
        "event_id": event_ids,
        "weight_scope": np.asarray("all_events"),
    }
    indices = torch.tensor([0, 3, 7])
    selected = subset_weight_payload(payload, indices, n_events=8)
    destination = tmp_path / "subset.npz"
    save_weight_payload(selected, destination)
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["weight_scope"].item() == "all_events"
        assert np.array_equal(archive["event_id"], event_ids[[0, 3, 7]])
