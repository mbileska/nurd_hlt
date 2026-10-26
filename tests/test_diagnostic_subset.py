import numpy as np
import pytest
import torch

from scripts.create_diagnostic_subset import (
    extract_weight_vector,
    main,
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


def test_decay8_subset_is_stratified_by_derived_targets(tmp_path):
    categories = torch.tensor([15, 0, 27, 6, 9, 12, 18, 21]).repeat(10)
    parents = torch.tensor([0, 1, 2, 3, 3, 3, 0, 0]).repeat(10)
    n_events = parents.numel()
    event_ids = torch.arange(n_events * 4).reshape(n_events, 4)
    sample = {
        "pf": torch.ones(n_events, 2, 7),
        "obj": torch.ones(n_events, 2, 4),
        "label": parents,
        "background_category": categories,
        "event_id": event_ids,
    }
    data_path = tmp_path / "sample.pt"
    weights_path = tmp_path / "weights.npz"
    output_data = tmp_path / "subset.pt"
    output_weights = tmp_path / "subset_weights.npz"
    metadata_path = tmp_path / "metadata.json"
    torch.save(sample, data_path)
    np.savez_compressed(
        weights_path,
        weights=np.ones(n_events),
        event_id=event_ids.numpy(),
        weight_scope=np.asarray("all_events"),
    )

    main([
        "--data", str(data_path),
        "--weights", str(weights_path),
        "--output-data", str(output_data),
        "--output-weights", str(output_weights),
        "--metadata", str(metadata_path),
        "--max-events", "40",
        "--target-scheme", "decay8",
        "--seed", "9",
    ])

    import json
    metadata = json.loads(metadata_path.read_text())
    assert metadata["target_scheme"] == "decay8"
    assert metadata["class_counts_selected"] == {
        str(label): 5 for label in range(8)
    }
    selected = torch.load(output_data, map_location="cpu", weights_only=False)
    assert "background_category" in selected
    assert selected["label"].numel() == 40
