import numpy as np
import pytest

from scripts.convert_background_nanoaod import (
    labels_from_samples,
    split_weight_payload,
    stratified_test_mask,
)


def test_background_label_contract():
    assert labels_from_samples([0, 1, 2, 3]).tolist() == [1, 3, 0, 2]
    with pytest.raises(ValueError, match="Unsupported"):
        labels_from_samples([4])


def test_fine_category_split_is_deterministic_and_complete():
    categories = np.repeat([0, 1, 8, 27], [20, 30, 40, 50])
    first = stratified_test_mask(categories, test_fraction=0.2, seed=42)
    second = stratified_test_mask(categories, test_fraction=0.2, seed=42)
    assert np.array_equal(first, second)
    assert first.sum() == 28
    for category in np.unique(categories):
        assert (categories[first] == category).any()
        assert (categories[~first] == category).any()


def test_weight_split_slices_all_event_fields_and_keeps_metadata():
    event_ids = np.arange(24, dtype=np.int64).reshape(6, 4)
    payload = {
        "weights": np.linspace(1, 2, 6),
        "event_id": event_ids,
        "categories": np.arange(6),
        "weight_scope": np.asarray("all_events"),
        "luminosity_pb_inverse": np.asarray(1.0),
    }
    mask = np.asarray([True, False, True, False, False, True])
    selected = split_weight_payload(payload, mask, n_events=6)
    assert np.array_equal(selected["event_id"], event_ids[mask])
    assert selected["weights"].shape == (3,)
    assert selected["luminosity_pb_inverse"].shape == ()
    assert selected["weight_scope"].item() == "all_events"
