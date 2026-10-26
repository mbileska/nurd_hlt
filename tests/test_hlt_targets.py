import pytest
import torch

from utils.hlt_targets import (
    DECAY8_NAMES,
    checkpoint_target_labels,
    derive_target_labels,
)


def _decay_sample(repeats=2):
    categories = torch.tensor([15, 0, 27, 6, 9, 12, 18, 21]).repeat(repeats)
    parents = torch.tensor([0, 1, 2, 3, 3, 3, 0, 0]).repeat(repeats)
    return {"label": parents, "background_category": categories}


def test_parent4_is_unchanged_without_fine_categories():
    labels = torch.tensor([0, 1, 2, 3, 1])
    derived, metadata = derive_target_labels({"label": labels}, "parent4")

    assert torch.equal(derived, labels)
    assert metadata["qcd_label"] == 1
    assert metadata["class_names"][3] == "W+jets"


def test_decay8_splits_w_and_z_but_not_ht_bins():
    sample = _decay_sample()
    labels, metadata = derive_target_labels(sample, "decay8")

    assert labels.tolist() == list(range(8)) * 2
    assert metadata["class_names"] == DECAY8_NAMES
    assert metadata["qcd_label"] == 1


def test_decay8_rejects_missing_or_inconsistent_metadata():
    with pytest.raises(KeyError, match="background_category"):
        derive_target_labels({"label": torch.tensor([0, 1])}, "decay8")

    sample = _decay_sample(repeats=1)
    sample["label"][0] = 3
    with pytest.raises(ValueError, match="disagree"):
        derive_target_labels(sample, "decay8")


def test_decay8_rejects_unconfigured_categories():
    with pytest.raises(ValueError, match="cannot map"):
        derive_target_labels({
            "label": torch.tensor([0]),
            "background_category": torch.tensor([24]),
        }, "decay8")


def test_checkpoint_taxonomy_is_reproduced_for_evaluation():
    sample = _decay_sample(repeats=1)
    labels = checkpoint_target_labels(sample, {
        "target_scheme": "decay8",
        "label_map": {value: value for value in range(8)},
    })
    assert labels.tolist() == list(range(8))
