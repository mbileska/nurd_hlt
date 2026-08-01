import numpy as np
import pytest
import torch

from eval_abcd_nurd import abcd_record_at_thresholds
from utils.event_weights import load_event_weights


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
