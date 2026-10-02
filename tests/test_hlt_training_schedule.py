import math

import pytest
import torch

from train_hlt import (
    EpochMetrics,
    checkpoint_selection_start_epoch,
    cosine_lr_multiplier,
    nurd_active_for_epoch,
)


def test_nurd_disabled_keeps_critic_off_and_allows_immediate_selection():
    assert checkpoint_selection_start_epoch(0, 20) == 1
    assert nurd_active_for_epoch(1, 0, 0) is False
    assert nurd_active_for_epoch(40, 0, 20) is False


def test_zero_start_activates_nurd_immediately_without_ramp():
    assert checkpoint_selection_start_epoch(1, 0) == 1
    assert nurd_active_for_epoch(1, 1, 0) is True
    assert nurd_active_for_epoch(40, 1, 0) is True


def test_positive_start_switches_nurd_on_sharply():
    assert checkpoint_selection_start_epoch(1, 6) == 6
    for epoch in range(1, 6):
        assert nurd_active_for_epoch(epoch, 1, 6) is False
    assert nurd_active_for_epoch(6, 1, 6) is True
    assert nurd_active_for_epoch(7, 1, 6) is True


def test_nurd_schedule_rejects_invalid_values():
    with pytest.raises(ValueError, match="one-based"):
        nurd_active_for_epoch(0, 1, 0)
    with pytest.raises(ValueError, match="zero or one"):
        checkpoint_selection_start_epoch(2, 0)
    with pytest.raises(ValueError, match="non-negative"):
        checkpoint_selection_start_epoch(1, -1)


def test_accuracy_metrics_remain_global_weighted_and_class_balanced():
    metrics = EpochMetrics(num_classes=2)
    labels = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([
        [2.0, 0.0],  # correct class 0, weight 1
        [0.0, 2.0],  # incorrect class 0, weight 3
        [0.0, 2.0],  # correct class 1, weight 2
        [0.0, 2.0],  # correct class 1, weight 4
    ])
    weights = torch.tensor([1.0, 3.0, 2.0, 4.0])
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    zero = torch.tensor(0.0)
    metrics.update_main(logits, labels, weights, ce, zero, zero, zero)
    summary = metrics.summary()

    assert summary["per_class_accuracy"] == {0: 0.5, 1: 1.0}
    assert summary["balanced_accuracy"] == pytest.approx(0.75)
    assert summary["weighted_accuracy"] == pytest.approx(0.7)


def test_cosine_lr_reaches_epoch_40_minimum_and_stays_there():
    assert cosine_lr_multiplier(0, 40) == pytest.approx(1.0)
    assert cosine_lr_multiplier(20, 40) == pytest.approx(0.5005)
    assert cosine_lr_multiplier(40, 40) == pytest.approx(0.001)
    assert cosine_lr_multiplier(80, 40) == pytest.approx(0.001)
