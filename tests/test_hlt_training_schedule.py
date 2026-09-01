import math

import pytest
import torch

from train_hlt import (
    EpochMetrics,
    cosine_lr_multiplier,
    effective_information_weight,
    information_schedule_end_epoch,
)


def test_immediate_information_schedule_preserves_previous_behavior():
    assert information_schedule_end_epoch(0, 0) == 1
    assert effective_information_weight(1, 1.0, 0, 0) == pytest.approx(1.0)
    assert effective_information_weight(20, 1.0, 0, 0) == pytest.approx(1.0)


def test_five_epoch_warmup_and_ten_epoch_cosine_ramp():
    assert information_schedule_end_epoch(5, 10) == 15
    for epoch in range(1, 6):
        assert effective_information_weight(epoch, 1.0, 5, 10) == 0.0
    assert effective_information_weight(6, 1.0, 5, 10) == 0.0
    expected_epoch_10 = 0.5 * (1.0 - math.cos(math.pi * 4.0 / 9.0))
    assert effective_information_weight(10, 1.0, 5, 10) == pytest.approx(
        expected_epoch_10)
    assert effective_information_weight(15, 1.0, 5, 10) == pytest.approx(1.0)
    assert effective_information_weight(16, 1.0, 5, 10) == pytest.approx(1.0)


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
