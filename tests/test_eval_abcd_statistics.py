import sys
import types

import numpy as np
import pytest


sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from eval_abcd_nurd import (  # noqa: E402
    abcd_region_statistics_at_thresholds,
    closure_ratio_and_uncertainty,
    statistically_valid_regions,
)


def test_unweighted_effective_counts_equal_raw_counts():
    axis_1 = np.asarray([1.0, 1.0, 0.0, 0.0])
    axis_2 = np.asarray([1.0, 0.0, 1.0, 0.0])
    statistics = abcd_region_statistics_at_thresholds(
        axis_1, axis_2, 0.5, 0.5)

    for region in ("A", "B", "C", "D"):
        assert statistics[region]["yield"] == 1.0
        assert statistics[region]["sumw2"] == 1.0
        assert statistics[region]["raw_count"] == 1
        assert statistics[region]["effective_count"] == 1.0


def test_weighted_closure_uncertainty_uses_sumw2():
    axis_1 = np.asarray([1.0, 1.0, 0.0, 0.0])
    axis_2 = np.asarray([1.0, 0.0, 1.0, 0.0])
    weights = np.asarray([2.0, 1.0, 3.0, 4.0])
    statistics = abcd_region_statistics_at_thresholds(
        axis_1, axis_2, 0.5, 0.5, weights=weights)

    ratio, uncertainty = closure_ratio_and_uncertainty(statistics)
    assert ratio == pytest.approx(3.0 / 8.0)
    # Each region contains one weighted event, so every relative-variance
    # contribution sumw2/sumw^2 is one.
    assert uncertainty == pytest.approx(2.0 * ratio)


def test_minimum_statistics_use_effective_not_weighted_yield():
    axis_1 = np.asarray([1.0, 1.0, 0.0, 0.0])
    axis_2 = np.asarray([1.0, 0.0, 1.0, 0.0])
    weights = np.full(4, 1000.0)
    statistics = abcd_region_statistics_at_thresholds(
        axis_1, axis_2, 0.5, 0.5, weights=weights)

    assert not statistically_valid_regions(
        statistics, {"A": 2, "B": 2, "C": 2, "D": 2})
    assert statistically_valid_regions(
        statistics, {"A": 1, "B": 1, "C": 1, "D": 1})
