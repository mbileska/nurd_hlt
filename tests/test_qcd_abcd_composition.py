import numpy as np
import pytest

from scripts.audit_qcd_abcd_composition import (
    region_codes,
    region_statistics,
    verify_report,
)


def test_region_composition_reproduces_weighted_global_yields():
    ae = np.array([2., 2., 0., 0.])
    md = np.array([2., 0., 2., 0.])
    weights = np.array([10., 1., 2., 3.])
    categories = np.array([0, 5, 0, 5])
    codes = region_codes(ae, md, 1., 1.)
    np.testing.assert_array_equal(codes, [0, 1, 2, 3])
    rows = region_statistics(codes, weights, categories)
    report = {
        region: {"raw_count": 1, "yield": yield_}
        for region, yield_ in zip("ABCD", (10., 1., 2., 3.))
    }
    verify_report(rows, report)
    assert next(r for r in rows if r["region"] == "A" and r["category"] == 0)[
        "fraction_of_region_weight"] == 1.0
    assert next(r for r in rows if r["region"] == "D" and r["category"] == 5)[
        "effective_count"] == 1.0


def test_region_report_mismatch_fails():
    rows = region_statistics(
        np.array([0, 1, 2, 3]), np.ones(4), np.zeros(4, dtype=int))
    report = {region: {"raw_count": 1, "yield": 1.} for region in "ABCD"}
    report["A"]["yield"] = 2.
    with pytest.raises(ValueError, match="weighted yield mismatch"):
        verify_report(rows, report)
