import numpy as np
import pytest
import torch

from scripts.audit_qcd_validation_hardness import (
    ht10_from_objects,
    ht_slice_edges,
    weighted_pearson,
    weighted_quantile,
)


def test_ht10_uses_only_first_ten_jets_above_threshold():
    objects = torch.zeros((2, 23, 4))
    objects[0, :3, 0] = torch.tensor([100., 29., 30.])
    objects[0, 10, 0] = 999.  # Muon, never included in jet HT.
    objects[1, :2, 0] = torch.tensor([31., 32.])
    np.testing.assert_array_equal(
        ht10_from_objects(objects, np.array([0, 1])), [130., 63.])


def test_weighted_thirds_fall_back_when_zero_ht_has_large_mass():
    ht = np.array([0., 0., 0., 20., 30., 40., 50., 60., 70.])
    weights = np.array([10., 10., 10., 1., 1., 1., 1., 1., 1.])
    edges, method = ht_slice_edges(ht, weights)
    assert np.all(np.diff(edges) > 0)
    assert method != "physics_weighted_thirds"


def test_weighted_pearson_and_quantile():
    x = np.array([1., 2., 3.])
    np.testing.assert_array_equal(weighted_quantile(x, [0.25, 0.75],
                                                     np.ones(3)), [1., 3.])
    assert weighted_pearson(x, x * 2, np.ones(3)) == pytest.approx(1.)
