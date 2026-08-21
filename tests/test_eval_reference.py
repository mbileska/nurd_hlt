import numpy as np
import sys
import types

sys.modules.setdefault("wandb", types.ModuleType("wandb"))
from eval_abcd_nurd import _fit_class_transform


def test_weighted_md_reference_uses_supplied_physics_measure():
    embeddings = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0]])
    mask = np.asarray([True, True, False])
    weights = np.asarray([1.0, 3.0, 100.0])
    mean, transform = _fit_class_transform(
        embeddings, mask, n_pca=None, class_name="QCD", weights=weights)
    assert np.allclose(mean, [1.5, 0.0])
    assert transform.shape == (2, 2)
    assert np.isfinite(transform).all()
