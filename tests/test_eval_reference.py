import numpy as np
import sys
import types
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))
from eval_abcd_nurd import _fit_class_transform, checkpoint_reference_indices


def test_weighted_md_reference_uses_supplied_physics_measure():
    embeddings = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0]])
    mask = np.asarray([True, True, False])
    weights = np.asarray([1.0, 3.0, 100.0])
    mean, transform = _fit_class_transform(
        embeddings, mask, n_pca=None, class_name="QCD", weights=weights)
    assert np.allclose(mean, [1.5, 0.0])
    assert transform.shape == (2, 2)
    assert np.isfinite(transform).all()


def test_checkpoint_indices_are_explicitly_moved_to_cpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = {
        "preprocessing": {
            "data_signature": {"n_events": 5, "token": "sample"},
            "weighting": {"generator": {
                "effective_physics_weight_sha256": "weights",
            }},
            "split": {
                "train_indices": torch.tensor([0, 2, 4], device=device),
                "validation_indices": torch.tensor([1, 3], device=device),
            },
        },
    }
    fit, selection = checkpoint_reference_indices(
        checkpoint,
        {"n_events": 5, "token": "sample"},
        {"effective_physics_weight_sha256": "weights"},
    )
    assert fit.dtype == np.int64
    assert selection.dtype == np.int64
    assert fit.tolist() == [0, 2, 4]
    assert selection.tolist() == [1, 3]
