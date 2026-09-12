import sys
import types

import numpy as np
import pytest
import torch


sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from scripts.scan_validation_checkpoints import (  # noqa: E402
    discover_checkpoints,
    validation_closure_metrics,
)


def test_discover_checkpoints_includes_selected_nonperiodic_epoch(tmp_path):
    torch.save({"epoch": 5}, tmp_path / "checkpoint_epoch_005.pth.tar")
    torch.save({"epoch": 8}, tmp_path / "checkpoint_main.pth.tar")
    torch.save({"epoch": 10}, tmp_path / "checkpoint_final.pth.tar")

    candidates = discover_checkpoints(tmp_path, [5, 8, 10])

    assert [(epoch, role) for epoch, _path, role in candidates] == [
        (5, "periodic"), (8, "selected"), (10, "final")]
    with pytest.raises(FileNotFoundError, match="requested epoch"):
        discover_checkpoints(tmp_path, [7])


def test_validation_closure_metrics_finds_exact_factorization():
    # Equal population in every quadrant gives exact ABCD closure.
    axis_1 = np.tile(np.asarray([0.0, 0.0, 1.0, 1.0]), 10)
    axis_2 = np.tile(np.asarray([0.0, 1.0, 0.0, 1.0]), 10)
    weights = np.ones_like(axis_1)
    result = validation_closure_metrics(
        axis_1,
        axis_2,
        weights,
        percentiles=[0.5],
        minimums={"A": 1, "B": 1, "C": 1, "D": 1},
    )

    assert result["grid_points"] == 1
    assert result["grid_median_absolute_nonclosure"] == 0.0
    assert result["curve"][0]["ratio"] == 1.0


def test_validation_closure_metrics_rejects_empty_sidebands():
    axis_1 = np.tile(np.asarray([0.0, 0.0, 1.0, 1.0]), 10)
    axis_2 = axis_1.copy()
    weights = np.ones_like(axis_1)

    try:
        validation_closure_metrics(
            axis_1,
            axis_2,
            weights,
            percentiles=[0.5],
            minimums={"A": 1, "B": 1, "C": 1, "D": 1},
        )
    except RuntimeError as error:
        assert "No statistically valid" in str(error)
    else:
        raise AssertionError("Correlated scores with empty sidebands should be rejected.")
