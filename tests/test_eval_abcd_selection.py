import sys
import types

import numpy as np

sys.modules.setdefault("wandb", types.SimpleNamespace())

from eval_abcd_nurd import scan_abcd_grid


def test_robust_abcd_selection_is_deterministic_and_fully_cross_checked():
    rng = np.random.default_rng(12)
    axis1 = rng.normal(size=8000)
    axis2 = rng.normal(size=8000)
    percent = np.linspace(0.50, 0.75, 12)
    kwargs = {
        "min_A": 50,
        "min_D": 500,
        "min_A_frac": 0.02,
        "min_region_frac": 0.01,
        "max_ratio_unc": 0.10,
        "selection_folds": 5,
        "selection_seed": 42,
        "selection_stat_weight": 0.5,
        "selection_neighbor_weight": 1.0,
    }

    first, summary = scan_abcd_grid(axis1, axis2, percent, **kwargs)
    second, _ = scan_abcd_grid(axis1, axis2, percent, **kwargs)

    assert first == second
    assert first["selection_folds"] == 5
    assert first["ratio_unc"] <= 0.10
    assert min(first[key] for key in ("A", "B", "C", "D")) >= 80
    assert summary["n_points"] > 0
