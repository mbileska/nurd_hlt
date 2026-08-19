import sys
import types

import numpy as np
import pytest
import torch

sys.modules.setdefault("wandb", types.SimpleNamespace())

from eval_abcd_nurd import (
    class_assignment_diagnostics,
    checkpoint_reference_splits,
    scan_abcd_grid,
    scan_legacy_same_sample,
    scan_main_legacy_qcd,
)


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
    assert summary["total_grid_points"] == len(percent) ** 2
    assert summary["region_eligible_points"] >= summary["candidate_points"] > 0
    assert summary["tuning_effective_sample_size"] == len(axis1)


def test_scan_explains_an_unattainable_uncertainty_cut():
    rng = np.random.default_rng(7)
    axis1 = rng.normal(size=2000)
    axis2 = rng.normal(size=2000)
    percent = np.linspace(0.50, 0.75, 8)

    best, summary = scan_abcd_grid(
        axis1, axis2, percent,
        min_A=20, min_D=100, min_A_frac=0.02,
        min_region_frac=0.01, max_ratio_unc=1e-6,
        selection_folds=5, selection_seed=42)

    assert "t1" not in best
    assert summary["region_eligible_points"] > 0
    assert summary["uncertainty_eligible_points"] == 0
    assert summary["minimum_ratio_unc"] > 1e-6


def test_legacy_same_sample_scan_keeps_historical_oracle_semantics():
    rng = np.random.default_rng(19)
    axis1 = rng.normal(size=5000)
    axis2 = rng.normal(size=5000)
    weights = np.linspace(0.5, 2.0, len(axis1))
    percent = np.linspace(0.50, 0.80, 10)

    best, summary = scan_legacy_same_sample(
        axis1, axis2, percent, min_A=20, min_D=100, weights=weights)

    assert "t1" in best
    assert best["selection_folds"] == 1
    assert summary["candidate_points"] > 0
    assert summary["tuning_effective_sample_size"] < len(axis1)


def test_main_legacy_qcd_scan_uses_unweighted_numpy_quantiles_and_old_sign():
    axis1 = np.arange(1000, dtype=np.float64)
    axis2 = np.roll(axis1, 137)
    percent = np.array([0.50, 0.60, 0.70])

    best, summary = scan_main_legacy_qcd(
        axis1, axis2, percent, min_A=10, min_D=10)

    assert best["t1"] == np.quantile(axis1, best["p1"])
    assert best["t2"] == np.quantile(axis2, best["p2"])
    assert np.isclose(
        best["nonclosure"],
        (best["A"] - best["A_hat"]) / best["A_hat"],
    )
    assert 0 < summary["n_points"] <= len(percent) ** 2


def test_class_assignment_uses_generator_weighted_measure():
    products = {
        "reference_labels": np.array([0, 1]),
        "classifier_route_index": np.array([0, 1, 1]),
        "gaussian_route_index": np.array([0, 1, 1]),
        "typicality_route_index": np.array([0, 1, 1]),
    }
    result = class_assignment_diagnostics(
        np.array([0, 0, 1]), products,
        weights=np.array([1.0, 9.0, 1.0]))

    classifier = result["classifier"]
    assert np.isclose(classifier["accuracy"], 2.0 / 11.0)
    assert np.isclose(classifier["balanced_accuracy"], 0.55)
    assert classifier["confusion_matrix"] == [[1, 1], [0, 1]]
    assert classifier["weighted_confusion_matrix"] == [[1.0, 9.0], [0.0, 1.0]]


def test_heldout_reference_roles_must_match_checkpoint_provenance():
    signature = {"n_events": 10, "label_sha256": "sample"}
    checkpoint = {
        "checkpoint_contract_version": 2,
        "data_split_indices": {
            "reference_fit": torch.tensor([0, 1, 2, 3, 4, 5]),
            "checkpoint_validation": torch.tensor([6, 7]),
            "threshold_tune": torch.tensor([8, 9]),
        },
        "data_provenance": {
            "sample": signature,
            "generator_weights": {"sha256": "weights"},
        },
    }
    metadata = {
        "sample_signature": signature,
        "sha256": "weights",
    }

    fit, checkpoint_val, tune = checkpoint_reference_splits(
        checkpoint, np.zeros(10, dtype=int), metadata)
    assert fit.tolist() == [0, 1, 2, 3, 4, 5]
    assert checkpoint_val.tolist() == [6, 7]
    assert tune.tolist() == [8, 9]

    with pytest.raises(ValueError, match="generator-weight digest"):
        checkpoint_reference_splits(
            checkpoint, np.zeros(10, dtype=int),
            {**metadata, "sha256": "wrong"})
