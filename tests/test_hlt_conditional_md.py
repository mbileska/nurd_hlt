import numpy as np
import pytest
import sys
import types

from utils.hlt_conditional_md import ConditionalMDCDF

sys.modules.setdefault("wandb", types.ModuleType("wandb"))
from scripts.scan_validation_checkpoints import validation_closure_metrics  # noqa: E402
from scripts.run_conditional_md_calibration import main  # noqa: E402


def test_weighted_conditional_cdf_reduces_independent_validation_dependence(tmp_path):
    rng = np.random.default_rng(17)
    n_train, n_val = 60_000, 30_000
    train_log_ae = rng.normal(size=n_train)
    val_log_ae = rng.normal(size=n_val)
    # A smooth common-hardness dependence, with heterogeneous physics weights.
    train_md = np.exp(0.9 * train_log_ae + rng.normal(size=n_train))
    val_md = np.exp(0.9 * val_log_ae + rng.normal(size=n_val))
    train_weights = np.exp(0.25 * train_log_ae)
    val_weights = np.exp(0.25 * val_log_ae)

    model = ConditionalMDCDF.fit(
        10 ** train_log_ae, train_md, train_weights,
        n_ae_bins=20, min_ess=200)
    calibrated = model.transform(10 ** val_log_ae, val_md)

    # The fit is based entirely on training rows and is reusable as an artifact.
    path = tmp_path / "conditional_md.npz"
    model.save(path)
    np.testing.assert_allclose(
        ConditionalMDCDF.load(path).transform(10 ** val_log_ae, val_md),
        calibrated)
    assert len(model.fit_counts) == 20
    assert np.min(model.fit_effective_counts) > 200
    assert np.all((calibrated >= 0) & (calibrated <= 1))

    # Held-out, physics-weighted MD percentiles should be nearly uniform in
    # every coarse AE bin even though raw MD rises strongly with AE.
    raw_bin_means = []
    calibrated_bin_means = []
    for bin_index in range(20):
        mask = model.bin_indices(10 ** val_log_ae) == bin_index
        raw_bin_means.append(np.average(val_md[mask], weights=val_weights[mask]))
        calibrated_bin_means.append(np.average(
            calibrated[mask], weights=val_weights[mask]))
    assert max(raw_bin_means) > 5 * min(raw_bin_means)
    assert max(abs(np.asarray(calibrated_bin_means) - 0.5)) < 0.04


def test_conditional_cdf_rejects_sparse_or_invalid_fit():
    ae = np.linspace(0.1, 1.0, 100)
    md = np.linspace(1.0, 10.0, 100)
    with pytest.raises(ValueError, match="ESS"):
        ConditionalMDCDF.fit(ae, md, np.ones(100), n_ae_bins=20, min_ess=10)
    with pytest.raises(ValueError, match="strictly positive"):
        ConditionalMDCDF.fit(np.zeros(100), md, np.ones(100))


def test_weighted_validation_abcd_improves_for_synthetic_common_hardness():
    rng = np.random.default_rng(17)
    train_log_ae = rng.normal(size=30_000)
    val_log_ae = rng.normal(size=12_000)
    train_ae = np.exp(train_log_ae)
    val_ae = np.exp(val_log_ae)
    train_md = np.exp(1.1 * train_log_ae + rng.normal(size=len(train_ae)))
    val_md = np.exp(1.1 * val_log_ae + rng.normal(size=len(val_ae)))
    train_weights = np.exp(0.2 * train_log_ae)
    val_weights = np.exp(0.2 * val_log_ae)
    calibrated = ConditionalMDCDF.fit(
        train_ae, train_md, train_weights, n_ae_bins=20,
        min_ess=100).transform(val_ae, val_md)
    settings = {
        "percentiles": np.linspace(0.5, 0.8, 4),
        "minimums": {key: 10 for key in "ABCD"},
    }
    raw = validation_closure_metrics(val_ae, val_md, val_weights, **settings)
    corrected = validation_closure_metrics(
        val_ae, calibrated, val_weights, **settings)
    assert raw["grid_points"] == corrected["grid_points"] == 16
    assert raw["grid_median_absolute_nonclosure"] > 5
    assert corrected["grid_median_absolute_nonclosure"] < 0.2


def test_conditional_md_experiment_refuses_outer_test_manifest(tmp_path):
    with pytest.raises(ValueError, match="outer train/sample.json"):
        main([
            "--checkpoint", str(tmp_path / "checkpoint.pth.tar"),
            "--ae-ckpt", str(tmp_path / "ae.pth"),
            "--data", str(tmp_path / "test/sample.json"),
            "--gen-weight-path", str(tmp_path / "weights.npz"),
            "--outdir", str(tmp_path / "output"),
        ])
    assert not (tmp_path / "output").exists()
