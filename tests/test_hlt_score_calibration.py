import numpy as np

from utils.hlt_score_calibration import (
    fit_class_references,
    fit_conditional_cdf,
    load_conditional_cdf,
    load_class_references,
    save_conditional_cdf,
    save_class_references,
    score_latents,
    validate_reference_metadata,
)


def _synthetic_reference(seed=7):
    rng = np.random.default_rng(seed)
    class_zero = rng.normal(loc=(-2.0, 0.0), scale=(0.5, 1.5), size=(400, 2))
    class_one = rng.normal(loc=(2.0, 0.0), scale=(1.5, 0.5), size=(400, 2))
    latents = np.concatenate([class_zero, class_one], axis=0)
    labels = np.concatenate([
        np.zeros(class_zero.shape[0], dtype=np.int64),
        np.ones(class_one.shape[0], dtype=np.int64),
    ])
    fit = np.concatenate([np.arange(0, 250), np.arange(400, 650)])
    calibration = np.concatenate([np.arange(250, 400), np.arange(650, 800)])
    references = fit_class_references(
        latents, labels, fit, calibration, [0, 1], n_components=2)
    return references


def test_calibrated_union_is_large_only_away_from_all_classes():
    references = _synthetic_reference()
    latents = np.asarray([[-2.0, 0.0], [2.0, 0.0], [12.0, 12.0]])
    logits = np.asarray([[5.0, -5.0], [-5.0, 5.0], [0.0, 0.0]])
    score, products = score_latents(
        latents, logits, references, score_mode="calibrated_union", qcd_label=1)

    assert score[2] > score[0]
    assert score[2] > score[1]
    assert products["classifier_route_index"].tolist()[:2] == [0, 1]
    assert products["md_per_class"].shape == (3, 2)


def test_score_modes_are_finite():
    references = _synthetic_reference()
    latents = np.asarray([[-2.0, 0.0], [2.0, 0.0], [6.0, 6.0]])
    logits = np.asarray([[2.0, -2.0], [-2.0, 2.0], [0.5, -0.5]])
    for mode in (
        "calibrated_union", "mixture_nll", "qcd_md", "min_md",
        "classifier_routed", "gaussian_routed",
    ):
        score, _ = score_latents(
            latents, logits, references, score_mode=mode, qcd_label=1)
        assert np.isfinite(score).all()


def test_reference_round_trip(tmp_path):
    references = _synthetic_reference()
    path = tmp_path / "references.npz"
    metadata = {"checkpoint": "example", "n_pca": 2}
    save_class_references(path, references, metadata)
    loaded, loaded_metadata = load_class_references(path)

    assert validate_reference_metadata(loaded_metadata, metadata)
    assert [ref.label for ref in loaded] == [0, 1]
    for original, restored in zip(references, loaded):
        np.testing.assert_allclose(original.mean, restored.mean)
        np.testing.assert_allclose(original.whitening, restored.whitening)
        np.testing.assert_allclose(original.calibration_md, restored.calibration_md)


def test_frozen_conditional_cdf_removes_heldout_ae_md_trend(tmp_path):
    rng = np.random.default_rng(31)

    def sample(size):
        log_ae = rng.uniform(-3.0, 2.0, size=size)
        ae = np.exp(log_ae)
        # Deliberately heteroscedastic and nonlinear MD dependence.
        md = np.exp(0.75 * log_ae + rng.normal(
            scale=0.25 + 0.08 * (log_ae + 3.0), size=size))
        weights = np.exp(rng.normal(scale=0.35, size=size))
        return ae, md, weights

    calibration_ae, calibration_md, calibration_weights = sample(12_000)
    test_ae, test_md, _ = sample(8_000)
    calibration = fit_conditional_cdf(
        calibration_ae, calibration_md, calibration_weights,
        n_conditioner_bins=20, n_target_quantiles=257,
        min_bin_events=100)
    transformed = calibration.transform(test_ae, test_md)

    raw_corr = np.corrcoef(np.log(test_ae), np.log(test_md))[0, 1]
    calibrated_corr = np.corrcoef(np.log(test_ae), transformed)[0, 1]
    assert raw_corr > 0.8
    assert abs(calibrated_corr) < 0.08
    assert np.isfinite(transformed).all()
    assert (transformed > 0.0).all()

    path = tmp_path / "conditional_cdf.npz"
    save_conditional_cdf(path, calibration, {"split": "calibration"})
    restored, metadata = load_conditional_cdf(path)
    np.testing.assert_allclose(
        restored.transform(test_ae, test_md), transformed)
    assert metadata == {"split": "calibration"}
