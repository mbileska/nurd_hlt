import numpy as np

from utils.hlt_score_calibration import (
    fit_class_references,
    load_class_references,
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
