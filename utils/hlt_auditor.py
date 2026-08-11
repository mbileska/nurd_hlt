"""Fresh post-hoc nuisance auditors for latent-independence diagnostics."""

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def nuisance_auditor(train_latents, train_nuisance, test_latents,
                     test_nuisance, bin_edges, seed=42,
                     train_weights=None, test_weights=None,
                     max_iter=100):
    """Train and score a fresh nuisance-bin auditor under a weighted measure."""
    edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    train_bins = np.searchsorted(
        edges[1:-1], train_nuisance, side="right")
    test_bins = np.searchsorted(
        edges[1:-1], test_nuisance, side="right")
    classes = np.arange(len(edges) - 1)
    train_weights = (
        np.ones(len(train_bins), dtype=np.float64) if train_weights is None
        else np.asarray(train_weights, dtype=np.float64).reshape(-1)
    )
    test_weights = (
        np.ones(len(test_bins), dtype=np.float64) if test_weights is None
        else np.asarray(test_weights, dtype=np.float64).reshape(-1)
    )
    if len(train_weights) != len(train_bins) or len(test_weights) != len(test_bins):
        raise ValueError("Auditor weights must align with nuisance values.")
    train_weights = train_weights / train_weights.sum()
    test_weights = test_weights / test_weights.sum()
    rng = np.random.default_rng(seed)
    physical_train_idx = rng.choice(
        len(train_bins), size=len(train_bins), replace=True,
        p=train_weights)
    auditor = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 64), activation="relu", alpha=1e-4,
            batch_size=1024, learning_rate_init=1e-3,
            max_iter=int(max_iter), early_stopping=True,
            validation_fraction=0.2, n_iter_no_change=8,
            random_state=seed,
        ),
    )
    auditor.fit(
        np.asarray(train_latents)[physical_train_idx],
        train_bins[physical_train_idx])
    probabilities = auditor.predict_proba(test_latents)
    aligned_probabilities = np.full(
        (len(test_bins), len(classes)), 1e-12, dtype=np.float64)
    aligned_probabilities[:, auditor.classes_.astype(int)] = probabilities
    aligned_probabilities /= aligned_probabilities.sum(axis=1, keepdims=True)

    train_prior = np.bincount(
        train_bins, weights=train_weights,
        minlength=len(classes)).astype(np.float64)
    train_prior /= train_prior.sum()
    chance_probabilities = np.broadcast_to(
        train_prior, aligned_probabilities.shape)
    try:
        macro_auc = roc_auc_score(
            test_bins, aligned_probabilities, labels=classes,
            multi_class="ovr", average="macro",
            sample_weight=test_weights)
    except ValueError:
        macro_auc = np.nan
    return {
        "train_n": int(len(train_bins)),
        "test_n": int(len(test_bins)),
        "n_bins": int(len(classes)),
        "measure": "generator_weighted_qcd",
        "accuracy": float(np.sum(
            test_weights
            * (auditor.predict(test_latents) == test_bins))),
        "majority_accuracy": float(np.bincount(
            test_bins, weights=test_weights,
            minlength=len(classes)).max()),
        "cross_entropy": float(log_loss(
            test_bins, aligned_probabilities, labels=classes,
            sample_weight=test_weights)),
        "prior_cross_entropy": float(log_loss(
            test_bins, chance_probabilities, labels=classes,
            sample_weight=test_weights)),
        "macro_ovr_auc": float(macro_auc),
        "iterations": int(auditor[-1].n_iter_),
    }
