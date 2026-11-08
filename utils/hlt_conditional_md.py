"""Training-fitted, physics-weighted conditional CDF for QCD MD scores.

The AE coordinate is divided into equal-*weight* bins. Within each bin the
MD score is mapped to its weighted empirical percentile. This is a deliberately
simple score-level diagnostic, not a trained replacement for NURD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _validated_arrays(ae, md, weights=None):
    ae = np.asarray(ae, dtype=np.float64).reshape(-1)
    md = np.asarray(md, dtype=np.float64).reshape(-1)
    if ae.shape != md.shape or not len(ae):
        raise ValueError("AE and MD must be aligned, nonempty arrays.")
    if not np.isfinite(ae).all() or (ae <= 0).any():
        raise ValueError("AE scores must be finite and strictly positive.")
    if not np.isfinite(md).all() or (md < 0).any():
        raise ValueError("MD scores must be finite and nonnegative.")
    if weights is None:
        return ae, md, None
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape != ae.shape or not np.isfinite(weights).all():
        raise ValueError("Weights must be finite and aligned with scores.")
    if (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Weights must be nonnegative with positive total mass.")
    return ae, md, weights


def _weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    indices = np.searchsorted(
        cumulative, np.asarray(probabilities) * cumulative[-1], side="left")
    return sorted_values[np.minimum(indices, len(sorted_values) - 1)]


@dataclass(frozen=True)
class ConditionalMDCDF:
    """Piecewise empirical P(MD <= md | log10(AE) bin), fitted on QCD only."""

    log_ae_edges: np.ndarray
    offsets: np.ndarray
    sorted_md: np.ndarray
    weighted_mid_cdf: np.ndarray
    fit_counts: np.ndarray
    fit_effective_counts: np.ndarray

    @classmethod
    def fit(cls, ae, md, physics_weights, *, n_ae_bins=20, min_ess=200):
        ae, md, weights = _validated_arrays(ae, md, physics_weights)
        if not 2 <= int(n_ae_bins) <= 200:
            raise ValueError("n_ae_bins must be between 2 and 200.")
        if float(min_ess) <= 0:
            raise ValueError("min_ess must be positive.")
        positive = weights > 0
        log_ae = np.log10(ae[positive])
        md = md[positive]
        weights = weights[positive]
        edges = np.unique(_weighted_quantiles(
            log_ae, weights, np.linspace(0, 1, int(n_ae_bins) + 1)))
        if len(edges) < 3:
            raise ValueError("AE scores have too few distinct weighted bins.")
        bins = np.searchsorted(edges[1:-1], log_ae, side="right")

        md_chunks = []
        cdf_chunks = []
        offsets = [0]
        fit_counts = []
        fit_ess = []
        for bin_index in range(len(edges) - 1):
            mask = bins == bin_index
            values = md[mask]
            mass = weights[mask]
            ess = float(mass.sum() ** 2 / np.square(mass).sum()) if len(mass) else 0.0
            if ess < min_ess:
                raise ValueError(
                    f"AE bin {bin_index} has ESS={ess:.1f} < {min_ess}; "
                    "use fewer bins or improve QCD statistics.")
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_mass = mass[order]
            unique_values, first = np.unique(sorted_values, return_index=True)
            grouped_mass = np.add.reduceat(sorted_mass, first)
            midpoint = (
                np.cumsum(grouped_mass) - 0.5 * grouped_mass
            ) / grouped_mass.sum()
            md_chunks.append(unique_values)
            cdf_chunks.append(midpoint)
            offsets.append(offsets[-1] + len(unique_values))
            fit_counts.append(len(values))
            fit_ess.append(ess)
        return cls(
            log_ae_edges=edges,
            offsets=np.asarray(offsets, dtype=np.int64),
            sorted_md=np.concatenate(md_chunks),
            weighted_mid_cdf=np.concatenate(cdf_chunks),
            fit_counts=np.asarray(fit_counts, dtype=np.int64),
            fit_effective_counts=np.asarray(fit_ess, dtype=np.float64),
        )

    def bin_indices(self, ae):
        ae = np.asarray(ae, dtype=np.float64).reshape(-1)
        if not np.isfinite(ae).all() or (ae <= 0).any():
            raise ValueError("AE scores must be finite and strictly positive.")
        return np.searchsorted(
            self.log_ae_edges[1:-1], np.log10(ae), side="right")

    def transform(self, ae, md):
        ae, md, _ = _validated_arrays(ae, md)
        bins = self.bin_indices(ae)
        result = np.empty(len(ae), dtype=np.float64)
        for bin_index in range(len(self.fit_counts)):
            mask = bins == bin_index
            start, stop = self.offsets[bin_index:bin_index + 2]
            result[mask] = np.interp(
                md[mask], self.sorted_md[start:stop],
                self.weighted_mid_cdf[start:stop], left=0.0, right=1.0)
        return result

    def save(self, path):
        path = Path(path)
        np.savez_compressed(
            path,
            log_ae_edges=self.log_ae_edges,
            offsets=self.offsets,
            sorted_md=self.sorted_md,
            weighted_mid_cdf=self.weighted_mid_cdf,
            fit_counts=self.fit_counts,
            fit_effective_counts=self.fit_effective_counts,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            return cls(**{key: saved[key] for key in (
                "log_ae_edges", "offsets", "sorted_md", "weighted_mid_cdf",
                "fit_counts", "fit_effective_counts")})
