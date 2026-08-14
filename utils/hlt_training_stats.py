import math

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold
from torch.utils.data import Sampler


def full_measure_scoped_mean(values, scope_weights, full_weights, eps=1e-8):
    """Weighted scoped contribution normalized by the full event measure.

    V4 added its QCD-only critic penalty to per-event losses before reducing
    over the complete all-background batch. Normalizing over QCD alone changes
    the loss coefficient by the inverse QCD fraction.
    """
    values = torch.as_tensor(values).reshape(-1)
    scope_weights = torch.as_tensor(
        scope_weights, device=values.device, dtype=values.dtype).reshape(-1)
    full_weights = torch.as_tensor(
        full_weights, device=values.device, dtype=values.dtype).reshape(-1)
    if values.numel() != scope_weights.numel():
        raise ValueError("values and scope_weights must align.")
    return (values * scope_weights).sum() / full_weights.sum().clamp(min=eps)

from utils.event_weights import weighted_quantile


def weighted_resample_indices(weights, n_samples=None):
    """Draw indices from the physical measure represented by event weights."""
    weights = torch.as_tensor(weights).float().view(-1)
    if weights.numel() == 0:
        return torch.empty(0, device=weights.device, dtype=torch.long)
    if n_samples is None:
        n_samples = weights.numel()
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    valid = torch.isfinite(weights) & (weights >= 0)
    if not valid.all() or float(weights.sum().item()) <= 0.0:
        raise ValueError(
            "Weighted resampling requires finite, non-negative weights with "
            "positive total mass."
        )
    probabilities = weights / weights.sum()
    return torch.multinomial(probabilities, n_samples, replacement=True)


def weighted_balanced_folds(weights, n_splits=2, seed=42):
    """Assign rows to folds while balancing total generator weight."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        return np.empty(0, dtype=np.int16)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Fold weights must be finite and non-negative.")
    n_splits = min(max(int(n_splits), 2), weights.size)
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(weights.size)
    order = shuffled[np.argsort(-weights[shuffled], kind="stable")]
    fold_mass = np.zeros(n_splits, dtype=np.float64)
    fold_size = np.zeros(n_splits, dtype=np.int64)
    fold_ids = np.empty(weights.size, dtype=np.int16)
    for index in order:
        # Size is a deterministic tie-breaker when several folds have equal mass.
        fold = min(range(n_splits), key=lambda value: (
            fold_mass[value], fold_size[value], value))
        fold_ids[index] = fold
        fold_mass[fold] += weights[index]
        fold_size[fold] += 1
    return fold_ids


def soft_conditioner_profile_loss(conditioner, target, n_bins=8,
                                  tail_weight=2.0, scale=12.0, eps=1e-8,
                                  weights=None):
    """Profile flatness with differentiable conditioner-bin membership."""
    conditioner = conditioner.float().view(-1)
    target = target.float().view(-1)
    weights = (
        torch.ones_like(conditioner) if weights is None
        else weights.float().view(-1).to(conditioner.device)
    )
    if conditioner.numel() < max(4, n_bins * 2):
        zero = (conditioner.sum() + target.sum()) * 0.0
        return zero, None
    with torch.no_grad():
        quantiles = torch.linspace(
            0, 1, n_bins + 1, device=conditioner.device)
        edges = weighted_quantile(
            conditioner.detach(), quantiles, weights.detach())
        width = (
            weighted_quantile(conditioner.detach(), [0.84], weights.detach())[0]
            - weighted_quantile(conditioner.detach(), [0.16], weights.detach())[0]
        ).clamp(min=eps) / max(n_bins, 1)
    total_weight = weights.sum().clamp(min=eps)
    global_mean = (weights * target).sum() / total_weight
    global_scale = torch.sqrt(
        (weights * (target - global_mean).square()).sum() / total_weight
    ).detach().clamp(min=eps)
    losses = []
    for i in range(n_bins):
        if i == 0:
            membership = torch.sigmoid(
                scale * (edges[1] - conditioner) / width)
        elif i == n_bins - 1:
            membership = torch.sigmoid(
                scale * (conditioner - edges[-2]) / width)
        else:
            lower = torch.sigmoid(
                scale * (conditioner - edges[i]) / width)
            upper = torch.sigmoid(
                scale * (edges[i + 1] - conditioner) / width)
            membership = lower * upper
        weighted_membership = membership * weights
        mass = weighted_membership.sum().clamp(min=eps)
        bin_mean = (weighted_membership * target).sum() / mass
        rel_tail = i / max(n_bins - 1, 1)
        weight = 1.0 + tail_weight * rel_tail * rel_tail
        losses.append(
            weight * ((bin_mean - global_mean) / global_scale).pow(2))
    loss = torch.stack(losses).mean()
    return loss, loss.detach().item()


def soft_copula_grid_loss(x, y, quantiles, scale=12.0,
                          tail_focus_weight=2.0, eps=1e-6, weights=None):
    """Normalized soft independence residuals over a quantile grid."""
    x = x.float().view(-1)
    y = y.float().view(-1)
    weights = (
        torch.ones_like(x) if weights is None
        else weights.float().view(-1).to(x.device)
    )
    weights = weights / weights.sum().clamp(min=eps)
    quantiles = [q for q in quantiles if 0.0 < q < 1.0]
    if x.numel() < 20 or not quantiles:
        zero = (x.sum() + y.sum()) * 0.0
        return zero, None
    with torch.no_grad():
        x_scale = (
            weighted_quantile(x.detach(), [0.84], weights.detach())[0]
            - weighted_quantile(x.detach(), [0.16], weights.detach())[0]
        ).clamp(min=eps)
        y_scale = (
            weighted_quantile(y.detach(), [0.84], weights.detach())[0]
            - weighted_quantile(y.detach(), [0.16], weights.detach())[0]
        ).clamp(min=eps)
        cuts_x = {
            q: weighted_quantile(x.detach(), [q], weights.detach())[0]
            for q in quantiles
        }
        cuts_y = {
            q: weighted_quantile(y.detach(), [q], weights.detach())[0]
            for q in quantiles
        }

    losses = []
    residuals = []
    for qx in quantiles:
        sx = torch.sigmoid(scale * (x - cuts_x[qx]) / x_scale)
        for qy in quantiles:
            sy = torch.sigmoid(scale * (y - cuts_y[qy]) / y_scale)
            px = (weights * sx).sum()
            py = (weights * sy).sum()
            joint = (weights * sx * sy).sum()
            denominator = torch.sqrt(
                px * (1.0 - px) * py * (1.0 - py) + eps)
            residual = (joint - px * py) / denominator
            tail = max(0.0, ((qx + qy) * 0.5 - 0.5) / 0.5)
            weight = 1.0 + tail_focus_weight * tail * tail
            losses.append(weight * residual.pow(2))
            residuals.append(residual.detach().abs())
    loss = torch.stack(losses).mean()
    return loss, torch.stack(residuals).mean().item()


class RunningQCDMDProxy:
    """Streaming Mahalanobis reference with weighted QCD moments.

    In ``ema`` mode each batch is scored before its detached moments update the
    active reference, which tracks a moving encoder without self-scoring. The
    legacy ``epoch`` mode keeps one frozen reference for a whole epoch.
    """

    def __init__(self, momentum=0.05, eps=1e-5, shrinkage=0.05,
                 mode="epoch"):
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1].")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1].")
        if mode not in {"ema", "epoch"}:
            raise ValueError("mode must be 'ema' or 'epoch'.")
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.shrinkage = float(shrinkage)
        self.mode = mode
        self.mean = None
        self.second_moment = None
        self.updates = 0
        self.begin_epoch()

    @property
    def ready(self):
        return self.mean is not None and self.second_moment is not None

    def begin_epoch(self):
        self._count = 0
        self._sum = None
        self._sum_outer = None

    @staticmethod
    def _batch_sums(latent, weights=None):
        values = latent.detach().float()
        if weights is None:
            weights = torch.ones(
                values.size(0), device=values.device, dtype=values.dtype)
        weights = weights.detach().float().view(-1).to(values.device)
        count = weights.sum()
        total = (values * weights[:, None]).sum(dim=0)
        total_outer = values.T @ (values * weights[:, None])
        return count, total, total_outer

    @staticmethod
    def _batch_moments(latent, weights=None):
        count, total, total_outer = RunningQCDMDProxy._batch_sums(
            latent, weights)
        return total / count, total_outer / count

    def _regularized_covariance(self, mean, second_moment):
        covariance = second_moment - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        diagonal_mean = covariance.diagonal().mean().clamp(min=self.eps)
        target = torch.eye(
            covariance.size(0), device=covariance.device,
            dtype=covariance.dtype
        ) * diagonal_mean
        covariance = (
            (1.0 - self.shrinkage) * covariance
            + self.shrinkage * target
        )
        return covariance + torch.eye(
            covariance.size(0), device=covariance.device,
            dtype=covariance.dtype
        ) * self.eps

    def _score(self, latent, mean, second_moment):
        mean = mean.to(device=latent.device, dtype=torch.float32)
        second_moment = second_moment.to(
            device=latent.device, dtype=torch.float32)
        covariance = self._regularized_covariance(mean, second_moment)
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            whitening = eigenvectors / eigenvalues.clamp(min=self.eps).sqrt()
        centered = latent.float() - mean
        whitened = centered @ whitening
        return (whitened * whitened).sum(dim=1).to(latent.dtype)

    def update(self, qcd_latent, weights=None):
        """Update EMA moments or accumulate moments for epoch replacement."""
        if qcd_latent.size(0) < 1:
            return
        count, total, total_outer = self._batch_sums(qcd_latent, weights)
        if float(count.item()) <= 0.0:
            return
        batch_mean = (total / count).cpu()
        batch_second = (total_outer / count).cpu()
        if self.mode == "ema":
            if not self.ready:
                self.mean = batch_mean
                self.second_moment = batch_second
            else:
                self.mean = (
                    (1.0 - self.momentum) * self.mean
                    + self.momentum * batch_mean
                )
                self.second_moment = (
                    (1.0 - self.momentum) * self.second_moment
                    + self.momentum * batch_second
                )
            self.updates += 1
            return
        total = total.cpu()
        total_outer = total_outer.cpu()
        self._count += float(count.item())
        self._sum = total if self._sum is None else self._sum + total
        self._sum_outer = (
            total_outer if self._sum_outer is None
            else self._sum_outer + total_outer
        )

    def finalize_epoch(self):
        """Atomically replace the active reference with this epoch's moments."""
        if self.mode == "ema":
            return False
        if self._count <= 0.0:
            return False
        self.mean = self._sum / self._count
        self.second_moment = self._sum_outer / self._count
        self.updates += 1
        self.begin_epoch()
        return True

    def md(self, latent, qcd_mask, update=True, weights=None):
        qcd_count = int(qcd_mask.sum().item())
        if qcd_count < 2:
            return torch.zeros(
                latent.size(0), device=latent.device, dtype=latent.dtype)

        qcd_latent = latent[qcd_mask]
        qcd_weights = None if weights is None else weights[qcd_mask]
        if self.ready:
            scores = self._score(latent, self.mean, self.second_moment)
        else:
            mean, second_moment = self._batch_moments(
                qcd_latent, qcd_weights)
            scores = self._score(latent, mean, second_moment)
        if update:
            self.update(qcd_latent, qcd_weights)
        return scores

    def state_dict(self):
        return {
            "momentum": self.momentum,
            "eps": self.eps,
            "shrinkage": self.shrinkage,
            "mode": self.mode,
            "updates": self.updates,
            "mean": None if self.mean is None else self.mean.detach().cpu(),
            "second_moment": (
                None if self.second_moment is None
                else self.second_moment.detach().cpu()
            ),
        }

    def load_state_dict(self, state):
        self.momentum = float(state.get("momentum", 1.0))
        self.eps = float(state["eps"])
        self.shrinkage = float(state.get("shrinkage", 0.05))
        self.mode = state.get("mode", "epoch")
        self.updates = int(state.get("updates", 0))
        self.mean = state.get("mean")
        self.second_moment = state.get("second_moment")
        self.begin_epoch()


def _weighted_covariance(values, weights, shrinkage=0.05, eps=1e-8):
    weights = np.asarray(weights, dtype=np.float64)
    total = max(weights.sum(), eps)
    mean = np.sum(values * weights[:, None], axis=0) / total
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered / total
    covariance = 0.5 * (covariance + covariance.T)
    diagonal_mean = max(float(np.trace(covariance) / covariance.shape[0]), eps)
    target = np.eye(covariance.shape[0]) * diagonal_mean
    covariance = (1.0 - shrinkage) * covariance + shrinkage * target
    covariance += np.eye(covariance.shape[0]) * eps
    return mean, covariance


def cross_fitted_mahalanobis(latents, n_splits=2, seed=42,
                             sample_weights=None, shrinkage=0.05):
    """Score every row against a reference fit on other folds."""
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("latents must be a two-dimensional array.")
    if values.shape[0] < 4:
        return np.full(values.shape[0], np.nan, dtype=np.float64)
    n_splits = min(max(int(n_splits), 2), values.shape[0])
    scores = np.empty(values.shape[0], dtype=np.float64)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if sample_weights.shape[0] != values.shape[0]:
            raise ValueError("sample_weights must align with latents.")
    if sample_weights is None:
        splitter = KFold(
            n_splits=n_splits, shuffle=True, random_state=int(seed))
        splits = splitter.split(values)
    else:
        fold_ids = weighted_balanced_folds(
            sample_weights, n_splits=n_splits, seed=seed)
        all_indices = np.arange(values.shape[0])
        splits = (
            (all_indices[fold_ids != fold], all_indices[fold_ids == fold])
            for fold in range(n_splits)
        )
    for fit_idx, score_idx in splits:
        if sample_weights is None:
            covariance = LedoitWolf(assume_centered=False).fit(values[fit_idx])
            mean = covariance.location_
            covariance_matrix = covariance.covariance_
        else:
            mean, covariance_matrix = _weighted_covariance(
                values[fit_idx], sample_weights[fit_idx],
                shrinkage=shrinkage)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvalues = np.clip(eigenvalues, 1e-8, None)
        whitening = eigenvectors / np.sqrt(eigenvalues)
        transformed = (values[score_idx] - mean) @ whitening
        scores[score_idx] = np.sum(transformed * transformed, axis=1)
    return scores


class QCDRichBatchSampler(Sampler):
    """Fixed-size batches with a controlled QCD fraction.

    QCD is cycled when necessary; other backgrounds are shuffled and cycled
    independently. ``sampling_correction`` returns the two constant importance
    factors needed to recover expectations under the natural training mixture.
    """

    def __init__(self, labels, batch_size, qcd_label=1, qcd_fraction=0.25,
                 drop_last=True, seed=42):
        labels = torch.as_tensor(labels).long().cpu().numpy()
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2.")
        if not 0.0 < qcd_fraction < 1.0:
            raise ValueError("qcd_fraction must be in (0, 1).")
        self.batch_size = int(batch_size)
        self.qcd_label = int(qcd_label)
        self.qcd_fraction = float(qcd_fraction)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.qcd_indices = np.flatnonzero(labels == self.qcd_label)
        self.other_indices = np.flatnonzero(labels != self.qcd_label)
        if self.qcd_indices.size == 0 or self.other_indices.size == 0:
            raise ValueError("QCD-rich sampling needs both QCD and non-QCD events.")
        self.n_events = int(labels.size)
        self.n_qcd = max(
            1, min(self.batch_size - 1,
                   int(round(self.batch_size * self.qcd_fraction))))
        self.n_other = self.batch_size - self.n_qcd
        self.n_batches = (
            self.n_events // self.batch_size if self.drop_last
            else int(math.ceil(self.n_events / self.batch_size))
        )
        self.natural_qcd_fraction = self.qcd_indices.size / self.n_events

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        qcd_pool = rng.permutation(self.qcd_indices)
        other_pool = rng.permutation(self.other_indices)
        qcd_cursor = other_cursor = 0

        def take(pool, cursor, count):
            pieces = []
            while count > 0:
                available = pool.size - cursor
                if available == 0:
                    pool = rng.permutation(pool)
                    cursor = 0
                    available = pool.size
                n_take = min(count, available)
                pieces.append(pool[cursor:cursor + n_take])
                cursor += n_take
                count -= n_take
            return np.concatenate(pieces), pool, cursor

        for _ in range(self.n_batches):
            qcd, qcd_pool, qcd_cursor = take(
                qcd_pool, qcd_cursor, self.n_qcd)
            other, other_pool, other_cursor = take(
                other_pool, other_cursor, self.n_other)
            batch = np.concatenate([qcd, other])
            rng.shuffle(batch)
            yield batch.tolist()

    def sampling_correction(self):
        natural = self.natural_qcd_fraction
        sampled = self.n_qcd / self.batch_size
        return {
            "qcd": natural / sampled,
            "other": (1.0 - natural) / (1.0 - sampled),
        }
