import math

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold
from torch.utils.data import Sampler


def soft_conditioner_profile_loss(conditioner, target, n_bins=8,
                                  tail_weight=2.0, scale=12.0, eps=1e-8):
    """Profile flatness with differentiable conditioner-bin membership."""
    conditioner = conditioner.float().view(-1)
    target = target.float().view(-1)
    if conditioner.numel() < max(4, n_bins * 2):
        zero = (conditioner.sum() + target.sum()) * 0.0
        return zero, None
    with torch.no_grad():
        edges = torch.quantile(
            conditioner.detach(),
            torch.linspace(0, 1, n_bins + 1, device=conditioner.device))
        width = (
            torch.quantile(conditioner.detach(), 0.84)
            - torch.quantile(conditioner.detach(), 0.16)
        ).clamp(min=eps) / max(n_bins, 1)
    global_mean = target.mean()
    global_scale = target.std(unbiased=False).detach().clamp(min=eps)
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
        mass = membership.sum().clamp(min=eps)
        bin_mean = (membership * target).sum() / mass
        rel_tail = i / max(n_bins - 1, 1)
        weight = 1.0 + tail_weight * rel_tail * rel_tail
        losses.append(
            weight * ((bin_mean - global_mean) / global_scale).pow(2))
    loss = torch.stack(losses).mean()
    return loss, loss.detach().item()


def soft_copula_grid_loss(x, y, quantiles, scale=12.0,
                          tail_focus_weight=2.0, eps=1e-6):
    """Normalized soft independence residuals over a quantile grid."""
    x = x.float().view(-1)
    y = y.float().view(-1)
    quantiles = [q for q in quantiles if 0.0 < q < 1.0]
    if x.numel() < 20 or not quantiles:
        zero = (x.sum() + y.sum()) * 0.0
        return zero, None
    with torch.no_grad():
        x_scale = (
            torch.quantile(x.detach(), 0.84)
            - torch.quantile(x.detach(), 0.16)
        ).clamp(min=eps)
        y_scale = (
            torch.quantile(y.detach(), 0.84)
            - torch.quantile(y.detach(), 0.16)
        ).clamp(min=eps)
        cuts_x = {q: torch.quantile(x.detach(), q) for q in quantiles}
        cuts_y = {q: torch.quantile(y.detach(), q) for q in quantiles}

    losses = []
    residuals = []
    for qx in quantiles:
        sx = torch.sigmoid(scale * (x - cuts_x[qx]) / x_scale)
        for qy in quantiles:
            sy = torch.sigmoid(scale * (y - cuts_y[qy]) / y_scale)
            px = sx.mean()
            py = sy.mean()
            joint = (sx * sy).mean()
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
    """Epoch-frozen Mahalanobis reference with streaming QCD moments.

    Every batch in an epoch is scored against the same reference. Detached QCD
    moments are accumulated during that epoch and become the reference only
    after ``finalize_epoch``. This avoids batch-order dependence and prevents a
    batch from changing the covariance used to score itself.
    """

    def __init__(self, momentum=1.0, eps=1e-5, shrinkage=0.05):
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1].")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1].")
        # momentum is retained in checkpoints for backward compatibility. New
        # references are intentionally replaced once per epoch.
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.shrinkage = float(shrinkage)
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
    def _batch_sums(latent):
        values = latent.detach().float()
        return values.size(0), values.sum(dim=0), values.T @ values

    @staticmethod
    def _batch_moments(latent):
        count, total, total_outer = RunningQCDMDProxy._batch_sums(latent)
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

    def update(self, qcd_latent):
        """Accumulate detached moments without changing the active reference."""
        if qcd_latent.size(0) < 1:
            return
        count, total, total_outer = self._batch_sums(qcd_latent)
        total = total.cpu()
        total_outer = total_outer.cpu()
        self._count += count
        self._sum = total if self._sum is None else self._sum + total
        self._sum_outer = (
            total_outer if self._sum_outer is None
            else self._sum_outer + total_outer
        )

    def finalize_epoch(self):
        """Atomically replace the active reference with this epoch's moments."""
        if self._count < 2:
            return False
        self.mean = self._sum / self._count
        self.second_moment = self._sum_outer / self._count
        self.updates += 1
        self.begin_epoch()
        return True

    def md(self, latent, qcd_mask, update=True):
        qcd_count = int(qcd_mask.sum().item())
        if qcd_count < 2:
            return torch.zeros(
                latent.size(0), device=latent.device, dtype=latent.dtype)

        qcd_latent = latent[qcd_mask]
        if self.ready:
            scores = self._score(latent, self.mean, self.second_moment)
        else:
            mean, second_moment = self._batch_moments(qcd_latent)
            scores = self._score(latent, mean, second_moment)
        if update:
            self.update(qcd_latent)
        return scores

    def state_dict(self):
        return {
            "momentum": self.momentum,
            "eps": self.eps,
            "shrinkage": self.shrinkage,
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
        self.updates = int(state.get("updates", 0))
        self.mean = state.get("mean")
        self.second_moment = state.get("second_moment")
        self.begin_epoch()


def cross_fitted_mahalanobis(latents, n_splits=2, seed=42):
    """Score every row against a Ledoit-Wolf reference fit on other folds."""
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("latents must be a two-dimensional array.")
    if values.shape[0] < 4:
        return np.full(values.shape[0], np.nan, dtype=np.float64)
    n_splits = min(max(int(n_splits), 2), values.shape[0])
    scores = np.empty(values.shape[0], dtype=np.float64)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for fit_idx, score_idx in splitter.split(values):
        covariance = LedoitWolf(assume_centered=False).fit(values[fit_idx])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.covariance_)
        eigenvalues = np.clip(eigenvalues, 1e-8, None)
        whitening = eigenvectors / np.sqrt(eigenvalues)
        transformed = (values[score_idx] - covariance.location_) @ whitening
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
