import torch


class RunningQCDMDProxy:
    """Lagged EMA Mahalanobis reference built from QCD latent moments."""

    def __init__(self, momentum=0.05, eps=1e-5):
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1].")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.mean = None
        self.second_moment = None
        self.updates = 0

    @property
    def ready(self):
        return self.mean is not None and self.second_moment is not None

    @staticmethod
    def _batch_moments(latent):
        values = latent.detach().float()
        mean = values.mean(dim=0)
        second_moment = values.T @ values / values.size(0)
        return mean, second_moment

    def _score(self, latent, mean, second_moment):
        mean = mean.to(device=latent.device, dtype=torch.float32)
        second_moment = second_moment.to(device=latent.device, dtype=torch.float32)
        covariance = second_moment - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        covariance = covariance + torch.eye(
            covariance.size(0), device=covariance.device,
            dtype=covariance.dtype
        ) * self.eps
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            whitening = eigenvectors / eigenvalues.clamp(min=self.eps).sqrt()
        centered = latent.float() - mean
        whitened = centered @ whitening
        return (whitened * whitened).sum(dim=1).to(latent.dtype)

    def update(self, qcd_latent):
        if qcd_latent.size(0) < 2:
            return
        mean, second_moment = self._batch_moments(qcd_latent)
        if not self.ready:
            self.mean = mean
            self.second_moment = second_moment
        else:
            momentum = self.momentum
            self.mean = (
                (1.0 - momentum) * self.mean.to(mean.device)
                + momentum * mean
            )
            self.second_moment = (
                (1.0 - momentum) * self.second_moment.to(second_moment.device)
                + momentum * second_moment
            )
        self.updates += 1

    def md(self, latent, qcd_mask, update=True):
        """Score with the previous reference, then optionally update from QCD."""
        if int(qcd_mask.sum().item()) < 2:
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
            "updates": self.updates,
            "mean": None if self.mean is None else self.mean.detach().cpu(),
            "second_moment": (
                None if self.second_moment is None
                else self.second_moment.detach().cpu()
            ),
        }

    def load_state_dict(self, state):
        self.momentum = float(state["momentum"])
        self.eps = float(state["eps"])
        self.updates = int(state.get("updates", 0))
        self.mean = state.get("mean")
        self.second_moment = state.get("second_moment")
