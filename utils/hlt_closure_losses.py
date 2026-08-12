"""Weighted versions of the successful V3/V4 QCD closure objectives."""

import torch

from utils.event_weights import weighted_quantile


def soft_abcd_tail_loss(x, y, quantiles, min_events=5, scale=12.0,
                        tail_focus_weight=2.0, eps=1e-6, weights=None):
    """Differentiable tail-ABCD residual under the physical event measure.

    This preserves the V3/V4 log-ratio objective while allowing generator
    weights. Weights are normalized to mean one so ``min_events`` remains an
    effective-event guard rather than depending on an arbitrary cross section.
    """
    x = x.float().view(-1)
    y = y.float().view(-1)
    weights = (
        torch.ones_like(x) if weights is None
        else weights.float().view(-1).to(x.device)
    )
    if x.shape != y.shape or x.shape != weights.shape:
        raise ValueError("x, y, and weights must have the same shape.")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Closure weights must be finite and non-negative.")
    quantiles = [q for q in quantiles if 0.0 < q < 1.0]
    if x.numel() < max(20, 4 * min_events) or not quantiles:
        zero = (x.sum() + y.sum()) * 0.0
        return zero, None
    weights = weights / weights.mean().clamp(min=eps)

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
    abs_log_ratios = []
    reliabilities = []
    for qx in quantiles:
        cut_x = cuts_x[qx]
        hard_x_high = x.detach() > cut_x
        sx_high = torch.sigmoid(scale * (x - cut_x) / x_scale)
        sx_low = 1.0 - sx_high
        for qy in quantiles:
            cut_y = cuts_y[qy]
            hard_y_high = y.detach() > cut_y
            hard_masks = (
                hard_x_high & hard_y_high,
                hard_x_high & ~hard_y_high,
                ~hard_x_high & hard_y_high,
                ~hard_x_high & ~hard_y_high,
            )
            with torch.no_grad():
                hard_masses = torch.stack([
                    weights[mask].sum() for mask in hard_masks
                ])
                hard_sumw2 = torch.stack([
                    weights[mask].square().sum() for mask in hard_masks
                ])
                effective_counts = (
                    hard_masses.square() / hard_sumw2.clamp(min=eps)
                )
                hard_min = effective_counts.min()
                reliability = torch.sigmoid(
                    (hard_min - float(min_events))
                    / max(float(min_events) * 0.25, 1.0)
                )

            sy_high = torch.sigmoid(scale * (y - cut_y) / y_scale)
            sy_low = 1.0 - sy_high
            A = (weights * sx_high * sy_high).sum()
            B = (weights * sx_high * sy_low).sum()
            C = (weights * sx_low * sy_high).sum()
            D = (weights * sx_low * sy_low).sum()
            log_ratio = (
                torch.log(B + eps) + torch.log(C + eps)
                - torch.log(D + eps) - torch.log(A + eps)
            )
            rel_tail = max(0.0, ((qx + qy) * 0.5 - 0.5) / 0.5)
            tail_weight = 1.0 + tail_focus_weight * rel_tail * rel_tail
            losses.append(reliability * tail_weight * log_ratio.square())
            abs_log_ratios.append(log_ratio.detach().abs())
            reliabilities.append(reliability)

    loss = torch.stack(losses).mean()
    reliability = torch.stack(reliabilities)
    metric = (
        torch.stack(abs_log_ratios) * reliability
    ).sum() / reliability.sum().clamp(min=eps)
    return loss, metric.item()
