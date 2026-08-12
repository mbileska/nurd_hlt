import torch

from utils.hlt_closure_losses import soft_abcd_tail_loss


def test_weighted_tail_abcd_has_gradients_for_both_axes():
    torch.manual_seed(7)
    x = torch.randn(128, requires_grad=True)
    y = (0.4 * x.detach() + torch.randn(128)).requires_grad_()
    weights = torch.linspace(0.2, 3.0, 128)

    loss, metric = soft_abcd_tail_loss(
        x, y, [0.5, 0.7, 0.85], weights=weights)
    loss.backward()

    assert torch.isfinite(loss)
    assert metric is not None
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert y.grad is not None and y.grad.abs().sum() > 0


def test_tail_abcd_rejects_misaligned_weights():
    with torch.no_grad():
        x = torch.arange(20, dtype=torch.float32)
        y = torch.arange(20, dtype=torch.float32)
        try:
            soft_abcd_tail_loss(x, y, [0.5], weights=torch.ones(19))
        except ValueError as exc:
            assert "same shape" in str(exc)
        else:
            raise AssertionError("misaligned weights were accepted")
