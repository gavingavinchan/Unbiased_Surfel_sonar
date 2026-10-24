"""CPU checks for the return-weighted sonar loss (experiment 260)."""
import torch
from utils.loss_utils import return_mask, return_weighted_loss, l1_loss, ssim


def test_mask_dilates_one_bin():
    t = torch.zeros(3, 20, 30); t[:, 10, 15] = 1
    m = return_mask(t, .5, 1)
    assert m.shape == (20, 30) and int(m.sum()) == 9 and bool(m[9:12, 14:17].all())


def test_equal_weights_and_fallback():
    torch.manual_seed(0)
    t = torch.zeros(3, 40, 50); t[:, 20, 10:20] = 1
    p = torch.rand(3, 40, 50) * .1
    loss, n = return_weighted_loss(p, t, .5)
    assert n == 3 * 12
    fg = return_mask(t, .5)
    err = (p - t).abs().mean(0)
    l1 = .5 * err[fg].mean() + .5 * err[~fg].mean()
    assert loss > .8 * l1 - 1e-6          # SSIM term is non-negative here
    empty = torch.zeros(3, 40, 50)
    loss0, n0 = return_weighted_loss(p, empty, .5)
    assert n0 == 0
    assert torch.allclose(loss0, .8 * l1_loss(p, empty) + .2 * (1 - ssim(p, empty)), atol=1e-6)


def test_gradient_reaches_returns():
    t = torch.zeros(3, 40, 50); t[:, 20, 25] = 1
    p = torch.zeros(3, 40, 50, requires_grad=True)
    loss, _ = return_weighted_loss(p, t, .5); loss.backward()
    g_fg = p.grad[0, 20, 25].abs(); g_bg = p.grad[0, 5, 5].abs()
    assert g_fg > 50 * g_bg
