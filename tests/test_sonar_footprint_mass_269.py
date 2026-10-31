"""269: SONAR_RASTER_FOOTPRINT_NORM=mass helper reproduces the forward kernel footprint mass (CPU-only)."""
import math

import pytest

torch = pytest.importorskip("torch")
gr = pytest.importorskip("gaussian_renderer")


def _mass(var_x, var_y, mu=(128.0, 100.0)):
    mu_t = torch.tensor([list(mu)], dtype=torch.float64)
    sigma = torch.tensor([[[var_x, 0.0], [0.0, var_y]]], dtype=torch.float64)
    return float(gr._sonar_footprint_pixel_mass(mu_t, sigma, 256, 200, 0.0625)[0])


def test_resolved_footprint_mass_matches_truncated_gaussian_integral():
    # alpha cut 0.0625*G >= 1/255 keeps rho <= -2 ln(16/255); mass = 2 pi sqrt(det) (1 - 16/255)
    expected = 2 * math.pi * 5.0 * 4.0 * (1 - 16 / 255)
    assert abs(_mass(25.0, 16.0) / expected - 1) < 0.01


def test_subpixel_footprint_mass_is_low_pass_floor():
    # min variance 0.1 px^2: low-pass exp(-|d|^2) dominates; the alpha cut keeps |d|^2 <= 2 on the
    # integer grid around an integer centre: 1 + 4 e^-1 + 4 e^-2 ~= 3.01 px (not area-proportional)
    assert abs(_mass(0.1, 0.1) - (1 + 4 * math.exp(-1) + 4 * math.exp(-2))) < 1e-6


def test_variance_clamp_caps_mass():
    assert abs(_mass(1e4, 1e4) - _mass(400.0, 400.0)) < 1e-6


def test_mass_is_differentiable_in_covariance():
    mu = torch.tensor([[128.0, 100.0]], dtype=torch.float64)
    sigma = torch.tensor([[[9.0, 1.0], [1.0, 4.0]]], dtype=torch.float64, requires_grad=True)
    m = gr._sonar_footprint_pixel_mass(mu, sigma, 256, 200, 0.0625).sum()
    m.backward()
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all() and float(sigma.grad[0, 0, 0]) > 0
