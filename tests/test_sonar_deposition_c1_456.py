"""Opt-in C1 numerical candidate: explicit rejection and positive-event conservation."""
import math
import pytest
import torch
from test_sonar_deposition_contract_337 import ns, args, enabled


def candidate(monkeypatch):
    enabled(monkeypatch)
    monkeypatch.setenv('SONAR_DEPOSITION_CONTRACT', 'intrinsic_screen_mass_c1_456_v1')
    monkeypatch.setenv('SONAR_USE_CUDA_RASTERIZER', '0')


@pytest.mark.parametrize('mode', ['1', 'true', 'yes', None])
def test_compiled_mode_rejected_before_dispatch(monkeypatch, mode):
    candidate(monkeypatch)
    if mode is None:
        monkeypatch.delenv('SONAR_USE_CUDA_RASTERIZER')
    else:
        monkeypatch.setenv('SONAR_USE_CUDA_RASTERIZER', mode)
    with pytest.raises(ValueError, match='compiled deposition is unsupported'):
        ns['_sonar_deposition_contract']()


def test_cropped_positive_events_and_independent_screen_kernel(monkeypatch):
    candidate(monkeypatch)
    a = args()
    a['vis_mu'] = torch.tensor([[.2, .3]], dtype=torch.float64, requires_grad=True)
    r, d, s, _ = ns['_rasterize_sonar_event_volume_matched_fallback'](**a)
    # Intrinsic covariance <= .5 I: the screen Gaussian wins everywhere.
    weights = []
    cutoff = 16 / 255
    for y in range(9):
        row = []
        for x in range(9):
            q = math.exp(-((x-.2)**2+(y-.3)**2))
            if q <= .75*cutoff:
                w = 0.
            elif q >= 1.25*cutoff:
                w = q
            else:
                t = (q-.75*cutoff)/(.5*cutoff)
                w = q*(3*t*t-2*t*t*t)
            row.append(w)
        weights.append(row)
    k = torch.tensor(weights, dtype=torch.float64)
    k /= k.sum()
    assert torch.max(torch.abs(r[:, 1, :].T-.7*k)) < 1e-14
    assert r.sum().item() == pytest.approx(.7, abs=1e-14)
    assert d.sum().item() == pytest.approx(.91, abs=1e-14)
    assert s.sum().item() == pytest.approx(.3, abs=1e-14)
    # Normalization must differentiate too: total event energy is position invariant.
    assert torch.max(torch.abs(torch.autograd.grad(r.sum(), a['vis_mu'])[0])) < 1e-13
    assert r[:, 0, :].count_nonzero() == 0
    for key in ['event_surfel_idx', 'event_ray_ids', 'event_returns', 'event_share']:
        a[key] = a[key][:0]
    assert all(x.count_nonzero() == 0 for x in ns['_rasterize_sonar_event_volume_matched_fallback'](**a))
