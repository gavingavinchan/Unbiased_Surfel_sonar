"""Screen branch CPU coverage; never count this as compiled CUDA execution."""
import pytest
import torch
import json
from pathlib import Path
from interior_screen_cases import CONFIG, cases, pack, observations, geometry, independent_vjp, eligible, tile_signature
from test_sonar_packed_factor_contract import native_backward, intensity_native


def relative(a, b):
    return float((a-b).norm()/b.norm().clamp_min(1e-20))


def score(case, dtype, strict=True):
    key = 'float32' if dtype == torch.float32 else 'float64'
    cfg = CONFIG[key]; tol = cfg['fd_relative_limit']
    mean = torch.tensor(CONFIG['mappings'][case[1]]['mean'], dtype=dtype, requires_grad=True)
    h = pack(case, dtype, mean=mean)
    pixels, weights = observations(case, dtype)
    signature = tile_signature(h.detach())
    frozen = json.loads((Path(__file__).parent/'fixtures/interior_endpoints_176.json').read_text())
    frozen = next(r for r in frozen if r['dtype']==key and r['case']==list(case))
    failures = []
    margins = []
    def check(matrix, label):
        endpoint = next(e for e in frozen['endpoints'] if e['label']==label)
        assert matrix.detach().tolist()==endpoint['matrix'], 'frozen endpoint reproduction drift'
        _, r3, r2, alpha = geometry(matrix, pixels)
        margins.append(dict(rho_margin=float((r3-r2).min()),
            alpha_lower_margin=float((alpha-1/255).min()), alpha_upper_margin=float((.99-alpha).min())))
        try:
            eligible(matrix, pixels, signature)
        except AssertionError as error:
            failures.append(dict(label=label, error=str(error) or 'eligibility assertion',
                base_tile=list(signature), actual_tile=list(tile_signature(matrix))))
    check(h.detach(), 'base')
    loss = lambda matrix: (intensity_native(matrix, pixels)*weights).sum()
    gh, gm = torch.autograd.grad(loss(h), (h, mean), retain_graph=True)
    direct = independent_vjp(h.detach(), pixels, weights)
    native = native_backward(h, pixels, weights)
    atol = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(direct, gh, atol=atol, rtol=atol)
    torch.testing.assert_close(native, gh, atol=atol, rtol=atol)
    mean_direct = torch.autograd.grad(h, mean, direct, retain_graph=True)[0]
    torch.testing.assert_close(mean_direct, gm, atol=atol, rtol=atol)
    b = torch.tensor(CONFIG['mappings'][case[1]]['affine'], dtype=dtype)
    mean_closed = b[:2,:2].T @ direct[:2,2]
    torch.testing.assert_close(mean_closed, gm, atol=atol, rtol=atol)
    torch.testing.assert_close(direct, gh, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(native, gh, atol=3e-5, rtol=3e-4)
    assert float(gm.norm()) > 0
    assert not torch.allclose(-direct, gh, atol=3e-5, rtol=3e-4), 'wrong-sign chain escaped rejection'
    finite = torch.zeros_like(h)
    for j, eps in enumerate(cfg['matrix_steps']):
        delta = torch.zeros_like(h); delta.reshape(-1)[j] = eps
        plus, minus = h.detach()+delta, h.detach()-delta
        check(plus, f'matrix_{j}_plus'); check(minus, f'matrix_{j}_minus')
        finite.reshape(-1)[j] = (loss(plus)-loss(minus))/(2*eps)
    finite_mean = torch.zeros_like(mean)
    for j in range(2):
        delta = torch.zeros_like(mean); delta[j] = cfg['mean_step']
        plus = pack(case, dtype, mean=mean.detach()+delta)
        minus = pack(case, dtype, mean=mean.detach()-delta)
        check(plus, f'mean_{j}_plus'); check(minus, f'mean_{j}_minus')
        finite_mean[j] = (loss(plus)-loss(minus))/(2*cfg['mean_step'])
    # Frozen norm-relative FD criterion; exact zeros additionally checked above.
    matrix_error, mean_error = relative(gh, finite), relative(gm, finite_mean)
    row = dict(case=list(case), dtype=key, matrix= h.detach().tolist(),
        covariance=(h[:2,:2]@h[:2,:2].T).detach().tolist(),
        matrix_gradient=gh.detach().tolist(), matrix_fd=finite.tolist(), mean_gradient=gm.detach().tolist(), mean_fd=finite_mean.tolist(),
        independent_gradient=direct.tolist(), native_equation_gradient=native.detach().tolist(),
        mean_closed_gradient=mean_closed.tolist(), mean_producer_vjp=mean_direct.tolist(),
        independent_absolute_error=(direct-gh).abs().detach().tolist(), native_absolute_error=(native-gh).abs().detach().tolist(),
        matrix_fd_absolute_error=(finite-gh).abs().detach().tolist(), mean_fd_absolute_error=(finite_mean-gm).abs().detach().tolist(),
        exact_zero_entries=[0,1,3,4], exact_zero_gradient_pass=bool((gh.reshape(-1)[[0,1,3,4]]==0).all()),
        matrix_fd_relative=matrix_error, mean_fd_relative=mean_error,
        independent_vjp_relative=relative(direct,gh), native_equation_vjp_relative=relative(native,gh),
        matrix_fd_limit_pass=matrix_error<=tol, mean_fd_limit_pass=mean_error<=tol,
        wrong_sign_max_tolerance_ratio=float((2*gh.abs()/(3e-5+3e-4*gh.abs())).max()),
        wrong_sign_rejected=True, eligibility_failures=failures, eligible=not failures, tile_signature=list(signature),
        **{k:min(m[k] for m in margins) for k in margins[0]})
    if strict:
        assert not failures, row
        assert matrix_error <= tol, row
        assert mean_error <= tol, row
    return row


@pytest.mark.parametrize('case', list(cases()))
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_screen_center_cpu(case, dtype):
    score(case, dtype)


@pytest.mark.parametrize('case', list(cases()))
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_screen_normal_fallback_agreement(case, dtype):
    ci,mi,_,batch=case
    normal, fallback = pack((ci,mi,False,batch),dtype), pack((ci,mi,True,batch),dtype)
    torch.testing.assert_close(normal,fallback,atol=2e-6 if dtype==torch.float32 else 1e-12,rtol=2e-6 if dtype==torch.float32 else 1e-12)
