"""Frozen 176 observations; quotient VJP carried from Codex experiment 172 and independent rational-center VJP (CPU or CUDA).

No renderer import or CUDA initialization at module import. The CPU suite uses
162's native-equation oracle separately, including its autograd center chain.
"""
import ast
import json
import math
from pathlib import Path
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'tests/fixtures/screen_center_176.json').read_text())


def producers():
    names = {'_bounded_sonar_linalg', '_condition_sigma_2d', '_covariance_cholesky_2x2',
             'sigma2d_to_transmat_precomp', '_sigma2d_to_transmat_precomp_batch'}
    source = ROOT/'gaussian_renderer/__init__.py'
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns


def cases():
    for ci in range(2):
        for mi in range(2):
            for fallback in (False, True):
                for batch in (False, True):
                    yield ci, mi, fallback, batch


def pack(case, dtype, device='cpu', mean=None):
    ci, mi, fallback, batch = case
    cfg = CONFIG['mappings'][mi]
    mu = torch.tensor(cfg['mean'], dtype=dtype, device=device) if mean is None else mean
    cov = torch.tensor(CONFIG['covariances'][ci], dtype=dtype, device=device)
    ns = producers()
    fn = ns['_sigma2d_to_transmat_precomp_batch' if batch else 'sigma2d_to_transmat_precomp']
    if batch:
        mu, cov = mu[None], cov[None]
    if fallback:
        with patch.object(torch.linalg, 'cholesky', side_effect=torch.linalg.LinAlgError('176 forced fallback')):
            h = fn(mu, cov)
    else:
        h = fn(mu, cov)
    b = torch.tensor(cfg['affine'], dtype=dtype, device=device)
    return b @ h.reshape(3, 3)


def observations(case, dtype, device='cpu'):
    return (torch.tensor(CONFIG['mappings'][case[1]]['pixels'], dtype=dtype, device=device),
            torch.tensor(CONFIG['weights'], dtype=dtype, device=device))


def geometry(h, pixels):
    u, v, w = h.unbind(0)
    k, l = pixels[:, 0, None]*w-u, pixels[:, 1, None]*w-v
    p = torch.linalg.cross(k, l)
    s = p[:, :2]/p[:, 2, None]
    t = h.new_tensor([9., 9., -1.])
    denominator = (t*w*w).sum()
    center = torch.stack(((t*u*w).sum(), (t*v*w).sum()))/denominator
    rho3 = s.square().sum(-1)
    rho2 = 2*(pixels-center).square().sum(-1)
    alpha = CONFIG['opacity']*torch.exp(-.5*torch.minimum(rho3, rho2))
    return center, rho3, rho2, alpha


def independent_vjp(h, pixels, weights):
    """Closed form quotient derivative, no autograd and no CPU oracle call."""
    u, v, w = h.unbind(0)
    t = h.new_tensor([9., 9., -1.])
    d = (t*w.square()).sum()
    center = torch.stack(((t*u*w).sum(), (t*v*w).sum()))/d
    delta = center-pixels
    a = CONFIG['opacity']*torch.exp(-delta.square().sum(-1))
    bar_c = (-2*weights[:, None]*a[:, None]*delta).sum(0)
    bar_u = bar_c[0]*t*w/d
    bar_v = bar_c[1]*t*w/d
    # dc_i/dw_j = t_j * (u_ij - 2*c_i*w_j) / d
    bar_w = t*(bar_c[0]*(u-2*center[0]*w)+bar_c[1]*(v-2*center[1]*w))/d
    return torch.stack((bar_u, bar_v, bar_w))


def tile_signature(h):
    # Source equations including exact affine-only footprint predicate.
    center = geometry(h, h.new_tensor([[0., 0.]]))[0]
    t = h.new_tensor([9., 9., -1.]); w = h[2]
    f = t/(t*w.square()).sum()
    extent = (center.square()-(f*h[:2].square()).sum(-1)).clamp_min(1e-4).sqrt()
    affine = bool(torch.equal(w, h.new_tensor([0., 0., 1.])))
    cutoff = math.sqrt(2*math.log(255*CONFIG['opacity'])) if affine else 3.
    if affine:
        extent = cutoff*h[:2, :2].square().sum(-1).sqrt()
    radius = torch.maximum(extent.max(), h.new_tensor(cutoff*.707106)).ceil()+int(affine)
    lo = ((center-radius)/16).to(torch.int64).clamp(0, 2)
    hi = ((center+radius+15)/16).to(torch.int64).clamp(0, 2)
    return tuple(torch.cat((lo, hi)).tolist())


def eligible(h, pixels, signature):
    center, r3, r2, alpha = geometry(h, pixels)
    e = CONFIG['eligibility']
    assert bool(((pixels-center).square().sum(-1)>0).all())
    assert float((r3-r2).min()) >= e['rho_margin_min']
    assert float((alpha-1/255).min()) >= e['alpha_lower_margin_min']
    assert float((.99-alpha).min()) >= e['alpha_upper_margin_min']
    assert tile_signature(h) == signature
    lo = pixels.new_tensor(signature[:2])*16; hi = pixels.new_tensor(signature[2:])*16
    assert bool(((pixels >= lo) & (pixels < hi)).all())
    assert torch.isfinite(alpha).all()
    return {'rho_margin': float((r3-r2).min()), 'alpha_lower_margin': float((alpha-1/255).min()),
            'alpha_upper_margin': float((.99-alpha).min())}


def native_screen_cases(test):
    """Run every exact arm in compiled native acceptance, with zero per-arm skips."""
    # Never permit native acceptance to hide a failed frozen CPU eligibility window.
    for case in cases():
        h = pack(case, torch.float64)
        pixels, _ = observations(case, torch.float64)
        signature = tile_signature(h)
        eligible(h, pixels, signature)
        for j, eps in enumerate(CONFIG['float64']['matrix_steps']):
            delta = torch.zeros_like(h); delta.reshape(-1)[j] = eps
            eligible(h+delta, pixels, signature)
            eligible(h-delta, pixels, signature)
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    eye = torch.eye(4, device='cuda')
    settings = GaussianRasterizationSettings(image_height=32, image_width=32,
        tanfovx=1., tanfovy=1., bg=torch.zeros(3, device='cuda'), ndc2world=eye,
        scale_modifier=1., viewmatrix=eye, projmatrix=eye, sh_degree=0,
        campos=torch.zeros(3, device='cuda'), prefiltered=False, additive_mode=True, debug=False)
    raster = GaussianRasterizer(settings)
    count = 0
    for case in cases():
        with test.subTest(screen_center_case=case):
            mean = torch.tensor(CONFIG['mappings'][case[1]]['mean'], device='cuda', requires_grad=True)
            h = pack(case, torch.float32, 'cuda', mean=mean)
            pixels, weights = observations(case, torch.float64)
            ref = h.detach().cpu().double().requires_grad_()
            eligible(ref.detach(), pixels, tile_signature(ref.detach()))
            image, radii, _, _ = raster(means3D=torch.tensor([[0., 0., 1.]], device='cuda'),
                means2D=torch.zeros(1, 3, device='cuda'), opacities=torch.tensor([[CONFIG['opacity']]], device='cuda'),
                colors_precomp=torch.ones(1, 3, device='cuda'), cov3D_precomp=h.reshape(1, 9).contiguous())
            x, y = pixels.long().unbind(-1)
            actual = image[0, y.to('cuda'), x.to('cuda')]
            expected = geometry(ref, pixels)[3]
            torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-6, rtol=2e-5)
            loss = (actual*weights.to(device='cuda', dtype=torch.float32)).sum()
            gh, gm = torch.autograd.grad(loss, (h, mean), retain_graph=True)
            direct = independent_vjp(ref.detach(), pixels, weights)
            torch.testing.assert_close(gh.cpu().double(), direct, atol=3e-5, rtol=3e-4)
            ref_mean = mean.detach().cpu().double().requires_grad_()
            ref_h = pack(case, torch.float64, mean=ref_mean)
            expected_mean = torch.autograd.grad(ref_h, ref_mean, direct)[0]
            torch.testing.assert_close(gm.cpu().double(), expected_mean, atol=3e-5, rtol=3e-4)
            test.assertGreater(float(gm.norm()), 0.)
            test.assertGreater(int(radii[0]), 0)
            count += 1
    test.assertEqual(count, 16)
