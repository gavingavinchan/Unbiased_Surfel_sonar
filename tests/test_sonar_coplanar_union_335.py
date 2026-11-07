"""335 coplanar-union helpers: grouping, counted-once weights and layer compositing (CPU, no compiled module)."""
import ast
import math
import os
from pathlib import Path
import pytest
import torch

source = Path(__file__).resolve().parents[1] / 'gaussian_renderer/__init__.py'
tree = ast.parse(source.read_text())
names = {'coplanar_groups', '_snap_edges', 'coplanar_union_weights', 'compose_coplanar_union_occlusion',
         'compose_ray_binned_occlusion', 'resolve_sonar_render_contract'}
body = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(t, 'id', '').startswith('COPLANAR_') for t in node.targets))]
assert {n.name for n in body if isinstance(n, ast.FunctionDef)} == names
ns = {'torch': torch, 'math': math, 'os': os}
exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), ns)
D = torch.float64


def plate_cells(n, center=(1.3, 0., 0.), w=.24, h=.18):
    c = torch.tensor(center, dtype=D)
    u = torch.tensor([0., 1., 0.], dtype=D); v = torch.tensor([0., 0., -1.], dtype=D)
    xs = [c + ((i + .5) / n - .5) * w * u + ((j + .5) / n - .5) * h * v for i in range(n) for j in range(n)]
    x = torch.stack(xs); m = x.shape[0]
    s = torch.tensor([[w / n / math.sqrt(2 * math.pi), h / n / math.sqrt(2 * math.pi)]], dtype=D).expand(m, 2)
    return x, torch.tensor([1., 0., 0.], dtype=D).expand(m, 3), u.expand(m, 3), v.expand(m, 3), s


def weights(x, nrm, u, v, s):
    lab = ns['coplanar_groups'](x, nrm)
    return lab, ns['coplanar_union_weights'](x, nrm, u, v, s, lab)[0]


def test_tiling_weights_one_and_duplicates_half():
    x, nrm, u, v, s = plate_cells(4)
    lab, w = weights(x, nrm, u, v, s)
    assert lab.unique().numel() == 1 and torch.allclose(w, torch.ones_like(w), atol=1e-12)
    cat = lambda t: torch.cat([t, t])
    lab, w = weights(cat(x), cat(nrm), cat(u), cat(v), cat(s))
    assert torch.allclose(w, torch.full_like(w, .5), atol=1e-12)


def test_partial_overlap_union_area_and_order_independence():
    a = plate_cells(1, (1.3, -.06, 0.)); b = plate_cells(1, (1.3, .06, 0.))
    parts = [torch.cat([p, q]) for p, q in zip(a, b)]
    lab, w = weights(*parts)
    assert torch.allclose(w, torch.full_like(w, .75), atol=1e-12)  # union .36x.18 over two .24x.18 cells
    perm = torch.tensor([1, 0])
    _, w2 = weights(*[p[perm] for p in parts])
    assert torch.allclose(w2, w[perm], atol=1e-12)


def test_distinct_layers_and_tilt_not_merged():
    x, nrm, u, v, s = plate_cells(1)
    far = x + torch.tensor([.005, 0., 0.], dtype=D)
    lab = ns['coplanar_groups'](torch.cat([x, far]), torch.cat([nrm, nrm]))
    assert lab.unique().numel() == 2
    ang = math.radians(1.)
    tilt = torch.tensor([[math.cos(ang), math.sin(ang), 0.]], dtype=D)
    lab = ns['coplanar_groups'](torch.cat([x, x]), torch.cat([nrm, tilt]))
    assert lab.unique().numel() == 2


def test_rotated_cells_use_quadrature_close_to_exact():
    x, nrm, u, v, s = plate_cells(1)
    ang = math.radians(30.)
    u2 = math.cos(ang) * u + math.sin(ang) * v; v2 = -math.sin(ang) * u + math.cos(ang) * v
    lab = ns['coplanar_groups'](torch.cat([x, x]), torch.cat([nrm, nrm]))
    w, info = ns['coplanar_union_weights'](torch.cat([x, x]), torch.cat([nrm, nrm]), torch.cat([u, u2]),
                                          torch.cat([v, v2]), torch.cat([s, s]), lab)
    assert info['quadrature_groups'] == 1 and (w < 1).all() and (w > .5).all()


def test_layer_compositing_coplanar_no_attenuation_and_opaque_front_blocks():
    ray = torch.tensor([0, 0, 0]); rng = torch.tensor([1.3, 1.31, 1.9], dtype=D)
    val = torch.tensor([1., 1., 1.], dtype=D); cov = torch.tensor([.6, .6, .5], dtype=D)
    grp = torch.tensor([0, 0, 1]); key = torch.tensor([1.3, 1.9], dtype=D)
    out = ns['compose_coplanar_union_occlusion'](ray, rng, val, cov, cov, grp, key, 1)
    assert torch.allclose(out['event_returns'], torch.tensor([1., 1., 1e-8], dtype=D), atol=1e-12)
    out = ns['compose_coplanar_union_occlusion'](ray, rng, val, cov * .5, cov, grp, key, 1)
    assert torch.allclose(out['event_returns'], torch.tensor([1., 1., .4], dtype=D), atol=1e-12)


def test_opt_in_mode(monkeypatch):
    monkeypatch.delenv('SONAR_OCCLUSION_MODE', raising=False)
    assert ns['resolve_sonar_render_contract']()['occlusion_mode'] == 'ray_binned'
    monkeypatch.setenv('SONAR_OCCLUSION_MODE', 'coplanar_union')
    assert ns['resolve_sonar_render_contract']()['occlusion_mode'] == 'coplanar_union'
