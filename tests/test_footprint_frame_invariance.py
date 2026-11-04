"""Changing world coordinates must not change a sonar image or its gradients."""
import math
from types import SimpleNamespace
import pytest
import torch
from test_renderer_baseline_r0_red_densification_contracts import (
    _load_renderer_fn, _fake_ranges_to_points, _fake_points_to_normals,
)


def quaternion_product(a, b):
    w, x, y, z = a.unbind(-1)
    v, i, j, k = b.unbind(-1)
    return torch.stack((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                        w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v), -1)


def render_fixture(mode, occlusion, angle, axis=2, dtype=torch.float64):
    q = torch.zeros(4, dtype=dtype)
    q[0] = math.cos(angle/2)
    q[axis+1] = math.sin(angle/2)
    # Independent Rodrigues rotation of the coordinate system.
    vector = torch.zeros(3, dtype=dtype); vector[axis] = 1
    vx, vy, vz = vector
    K = torch.tensor([[0., -vz, vy], [vz, 0., -vx], [-vy, vx, 0.]], dtype=dtype)
    Q = torch.eye(3, dtype=dtype) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)
    w2v = torch.eye(4, dtype=dtype); w2v[:3,:3] = Q.T
    render = _load_renderer_fn('render_sonar', {
        'get_scaled_world_to_view_transform': lambda *args, **kw: w2v,
        'sonar_ranges_to_points': _fake_ranges_to_points,
        'sonar_points_to_normals': _fake_points_to_normals,
    })
    xyz0 = torch.tensor([[.04,.01,1.2], [.13,-.02,1.4]], dtype=dtype)
    quaternion0 = torch.tensor([[0.,0.,1.,0.],[0.,0.,1.,0.]], dtype=dtype)
    xyz = (xyz0 @ Q.T).requires_grad_()
    rot = quaternion_product(q.expand(2,4), quaternion0).requires_grad_()
    scales = torch.tensor([[.10,.025],[.07,.02]], dtype=dtype, requires_grad=True)
    opacity = torch.full((2,1), .7, dtype=dtype, requires_grad=True)
    pc = SimpleNamespace(get_xyz=xyz, get_rotation=rot, get_scaling=scales, get_opacity=opacity)
    config = SimpleNamespace(half_azimuth_rad=math.radians(60),
        half_elevation_rad=math.radians(10), range_min=.2, range_max=3.)
    result = render(SimpleNamespace(image_height=32,image_width=48), pc,
                    torch.zeros(3,dtype=dtype), config)
    loss = (result['render'] * torch.linspace(.5,1.5,48,dtype=dtype)).sum()
    grads = torch.autograd.grad(loss, (xyz, scales, opacity))
    return result['render'].detach(), grads[0] @ Q, grads[1], grads[2]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('mode', ['2dgs','2dgs_nonlinear'])
@pytest.mark.parametrize('occlusion', ['none','ray_binned'])
@pytest.mark.parametrize('axis', [0,1,2])
def test_complete_render_is_independent_of_world_axes(monkeypatch, mode, occlusion, axis, dtype):
    monkeypatch.setenv('SONAR_RENDER_MODE',mode)
    monkeypatch.setenv('SONAR_OCCLUSION_MODE',occlusion)
    monkeypatch.setenv('SONAR_LAMBERTIAN_MODE','clamp0')
    monkeypatch.setenv('SONAR_ELEV_BINS','3')
    a = render_fixture(mode,occlusion,0.,axis,dtype)
    b = render_fixture(mode,occlusion,.7,axis,dtype)
    assert a[0].sum() > 0
    for left,right in zip(a,b):
        assert torch.isfinite(right).all()
        torch.testing.assert_close(left,right,atol=2e-6 if dtype == torch.float32 else 2e-8,
                                   rtol=2e-5 if dtype == torch.float32 else 2e-7)
