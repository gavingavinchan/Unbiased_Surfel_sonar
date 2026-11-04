"""World-unit changes preserve physical surfel extents, images and derivatives."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_renderer_baseline_r0_red_densification_contracts import (
    _load_renderer_fn, _fake_ranges_to_points, _fake_points_to_normals,
)


def actual_scaled_transform():
    path = Path(__file__).resolve().parents[1] / 'utils/sonar_utils.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'get_scaled_world_to_view_transform')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns[node.name]


def render_units(unit_scale, dtype=torch.float64, log_shift=0., differentiate=True, use_factor=True):
    # Same two physical patches, expressed in different arbitrary world units.
    # A translated sonar exercises the real translation-scaling helper too.
    xyz = (torch.tensor([[.04,.01,1.2],[.13,-.02,1.4]],dtype=dtype)/unit_scale).requires_grad_()
    sizes = (torch.tensor([[.10,.025],[.07,.02]],dtype=dtype)/unit_scale).requires_grad_()
    quat = torch.tensor([[0.,0.,1.,0.],[0.,0.,1.,0.]],dtype=dtype,requires_grad=True)
    opacity = torch.full((2,1),.7,dtype=dtype,requires_grad=True)
    log_scale = torch.tensor(math.log(unit_scale)+log_shift,dtype=dtype,requires_grad=True)
    w2v = torch.eye(4,dtype=dtype)
    w2v[3,:3] = torch.tensor([.015,-.005,.08],dtype=dtype)/unit_scale
    camera = SimpleNamespace(image_height=32,image_width=48,world_view_transform=w2v)
    render = _load_renderer_fn('render_sonar', {
        'get_scaled_world_to_view_transform': actual_scaled_transform(),
        'sonar_ranges_to_points': _fake_ranges_to_points,
        'sonar_points_to_normals': _fake_points_to_normals,
    })
    pc = SimpleNamespace(get_xyz=xyz,get_scaling=sizes,get_rotation=quat,get_opacity=opacity)
    cfg = SimpleNamespace(half_azimuth_rad=math.radians(60),half_elevation_rad=math.radians(10),range_min=.2,range_max=3.)
    image = render(camera,pc,torch.zeros(3,dtype=dtype),cfg,scale_factor=SimpleNamespace(scale=log_scale.exp()) if use_factor else None)['render']
    if not differentiate:
        return image.detach()
    weights = torch.linspace(.5,1.5,48,dtype=dtype)
    loss = (image * weights).sum()
    grads = torch.autograd.grad(loss,(xyz,sizes,quat,opacity,log_scale))
    return image.detach(),grads[0]/unit_scale,grads[1]/unit_scale,grads[2],grads[3],grads[4],loss.detach()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mode',['2dgs','2dgs_nonlinear'])
@pytest.mark.parametrize('occlusion',['none','ray_binned'])
@pytest.mark.parametrize('unit_scale',[.5,.7,1.3,2.,3.1,4.])
def test_world_unit_change_preserves_complete_render(monkeypatch,dtype,mode,occlusion,unit_scale):
    monkeypatch.setenv('SONAR_RENDER_MODE',mode)
    monkeypatch.setenv('SONAR_OCCLUSION_MODE',occlusion)
    monkeypatch.setenv('SONAR_LAMBERTIAN_MODE','clamp0')
    monkeypatch.setenv('SONAR_ELEV_BINS','3')
    a,b = render_units(1.,dtype),render_units(unit_scale,dtype)
    assert a[0].sum() > 0
    for left,right in zip(a,b):
        assert torch.isfinite(right).all()
        torch.testing.assert_close(left,right,atol=3e-6 if dtype==torch.float32 else 2e-8,
                                   rtol=3e-5 if dtype==torch.float32 else 2e-7)


@pytest.mark.parametrize('mode',['2dgs','2dgs_nonlinear'])
@pytest.mark.parametrize('occlusion',['none','ray_binned'])
def test_metric_scale_gradient_matches_finite_difference(monkeypatch,mode,occlusion):
    monkeypatch.setenv('SONAR_RENDER_MODE',mode)
    monkeypatch.setenv('SONAR_OCCLUSION_MODE',occlusion)
    monkeypatch.setenv('SONAR_LAMBERTIAN_MODE','clamp0')
    monkeypatch.setenv('SONAR_ELEV_BINS','3')
    center = render_units(2.)
    h = 1e-6
    finite = (render_units(2.,log_shift=h)[-1]-render_units(2.,log_shift=-h)[-1])/(2*h)
    assert center[-2].abs() > 1e-5
    torch.testing.assert_close(center[-2],finite,atol=1e-6,rtol=2e-5)


@pytest.mark.parametrize('mode',['2dgs','2dgs_nonlinear'])
@pytest.mark.parametrize('occlusion',['none','ray_binned'])
def test_unit_metric_scale_preserves_no_scale_forward_path(monkeypatch,mode,occlusion):
    monkeypatch.setenv('SONAR_RENDER_MODE',mode)
    monkeypatch.setenv('SONAR_OCCLUSION_MODE',occlusion)
    monkeypatch.setenv('SONAR_LAMBERTIAN_MODE','clamp0')
    monkeypatch.setenv('SONAR_ELEV_BINS','3')
    a=render_units(1.,differentiate=False,use_factor=False)
    b=render_units(1.,differentiate=False,use_factor=True)
    torch.testing.assert_close(a,b,atol=0.,rtol=0.)
