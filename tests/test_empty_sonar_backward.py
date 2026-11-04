"""Hard-culling all patches leaves a constant objective with zero derivatives."""
import math
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_footprint_metric_scale import actual_scaled_transform
from test_renderer_baseline_r0_red_densification_contracts import (
    _load_renderer_fn, _fake_ranges_to_points, _fake_points_to_normals,
)

OUTPUTS = ('render', 'rend_alpha', 'rend_normal', 'rend_dist', 'surf_depth', 'surf_normal')


def render_case(case, dtype=torch.float64, trainable='all', grad_enabled=True, actual_normals=False, use_scale=True):
    xyzs = {
        'empty': [],
        'behind': [[.1, .02, -1.2], [-.1, -.02, -1.5]],
        'elevation': [[.1, 1., 1.2], [-.1, -1., 1.5]],
        'range': [[0., 0., .05], [0., 0., 4.]],
        'azimuth': [[2., 0., .3], [-2., 0., .3]],
        'visible': [[.04, .01, 1.2], [.13, -.02, 1.4]],
    }[case]
    count = len(xyzs)
    tensors = {
        'xyz': torch.tensor(xyzs, dtype=dtype).reshape(count, 3),
        'size': torch.full((count, 2), .03, dtype=dtype),
        'quat': torch.tensor([[0., 0., 1., 0.]] * count, dtype=dtype).reshape(count, 4),
        'opacity': torch.full((count, 1), .7, dtype=dtype),
        'log_scale': torch.zeros((), dtype=dtype),
        'pose': torch.eye(4, dtype=dtype),
    }
    for key, value in tensors.items():
        value.requires_grad_(trainable == 'all' or trainable == key)
    camera = SimpleNamespace(image_height=32, image_width=48,
                             world_view_transform=tensors['pose'])
    cfg = SimpleNamespace(half_azimuth_rad=math.radians(60),
                          half_elevation_rad=math.radians(10), range_min=.2, range_max=3.)
    pc = SimpleNamespace(get_xyz=tensors['xyz'], get_scaling=tensors['size'],
                         get_rotation=tensors['quat'], get_opacity=tensors['opacity'])
    cfg.azimuth_grid = torch.linspace(cfg.half_azimuth_rad, -cfg.half_azimuth_rad, camera.image_width, dtype=dtype)
    ranges_fn, normals_fn = _fake_ranges_to_points, _fake_points_to_normals
    if actual_normals:
        path = Path(__file__).resolve().parents[1] / 'utils/point_utils.py'
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in ('sonar_ranges_to_points', 'sonar_points_to_normals')]
        namespace = {'torch': torch, 'F': torch.nn.functional}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        ranges_fn, normals_fn = namespace['sonar_ranges_to_points'], namespace['sonar_points_to_normals']
    render = _load_renderer_fn('render_sonar', {
        'get_scaled_world_to_view_transform': actual_scaled_transform(),
        'sonar_ranges_to_points': ranges_fn,
        'sonar_points_to_normals': normals_fn,
    })
    with torch.set_grad_enabled(grad_enabled):
        output = render(camera, pc, torch.zeros(3, dtype=dtype), cfg,
                        scale_factor=SimpleNamespace(scale=tensors['log_scale'].exp()) if use_scale else None)
    return output, tensors


@pytest.mark.parametrize('mode', ['2dgs', '2dgs_nonlinear'])
@pytest.mark.parametrize('occlusion', ['none', 'ray_binned'])
@pytest.mark.parametrize('case', ['empty', 'behind', 'elevation', 'range', 'azimuth'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_empty_render_backward_is_zero(monkeypatch, mode, occlusion, case, dtype):
    monkeypatch.setenv('SONAR_RENDER_MODE', mode)
    monkeypatch.setenv('SONAR_OCCLUSION_MODE', occlusion)
    monkeypatch.setenv('SONAR_ELEV_BINS', '3')
    output, tensors = render_case(case, dtype)
    assert not output['visibility_filter'].any()
    # A nonzero target is the normal training trigger for this failure.
    loss = sum((output[key] - .1).square().sum() for key in OUTPUTS)
    loss.backward()
    assert output['viewspace_points'].grad is not None
    assert torch.count_nonzero(output['viewspace_points'].grad) == 0
    for key in OUTPUTS:
        assert output[key].dtype == dtype
        assert torch.count_nonzero(output[key]) == 0
    for key, tensor in tensors.items():
        assert tensor.grad is not None, key
        assert torch.isfinite(tensor.grad).all(), key
        assert torch.count_nonzero(tensor.grad) == 0, key


@pytest.mark.parametrize('trainable', ['xyz', 'size', 'quat', 'opacity', 'log_scale', 'pose'])
@pytest.mark.parametrize('case', ['empty', 'behind'])
def test_only_one_parameter_group_trainable(monkeypatch, trainable, case):
    monkeypatch.setenv('SONAR_RENDER_MODE', '2dgs')
    monkeypatch.setenv('SONAR_OCCLUSION_MODE', 'ray_binned')
    monkeypatch.setenv('SONAR_ELEV_BINS', '3')
    output, tensors = render_case(case, trainable=trainable)
    (output['render'] - .1).square().sum().backward()
    for key, tensor in tensors.items():
        if key == trainable:
            assert tensor.grad is not None
            assert torch.count_nonzero(tensor.grad) == 0
        else:
            assert tensor.grad is None


@pytest.mark.parametrize('grad_enabled', [True, False])
@pytest.mark.parametrize('case', ['empty', 'behind', 'visible'])
def test_inference_has_no_created_gradient_graph(monkeypatch, grad_enabled, case):
    monkeypatch.setenv('SONAR_RENDER_MODE', '2dgs')
    monkeypatch.setenv('SONAR_OCCLUSION_MODE', 'ray_binned')
    monkeypatch.setenv('SONAR_ELEV_BINS', '3')
    output, _ = render_case(case, trainable='none', grad_enabled=grad_enabled)
    for key in OUTPUTS:
        assert not output[key].requires_grad
        assert torch.isfinite(output[key]).all()


@pytest.mark.parametrize('mode', ['2dgs', '2dgs_nonlinear'])
@pytest.mark.parametrize('case', ['empty', 'behind'])
@pytest.mark.parametrize('use_scale', [True, False])
def test_empty_auxiliary_maps_with_real_normal_helpers(monkeypatch, mode, case, use_scale):
    monkeypatch.setenv('SONAR_RENDER_MODE', mode)
    monkeypatch.setenv('SONAR_OCCLUSION_MODE', 'ray_binned')
    monkeypatch.setenv('SONAR_ELEV_BINS', '3')
    output, tensors = render_case(case, actual_normals=True, use_scale=use_scale)
    # Include only auxiliary maps to exercise their actual backprojection and
    # finite-difference normal calculation, independent of the brightness loss.
    sum((output[key] - .1).square().sum() for key in OUTPUTS[1:]).backward()
    for key, tensor in tensors.items():
        if key == 'log_scale' and not use_scale:
            assert tensor.grad is None
        else:
            assert tensor.grad is not None, key
            assert torch.isfinite(tensor.grad).all(), key
            assert torch.count_nonzero(tensor.grad) == 0, key
