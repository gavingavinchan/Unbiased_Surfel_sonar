"""bin_edges_v1 membership policy (035): versioned, independent expectations.

Policy v1: integer cell membership from explicit bin edges, evaluated in float64,
with a zero-width acceptance band; physical sample coordinates are unchanged.
These expectations are separate from the quantized_clearance_v1 tests, which
remain the default policy and are not edited.
"""
import ast
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from utils.sonar_utils import (SonarConfig, sonar_bin_membership,
                               get_scaled_world_to_view_transform)

ROOT = Path(__file__).resolve().parents[1]
functions = {'SonarProjection', '_transform_world_points_to_sonar_frame',
             'sonar_project_points', '_sonar_config_values', '_project_points_to_sonar_batch'}
ns = dict(torch=torch, math=math, dataclass=dataclass,
          get_scaled_world_to_view_transform=get_scaled_world_to_view_transform)
tree = ast.parse((ROOT / 'gaussian_renderer/__init__.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                            and n.name in functions], type_ignores=[]), 'renderer_geometry', 'exec'), ns)
project = ns['sonar_project_points']
batch_project = ns['_project_points_to_sonar_batch']
W, H = 256, 200
# Use the configuration's own float half-angles: math.radians(120)/2 and pi/3 can
# differ by one ulp, and "the" endpoint is the configured one.
HA = SonarConfig(device='cpu').half_azimuth_rad
HE = SonarConfig(device='cpu').half_elevation_rad


def alt_cfg(offset=0.5, origin=0.0, span=3.0, edges=None):
    return SonarConfig(range_origin=origin, range_span=span, pixel_center_offset=offset, device='cpu',
                       pose_mode='poses_are_sonar', membership_policy='bin_edges_v1', bin_edge_offsets=edges)


def membership(cfg, az, el, r):
    az = torch.tensor(az, dtype=torch.float64); el = torch.tensor(el, dtype=torch.float64)
    r = torch.tensor(r, dtype=torch.float64)
    col = (-az / cfg.half_azimuth_rad + 1) * (W / 2) - cfg.pixel_center_offset
    row = (r - cfg.range_min) / (cfg.range_max - cfg.range_min) * H - cfg.pixel_center_offset
    return sonar_bin_membership(az, el, r, torch.ones_like(r), col, row, cfg, W, H)


def test_default_policy_is_unchanged_and_new_policy_requires_explicit_choices():
    assert SonarConfig(device='cpu').membership_policy == 'quantized_clearance_v1'
    with pytest.raises(ValueError, match='explicit pose_mode'):
        SonarConfig(device='cpu', membership_policy='bin_edges_v1')
    with pytest.raises(ValueError, match='explicit bin_edge_offsets'):
        SonarConfig(device='cpu', pose_mode='poses_are_sonar', pixel_center_offset=0.0, membership_policy='bin_edges_v1')
    with pytest.raises(ValueError, match='membership_policy'):
        SonarConfig(device='cpu', membership_policy='nearest')
    assert alt_cfg().bin_edge_offsets == (-0.5, -0.5)


# (azimuth, elevation, range, expected valid, expected cell_col, expected cell_row)
CENTRE = [
    (HA, 0., 1.5, True, 0, 100), (-HA, 0., 1.5, True, W - 1, 100),
    (math.nextafter(HA, math.inf), 0., 1.5, False, -1, -1), (math.nextafter(-HA, -math.inf), 0., 1.5, False, -1, -1),
    (0., HE, 1.5, True, 128, 100), (0., -HE, 1.5, True, 128, 100),
    (0., math.nextafter(HE, math.inf), 1.5, False, -1, -1), (0., math.nextafter(-HE, -math.inf), 1.5, False, -1, -1),
    (0.3, 0., 0.0 + 1e-300, True, None, 0), (0.3, 0., 3.0, True, None, H - 1),
    (0.3, 0., math.nextafter(3.0, math.inf), False, -1, -1), (HA, HE, 3.0, True, 0, H - 1),
    (-HA, -HE, 3.0, True, W - 1, H - 1), (math.nextafter(HA, math.inf), HE, 3.0, False, -1, -1),
]


@pytest.mark.parametrize('az,el,r,valid,ccol,crow', CENTRE)
def test_centre_grid_closed_endpoints_zero_band(az, el, r, valid, ccol, crow):
    v, c, rr = membership(alt_cfg(), [az], [el], [r])
    assert bool(v[0]) is valid
    if ccol is not None:
        assert int(c[0]) == ccol
    assert int(rr[0]) == crow


def cells_at(cfg, col, row):
    col = torch.tensor(col, dtype=torch.float64); row = torch.tensor(row, dtype=torch.float64)
    z = torch.zeros_like(col)
    return sonar_bin_membership(z, z, torch.full_like(col, 1.5), torch.ones_like(col), col, row, cfg, W, H)


def test_interior_edges_are_half_open_and_far_endpoint_is_closed():
    cfg = alt_cfg()
    k = torch.arange(0, W + 1, dtype=torch.float64)
    v, c, _ = cells_at(cfg, (k - 0.5).tolist(), [10.] * len(k))
    assert bool(v.all())
    assert c.tolist() == list(range(W)) + [W - 1]  # edge k-0.5 opens cell k; far edge closed
    v, c, _ = cells_at(cfg, [math.nextafter(-0.5, -1), math.nextafter(W - 0.5, W)], [10., 10.])
    assert v.tolist() == [False, False]
    # One ulp below an interior edge stays in the lower cell (no rounding across the edge).
    below = [math.nextafter(kk - 0.5, -math.inf) for kk in range(1, W)]
    v, c, _ = cells_at(cfg, below, [10.] * len(below))
    assert bool(v.all()) and c.tolist() == list(range(0, W - 1))
    j = torch.arange(0, H + 1, dtype=torch.float64)
    v, _, r = cells_at(cfg, [3.] * len(j), (j - 0.5).tolist())
    assert bool(v.all()) and r.tolist() == list(range(H)) + [H - 1]


def test_archive_grid_uses_integer_azimuth_samples_and_floor_range_bins():
    cfg = alt_cfg(offset=0.0, origin=0.2, span=2.8, edges=(-0.5, 0.0))
    # +60 deg is sample col 0 (cell 0); -60 deg is col 256, beyond the last cell edge 255.5.
    v, c, r = membership(cfg, [HA, -HA, 0.], [0., 0., 0.], [1.0, 1.0, 0.2 + 73.7 * 2.8 / 200])
    assert v.tolist() == [True, False, True]
    assert int(c[0]) == 0 and int(r[2]) == 73
    v, c, r = cells_at(cfg, [255.5, math.nextafter(255.5, 256), 3.0, 3.0], [10., 10., 73.0, 200.0])
    assert v.tolist() == [True, False, True, True]
    assert int(c[0]) == W - 1 and r.tolist()[2:] == [73, H - 1]


def _camera(seed=7):
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', [17, 30, -23], degrees=True).as_matrix(); t = np.array([.31, -.22, .83])
    C = np.eye(4); C[:3, :3] = R; C[:3, 3] = t
    cam = SimpleNamespace(world_view_transform=torch.tensor(C.T, dtype=torch.float32),
                          world_view_transform_precise=torch.tensor(C.T, dtype=torch.float64),
                          image_height=H, image_width=W)
    rng = np.random.default_rng(seed)
    loc = np.stack([rng.uniform(-2, 2, 4000), rng.uniform(-.6, .6, 4000), rng.uniform(-.2, 3.2, 4000)], -1)
    return cam, torch.tensor((loc - t) @ R, dtype=torch.float32)


def test_physical_samples_are_unchanged_by_the_membership_policy():
    cam, w = _camera()
    a = project(w, cam, alt_cfg())
    b = project(w, cam, SonarConfig(range_min=0., range_max=3., device='cpu', pose_mode='poses_are_sonar'))
    assert torch.equal(a.col, b.col) and torch.equal(a.row, b.row) and torch.equal(a.range_vals, b.range_vals)
    assert a.cell_col is not None and b.cell_col is None
    inside = a.valid
    assert torch.equal(a.cell_col[inside], torch.floor(a.col[inside] + 0.5).long().clamp(0, W - 1))
    assert bool((a.cell_col[~inside] == -1).all())


def test_policies_differ_only_within_the_documented_band():
    cam, w = _camera(11)
    a = project(w, cam, alt_cfg())
    b = project(w, cam, SonarConfig(range_min=0., range_max=3., device='cpu', pose_mode='poses_are_sonar'))
    diff = a.valid != b.valid
    clearance = torch.minimum(HA - a.azimuth.abs(), torch.full_like(a.azimuth, 1.0))
    # Any disagreement must lie within half a quantum of the angular aperture.
    assert bool((clearance[diff].abs() <= 2.0 ** -20 + 1e-12).all())


def test_footprint_and_projection_agree_away_from_float32_boundaries():
    cam, w = _camera(13)
    cfg = alt_cfg()
    a = project(w, cam, cfg)
    vs, _ = ns['_transform_world_points_to_sonar_frame'](w, cam)
    fb = batch_project(vs, SimpleNamespace(image_width=W, image_height=H, half_azimuth_rad=HA, half_elevation_rad=HE,
                                          range_min=0., range_max=3., pixel_center_offset=0.5,
                                          membership_policy='bin_edges_v1', bin_edge_offsets=(-0.5, -0.5)))
    far = ((HA - a.azimuth.abs()).abs() > 1e-5) & ((a.range_vals - 3.0).abs() > 1e-5)
    el = torch.atan2(vs[:, 1].double(), torch.hypot(vs[:, 0].double(), vs[:, 2].double()))
    far &= (HE - el.abs()).abs() > 1e-5
    assert torch.equal(a.valid[far], fb['valid'][far])
