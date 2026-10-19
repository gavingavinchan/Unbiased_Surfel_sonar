"""Regression checks against conventional rigid transforms, not inverse cancellation."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import math
import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('geometry_sonar_utils', ROOT / 'utils/sonar_utils.py')
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)
# Use the actual renderer functions without importing optional CUDA extensions on CI.
names = {'SonarProjection', '_transform_world_points_to_sonar_frame', 'sonar_project_points'}
tree = ast.parse((ROOT / 'gaussian_renderer/__init__.py').read_text())
ns = {'torch': torch, 'dataclass': dataclass, 'get_scaled_world_to_view_transform': su.get_scaled_world_to_view_transform}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names], type_ignores=[]), 'renderer_geometry', 'exec'), ns)


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize('scale', [.65, 1., 1.7])
@pytest.mark.parametrize('with_extrinsic', [False, True])
def test_nonidentity_projection_and_backprojection_agree_with_analytic_pose(scale, with_extrinsic):
    R = rz(.43) @ ry(.52) @ rx(-.31)
    t = np.array([.3, -.2, .8])
    C = np.eye(4); C[:3, :3] = R; C[:3, 3] = t
    E = np.eye(4); E[:3, :3] = ry(-.2) @ rx(.13); E[:3, 3] = [.04, -.12, .07]
    ex = su.SonarExtrinsic(device='cpu', camera_to_sonar=E) if with_extrinsic else None
    scaled_C = C.copy(); scaled_C[:3, 3] *= scale
    expected_transform = E @ scaled_C if with_extrinsic else scaled_C
    cfg = su.SonarConfig(range_min=0., device='cpu')
    cam = SimpleNamespace(world_view_transform=torch.tensor(C.T, dtype=torch.float64), image_height=200, image_width=256)
    sf = SimpleNamespace(scale=torch.tensor(scale, dtype=torch.float64))
    row, col = 85, 125
    theta = math.pi / 3 - (col + .5) * (2 * math.pi / 3) / 256
    radius = (row + .5) * 3 / 200
    elevation = .07
    local = np.array([-radius * math.sin(theta) * math.cos(elevation), radius * math.sin(elevation), radius * math.cos(theta) * math.cos(elevation)])
    expected_world = (expected_transform[:3, :3].T @ (local - expected_transform[:3, 3])) / scale
    points = torch.tensor(expected_world[None], dtype=torch.float64)
    actual, stored = ns['_transform_world_points_to_sonar_frame'](points, cam, sf, ex)
    np.testing.assert_allclose(actual.numpy()[0], local, atol=1e-10)
    np.testing.assert_allclose(stored.T.numpy(), expected_transform, atol=1e-10)
    projection = ns['sonar_project_points'](points, cam, cfg, sf, ex)
    assert projection.col.item() == pytest.approx(col, abs=1e-8)
    assert projection.row.item() == pytest.approx(row, abs=1e-8)
    assert projection.valid.item()
    inverse = su.view_points_to_world(torch.tensor(local[None]), stored, sf)
    np.testing.assert_allclose(inverse.numpy()[0], expected_world, atol=1e-10)
    bins = su.back_project_bins(0, torch.tensor([row]), torch.tensor([col]), torch.tensor([elevation]), cameras=[cam], sonar_config=cfg, scale_factor=sf, sonar_extrinsic=ex)
    np.testing.assert_allclose(bins.numpy()[0, 0], expected_world, atol=2e-7)


def test_grid_is_200_range_rows_by_256_azimuth_columns_at_bin_centres():
    cfg = su.SonarConfig(range_min=0., device='cpu')
    assert cfg.range_mesh.shape == (200, 256)
    assert cfg.azimuth_mesh.shape == (200, 256)
    assert cfg.range_mesh[0, 0].item() == pytest.approx(.0075)
    assert cfg.range_mesh[-1, -1].item() == pytest.approx(2.9925)
    assert cfg.azimuth_mesh[0, 0].item() == pytest.approx(math.radians(59.765625))
    assert cfg.azimuth_mesh[-1, -1].item() == pytest.approx(math.radians(-59.765625))
    # Downsampling changes sample centres across the full aperture, not by slicing.
    az, radius = cfg.pixel_to_polar(torch.tensor(127.), torch.tensor(99.), image_width=128, image_height=100)
    assert az.item() == pytest.approx(math.radians(-59.53125))
    assert radius.item() == pytest.approx(2.985)


def test_provisional_default_extrinsic_is_transposed_conventional_rigid_matrix():
    mount = np.array(su.SONAR_MOUNT_TRANSLATION_CAM)
    R = rx(math.radians(su.SONAR_MOUNT_PITCH_DEG))
    expected = np.eye(4); expected[:3, :3] = R; expected[:3, 3] = -R @ mount
    np.testing.assert_allclose(su.get_camera_to_sonar_transform(device='cpu').T.numpy(), expected, atol=1e-7)


def test_extrinsic_rejects_nonrigid_matrix_and_reflection():
    for E in [np.diag([1, 1, 1, 2]), np.diag([-1, 1, 1, 1])]:
        with pytest.raises(ValueError, match='proper rigid'):
            su.SonarExtrinsic(device='cpu', camera_to_sonar=E)
