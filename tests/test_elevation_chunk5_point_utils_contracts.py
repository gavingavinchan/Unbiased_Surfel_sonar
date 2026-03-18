import math
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.point_utils import sonar_ranges_to_points
from utils.sonar_utils import (
    SonarConfig,
    get_scaled_world_to_view_transform,
    sonar_polar_to_points,
    view_points_to_world,
)


def _make_identity_view(device="cpu"):
    return SimpleNamespace(world_view_transform=torch.eye(4, dtype=torch.float32, device=device))


def _make_translated_view(translation_row, device="cpu"):
    w2v = torch.eye(4, dtype=torch.float32, device=device)
    w2v[3, :3] = torch.tensor(translation_row, dtype=torch.float32, device=device)
    return SimpleNamespace(world_view_transform=w2v)


def test_c5_t02_sonar_ranges_to_points_none_elevation_matches_zero_elevation():
    sonar_config = SonarConfig(
        image_width=8,
        image_height=6,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )
    view = _make_identity_view()
    range_image = torch.full((1, 6, 8), 1.5, dtype=torch.float32)
    zero_elevation = torch.zeros((1, 6, 8), dtype=torch.float32)

    pts_none = sonar_ranges_to_points(view, range_image, sonar_config, scale_factor=None, elevation_image=None)
    pts_zero = sonar_ranges_to_points(view, range_image, sonar_config, scale_factor=None, elevation_image=zero_elevation)

    assert pts_none.shape == (6, 8, 3)
    assert torch.allclose(pts_none, pts_zero, atol=1e-6)


def test_c5_t02_sonar_ranges_to_points_explicit_elevation_changes_geometry():
    sonar_config = SonarConfig(
        image_width=8,
        image_height=6,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )
    view = _make_identity_view()
    range_image = torch.full((1, 6, 8), 2.0, dtype=torch.float32)
    elevation_image = torch.zeros((1, 6, 8), dtype=torch.float32)
    elevation_image[:, 2:4, 3:5] = math.radians(10.0)

    pts_flat = sonar_ranges_to_points(view, range_image, sonar_config, scale_factor=None, elevation_image=None)
    pts_tilt = sonar_ranges_to_points(view, range_image, sonar_config, scale_factor=None, elevation_image=elevation_image)
    cos_elev = math.cos(math.radians(10.0))
    sin_elev = math.sin(math.radians(10.0))

    assert torch.allclose(pts_flat[0, 0], pts_tilt[0, 0], atol=1e-6)
    assert pts_tilt[2, 3, 1].item() == pytest.approx(2.0 * sin_elev, abs=1e-5)
    assert pts_tilt[2, 3, 0].item() == pytest.approx(pts_flat[2, 3, 0].item() * cos_elev, abs=1e-5)
    assert pts_tilt[2, 3, 2].item() == pytest.approx(pts_flat[2, 3, 2].item() * cos_elev, abs=1e-5)
    assert not torch.allclose(pts_flat[2, 3], pts_tilt[2, 3], atol=1e-6)


def test_c5_t02_sonar_ranges_to_points_rejects_mismatched_elevation_shape():
    sonar_config = SonarConfig(
        image_width=8,
        image_height=6,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )
    view = _make_identity_view()
    range_image = torch.full((1, 6, 8), 1.0, dtype=torch.float32)
    wrong_shape = torch.zeros((1, 5, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="elevation_image shape"):
        sonar_ranges_to_points(view, range_image, sonar_config, scale_factor=None, elevation_image=wrong_shape)


def test_c5_t02_sonar_polar_to_points_preserves_sign_conventions():
    azimuth = torch.tensor([0.0, 0.2, -0.2], dtype=torch.float32)
    elevation = torch.tensor([0.1, 0.0, -0.1], dtype=torch.float32)
    range_vals = torch.ones((3,), dtype=torch.float32)

    points = sonar_polar_to_points(azimuth, elevation, range_vals)

    assert points.shape == (3, 3)
    assert points[0, 1].item() > 0.0
    assert points[1, 0].item() < 0.0
    assert points[2, 0].item() > 0.0
    assert points[2, 1].item() < 0.0


def test_c5_t02_scaled_world_to_view_roundtrip_contract():
    view = _make_translated_view([1.0, 2.0, 3.0])
    scale_factor = SimpleNamespace(scale=torch.tensor(2.0, dtype=torch.float32))

    scaled_w2v = get_scaled_world_to_view_transform(view, scale_factor=scale_factor, sonar_extrinsic=None)
    points_view = torch.tensor([[4.0, 8.0, 12.0]], dtype=torch.float32)
    points_world = view_points_to_world(points_view, scaled_w2v, scale_factor=scale_factor)

    assert torch.allclose(scaled_w2v[3, :3], torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(points_world, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), atol=1e-6)
