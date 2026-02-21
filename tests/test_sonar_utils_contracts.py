import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
SONAR_UTILS_PATH = REPO_ROOT / "utils" / "sonar_utils.py"


def _load_sonar_utils_module():
    spec = importlib.util.spec_from_file_location("sonar_utils", SONAR_UTILS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sonar_utils = _load_sonar_utils_module()
SonarConfig = sonar_utils.SonarConfig
back_project_bins = sonar_utils.back_project_bins
run_sonar_convention_asserts = sonar_utils.run_sonar_convention_asserts
sonar_frame_to_points = sonar_utils.sonar_frame_to_points


def _make_camera(*, image, translation=(0.0, 0.0, 0.0)):
    w2v = torch.eye(4, dtype=torch.float32)
    w2v[3, :3] = torch.tensor(translation, dtype=torch.float32)
    return SimpleNamespace(
        world_view_transform=w2v,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
        original_image=image,
    )


def test_pixel_polar_roundtrip_is_consistent():
    cfg = SonarConfig(
        image_width=256,
        image_height=200,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )

    cols = torch.tensor([0.0, 64.25, 128.0, 191.75, 255.0], dtype=torch.float32)
    rows = torch.tensor([0.0, 50.5, 100.0, 150.25, 199.0], dtype=torch.float32)

    az, rng = cfg.pixel_to_polar(cols, rows)
    cols_rt, rows_rt = cfg.polar_to_pixel(az, rng)

    assert torch.allclose(cols_rt, cols, atol=1e-5)
    assert torch.allclose(rows_rt, rows, atol=1e-5)


def test_back_project_bins_shape_and_elevation_sign_contract():
    cfg = SonarConfig(
        image_width=128,
        image_height=64,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )
    camera = _make_camera(image=torch.zeros((1, 64, 128), dtype=torch.float32))

    rows = torch.tensor([12, 20], dtype=torch.long)
    cols = torch.tensor([40, 80], dtype=torch.long)
    elev_bins = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32)

    points = back_project_bins(
        frame_idx=0,
        rows=rows,
        cols=cols,
        elev_bins=elev_bins,
        cameras=[camera],
        sonar_config=cfg,
        scale_factor=None,
    )

    assert points.shape == (2, 3, 3)
    assert torch.isfinite(points).all()
    assert points[0, 0, 1].item() < 0.0
    assert abs(points[0, 1, 1].item()) < 1e-6
    assert points[0, 2, 1].item() > 0.0


def test_back_project_bins_rejects_invalid_inputs():
    cfg = SonarConfig(device="cpu")
    camera = _make_camera(image=torch.zeros((1, 200, 256), dtype=torch.float32))

    with pytest.raises(ValueError, match="rows and cols must be 1D"):
        back_project_bins(
            frame_idx=0,
            rows=torch.tensor([[1, 2]], dtype=torch.long),
            cols=torch.tensor([1, 2], dtype=torch.long),
            elev_bins=torch.tensor([0.0], dtype=torch.float32),
            cameras=[camera],
            sonar_config=cfg,
            scale_factor=None,
        )

    with pytest.raises(ValueError, match="rows/cols length mismatch"):
        back_project_bins(
            frame_idx=0,
            rows=torch.tensor([1, 2], dtype=torch.long),
            cols=torch.tensor([1], dtype=torch.long),
            elev_bins=torch.tensor([0.0], dtype=torch.float32),
            cameras=[camera],
            sonar_config=cfg,
            scale_factor=None,
        )


def test_sonar_frame_to_points_zero_mode_has_no_elevation_spread():
    image = torch.zeros((1, 8, 8), dtype=torch.float32)
    image[0, 3, 2] = 1.0
    image[0, 5, 6] = 0.9

    cfg = SonarConfig(
        image_width=8,
        image_height=8,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )
    camera = _make_camera(image=image)

    points, colors, debug = sonar_frame_to_points(
        camera,
        cfg,
        intensity_threshold=0.5,
        mask_top_rows=0,
        scale_factor=1.0,
        elevation_mode="zero",
        return_debug=True,
    )

    assert points.shape == (2, 3)
    assert colors.shape == (2, 3)
    assert debug["num_points"] == 2
    assert debug["elevation_min_rad"] == 0.0
    assert debug["elevation_max_rad"] == 0.0
    assert abs(debug["y_cam_min"]) < 1e-8
    assert abs(debug["y_cam_max"]) < 1e-8


def test_sonar_frame_to_points_random_mode_is_reproducible_with_seeded_rng():
    image = torch.zeros((1, 8, 8), dtype=torch.float32)
    image[0, 3, 2] = 1.0
    image[0, 4, 3] = 0.8
    image[0, 5, 6] = 0.9

    cfg = SonarConfig(
        image_width=8,
        image_height=8,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )
    camera = _make_camera(image=image)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    points1, _, debug1 = sonar_frame_to_points(
        camera,
        cfg,
        intensity_threshold=0.5,
        mask_top_rows=0,
        scale_factor=1.0,
        elevation_mode="random",
        rng=rng1,
        return_debug=True,
    )
    points2, _, debug2 = sonar_frame_to_points(
        camera,
        cfg,
        intensity_threshold=0.5,
        mask_top_rows=0,
        scale_factor=1.0,
        elevation_mode="random",
        rng=rng2,
        return_debug=True,
    )

    assert np.allclose(points1, points2)
    assert debug1["elevation_min_rad"] == pytest.approx(debug2["elevation_min_rad"])
    assert debug1["elevation_max_rad"] == pytest.approx(debug2["elevation_max_rad"])
    assert math.isfinite(debug1["y_cam_sum"])


def test_run_sonar_convention_asserts_bundle_passes_on_cpu():
    cfg = SonarConfig(device="cpu")
    camera = _make_camera(image=torch.zeros((1, cfg.image_height, cfg.image_width), dtype=torch.float32))

    report = run_sonar_convention_asserts(cfg, sample_camera=camera, device="cpu")

    assert report.azimuth_left_rad > 0.0
    assert report.azimuth_right_rad < 0.0
    assert report.positive_elevation_y > 0.0
    assert report.negative_elevation_y < 0.0
    assert report.extrinsic_roundtrip_max_abs < 1e-5
    assert report.layout_roundtrip_max_abs < 1e-5
