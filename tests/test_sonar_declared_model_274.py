"""274: declared image model (SONAR_ELEV_MODEL=overlap_sum, SONAR_INTENSITY_GAIN, SONAR_RASTER_FOOTPRINT_NORM=mass).

CPU tests check the elevation-overlap helpers and the fallback totals; the CUDA parity test (skipped without the
compiled sonar rasterizer) checks the training path deposits the same total as the fallback for isolated patches.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gr = pytest.importorskip("gaussian_renderer")


def test_overlap_weights_sum_to_in_aperture_coverage():
    half = math.radians(10)
    e = torch.tensor([0.0, math.radians(9.0), math.radians(-4.0)], dtype=torch.float64)
    s = torch.tensor([0.02, 0.05, 1e-3], dtype=torch.float64)
    for bins in (1, 8, 32):
        w = gr._elevation_overlap_weights(e, s, half, bins)
        cov = 0.5 * (torch.erf((half - e) / (s * math.sqrt(2))) - torch.erf((-half - e) / (s * math.sqrt(2))))
        assert torch.allclose(w.sum(dim=1), cov, atol=1e-12)


def test_elevation_sigma_of_facing_patch_is_size_over_range():
    # sonar frame right/down/forward; patch 2 m ahead, tangents along down and right, scale 0.05
    mean = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)   # tangents = x (right), y (down)
    s = gr._elevation_sigma_batch(mean, torch.tensor([[0.05, 0.05]], dtype=torch.float64), quat, min_sigma=0.0)
    assert abs(float(s[0]) - 0.025) < 1e-9


def _render(patches, bins, cuda_raster, device, monkeypatch, gain="1"):
    from utils.sonar_utils import build_debug_sonar_config
    monkeypatch.setenv("SONAR_RENDER_MODE", "2dgs")
    monkeypatch.setenv("SONAR_OCCLUSION_MODE", "ray_binned")
    monkeypatch.setenv("SONAR_ELEV_MODEL", "overlap_sum")
    monkeypatch.setenv("SONAR_RASTER_FOOTPRINT_NORM", "mass")
    monkeypatch.setenv("SONAR_INTENSITY_GAIN", gain)
    monkeypatch.setenv("SONAR_ELEV_BINS", str(bins))
    monkeypatch.setenv("SONAR_USE_CUDA_RASTERIZER", "1" if cuda_raster else "0")
    sonar = build_debug_sonar_config({"SONAR_RANGE_ORIGIN": "0", "SONAR_RANGE_SPAN": "3",
        "SONAR_PIXEL_CENTER_OFFSET": ".5", "SONAR_POSE_MODE": "poses_are_sonar"}, device=device)
    basis = np.eye(4); basis[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    cam = SimpleNamespace(image_height=200, image_width=256,
        world_view_transform=torch.as_tensor(np.linalg.inv(basis).T, device=device))
    t = lambda v: torch.tensor(v, device=device, dtype=torch.float32)
    # facing the sonar (FLU world): tangents world Z and Y, normal -X  -> quaternion of [[0,0,-1],[0,1,0],[1,0,0]]
    q = [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
    pc = SimpleNamespace(get_xyz=t([p[0] for p in patches]), get_rotation=t([q] * len(patches)),
        get_scaling=t([[p[1], p[1]] for p in patches]), get_opacity=t([[0.5]] * len(patches)))
    with torch.no_grad():
        return gr.render_sonar(cam, pc, torch.zeros(3, device=device), sonar)["render"][0].double().cpu()


@pytest.mark.parametrize("size", [0.01, 0.05])
def test_fallback_total_is_q_and_bin_independent(size, monkeypatch):
    totals = [float(_render([([2.0, 0.0, 0.0], size)], b, False, "cpu", monkeypatch).sum()) for b in (8, 16, 32)]
    q = 0.5 / (2.0 ** 2 + 1e-6)
    for t in totals:
        assert abs(t / q - 1) < 1e-3


def test_gain_is_linear(monkeypatch):
    a = _render([([2.0, 0.0, 0.0], 0.05)], 8, False, "cpu", monkeypatch, gain="1")
    b = _render([([2.0, 0.0, 0.0], 0.05)], 8, False, "cpu", monkeypatch, gain="3")
    assert torch.allclose(b, 3 * a, rtol=1e-5, atol=1e-9)


@pytest.mark.cuda
def test_cuda_parity_isolated_patches(monkeypatch):
    if not torch.cuda.is_available() or not gr._can_use_sonar_cuda_rasterizer(torch.device("cuda")):
        pytest.skip("compiled sonar CUDA rasterizer not available")
    for size in (0.01, 0.05, 0.2):
        for r in (1.0, 2.0):
            for bins in (8, 32):
                cu = float(_render([([r, 0.0, 0.0], size)], bins, True, "cuda", monkeypatch).sum())
                fb = float(_render([([r, 0.0, 0.0], size)], bins, False, "cuda", monkeypatch).sum())
                assert fb > 0 and abs(cu / fb - 1) < 0.01, (size, r, bins, cu, fb)
