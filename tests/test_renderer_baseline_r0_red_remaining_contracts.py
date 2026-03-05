import ast
import copy
import importlib.util
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = REPO_ROOT / "gaussian_renderer" / "__init__.py"
SONAR_UTILS_PATH = REPO_ROOT / "utils" / "sonar_utils.py"


def _parse_renderer():
    return ast.parse(RENDERER_PATH.read_text(encoding="utf-8"))


def _load_renderer_fn(fn_name: str):
    tree = _parse_renderer()
    found = any(isinstance(node, ast.FunctionDef) and node.name == fn_name for node in tree.body)
    if not found:
        return None
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            selected_nodes.append(copy.deepcopy(node))
    mod = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"torch": torch, "math": math, "os": os, "dataclass": dataclass, "GaussianModel": object}
    exec(compile(mod, str(RENDERER_PATH), "exec"), ns)
    return ns[fn_name]


def _require_renderer_fn(fn_name: str, contract: str):
    fn = _load_renderer_fn(fn_name)
    if fn is None:
        pytest.fail(f"{contract}: missing renderer callable `{fn_name}`")
    return fn


def _load_sonar_utils_module():
    spec = importlib.util.spec_from_file_location("sonar_utils", SONAR_UTILS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_sonar_config():
    sonar_utils = _load_sonar_utils_module()
    return sonar_utils.SonarConfig(
        image_width=256,
        image_height=200,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_kwargs:
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _quat_to_rotation_matrix(quat_wxyz):
    q = torch.nn.functional.normalize(quat_wxyz, dim=0)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ],
        dim=0,
    )


def _axis_angle_to_quat(axis, angle_rad: float):
    axis_n = axis / axis.norm().clamp_min(1e-8)
    half = 0.5 * angle_rad
    return torch.cat([
        torch.tensor([math.cos(half)], dtype=torch.float32),
        axis_n * math.sin(half),
    ])


def _surfel_disc_points_world(
    mean_3d,
    quat_wxyz,
    scale_xy,
    samples_per_axis: int = 31,
):
    coords = torch.linspace(-1.0, 1.0, samples_per_axis, dtype=torch.float32)
    uu, vv = torch.meshgrid(coords, coords, indexing="xy")
    disc_mask = (uu * uu + vv * vv) <= 1.0
    u = uu[disc_mask]
    v = vv[disc_mask]

    local = torch.stack([u * scale_xy[0], v * scale_xy[1], torch.zeros_like(u)], dim=-1)
    rot = _quat_to_rotation_matrix(quat_wxyz)
    world = local @ rot.T + mean_3d.unsqueeze(0)
    return world


def _project_points_to_sonar_pixels(points_world, sonar_cfg):
    x = points_world[:, 0]
    y = points_world[:, 1]
    z = points_world[:, 2]

    azimuth = -torch.atan2(x, z)
    range_vals = torch.sqrt(x * x + y * y + z * z)
    col, row = sonar_cfg.polar_to_pixel(azimuth, range_vals)

    valid = (
        (torch.abs(azimuth) <= sonar_cfg.half_azimuth_rad)
        & (range_vals >= sonar_cfg.range_min)
        & (range_vals <= sonar_cfg.range_max)
        & (col >= 0)
        & (col <= sonar_cfg.image_width - 1)
        & (row >= 0)
        & (row <= sonar_cfg.image_height - 1)
        & (z > 0)
    )
    pts_2d = torch.stack([col[valid], row[valid]], dim=-1)
    return pts_2d.detach().cpu().numpy()


def _oracle_covariance_from_points(points_2d: np.ndarray) -> np.ndarray:
    mu = points_2d.mean(axis=0)
    centered = points_2d - mu
    cov = centered.T @ centered / max(points_2d.shape[0] - 1, 1)
    return cov


def test_rb_t07_footprint_support_grows_with_surfel_scale_contract():
    project_footprint = _require_renderer_fn("project_sonar_footprint", "RB-T07")
    sonar_cfg = _make_sonar_config()

    mean_3d = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float32)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    small = project_footprint(
        mean_3d=mean_3d,
        scale_xy=torch.tensor([0.02, 0.02], dtype=torch.float32),
        quat_wxyz=quat,
        mode="2dgs",
        sonar_config=sonar_cfg,
    )
    large = project_footprint(
        mean_3d=mean_3d,
        scale_xy=torch.tensor([0.12, 0.12], dtype=torch.float32),
        quat_wxyz=quat,
        mode="2dgs",
        sonar_config=sonar_cfg,
    )

    det_small = torch.det(small["sigma_2d"]).item()
    det_large = torch.det(large["sigma_2d"]).item()
    assert det_large > det_small * 4.0, "RB-T07: larger surfel scale must increase image-space support"


def test_rb_t15_sigma_point_mode_is_closer_to_nonlinear_oracle_than_jacobian_contract():
    project_footprint = _require_renderer_fn("project_sonar_footprint", "RB-T15")
    sonar_cfg = _make_sonar_config()

    mean_3d = torch.tensor([0.65, 0.18, 1.45], dtype=torch.float32)
    quat = _axis_angle_to_quat(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32), math.radians(50.0))
    scale_xy = torch.tensor([0.32, 0.10], dtype=torch.float32)

    jac = _call_with_supported_kwargs(
        project_footprint,
        mean_3d=mean_3d,
        scale_xy=scale_xy,
        quat_wxyz=quat,
        mode="2dgs",
        sonar_config=sonar_cfg,
    )
    nonlin = _call_with_supported_kwargs(
        project_footprint,
        mean_3d=mean_3d,
        scale_xy=scale_xy,
        quat_wxyz=quat,
        mode="2dgs_nonlinear",
        sigma_point_config={"kappa": 0.0, "alpha": 1.0, "beta": 2.0},
        sonar_config=sonar_cfg,
    )

    disc_points = _surfel_disc_points_world(mean_3d, quat, scale_xy, samples_per_axis=35)
    oracle_pixels = _project_points_to_sonar_pixels(disc_points, sonar_cfg)
    assert oracle_pixels.shape[0] >= 20, "RB-T15 setup: insufficient projected samples for robust oracle covariance"
    oracle_cov = _oracle_covariance_from_points(oracle_pixels)

    jac_err = float(np.linalg.norm(jac["sigma_2d"].detach().cpu().numpy() - oracle_cov))
    nonlin_err = float(np.linalg.norm(nonlin["sigma_2d"].detach().cpu().numpy() - oracle_cov))
    assert nonlin_err < jac_err, "RB-T15: nonlinear mode should match nonlinear projection oracle better than Jacobian"


def test_rb_t19_sigma2d_to_t_roundtrip_and_gradient_contract():
    sigma2d_to_t = _require_renderer_fn("sigma2d_to_transmat_precomp", "RB-T19")
    t_to_sigma2d = _require_renderer_fn("transmat_precomp_to_sigma2d", "RB-T19")

    mu_2d = torch.tensor([12.0, 34.0], dtype=torch.float32, requires_grad=True)
    sigma_2d = torch.tensor(
        [[2.5, 0.4], [0.4, 1.8]],
        dtype=torch.float32,
        requires_grad=True,
    )

    t = sigma2d_to_t(mu_2d=mu_2d, sigma_2d=sigma_2d)
    sigma_rt = t_to_sigma2d(t)
    assert sigma_rt.shape == (2, 2)
    assert torch.allclose(sigma_rt, sigma_2d, atol=1e-4), "RB-T19: Sigma_2D <-> T conversion must roundtrip"

    loss = t.sum()
    loss.backward()
    assert sigma_2d.grad is not None
    assert torch.isfinite(sigma_2d.grad).all()
    assert sigma_2d.grad.abs().sum().item() > 0.0, "RB-T19: gradients must flow through Sigma_2D->T conversion"
