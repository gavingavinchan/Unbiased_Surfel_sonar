import ast
import copy
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path

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


def _make_sonar_cfg():
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


def test_rb_t15_nonlinear_mode_reports_boundary_validity_histogram_contract():
    project = _require_renderer_fn("project_sonar_footprint", "RB-T15")
    sonar_cfg = _make_sonar_cfg()

    out = project(
        mean_3d=torch.tensor([0.95, 0.12, 1.0], dtype=torch.float32),
        scale_xy=torch.tensor([0.28, 0.18], dtype=torch.float32),
        quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        mode="2dgs_nonlinear",
        sigma_point_config={"kappa": 0.0, "alpha": 1.0, "beta": 2.0},
        sonar_config=sonar_cfg,
    )

    required = {
        "sigma_valid_count",
        "sigma_invalid_reason_counts",
        "edge_band_fraction",
        "fallback_used",
    }
    missing = sorted(required - set(out.keys()))
    assert not missing, f"RB-T15: nonlinear boundary telemetry missing keys: {missing}"


def test_rb_t15_nonlinear_boundary_k_lt3_falls_back_to_jacobian_when_center_valid_contract():
    project = _require_renderer_fn("project_sonar_footprint", "RB-T15")
    sonar_cfg = _make_sonar_cfg()

    out = project(
        mean_3d=torch.tensor([1.45, 0.0, 0.65], dtype=torch.float32),
        scale_xy=torch.tensor([0.42, 0.22], dtype=torch.float32),
        quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        mode="2dgs_nonlinear",
        sigma_point_config={"kappa": 0.0, "alpha": 1.0, "beta": 2.0},
        sonar_config=sonar_cfg,
    )

    assert out.get("fallback_used", False), "RB-T15: K<3 with center valid must use Jacobian fallback"
    assert out.get("effective_mode") == "2dgs", "RB-T15: fallback path should switch effective mode to Jacobian"


def test_rb_t15_nonlinear_boundary_hysteresis_keeps_fallback_active_at_k_eq3_contract():
    project = _require_renderer_fn("project_sonar_footprint", "RB-T15")
    sonar_cfg = _make_sonar_cfg()

    out = project(
        mean_3d=torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
        scale_xy=torch.tensor([0.42, 0.22], dtype=torch.float32),
        quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        mode="2dgs_nonlinear",
        sigma_point_config={"kappa": 0.0, "alpha": 1.0, "beta": 2.0},
        sonar_config=sonar_cfg,
        previous_fallback_used=True,
    )

    assert out.get("sigma_valid_count") == 3, "RB-T15 setup: fixture must land on hysteresis boundary K=3"
    assert out.get("fallback_used", False), "RB-T15: prior fallback state must persist until K>=4"
    assert out.get("effective_mode") == "2dgs", "RB-T15: hysteresis hold should stay on Jacobian fallback"


def test_rb_t15_nonlinear_boundary_hysteresis_exits_fallback_at_k_ge4_contract():
    project = _require_renderer_fn("project_sonar_footprint", "RB-T15")
    sonar_cfg = _make_sonar_cfg()

    out = project(
        mean_3d=torch.tensor([0.95, 0.12, 1.0], dtype=torch.float32),
        scale_xy=torch.tensor([0.28, 0.18], dtype=torch.float32),
        quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        mode="2dgs_nonlinear",
        sigma_point_config={"kappa": 0.0, "alpha": 1.0, "beta": 2.0},
        sonar_config=sonar_cfg,
        previous_fallback_used=True,
    )

    assert out.get("sigma_valid_count") >= 4, "RB-T15 setup: fixture must satisfy hysteresis exit threshold"
    assert not out.get("fallback_used", True), "RB-T15: fallback should exit once K>=4 with valid center"
    assert out.get("effective_mode") == "2dgs_nonlinear", "RB-T15: hysteresis exit should restore nonlinear mode"


def test_rb_t15_nonlinear_center_invalid_skips_rendering_contract():
    project = _require_renderer_fn("project_sonar_footprint", "RB-T15")
    sonar_cfg = _make_sonar_cfg()

    out = project(
        mean_3d=torch.tensor([0.0, 0.0, -0.5], dtype=torch.float32),
        scale_xy=torch.tensor([0.2, 0.2], dtype=torch.float32),
        quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        mode="2dgs_nonlinear",
        sigma_point_config={"kappa": 0.0, "alpha": 1.0, "beta": 2.0},
        sonar_config=sonar_cfg,
    )

    assert out.get("skipped", False), "RB-T15: center-invalid surfel must be skipped, not clamped/wrapped"


def test_rb_t19_nonlinear_covariance_conditioning_clamps_eigenvalues_contract():
    project = _require_renderer_fn("project_sonar_footprint", "RB-T19")
    sonar_cfg = _make_sonar_cfg()

    out = project(
        mean_3d=torch.tensor([0.4, 0.2, 0.9], dtype=torch.float32),
        scale_xy=torch.tensor([1.2, 0.02], dtype=torch.float32),
        quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        mode="2dgs_nonlinear",
        sigma_point_config={"kappa": 0.0, "alpha": 1.0, "beta": 2.0},
        sonar_config=sonar_cfg,
    )

    sigma = out["sigma_2d"]
    eigvals = torch.linalg.eigvalsh((sigma + sigma.T) * 0.5)
    assert eigvals.min().item() >= 0.1 - 1e-4, "RB-T19: Sigma_2D eigenvalues must respect lower clamp"
    assert eigvals.max().item() <= 400.0 + 1e-3, "RB-T19: Sigma_2D eigenvalues must respect upper clamp"
