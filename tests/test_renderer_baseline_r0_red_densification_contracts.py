import ast
import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = REPO_ROOT / "gaussian_renderer" / "__init__.py"


def _load_renderer_fn(fn_name: str, extra_globals=None):
    tree = ast.parse(RENDERER_PATH.read_text(encoding="utf-8"))
    selected_nodes = [copy.deepcopy(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    mod = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(mod)

    ns = {
        "torch": torch,
        "math": math,
        "os": os,
        "dataclass": dataclass,
        "F": torch.nn.functional,
        "GaussianModel": object,
    }
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(mod, str(RENDERER_PATH), "exec"), ns)
    fn = ns.get(fn_name)
    if fn is None:
        pytest.fail(f"RB-T08: missing renderer callable `{fn_name}`")
    return fn


class _DummyPointCloud:
    def __init__(self):
        self._xyz = torch.tensor(
            [[0.00, 0.00, 1.20], [0.12, 0.00, 1.35]],
            dtype=torch.float32,
            requires_grad=True,
        )
        self._rotation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        self._opacity = torch.full((2, 1), 0.7, dtype=torch.float32, requires_grad=True)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_opacity(self):
        return self._opacity


def _identity_world_to_view(*args, **kwargs):
    return torch.eye(4, dtype=torch.float32)


def _fake_ranges_to_points(viewpoint_camera, range_image, sonar_config, scale_factor):
    h, w = range_image.shape[-2:]
    return torch.zeros((h, w, 3), dtype=range_image.dtype, device=range_image.device)


def _fake_points_to_normals(points, range_image):
    h, w = range_image.shape[-2:]
    return torch.zeros((h, w, 3), dtype=range_image.dtype, device=range_image.device)


@pytest.mark.parametrize("render_mode", ["2dgs", "2dgs_nonlinear"])
def test_rb_t08_render_sonar_emits_meaningful_densification_signals_for_both_modes_contract(monkeypatch, render_mode):
    render_sonar = _load_renderer_fn(
        "render_sonar",
        extra_globals={
            "get_scaled_world_to_view_transform": _identity_world_to_view,
            "sonar_ranges_to_points": _fake_ranges_to_points,
            "sonar_points_to_normals": _fake_points_to_normals,
        },
    )

    monkeypatch.setenv("SONAR_RENDER_MODE", render_mode)
    monkeypatch.setenv("SONAR_OCCLUSION_MODE", "none")
    monkeypatch.setenv("SONAR_LAMBERTIAN_MODE", "clamp0")

    viewpoint = SimpleNamespace(image_height=32, image_width=48)
    sonar_cfg = SimpleNamespace(
        half_azimuth_rad=math.radians(60.0),
        half_elevation_rad=math.radians(10.0),
        range_min=0.2,
        range_max=3.0,
    )
    pc = _DummyPointCloud()

    out = render_sonar(
        viewpoint_camera=viewpoint,
        pc=pc,
        bg_color=torch.zeros(3, dtype=torch.float32),
        sonar_config=sonar_cfg,
        scale_factor=None,
        sonar_extrinsic=None,
    )

    render_loss = out["render"].sum()
    render_loss.backward()

    radii = out["radii"]
    viewspace_points = out["viewspace_points"]

    assert radii.shape[0] == pc.get_xyz.shape[0]
    assert torch.isfinite(radii).all()
    assert torch.any(radii > 0), "RB-T08: radii must be meaningful (not all-zero placeholders)"

    assert viewspace_points.grad is not None
    assert torch.isfinite(viewspace_points.grad).all()
    assert (
        viewspace_points.grad.abs().sum().item() > 0.0
    ), "RB-T08: viewspace_points must carry non-zero densification gradient signal"
