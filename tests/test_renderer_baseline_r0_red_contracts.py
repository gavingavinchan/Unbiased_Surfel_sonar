import ast
import copy
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_MODEL_PATH = REPO_ROOT / "scene" / "gaussian_model.py"
RENDERER_PATH = REPO_ROOT / "gaussian_renderer" / "__init__.py"
DEBUG_MULTIFRAME_PATH = REPO_ROOT / "debug_multiframe.py"
SONAR_UTILS_PATH = REPO_ROOT / "utils" / "sonar_utils.py"


def _parse_file(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _load_top_level_function(path: Path, fn_name: str, extra_globals=None):
    tree = _parse_file(path)
    fn_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            fn_node = copy.deepcopy(node)
            break
    if fn_node is None:
        raise AssertionError(f"Could not find function {fn_name} in {path}")
    module = ast.Module(body=[fn_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "os": os, "math": math}
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[fn_name]


def _load_class_method_as_function(path: Path, class_name: str, method_name: str, extra_globals=None):
    tree = _parse_file(path)
    method_node = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                method_node = copy.deepcopy(item)
                break
    if method_node is None:
        raise AssertionError(f"Could not find {class_name}.{method_name} in {path}")
    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "np": np}
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _quat_to_normal(quaternions):
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    normal_x = 2 * (x * z + w * y)
    normal_y = 2 * (y * z - w * x)
    normal_z = 1 - 2 * (x * x + y * y)
    return torch.stack([normal_x, normal_y, normal_z], dim=-1)


def _load_sonar_utils_module():
    spec = importlib.util.spec_from_file_location("sonar_utils", SONAR_UTILS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rb_t01_create_from_pcd_uses_input_normals_for_rotation_contract():
    if not torch.cuda.is_available():
        pytest.skip("RB-T01 contract requires CUDA path used by create_from_pcd")

    normals_to_quat = _load_top_level_function(
        GAUSSIAN_MODEL_PATH,
        "_normals_to_quaternions",
        extra_globals={"F": torch.nn.functional},
    )
    create_from_pcd = _load_class_method_as_function(
        GAUSSIAN_MODEL_PATH,
        "GaussianModel",
        "create_from_pcd",
        extra_globals={
            "BasicPointCloud": object,
            "RGB2SH": lambda x: x,
            "distCUDA2": lambda x: torch.ones(x.shape[0], device=x.device, dtype=x.dtype),
            "nn": torch.nn,
            "_normals_to_quaternions": normals_to_quat,
        },
    )

    class DummyModel:
        max_sh_degree = 0

        def inverse_opacity_activation(self, x):
            return x

        @property
        def get_xyz(self):
            return self._xyz  # type: ignore[attr-defined]

    pcd = SimpleNamespace(
        points=np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 1.2]], dtype=np.float32),
        colors=np.array([[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]], dtype=np.float32),
        normals=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )

    model = DummyModel()
    create_from_pcd(model, pcd, 1.0)
    out_normals = _quat_to_normal(model._rotation.detach())  # type: ignore[attr-defined]
    expected = torch.tensor(pcd.normals, dtype=torch.float32, device=out_normals.device)
    cosine = torch.sum(out_normals * expected, dim=-1)

    assert torch.all(cosine > 0.999), "RB-T01: output quaternion normals must align with input pcd.normals"


def test_rb_t02_normal_based_rotation_is_seed_stable_contract():
    if not torch.cuda.is_available():
        pytest.skip("RB-T02 contract requires CUDA path used by create_from_pcd")

    normals_to_quat = _load_top_level_function(
        GAUSSIAN_MODEL_PATH,
        "_normals_to_quaternions",
        extra_globals={"F": torch.nn.functional},
    )
    create_from_pcd = _load_class_method_as_function(
        GAUSSIAN_MODEL_PATH,
        "GaussianModel",
        "create_from_pcd",
        extra_globals={
            "BasicPointCloud": object,
            "RGB2SH": lambda x: x,
            "distCUDA2": lambda x: torch.ones(x.shape[0], device=x.device, dtype=x.dtype),
            "nn": torch.nn,
            "_normals_to_quaternions": normals_to_quat,
        },
    )

    class DummyModel:
        max_sh_degree = 0

        def inverse_opacity_activation(self, x):
            return x

        @property
        def get_xyz(self):
            return self._xyz  # type: ignore[attr-defined]

    pcd = SimpleNamespace(
        points=np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 1.2]], dtype=np.float32),
        colors=np.array([[0.5, 0.5, 0.5], [0.7, 0.7, 0.7]], dtype=np.float32),
        normals=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )

    torch.manual_seed(42)
    m1 = DummyModel()
    create_from_pcd(m1, pcd, 1.0)
    q1 = m1._rotation.detach().clone()  # type: ignore[attr-defined]

    torch.manual_seed(42)
    m2 = DummyModel()
    create_from_pcd(m2, pcd, 1.0)
    q2 = m2._rotation.detach().clone()  # type: ignore[attr-defined]

    assert torch.allclose(q1, q2, atol=1e-7), "RB-T02: normal-driven quaternion init must be deterministic"


def test_rb_t03_leaky_lambertian_keeps_gradient_for_wrong_facing_surfel_contract():
    compute_sonar_lambertian = _load_top_level_function(RENDERER_PATH, "compute_sonar_lambertian")

    dot_leaky = torch.tensor([-0.8, -0.2], dtype=torch.float32, requires_grad=True)
    out_leaky, mode = compute_sonar_lambertian(dot_leaky, mode="leaky", alpha=0.05)
    out_leaky.sum().backward()

    assert mode == "leaky"
    assert torch.all(dot_leaky.grad > 0), "RB-T03: wrong-facing surfels must keep non-zero gradient in leaky mode"


def test_rb_t04_clamp0_matches_hard_nonnegative_transfer_contract():
    compute_sonar_lambertian = _load_top_level_function(RENDERER_PATH, "compute_sonar_lambertian")

    dots = torch.tensor([-1.2, -0.2, 0.0, 0.3, 1.4], dtype=torch.float32)
    out, mode = compute_sonar_lambertian(dots, mode="clamp0")
    expected = torch.clamp(dots, min=0.0)

    assert mode == "clamp0"
    assert torch.allclose(out, expected, atol=1e-8), "RB-T04: clamp0 mode must match prior clamp(min=0) behavior"


def test_rb_t09_prune_require_all_and_any_diverge_on_mixed_visibility_contract():
    masks = [
        torch.tensor([True, False, True], dtype=torch.bool),
        torch.tensor([False, True, False], dtype=torch.bool),
    ]

    class DummyGaussians:
        def __init__(self):
            self.get_xyz = torch.zeros((3, 3), dtype=torch.float32)
            self.get_scaling = torch.ones((3, 2), dtype=torch.float32)
            self.last_prune_mask = None

        def prune_points(self, prune_mask):
            self.last_prune_mask = prune_mask.clone()

    state = {"idx": 0}

    def fake_is_fully_in_sonar_fov(*args, **kwargs):
        out = masks[state["idx"]]
        state["idx"] += 1
        return out

    prune_outside_fov = _load_top_level_function(
        DEBUG_MULTIFRAME_PATH,
        "prune_outside_fov",
        extra_globals={
            "is_fully_in_sonar_fov": fake_is_fully_in_sonar_fov,
            "is_in_sonar_fov": fake_is_fully_in_sonar_fov,
            "ELEV_CHUNK4_CFG": {},
            "ensure_chunk4_state_capacity": lambda *args, **kwargs: None,
            "apply_row_prune_with_chunk4_state": lambda *args, **kwargs: 0,
        },
    )

    g_all = DummyGaussians()
    state["idx"] = 0
    pruned_all = prune_outside_fov(
        g_all,
        training_frames=[object(), object()],
        sonar_config=object(),
        scale_factor=object(),
        require_all=True,
        check_size=True,
    )

    g_any = DummyGaussians()
    state["idx"] = 0
    pruned_any = prune_outside_fov(
        g_any,
        training_frames=[object(), object()],
        sonar_config=object(),
        scale_factor=object(),
        require_all=False,
        check_size=True,
    )

    assert pruned_all == 3
    assert pruned_any == 0
    assert torch.equal(g_all.last_prune_mask, torch.tensor([True, True, True]))
    assert g_any.last_prune_mask is None, "RB-T09: no prune_points call expected when nothing to prune"


def test_rb_t10_forward_then_backward_projection_roundtrip_contract():
    sonar_utils = _load_sonar_utils_module()
    cfg = sonar_utils.SonarConfig(
        image_width=64,
        image_height=48,
        azimuth_fov=120.0,
        elevation_fov=20.0,
        range_min=0.2,
        range_max=3.0,
        device="cpu",
    )

    point_cam = np.array([0.35, 0.0, 1.2], dtype=np.float32)
    azimuth = -math.atan2(float(point_cam[0]), float(point_cam[2]))
    range_val = float(np.linalg.norm(point_cam))

    col, row = cfg.polar_to_pixel(
        torch.tensor([azimuth], dtype=torch.float32),
        torch.tensor([range_val], dtype=torch.float32),
    )
    col_i = int(torch.round(col).item())
    row_i = int(torch.round(row).item())
    assert 0 <= col_i < cfg.image_width
    assert 0 <= row_i < cfg.image_height

    image = torch.zeros((1, cfg.image_height, cfg.image_width), dtype=torch.float32)
    image[0, row_i, col_i] = 1.0
    camera = SimpleNamespace(
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
        original_image=image,
    )

    points_world, _ = sonar_utils.sonar_frame_to_points(
        camera,
        cfg,
        intensity_threshold=0.5,
        mask_top_rows=0,
        scale_factor=1.0,
        elevation_mode="zero",
        return_debug=False,
    )

    assert points_world.shape[0] == 1
    err = float(np.linalg.norm(points_world[0] - point_cam))
    assert err < 0.10, f"RB-T10: forward/backward projection drift too large: {err:.4f}m"


def test_rb_t22_legacy_render_mode_is_rejected_contract(monkeypatch):
    resolve_sonar_render_contract = _load_top_level_function(RENDERER_PATH, "resolve_sonar_render_contract")
    monkeypatch.setenv("SONAR_RENDER_MODE", "legacy")

    with pytest.raises(ValueError, match="legacy"):
        resolve_sonar_render_contract()
