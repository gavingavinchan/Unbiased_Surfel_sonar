import ast
import copy
import importlib.util
import math
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = REPO_ROOT / "gaussian_renderer" / "__init__.py"
DEBUG_SCRIPT = REPO_ROOT / "debug_multiframe.py"
HELPER_PATH = REPO_ROOT / "utils" / "elevation_chunk5_helpers.py"


def _parse_module(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _load_function_from_ast(path: Path, fn_name: str, extra_ns=None):
    if torch is None:
        pytest.skip(f"{fn_name} behavioral contract requires torch")
    tree = _parse_module(path)
    found = any(isinstance(node, ast.FunctionDef) and node.name == fn_name for node in tree.body)
    if not found:
        return None

    selected_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            selected_nodes.append(copy.deepcopy(node))

    mod = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"torch": torch, "math": math}
    if extra_ns:
        ns.update(extra_ns)
    exec(compile(mod, str(path), "exec"), ns)
    return ns[fn_name]


def _load_helper_module():
    if torch is None:
        pytest.skip("Chunk-5 helper behavioral contract requires torch")
    spec = importlib.util.spec_from_file_location("elevation_chunk5_helpers", HELPER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_function(path: Path, fn_name: str, contract: str):
    tree = _parse_module(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return node
    pytest.fail(f"{contract}: missing function `{fn_name}` in {path.name}")


def _dict_entries(dict_node: ast.Dict):
    out = {}
    for key_node, value_node in zip(dict_node.keys, dict_node.values):
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            out[key_node.value] = value_node
    return out


def _require_last_return_dict(fn_node: ast.FunctionDef, contract: str):
    return_nodes = [
        node.value
        for node in ast.walk(fn_node)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    if not return_nodes:
        pytest.fail(f"{contract}: expected `{fn_node.name}` to return a dict")
    return max(return_nodes, key=lambda node: getattr(node, "lineno", -1))


def _extract_string_literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_render_pkg_keys(fn_node: ast.FunctionDef):
    keys = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "render_pkg":
            key = _extract_string_literal(node.slice)
            if key is not None:
                keys.add(key)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "render_pkg"
            and node.args
        ):
            key = _extract_string_literal(node.args[0])
            if key is not None:
                keys.add(key)
    return keys


def _collect_named_assignment_exprs(tree, target_name: str):
    exprs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    exprs.append(node.value)
    return exprs


def _collect_named_assignments(tree, target_name: str):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    out.append(node)
    return out


def _contains_direct_zero_constructor(expr_node):
    for node in ast.walk(expr_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"zeros", "zeros_like", "new_zeros"}:
                return True
    return False


def _is_name(node, name: str):
    return isinstance(node, ast.Name) and node.id == name


def _expr_contains_name(expr_node, name: str):
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(expr_node))


def _expr_contains_string_literal(expr_node, value: str):
    return any(isinstance(node, ast.Constant) and node.value == value for node in ast.walk(expr_node))


def _expr_contains_any_tokens(expr_node, tokens):
    token_set = set(tokens)
    for node in ast.walk(expr_node):
        if isinstance(node, ast.Name) and node.id in token_set:
            return True
        if isinstance(node, ast.Attribute) and node.attr in token_set:
            return True
        if isinstance(node, ast.Constant) and node.value in token_set:
            return True
    return False


def _if_test_mentions_names(test_node, names):
    required = set(names)
    seen = {node.id for node in ast.walk(test_node) if isinstance(node, ast.Name)}
    return required.issubset(seen)


def _find_assignments(tree, target_name: str):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    out.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == target_name:
            out.append(node.value)
    return out


def _call_references_render_pkg_key(call_node: ast.Call, key: str):
    value = call_node.func.value
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id == "render_pkg":
        return _extract_string_literal(value.slice) == key
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "get"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "render_pkg"
        and value.args
    ):
        return _extract_string_literal(value.args[0]) == key
    return False


def _function_references_attribute(fn_node: ast.FunctionDef, root_name: str, attr_name: str):
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Attribute) and node.attr == attr_name:
            if isinstance(node.value, ast.Name) and node.value.id == root_name:
                return True
    return False


def _has_quaternion_to_normal_from_gaussian_rotation(fn_node: ast.FunctionDef):
    for node in ast.walk(fn_node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "quaternion_to_normal"):
            continue
        for arg in node.args:
            for subnode in ast.walk(arg):
                if (
                    isinstance(subnode, ast.Attribute)
                    and subnode.attr == "get_rotation"
                    and isinstance(subnode.value, ast.Name)
                    and subnode.value.id == "gaussians"
                ):
                    return True
    return False


def _collect_write_string_literals(fn_node: ast.FunctionDef):
    out = []
    for node in ast.walk(fn_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            out.append(node.args[0].value)
    return out


def test_c56_t01_render_sonar_regularizer_outputs_are_not_placeholders_contract():
    fn_node = _require_function(RENDERER_PATH, "render_sonar", "C56-T01")
    return_dict = _require_last_return_dict(fn_node, "C56-T01")
    entries = _dict_entries(return_dict)

    required = {"rend_alpha", "rend_normal", "rend_dist", "surf_depth", "surf_normal"}
    missing = required.difference(entries)
    assert not missing, f"C56-T01: render_sonar missing regularizer outputs: {sorted(missing)}"

    assert not any(isinstance(node, ast.Compare) for node in ast.walk(entries["rend_alpha"])), (
        "C56-T01: rend_alpha must be accumulated opacity/support mass, not a binary occupancy threshold"
    )
    assert not _is_name(entries["rend_normal"], "surf_normal"), (
        "C56-T01: rend_normal must be a distinct accumulated surfel-orientation field, not an alias of surf_normal"
    )
    assert not _expr_contains_name(entries["surf_depth"], "range_image"), (
        "C56-T01: surf_depth must be a hard/unbiased surface depth, not the support-weighted range image placeholder"
    )
    assert not _contains_direct_zero_constructor(entries["rend_dist"]), (
        "C56-T01: rend_dist must expose a real distortion/depth-spread signal, not a zero placeholder"
    )


def test_c56_t02_compute_chunk5_normal_for_frame_reads_renderer_connected_maps_contract():
    fn_node = _require_function(DEBUG_SCRIPT, "compute_chunk5_normal_for_frame", "C56-T02")
    render_pkg_keys = _collect_render_pkg_keys(fn_node)

    required = {"rend_alpha", "rend_normal", "surf_depth", "surf_normal"}
    missing = required.difference(render_pkg_keys)
    assert not missing, (
        "C56-T02: compute_chunk5_normal_for_frame must consume renderer-connected regularizer maps "
        f"directly; missing render_pkg keys {sorted(missing)}"
    )


def test_c56_t03_compute_chunk5_normal_for_frame_no_longer_uses_sparse_geometry_helper_contract():
    fn_node = _require_function(DEBUG_SCRIPT, "compute_chunk5_normal_for_frame", "C56-T03")

    helper_calls = [
        node
        for node in ast.walk(fn_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_compute_chunk5_local_expected_geometry"
    ]
    assert not helper_calls, (
        "C56-T03: active Chunk-5.6 normal path must be rewritten away from "
        "_compute_chunk5_local_expected_geometry"
    )


def test_c56_t04_compose_ray_binned_occlusion_exposes_regularizer_state_contract():
    fn_node = _require_function(RENDERER_PATH, "compose_ray_binned_occlusion", "C56-T04")
    return_dict = _require_last_return_dict(fn_node, "C56-T04")
    entries = _dict_entries(return_dict)

    required = {
        "event_returns",
        "ray_returns",
        "final_transmittance",
        "transmittance_before_event",
        "alpha_sorted",
        "range_sorted",
        "ray_ids_sorted",
        "segment_starts",
        "event_sort_order",
    }
    missing = required.difference(entries)
    assert not missing, (
        "C56-T04: compose_ray_binned_occlusion must expose compositing state needed for "
        f"rend_normal / hard-depth / rend_dist derivation; missing keys {sorted(missing)}"
    )


def test_c56_t05_sonar_fixed_opacity_default_is_not_enabled_for_chunk56_contract():
    tree = _parse_module(DEBUG_SCRIPT)
    assignments = _find_assignments(tree, "SONAR_FIXED_OPACITY")
    assert assignments, "C56-T05: expected a SONAR_FIXED_OPACITY assignment in debug_multiframe.py"

    matched_env_bool = False
    for value in assignments:
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "env_bool"
            and len(value.args) >= 2
            and _extract_string_literal(value.args[0]) == "SONAR_FIXED_OPACITY"
        ):
            matched_env_bool = True
            default_value = value.args[1]
            assert isinstance(default_value, ast.Constant) and default_value.value is False, (
                "C56-T05: Chunk-5.6 validation must not default SONAR_FIXED_OPACITY to true"
            )

    assert matched_env_bool, "C56-T05: expected SONAR_FIXED_OPACITY to be configured via env_bool"


def test_c56_t06_compute_chunk5_normal_for_frame_does_not_source_normals_from_gaussians_contract():
    fn_node = _require_function(DEBUG_SCRIPT, "compute_chunk5_normal_for_frame", "C56-T06")
    assert not _has_quaternion_to_normal_from_gaussian_rotation(fn_node), (
        "C56-T06: active Chunk-5.6 normal path must use renderer-produced rend_normal, "
        "not direct quaternion_to_normal(gaussians.get_rotation...)"
    )


def test_c56_t07_compute_chunk5_normal_for_frame_detaches_rend_alpha_for_comparison_contract():
    fn_node = _require_function(DEBUG_SCRIPT, "compute_chunk5_normal_for_frame", "C56-T07")

    matching_detach_calls = [
        node
        for node in ast.walk(fn_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "detach"
        and _call_references_render_pkg_key(node, "rend_alpha")
    ]
    assert matching_detach_calls, (
        "C56-T07: compute_chunk5_normal_for_frame must detach render_pkg['rend_alpha'] "
        "before comparing rend_normal against surf_normal"
    )


def test_c56_t08_compute_chunk5_normal_for_frame_no_longer_gates_on_stage1_posteriors_contract():
    fn_node = _require_function(DEBUG_SCRIPT, "compute_chunk5_normal_for_frame", "C56-T08")

    blocking_ifs = []
    for node in ast.walk(fn_node):
        if isinstance(node, ast.If):
            if _if_test_mentions_names(node.test, {"p_post_frame", "support_mask_frame"}):
                if any(isinstance(stmt, ast.Return) for stmt in node.body):
                    blocking_ifs.append(ast.dump(node.test))
    assert not blocking_ifs, (
        "C56-T08: active Chunk-5.6 normal path must not early-return when Stage-1 posterior tensors are absent; "
        f"found gates {blocking_ifs}"
    )


def test_c56_t09_chunk5_diag_summary_includes_renderer_connected_coverage_and_depth_stats_contract():
    fn_node = _require_function(DEBUG_SCRIPT, "summarize_chunk5_batch_diagnostics", "C56-T09")
    return_dict = _require_last_return_dict(fn_node, "C56-T09")
    entries = _dict_entries(return_dict)

    required = {
        "rend_normal_valid_count",
        "rend_normal_valid_frac",
        "surf_depth_valid_count",
        "surf_depth_valid_frac",
        "surf_normal_valid_count",
        "surf_normal_valid_frac",
        "rend_alpha_mean",
        "surf_depth_mean",
        "rend_dist_mean",
    }
    missing = required.difference(entries)
    assert not missing, (
        "C56-T09: Chunk-5.6 diagnostics summary must include renderer-connected normal/depth coverage and stats; "
        f"missing keys {sorted(missing)}"
    )


def test_c56_t10_chunk5_diag_csv_header_includes_normal_depth_and_opacity_contract():
    fn_node = _require_function(DEBUG_SCRIPT, "init_chunk5_diag_log", "C56-T10")
    header_text = "\n".join(_collect_write_string_literals(fn_node))
    assert header_text, "C56-T10: expected init_chunk5_diag_log to write a CSV header"

    required_columns = {
        "normal_mode",
        "w_normal_mean",
        "rend_normal_valid_frac",
        "surf_normal_valid_frac",
        "surf_depth_valid_frac",
        "rend_alpha_mean",
        "surf_depth_mean",
        "rend_dist_mean",
        "opacity_fixed",
        "opacity_grad_enabled",
    }
    missing = {col for col in required_columns if col not in header_text}
    assert not missing, (
        "C56-T10: Chunk-5.6 CSV diagnostics must log normal/depth/opacity contract fields; "
        f"missing columns {sorted(missing)}"
    )


def test_c56_t11_debug_multiframe_consumes_renderer_distortion_map_contract():
    tree = _parse_module(DEBUG_SCRIPT)
    render_pkg_keys = _collect_render_pkg_keys(tree)
    assert "rend_dist" in render_pkg_keys, (
        "C56-T11: debug_multiframe.py must consume render_pkg['rend_dist'] for the restored depth regularizer path"
    )


def test_c56_t12_stage_loss_assignments_include_depth_term_contract():
    tree = _parse_module(DEBUG_SCRIPT)
    loss_exprs = _collect_named_assignment_exprs(tree, "loss_i")
    assert loss_exprs, "C56-T12: expected at least one `loss_i` assignment in debug_multiframe.py"

    depth_loss_exprs = [
        expr for expr in loss_exprs
        if _expr_contains_any_tokens(expr, {"loss_depth_i", "depth_term_i", "rend_dist", "surf_depth"})
    ]
    assert len(depth_loss_exprs) >= 2, (
        "C56-T12: Stage-2 and Stage-3 unified loss assignments must each include a depth term when the "
        "Chunk-5.6 depth rollout lands"
    )


def test_c56_t13_compose_ray_binned_occlusion_reports_correct_transmittance_state_contract():
    compose = _load_function_from_ast(RENDERER_PATH, "compose_ray_binned_occlusion")
    assert compose is not None, "C56-T13: missing compose_ray_binned_occlusion"

    out = compose(
        ray_ids=torch.tensor([0, 0, 0], dtype=torch.long),
        range_vals=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        alpha_vals=torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32),
        value_vals=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        num_rays=1,
    )

    for key in {
        "transmittance_before_event",
        "alpha_sorted",
        "range_sorted",
        "ray_ids_sorted",
        "segment_starts",
        "event_sort_order",
    }:
        assert key in out, f"C56-T13: compose output missing `{key}`"

    assert torch.allclose(
        out["transmittance_before_event"],
        torch.tensor([1.0, 0.8, 0.56], dtype=torch.float32),
        atol=1e-6,
    ), "C56-T13: transmittance-before-event must match cumulative occlusion math"
    assert torch.allclose(
        out["alpha_sorted"],
        torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32),
        atol=1e-6,
    )
    assert torch.allclose(
        out["range_sorted"],
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        atol=1e-6,
    )
    assert out["ray_ids_sorted"].tolist() == [0, 0, 0]
    assert out["segment_starts"].tolist() == [0]
    assert out["event_sort_order"].tolist() == [0, 1, 2]


def test_c56_t14_apply_opacity_policy_enables_learnable_opacity_contract():
    apply_opacity_policy = _load_function_from_ast(
        DEBUG_SCRIPT,
        "apply_opacity_policy",
        extra_ns={
            "inverse_sigmoid": lambda x: torch.log(x / (1.0 - x).clamp_min(1e-8)),
            "FIXED_OPACITY_TARGET": 0.999,
            "GAUSSIAN_OPACITY_LR": 0.05,
        },
    )
    assert apply_opacity_policy is not None, "C56-T14: missing apply_opacity_policy"

    opacity = torch.nn.Parameter(torch.zeros((4, 1), dtype=torch.float32), requires_grad=False)
    optimizer = SimpleNamespace(param_groups=[{"name": "opacity", "lr": 0.0}])
    gaussians = SimpleNamespace(_opacity=opacity, optimizer=optimizer)

    apply_opacity_policy(gaussians, fixed_opacity=False, learnable_opacity_lr=0.123)

    assert gaussians._opacity.requires_grad is True, (
        "C56-T14: learnable-opacity mode must enable gradients on gaussians._opacity"
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.123), (
        "C56-T14: learnable-opacity mode must restore a nonzero opacity optimizer LR"
    )


def test_c56_t15_mode_gating_helper_contracts_for_off_shadow_active():
    chunk5 = _load_helper_module()

    assert chunk5.mode_enables_normal_loss("off") is False
    assert chunk5.mode_enables_normal_loss("shadow") is False
    assert chunk5.mode_enables_normal_loss("active") is True

    assert chunk5.resolve_effective_chunk5_modes(
        elevation_aware=False,
        requested_normal_mode="active",
        densify_enabled=True,
        requested_densify_mode="active",
    ) == ("off", "off")


def test_c56_t16_stage2_and_stage3_normal_term_alignment_contract():
    tree = _parse_module(DEBUG_SCRIPT)
    normal_term_assignments = _collect_named_assignments(tree, "normal_term_i")
    assert len(normal_term_assignments) >= 2, (
        "C56-T16: expected Stage-2 and Stage-3 normal_term_i assignments in debug_multiframe.py"
    )

    matching = []
    for assign in normal_term_assignments:
        expr = assign.value
        if not isinstance(expr, ast.IfExp):
            continue
        if not _expr_contains_name(expr.test, "normal_loss_enabled"):
            continue
        if not (_expr_contains_name(expr.body, "normal_weight_iter") and _expr_contains_name(expr.body, "loss_normal_i")):
            continue
        matching.append(assign)

    assert len(matching) >= 2, (
        "C56-T16: Stage-2 and Stage-3 must gate normal_term_i the same way with normal_loss_enabled, "
        "normal_weight_iter, and loss_normal_i"
    )


def test_c56_t17_stage2_and_stage3_loss_include_normal_term_once_contract():
    tree = _parse_module(DEBUG_SCRIPT)
    loss_assignments = _collect_named_assignments(tree, "loss_i")
    assert len(loss_assignments) >= 2, "C56-T17: expected Stage-2 and Stage-3 loss_i assignments"

    matching = []
    for assign in loss_assignments:
        count = sum(
            1
            for node in ast.walk(assign.value)
            if isinstance(node, ast.Name) and node.id == "normal_term_i"
        )
        if count == 1:
            matching.append(assign)

    assert len(matching) >= 2, (
        "C56-T17: Stage-2 and Stage-3 unified loss expressions must include normal_term_i exactly once"
    )


def test_c56_t18_synthetic_surface_diagnostics_sphere_behavior_contract():
    fn = _load_function_from_ast(
        DEBUG_SCRIPT,
        "compute_synthetic_surface_diagnostics_from_arrays",
        extra_ns={"np": np},
    )

    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.02],
        ],
        dtype=np.float64,
    )
    normals = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    opacity = np.asarray([0.95, 0.85, 0.40], dtype=np.float64)
    geometry = {"shape": "sphere", "sphere_center_m": [0.0, 0.0, 0.0], "sphere_radius_m": 1.0}

    diag = fn(
        points,
        normals,
        opacity,
        geometry,
        near_surface_thresh_m=0.05,
        high_opacity_thresh=0.80,
        orientation_good_cos=0.95,
    )

    assert diag["shape"] == "sphere"
    assert diag["near_surface"]["count"] == 3
    assert diag["near_surface"]["high_opacity_frac"] == pytest.approx(2.0 / 3.0)
    assert diag["near_surface"]["good_orientation_frac"] == pytest.approx(1.0)
    assert diag["near_surface"]["orientation_abs_cos"]["mean"] == pytest.approx(1.0)


def test_c56_t19_synthetic_surface_diagnostics_cube_behavior_contract():
    fn = _load_function_from_ast(
        DEBUG_SCRIPT,
        "compute_synthetic_surface_diagnostics_from_arrays",
        extra_ns={"np": np},
    )

    points = np.asarray(
        [
            [1.02, 0.10, 0.00],
            [-1.01, 0.00, 0.10],
            [0.00, 0.00, 1.03],
            [0.00, 0.00, 0.20],
        ],
        dtype=np.float64,
    )
    normals = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    opacity = np.asarray([0.92, 0.88, 0.30, 0.10], dtype=np.float64)
    geometry = {"shape": "cube", "cube_center_m": [0.0, 0.0, 0.0], "cube_half_extent_m": 1.0}

    diag = fn(
        points,
        normals,
        opacity,
        geometry,
        near_surface_thresh_m=0.05,
        high_opacity_thresh=0.80,
        orientation_good_cos=0.95,
    )

    assert diag["shape"] == "cube"
    assert diag["near_surface"]["count"] == 3
    assert diag["near_surface"]["high_opacity_frac"] == pytest.approx(2.0 / 3.0)
    assert diag["near_surface"]["good_orientation_frac"] == pytest.approx(1.0)
    assert diag["near_surface"]["high_opacity_majority"] is True


def test_c56_t20_visualizer_exports_include_synthetic_surface_diagnostics_contract():
    build_refs = _require_function(DEBUG_SCRIPT, "build_visualizer_metadata_refs", "C56-T20")
    export_stage = _require_function(DEBUG_SCRIPT, "export_stage_visualizer_state", "C56-T20")

    assert _expr_contains_string_literal(build_refs, "synthetic_surface_diagnostics"), (
        "C56-T20: build_visualizer_metadata_refs must expose synthetic_surface_diagnostics.json when present"
    )
    assert _expr_contains_string_literal(export_stage, "synthetic_surface_diagnostics"), (
        "C56-T20: export_stage_visualizer_state must carry per-stage synthetic surface diagnostics into the visualizer artifact"
    )
