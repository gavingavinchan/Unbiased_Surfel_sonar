import ast
import copy
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = REPO_ROOT / "gaussian_renderer" / "__init__.py"
DEBUG_MULTIFRAME_PATH = REPO_ROOT / "debug_multiframe.py"


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
        pytest.fail(f"Missing top-level function `{fn_name}` in {path}")

    module = ast.Module(body=[fn_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "os": os, "math": math}
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[fn_name]


def _extract_final_checkpoint_metadata_keys():
    tree = _parse_file(DEBUG_MULTIFRAME_PATH)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "save_training_checkpoint":
            continue

        metadata_kw = None
        for kw in node.keywords:
            if kw.arg == "metadata":
                metadata_kw = kw.value
                break
        if not isinstance(metadata_kw, ast.Dict):
            continue

        keys = set()
        for key_node in metadata_kw.keys:
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                keys.add(key_node.value)
        if keys:
            return keys

    pytest.fail("RB-T14: could not find final save_training_checkpoint(..., metadata={...}) payload")


def _extract_render_sonar_diagnostics_keys():
    tree = _parse_file(RENDERER_PATH)
    render_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "render_sonar":
            render_fn = node
            break
    if render_fn is None:
        pytest.fail("RB-T17: missing render_sonar function")

    for node in ast.walk(render_fn):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "sonar_diagnostics":
            continue
        if not isinstance(node.value, ast.Dict):
            continue

        keys = set()
        for key_node in node.value.keys:
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                keys.add(key_node.value)
        if keys:
            return keys

    pytest.fail("RB-T17: render_sonar does not define sonar_diagnostics={...} dictionary")


def _extract_render_sonar_diagnostics_dict_node():
    tree = _parse_file(RENDERER_PATH)
    render_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "render_sonar":
            render_fn = node
            break
    if render_fn is None:
        pytest.fail("RB-T21: missing render_sonar function")

    for node in ast.walk(render_fn):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "sonar_diagnostics" and isinstance(node.value, ast.Dict):
            return node.value
    pytest.fail("RB-T21: render_sonar sonar_diagnostics dict assignment not found")


def test_rb_t13_default_occlusion_mode_is_ray_binned_contract(monkeypatch):
    resolve = _load_top_level_function(RENDERER_PATH, "resolve_sonar_render_contract")
    monkeypatch.delenv("SONAR_OCCLUSION_MODE", raising=False)
    monkeypatch.delenv("SONAR_RENDER_MODE", raising=False)

    cfg = resolve()
    assert cfg["occlusion_mode"] == "ray_binned", (
        "RB-T13: default occlusion mode must be ray_binned for active contract safety"
    )


def test_rb_t14_checkpoint_metadata_contains_renderer_fingerprint_fields_contract():
    keys = _extract_final_checkpoint_metadata_keys()
    required = {
        "renderer_semantics_version",
        "normal_init_mode",
        "lambertian_transfer",
        "occlusion_model",
        "render_sonar_contract_hash",
        "sonar_render_mode",
        "sonar_occlusion_mode",
        "occlusion_space",
        "occlusion_footprint_policy",
        "occlusion_support_cap_config",
        "occlusion_support_cap_mass_loss",
        "compat_reference_id",
        "elevation_bin_count",
        "elevation_weight_mode",
        "surfel_size_stats_schema_version",
    }
    missing = sorted(required - keys)
    assert not missing, f"RB-T14: final checkpoint metadata missing required renderer fingerprint keys: {missing}"


def test_rb_t17_sonar_diagnostics_exposes_surfel_size_stats_contract():
    keys = _extract_render_sonar_diagnostics_keys()
    required = {
        "surfel_size_stats_schema_version",
        "surfel_size_stats_world",
        "surfel_size_stats_image",
        "surfel_size_stats_in_fov",
        "surfel_size_stats_every",
    }
    missing = sorted(required - keys)
    assert not missing, f"RB-T17: sonar diagnostics missing surfel-size telemetry keys: {missing}"


def test_rb_t17_print_diagnostics_emits_surfel_and_mass_loss_fields_contract(capsys):
    print_sonar_diagnostics = _load_top_level_function(DEBUG_MULTIFRAME_PATH, "print_sonar_diagnostics")
    diag = {
        "attenuation_gain_mode": "manual",
        "attenuation_enabled": True,
        "attenuation_effective_gain": 1.0,
        "attenuation_exp": 2.0,
        "attenuation_r0": 0.35,
        "attenuation_eps": 1e-6,
        "near_range_mean_intensity": 0.7,
        "far_range_mean_intensity": 0.2,
        "far_over_near_ratio": 0.28,
        "near_range_saturation_rate": 0.0,
        "nan_inf_count": 0,
        "surfel_size_stats_schema_version": "v1",
        "surfel_size_stats_world": {"r_eq": {"p95": 0.3}},
        "occlusion_support_cap_mass_loss": {"p99": 0.01},
    }

    print_sonar_diagnostics(diag, prefix="[rb-t17] ")
    out = capsys.readouterr().out.lower()
    assert "surfel" in out, "RB-T17: diagnostics printout must include surfel-size telemetry"
    assert "mass_loss" in out, "RB-T17: diagnostics printout must include support-cap mass-loss telemetry"


def test_rb_t12_print_diagnostics_emits_render_mode_and_occlusion_mode_contract(capsys):
    print_sonar_diagnostics = _load_top_level_function(DEBUG_MULTIFRAME_PATH, "print_sonar_diagnostics")
    diag = {
        "attenuation_gain_mode": "manual",
        "attenuation_enabled": True,
        "attenuation_effective_gain": 1.0,
        "attenuation_exp": 2.0,
        "attenuation_r0": 0.35,
        "attenuation_eps": 1e-6,
        "near_range_mean_intensity": 0.7,
        "far_range_mean_intensity": 0.2,
        "far_over_near_ratio": 0.28,
        "near_range_saturation_rate": 0.0,
        "nan_inf_count": 0,
        "sonar_render_mode": "2dgs",
        "sonar_occlusion_mode": "ray_binned",
    }

    print_sonar_diagnostics(diag, prefix="[rb-t12] ")
    out = capsys.readouterr().out.lower()
    assert "render_mode" in out, "RB-T12: diagnostics printout must include sonar render mode"
    assert "occlusion_mode" in out, "RB-T12: diagnostics printout must include sonar occlusion mode"


def test_rb_t21_mass_loss_diagnostics_are_not_hardcoded_zero_contract():
    diag_dict = _extract_render_sonar_diagnostics_dict_node()
    mass_loss_node = None
    for key_node, value_node in zip(diag_dict.keys, diag_dict.values):
        if isinstance(key_node, ast.Constant) and key_node.value == "occlusion_support_cap_mass_loss":
            mass_loss_node = value_node
            break

    assert isinstance(mass_loss_node, ast.Dict), "RB-T21: missing occlusion_support_cap_mass_loss dict in diagnostics"

    hardcoded_zero_keys = []
    for key_node, value_node in zip(mass_loss_node.keys, mass_loss_node.values):
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, (int, float)) and float(value_node.value) == 0.0:
            hardcoded_zero_keys.append(key_node.value)

    assert not hardcoded_zero_keys, (
        "RB-T21: mass-loss telemetry must be computed from support-cap statistics, "
        f"not hard-coded zeros for keys {hardcoded_zero_keys}"
    )


def test_rb_t14_checkpoint_metadata_contains_sigma_point_fields_for_nonlinear_mode_contract():
    keys = _extract_final_checkpoint_metadata_keys()
    required = {
        "sigma_point_config",
        "sigma_point_boundary_policy_version",
        "sigma_point_fallback_fraction",
    }
    missing = sorted(required - keys)
    assert not missing, (
        "RB-T14: checkpoint metadata must carry nonlinear footprint provenance fields "
        f"for 2dgs_nonlinear audits: {missing}"
    )


def test_rb_t17_surfel_stats_cadence_env_var_is_wired_contract():
    text = DEBUG_MULTIFRAME_PATH.read_text(encoding="utf-8")
    assert "SONAR_SURFEL_STATS_EVERY" in text, (
        "RB-T17: debug runtime must wire SONAR_SURFEL_STATS_EVERY for telemetry cadence control"
    )


def test_rb_t12_print_diagnostics_emits_lambertian_and_visibility_fields_contract(capsys):
    print_sonar_diagnostics = _load_top_level_function(DEBUG_MULTIFRAME_PATH, "print_sonar_diagnostics")
    diag = {
        "attenuation_gain_mode": "manual",
        "attenuation_enabled": True,
        "attenuation_effective_gain": 1.0,
        "attenuation_exp": 2.0,
        "attenuation_r0": 0.35,
        "attenuation_eps": 1e-6,
        "near_range_mean_intensity": 0.7,
        "far_range_mean_intensity": 0.2,
        "far_over_near_ratio": 0.28,
        "near_range_saturation_rate": 0.0,
        "nan_inf_count": 0,
        "lambertian_mode": "leaky",
        "visible_surfel_ratio": 0.75,
    }

    print_sonar_diagnostics(diag, prefix="[rb-t12-lambert] ")
    out = capsys.readouterr().out.lower()
    assert "lambertian" in out, "RB-T12: diagnostics printout must include lambertian mode"
    assert "visible" in out, "RB-T12: diagnostics printout must include visible surfel ratio"
