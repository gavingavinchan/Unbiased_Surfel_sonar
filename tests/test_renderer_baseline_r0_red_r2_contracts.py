"""R2 ray-occlusion contract tests.

Compositing contract used by RB-T05/RB-T06/RB-T16:
- Sort events by increasing sonar range per ray.
- Event return uses transmittance-only scaling: `event_return = T * value`.
- Transmittance update is alpha-gated: `T <- T * (1 - alpha)`.

This is intentionally not standard RGB alpha compositing (`alpha * T * value`).
"""

import ast
import copy
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = REPO_ROOT / "gaussian_renderer" / "__init__.py"


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
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"torch": torch, "math": math, "os": os, "dataclass": dataclass, "GaussianModel": object}
    exec(compile(module, str(RENDERER_PATH), "exec"), ns)
    return ns[fn_name]


def _require_renderer_fn(fn_name: str, contract: str):
    fn = _load_renderer_fn(fn_name)
    if fn is None:
        pytest.fail(f"{contract}: missing renderer callable `{fn_name}`")
    return fn


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_kwargs:
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _expected_event_returns(range_vals, alpha_vals, value_vals, ray_ids, num_rays):
    event_returns = torch.zeros_like(value_vals)
    trans = torch.ones(num_rays, dtype=value_vals.dtype)
    order = torch.argsort(range_vals)
    for idx in order.tolist():
        ray = int(ray_ids[idx].item())
        event_returns[idx] = trans[ray] * value_vals[idx]
        trans[ray] = trans[ray] * (1.0 - alpha_vals[idx])
    return event_returns


def test_rb_t05_same_ray_front_to_back_occlusion_contract():
    compose = _require_renderer_fn(
        "compose_ray_binned_occlusion",
        "RB-T05",
    )

    out = _call_with_supported_kwargs(
        compose,
        ray_ids=torch.tensor([0, 0], dtype=torch.long),
        range_vals=torch.tensor([1.0, 2.0], dtype=torch.float32),
        alpha_vals=torch.tensor([0.7, 0.7], dtype=torch.float32),
        value_vals=torch.tensor([1.0, 1.0], dtype=torch.float32),
        num_rays=1,
    )

    event_returns = out["event_returns"]
    expected = _expected_event_returns(
        range_vals=torch.tensor([1.0, 2.0], dtype=torch.float32),
        alpha_vals=torch.tensor([0.7, 0.7], dtype=torch.float32),
        value_vals=torch.tensor([1.0, 1.0], dtype=torch.float32),
        ray_ids=torch.tensor([0, 0], dtype=torch.long),
        num_rays=1,
    )
    assert event_returns.shape[0] == 2
    assert event_returns[0].item() > event_returns[1].item(), "RB-T05: near event must suppress far event on same ray"
    assert torch.allclose(event_returns, expected, atol=1e-6)


def test_rb_t06_different_rays_do_not_occlude_each_other_contract():
    compose = _require_renderer_fn(
        "compose_ray_binned_occlusion",
        "RB-T06",
    )

    out = _call_with_supported_kwargs(
        compose,
        ray_ids=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        range_vals=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32),
        alpha_vals=torch.tensor([0.8, 0.8, 0.8, 0.8], dtype=torch.float32),
        value_vals=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        num_rays=2,
    )

    event_returns = out["event_returns"]
    expected = _expected_event_returns(
        range_vals=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32),
        alpha_vals=torch.tensor([0.8, 0.8, 0.8, 0.8], dtype=torch.float32),
        value_vals=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        ray_ids=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        num_rays=2,
    )
    assert event_returns.shape[0] == 4
    assert torch.allclose(event_returns, expected, atol=1e-6)


def test_rb_t16_occlusion_order_uses_sonar_range_not_camera_z_contract():
    compose = _require_renderer_fn(
        "compose_ray_binned_occlusion",
        "RB-T16",
    )

    out = _call_with_supported_kwargs(
        compose,
        ray_ids=torch.tensor([0, 0], dtype=torch.long),
        range_vals=torch.tensor([2.0, 1.0], dtype=torch.float32),
        camera_z_vals=torch.tensor([1.0, 2.0], dtype=torch.float32),
        alpha_vals=torch.tensor([0.6, 0.6], dtype=torch.float32),
        value_vals=torch.tensor([0.2, 1.0], dtype=torch.float32),
        num_rays=1,
    )

    ray_returns = out["ray_returns"]
    expected_event_returns = _expected_event_returns(
        range_vals=torch.tensor([2.0, 1.0], dtype=torch.float32),
        alpha_vals=torch.tensor([0.6, 0.6], dtype=torch.float32),
        value_vals=torch.tensor([0.2, 1.0], dtype=torch.float32),
        ray_ids=torch.tensor([0, 0], dtype=torch.long),
        num_rays=1,
    )
    assert ray_returns.shape[0] == 1
    assert ray_returns[0].item() == pytest.approx(expected_event_returns.sum().item(), abs=1e-6), (
        "RB-T16: compositor must sort by sonar range for same-ray occlusion"
    )


def test_rb_t18_elevation_marginalization_matches_weighted_sum_contract():
    marginalize = _require_renderer_fn(
        "marginalize_elevation_bins",
        "RB-T18",
    )

    returns_aer = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    elev_weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    out = _call_with_supported_kwargs(marginalize, returns_aer=returns_aer, elev_weights=elev_weights)
    expected = torch.einsum("aer,e->ar", returns_aer, elev_weights)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-6), "RB-T18: marginalization must satisfy I[a,r] = sum_e w_e*R[a,e,r]"
