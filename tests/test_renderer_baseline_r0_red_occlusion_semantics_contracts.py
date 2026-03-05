import ast
import copy
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Compositing semantics follow RB-T05/RB-T06/RB-T16:
# event_return = transmittance * value, with transmittance updated by (1 - alpha).

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


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_kwargs:
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def test_rb_t20_wide_surfel_can_participate_in_multiple_rays_independently_contract():
    compose = _require_renderer_fn("compose_ray_binned_occlusion", "RB-T20")

    out = _call_with_supported_kwargs(
        compose,
        ray_ids=torch.tensor([0, 1, 0], dtype=torch.long),
        range_vals=torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
        alpha_vals=torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32),
        value_vals=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        surfel_ids=torch.tensor([42, 42, 7], dtype=torch.long),
        num_rays=2,
    )

    event_returns = out["event_returns"]
    assert event_returns[1].item() == pytest.approx(1.0, abs=1e-4), (
        "RB-T20: same surfel in ray 1 must remain independent from ray 0 occlusion"
    )
    assert event_returns[2].item() == pytest.approx(0.3, abs=1e-4)


def test_rb_t21_support_cap_preserves_weight_normalization_and_reports_mass_loss_contract():
    cap_support = _require_renderer_fn("cap_support_weights", "RB-T21")

    out = cap_support(
        weights=torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32),
        topk=2,
        floor_rel=0.0,
    )

    kept = out["weights"]
    stats = out["mass_loss"]

    assert torch.allclose(kept.sum(), torch.tensor(1.0), atol=1e-6)
    assert kept[0].item() == pytest.approx(4.0 / 7.0, abs=1e-6)
    assert kept[1].item() == pytest.approx(3.0 / 7.0, abs=1e-6)
    assert kept[2].item() == pytest.approx(0.0, abs=1e-6)
    assert kept[3].item() == pytest.approx(0.0, abs=1e-6)
    assert stats["mass_lost"].item() == pytest.approx(0.3, abs=1e-6)
    assert stats["p95"].item() >= 0.0
    assert stats["p99"].item() >= stats["p95"].item()
