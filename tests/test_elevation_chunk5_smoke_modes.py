"""Chunk-5 smoke-mode placeholders.

These tests lock the basic mode semantics before the runtime path exists. They
are intentionally lightweight and should later be complemented by real runtime
smokes for late normals and optional densification.
"""

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "utils" / "elevation_chunk5_helpers.py"
HAS_HELPER = HELPER_PATH.exists()


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("elevation_chunk5_helpers", HELPER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chunk5():
    if not HAS_HELPER:
        pytest.skip("Chunk5 helper module not created yet")
    return _load_helper_module()


@pytest.fixture(autouse=True)
def _deterministic_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_c5_t04_placeholder_off_mode_parity_toy_aggregation(chunk5):
    baseline = 1.25
    loss_normal = 0.40
    w_normal = 0.10

    normal_mode, densify_mode = chunk5.resolve_effective_chunk5_modes(
        elevation_aware=False,
        requested_normal_mode="active",
        densify_enabled=True,
        requested_densify_mode="active",
    )

    weighted_normal = (w_normal * loss_normal) if chunk5.mode_enables_normal_loss(normal_mode) else 0.0
    total = baseline + weighted_normal

    assert normal_mode == "off"
    assert densify_mode == "off"
    assert total == pytest.approx(baseline)
