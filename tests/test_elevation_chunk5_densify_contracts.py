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


def test_c5_t05_densify_mode_gate_contract(chunk5):
    normal_mode, densify_mode = chunk5.resolve_effective_chunk5_modes(
        elevation_aware=True,
        requested_normal_mode="shadow",
        densify_enabled=False,
        requested_densify_mode="active",
    )
    assert normal_mode == "shadow"
    assert densify_mode == "off"
    assert not chunk5.mode_enables_normal_loss(normal_mode)
    assert not chunk5.mode_enables_densify_candidates(densify_mode)
    assert not chunk5.mode_enables_densify_spawn(densify_mode)

    normal_mode, densify_mode = chunk5.resolve_effective_chunk5_modes(
        elevation_aware=True,
        requested_normal_mode="active",
        densify_enabled=True,
        requested_densify_mode="shadow",
    )
    assert normal_mode == "active"
    assert densify_mode == "shadow"
    assert chunk5.mode_enables_normal_loss(normal_mode)
    assert chunk5.mode_enables_densify_candidates(densify_mode)
    assert not chunk5.mode_enables_densify_spawn(densify_mode)

    normal_mode, densify_mode = chunk5.resolve_effective_chunk5_modes(
        elevation_aware=True,
        requested_normal_mode="off",
        densify_enabled=True,
        requested_densify_mode="active",
    )
    assert normal_mode == "off"
    assert densify_mode == "active"
    assert chunk5.mode_enables_densify_candidates(densify_mode)
    assert chunk5.mode_enables_densify_spawn(densify_mode)


def test_c5_t05_densify_iteration_and_interval_eligibility_contract(chunk5):
    assert not chunk5.is_densify_iteration_eligible(
        iteration=199,
        stage2_start_iter=200,
        densify_interval=50,
    )
    assert chunk5.is_densify_iteration_eligible(
        iteration=200,
        stage2_start_iter=200,
        densify_interval=50,
    )
    assert not chunk5.is_densify_iteration_eligible(
        iteration=225,
        stage2_start_iter=200,
        densify_interval=50,
    )
    assert chunk5.is_densify_iteration_eligible(
        iteration=250,
        stage2_start_iter=200,
        densify_interval=50,
    )


def test_c5_t06_arc_bin_scoring_formula_contract(chunk5):
    gt_intensity = torch.tensor(
        [
            [0.9, 0.2, 0.1],
            [0.5, 0.4, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=torch.bool,
    )
    reliability = torch.tensor([1.0, 0.5], dtype=torch.float32)

    scores = chunk5.compute_arc_bin_scores(
        gt_intensity=gt_intensity,
        valid_mask=valid_mask,
        reliability=reliability,
    )

    expected = torch.tensor([1.15, 0.2, 0.0], dtype=torch.float32)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, expected, atol=1e-6)


def test_c5_t06_arc_peak_selection_argmax_and_skip_contract(chunk5):
    peak = chunk5.select_arc_peak_bin(
        scores=torch.tensor([0.15, 0.55, 0.30], dtype=torch.float32),
        min_score=0.30,
    )
    assert peak == 1

    skipped = chunk5.select_arc_peak_bin(
        scores=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        min_score=0.30,
    )
    assert skipped is None

    skipped_nontrivial = chunk5.select_arc_peak_bin(
        scores=torch.tensor([0.10, 0.20, 0.15], dtype=torch.float32),
        min_score=0.30,
    )
    assert skipped_nontrivial is None
