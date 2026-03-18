import importlib.util
import math
import py_compile
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "utils" / "elevation_chunk5_helpers.py"
DEBUG_SCRIPT = REPO_ROOT / "debug_multiframe.py"
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


def test_c5_t01_chunk5_helper_module_exists_red_until_helper_added():
    assert HELPER_PATH.exists(), "Missing helper module: utils/elevation_chunk5_helpers.py"


@pytest.mark.skipif(not HAS_HELPER, reason="Chunk5 helper module not created yet")
def test_c5_t01_chunk5_helper_module_compiles_for_smoke():
    py_compile.compile(str(HELPER_PATH), doraise=True)


def test_c5_t01_debug_multiframe_script_compiles_for_smoke():
    py_compile.compile(str(DEBUG_SCRIPT), doraise=True)


def test_c5_t01_normal_ramp_schedule_contract(chunk5):
    cfg = {
        "ramp_start_iter": 4000,
        "ramp_end_iter": 8000,
        "weight_early": 0.01,
        "weight_late": 0.10,
    }

    assert chunk5.resolve_normal_weight(iteration=3999, **cfg) == pytest.approx(0.01)
    assert chunk5.resolve_normal_weight(iteration=4000, **cfg) == pytest.approx(0.01)
    assert chunk5.resolve_normal_weight(iteration=6000, **cfg) == pytest.approx(0.055)
    assert chunk5.resolve_normal_weight(iteration=8000, **cfg) == pytest.approx(0.10)
    assert chunk5.resolve_normal_weight(iteration=8001, **cfg) == pytest.approx(0.10)


def test_c5_t01_normal_ramp_zero_span_contract(chunk5):
    cfg = {
        "ramp_start_iter": 4000,
        "ramp_end_iter": 4000,
        "weight_early": 0.01,
        "weight_late": 0.10,
    }

    assert chunk5.resolve_normal_weight(iteration=3999, **cfg) == pytest.approx(0.01)
    assert chunk5.resolve_normal_weight(iteration=4000, **cfg) == pytest.approx(0.01)
    assert chunk5.resolve_normal_weight(iteration=4001, **cfg) == pytest.approx(0.10)


def test_c5_t02_expected_elevation_contract(chunk5):
    probs = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    elev_bins = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)

    expected = chunk5.compute_expected_elevation(probs=probs, elev_bins=elev_bins)

    assert expected.shape == (2,)
    assert torch.isfinite(expected).all()
    assert expected[0].item() == pytest.approx(0.15)
    assert expected[1].item() == pytest.approx(0.0)


def test_c5_t02_finite_difference_normal_contract(chunk5):
    pts_left = torch.tensor([[0.0, 1.0, 2.0], [1.0, -1.0, 0.0]], dtype=torch.float32)
    pts_right = torch.tensor([[2.0, 1.0, 2.0], [3.0, -1.0, 0.0]], dtype=torch.float32)
    pts_up = torch.tensor([[1.0, 0.0, 2.0], [2.0, -2.0, 0.0]], dtype=torch.float32)
    pts_down = torch.tensor([[1.0, 2.0, 2.0], [2.0, 0.0, 0.0]], dtype=torch.float32)

    normals = chunk5.compute_finite_difference_normals(
        pts_left=pts_left,
        pts_right=pts_right,
        pts_up=pts_up,
        pts_down=pts_down,
        eps=1e-8,
    )

    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    assert normals.shape == expected.shape
    assert torch.isfinite(normals).all()
    assert torch.allclose(normals, expected, atol=1e-6)
    assert torch.allclose(torch.norm(normals, dim=-1), torch.ones((2,), dtype=torch.float32), atol=1e-6)


def test_c5_t02_finite_difference_normal_tilted_surface_contract(chunk5):
    pts_left = torch.tensor([[-1.0, 0.0, -1.0]], dtype=torch.float32)
    pts_right = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
    pts_up = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32)
    pts_down = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    normals = chunk5.compute_finite_difference_normals(
        pts_left=pts_left,
        pts_right=pts_right,
        pts_up=pts_up,
        pts_down=pts_down,
        eps=1e-8,
    )

    expected = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
    expected = expected / torch.norm(expected, dim=-1, keepdim=True)
    assert normals.shape == expected.shape
    assert torch.isfinite(normals).all()
    assert torch.allclose(normals, expected, atol=1e-6)


def test_c5_t03_confidence_mask_contract(chunk5):
    probs = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.34, 0.33, 0.33],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    support_mask = torch.tensor(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=torch.bool,
    )

    confident, entropy = chunk5.compute_confidence_mask(
        probs=probs,
        support_mask=support_mask,
        confidence_thresh=0.5,
    )

    assert confident.tolist() == [True, False, False]
    assert torch.isfinite(entropy).all()
    assert entropy[0].item() == pytest.approx(0.0)
    assert entropy[1].item() == pytest.approx(math.log(3.0), rel=5e-3)
    assert entropy[2].item() == pytest.approx(0.0)


def test_c5_t03_cosine_distance_normal_loss_contract(chunk5):
    n_quat = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    n_expected = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    loss = chunk5.compute_normal_supervision_loss(
        n_quat=n_quat,
        n_expected=n_expected,
    )

    expected_terms = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected_terms.mean().item())
