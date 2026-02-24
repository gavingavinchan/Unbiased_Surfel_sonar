import importlib.util
import math
import py_compile
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "utils" / "elevation_chunk4_helpers.py"
HAS_HELPER = HELPER_PATH.exists()


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("elevation_chunk4_helpers", HELPER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chunk4():
    if not HAS_HELPER:
        pytest.skip("Chunk4 helper module not created yet")
    return _load_helper_module()


@pytest.fixture(autouse=True)
def _deterministic_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def test_c4_t01_chunk4_helper_module_exists_red_until_helper_added():
    assert HELPER_PATH.exists(), "Missing helper module: utils/elevation_chunk4_helpers.py"


@pytest.mark.skipif(not HAS_HELPER, reason="Chunk4 helper module not created yet")
def test_c4_t01_chunk4_helper_module_compiles_for_smoke():
    py_compile.compile(str(HELPER_PATH), doraise=True)


def test_c4_t01_debug_multiframe_script_compiles_for_smoke():
    target = REPO_ROOT / "debug_multiframe.py"
    py_compile.compile(str(target), doraise=True)


def test_c4_t02_association_gating_contract_with_no_match_safety(chunk4):
    exp_row = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32)
    exp_col = torch.tensor([10.0, 20.0, 12.0, 10.0], dtype=torch.float32)
    exp_depth = torch.tensor([1.0, 1.0, 1.25, 1.0], dtype=torch.float32)
    exp_valid = torch.tensor([1, 1, 1, 0], dtype=torch.bool)

    surf_row = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
    surf_col = torch.tensor([10.0, 14.0, 13.0], dtype=torch.float32)
    surf_depth = torch.tensor([1.0, 1.0, 1.20], dtype=torch.float32)
    surf_valid = torch.tensor([1, 1, 0], dtype=torch.bool)

    surf_idx, assoc_w, match_valid = chunk4.associate_expected_points_to_surfels(
        exp_row=exp_row,
        exp_col=exp_col,
        exp_depth=exp_depth,
        exp_valid=exp_valid,
        surf_row=surf_row,
        surf_col=surf_col,
        surf_depth=surf_depth,
        surf_valid=surf_valid,
        max_pix_err=3.0,
        max_depth_err=0.08,
        sigma_pix=2.0,
        sigma_depth=0.05,
        min_w=0.10,
    )

    assert torch.equal(match_valid, torch.tensor([1, 0, 0, 0], dtype=torch.bool))
    assert int(surf_idx[0].item()) == 0
    assert assoc_w[0].item() == pytest.approx(1.0)

    unmatched = ~match_valid
    assert (surf_idx[unmatched] >= 0).all()
    assert (surf_idx[unmatched] < surf_row.shape[0]).all()
    assert (assoc_w[match_valid] >= 0.10).all()
    assert (assoc_w[match_valid] <= 1.0).all()


def test_c4_t02_association_weight_formula_and_min_clamp(chunk4):
    exp_row = torch.tensor([10.0, 10.0], dtype=torch.float32)
    exp_col = torch.tensor([10.0, 10.0], dtype=torch.float32)
    exp_depth = torch.tensor([1.00, 1.16], dtype=torch.float32)
    exp_valid = torch.tensor([1, 1], dtype=torch.bool)

    surf_row = torch.tensor([10.0, 10.0], dtype=torch.float32)
    surf_col = torch.tensor([12.0, 13.0], dtype=torch.float32)
    surf_depth = torch.tensor([1.05, 1.08], dtype=torch.float32)
    surf_valid = torch.tensor([1, 1], dtype=torch.bool)

    surf_idx, assoc_w, match_valid = chunk4.associate_expected_points_to_surfels(
        exp_row=exp_row,
        exp_col=exp_col,
        exp_depth=exp_depth,
        exp_valid=exp_valid,
        surf_row=surf_row,
        surf_col=surf_col,
        surf_depth=surf_depth,
        surf_valid=surf_valid,
        max_pix_err=3.0,
        max_depth_err=0.08,
        sigma_pix=2.0,
        sigma_depth=0.05,
        min_w=0.10,
    )

    assert match_valid.tolist() == [True, True]
    assert surf_idx.tolist() == [0, 1]

    expected_w0 = math.exp(-0.5 * (1.0 + 1.0))
    expected_w1 = max(0.10, math.exp(-0.5 * (2.25 + 2.56)))
    assert assoc_w[0].item() == pytest.approx(expected_w0, rel=1e-6)
    assert assoc_w[1].item() == pytest.approx(expected_w1, rel=1e-6)


def test_c4_t03_coupling_reduction_dense_and_zero_match_contracts(chunk4):
    pts_expected = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    surfel_xyz = torch.tensor([[0.01, 0.0, 0.0], [0.97, 0.0, 0.0]], dtype=torch.float32)
    surf_idx = torch.tensor([0, 1], dtype=torch.long)
    assoc_w = torch.tensor([1.0, 0.5], dtype=torch.float32)
    match_valid = torch.tensor([1, 1], dtype=torch.bool)

    loss_dense = chunk4.reduce_coupling_loss(
        pts_expected=pts_expected,
        surfel_xyz=surfel_xyz,
        surf_idx=surf_idx,
        assoc_w=assoc_w,
        match_valid=match_valid,
        huber_delta=0.03,
    )

    assert torch.isfinite(loss_dense)
    assert loss_dense.item() > 0.0

    zero_loss = chunk4.reduce_coupling_loss(
        pts_expected=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        surfel_xyz=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        surf_idx=torch.tensor([12345], dtype=torch.long),
        assoc_w=torch.tensor([0.2], dtype=torch.float32),
        match_valid=torch.tensor([0], dtype=torch.bool),
        huber_delta=0.03,
    )

    assert torch.isfinite(zero_loss)
    assert zero_loss.item() == pytest.approx(0.0)


def test_c4_t02_association_is_independent_per_expected_point(chunk4):
    exp_row = torch.tensor([10.0, 10.0], dtype=torch.float32)
    exp_col = torch.tensor([10.2, 9.7], dtype=torch.float32)
    exp_depth = torch.tensor([1.0, 1.0], dtype=torch.float32)
    exp_valid = torch.tensor([1, 1], dtype=torch.bool)

    surf_row = torch.tensor([10.0, 40.0], dtype=torch.float32)
    surf_col = torch.tensor([10.0, 40.0], dtype=torch.float32)
    surf_depth = torch.tensor([1.0, 1.0], dtype=torch.float32)
    surf_valid = torch.tensor([1, 0], dtype=torch.bool)

    surf_idx, assoc_w, match_valid = chunk4.associate_expected_points_to_surfels(
        exp_row=exp_row,
        exp_col=exp_col,
        exp_depth=exp_depth,
        exp_valid=exp_valid,
        surf_row=surf_row,
        surf_col=surf_col,
        surf_depth=surf_depth,
        surf_valid=surf_valid,
        max_pix_err=3.0,
        max_depth_err=0.08,
        sigma_pix=2.0,
        sigma_depth=0.05,
        min_w=0.10,
    )

    assert match_valid.tolist() == [True, True]
    assert surf_idx.tolist() == [0, 0]
    assert (assoc_w >= 0.10).all()
    assert (assoc_w <= 1.0).all()


def test_c4_t10_mode_gate_contract_for_off_shadow_active(chunk4):
    couple_mode, support_mode = chunk4.resolve_effective_chunk4_modes(
        elevation_aware=False,
        requested_couple_mode="active",
        requested_support_mode="active",
    )
    assert couple_mode == "off"
    assert support_mode == "off"

    couple_mode, support_mode = chunk4.resolve_effective_chunk4_modes(
        elevation_aware=True,
        requested_couple_mode="shadow",
        requested_support_mode="active",
    )
    assert couple_mode == "shadow"
    assert support_mode == "active"

    assert not chunk4.mode_enables_weighted_coupling("off")
    assert not chunk4.mode_enables_weighted_coupling("shadow")
    assert chunk4.mode_enables_weighted_coupling("active")

    assert not chunk4.mode_enables_hard_prune("off")
    assert not chunk4.mode_enables_hard_prune("shadow")
    assert chunk4.mode_enables_hard_prune("active")


def test_c4_t10_mode_rejects_invalid_values(chunk4):
    with pytest.raises(ValueError, match="couple"):
        chunk4.resolve_effective_chunk4_modes(
            elevation_aware=True,
            requested_couple_mode="invalid",
            requested_support_mode="off",
        )

    with pytest.raises(ValueError, match="support"):
        chunk4.resolve_effective_chunk4_modes(
            elevation_aware=True,
            requested_couple_mode="off",
            requested_support_mode="invalid",
        )
