import importlib.util
import math
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


def test_c5_t05_stage2_start_default_stays_fixed_for_short_and_long_runs(chunk5):
    assert chunk5.resolve_chunk5_stage2_start_default(stage2_iters=0) == 12000
    assert chunk5.resolve_chunk5_stage2_start_default(stage2_iters=5000) == 12000
    assert chunk5.resolve_chunk5_stage2_start_default(stage2_iters=50000) == 12000


def test_c5_t05_renderer_contract_gate_requires_2dgs_and_ray_binned(chunk5):
    assert chunk5.chunk5_renderer_contract_is_active(render_mode="2dgs", occlusion_mode="ray_binned")
    assert not chunk5.chunk5_renderer_contract_is_active(render_mode="2dgs", occlusion_mode="legacy")
    assert not chunk5.chunk5_renderer_contract_is_active(render_mode="pinhole", occlusion_mode="ray_binned")


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


def test_c5_t06_arc_score_contribution_preserves_per_pixel_scores(chunk5):
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

    contribution = chunk5.compute_arc_score_contribution(
        gt_intensity=gt_intensity,
        valid_mask=valid_mask,
        reliability=reliability,
    )

    expected = torch.tensor(
        [
            [0.9, 0.2, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert contribution.shape == gt_intensity.shape
    assert torch.allclose(contribution, expected, atol=1e-6)


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


def test_c5_t06_high_error_tracker_key_roundtrip_and_ranking_contract(chunk5):
    key = chunk5.make_high_error_key("frame_001", 12, 34)
    assert key == "frame_001:12:34"
    assert chunk5.parse_high_error_key(key) == ("frame_001", 12, 34)

    tracker = {
        "frame_001:10:20": 2,
        "frame_001:5:3": 4,
        "frame_002:7:9": 3,
        "frame_003:1:1": 10,
    }
    selected = chunk5.select_persistent_high_error_candidates(
        high_error_tracker=tracker,
        frame_keys=["frame_001", "frame_002"],
        min_hits=2,
        max_candidates=3,
    )

    assert selected == [
        ("frame_001", 5, 3, 4),
        ("frame_002", 7, 9, 3),
        ("frame_001", 10, 20, 2),
    ]


def test_c5_t06_peak_selection_from_loglik_contract(chunk5):
    peak, scores = chunk5.select_peak_bin_from_loglik(
        loglik=torch.tensor([-0.1, -1.0, -0.2], dtype=torch.float32),
        support_mask=torch.tensor([1, 0, 1], dtype=torch.bool),
        min_score=0.5,
    )

    assert peak == 0
    assert torch.allclose(scores, torch.tensor([math.exp(-0.1), 0.0, math.exp(-0.2)], dtype=torch.float32), atol=1e-6)


def test_c5_t06_peak_selection_from_loglik_skips_unsupported_bins(chunk5):
    peak, scores = chunk5.select_peak_bin_from_loglik(
        loglik=torch.tensor([-0.01, -0.02], dtype=torch.float32),
        support_mask=torch.tensor([0, 0], dtype=torch.bool),
        min_score=0.1,
    )

    assert peak is None
    assert torch.equal(scores, torch.zeros((2,), dtype=torch.float32))


def test_c5_t07_quaternion_from_normal_contract(chunk5):
    normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )

    quats = chunk5.quaternions_from_normals(normals)

    assert quats.shape == (3, 4)
    assert torch.allclose(torch.norm(quats, dim=-1), torch.ones((3,), dtype=torch.float32), atol=1e-6)
    z_rotated = torch.stack(
        [
            2 * (quats[:, 1] * quats[:, 3] + quats[:, 0] * quats[:, 2]),
            2 * (quats[:, 2] * quats[:, 3] - quats[:, 0] * quats[:, 1]),
            1 - 2 * (quats[:, 1] * quats[:, 1] + quats[:, 2] * quats[:, 2]),
        ],
        dim=-1,
    )
    assert torch.allclose(z_rotated, normals, atol=1e-5)
