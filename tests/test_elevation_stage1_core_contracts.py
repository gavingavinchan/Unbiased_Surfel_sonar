import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "utils" / "elevation_stage1_helpers.py"
assert HELPER_PATH.exists(), "Missing helper module: utils/elevation_stage1_helpers.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("elevation_stage1_helpers", HELPER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage1 = _load_helper_module()


def test_pose_only_score_combination_formula():
    score = stage1.combine_pose_overlap_score(
        yaw_score=0.5,
        baseline_score=0.8,
        w_yaw=0.6,
        w_base=0.4,
    )
    assert score == pytest.approx(0.62)


def test_overlap_hard_gate_contract():
    assert stage1.pose_only_hard_gate(
        baseline_m=0.08,
        yaw_deg=30.0,
        min_baseline_m=0.06,
        max_yaw_deg=40.0,
    )
    assert not stage1.pose_only_hard_gate(
        baseline_m=0.02,
        yaw_deg=30.0,
        min_baseline_m=0.06,
        max_yaw_deg=40.0,
    )
    assert not stage1.pose_only_hard_gate(
        baseline_m=0.08,
        yaw_deg=70.0,
        min_baseline_m=0.06,
        max_yaw_deg=40.0,
    )
    assert stage1.pose_only_hard_gate(
        baseline_m=0.06,
        yaw_deg=-40.0,
        min_baseline_m=0.06,
        max_yaw_deg=40.0,
    )


def test_deterministic_overlap_ranking_tie_break():
    candidates = [
        ("frame_b", 0.8),
        ("frame_a", 0.8),
        ("frame_c", 0.4),
    ]
    ranked = stage1.rank_overlap_candidates(candidates, topk=2)
    assert ranked == [("frame_a", 0.8), ("frame_b", 0.8)]


def test_round_robin_sampler_is_deterministic_and_covers_all_frames():
    state = stage1.RoundRobinSamplerState(frame_keys=["f0", "f1", "f2", "f3"], cursor=0, epoch=0)

    seen = []
    for _ in range(2):
        batch, state = stage1.round_robin_sample(state, frames_per_iter=3)
        seen.extend(batch)

    assert state.epoch >= 1
    assert set(seen) == {"f0", "f1", "f2", "f3"}


def test_round_robin_sampler_same_state_same_sequence():
    s1 = stage1.RoundRobinSamplerState(frame_keys=["f0", "f1", "f2"], cursor=1, epoch=2)
    s2 = stage1.RoundRobinSamplerState(frame_keys=["f0", "f1", "f2"], cursor=1, epoch=2)

    b1, n1 = stage1.round_robin_sample(s1, frames_per_iter=5)
    b2, n2 = stage1.round_robin_sample(s2, frames_per_iter=5)

    assert b1 == b2
    assert n1 == n2


def test_round_robin_sampler_wraps_for_large_batch():
    state = stage1.RoundRobinSamplerState(frame_keys=["f0", "f1", "f2"], cursor=0, epoch=0)
    batch, nxt = stage1.round_robin_sample(state, frames_per_iter=5)

    assert batch == ["f0", "f1", "f2", "f0", "f1"]
    assert nxt.cursor == 2
    assert nxt.epoch == 1


def test_normalize_by_percentiles_clamps_and_handles_degenerate_span():
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)
    y = stage1.normalize_by_percentiles(x, lo=0.0, hi=1.0, eps=1e-6)
    assert torch.allclose(y, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))

    z = stage1.normalize_by_percentiles(x, lo=1.0, hi=1.0, eps=1e-6)
    assert torch.isfinite(z).all()
    assert (z >= 0.0).all()
    assert (z <= 1.0).all()


def test_masked_softmax_all_masked_rows_are_finite_and_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]], dtype=torch.float32)
    support_mask = torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.bool)

    probs = stage1.masked_softmax(logits, support_mask, dim=-1, min_support=1e-6)

    assert torch.isfinite(probs).all()
    assert torch.allclose(probs[0], torch.zeros_like(probs[0]))
    assert probs[1, 1].item() == pytest.approx(0.0)
    assert probs[1].sum().item() == pytest.approx(1.0)


def test_linear_temperature_schedule_contract():
    mid = stage1.linear_anneal(iteration=5, start=2.0, end=0.1, horizon=10)
    assert mid == pytest.approx(1.05)

    clamped_end = stage1.linear_anneal(iteration=999, start=2.0, end=0.1, horizon=10)
    assert clamped_end == pytest.approx(0.1)

    fallback_horizon = stage1.linear_anneal(iteration=5, start=2.0, end=0.1, horizon=0)
    assert fallback_horizon == pytest.approx(0.1)


def test_shared_vs_decoupled_post_temperature_schedule():
    t_model_shared, t_post_shared = stage1.resolve_temperatures(
        iteration=5,
        temp_start=2.0,
        temp_end=0.1,
        temp_post_mode="shared",
        temp_post_start=3.0,
        temp_post_end=1.0,
        horizon=10,
    )
    assert t_model_shared == pytest.approx(1.05)
    assert t_post_shared == pytest.approx(t_model_shared)

    t_model_dec, t_post_dec = stage1.resolve_temperatures(
        iteration=5,
        temp_start=2.0,
        temp_end=0.1,
        temp_post_mode="decoupled",
        temp_post_start=3.0,
        temp_post_end=1.0,
        horizon=10,
    )
    assert t_model_dec == pytest.approx(1.05)
    assert t_post_dec == pytest.approx(2.0)
    assert t_post_dec != pytest.approx(t_model_dec)


def test_refresh_interval_trigger_contract():
    assert not stage1.should_refresh_pixel_bank(iteration=10, refresh_interval=0)
    assert stage1.should_refresh_pixel_bank(iteration=10, refresh_interval=5)
    assert not stage1.should_refresh_pixel_bank(iteration=11, refresh_interval=5)


def test_refresh_remap_nearest_and_far_reset_contract():
    old_rows = torch.tensor([10, 20], dtype=torch.long)
    old_cols = torch.tensor([10, 20], dtype=torch.long)
    old_logits = torch.tensor([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]], dtype=torch.float32)

    new_rows = torch.tensor([11, 40], dtype=torch.long)
    new_cols = torch.tensor([11, 40], dtype=torch.long)

    out = stage1.remap_or_reset_pixel_logits(
        old_rows=old_rows,
        old_cols=old_cols,
        old_logits=old_logits,
        new_rows=new_rows,
        new_cols=new_cols,
        remap_mode="nearest",
        remap_max_dist=6,
    )

    assert torch.allclose(out[0], old_logits[0])
    assert torch.allclose(out[1], torch.zeros_like(out[1]))


def test_refresh_remap_reset_mode_zeros_all_logits():
    old_rows = torch.tensor([10, 20], dtype=torch.long)
    old_cols = torch.tensor([10, 20], dtype=torch.long)
    old_logits = torch.tensor([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]], dtype=torch.float32)

    new_rows = torch.tensor([11, 21], dtype=torch.long)
    new_cols = torch.tensor([11, 21], dtype=torch.long)

    out = stage1.remap_or_reset_pixel_logits(
        old_rows=old_rows,
        old_cols=old_cols,
        old_logits=old_logits,
        new_rows=new_rows,
        new_cols=new_cols,
        remap_mode="reset",
        remap_max_dist=6,
    )

    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.zeros_like(out))


def test_optimizer_rebuild_required_when_shape_changes():
    old_logits = torch.zeros((2, 7), dtype=torch.float32)
    same_shape = torch.zeros((2, 7), dtype=torch.float32)
    new_shape = torch.zeros((3, 7), dtype=torch.float32)

    assert not stage1.optimizer_rebuild_required(old_logits, same_shape)
    assert stage1.optimizer_rebuild_required(old_logits, new_shape)


def test_frame_stats_low_valid_uses_reliability_floor():
    gt = torch.zeros((4, 4), dtype=torch.float32)
    gt[0, 0] = 1.0
    valid = torch.zeros((4, 4), dtype=torch.bool)
    valid[0, 0] = True

    stats = stage1.compute_frame_stats(
        gt_frame=gt,
        valid_mask=valid,
        p_lo=10,
        p_hi=99,
        rel_floor=0.3,
        rel_valid_min=0.5,
        rel_valid_max=0.9,
        rel_dyn_min=0.1,
        rel_dyn_max=0.8,
    )

    assert math.isfinite(stats["p_lo"])
    assert math.isfinite(stats["p_hi"])
    assert math.isfinite(stats["dyn_range"])
    assert stats["reliability"] == pytest.approx(0.3)


def test_frame_stats_zero_valid_returns_floor_and_zero_stats():
    gt = torch.zeros((2, 2), dtype=torch.float32)
    valid = torch.zeros((2, 2), dtype=torch.bool)
    stats = stage1.compute_frame_stats(
        gt_frame=gt,
        valid_mask=valid,
        p_lo=10,
        p_hi=99,
        rel_floor=0.3,
        rel_valid_min=0.03,
        rel_valid_max=0.30,
        rel_dyn_min=0.08,
        rel_dyn_max=0.50,
    )
    assert stats["p_lo"] == 0.0
    assert stats["p_hi"] == 0.0
    assert stats["dyn_range"] == 0.0
    assert stats["reliability"] == pytest.approx(0.3)


def test_frame_stats_normal_path_respects_floor_bound():
    gt = torch.tensor([[0.1, 0.4], [0.7, 1.0]], dtype=torch.float32)
    valid = torch.ones((2, 2), dtype=torch.bool)
    stats = stage1.compute_frame_stats(
        gt_frame=gt,
        valid_mask=valid,
        p_lo=10,
        p_hi=99,
        rel_floor=0.3,
        rel_valid_min=0.03,
        rel_valid_max=0.30,
        rel_dyn_min=0.08,
        rel_dyn_max=0.50,
    )
    assert stats["reliability"] == pytest.approx(1.0)
