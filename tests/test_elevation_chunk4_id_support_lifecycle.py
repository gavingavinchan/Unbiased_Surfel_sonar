import importlib.util
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


def test_c4_t04_persistent_id_lifecycle_densify_and_prune_reorder(chunk4):
    state = chunk4.initialize_persistent_surfel_state(num_surfels=8, device="cpu", init_birth_iter=0)
    assert torch.equal(state["surfel_ids"], torch.arange(8, dtype=torch.long))
    assert int(state["next_surfel_id"]) == 8

    state = chunk4.apply_densify_to_surfel_state(state, n_new=3, current_iter=11)
    assert torch.equal(state["surfel_ids"][-3:], torch.tensor([8, 9, 10], dtype=torch.long))
    assert int(state["next_surfel_id"]) == 11
    assert torch.equal(state["birth_iter_by_id"][8:11], torch.tensor([11, 11, 11], dtype=torch.long))

    keep_row_idx = torch.tensor([10, 1, 7, 0, 8], dtype=torch.long)
    state = chunk4.apply_prune_reorder_to_surfel_state(state, keep_row_idx=keep_row_idx)
    assert torch.equal(state["surfel_ids"], torch.tensor([10, 1, 7, 0, 8], dtype=torch.long))

    id_to_row = state["id_to_row"]
    assert int(id_to_row[10].item()) == 0
    assert int(id_to_row[1].item()) == 1
    assert int(id_to_row[7].item()) == 2
    assert int(id_to_row[0].item()) == 3
    assert int(id_to_row[8].item()) == 4
    assert int(id_to_row[2].item()) == -1
    assert int(id_to_row[9].item()) == -1

    chunk4.assert_surfel_id_integrity(state)


def test_c4_t04_duplicate_id_fail_fast_contract(chunk4):
    bad_state = {
        "surfel_ids": torch.tensor([0, 1, 1], dtype=torch.long),
        "next_surfel_id": 3,
        "id_to_row": torch.tensor([0, 2, -1], dtype=torch.long),
        "ema_by_id": torch.zeros(3, dtype=torch.float32),
        "last_raw_by_id": torch.zeros(3, dtype=torch.float32),
        "birth_iter_by_id": torch.zeros(3, dtype=torch.long),
        "fail_streak_by_id": torch.zeros(3, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="Duplicate"):
        chunk4.assert_surfel_id_integrity(bad_state)


def test_c4_t05_support_buffer_update_is_id_invariant_under_reindex(chunk4):
    raw_by_id = torch.tensor([0.2, 0.9, 0.1, 0.8, 0.4, 0.3, 0.7, 0.5], dtype=torch.float32)

    state_a = chunk4.initialize_persistent_surfel_state(num_surfels=8, device="cpu", init_birth_iter=0)
    state_b = chunk4.initialize_persistent_surfel_state(num_surfels=8, device="cpu", init_birth_iter=0)

    row_raw_a = raw_by_id[state_a["surfel_ids"]]
    state_a = chunk4.update_support_buffers_by_id(state_a, row_support_raw=row_raw_a, ema_decay=0.90)

    perm = torch.tensor([3, 1, 7, 0, 2, 6, 4, 5], dtype=torch.long)
    state_b = chunk4.apply_prune_reorder_to_surfel_state(state_b, keep_row_idx=perm)
    row_raw_b = raw_by_id[state_b["surfel_ids"]]
    state_b = chunk4.update_support_buffers_by_id(state_b, row_support_raw=row_raw_b, ema_decay=0.90)

    assert torch.allclose(state_a["ema_by_id"], state_b["ema_by_id"])
    assert torch.allclose(state_a["last_raw_by_id"], state_b["last_raw_by_id"])

    before = state_b["ema_by_id"].clone()
    state_b = chunk4.apply_prune_reorder_to_surfel_state(
        state_b,
        keep_row_idx=torch.tensor([0, 2, 4, 6], dtype=torch.long),
    )
    row_raw_small = torch.tensor([0.3, 0.2, 0.1, 0.0], dtype=torch.float32)
    state_b = chunk4.update_support_buffers_by_id(state_b, row_support_raw=row_raw_small, ema_decay=0.90)

    retired = torch.ones(int(state_b["next_surfel_id"]), dtype=torch.bool)
    retired[state_b["surfel_ids"]] = False
    assert torch.allclose(state_b["ema_by_id"][retired], before[retired])


def test_c4_t06_support_schedule_warmup_and_effective_floor_contract(chunk4):
    support_count = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
    diverse_count = torch.tensor([0, 1, 2, 3, 10], dtype=torch.float32)

    warmup_fail, warmup_floor = chunk4.compute_support_failure_mask(
        iteration=1000,
        support_count_by_id=support_count,
        diverse_candidate_count_by_id=diverse_count,
        warmup_iters=4000,
        late_phase_start_iter=8000,
        min_ratio_mid=0.25,
        min_ratio_late=0.45,
        min_count_mid=2,
        min_count_late=4,
        use_ratio=True,
    )
    assert not warmup_fail.any()
    assert torch.equal(warmup_floor, torch.zeros_like(warmup_floor))

    mid_fail, mid_floor = chunk4.compute_support_failure_mask(
        iteration=5000,
        support_count_by_id=support_count,
        diverse_candidate_count_by_id=diverse_count,
        warmup_iters=4000,
        late_phase_start_iter=8000,
        min_ratio_mid=0.25,
        min_ratio_late=0.45,
        min_count_mid=2,
        min_count_late=4,
        use_ratio=True,
    )
    assert torch.equal(mid_floor, torch.tensor([0, 1, 2, 2, 2], dtype=torch.float32))
    assert torch.equal(mid_fail, torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool))

    late_fail, late_floor = chunk4.compute_support_failure_mask(
        iteration=9000,
        support_count_by_id=support_count,
        diverse_candidate_count_by_id=diverse_count,
        warmup_iters=4000,
        late_phase_start_iter=8000,
        min_ratio_mid=0.25,
        min_ratio_late=0.45,
        min_count_mid=2,
        min_count_late=4,
        use_ratio=True,
    )
    assert torch.equal(late_floor, torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))
    assert torch.equal(late_fail, torch.tensor([1, 0, 0, 0, 1], dtype=torch.bool))


def test_c4_t06_zero_diverse_ratio_fails_when_support_zero(chunk4):
    # Contract choice for v1: no diverse candidates means no support evidence,
    # so the surfel is treated as failing the hard support check post-warmup.
    fail_mask, floor_eff = chunk4.compute_support_failure_mask(
        iteration=5000,
        support_count_by_id=torch.tensor([0.0], dtype=torch.float32),
        diverse_candidate_count_by_id=torch.tensor([0.0], dtype=torch.float32),
        warmup_iters=4000,
        late_phase_start_iter=8000,
        min_ratio_mid=0.25,
        min_ratio_late=0.45,
        min_count_mid=2,
        min_count_late=4,
        use_ratio=True,
    )
    assert floor_eff.item() == pytest.approx(0.0)
    assert bool(fail_mask.item())


def test_c4_t07_hysteresis_requires_consecutive_failures(chunk4):
    fail_streak = torch.zeros(1, dtype=torch.long)
    pattern = [True, True, False, True, True, True]
    observed_prune = []

    for fail in pattern:
        fail_streak, prune_mask = chunk4.apply_prune_hysteresis(
            fail_streak_by_id=fail_streak,
            fail_mask_by_id=torch.tensor([fail], dtype=torch.bool),
            patience=3,
        )
        observed_prune.append(bool(prune_mask.item()))

    assert observed_prune == [False, False, False, False, False, True]
    assert int(fail_streak.item()) == 3


def test_c4_t08_new_surfel_grace_skips_until_age_threshold_only(chunk4):
    prune_mask = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    birth_iter_by_id = torch.tensor([0, 90, 95, 99], dtype=torch.long)

    out_enabled = chunk4.apply_new_surfel_grace(
        prune_mask_by_id=prune_mask,
        birth_iter_by_id=birth_iter_by_id,
        current_iter=100,
        grace_iters=10,
        enabled=True,
    )
    assert torch.equal(out_enabled, torch.tensor([1, 1, 0, 0], dtype=torch.bool))

    out_disabled = chunk4.apply_new_surfel_grace(
        prune_mask_by_id=prune_mask,
        birth_iter_by_id=birth_iter_by_id,
        current_iter=100,
        grace_iters=10,
        enabled=False,
    )
    assert torch.equal(out_disabled, prune_mask)
