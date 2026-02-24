import hashlib
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


def _assert_payload_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        a_val = a[key]
        b_val = b[key]
        if isinstance(a_val, torch.Tensor):
            assert isinstance(b_val, torch.Tensor)
            assert torch.equal(a_val, b_val)
        else:
            assert a_val == b_val


def test_c4_t09_checkpoint_schema_constant_contract(chunk4):
    assert chunk4.CHECKPOINT_SCHEMA_VERSION == "chunk4_coupling_support_v1"


def test_c4_t09_checkpoint_payload_contains_required_chunk4_fields(chunk4):
    frame_keys = ["frame_001", "frame_010", "frame_020"]
    expected_fp = hashlib.sha256("\n".join(frame_keys).encode("utf-8")).hexdigest()

    payload = chunk4.build_chunk4_checkpoint_payload(
        surfel_ids=torch.tensor([2, 5, 7], dtype=torch.long),
        next_surfel_id=9,
        id_to_row=torch.tensor([-1, -1, 0, -1, -1, 1, -1, 2, -1], dtype=torch.long),
        ema_by_id=torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=torch.float32),
        last_raw_by_id=torch.tensor([0.0, 0.0, 1.0, 0.5, 0.2, 0.9, 0.1, 0.8, 0.0], dtype=torch.float32),
        birth_iter_by_id=torch.tensor([0, 0, 10, 0, 0, 12, 0, 15, 0], dtype=torch.long),
        fail_streak_by_id=torch.tensor([0, 0, 1, 0, 0, 2, 0, 0, 0], dtype=torch.long),
        active_frame_keys=frame_keys,
        couple_mode="active",
        support_mode="shadow",
        support_scheduler_state={"phase": "mid", "patience": 4},
    )

    assert payload["checkpoint_schema_version"] == "chunk4_coupling_support_v1"
    assert payload["active_frame_keys"] == frame_keys
    assert payload["active_frame_fingerprint"] == expected_fp
    assert "surfel_ids" in payload
    assert "next_surfel_id" in payload
    assert "id_to_row" in payload
    assert "ema_by_id" in payload
    assert "last_raw_by_id" in payload
    assert "birth_iter_by_id" in payload
    assert "fail_streak_by_id" in payload
    assert "couple_mode" in payload
    assert "support_mode" in payload


@pytest.mark.filterwarnings("ignore:You are using `torch.load` with `weights_only=False`.*:FutureWarning")
def test_c4_t09_checkpoint_payload_torch_roundtrip(tmp_path, chunk4):
    payload = chunk4.build_chunk4_checkpoint_payload(
        surfel_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        next_surfel_id=3,
        id_to_row=torch.tensor([0, 1, 2], dtype=torch.long),
        ema_by_id=torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32),
        last_raw_by_id=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        birth_iter_by_id=torch.tensor([0, 5, 10], dtype=torch.long),
        fail_streak_by_id=torch.tensor([0, 1, 2], dtype=torch.long),
        active_frame_keys=["f0", "f1"],
        couple_mode="shadow",
        support_mode="active",
        support_scheduler_state={"phase": "late", "patience": 4},
    )

    p = tmp_path / "chunk4_payload.pth"
    torch.save(payload, p)
    loaded = torch.load(p, map_location="cpu")
    _assert_payload_equal(payload, loaded)


def test_c4_t09_resume_policy_strict_raises_on_schema_mismatch(chunk4):
    with pytest.raises(ValueError, match="schema"):
        chunk4.resolve_chunk4_resume_action(
            checkpoint_schema_version="legacy",
            runtime_schema_version="chunk4_coupling_support_v1",
            frame_fingerprint_matches=True,
            mismatch_policy="strict",
        )


def test_c4_t09_resume_policy_strict_raises_on_frame_set_mismatch(chunk4):
    with pytest.raises(ValueError, match="frame-set mismatch"):
        chunk4.resolve_chunk4_resume_action(
            checkpoint_schema_version="chunk4_coupling_support_v1",
            runtime_schema_version="chunk4_coupling_support_v1",
            frame_fingerprint_matches=False,
            mismatch_policy="strict",
        )


def test_c4_t09_resume_policy_reset_paths_are_deterministic(chunk4):
    reset_chunk4 = chunk4.resolve_chunk4_resume_action(
        checkpoint_schema_version="legacy",
        runtime_schema_version="chunk4_coupling_support_v1",
        frame_fingerprint_matches=False,
        mismatch_policy="reset_chunk4",
    )
    assert reset_chunk4 == "reset_chunk4"

    reset_all = chunk4.resolve_chunk4_resume_action(
        checkpoint_schema_version="legacy",
        runtime_schema_version="chunk4_coupling_support_v1",
        frame_fingerprint_matches=False,
        mismatch_policy="reset_all",
    )
    assert reset_all == "reset_all"


def test_c4_t09_resume_policy_load_when_schema_and_fingerprint_match(chunk4):
    action = chunk4.resolve_chunk4_resume_action(
        checkpoint_schema_version="chunk4_coupling_support_v1",
        runtime_schema_version="chunk4_coupling_support_v1",
        frame_fingerprint_matches=True,
        mismatch_policy="strict",
    )
    assert action == "load"
