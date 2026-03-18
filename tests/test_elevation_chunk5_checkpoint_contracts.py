import hashlib
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


def test_c5_t09_checkpoint_schema_and_payload_key_constants(chunk5):
    assert chunk5.CHECKPOINT_SCHEMA_VERSION == "chunk5_normals_densify_v1"
    assert chunk5.CHECKPOINT_PAYLOAD_KEY == "elevation_chunk5_state"


def test_c5_t09_checkpoint_payload_contains_required_fields_and_omits_derived_weight(chunk5):
    frame_keys = ["frame_001", "frame_010", "frame_020"]
    expected_fp = hashlib.sha256("\n".join(frame_keys).encode("utf-8")).hexdigest()
    tracker = {
        "frame_001:10:20": 2,
        "frame_010:11:21": 1,
    }

    payload = chunk5.build_chunk5_checkpoint_payload(
        normal_mode="shadow",
        densify_enabled=True,
        densify_mode="active",
        densify_event_count=3,
        high_error_tracker=tracker,
        active_frame_keys=frame_keys,
    )

    assert payload["checkpoint_schema_version"] == "chunk5_normals_densify_v1"
    assert payload["active_frame_keys"] == frame_keys
    assert payload["active_frame_fingerprint"] == expected_fp
    assert payload["normal_mode"] == "shadow"
    assert payload["densify_enabled"] is True
    assert payload["densify_mode"] == "active"
    assert payload["densify_event_count"] == 3
    assert payload["high_error_tracker"] == tracker
    assert "w_normal" not in payload


@pytest.mark.filterwarnings("ignore:You are using `torch.load` with `weights_only=False`.*:FutureWarning")
def test_c5_t09_checkpoint_payload_torch_roundtrip(tmp_path, chunk5):
    payload = chunk5.build_chunk5_checkpoint_payload(
        normal_mode="active",
        densify_enabled=False,
        densify_mode="off",
        densify_event_count=0,
        high_error_tracker={"frame_000:5:7": 2},
        active_frame_keys=["f0", "f1"],
    )

    p = tmp_path / "chunk5_payload.pth"
    torch.save(payload, p)
    loaded = torch.load(p, map_location="cpu")
    _assert_payload_equal(payload, loaded)


def test_c5_t09_resume_policy_strict_raises_on_schema_mismatch(chunk5):
    with pytest.raises(ValueError, match="schema"):
        chunk5.resolve_chunk5_resume_action(
            checkpoint_schema_version="legacy",
            runtime_schema_version="chunk5_normals_densify_v1",
            frame_fingerprint_matches=True,
            mismatch_policy="strict",
        )


def test_c5_t09_resume_policy_strict_raises_on_frame_set_mismatch(chunk5):
    with pytest.raises(ValueError, match="frame-set mismatch"):
        chunk5.resolve_chunk5_resume_action(
            checkpoint_schema_version="chunk5_normals_densify_v1",
            runtime_schema_version="chunk5_normals_densify_v1",
            frame_fingerprint_matches=False,
            mismatch_policy="strict",
        )


def test_c5_t09_resume_policy_reset_paths_are_deterministic(chunk5):
    reset_frame = chunk5.resolve_chunk5_resume_action(
        checkpoint_schema_version="legacy",
        runtime_schema_version="chunk5_normals_densify_v1",
        frame_fingerprint_matches=False,
        mismatch_policy="reset_frame",
    )
    assert reset_frame == "reset_frame"

    reset_all = chunk5.resolve_chunk5_resume_action(
        checkpoint_schema_version="legacy",
        runtime_schema_version="chunk5_normals_densify_v1",
        frame_fingerprint_matches=False,
        mismatch_policy="reset_all",
    )
    assert reset_all == "reset_all"


def test_c5_t09_resume_policy_load_when_schema_and_fingerprint_match(chunk5):
    action = chunk5.resolve_chunk5_resume_action(
        checkpoint_schema_version="chunk5_normals_densify_v1",
        runtime_schema_version="chunk5_normals_densify_v1",
        frame_fingerprint_matches=True,
        mismatch_policy="strict",
    )
    assert action == "load"
