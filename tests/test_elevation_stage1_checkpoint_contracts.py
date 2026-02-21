import hashlib
import importlib.util
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


def test_active_frame_fingerprint_matches_sha256_contract():
    frame_keys = ["frame_001", "frame_010", "frame_020"]
    expected = hashlib.sha256("\n".join(frame_keys).encode("utf-8")).hexdigest()
    got = stage1.compute_active_frame_fingerprint(frame_keys)
    assert got == expected


def test_active_frame_fingerprint_empty_and_singleton_contracts():
    empty_expected = hashlib.sha256(b"").hexdigest()
    one_expected = hashlib.sha256("frame_only".encode("utf-8")).hexdigest()

    assert stage1.compute_active_frame_fingerprint([]) == empty_expected
    assert stage1.compute_active_frame_fingerprint(["frame_only"]) == one_expected


def test_checkpoint_payload_contains_required_stage1_fields():
    payload = stage1.build_stage1_checkpoint_payload(
        overlap_table={"frame_001": [("frame_010", 0.7)]},
        sampler_state={"cursor": 2, "epoch": 1},
        pixel_bank={"frame_001": {"rows": [1, 2], "cols": [3, 4], "logits_key": "frame_001"}},
        pixel_logits={"frame_001": [[0.0, 0.1, 0.2]]},
        optim_elev_state={"state": {}, "param_groups": []},
        active_frame_keys=["frame_001", "frame_010"],
    )

    assert payload["checkpoint_schema_version"] == "chunk3_stage1_v1"
    assert payload["active_frame_keys"] == ["frame_001", "frame_010"]
    assert payload["active_frame_fingerprint"] == stage1.compute_active_frame_fingerprint(
        ["frame_001", "frame_010"]
    )
    assert "overlap_table" in payload
    assert "pixel_bank" in payload
    assert "pixel_logits" in payload
    assert "optim_elev_state" in payload


def test_resume_policy_strict_raises_on_schema_mismatch():
    with pytest.raises(ValueError, match="schema"):
        stage1.resolve_stage1_resume_action(
            checkpoint_schema_version="legacy",
            runtime_schema_version="chunk3_stage1_v1",
            frame_fingerprint_matches=True,
            mismatch_policy="strict",
        )


def test_resume_policy_strict_raises_on_frame_set_mismatch_with_matching_schema():
    with pytest.raises(ValueError, match="frame-set mismatch"):
        stage1.resolve_stage1_resume_action(
            checkpoint_schema_version="chunk3_stage1_v1",
            runtime_schema_version="chunk3_stage1_v1",
            frame_fingerprint_matches=False,
            mismatch_policy="strict",
        )


def test_resume_policy_reset_frame_returns_reset_frame():
    action = stage1.resolve_stage1_resume_action(
        checkpoint_schema_version="legacy",
        runtime_schema_version="chunk3_stage1_v1",
        frame_fingerprint_matches=False,
        mismatch_policy="reset_frame",
    )
    assert action == "reset_frame"


def test_resume_policy_reset_all_returns_reset_all():
    action = stage1.resolve_stage1_resume_action(
        checkpoint_schema_version="legacy",
        runtime_schema_version="chunk3_stage1_v1",
        frame_fingerprint_matches=False,
        mismatch_policy="reset_all",
    )
    assert action == "reset_all"


def test_resume_policy_load_when_schema_and_fingerprint_match():
    action = stage1.resolve_stage1_resume_action(
        checkpoint_schema_version="chunk3_stage1_v1",
        runtime_schema_version="chunk3_stage1_v1",
        frame_fingerprint_matches=True,
        mismatch_policy="strict",
    )
    assert action == "load"


def test_checkpoint_payload_torch_roundtrip(tmp_path):
    payload = stage1.build_stage1_checkpoint_payload(
        overlap_table={"frame_001": [("frame_010", 0.7)]},
        sampler_state={"cursor": 2, "epoch": 1},
        pixel_bank={"frame_001": {"rows": [1, 2], "cols": [3, 4], "logits_key": "frame_001"}},
        pixel_logits={"frame_001": [[0.0, 0.1, 0.2]]},
        optim_elev_state={"state": {}, "param_groups": []},
        active_frame_keys=["frame_001", "frame_010"],
    )
    p = tmp_path / "stage1_payload.pth"
    torch.save(payload, p)
    loaded = torch.load(p, map_location="cpu", weights_only=True)
    assert loaded == payload


def test_frame_key_uniqueness_contract():
    stage1.assert_frame_keys_unique(["f0", "f1", "f2"])
    with pytest.raises(ValueError, match="Duplicate frame_key values"):
        stage1.assert_frame_keys_unique(["f0", "f1", "f0"])
