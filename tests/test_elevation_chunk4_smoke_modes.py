"""Chunk-4 smoke-mode placeholders.

These tests lock mode-gate semantics in a tiny deterministic fixture before the
runtime integration path is fully wired. They are intentionally lightweight and
are expected to be complemented by real debug_multiframe runtime smokes.
"""

import importlib.util
import csv
import math
import os
import py_compile
import random
import subprocess
import sys
from typing import Optional
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "utils" / "elevation_chunk4_helpers.py"
DEBUG_SCRIPT = REPO_ROOT / "debug_multiframe.py"
HAS_HELPER = HELPER_PATH.exists()
RUN_RUNTIME_SMOKES = os.environ.get("RUN_CHUNK4_RUNTIME_SMOKES", "0") == "1"
RUNTIME_DATASET_PATH = Path(
    os.environ.get(
        "CHUNK4_RUNTIME_DATASET_PATH",
        str(REPO_ROOT / "synthetic_datasets" / "synthetic_sphere_A_clean"),
    )
)


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


@pytest.mark.skipif(not HAS_HELPER, reason="Chunk4 helper module not created yet")
def test_c4_t01_chunk4_helper_module_compiles_for_smoke_modes():
    py_compile.compile(str(HELPER_PATH), doraise=True)


def test_c4_t01_debug_multiframe_compiles_for_smoke_modes():
    target = REPO_ROOT / "debug_multiframe.py"
    py_compile.compile(str(target), doraise=True)


def test_c4_t11_placeholder_off_mode_parity_toy_aggregation(chunk4):
    baseline = 1.25
    loss_couple = 0.75
    w_couple = 0.50

    couple_mode, support_mode = chunk4.resolve_effective_chunk4_modes(
        elevation_aware=True,
        requested_couple_mode="off",
        requested_support_mode="off",
    )

    weighted_couple = (w_couple * loss_couple) if chunk4.mode_enables_weighted_coupling(couple_mode) else 0.0
    total = baseline + weighted_couple

    assert support_mode == "off"
    assert total == pytest.approx(baseline)


def test_c4_t12_placeholder_shadow_mode_smoke_diagnostics_only(chunk4):
    couple_mode, support_mode = chunk4.resolve_effective_chunk4_modes(
        elevation_aware=True,
        requested_couple_mode="shadow",
        requested_support_mode="shadow",
    )
    assert couple_mode == "shadow"
    assert support_mode == "shadow"
    assert not chunk4.mode_enables_weighted_coupling(couple_mode)
    assert not chunk4.mode_enables_hard_prune(support_mode)


def test_c4_t13_placeholder_active_mode_smoke_enables_coupling_and_prune(chunk4):
    couple_mode, support_mode = chunk4.resolve_effective_chunk4_modes(
        elevation_aware=True,
        requested_couple_mode="active",
        requested_support_mode="active",
    )
    assert chunk4.mode_enables_weighted_coupling(couple_mode)
    assert chunk4.mode_enables_hard_prune(support_mode)


def test_c4_t14_placeholder_resume_smoke_matching_schema_load_action(chunk4):
    action = chunk4.resolve_chunk4_resume_action(
        checkpoint_schema_version=chunk4.CHECKPOINT_SCHEMA_VERSION,
        runtime_schema_version=chunk4.CHECKPOINT_SCHEMA_VERSION,
        frame_fingerprint_matches=True,
        mismatch_policy="strict",
    )
    assert action == "load"


def _read_final_eval_stats(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise AssertionError(f"No rows in {csv_path}")

    losses = [float(r["total_loss"]) for r in rows]
    ssims = [float(r["ssim"]) for r in rows]
    if not all(math.isfinite(v) for v in losses + ssims):
        raise AssertionError("Non-finite loss/ssim values in final eval CSV")
    return {
        "loss_mean": sum(losses) / len(losses),
        "ssim_mean": sum(ssims) / len(ssims),
    }


def _run_debug_smoke(
    run_dir: Path,
    *,
    couple_mode: str,
    support_mode: str,
    elevation_aware: str = "1",
    save_checkpoint: Optional[Path] = None,
    load_checkpoint: Optional[Path] = None,
):
    env = os.environ.copy()
    env.update(
        {
            "SONAR_DATASET": "synthetic_a_clean",
            "SONAR_DATASET_PATH": str(RUNTIME_DATASET_PATH),
            "SONAR_OUTPUT_DIR": str(run_dir),
            "SONAR_NUM_FRAMES": os.environ.get("CHUNK4_RUNTIME_NUM_FRAMES", "16"),
            "SONAR_STAGE2_ITERS": os.environ.get("CHUNK4_RUNTIME_STAGE2_ITERS", "8"),
            "SONAR_STAGE3_ITERS": os.environ.get("CHUNK4_RUNTIME_STAGE3_ITERS", "1"),
            "SONAR_FREEZE_SCALE": "1",
            "ELEVATION_AWARE": elevation_aware,
            "ELEV_STAGE1_MODE": "active",
            "ELEV_COUPLE_MODE": couple_mode,
            "ELEV_SUPPORT_MODE": support_mode,
        }
    )
    if save_checkpoint is not None:
        env["SONAR_SAVE_CHECKPOINT"] = str(save_checkpoint)
    if load_checkpoint is not None:
        env["SONAR_LOAD_CHECKPOINT"] = str(load_checkpoint)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "runtime_smoke.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            [sys.executable, str(DEBUG_SCRIPT)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise AssertionError(f"debug_multiframe.py failed (code={proc.returncode}); see {log_path}")

    surfel_path = run_dir / "surfels_after_training.ply"
    eval_csv = run_dir / "final_eval_train_frames.csv"
    support_csv = run_dir / "support_metrics_train.csv"
    visits_csv = run_dir / "frame_training_visits.csv"
    assert surfel_path.exists(), f"Missing artifact: {surfel_path}"
    assert eval_csv.exists(), f"Missing artifact: {eval_csv}"
    assert support_csv.exists(), f"Missing artifact: {support_csv}"
    assert visits_csv.exists(), f"Missing artifact: {visits_csv}"

    return {
        "run_dir": run_dir,
        "log_path": log_path,
        "eval_csv": eval_csv,
        "stats": _read_final_eval_stats(eval_csv),
    }


@pytest.mark.skipif(not RUN_RUNTIME_SMOKES, reason="Set RUN_CHUNK4_RUNTIME_SMOKES=1 to run runtime smokes")
def test_c4_t11_runtime_off_mode_parity_smoke(tmp_path):
    if not RUNTIME_DATASET_PATH.exists():
        pytest.skip(f"Runtime dataset not found: {RUNTIME_DATASET_PATH}")

    baseline = _run_debug_smoke(
        tmp_path / "baseline_elev0",
        couple_mode="off",
        support_mode="off",
        elevation_aware="0",
    )
    offmode = _run_debug_smoke(
        tmp_path / "offmode_elev1",
        couple_mode="off",
        support_mode="off",
        elevation_aware="1",
    )

    b_loss = baseline["stats"]["loss_mean"]
    o_loss = offmode["stats"]["loss_mean"]
    rel_loss_delta = abs(o_loss - b_loss) / max(abs(b_loss), 1e-8)
    abs_ssim_delta = abs(offmode["stats"]["ssim_mean"] - baseline["stats"]["ssim_mean"])

    assert rel_loss_delta <= 0.05
    assert abs_ssim_delta <= 0.01


@pytest.mark.skipif(not RUN_RUNTIME_SMOKES, reason="Set RUN_CHUNK4_RUNTIME_SMOKES=1 to run runtime smokes")
def test_c4_t12_runtime_shadow_mode_smoke(tmp_path):
    if not RUNTIME_DATASET_PATH.exists():
        pytest.skip(f"Runtime dataset not found: {RUNTIME_DATASET_PATH}")
    out = _run_debug_smoke(
        tmp_path / "shadow_mode",
        couple_mode="shadow",
        support_mode="shadow",
    )
    assert math.isfinite(out["stats"]["loss_mean"])
    assert math.isfinite(out["stats"]["ssim_mean"])


@pytest.mark.skipif(not RUN_RUNTIME_SMOKES, reason="Set RUN_CHUNK4_RUNTIME_SMOKES=1 to run runtime smokes")
def test_c4_t13_runtime_active_mode_smoke(tmp_path):
    if not RUNTIME_DATASET_PATH.exists():
        pytest.skip(f"Runtime dataset not found: {RUNTIME_DATASET_PATH}")
    out = _run_debug_smoke(
        tmp_path / "active_mode",
        couple_mode="active",
        support_mode="active",
    )
    assert math.isfinite(out["stats"]["loss_mean"])
    assert math.isfinite(out["stats"]["ssim_mean"])


@pytest.mark.skipif(not RUN_RUNTIME_SMOKES, reason="Set RUN_CHUNK4_RUNTIME_SMOKES=1 to run runtime smokes")
def test_c4_t14_runtime_resume_smoke(tmp_path):
    if not RUNTIME_DATASET_PATH.exists():
        pytest.skip(f"Runtime dataset not found: {RUNTIME_DATASET_PATH}")

    ckpt = tmp_path / "chunk4_runtime_ckpt.pth"
    _ = _run_debug_smoke(
        tmp_path / "resume_run1",
        couple_mode="active",
        support_mode="active",
        save_checkpoint=ckpt,
    )
    assert ckpt.exists(), f"Missing checkpoint: {ckpt}"

    resumed = _run_debug_smoke(
        tmp_path / "resume_run2",
        couple_mode="active",
        support_mode="active",
        load_checkpoint=ckpt,
    )
    log_text = resumed["log_path"].read_text(encoding="utf-8")
    assert "[Checkpoint] Loaded:" in log_text
