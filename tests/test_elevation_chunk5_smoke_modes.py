"""Chunk-5 smoke tests.

This file keeps tiny placeholder checks and opt-in runtime smokes for the new
late-normal and densify paths in ``debug_multiframe.py``.
"""

import csv
import importlib.util
import math
import os
import py_compile
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "utils" / "elevation_chunk5_helpers.py"
DEBUG_SCRIPT = REPO_ROOT / "debug_multiframe.py"
HAS_HELPER = HELPER_PATH.exists()
RUN_RUNTIME_SMOKES = os.environ.get("RUN_CHUNK5_RUNTIME_SMOKES", "0") == "1"
RUNTIME_DATASET_PATH = Path(
    os.environ.get(
        "CHUNK5_RUNTIME_DATASET_PATH",
        str(REPO_ROOT / "synthetic_datasets" / "synthetic_sphere_A_clean"),
    )
)


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


def test_c5_t01_debug_multiframe_compiles_for_smoke_modes():
    py_compile.compile(str(DEBUG_SCRIPT), doraise=True)


def test_c5_t04_placeholder_off_mode_parity_toy_aggregation(chunk5):
    baseline = 1.25
    loss_normal = 0.40
    w_normal = 0.10

    normal_mode, densify_mode = chunk5.resolve_effective_chunk5_modes(
        elevation_aware=False,
        requested_normal_mode="active",
        densify_enabled=True,
        requested_densify_mode="active",
    )

    weighted_normal = (w_normal * loss_normal) if chunk5.mode_enables_normal_loss(normal_mode) else 0.0
    total = baseline + weighted_normal

    assert normal_mode == "off"
    assert densify_mode == "off"
    assert total == pytest.approx(baseline)


def _read_final_eval_stats(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
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
    normal_mode: str,
    densify_mode: str,
    densify_enabled: str,
    stage2_iters: str,
    num_frames: str,
    extra_env=None,
):
    env = os.environ.copy()
    env.update(
        {
            "SONAR_DATASET": "synthetic_a_clean",
            "SONAR_DATASET_PATH": str(RUNTIME_DATASET_PATH),
            "SONAR_OUTPUT_DIR": str(run_dir),
            "SONAR_NUM_FRAMES": num_frames,
            "SONAR_STAGE2_ITERS": stage2_iters,
            "SONAR_STAGE3_ITERS": os.environ.get("CHUNK5_RUNTIME_STAGE3_ITERS", "1"),
            "SONAR_FREEZE_SCALE": "1",
            "SONAR_RENDER_MODE": "2dgs",
            "SONAR_OCCLUSION_MODE": "ray_binned",
            "SONAR_LAMBERTIAN_MODE": "leaky",
            "ELEVATION_AWARE": "1",
            "ELEV_STAGE1_MODE": "active",
            "ELEV_NORMAL_MODE": normal_mode,
            "ELEV_NORMAL_RAMP_START_ITER": "1",
            "ELEV_NORMAL_RAMP_END_ITER": "4",
            "ELEV_NORMAL_ELEV_START_ITER": "1",
            "ELEV_DENSIFY": densify_enabled,
            "ELEV_DENSIFY_MODE": densify_mode,
            "ELEV_STAGE2_START_ITER": "1",
            "ELEV_DENSIFY_INTERVAL": "1",
            "ELEV_DENSIFY_MIN_INTENSITY": "0.0",
            "ELEV_DENSIFY_RESIDUAL_THRESH": "0.0",
            "ELEV_DENSIFY_MAX_PER_EVENT": "8",
            "ELEV_DENSIFY_MULTI_VIEW_MIN_SCORE": "0.0",
        }
    )
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

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
    assert surfel_path.exists(), f"Missing artifact: {surfel_path}"
    assert eval_csv.exists(), f"Missing artifact: {eval_csv}"

    log_text = log_path.read_text(encoding="utf-8")
    return {
        "run_dir": run_dir,
        "log_path": log_path,
        "eval_csv": eval_csv,
        "log_text": log_text,
        "stats": _read_final_eval_stats(eval_csv),
    }


@pytest.mark.skipif(not RUN_RUNTIME_SMOKES, reason="Set RUN_CHUNK5_RUNTIME_SMOKES=1 to run runtime smokes")
def test_c5_t11_runtime_active_normals_smoke(tmp_path):
    if not RUNTIME_DATASET_PATH.exists():
        pytest.skip(f"Runtime dataset not found: {RUNTIME_DATASET_PATH}")

    out = _run_debug_smoke(
        tmp_path / "active_normals",
        normal_mode="active",
        densify_mode="off",
        densify_enabled="0",
        stage2_iters=os.environ.get("CHUNK5_RUNTIME_STAGE2_ITERS", "8"),
        num_frames=os.environ.get("CHUNK5_RUNTIME_NUM_FRAMES", "8"),
    )

    assert math.isfinite(out["stats"]["loss_mean"])
    assert math.isfinite(out["stats"]["ssim_mean"])
    assert "normal=" in out["log_text"]
    assert "conf_cov=" in out["log_text"]


@pytest.mark.skipif(not RUN_RUNTIME_SMOKES, reason="Set RUN_CHUNK5_RUNTIME_SMOKES=1 to run runtime smokes")
def test_c5_t13_runtime_active_densify_smoke(tmp_path):
    if not RUNTIME_DATASET_PATH.exists():
        pytest.skip(f"Runtime dataset not found: {RUNTIME_DATASET_PATH}")

    out = _run_debug_smoke(
        tmp_path / "active_densify",
        normal_mode="shadow",
        densify_mode="active",
        densify_enabled="1",
        stage2_iters=os.environ.get("CHUNK5_RUNTIME_DENSIFY_STAGE2_ITERS", "10"),
        num_frames=os.environ.get("CHUNK5_RUNTIME_DENSIFY_NUM_FRAMES", "2"),
    )

    assert math.isfinite(out["stats"]["loss_mean"])
    assert math.isfinite(out["stats"]["ssim_mean"])
    assert "densify_mode=active" in out["log_text"]
    assert "spawn=" in out["log_text"]
    assert "cand=" in out["log_text"]
    assert "supp=" in out["log_text"]
