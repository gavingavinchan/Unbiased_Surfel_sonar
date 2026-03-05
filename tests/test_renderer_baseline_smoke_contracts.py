"""Renderer baseline smoke/integration contracts (RB-T11, RB-T12 runtime only)."""

import csv
import math
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_SCRIPT = REPO_ROOT / "debug_multiframe.py"
RUN_RUNTIME_SMOKES = os.environ.get("RUN_RENDERER_BASELINE_RUNTIME_SMOKES", "0") == "1"
RUNTIME_DATASET_PATH = Path(
    os.environ.get(
        "RENDERER_BASELINE_RUNTIME_DATASET_PATH",
        str(REPO_ROOT / "synthetic_datasets" / "synthetic_sphere_A_clean"),
    )
)


def _read_final_eval_stats(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"No rows in {csv_path}")

    losses = [float(r["total_loss"]) for r in rows]
    ssims = [float(r["ssim"]) for r in rows]
    assert all(math.isfinite(v) for v in losses + ssims), "Non-finite loss/ssim in final eval CSV"
    return {
        "loss_mean": sum(losses) / len(losses),
        "ssim_mean": sum(ssims) / len(ssims),
    }


def _run_debug_smoke(run_dir: Path, *, stage2_iters: str = "8"):
    env = os.environ.copy()
    env.update(
        {
            "SONAR_DATASET": "synthetic_a_clean",
            "SONAR_DATASET_PATH": str(RUNTIME_DATASET_PATH),
            "SONAR_OUTPUT_DIR": str(run_dir),
            "SONAR_NUM_FRAMES": os.environ.get("RENDERER_BASELINE_RUNTIME_NUM_FRAMES", "16"),
            "SONAR_STAGE2_ITERS": stage2_iters,
            "SONAR_STAGE3_ITERS": os.environ.get("RENDERER_BASELINE_RUNTIME_STAGE3_ITERS", "1"),
            "SONAR_FREEZE_SCALE": "1",
            "SONAR_RENDER_MODE": os.environ.get("RENDERER_BASELINE_RUNTIME_RENDER_MODE", "2dgs"),
            "SONAR_OCCLUSION_MODE": os.environ.get("RENDERER_BASELINE_RUNTIME_OCCLUSION_MODE", "none"),
            "SONAR_LAMBERTIAN_MODE": os.environ.get("RENDERER_BASELINE_RUNTIME_LAMBERTIAN_MODE", "clamp0"),
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "renderer_smoke.log"
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

    eval_csv = run_dir / "final_eval_train_frames.csv"
    assert eval_csv.exists(), f"Missing artifact: {eval_csv}"

    return {
        "log_path": log_path,
        "stats": _read_final_eval_stats(eval_csv),
    }


def test_rb_t11_rb_t12_debug_script_compiles_for_smoke_contracts():
    py_compile.compile(str(DEBUG_SCRIPT), doraise=True)


@pytest.mark.skipif(
    not RUN_RUNTIME_SMOKES,
    reason="Set RUN_RENDERER_BASELINE_RUNTIME_SMOKES=1 to run runtime smokes",
)
def test_rb_t11_runtime_shadow_mode_has_finite_losses(tmp_path):
    if not RUNTIME_DATASET_PATH.exists():
        pytest.skip(f"Runtime dataset not found: {RUNTIME_DATASET_PATH}")

    out = _run_debug_smoke(tmp_path / "rb_t11_shadow", stage2_iters="8")
    assert math.isfinite(out["stats"]["loss_mean"])
    assert math.isfinite(out["stats"]["ssim_mean"])


@pytest.mark.skipif(
    not RUN_RUNTIME_SMOKES,
    reason="Set RUN_RENDERER_BASELINE_RUNTIME_SMOKES=1 to run runtime smokes",
)
def test_rb_t12_runtime_log_emits_sonar_diagnostics(tmp_path):
    if not RUNTIME_DATASET_PATH.exists():
        pytest.skip(f"Runtime dataset not found: {RUNTIME_DATASET_PATH}")

    out = _run_debug_smoke(tmp_path / "rb_t12_active", stage2_iters="8")
    log_text = out["log_path"].read_text(encoding="utf-8")
    assert "attenuation enabled=" in log_text
    assert "nan_inf=" in log_text
