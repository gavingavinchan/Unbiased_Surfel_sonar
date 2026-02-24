import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_A_GATE = REPO_ROOT / "scripts" / "run_synthetic_a_gate.py"
RUN_C_GATE = REPO_ROOT / "scripts" / "run_synthetic_c_gate.py"
EVAL_CUBE = REPO_ROOT / "scripts" / "eval_synthetic_cube.py"
DEBUG_SCRIPT = REPO_ROOT / "debug_multiframe.py"

RUN_SYNTH_MATRIX = os.environ.get("RUN_CHUNK4_SYNTHETIC_MATRIX", "0") == "1"
DATASET_A = Path(
    os.environ.get(
        "CHUNK4_DATASET_A_PATH",
        str(REPO_ROOT / "synthetic_datasets" / "synthetic_sphere_A_clean"),
    )
)
DATASET_C = Path(
    os.environ.get(
        "CHUNK4_DATASET_C_PATH",
        str(REPO_ROOT / "synthetic_datasets" / "synthetic_cube_C_clean"),
    )
)


def _run_command(cmd, *, env=None, log_path: Path):
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def _allow_gate_exit(code: int):
    # Gate scripts/evaluators return 2 when thresholds fail, which is acceptable
    # for low-iteration smoke/regression plumbing tests.
    assert code in {0, 2}, f"Unexpected process exit code: {code}"


@pytest.mark.skipif(not RUN_SYNTH_MATRIX, reason="Set RUN_CHUNK4_SYNTHETIC_MATRIX=1 to run synthetic matrix tests")
def test_c4_t15_synthetic_matrix_gate_scripts(tmp_path):
    if not DATASET_A.exists():
        pytest.skip(f"Dataset A not found: {DATASET_A}")
    if not DATASET_C.exists():
        pytest.skip(f"Dataset C not found: {DATASET_C}")

    output_root = tmp_path / "synthetic_matrix"
    output_root.mkdir(parents=True, exist_ok=True)

    a_summary_json = output_root / "a_gate_summary.json"
    a_summary_md = output_root / "a_gate_summary.md"
    a_cmd = [
        sys.executable,
        str(RUN_A_GATE),
        "--dataset-root",
        str(DATASET_A),
        "--reuse-dataset",
        "--num-frames",
        os.environ.get("CHUNK4_SYNTH_NUM_FRAMES", "32"),
        "--stage2-iters",
        os.environ.get("CHUNK4_SYNTH_STAGE2_ITERS", "12"),
        "--stage3-iters",
        os.environ.get("CHUNK4_SYNTH_STAGE3_ITERS", "1"),
        "--output-root",
        str(output_root),
        "--run-prefix",
        "chunk4_synth_a",
        "--overwrite-runs",
        "--summary-json",
        str(a_summary_json),
        "--summary-md",
        str(a_summary_md),
    ]
    a_code = _run_command(a_cmd, log_path=output_root / "a_gate.log")
    _allow_gate_exit(a_code)
    assert a_summary_json.exists()
    assert a_summary_md.exists()
    a_summary = json.loads(a_summary_json.read_text(encoding="utf-8"))
    assert "runs" in a_summary
    assert len(a_summary["runs"]) == 2

    c_summary_json = output_root / "c_gate_summary.json"
    c_summary_md = output_root / "c_gate_summary.md"
    c_cmd = [
        sys.executable,
        str(RUN_C_GATE),
        "--dataset-root",
        str(DATASET_C),
        "--reuse-dataset",
        "--num-frames",
        os.environ.get("CHUNK4_SYNTH_NUM_FRAMES", "32"),
        "--stage2-iters",
        os.environ.get("CHUNK4_SYNTH_STAGE2_ITERS", "12"),
        "--stage3-iters",
        os.environ.get("CHUNK4_SYNTH_STAGE3_ITERS", "1"),
        "--output-root",
        str(output_root),
        "--run-prefix",
        "chunk4_synth_c",
        "--overwrite-runs",
        "--summary-json",
        str(c_summary_json),
        "--summary-md",
        str(c_summary_md),
    ]
    c_code = _run_command(c_cmd, log_path=output_root / "c_gate.log")
    _allow_gate_exit(c_code)
    assert c_summary_json.exists()
    assert c_summary_md.exists()
    c_summary = json.loads(c_summary_json.read_text(encoding="utf-8"))
    assert "runs" in c_summary
    assert len(c_summary["runs"]) == 2


@pytest.mark.skipif(not RUN_SYNTH_MATRIX, reason="Set RUN_CHUNK4_SYNTHETIC_MATRIX=1 to run synthetic matrix tests")
def test_c4_t16_synthetic_continuation_resume_state_check(tmp_path):
    if not DATASET_C.exists():
        pytest.skip(f"Dataset C not found: {DATASET_C}")

    run1 = tmp_path / "c4_s4_run1"
    run2 = tmp_path / "c4_s4_run2"
    checkpoint = run1 / "chunk4_ckpt.pth"

    env1 = os.environ.copy()
    env1.update(
        {
            "SONAR_DATASET": "synthetic_c_clean",
            "SONAR_DATASET_PATH": str(DATASET_C),
            "SONAR_OUTPUT_DIR": str(run1),
            "SONAR_NUM_FRAMES": os.environ.get("CHUNK4_SYNTH_NUM_FRAMES", "32"),
            "SONAR_STAGE2_ITERS": os.environ.get("CHUNK4_SYNTH_STAGE2_ITERS", "12"),
            "SONAR_STAGE3_ITERS": os.environ.get("CHUNK4_SYNTH_STAGE3_ITERS", "1"),
            "SONAR_FREEZE_SCALE": "1",
            "ELEVATION_AWARE": "1",
            "ELEV_STAGE1_MODE": "active",
            "ELEV_COUPLE_MODE": "active",
            "ELEV_SUPPORT_MODE": "active",
            "SONAR_SAVE_CHECKPOINT": str(checkpoint),
        }
    )
    code1 = _run_command([sys.executable, str(DEBUG_SCRIPT)], env=env1, log_path=tmp_path / "c4_s4_run1.log")
    assert code1 == 0
    assert checkpoint.exists()
    assert (run1 / "surfels_after_training.ply").exists()

    env2 = os.environ.copy()
    env2.update(
        {
            "SONAR_DATASET": "synthetic_c_clean",
            "SONAR_DATASET_PATH": str(DATASET_C),
            "SONAR_OUTPUT_DIR": str(run2),
            "SONAR_NUM_FRAMES": os.environ.get("CHUNK4_SYNTH_NUM_FRAMES", "32"),
            "SONAR_STAGE2_ITERS": os.environ.get("CHUNK4_SYNTH_STAGE2_ITERS", "12"),
            "SONAR_STAGE3_ITERS": os.environ.get("CHUNK4_SYNTH_STAGE3_ITERS", "1"),
            "SONAR_FREEZE_SCALE": "1",
            "ELEVATION_AWARE": "1",
            "ELEV_STAGE1_MODE": "active",
            "ELEV_COUPLE_MODE": "active",
            "ELEV_SUPPORT_MODE": "active",
            "SONAR_LOAD_CHECKPOINT": str(checkpoint),
        }
    )
    run2_log = tmp_path / "c4_s4_run2.log"
    code2 = _run_command([sys.executable, str(DEBUG_SCRIPT)], env=env2, log_path=run2_log)
    assert code2 == 0
    assert (run2 / "surfels_after_training.ply").exists()
    assert (run2 / "support_metrics_train.csv").exists()

    log_text = run2_log.read_text(encoding="utf-8")
    assert "[Checkpoint] Loaded:" in log_text

    eval_dir = run2 / "eval_surfel"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_json = eval_dir / "cube_eval.json"
    eval_cmd = [
        sys.executable,
        str(EVAL_CUBE),
        "--reconstruction",
        str(run2 / "surfels_after_training.ply"),
        "--dataset-root",
        str(DATASET_C),
        "--output-dir",
        str(eval_dir),
        "--fit-mode",
        "both",
    ]
    eval_code = _run_command(eval_cmd, log_path=tmp_path / "c4_s4_eval.log")
    _allow_gate_exit(eval_code)
    assert eval_json.exists()


def test_c4_t17_manual_visual_panel_required_placeholder():
    pytest.skip(
        "Manual required: inspect Dataset-C artifact panel and record improved|unchanged|regressed verdict"
    )
