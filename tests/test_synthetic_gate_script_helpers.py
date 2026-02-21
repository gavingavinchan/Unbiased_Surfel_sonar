import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



@pytest.fixture(scope="module")
def gate_a():
    return _load_module(REPO_ROOT / "scripts" / "run_synthetic_a_gate.py", "run_synthetic_a_gate")


@pytest.fixture(scope="module")
def gate_c():
    return _load_module(REPO_ROOT / "scripts" / "run_synthetic_c_gate.py", "run_synthetic_c_gate")


def test_extract_eval_metrics_a_prefers_gt_trimmed_center_error(gate_a):
    payload = {
        "overall_pass": True,
        "ground_truth": {"radial_stats": {"mean": 0.01, "p95": 0.02}},
        "fit_sphere": {
            "center_error_m": 0.20,
            "by_mode": {
                "gt_trimmed": {
                    "center_error_m": 0.03,
                }
            },
        },
    }

    metrics = gate_a.extract_eval_metrics(payload)
    assert metrics["overall_pass"] is True
    assert metrics["mean_radial_error_m"] == 0.01
    assert metrics["p95_radial_error_m"] == 0.02
    assert metrics["center_error_m"] == 0.03


def test_extract_eval_metrics_a_falls_back_to_top_level_center_error(gate_a):
    payload = {
        "overall_pass": False,
        "ground_truth": {"radial_stats": {"mean": 0.11, "p95": 0.22}},
        "fit_sphere": {
            "center_error_m": 0.15,
            "by_mode": {},
        },
    }

    metrics = gate_a.extract_eval_metrics(payload)
    assert metrics["overall_pass"] is False
    assert metrics["center_error_m"] == 0.15


def test_extract_eval_metrics_c_prefers_gt_trimmed_center_error(gate_c):
    payload = {
        "overall_pass": True,
        "ground_truth": {"surface_stats": {"mean": 0.03, "p95": 0.07}},
        "fit_cube": {
            "center_error_m": 0.40,
            "by_mode": {
                "gt_trimmed": {
                    "center_error_m": 0.012,
                }
            },
        },
    }

    metrics = gate_c.extract_eval_metrics(payload)
    assert metrics["overall_pass"] is True
    assert metrics["mean_surface_error_m"] == 0.03
    assert metrics["p95_surface_error_m"] == 0.07
    assert metrics["center_error_m"] == 0.012


def test_extract_eval_metrics_c_falls_back_to_top_level_center_error(gate_c):
    payload = {
        "overall_pass": False,
        "ground_truth": {"surface_stats": {"mean": 0.13, "p95": 0.31}},
        "fit_cube": {
            "center_error_m": 0.19,
            "by_mode": {},
        },
    }

    metrics = gate_c.extract_eval_metrics(payload)
    assert metrics["overall_pass"] is False
    assert metrics["center_error_m"] == 0.19


def test_write_summary_markdown_a_includes_gate_and_drift_sections(tmp_path, gate_a):
    summary = {
        "generated_utc": "2026-02-18T00:00:00+00:00",
        "dataset_root": "/tmp/synth_a",
        "pose_mode": "sonar_equivalent",
        "overall_pass": True,
        "consistency_gate": {
            "pass": True,
            "median_pixel_error": 0.0,
            "mean_radial_residual_m": 0.01,
            "p95_radial_residual_m": 0.02,
            "fitted_center_error_m": 0.005,
        },
        "runs": [
            {
                "output_dir": "/tmp/run1",
                "metrics": {
                    "overall_pass": True,
                    "mean_radial_error_m": 0.01,
                    "p95_radial_error_m": 0.02,
                    "center_error_m": 0.005,
                },
            },
            {
                "output_dir": "/tmp/run2",
                "metrics": {
                    "overall_pass": True,
                    "mean_radial_error_m": 0.011,
                    "p95_radial_error_m": 0.021,
                    "center_error_m": 0.006,
                },
            },
        ],
        "reproducibility": {
            "pass": True,
            "thresholds": {
                "mean_radial_error_m_le": 0.005,
                "p95_radial_error_m_le": 0.010,
                "center_error_m_le": 0.005,
            },
            "drift": {
                "mean_radial_error_m": 0.001,
                "p95_radial_error_m": 0.001,
                "center_error_m": 0.001,
            },
        },
    }

    out_path = tmp_path / "a_summary.md"
    gate_a.write_summary_markdown(out_path, summary)

    text = out_path.read_text(encoding="utf-8")
    assert "# Synthetic Dataset A Gate Summary" in text
    assert "## Consistency Gate" in text
    assert "## Reproducibility Drift" in text
    assert "run1 pass" in text
    assert "run2 pass" in text


def test_write_summary_markdown_c_supports_surface_residual_keys(tmp_path, gate_c):
    summary = {
        "generated_utc": "2026-02-18T00:00:00+00:00",
        "dataset_root": "/tmp/synth_c",
        "pose_mode": "sonar_equivalent",
        "overall_pass": False,
        "consistency_gate": {
            "pass": True,
            "median_pixel_error": 0.0,
            "mean_surface_residual_m": 0.08,
            "p95_surface_residual_m": 0.20,
            "fitted_center_error_m": 0.01,
        },
        "runs": [],
        "reproducibility": {
            "pass": False,
            "thresholds": {
                "mean_surface_error_m_le": 0.005,
                "p95_surface_error_m_le": 0.010,
                "center_error_m_le": 0.005,
            },
            "drift": {
                "mean_surface_error_m": 0.0,
                "p95_surface_error_m": 0.0,
                "center_error_m": 0.0,
            },
        },
    }

    out_path = tmp_path / "c_summary.md"
    gate_c.write_summary_markdown(out_path, summary)

    text = out_path.read_text(encoding="utf-8")
    assert "# Synthetic Dataset C Gate Summary" in text
    assert "mean surface residual" in text
    assert "p95 surface residual" in text
    assert "- skipped: `true`" in text
