#!/usr/bin/env python3
"""
Run the synthetic Dataset C gate end-to-end:
generate -> consistency gate -> train x2 -> evaluate x2 -> reproducibility summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "generate_synthetic_sonar_dataset.py"
EVALUATOR_SCRIPT = REPO_ROOT / "scripts" / "eval_synthetic_cube.py"
TRAIN_SCRIPT = REPO_ROOT / "debug_multiframe.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic Dataset C acceptance gate")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./synthetic_datasets/synthetic_cube_C_clean"),
        help="Dataset output path used by generator and training",
    )
    parser.add_argument("--variant", choices=["C_clean", "C_noisy"], default="C_clean")
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elevation-samples", type=int, default=64)
    parser.add_argument(
        "--pose-mode",
        choices=["sonar_equivalent", "camera_with_extrinsic"],
        default="sonar_equivalent",
        help=(
            "Generator pose mode. sonar_equivalent is the canonical Dataset C gate mode. "
            "camera_with_extrinsic is diagnostic-only."
        ),
    )
    parser.add_argument(
        "--pose-policy",
        choices=["auto", "orbit_sweep", "multi_band"],
        default="auto",
        help="Pose coverage policy. Dataset C default is multi_band via auto.",
    )
    parser.add_argument("--reuse-dataset", action="store_true", help="Skip generation and reuse existing dataset")
    parser.add_argument("--no-loader-check", dest="run_loader_check", action="store_false")
    parser.set_defaults(run_loader_check=True)

    parser.add_argument("--output-root", type=Path, default=Path("./output"))
    parser.add_argument("--run-prefix", type=str, default="debug_multiframe_synth_c")
    parser.add_argument("--stage2-iters", type=int, default=1000)
    parser.add_argument("--stage3-iters", type=int, default=1)
    parser.add_argument("--freeze-scale", type=int, choices=[0, 1], default=1)
    parser.add_argument("--overwrite-runs", action="store_true", help="Delete existing run output folders")
    parser.add_argument("--skip-training", action="store_true", help="Stop after dataset generation and gate")

    parser.add_argument("--drift-mean-threshold", type=float, default=0.005)
    parser.add_argument("--drift-p95-threshold", type=float, default=0.010)
    parser.add_argument("--drift-center-threshold", type=float, default=0.005)

    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    return parser.parse_args()


def run_command(
    command: List[str],
    *,
    env: Dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(command)}")
    proc = subprocess.run(command, check=False, env=env, cwd=str(cwd))
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return proc


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_eval_metrics(eval_json: Dict) -> Dict[str, float | bool]:
    gt = eval_json["ground_truth"]["surface_stats"]
    fit = eval_json.get("fit_cube", {})
    by_mode = fit.get("by_mode", {}) if isinstance(fit, dict) else {}
    if isinstance(by_mode, dict) and "gt_trimmed" in by_mode:
        center_error = float(by_mode["gt_trimmed"]["center_error_m"])
    else:
        center_error = float(fit.get("center_error_m", float("inf")))

    return {
        "overall_pass": bool(eval_json.get("overall_pass", False)),
        "mean_surface_error_m": float(gt["mean"]),
        "p95_surface_error_m": float(gt["p95"]),
        "center_error_m": center_error,
    }


def write_summary_markdown(path: Path, summary: Dict) -> None:
    gate = summary["consistency_gate"]
    rep = summary["reproducibility"]
    run1 = summary["runs"][0] if summary["runs"] else None
    run2 = summary["runs"][1] if len(summary["runs"]) > 1 else None

    mean_key = "mean_surface_residual_m" if "mean_surface_residual_m" in gate else "mean_radial_residual_m"
    p95_key = "p95_surface_residual_m" if "p95_surface_residual_m" in gate else "p95_radial_residual_m"

    lines = [
        "# Synthetic Dataset C Gate Summary",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Pose mode: `{summary['pose_mode']}`",
        f"- Overall pass: `{summary['overall_pass']}`",
        "",
        "## Consistency Gate",
        "",
        f"- pass: `{gate.get('pass', False)}`",
        f"- median pixel error: `{gate.get('median_pixel_error', float('inf')):.6f}`",
        f"- mean surface residual (m): `{gate.get(mean_key, float('inf')):.6f}`",
        f"- p95 surface residual (m): `{gate.get(p95_key, float('inf')):.6f}`",
        f"- fitted center error (m): `{gate.get('fitted_center_error_m', float('inf')):.6f}`",
        "",
        "## Training Runs",
        "",
    ]

    if run1 is None:
        lines.append("- skipped: `true`")
    else:
        lines.extend(
            [
                f"- run1 pass: `{run1['metrics']['overall_pass']}` @ `{run1['output_dir']}`",
                f"- run1 mean/p95/center (m): `{run1['metrics']['mean_surface_error_m']:.6f} / {run1['metrics']['p95_surface_error_m']:.6f} / {run1['metrics']['center_error_m']:.6f}`",
            ]
        )

    if run2 is not None:
        lines.extend(
            [
                f"- run2 pass: `{run2['metrics']['overall_pass']}` @ `{run2['output_dir']}`",
                f"- run2 mean/p95/center (m): `{run2['metrics']['mean_surface_error_m']:.6f} / {run2['metrics']['p95_surface_error_m']:.6f} / {run2['metrics']['center_error_m']:.6f}`",
                "",
                "## Reproducibility Drift",
                "",
                f"- pass: `{rep['pass']}`",
                f"- |delta mean| (m): `{rep['drift']['mean_surface_error_m']:.6f}` (<= `{rep['thresholds']['mean_surface_error_m_le']:.6f}`)",
                f"- |delta p95| (m): `{rep['drift']['p95_surface_error_m']:.6f}` (<= `{rep['thresholds']['p95_surface_error_m_le']:.6f}`)",
                f"- |delta center| (m): `{rep['drift']['center_error_m']:.6f}` (<= `{rep['thresholds']['center_error_m_le']:.6f}`)",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.summary_json is None:
        summary_json = output_root / f"{args.run_prefix}_gate_summary.json"
    else:
        summary_json = args.summary_json.expanduser().resolve()
    if args.summary_md is None:
        summary_md = output_root / f"{args.run_prefix}_gate_summary.md"
    else:
        summary_md = args.summary_md.expanduser().resolve()

    if not args.reuse_dataset:
        generator_cmd = [
            args.python,
            str(GENERATOR_SCRIPT),
            "--output-dir",
            str(dataset_root),
            "--variant",
            args.variant,
            "--num-frames",
            str(args.num_frames),
            "--seed",
            str(args.seed),
            "--elevation-samples",
            str(args.elevation_samples),
            "--pose-mode",
            args.pose_mode,
            "--pose-policy",
            args.pose_policy,
            "--overwrite",
        ]
        if not args.run_loader_check:
            generator_cmd.append("--no-loader-check")
        run_command(generator_cmd)

    consistency_gate_path = dataset_root / "consistency_gate.json"
    consistency_gate = load_json(consistency_gate_path)

    summary = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_root": str(dataset_root),
        "pose_mode": args.pose_mode,
        "consistency_gate": consistency_gate,
        "runs": [],
        "reproducibility": {
            "pass": False,
            "thresholds": {
                "mean_surface_error_m_le": float(args.drift_mean_threshold),
                "p95_surface_error_m_le": float(args.drift_p95_threshold),
                "center_error_m_le": float(args.drift_center_threshold),
            },
            "drift": {
                "mean_surface_error_m": float("inf"),
                "p95_surface_error_m": float("inf"),
                "center_error_m": float("inf"),
            },
        },
        "overall_pass": bool(consistency_gate.get("pass", False)),
    }

    if not args.skip_training:
        run_names = [f"{args.run_prefix}_run1", f"{args.run_prefix}_run2"]
        for run_name in run_names:
            run_dir = (output_root / run_name).resolve()
            if run_dir.exists():
                if args.overwrite_runs:
                    shutil.rmtree(run_dir)
                else:
                    raise FileExistsError(
                        f"Run output directory exists: {run_dir}. Use --overwrite-runs to replace it."
                    )

            train_env = os.environ.copy()
            train_env.update(
                {
                    "SONAR_DATASET": "synthetic_c_clean",
                    "SONAR_DATASET_PATH": str(dataset_root),
                    "SONAR_OUTPUT_DIR": str(run_dir),
                    "SONAR_NUM_FRAMES": str(args.num_frames),
                    "SONAR_STAGE2_ITERS": str(args.stage2_iters),
                    "SONAR_STAGE3_ITERS": str(args.stage3_iters),
                    "SONAR_FREEZE_SCALE": str(args.freeze_scale),
                }
            )

            train_proc = run_command([args.python, str(TRAIN_SCRIPT)], env=train_env, check=False)

            recon_path = run_dir / "surfels_after_training.ply"
            if not recon_path.exists():
                if train_proc.returncode != 0:
                    print(f"[warn] Training process returned code {train_proc.returncode}")
                raise FileNotFoundError(f"Missing reconstruction output: {recon_path}")
            if train_proc.returncode != 0:
                print(
                    f"[warn] Training process returned code {train_proc.returncode} "
                    "but reconstruction artifact exists; continuing."
                )

            eval_dir = run_dir / "eval_surfel"
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_cmd = [
                args.python,
                str(EVALUATOR_SCRIPT),
                "--reconstruction",
                str(recon_path),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(eval_dir),
                "--fit-mode",
                "both",
            ]
            eval_json_path = eval_dir / "cube_eval.json"
            eval_proc = run_command(eval_cmd, check=False)
            if not eval_json_path.exists():
                if eval_proc.returncode != 0:
                    print(f"[warn] Evaluator process returned code {eval_proc.returncode}")
                raise FileNotFoundError(f"Missing evaluator output: {eval_json_path}")
            if eval_proc.returncode != 0:
                print(
                    f"[warn] Evaluator process returned code {eval_proc.returncode} "
                    "(threshold failure expected for low-iteration smoke runs)."
                )

            eval_json = load_json(eval_json_path)
            metrics = extract_eval_metrics(eval_json)
            summary["runs"].append(
                {
                    "name": run_name,
                    "output_dir": str(run_dir),
                    "eval_json": str(eval_json_path),
                    "metrics": metrics,
                }
            )

        run1 = summary["runs"][0]["metrics"]
        run2 = summary["runs"][1]["metrics"]
        drift = {
            "mean_surface_error_m": abs(float(run1["mean_surface_error_m"]) - float(run2["mean_surface_error_m"])),
            "p95_surface_error_m": abs(float(run1["p95_surface_error_m"]) - float(run2["p95_surface_error_m"])),
            "center_error_m": abs(float(run1["center_error_m"]) - float(run2["center_error_m"])),
        }

        thresholds = summary["reproducibility"]["thresholds"]
        reproducibility_pass = (
            drift["mean_surface_error_m"] <= float(thresholds["mean_surface_error_m_le"])
            and drift["p95_surface_error_m"] <= float(thresholds["p95_surface_error_m_le"])
            and drift["center_error_m"] <= float(thresholds["center_error_m_le"])
        )

        summary["reproducibility"] = {
            "pass": bool(reproducibility_pass),
            "thresholds": thresholds,
            "drift": drift,
        }

        summary["overall_pass"] = bool(
            summary["consistency_gate"].get("pass", False)
            and run1["overall_pass"]
            and run2["overall_pass"]
            and reproducibility_pass
        )

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_summary_markdown(summary_md, summary)

    print("Synthetic Dataset C gate run complete")
    print(f"  overall_pass: {summary['overall_pass']}")
    print(f"  wrote: {summary_json}")
    print(f"  wrote: {summary_md}")
    raise SystemExit(0 if summary["overall_pass"] else 2)


if __name__ == "__main__":
    main()
