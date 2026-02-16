#!/usr/bin/env python3
"""
Evaluate synthetic sphere reconstruction quality against known ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic sphere reconstruction")
    parser.add_argument("--reconstruction", type=Path, required=True, help="Path to reconstruction PLY")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to manifest.json")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root containing manifest.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for eval artifacts")

    parser.add_argument("--mean-threshold", type=float, default=0.05)
    parser.add_argument("--p95-threshold", type=float, default=0.10)
    parser.add_argument("--center-threshold", type=float, default=0.03)
    parser.add_argument("--hist-bins", type=int, default=80)
    parser.add_argument(
        "--fit-mode",
        choices=["least_squares", "gt_trimmed", "both"],
        default="gt_trimmed",
    )
    parser.add_argument("--robust-mad-scale", type=float, default=3.0)
    parser.add_argument("--robust-iters", type=int, default=2)
    parser.add_argument("--no-fit-sphere", dest="fit_sphere", action="store_false")
    parser.set_defaults(fit_sphere=True)
    return parser.parse_args()


def load_manifest(args: argparse.Namespace) -> Dict:
    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
    elif args.dataset_root is not None:
        manifest_path = args.dataset_root.expanduser().resolve() / "manifest.json"
    else:
        raise ValueError("Provide --manifest or --dataset-root")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_vertices_from_ply(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PLY not found: {path}")

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise RuntimeError(f"PLY has no vertex element: {path}")

    verts = ply["vertex"]
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(np.float64)
    if xyz.shape[0] == 0:
        raise RuntimeError(f"PLY has zero vertices: {path}")
    return xyz


def fit_sphere_least_squares(points: np.ndarray) -> Tuple[np.ndarray, float]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    A = np.column_stack([2.0 * x, 2.0 * y, 2.0 * z, np.ones_like(x)])
    b = x * x + y * y + z * z
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    center = params[:3]
    radius_sq = params[3] + np.dot(center, center)
    if radius_sq <= 0.0:
        raise RuntimeError(f"Fitted non-positive radius^2: {radius_sq}")
    return center, float(math.sqrt(radius_sq))


def fit_sphere_gt_trimmed(
    points: np.ndarray,
    gt_center: np.ndarray,
    gt_radius: float,
    mad_scale: float = 3.0,
    iters: int = 1,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    if points.shape[0] < 8:
        center, radius = fit_sphere_least_squares(points)
        return center, radius, {
            "kept_count": int(points.shape[0]),
            "total_count": int(points.shape[0]),
            "kept_fraction": 1.0,
        }

    mask = np.ones(points.shape[0], dtype=bool)
    for _ in range(max(1, int(iters))):
        residual = np.abs(np.linalg.norm(points - gt_center[None, :], axis=1) - gt_radius)
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        robust_sigma = mad * 1.4826
        thresh = med + float(mad_scale) * max(robust_sigma, 1e-6)
        new_mask = residual <= thresh

        if new_mask.sum() < 8:
            break
        mask = new_mask
        break

    final_points = points[mask]
    center, radius = fit_sphere_least_squares(final_points)
    info = {
        "kept_count": int(final_points.shape[0]),
        "total_count": int(points.shape[0]),
        "kept_fraction": float(final_points.shape[0] / points.shape[0]),
    }
    return center, radius, info


def summarize_residuals(residuals: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(residuals.shape[0]),
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "std": float(np.std(residuals)),
        "p90": float(np.percentile(residuals, 90.0)),
        "p95": float(np.percentile(residuals, 95.0)),
        "p99": float(np.percentile(residuals, 99.0)),
        "max": float(np.max(residuals)),
        "min": float(np.min(residuals)),
    }


def save_histogram(path: Path, residuals: np.ndarray, bins: int) -> None:
    fig = plt.figure(figsize=(7, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(residuals, bins=bins, color="#2f7fbd", alpha=0.85)
    ax.set_title("Sphere Residual Histogram")
    ax.set_xlabel("Absolute radial residual (m)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_fit_mode(
    mode: str,
    points: np.ndarray,
    gt_center: np.ndarray,
    gt_radius: float,
    robust_mad_scale: float,
    robust_iters: int,
) -> Dict:
    if mode == "gt_trimmed":
        fit_center, fit_radius, robust_info = fit_sphere_gt_trimmed(
            points,
            gt_center=gt_center,
            gt_radius=gt_radius,
            mad_scale=robust_mad_scale,
            iters=robust_iters,
        )
    elif mode == "least_squares":
        fit_center, fit_radius = fit_sphere_least_squares(points)
        robust_info = None
    else:
        raise ValueError(f"Unsupported fit mode: {mode}")

    fit_radial = np.abs(np.linalg.norm(points - fit_center[None, :], axis=1) - fit_radius)
    return {
        "mode": mode,
        "center_m": fit_center.tolist(),
        "radius_m": float(fit_radius),
        "center_error_m": float(np.linalg.norm(fit_center - gt_center)),
        "radius_error_m": float(abs(fit_radius - gt_radius)),
        "radial_stats": summarize_residuals(fit_radial),
        "robust_info": robust_info,
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args)

    gt_center = np.asarray(manifest["geometry"]["sphere_center_m"], dtype=np.float64)
    gt_radius = float(manifest["geometry"]["sphere_radius_m"])

    points = load_vertices_from_ply(args.reconstruction)
    gt_radial = np.abs(np.linalg.norm(points - gt_center[None, :], axis=1) - gt_radius)
    gt_stats = summarize_residuals(gt_radial)

    fit_result = {
        "enabled": bool(args.fit_sphere),
        "mode": args.fit_mode,
        "primary_mode": None,
        "center_m": None,
        "radius_m": None,
        "center_error_m": None,
        "radius_error_m": None,
        "radial_stats": None,
        "robust_info": None,
        "by_mode": {},
    }

    if args.fit_sphere:
        fit_modes = ["least_squares", "gt_trimmed"] if args.fit_mode == "both" else [args.fit_mode]
        by_mode = {
            mode: run_fit_mode(
                mode,
                points=points,
                gt_center=gt_center,
                gt_radius=gt_radius,
                robust_mad_scale=float(args.robust_mad_scale),
                robust_iters=int(args.robust_iters),
            )
            for mode in fit_modes
        }

        primary_mode = "gt_trimmed" if args.fit_mode == "both" else args.fit_mode
        primary = by_mode[primary_mode]
        fit_result = {
            "enabled": True,
            "mode": args.fit_mode,
            "primary_mode": primary_mode,
            "center_m": primary["center_m"],
            "radius_m": primary["radius_m"],
            "center_error_m": primary["center_error_m"],
            "radius_error_m": primary["radius_error_m"],
            "radial_stats": primary["radial_stats"],
            "robust_info": primary["robust_info"],
            "by_mode": by_mode,
        }

    mean_ok = gt_stats["mean"] <= float(args.mean_threshold)
    p95_ok = gt_stats["p95"] <= float(args.p95_threshold)
    center_ok = (
        fit_result["center_error_m"] is not None
        and fit_result["center_error_m"] <= float(args.center_threshold)
    )
    overall_pass = bool(mean_ok and p95_ok and center_ok)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.reconstruction.expanduser().resolve().parent
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hist_path = output_dir / "sphere_residual_hist.png"
    save_histogram(hist_path, gt_radial, bins=int(args.hist_bins))

    result = {
        "reconstruction": str(args.reconstruction.expanduser().resolve()),
        "manifest_dataset_id": manifest.get("dataset_id", "unknown"),
        "manifest_variant": manifest.get("variant", "unknown"),
        "thresholds": {
            "mean_radial_error_m_le": float(args.mean_threshold),
            "p95_radial_error_m_le": float(args.p95_threshold),
            "center_error_m_le": float(args.center_threshold),
        },
        "ground_truth": {
            "sphere_center_m": gt_center.tolist(),
            "sphere_radius_m": gt_radius,
            "radial_stats": gt_stats,
        },
        "fit_sphere": fit_result,
        "checks": {
            "mean_pass": bool(mean_ok),
            "p95_pass": bool(p95_ok),
            "center_pass": bool(center_ok),
        },
        "overall_pass": overall_pass,
        "artifacts": {
            "residual_histogram": str(hist_path),
        },
    }

    out_json = output_dir / "sphere_eval.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Synthetic sphere evaluation complete")
    print(f"  points: {points.shape[0]}")
    print(f"  GT mean radial error: {gt_stats['mean']:.6f} m")
    print(f"  GT p95 radial error:  {gt_stats['p95']:.6f} m")
    if fit_result["center_error_m"] is not None:
        if fit_result["mode"] == "both":
            for mode_name in ("least_squares", "gt_trimmed"):
                mode_result = fit_result["by_mode"][mode_name]
                print(
                    f"  [{mode_name}] center error: {mode_result['center_error_m']:.6f} m, "
                    f"radius error: {mode_result['radius_error_m']:.6f} m"
                )
        else:
            print(f"  Fitted center error: {fit_result['center_error_m']:.6f} m")
            print(f"  Fitted radius error: {fit_result['radius_error_m']:.6f} m")
    print(f"  pass: {overall_pass}")
    print(f"  wrote: {out_json}")
    print(f"  wrote: {hist_path}")

    raise SystemExit(0 if overall_pass else 2)


if __name__ == "__main__":
    main()
