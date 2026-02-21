import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("matplotlib")
pytest.importorskip("plyfile")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



@pytest.fixture(scope="module")
def sphere_eval():
    return _load_module(REPO_ROOT / "scripts" / "eval_synthetic_sphere.py", "eval_synthetic_sphere")


@pytest.fixture(scope="module")
def cube_eval():
    return _load_module(REPO_ROOT / "scripts" / "eval_synthetic_cube.py", "eval_synthetic_cube")


def _sample_sphere(center, radius, n=200, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(center, dtype=np.float64)[None, :] + radius * v


def _sample_cube_surface(center, half_extent, n=300, seed=0):
    rng = np.random.default_rng(seed)
    points = np.empty((n, 3), dtype=np.float64)
    faces = rng.integers(0, 6, size=n)
    uv = rng.uniform(-half_extent, half_extent, size=(n, 2))
    for i, face in enumerate(faces):
        if face == 0:
            points[i] = [half_extent, uv[i, 0], uv[i, 1]]
        elif face == 1:
            points[i] = [-half_extent, uv[i, 0], uv[i, 1]]
        elif face == 2:
            points[i] = [uv[i, 0], half_extent, uv[i, 1]]
        elif face == 3:
            points[i] = [uv[i, 0], -half_extent, uv[i, 1]]
        elif face == 4:
            points[i] = [uv[i, 0], uv[i, 1], half_extent]
        else:
            points[i] = [uv[i, 0], uv[i, 1], -half_extent]
    return points + np.asarray(center, dtype=np.float64)[None, :]


def test_fit_sphere_least_squares_recovers_center_and_radius(sphere_eval):
    true_center = np.array([0.3, -0.2, 0.1], dtype=np.float64)
    true_radius = 0.8
    points = _sample_sphere(true_center, true_radius, n=500, seed=1)

    center, radius = sphere_eval.fit_sphere_least_squares(points)

    assert np.linalg.norm(center - true_center) < 1e-3
    assert abs(radius - true_radius) < 1e-3


def test_fit_sphere_gt_trimmed_reduces_outlier_bias(sphere_eval):
    true_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    true_radius = 1.0
    inliers = _sample_sphere(true_center, true_radius, n=300, seed=2)
    outliers = np.array([[5.0, 5.0, 5.0], [-4.0, -3.0, 4.0]], dtype=np.float64)
    points = np.vstack([inliers, outliers])

    center, radius, info = sphere_eval.fit_sphere_gt_trimmed(
        points,
        gt_center=true_center,
        gt_radius=true_radius,
        mad_scale=3.0,
        iters=2,
    )

    assert np.linalg.norm(center - true_center) < 0.02
    assert abs(radius - true_radius) < 0.02
    assert info["kept_count"] < points.shape[0]
    assert 0.0 < info["kept_fraction"] < 1.0


def test_sphere_run_fit_mode_rejects_invalid_mode(sphere_eval):
    points = _sample_sphere([0.0, 0.0, 0.0], 1.0, n=50, seed=4)
    with pytest.raises(ValueError, match="Unsupported fit mode"):
        sphere_eval.run_fit_mode(
            "bad_mode",
            points=points,
            gt_center=np.zeros(3, dtype=np.float64),
            gt_radius=1.0,
            robust_mad_scale=3.0,
            robust_iters=1,
        )


def test_cube_signed_distance_signs_match_inside_surface_outside(cube_eval):
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    he = 1.0
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.1],
            [1.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    d = cube_eval.cube_signed_distance(pts, center, he)

    assert d[0] < 0.0
    assert abs(d[1]) < 1e-9
    assert d[2] > 0.0


def test_fit_cube_bbox_midpoint_recovers_axis_aligned_cube(cube_eval):
    center = np.array([0.2, -0.3, 0.4], dtype=np.float64)
    he = 0.8
    points = _sample_cube_surface(center, he, n=600, seed=10)

    fit_center, fit_he = cube_eval.fit_cube_bbox_midpoint(points)

    assert np.linalg.norm(fit_center - center) < 0.05
    assert abs(fit_he - he) < 0.05


def test_fit_cube_gt_trimmed_handles_far_outliers(cube_eval):
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    he = 1.0
    inliers = _sample_cube_surface(center, he, n=400, seed=11)
    outliers = np.array([[8.0, 8.0, 8.0], [-7.0, 3.0, 6.0]], dtype=np.float64)
    points = np.vstack([inliers, outliers])

    fit_center, fit_he, info = cube_eval.fit_cube_gt_trimmed(
        points,
        gt_center=center,
        gt_half_extent=he,
        mad_scale=3.0,
        iters=2,
    )

    assert np.linalg.norm(fit_center - center) < 0.1
    assert abs(fit_he - he) < 0.1
    assert info["kept_count"] < points.shape[0]
    assert info["kept_fraction"] < 1.0


def test_cube_run_fit_mode_rejects_invalid_mode(cube_eval):
    points = _sample_cube_surface([0.0, 0.0, 0.0], 1.0, n=80, seed=12)
    with pytest.raises(ValueError, match="Unsupported fit mode"):
        cube_eval.run_fit_mode(
            "invalid",
            points=points,
            gt_center=np.zeros(3, dtype=np.float64),
            gt_half_extent=1.0,
            robust_mad_scale=3.0,
            robust_iters=1,
        )


def test_summarize_residuals_reports_expected_keys_and_values(sphere_eval):
    residuals = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    stats = sphere_eval.summarize_residuals(residuals)

    assert set(stats.keys()) == {
        "count",
        "mean",
        "median",
        "std",
        "p90",
        "p95",
        "p99",
        "max",
        "min",
    }
    assert stats["count"] == 5
    assert math.isclose(stats["mean"], 0.4)
    assert stats["min"] == 0.0
    assert stats["max"] == 0.8
