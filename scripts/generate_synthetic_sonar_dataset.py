#!/usr/bin/env python3
"""
Generate synthetic sonar datasets compatible with the existing COLMAP+sonar layout.

Supported synthetic tracks:
- Dataset A: sphere in vacuum
- Dataset C: cube in vacuum
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import rotmat2qvec
from scene.dataset_readers import readColmapSceneInfo
from utils.sonar_utils import SonarConfig, get_camera_to_sonar_transform, sonar_frame_to_points


SCRIPT_VERSION = "2026-03-22"

SPHERE_VARIANTS = {"A_clean", "A_noisy"}
CUBE_VARIANTS = {"C_clean", "C_noisy"}
ALL_VARIANTS = tuple(sorted(SPHERE_VARIANTS | CUBE_VARIANTS))


@dataclass
class PoseRecord:
    image_id: int
    image_name: str
    camera_center_world: np.ndarray
    sonar_center_world: np.ndarray
    R_w2c: np.ndarray
    t_w2c: np.ndarray
    R_w2s: np.ndarray
    t_w2s: np.ndarray
    qvec: np.ndarray
    pose_mode: str

    @property
    def R_c2w(self) -> np.ndarray:
        return self.R_w2s.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic sonar datasets (sphere/cube)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./synthetic_datasets/synthetic_sphere_A_clean"),
        help="Dataset output directory",
    )
    parser.add_argument("--variant", choices=ALL_VARIANTS, default="A_clean")
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir if it exists")
    parser.add_argument(
        "--pose-mode",
        choices=["sonar_equivalent", "camera_with_extrinsic"],
        default="sonar_equivalent",
        help=(
            "Pose export mode. 'sonar_equivalent' preserves previous behavior. "
            "'camera_with_extrinsic' exports camera poses so camera->sonar extrinsic yields the intended sonar orbit."
        ),
    )

    parser.add_argument("--sphere-radius", type=float, default=0.8)
    parser.add_argument("--sphere-center", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--cube-half-extent", type=float, default=0.8)
    parser.add_argument("--cube-center", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--orbit-radius", type=float, default=2.0)
    parser.add_argument("--elevation-min-deg", type=float, default=-12.0)
    parser.add_argument("--elevation-max-deg", type=float, default=12.0)
    parser.add_argument(
        "--pose-policy",
        choices=["auto", "orbit_sweep", "multi_band"],
        default="auto",
        help=(
            "Pose coverage policy. 'auto' selects orbit_sweep for Dataset A and "
            "multi_band for Dataset C."
        ),
    )
    parser.add_argument(
        "--pose-bands-deg",
        type=str,
        default="-12,-6,0,6,12",
        help="Comma-separated elevation bands in degrees for multi_band policy",
    )
    parser.add_argument(
        "--pose-band-jitter-frac",
        type=float,
        default=0.2,
        help=(
            "Uniform elevation jitter as a fraction of the minimum pose-band spacing for "
            "multi_band policy; set to 0 to keep band positions deterministic"
        ),
    )
    parser.add_argument("--translation-jitter-sigma", type=float, default=0.02)
    parser.add_argument("--rotation-jitter-sigma-deg", type=float, default=1.5)
    parser.add_argument(
        "--orientation-azimuth-jitter-max-deg",
        type=float,
        default=0.0,
        help=(
            "Uniform azimuth-only orientation jitter around the world up axis, in degrees, "
            "applied to the center-facing pose"
        ),
    )

    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=200)
    parser.add_argument("--azimuth-fov-deg", type=float, default=120.0)
    parser.add_argument("--elevation-fov-deg", type=float, default=20.0)
    parser.add_argument("--range-min", type=float, default=0.2)
    parser.add_argument("--range-max", type=float, default=3.0)
    parser.add_argument("--elevation-samples", type=int, default=64)

    parser.add_argument("--noise-speckle-sigma", type=float, default=0.12)
    parser.add_argument("--noise-dropout-prob", type=float, default=0.01)

    parser.add_argument("--points3d-count", type=int, default=4096)
    parser.add_argument("--run-loader-check", action="store_true", default=True)
    parser.add_argument("--no-loader-check", dest="run_loader_check", action="store_false")
    parser.add_argument("--run-consistency-gate", action="store_true", default=True)
    parser.add_argument("--no-consistency-gate", dest="run_consistency_gate", action="store_false")
    parser.add_argument("--gate-mask-top-rows", type=int, default=10)
    parser.add_argument("--gate-intensity-threshold", type=float, default=1.0 / 255.0)

    return parser.parse_args()


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def variant_is_sphere(variant: str) -> bool:
    return variant in SPHERE_VARIANTS


def variant_is_cube(variant: str) -> bool:
    return variant in CUBE_VARIANTS


def variant_shape(variant: str) -> str:
    if variant_is_sphere(variant):
        return "sphere"
    if variant_is_cube(variant):
        return "cube"
    raise ValueError(f"Unsupported variant: {variant}")


def variant_is_noisy(variant: str) -> bool:
    return variant.endswith("_noisy")


def resolve_pose_policy(args: argparse.Namespace) -> str:
    if args.pose_policy != "auto":
        return args.pose_policy
    return "multi_band" if variant_is_cube(args.variant) else "orbit_sweep"


def parse_pose_bands_deg(raw: str) -> np.ndarray:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("--pose-bands-deg must include at least one value")
    bands = np.array([float(v) for v in values], dtype=np.float64)
    return bands


def geometry_vec3(geometry: Dict[str, object], key: str) -> np.ndarray:
    value = geometry.get(key)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"Geometry key '{key}' must be a 3-vector")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"Geometry key '{key}' must have shape (3,), got {arr.shape}")
    return arr


def geometry_scalar(geometry: Dict[str, object], key: str) -> float:
    value = geometry.get(key)
    if not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"Geometry key '{key}' must be numeric")
    return float(value)


def rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def build_row_major_pose_matrix(R_w2v: np.ndarray, t_w2v: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_w2v
    T[3, :3] = t_w2v
    return T


def extract_rt_from_row_major_pose(T_w2v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return T_w2v[:3, :3].copy(), T_w2v[3, :3].copy()


def camera_center_from_w2v(R_w2v: np.ndarray, t_w2v: np.ndarray) -> np.ndarray:
    return -R_w2v.T @ t_w2v


def look_at_rotation_w2c(center_world: np.ndarray, target_world: np.ndarray) -> np.ndarray:
    """
    Build world->camera rotation for camera convention +X right, +Y down, +Z forward.
    """
    forward = normalize(target_world - center_world)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))
    down = -up

    R_w2c = np.stack([right, down, forward], axis=0)
    return R_w2c


def generate_orbit_angles(
    *,
    num_frames: int,
    elev_min_deg: float,
    elev_max_deg: float,
    policy: str,
    pose_bands_deg: np.ndarray,
    pose_band_jitter_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    azimuths = np.linspace(0.0, 2.0 * math.pi, num_frames, endpoint=False, dtype=np.float64)

    if policy == "orbit_sweep":
        elevations = np.linspace(
            math.radians(elev_min_deg),
            math.radians(elev_max_deg),
            num_frames,
            dtype=np.float64,
        )
        return azimuths, elevations

    if policy != "multi_band":
        raise ValueError(f"Unsupported pose policy: {policy}")

    bands = np.deg2rad(np.asarray(pose_bands_deg, dtype=np.float64))
    bands = bands[(bands >= math.radians(elev_min_deg)) & (bands <= math.radians(elev_max_deg))]
    if bands.size == 0:
        bands = np.linspace(math.radians(elev_min_deg), math.radians(elev_max_deg), 5, dtype=np.float64)

    band_idx = np.arange(num_frames, dtype=np.int64) % bands.shape[0]
    elevations = bands[band_idx]

    if bands.shape[0] > 1:
        sorted_bands = np.sort(np.unique(bands))
        if sorted_bands.shape[0] > 1:
            min_step = float(np.min(np.diff(sorted_bands)))
            if min_step > 1e-9 and pose_band_jitter_frac > 0.0:
                jitter = rng.uniform(
                    low=-float(pose_band_jitter_frac) * min_step,
                    high=float(pose_band_jitter_frac) * min_step,
                    size=num_frames,
                )
                elevations = elevations + jitter

    elevations = np.clip(elevations, math.radians(elev_min_deg), math.radians(elev_max_deg))
    return azimuths, elevations


def build_pose_records(args: argparse.Namespace, rng: np.random.Generator) -> List[PoseRecord]:
    shape = variant_shape(args.variant)
    if shape == "sphere":
        center = np.asarray(args.sphere_center, dtype=np.float64)
    else:
        center = np.asarray(args.cube_center, dtype=np.float64)

    orbit_radius = float(args.orbit_radius)
    num_frames = int(args.num_frames)
    pose_mode = str(args.pose_mode)
    if pose_mode not in ("sonar_equivalent", "camera_with_extrinsic"):
        raise ValueError(f"Unsupported --pose-mode: {pose_mode}")

    if pose_mode == "camera_with_extrinsic":
        T_c2s = get_camera_to_sonar_transform(device="cpu").detach().cpu().numpy().astype(np.float64)
        T_s2c = np.linalg.inv(T_c2s)
    else:
        T_s2c = np.eye(4, dtype=np.float64)

    pose_policy = resolve_pose_policy(args)
    pose_bands_deg = parse_pose_bands_deg(args.pose_bands_deg)
    azimuths, elevations = generate_orbit_angles(
        num_frames=num_frames,
        elev_min_deg=float(args.elevation_min_deg),
        elev_max_deg=float(args.elevation_max_deg),
        policy=pose_policy,
        pose_bands_deg=pose_bands_deg,
        pose_band_jitter_frac=float(args.pose_band_jitter_frac),
        rng=rng,
    )

    poses: List[PoseRecord] = []
    rot_sigma_rad = math.radians(float(args.rotation_jitter_sigma_deg))
    orientation_azimuth_jitter_max_rad = math.radians(float(args.orientation_azimuth_jitter_max_deg))
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    for i in range(num_frames):
        az = azimuths[i]
        el = elevations[i]

        base_sonar_center = center + orbit_radius * np.array(
            [
                math.cos(el) * math.cos(az),
                math.sin(el),
                math.cos(el) * math.sin(az),
            ],
            dtype=np.float64,
        )

        trans_jitter = rng.normal(loc=0.0, scale=float(args.translation_jitter_sigma), size=3)
        sonar_center = base_sonar_center + trans_jitter

        target_world = center
        if orientation_azimuth_jitter_max_rad > 0.0:
            delta_az = rng.uniform(
                low=-orientation_azimuth_jitter_max_rad,
                high=orientation_azimuth_jitter_max_rad,
            )
            base_forward = normalize(center - sonar_center)
            R_az = rotation_matrix_from_rotvec(world_up * delta_az)
            target_world = sonar_center + (R_az @ base_forward)

        R_w2s = look_at_rotation_w2c(sonar_center, target_world)

        rotvec_jitter = rng.normal(loc=0.0, scale=rot_sigma_rad, size=3)
        R_delta = rotation_matrix_from_rotvec(rotvec_jitter)
        R_w2s = R_delta @ R_w2s

        t_w2s = (R_w2s @ sonar_center) * -1.0
        T_w2s = build_row_major_pose_matrix(R_w2s, t_w2s)

        T_w2c = T_w2s @ T_s2c
        R_w2c, t_w2c = extract_rt_from_row_major_pose(T_w2c)
        camera_center = camera_center_from_w2v(R_w2c, t_w2c)
        qvec = rotmat2qvec(R_w2c)

        poses.append(
            PoseRecord(
                image_id=i + 1,
                image_name=f"sonar_{i:06d}.png",
                camera_center_world=camera_center,
                sonar_center_world=sonar_center,
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                R_w2s=R_w2s,
                t_w2s=t_w2s,
                qvec=qvec,
                pose_mode=pose_mode,
            )
        )

    return poses


def precompute_sonar_directions(sonar_cfg: SonarConfig, elev_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    az_cols = sonar_cfg.azimuth_grid.detach().cpu().numpy().astype(np.float64)
    elev_vals = np.linspace(
        -float(sonar_cfg.half_elevation_rad),
        float(sonar_cfg.half_elevation_rad),
        elev_samples,
        dtype=np.float64,
    )

    az_mesh, el_mesh = np.meshgrid(az_cols, elev_vals, indexing="ij")
    dirs_s = np.stack(
        [
            -np.sin(az_mesh) * np.cos(el_mesh),
            np.sin(el_mesh),
            np.cos(az_mesh) * np.cos(el_mesh),
        ],
        axis=-1,
    )

    dirs_s = dirs_s.reshape(-1, 3)
    dirs_s = dirs_s / np.linalg.norm(dirs_s, axis=1, keepdims=True).clip(min=1e-12)
    col_idx = np.repeat(np.arange(sonar_cfg.image_width, dtype=np.int64), elev_samples)
    return dirs_s, col_idx


def intersect_sphere_nearest_positive(
    origin_world: np.ndarray,
    dirs_world: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    oc = origin_world[None, :] - sphere_center[None, :]
    b = 2.0 * np.sum(dirs_world * oc, axis=1)
    c = float(np.dot(oc[0], oc[0]) - sphere_radius * sphere_radius)
    disc = b * b - 4.0 * c

    t = np.full(dirs_world.shape[0], np.nan, dtype=np.float64)
    valid = disc >= 0.0
    if not np.any(valid):
        return t

    sqrt_disc = np.sqrt(disc[valid])
    b_valid = b[valid]
    t0 = (-b_valid - sqrt_disc) / 2.0
    t1 = (-b_valid + sqrt_disc) / 2.0

    best = np.where(t0 > 1e-8, t0, np.where(t1 > 1e-8, t1, np.nan))
    t[valid] = best
    return t


def intersect_axis_aligned_cube_nearest_positive(
    origin_world: np.ndarray,
    dirs_world: np.ndarray,
    cube_center: np.ndarray,
    cube_half_extent: float,
) -> np.ndarray:
    box_min = cube_center - float(cube_half_extent)
    box_max = cube_center + float(cube_half_extent)

    t = np.full(dirs_world.shape[0], np.nan, dtype=np.float64)

    parallel = np.abs(dirs_world) < 1e-12
    outside_parallel = parallel & ((origin_world < box_min) | (origin_world > box_max))
    impossible = np.any(outside_parallel, axis=1)

    safe_dirs = np.where(parallel, 1.0, dirs_world)
    inv_dir = 1.0 / safe_dirs

    t1 = (box_min[None, :] - origin_world[None, :]) * inv_dir
    t2 = (box_max[None, :] - origin_world[None, :]) * inv_dir

    t_near = np.max(np.minimum(t1, t2), axis=1)
    t_far = np.min(np.maximum(t1, t2), axis=1)

    valid = (~impossible) & (t_far >= np.maximum(t_near, 1e-8))
    if not np.any(valid):
        return t

    t_near_v = t_near[valid]
    t_far_v = t_far[valid]
    best = np.where(t_near_v > 1e-8, t_near_v, np.where(t_far_v > 1e-8, t_far_v, np.nan))
    t[valid] = best
    return t


def render_sonar_frame(
    pose: PoseRecord,
    dirs_s_flat: np.ndarray,
    col_idx_flat: np.ndarray,
    sonar_cfg: SonarConfig,
    shape: str,
    geometry: Dict[str, object],
    elev_samples: int,
) -> np.ndarray:
    """
    Elevation-integrated forward model:
      - For each azimuth column, cast rays across elevation samples.
      - Resolve occlusion per (azimuth, elevation) ray via first hit.
      - Collapse the elevation fan to the front envelope in range.
      - Deposit the elevation hit fraction at the nearest range bin(s).
    """
    R_s2w = pose.R_w2s.T
    dirs_w_flat = dirs_s_flat @ R_s2w.T
    dirs_w_flat = dirs_w_flat / np.linalg.norm(dirs_w_flat, axis=1, keepdims=True).clip(min=1e-12)

    if shape == "sphere":
        hit_ranges = intersect_sphere_nearest_positive(
            origin_world=pose.sonar_center_world,
            dirs_world=dirs_w_flat,
            sphere_center=geometry_vec3(geometry, "sphere_center_m"),
            sphere_radius=geometry_scalar(geometry, "sphere_radius_m"),
        )
    elif shape == "cube":
        hit_ranges = intersect_axis_aligned_cube_nearest_positive(
            origin_world=pose.sonar_center_world,
            dirs_world=dirs_w_flat,
            cube_center=geometry_vec3(geometry, "cube_center_m"),
            cube_half_extent=geometry_scalar(geometry, "cube_half_extent_m"),
        )
    else:
        raise ValueError(f"Unsupported shape for rendering: {shape}")

    H = int(sonar_cfg.image_height)
    W = int(sonar_cfg.image_width)
    span = float(sonar_cfg.range_max - sonar_cfg.range_min)

    counts = np.zeros((H, W), dtype=np.float64)
    hit_ranges_grid = hit_ranges.reshape(W, elev_samples)
    valid_grid = np.isfinite(hit_ranges_grid)
    if np.any(valid_grid):
        valid_counts = valid_grid.sum(axis=1).astype(np.float64)
        front_ranges = np.min(np.where(valid_grid, hit_ranges_grid, np.inf), axis=1)
        front_valid = np.isfinite(front_ranges)
        if np.any(front_valid):
            front_row_f = (front_ranges[front_valid] - float(sonar_cfg.range_min)) / span * H
            in_bounds = (front_row_f >= 0.0) & (front_row_f <= float(H - 1))
            if np.any(in_bounds):
                cols = np.nonzero(front_valid)[0][in_bounds]
                rows = front_row_f[in_bounds]
                weights = valid_counts[cols] / float(elev_samples)

                row0 = np.floor(rows).astype(np.int64)
                frac = rows - row0.astype(np.float64)
                row1 = np.clip(row0 + 1, 0, H - 1)

                np.add.at(counts, (row0, cols), weights * (1.0 - frac))
                upper_mask = row1 != row0
                if np.any(upper_mask):
                    np.add.at(counts, (row1[upper_mask], cols[upper_mask]), weights[upper_mask] * frac[upper_mask])

    img_float = counts
    img_float = np.clip(img_float, 0.0, 1.0)
    return img_float.astype(np.float32)


def apply_noise_if_needed(
    img_float: np.ndarray,
    variant: str,
    rng: np.random.Generator,
    speckle_sigma: float,
    dropout_prob: float,
) -> np.ndarray:
    if not variant_is_noisy(variant):
        return img_float

    noisy = img_float.copy()
    if speckle_sigma > 0.0:
        speckle = rng.lognormal(mean=0.0, sigma=speckle_sigma, size=noisy.shape)
        noisy *= speckle.astype(np.float32)
    if dropout_prob > 0.0:
        dropout_mask = rng.random(size=noisy.shape) < dropout_prob
        noisy[dropout_mask] = 0.0
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.astype(np.float32)


def write_cameras_txt(path: Path, args: argparse.Namespace) -> Dict[str, float | str]:
    fx = args.image_width / (2.0 * math.tan(math.radians(args.azimuth_fov_deg / 2.0)))
    fy = args.image_height / (2.0 * math.tan(math.radians(args.elevation_fov_deg / 2.0)))
    cx = args.image_width / 2.0
    cy = args.image_height / 2.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            f"1 PINHOLE {args.image_width} {args.image_height} "
            f"{fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n"
        )

    return {
        "model": "PINHOLE",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def write_images_txt(path: Path, poses: List[PoseRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}\n")
        for pose in poses:
            qw, qx, qy, qz = pose.qvec.tolist()
            tx, ty, tz = pose.t_w2c.tolist()
            f.write(
                f"{pose.image_id} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                f"{tx:.12f} {ty:.12f} {tz:.12f} 1 {pose.image_name}\n"
            )
            f.write("\n")


def sample_sphere_points(center: np.ndarray, radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(count)
    v = rng.random(count)
    theta = 2.0 * math.pi * u
    phi = np.arccos(1.0 - 2.0 * v)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=1) * radius + center[None, :]
    return points.astype(np.float64)


def sample_cube_surface_points(
    center: np.ndarray,
    half_extent: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    half = float(half_extent)
    face_idx = rng.integers(low=0, high=6, size=count)
    uv = rng.uniform(low=-half, high=half, size=(count, 2))
    pts = np.zeros((count, 3), dtype=np.float64)

    # +X
    mask = face_idx == 0
    pts[mask, 0] = half
    pts[mask, 1:] = uv[mask]
    # -X
    mask = face_idx == 1
    pts[mask, 0] = -half
    pts[mask, 1:] = uv[mask]
    # +Y
    mask = face_idx == 2
    pts[mask, 1] = half
    pts[mask, 0] = uv[mask, 0]
    pts[mask, 2] = uv[mask, 1]
    # -Y
    mask = face_idx == 3
    pts[mask, 1] = -half
    pts[mask, 0] = uv[mask, 0]
    pts[mask, 2] = uv[mask, 1]
    # +Z
    mask = face_idx == 4
    pts[mask, 2] = half
    pts[mask, :2] = uv[mask]
    # -Z
    mask = face_idx == 5
    pts[mask, 2] = -half
    pts[mask, :2] = uv[mask]

    pts += center[None, :]
    return pts


def write_points3d_txt(path: Path, points: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        f.write(f"# Number of points: {points.shape[0]}\n")
        for i, p in enumerate(points, start=1):
            f.write(f"{i} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} 255 255 255 0.0\n")


def write_manifest(path: Path, manifest: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def write_dataset_settings_md(path: Path, manifest: Dict) -> None:
    geometry = manifest["geometry"]
    sonar = manifest["sonar_model"]
    pose = manifest["pose_generation"]
    sim = manifest["simulation"]
    seeds = manifest["seeds"]
    gate = manifest.get("consistency_gate", {})

    lines = [
        "# Synthetic Dataset Settings",
        "",
        f"- Dataset id: `{manifest['dataset_id']}`",
        f"- Variant: `{manifest['variant']}`",
        f"- Generated UTC: `{manifest['generated_utc']}`",
        f"- Generator: `scripts/generate_synthetic_sonar_dataset.py` ({manifest['generator_version']})",
        "",
        "## Geometry",
        "",
        f"- Shape: `{geometry['shape']}`",
    ]

    if geometry["shape"] == "sphere":
        lines.extend(
            [
                f"- Sphere center (m): `{geometry['sphere_center_m']}`",
                f"- Sphere radius (m): `{geometry['sphere_radius_m']}`",
            ]
        )
    elif geometry["shape"] == "cube":
        lines.extend(
            [
                f"- Cube center (m): `{geometry['cube_center_m']}`",
                f"- Cube half extent (m): `{geometry['cube_half_extent_m']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Sonar Model",
            "",
            f"- Image size: `{sonar['image_width']}x{sonar['image_height']}`",
            f"- Azimuth FOV (deg): `{sonar['azimuth_fov_deg']}`",
            f"- Elevation FOV (deg): `{sonar['elevation_fov_deg']}`",
            f"- Range limits (m): `[{sonar['range_min_m']}, {sonar['range_max_m']}]`",
            "",
            "## Pose Policy",
            "",
            f"- Frames: `{pose['num_frames']}`",
            f"- Orbit radius (m): `{pose['orbit_radius_m']}`",
            f"- Elevation sweep (deg): `{pose['elevation_sweep_deg']}`",
            f"- Pose policy: `{pose['pose_policy']}`",
            f"- Pose bands (deg): `{pose.get('pose_bands_deg', [])}`",
            f"- Pose band jitter frac: `{pose.get('pose_band_jitter_frac', 0.0)}`",
            f"- Translation jitter sigma (m): `{pose['translation_jitter_sigma_m']}`",
            f"- Rotation jitter sigma (deg): `{pose['rotation_jitter_sigma_deg']}`",
            f"- Orientation azimuth jitter max (deg): `{pose.get('orientation_azimuth_jitter_max_deg', 0.0)}`",
            f"- Orientation policy: `{pose['orientation_policy']}`",
            f"- Pose mode: `{pose['pose_mode']}`",
            f"- Pose export contract: `{pose['pose_export_contract']}`",
            "",
            "## Simulation",
            "",
            f"- Elevation samples per azimuth: `{sim['elevation_samples']}`",
            f"- Intersection policy: `{sim['intersection_policy']}`",
            f"- Intensity model: `{sim['intensity_model']}`",
            f"- Normalization: `{sim['normalization']}`",
            "",
            "## Seed Contract",
            "",
            f"- master: `{seeds['master']}`",
            f"- pose: `{seeds['pose']}`",
            f"- noise: `{seeds['noise']}`",
            f"- points3d: `{seeds['points3d']}`",
            "",
            "## Extrinsic Policy",
            "",
            f"- `{manifest['extrinsic_policy']}`",
        ]
    )

    if gate:
        lines.extend(
            [
                "",
                "## Backward Projection Gate",
                "",
                f"- Enabled: `{gate.get('enabled', False)}`",
                f"- Criteria: `{gate.get('criteria', {})}`",
                f"- Result file: `consistency_gate.json`",
            ]
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_points_ply(path: Path, points: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def project_points_to_source_pixels(
    points_world: np.ndarray,
    camera_stub: SimpleNamespace,
    sonar_cfg: SonarConfig,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    points_world_metric = points_world * scale_factor
    camera_center_colmap = -camera_stub.R.T @ camera_stub.T
    camera_center_metric = camera_center_colmap * scale_factor

    points_cam_metric = (camera_stub.R @ (points_world_metric - camera_center_metric).T).T
    x_cam = points_cam_metric[:, 0]
    y_cam = points_cam_metric[:, 1]
    z_cam = points_cam_metric[:, 2]

    azimuth = -np.arctan2(x_cam, z_cam)
    ranges = np.sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam)

    col = (-azimuth / float(sonar_cfg.half_azimuth_rad) + 1.0) * (sonar_cfg.image_width / 2.0)
    row = (ranges - float(sonar_cfg.range_min)) / float(sonar_cfg.range_max - sonar_cfg.range_min) * sonar_cfg.image_height
    return row, col


def fit_sphere_least_squares(points: np.ndarray) -> Tuple[np.ndarray, float] | None:
    if points.shape[0] < 8:
        return None

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    A = np.column_stack([2.0 * x, 2.0 * y, 2.0 * z, np.ones_like(x)])
    b = x * x + y * y + z * z

    try:
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    center = params[:3]
    radius_sq = params[3] + np.dot(center, center)
    if radius_sq <= 0.0:
        return None
    return center, float(math.sqrt(radius_sq))


def cube_signed_distance(points: np.ndarray, cube_center: np.ndarray, cube_half_extent: float) -> np.ndarray:
    q = np.abs(points - cube_center[None, :]) - float(cube_half_extent)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def fit_cube_center_median(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 8:
        return None
    return np.median(points, axis=0)


def run_backward_projection_gate(
    dataset_root: Path,
    poses: List[PoseRecord],
    images_uint8: List[np.ndarray],
    sonar_cfg: SonarConfig,
    shape: str,
    geometry: Dict[str, object],
    threshold: float,
    mask_top_rows: int,
) -> Dict:
    all_pixel_errors = []
    all_surface_residuals = []
    all_points = []
    has_nan_inf = False

    for pose, img_u8 in zip(poses, images_uint8):
        img_float = (img_u8.astype(np.float32) / 255.0)
        camera_stub = SimpleNamespace(
            original_image=torch.from_numpy(img_float[None, :, :].copy()),
            R=pose.R_c2w.astype(np.float64),
            T=pose.t_w2s.astype(np.float64),
        )

        backproj = sonar_frame_to_points(
            camera_stub,
            sonar_cfg,
            intensity_threshold=float(threshold),
            mask_top_rows=int(mask_top_rows),
            scale_factor=1.0,
            elevation_mode="zero",
            rng=None,
            return_debug=False,
        )
        points_world, _ = backproj[0], backproj[1]

        if points_world.size == 0:
            continue

        valid_mask = img_float > float(threshold)
        if mask_top_rows > 0:
            valid_mask[:mask_top_rows, :] = False
        src_rows, src_cols = np.where(valid_mask)

        if points_world.shape[0] != src_rows.shape[0]:
            min_count = min(points_world.shape[0], src_rows.shape[0])
            points_world = points_world[:min_count]
            src_rows = src_rows[:min_count]
            src_cols = src_cols[:min_count]

        proj_rows, proj_cols = project_points_to_source_pixels(points_world, camera_stub, sonar_cfg, scale_factor=1.0)
        pixel_err = np.sqrt((proj_rows - src_rows) ** 2 + (proj_cols - src_cols) ** 2)

        if shape == "sphere":
            sphere_center = geometry_vec3(geometry, "sphere_center_m")
            sphere_radius = geometry_scalar(geometry, "sphere_radius_m")
            residual = np.abs(np.linalg.norm(points_world - sphere_center[None, :], axis=1) - sphere_radius)
        elif shape == "cube":
            cube_center = geometry_vec3(geometry, "cube_center_m")
            cube_half_extent = geometry_scalar(geometry, "cube_half_extent_m")
            residual = np.abs(cube_signed_distance(points_world, cube_center, cube_half_extent))
        else:
            raise ValueError(f"Unsupported shape in consistency gate: {shape}")

        all_pixel_errors.append(pixel_err)
        all_surface_residuals.append(residual)
        all_points.append(points_world)

        if (not np.isfinite(points_world).all()) or (not np.isfinite(pixel_err).all()) or (not np.isfinite(residual).all()):
            has_nan_inf = True

    if all_pixel_errors:
        pixel_errors = np.concatenate(all_pixel_errors, axis=0)
        surface_residuals = np.concatenate(all_surface_residuals, axis=0)
        recovered_points = np.concatenate(all_points, axis=0)
    else:
        pixel_errors = np.zeros((0,), dtype=np.float64)
        surface_residuals = np.zeros((0,), dtype=np.float64)
        recovered_points = np.zeros((0, 3), dtype=np.float64)

    if shape == "sphere":
        gt_center = geometry_vec3(geometry, "sphere_center_m")
        gt_radius = geometry_scalar(geometry, "sphere_radius_m")
        fitted = fit_sphere_least_squares(recovered_points)
        if fitted is None:
            fitted_center = None
            fitted_size = None
            fitted_center_error = float("inf")
            fitted_size_error = float("inf")
        else:
            fitted_center, fitted_radius = fitted
            fitted_size = float(fitted_radius)
            fitted_center_error = float(np.linalg.norm(fitted_center - gt_center))
            fitted_size_error = float(abs(fitted_radius - gt_radius))
        residual_key = "radial"
    else:
        gt_center = geometry_vec3(geometry, "cube_center_m")
        gt_half_extent = geometry_scalar(geometry, "cube_half_extent_m")
        fitted_center = fit_cube_center_median(recovered_points)
        if fitted_center is None:
            fitted_size = None
            fitted_center_error = float("inf")
            fitted_size_error = float("inf")
        else:
            centered = np.abs(recovered_points - fitted_center[None, :])
            estimated_extent = float(np.percentile(np.max(centered, axis=1), 90.0))
            fitted_size = estimated_extent
            fitted_center_error = float(np.linalg.norm(fitted_center - gt_center))
            fitted_size_error = float(abs(estimated_extent - gt_half_extent))
        residual_key = "surface"

    gate = {
        "shape": shape,
        "num_recovered_points": int(recovered_points.shape[0]),
        "nan_inf_found": bool(has_nan_inf),
        "median_pixel_error": float(np.median(pixel_errors)) if pixel_errors.size > 0 else float("inf"),
        f"mean_{residual_key}_residual_m": float(np.mean(surface_residuals)) if surface_residuals.size > 0 else float("inf"),
        f"p95_{residual_key}_residual_m": float(np.percentile(surface_residuals, 95.0)) if surface_residuals.size > 0 else float("inf"),
        "fitted_center_m": fitted_center.tolist() if fitted_center is not None else None,
        "fitted_size_m": fitted_size,
        "fitted_center_error_m": fitted_center_error,
        "fitted_size_error_m": fitted_size_error,
        "criteria": {
            "median_pixel_error_le": 1.0,
            f"mean_{residual_key}_residual_m_le": 0.15,
            f"p95_{residual_key}_residual_m_le": 0.30,
            "fitted_center_error_m_le": 0.10,
        },
    }

    if shape == "sphere":
        gate["fitted_radius_m"] = fitted_size
        gate["fitted_radius_error_m"] = fitted_size_error
    else:
        gate["fitted_half_extent_m"] = fitted_size
        gate["fitted_half_extent_error_m"] = fitted_size_error

    criteria = gate["criteria"]

    gate["pass"] = (
        (not gate["nan_inf_found"])
        and gate["median_pixel_error"] <= criteria["median_pixel_error_le"]
        and gate[f"mean_{residual_key}_residual_m"] <= criteria[f"mean_{residual_key}_residual_m_le"]
        and gate[f"p95_{residual_key}_residual_m"] <= criteria[f"p95_{residual_key}_residual_m_le"]
        and gate["fitted_center_error_m"] <= criteria["fitted_center_error_m_le"]
    )

    write_manifest(dataset_root / "consistency_gate.json", gate)
    if recovered_points.shape[0] > 0:
        write_points_ply(dataset_root / "consistency_backprojected_points.ply", recovered_points)

    return gate


def write_dataset(
    args: argparse.Namespace,
    poses: List[PoseRecord],
    images_uint8: List[np.ndarray],
    intrinsics: Dict[str, float | str],
    seeds: Dict[str, int],
    gate_result: Dict | None,
) -> None:
    output_dir = args.output_dir
    sparse_dir = output_dir / "sparse" / "0"
    sonar_dir = output_dir / "sonar"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    sonar_dir.mkdir(parents=True, exist_ok=True)

    for pose, img_u8 in zip(poses, images_uint8):
        img_rgb = np.repeat(img_u8[:, :, None], 3, axis=2)
        Image.fromarray(img_rgb, mode="RGB").save(sonar_dir / pose.image_name)

    write_images_txt(sparse_dir / "images.txt", poses)

    shape = variant_shape(args.variant)
    points_rng = np.random.default_rng(seeds["points3d"])
    if shape == "sphere":
        geom_points = sample_sphere_points(
            center=np.asarray(args.sphere_center, dtype=np.float64),
            radius=float(args.sphere_radius),
            count=int(args.points3d_count),
            rng=points_rng,
        )
        geometry = {
            "shape": "sphere",
            "sphere_center_m": [float(x) for x in args.sphere_center],
            "sphere_radius_m": float(args.sphere_radius),
        }
        dataset_id = "synthetic_sphere_A"
        target_name = "sphere"
        intersection_policy = "nearest positive sphere root only"
    else:
        geom_points = sample_cube_surface_points(
            center=np.asarray(args.cube_center, dtype=np.float64),
            half_extent=float(args.cube_half_extent),
            count=int(args.points3d_count),
            rng=points_rng,
        )
        geometry = {
            "shape": "cube",
            "cube_center_m": [float(x) for x in args.cube_center],
            "cube_half_extent_m": float(args.cube_half_extent),
        }
        dataset_id = "synthetic_cube_C"
        target_name = "cube"
        intersection_policy = "nearest positive axis-aligned cube slab hit only"

    write_points3d_txt(sparse_dir / "points3D.txt", geom_points)

    generated_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    T_c2s = get_camera_to_sonar_transform(device="cpu").detach().cpu().numpy().astype(np.float64)
    if args.pose_mode == "camera_with_extrinsic":
        extrinsic_policy = (
            "Synthetic orbit/forward simulation uses sonar poses. "
            "Exported COLMAP poses are camera poses derived by T_w2c = T_w2s @ inv(T_c2s), "
            "so the runtime camera->sonar extrinsic path is exercised."
        )
        pose_export_contract = "images.txt stores camera poses; sonar pose recovered via camera->sonar extrinsic"
    else:
        extrinsic_policy = (
            "Synthetic poses are emitted as sonar-equivalent world->camera poses "
            "(camera/sonar co-located for this dataset)."
        )
        pose_export_contract = "images.txt stores sonar-equivalent poses (legacy compatibility mode)"

    pose_policy = resolve_pose_policy(args)
    pose_bands_deg = parse_pose_bands_deg(args.pose_bands_deg)
    orientation_policy_terms = [f"look-at {target_name} center"]
    if float(args.orientation_azimuth_jitter_max_deg) > 0.0:
        orientation_policy_terms.append(
            f"uniform azimuth jitter +/-{float(args.orientation_azimuth_jitter_max_deg):.3f} deg"
        )
    if float(args.rotation_jitter_sigma_deg) > 0.0:
        orientation_policy_terms.append(
            f"rotation jitter sigma {float(args.rotation_jitter_sigma_deg):.3f} deg"
        )
    orientation_policy = " + ".join(orientation_policy_terms)

    manifest = {
        "dataset_id": dataset_id,
        "variant": args.variant,
        "generator_version": SCRIPT_VERSION,
        "generated_utc": generated_utc,
        "layout_contract": "COLMAP text sparse/0 + sonar/*.png (R2-compatible)",
        "geometry": geometry,
        "sonar_model": {
            "image_width": int(args.image_width),
            "image_height": int(args.image_height),
            "azimuth_fov_deg": float(args.azimuth_fov_deg),
            "elevation_fov_deg": float(args.elevation_fov_deg),
            "range_min_m": float(args.range_min),
            "range_max_m": float(args.range_max),
        },
        "pose_generation": {
            "num_frames": int(args.num_frames),
            "orbit_radius_m": float(args.orbit_radius),
            "azimuth_coverage_deg": 360.0,
            "elevation_sweep_deg": [float(args.elevation_min_deg), float(args.elevation_max_deg)],
            "pose_policy": pose_policy,
            "pose_bands_deg": [float(x) for x in pose_bands_deg.tolist()],
            "pose_band_jitter_frac": float(args.pose_band_jitter_frac),
            "translation_jitter_sigma_m": float(args.translation_jitter_sigma),
            "rotation_jitter_sigma_deg": float(args.rotation_jitter_sigma_deg),
            "orientation_azimuth_jitter_max_deg": float(args.orientation_azimuth_jitter_max_deg),
            "orientation_policy": orientation_policy,
            "pose_mode": args.pose_mode,
            "pose_export_contract": pose_export_contract,
        },
        "simulation": {
            "elevation_samples": int(args.elevation_samples),
            "intersection_policy": intersection_policy,
            "intensity_model": "first-hit per (azimuth,elevation) ray, collapsed to front envelope per azimuth",
            "normalization": "img_float deposits elevation hit fraction at the nearest range bin(s)",
        },
        "noise_model": {
            "enabled": bool(variant_is_noisy(args.variant)),
            "speckle_sigma": float(args.noise_speckle_sigma),
            "dropout_prob": float(args.noise_dropout_prob),
        },
        "camera_intrinsics": intrinsics,
        "camera_to_sonar_transform_row_major": T_c2s.tolist(),
        "extrinsic_policy": extrinsic_policy,
        "seeds": seeds,
        "artifacts": {
            "images_txt": "sparse/0/images.txt",
            "cameras_txt": "sparse/0/cameras.txt",
            "points3d_txt": "sparse/0/points3D.txt",
            "sonar_dir": "sonar",
            "manifest": "manifest.json",
            "settings": "DATASET_SETTINGS.md",
        },
    }

    if gate_result is not None:
        manifest["consistency_gate"] = {
            "enabled": True,
            "result_path": "consistency_gate.json",
            "criteria": gate_result["criteria"],
            "pass": bool(gate_result["pass"]),
        }
    else:
        manifest["consistency_gate"] = {
            "enabled": False,
        }

    write_manifest(output_dir / "manifest.json", manifest)
    write_dataset_settings_md(output_dir / "DATASET_SETTINGS.md", manifest)


def run_loader_compat_check(dataset_root: Path) -> None:
    _ = readColmapSceneInfo(
        str(dataset_root),
        images=None,
        eval=False,
        sonar_mode=True,
        sonar_images="sonar",
    )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.elevation_samples <= 0:
        raise ValueError("--elevation-samples must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sonar").mkdir(parents=True, exist_ok=True)

    seed_seq = np.random.SeedSequence(int(args.seed))
    child_seqs = seed_seq.spawn(3)
    pose_seed = int(child_seqs[0].generate_state(1)[0])
    noise_seed = int(child_seqs[1].generate_state(1)[0])
    points_seed = int(child_seqs[2].generate_state(1)[0])
    seeds = {
        "master": int(args.seed),
        "pose": pose_seed,
        "noise": noise_seed,
        "points3d": points_seed,
    }

    pose_rng = np.random.default_rng(pose_seed)
    noise_rng = np.random.default_rng(noise_seed)

    sonar_cfg = SonarConfig(
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        azimuth_fov=float(args.azimuth_fov_deg),
        elevation_fov=float(args.elevation_fov_deg),
        range_min=float(args.range_min),
        range_max=float(args.range_max),
        intensity_threshold=0.0,
        device="cpu",
    )

    print(f"[1/6] Building poses ({args.num_frames} frames)")
    poses = build_pose_records(args, pose_rng)

    print(f"[2/6] Precomputing sonar rays (W={args.image_width}, elev_samples={args.elevation_samples})")
    dirs_s_flat, col_idx_flat = precompute_sonar_directions(sonar_cfg, args.elevation_samples)

    print("[3/6] Rendering sonar frames")
    images_uint8: List[np.ndarray] = []
    shape = variant_shape(args.variant)
    geometry: Dict[str, object]
    if shape == "sphere":
        geometry = {
            "sphere_center_m": [float(x) for x in args.sphere_center],
            "sphere_radius_m": float(args.sphere_radius),
        }
    else:
        geometry = {
            "cube_center_m": [float(x) for x in args.cube_center],
            "cube_half_extent_m": float(args.cube_half_extent),
        }

    for idx, pose in enumerate(poses):
        img_float = render_sonar_frame(
            pose=pose,
            dirs_s_flat=dirs_s_flat,
            col_idx_flat=col_idx_flat,
            sonar_cfg=sonar_cfg,
            shape=shape,
            geometry=geometry,
            elev_samples=int(args.elevation_samples),
        )
        img_float = apply_noise_if_needed(
            img_float,
            variant=args.variant,
            rng=noise_rng,
            speckle_sigma=float(args.noise_speckle_sigma),
            dropout_prob=float(args.noise_dropout_prob),
        )
        img_u8 = np.round(np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)
        images_uint8.append(img_u8)

        if (idx + 1) % max(1, args.num_frames // 10) == 0 or (idx + 1) == args.num_frames:
            print(f"  rendered {idx + 1}/{args.num_frames}")

    print("[4/6] Writing COLMAP + sonar dataset files")
    intrinsics = write_cameras_txt(args.output_dir / "sparse" / "0" / "cameras.txt", args)
    write_dataset(
        args=args,
        poses=poses,
        images_uint8=images_uint8,
        intrinsics=intrinsics,
        seeds=seeds,
        gate_result=None,
    )

    gate_result = None
    if args.run_consistency_gate:
        print("[5/6] Running backward projection consistency gate")
        gate_result = run_backward_projection_gate(
            dataset_root=args.output_dir,
            poses=poses,
            images_uint8=images_uint8,
            sonar_cfg=sonar_cfg,
            shape=shape,
            geometry=geometry,
            threshold=float(args.gate_intensity_threshold),
            mask_top_rows=int(args.gate_mask_top_rows),
        )
        residual_key = "radial" if shape == "sphere" else "surface"
        print(
            "  gate: "
            f"pass={gate_result['pass']}, "
            f"median_px={gate_result['median_pixel_error']:.4f}, "
            f"mean_res={gate_result[f'mean_{residual_key}_residual_m']:.4f}m, "
            f"p95_res={gate_result[f'p95_{residual_key}_residual_m']:.4f}m"
        )

        manifest_path = args.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["consistency_gate"] = {
            "enabled": True,
            "result_path": "consistency_gate.json",
            "criteria": gate_result["criteria"],
            "pass": bool(gate_result["pass"]),
        }
        write_manifest(manifest_path, manifest)
        write_dataset_settings_md(args.output_dir / "DATASET_SETTINGS.md", manifest)

    if args.run_loader_check:
        print("[6/6] Loader compatibility check via readColmapSceneInfo")
        run_loader_compat_check(args.output_dir)
    else:
        print("[6/6] Loader compatibility check skipped")

    print("Done.")
    print(f"Dataset root: {args.output_dir}")
    print("Expected debug command:")
    dataset_key = "synthetic_a_clean" if shape == "sphere" else "synthetic_c_clean"
    print(
        f"SONAR_DATASET={dataset_key} "
        "SONAR_DATASET_PATH="
        f"{args.output_dir} "
        "SONAR_INIT_SCALE_FACTOR=1.0 "
        "python debug_multiframe.py"
    )


if __name__ == "__main__":
    main()
