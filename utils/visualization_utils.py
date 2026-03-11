import json
import math
import os
import re
from typing import Dict, Optional, Sequence

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def _require_open3d():
    if o3d is None:
        raise ImportError("open3d is required for mesh/PLY export functions")


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def ensure_visualizer_dirs(output_dir: str):
    visualizer_dir = os.path.join(output_dir, "visualizer")
    rendered_dir = os.path.join(visualizer_dir, "rendered")
    os.makedirs(visualizer_dir, exist_ok=True)
    os.makedirs(rendered_dir, exist_ok=True)
    return visualizer_dir, rendered_dir


def sanitize_image_name(image_name: str) -> str:
    base = os.path.splitext(os.path.basename(str(image_name)))[0]
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._")
    return sanitized or "frame"


def build_frame_stem(frame_idx: int, image_name: str, width: int = 3) -> str:
    return f"{frame_idx:0{width}d}_{sanitize_image_name(image_name)}"


def activate_scales(scales) -> np.ndarray:
    scales_np = _to_numpy(scales).astype(np.float64)
    return np.exp(scales_np)


def normalize_quaternions(quaternions) -> np.ndarray:
    q = _to_numpy(quaternions).astype(np.float64)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return q / norms


def quaternion_to_rotation_matrices(quaternions) -> np.ndarray:
    q = normalize_quaternions(quaternions)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - w * z)
    rot[:, 0, 2] = 2.0 * (x * z + w * y)
    rot[:, 1, 0] = 2.0 * (x * y + w * z)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - w * x)
    rot[:, 2, 0] = 2.0 * (x * z - w * y)
    rot[:, 2, 1] = 2.0 * (y * z + w * x)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def compute_surfel_axes(quaternions):
    rot = quaternion_to_rotation_matrices(quaternions)
    tangent_u = rot[:, :, 0]
    tangent_v = rot[:, :, 1]
    normals = rot[:, :, 2]
    return tangent_u, tangent_v, normals


def compute_equivalent_radius(scales) -> np.ndarray:
    scales_np = _to_numpy(scales).astype(np.float64)
    if scales_np.shape[1] < 2:
        raise ValueError(f"Expected at least 2 scale channels, got {scales_np.shape}")
    return np.sqrt(np.clip(scales_np[:, 0] * scales_np[:, 1], 0.0, None))


def prepare_surfel_visualization_state(
    *,
    centers,
    scales,
    rotations,
    opacity=None,
    scales_are_latent: bool = True,
    rotations_are_normalized: bool = False,
) -> Dict[str, np.ndarray]:
    centers_np = _to_numpy(centers).astype(np.float64)
    scales_np = activate_scales(scales) if scales_are_latent else _to_numpy(scales).astype(np.float64)
    rotations_np = normalize_quaternions(rotations) if not rotations_are_normalized else _to_numpy(rotations).astype(np.float64)
    tangent_u, tangent_v, normals = compute_surfel_axes(rotations_np)
    eq_radius = compute_equivalent_radius(scales_np)
    opacity_np = None if opacity is None else _to_numpy(opacity).reshape(-1).astype(np.float64)
    return {
        "centers": centers_np,
        "scales": scales_np,
        "rotations": rotations_np,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "normals": normals,
        "equivalent_radius": eq_radius,
        "opacity": opacity_np,
    }


def deterministic_glyph_indices(
    opacity,
    eq_radius,
    *,
    max_count: Optional[int],
    opacity_percentile: float = 0.0,
    eq_radius_percentile: float = 100.0,
) -> np.ndarray:
    opacity_np = np.asarray(opacity, dtype=np.float64).reshape(-1)
    eq_np = np.asarray(eq_radius, dtype=np.float64).reshape(-1)
    if opacity_np.shape != eq_np.shape:
        raise ValueError("opacity and eq_radius must have matching shapes")
    if opacity_np.size == 0:
        return np.zeros(0, dtype=np.int64)

    keep = np.ones(opacity_np.shape[0], dtype=bool)
    if opacity_percentile > 0.0:
        opacity_cut = np.percentile(opacity_np, opacity_percentile)
        keep &= opacity_np >= opacity_cut
    if eq_radius_percentile < 100.0:
        eq_cut = np.percentile(eq_np, eq_radius_percentile)
        keep &= eq_np <= eq_cut

    kept = np.flatnonzero(keep)
    if kept.size == 0:
        kept = np.arange(opacity_np.shape[0], dtype=np.int64)

    order = np.lexsort((kept, -eq_np[kept], -opacity_np[kept]))
    ranked = kept[order]
    if max_count is None or max_count <= 0 or ranked.size <= max_count:
        return ranked
    return ranked[:max_count]


def build_point_cloud(points, colors=None):
    _require_open3d()
    point_cloud = o3d.geometry.PointCloud()
    points_np = _to_numpy(points).astype(np.float64)
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    if colors is not None:
        colors_np = np.clip(_to_numpy(colors).astype(np.float64), 0.0, 1.0)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_np)
    return point_cloud


def write_point_cloud(path: str, points, colors=None) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, build_point_cloud(points, colors))
    return path


def legacy_pose_pyramid_vertices(position, rotation_matrix, depth, azimuth_fov, elevation_fov):
    half_az = math.radians(float(azimuth_fov) / 2.0)
    half_el = math.radians(float(elevation_fov) / 2.0)
    width = 2.0 * depth * math.tan(half_az)
    height = 2.0 * depth * math.tan(half_el)
    vertices_local = np.array(
        [
            [0.0, 0.0, 0.0],
            [depth, -width / 2.0, -height / 2.0],
            [depth, width / 2.0, -height / 2.0],
            [depth, width / 2.0, height / 2.0],
            [depth, -width / 2.0, height / 2.0],
        ],
        dtype=np.float64,
    )
    return _pose_vertices_local_to_world(vertices_local, position, rotation_matrix)


def sonar_fov_vertices_local(range_value, azimuth_fov, elevation_fov):
    half_az = math.radians(float(azimuth_fov) / 2.0)
    half_el = math.radians(float(elevation_fov) / 2.0)
    vertices_local = [[0.0, 0.0, 0.0]]
    corner_angles = [
        (half_az, -half_el),
        (-half_az, -half_el),
        (-half_az, half_el),
        (half_az, half_el),
    ]
    for azimuth, elevation in corner_angles:
        right = -range_value * math.sin(azimuth) * math.cos(elevation)
        down = range_value * math.sin(elevation)
        forward = range_value * math.cos(azimuth) * math.cos(elevation)
        vertices_local.append([forward, right, down])
    return np.asarray(vertices_local, dtype=np.float64)


def full_range_fov_vertices(position, rotation_matrix, range_value, azimuth_fov, elevation_fov):
    vertices_local = sonar_fov_vertices_local(range_value, azimuth_fov, elevation_fov)
    return _pose_vertices_local_to_world(vertices_local, position, rotation_matrix)


def _pose_vertices_local_to_world(vertices_local, position, rotation_matrix):
    position_np = _to_numpy(position).astype(np.float64)
    rotation_np = _to_numpy(rotation_matrix).astype(np.float64)
    cam_z_world = rotation_np[:, 2]
    cam_x_world = rotation_np[:, 0]
    cam_y_world = rotation_np[:, 1]
    rotation_local_to_world = np.column_stack([cam_z_world, cam_x_world, cam_y_world])
    return (rotation_local_to_world @ vertices_local.T).T + position_np


def create_pose_wireframe(
    position,
    rotation_matrix,
    *,
    depth: float,
    azimuth_fov: float,
    elevation_fov: float,
    color: Sequence[float],
    mode: str = "near",
):
    _require_open3d()
    if mode == "near":
        vertices = full_range_fov_vertices(position, rotation_matrix, depth, azimuth_fov, elevation_fov)
    elif mode == "full_range":
        vertices = full_range_fov_vertices(position, rotation_matrix, depth, azimuth_fov, elevation_fov)
    else:
        raise ValueError(f"Unsupported wireframe mode: {mode}")

    edges = np.asarray([[0, 1], [0, 2], [0, 4], [0, 3], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
    wireframe = o3d.geometry.LineSet()
    wireframe.points = o3d.utility.Vector3dVector(vertices)
    wireframe.lines = o3d.utility.Vector2iVector(edges)
    wireframe.paint_uniform_color(np.asarray(color, dtype=np.float64))
    return wireframe


def write_line_set(path: str, line_set) -> str:
    _require_open3d()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_line_set(path, line_set)
    return path


def _resolve_scale_value(scale_factor) -> float:
    if scale_factor is None:
        return 1.0
    if hasattr(scale_factor, "scale"):
        scale_value = scale_factor.scale
    elif hasattr(scale_factor, "get_scale_value"):
        scale_value = scale_factor.get_scale_value()
    else:
        scale_value = scale_factor
    scale_np = _to_numpy(scale_value).reshape(-1)
    return float(scale_np[0])


def compute_frame_surfel_membership(centers, scales, rotations, camera, sonar_config, scale_factor=None):
    centers_np = _to_numpy(centers).astype(np.float64)
    scales_np = _to_numpy(scales).astype(np.float64)
    rotations_np = normalize_quaternions(rotations)
    _, _, normals_world = compute_surfel_axes(rotations_np)

    w2v = _to_numpy(camera.world_view_transform).astype(np.float64)
    rotation = w2v[:3, :3]
    translation = w2v[3, :3]
    scale_value = _resolve_scale_value(scale_factor)

    points_view = (centers_np * scale_value) @ rotation.T + (translation * scale_value)
    normals_view = normals_world @ rotation.T

    right = points_view[:, 0]
    down = points_view[:, 1]
    forward = points_view[:, 2]
    horiz_dist = np.sqrt(np.clip(right * right + forward * forward, 0.0, None))
    range_vals = np.linalg.norm(points_view, axis=1)
    azimuth = np.arctan2(right, forward)
    elevation = np.arctan2(down, np.clip(horiz_dist, 1e-12, None))

    half_az = math.radians(float(sonar_config.azimuth_fov) / 2.0)
    half_el = math.radians(float(sonar_config.elevation_fov) / 2.0)
    az_margin = (half_az - np.abs(azimuth)) * range_vals
    el_margin = (half_el - np.abs(elevation)) * range_vals
    range_margin_near = range_vals - float(sonar_config.range_min)
    range_margin_far = float(sonar_config.range_max) - range_vals
    fov_margin = np.min(
        np.stack([az_margin, el_margin, range_margin_near, range_margin_far], axis=0),
        axis=0,
    )
    surfel_radius = np.max(scales_np[:, :2], axis=1)
    in_front = forward > 0.0
    center_in_fov = (
        in_front
        & (np.abs(azimuth) <= half_az)
        & (np.abs(elevation) <= half_el)
        & (range_vals >= float(sonar_config.range_min))
        & (range_vals <= float(sonar_config.range_max))
    )
    overlap_mask = in_front & ((fov_margin + surfel_radius) > 0.0)

    dir_to_sonar = -points_view / np.clip(range_vals[:, None], 1e-12, None)
    facing_score = np.sum(normals_view * dir_to_sonar, axis=1)

    return {
        "points_view": points_view,
        "center_in_fov": center_in_fov,
        "overlap_mask": overlap_mask,
        "fov_margin": fov_margin,
        "surfel_radius": surfel_radius,
        "range_vals": range_vals,
        "azimuth_rad": azimuth,
        "elevation_rad": elevation,
        "facing_score": facing_score,
    }


def select_frame_surfel_indices(
    center_in_fov,
    overlap_mask,
    fov_margin,
    facing_score,
    opacity,
    eq_radius,
    *,
    max_count: int,
):
    center_in_fov = np.asarray(center_in_fov, dtype=bool).reshape(-1)
    overlap_mask = np.asarray(overlap_mask, dtype=bool).reshape(-1)
    fov_margin = np.asarray(fov_margin, dtype=np.float64).reshape(-1)
    facing_score = np.asarray(facing_score, dtype=np.float64).reshape(-1)
    opacity = np.asarray(opacity, dtype=np.float64).reshape(-1)
    eq_radius = np.asarray(eq_radius, dtype=np.float64).reshape(-1)

    candidate_idx = np.flatnonzero(overlap_mask)
    if candidate_idx.size == 0 or max_count <= 0:
        return np.zeros(0, dtype=np.int64)

    order = np.lexsort(
        (
            candidate_idx,
            eq_radius[candidate_idx],
            -opacity[candidate_idx],
            -facing_score[candidate_idx],
            -fov_margin[candidate_idx],
            ~center_in_fov[candidate_idx],
        )
    )
    ranked = candidate_idx[order]
    return ranked[:max_count]


def diagnostic_colors_from_metrics(facing_score, range_vals, range_min, range_max):
    facing_np = np.asarray(facing_score, dtype=np.float64).reshape(-1)
    range_np = np.asarray(range_vals, dtype=np.float64).reshape(-1)
    facing_mix = np.clip((facing_np + 1.0) * 0.5, 0.0, 1.0)
    range_span = max(float(range_max) - float(range_min), 1e-12)
    range_mix = np.clip((range_np - float(range_min)) / range_span, 0.0, 1.0)

    facing_bad = np.array([0.88, 0.25, 0.16], dtype=np.float64)
    facing_good = np.array([0.24, 0.78, 0.35], dtype=np.float64)
    near_color = np.array([0.98, 0.84, 0.20], dtype=np.float64)
    far_color = np.array([0.20, 0.66, 0.96], dtype=np.float64)

    facing_color = facing_bad[None, :] * (1.0 - facing_mix[:, None]) + facing_good[None, :] * facing_mix[:, None]
    range_color = near_color[None, :] * (1.0 - range_mix[:, None]) + far_color[None, :] * range_mix[:, None]
    return np.clip(0.55 * facing_color + 0.45 * range_color, 0.0, 1.0)


def build_surfel_glyph_geometry(
    surfel_state: Dict[str, np.ndarray],
    *,
    indices=None,
    face_mode: str = "double",
    base_colors=None,
    ellipse_segments: int = 12,
    face_offset_scale: float = 0.04,
    normal_stem_scale: float = 0.35,
    stem_width_scale: float = 0.08,
):
    centers = surfel_state["centers"]
    scales = surfel_state["scales"]
    tangent_u = surfel_state["tangent_u"]
    tangent_v = surfel_state["tangent_v"]
    normals = surfel_state["normals"]
    eq_radius = surfel_state["equivalent_radius"]

    if indices is None:
        selected = np.arange(centers.shape[0], dtype=np.int64)
    else:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)

    if selected.size == 0:
        return {
            "vertices": np.zeros((0, 3), dtype=np.float64),
            "triangles": np.zeros((0, 3), dtype=np.int32),
            "vertex_colors": np.zeros((0, 3), dtype=np.float64),
        }

    if base_colors is None:
        base_colors = np.tile(np.array([[0.75, 0.75, 0.75]], dtype=np.float64), (selected.size, 1))
    else:
        base_colors = np.clip(np.asarray(base_colors, dtype=np.float64), 0.0, 1.0)
        if base_colors.shape[0] != selected.size:
            raise ValueError("base_colors must match selected glyph count")

    if face_mode == "front":
        face_signs = [1.0]
    elif face_mode == "back":
        face_signs = [-1.0]
    elif face_mode == "double":
        face_signs = [1.0, -1.0]
    else:
        raise ValueError(f"Unsupported face_mode: {face_mode}")

    front_tint = np.array([0.96, 0.42, 0.18], dtype=np.float64)
    back_tint = np.array([0.15, 0.58, 0.95], dtype=np.float64)
    stem_tint = np.array([0.97, 0.90, 0.58], dtype=np.float64)

    vertices = []
    triangles = []
    colors = []

    for row_idx, surfel_idx in enumerate(selected):
        center = centers[surfel_idx]
        axis_u = tangent_u[surfel_idx] * scales[surfel_idx, 0]
        axis_v = tangent_v[surfel_idx] * scales[surfel_idx, 1]
        normal = normals[surfel_idx]
        radius = max(float(eq_radius[surfel_idx]), 1e-6)
        base_color = base_colors[row_idx]
        offset = normal * max(radius * face_offset_scale, 1e-6)

        for face_sign in face_signs:
            center_face = center + face_sign * offset
            face_tint = front_tint if face_sign > 0.0 else back_tint
            face_color = np.clip(0.6 * face_tint + 0.4 * base_color, 0.0, 1.0)

            center_idx = len(vertices)
            vertices.append(center_face)
            colors.append(face_color)
            ring_indices = []
            for segment_idx in range(ellipse_segments):
                angle = 2.0 * math.pi * float(segment_idx) / float(ellipse_segments)
                point = center_face + math.cos(angle) * axis_u + math.sin(angle) * axis_v
                ring_indices.append(len(vertices))
                vertices.append(point)
                colors.append(face_color)

            for segment_idx in range(ellipse_segments):
                a_idx = ring_indices[segment_idx]
                b_idx = ring_indices[(segment_idx + 1) % ellipse_segments]
                if face_sign > 0.0:
                    triangles.append([center_idx, a_idx, b_idx])
                else:
                    triangles.append([center_idx, b_idx, a_idx])

        stem_length = max(radius * normal_stem_scale, 1e-6)
        stem_half_width = max(min(scales[surfel_idx, 0], scales[surfel_idx, 1]) * stem_width_scale, radius * 0.03)
        stem_base = center
        stem_tip = center + normal * stem_length
        stem_side = tangent_u[surfel_idx] * stem_half_width
        stem_color = np.clip(0.5 * stem_tint + 0.5 * base_color, 0.0, 1.0)
        stem_vertices = [
            stem_base - stem_side,
            stem_base + stem_side,
            stem_tip + stem_side,
            stem_tip - stem_side,
        ]
        stem_start = len(vertices)
        vertices.extend(stem_vertices)
        colors.extend([stem_color] * 4)
        triangles.append([stem_start + 0, stem_start + 1, stem_start + 2])
        triangles.append([stem_start + 0, stem_start + 2, stem_start + 3])

    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "triangles": np.asarray(triangles, dtype=np.int32),
        "vertex_colors": np.asarray(colors, dtype=np.float64),
    }


def build_surfel_glyph_mesh(
    surfel_state: Dict[str, np.ndarray],
    *,
    indices=None,
    face_mode: str = "double",
    base_colors=None,
    ellipse_segments: int = 12,
    face_offset_scale: float = 0.04,
    normal_stem_scale: float = 0.35,
    stem_width_scale: float = 0.08,
):
    _require_open3d()
    geometry = build_surfel_glyph_geometry(
        surfel_state,
        indices=indices,
        face_mode=face_mode,
        base_colors=base_colors,
        ellipse_segments=ellipse_segments,
        face_offset_scale=face_offset_scale,
        normal_stem_scale=normal_stem_scale,
        stem_width_scale=stem_width_scale,
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(geometry["vertices"])
    mesh.triangles = o3d.utility.Vector3iVector(geometry["triangles"])
    mesh.vertex_colors = o3d.utility.Vector3dVector(geometry["vertex_colors"])
    return mesh


def write_triangle_mesh(path: str, mesh) -> str:
    _require_open3d()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_triangle_mesh(path, mesh)
    return path


def write_visualizer_manifest(path: str, manifest: Dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path
