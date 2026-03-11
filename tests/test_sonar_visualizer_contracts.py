import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_UTILS_PATH = REPO_ROOT / "utils" / "visualization_utils.py"


def _load_visualization_utils_module():
    spec = importlib.util.spec_from_file_location("visualization_utils", VIS_UTILS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vis = _load_visualization_utils_module()


def test_full_range_wireframe_corners_stay_on_sonar_range_shell():
    position = np.zeros(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    range_max = 3.0

    full_vertices = vis.full_range_fov_vertices(
        position,
        rotation,
        range_value=range_max,
        azimuth_fov=120.0,
        elevation_fov=20.0,
    )
    near_vertices = vis.legacy_pose_pyramid_vertices(
        position,
        rotation,
        depth=range_max,
        azimuth_fov=120.0,
        elevation_fov=20.0,
    )

    full_corner_dists = np.linalg.norm(full_vertices[1:], axis=1)
    near_corner_dists = np.linalg.norm(near_vertices[1:], axis=1)

    assert np.allclose(full_corner_dists, range_max, atol=1e-6)
    assert not np.allclose(near_corner_dists, range_max, atol=1e-3)


def test_near_and_full_range_wireframes_share_same_local_fov_angles():
    near_local = vis.sonar_fov_vertices_local(0.5, azimuth_fov=120.0, elevation_fov=20.0)
    far_local = vis.sonar_fov_vertices_local(3.0, azimuth_fov=120.0, elevation_fov=20.0)

    def _angles(vertices_local):
        rel = vertices_local[1:]
        forward = rel[:, 0]
        right = rel[:, 1]
        down = rel[:, 2]
        azimuth = np.degrees(np.arctan2(right, forward))
        elevation = np.degrees(np.arctan2(down, np.sqrt(right * right + forward * forward)))
        return azimuth, elevation

    near_az, near_el = _angles(near_local)
    far_az, far_el = _angles(far_local)
    assert np.allclose(near_az, far_az, atol=1e-6)
    assert np.allclose(near_el, far_el, atol=1e-6)
    assert np.allclose(np.abs(near_az), 60.0, atol=1e-6)
    assert np.allclose(np.abs(near_el), 10.0, atol=1e-6)


def test_prepare_surfel_visualization_state_activates_scales_and_normalizes_quaternions():
    state = vis.prepare_surfel_visualization_state(
        centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        scales=np.log(np.array([[2.0, 4.0]], dtype=np.float64)),
        rotations=np.array([[2.0, 0.0, 2.0, 0.0]], dtype=np.float64),
        opacity=np.array([0.75], dtype=np.float64),
        scales_are_latent=True,
        rotations_are_normalized=False,
    )

    assert np.allclose(state["scales"], np.array([[2.0, 4.0]], dtype=np.float64))
    assert np.allclose(np.linalg.norm(state["rotations"], axis=1), np.array([1.0], dtype=np.float64))
    assert np.allclose(state["normals"], np.array([[1.0, 0.0, 0.0]], dtype=np.float64), atol=1e-6)
    assert state["equivalent_radius"][0] == pytest.approx(np.sqrt(8.0))


def test_front_and_back_face_exports_land_on_opposite_sides_of_normal():
    state = vis.prepare_surfel_visualization_state(
        centers=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        scales=np.log(np.array([[1.5, 0.5]], dtype=np.float64)),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        opacity=np.array([1.0], dtype=np.float64),
        scales_are_latent=True,
        rotations_are_normalized=False,
    )

    front_geom = vis.build_surfel_glyph_geometry(state, indices=[0], face_mode="front", ellipse_segments=8)
    back_geom = vis.build_surfel_glyph_geometry(state, indices=[0], face_mode="back", ellipse_segments=8)

    front_center = front_geom["vertices"][0]
    back_center = back_geom["vertices"][0]
    center = state["centers"][0]
    normal = state["normals"][0]

    assert np.dot(front_center - center, normal) > 0.0
    assert np.dot(back_center - center, normal) < 0.0


def test_visualizer_manifest_paths_resolve_to_real_files(tmp_path):
    visualizer_dir, rendered_dir = vis.ensure_visualizer_dirs(str(tmp_path))
    near_path = Path(visualizer_dir) / "000_frame_wireframe_near.ply"
    full_path = Path(visualizer_dir) / "000_frame_wireframe_full_range.ply"
    glyph_path = Path(visualizer_dir) / "000_frame_surfels_in_fov.ply"
    rendered_path = Path(rendered_dir) / "000_frame.png"

    near_path.write_text("near", encoding="utf-8")
    full_path.write_text("full", encoding="utf-8")
    glyph_path.write_text("glyph", encoding="utf-8")
    rendered_path.write_bytes(b"png")

    manifest = {
        "frame_artifacts": [
            {
                "wireframe_near": near_path.name,
                "wireframe_full_range": full_path.name,
                "surfels_in_fov": glyph_path.name,
                "rendered_image": f"rendered/{rendered_path.name}",
            }
        ]
    }
    manifest_path = Path(visualizer_dir) / "manifest.json"
    vis.write_visualizer_manifest(str(manifest_path), manifest)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_entry = loaded["frame_artifacts"][0]
    for key in ("wireframe_near", "wireframe_full_range", "surfels_in_fov", "rendered_image"):
        assert (Path(visualizer_dir) / frame_entry[key]).exists()


def test_frame_surfel_selection_prioritizes_centers_inside_fov():
    selected = vis.select_frame_surfel_indices(
        center_in_fov=np.array([False, True, True, False]),
        overlap_mask=np.array([True, True, True, True]),
        fov_margin=np.array([0.9, 0.1, 0.8, 2.0]),
        facing_score=np.array([0.9, 0.2, 0.3, 0.99]),
        opacity=np.array([0.9, 0.9, 0.9, 0.9]),
        eq_radius=np.array([0.1, 0.1, 0.1, 0.1]),
        max_count=2,
    )
    assert selected.tolist() == [2, 1]
