#!/usr/bin/env python3
"""
Generate a mesh of pyramids showing sonar poses and FOV.

Each pyramid:
- Tip at the sonar/camera position
- Base oriented along the viewing direction
- Base dimensions match sonar FOV (120° azimuth, 20° elevation)
- Depth of pyramid = configurable (default 0.5m for visibility)
"""

import os
import sys
import numpy as np
import open3d as o3d
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argparse import Namespace
from scene import Scene, GaussianModel

# Configuration
DATASET_PATH = "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs"
OUTPUT_DIR = "./output/debug_before_after"
PYRAMID_DEPTH = 0.3  # meters - how far the pyramid extends from the pose
AZIMUTH_FOV = 120.0  # degrees
ELEVATION_FOV = 20.0  # degrees

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_sonar_pyramid(position, rotation_matrix, depth=PYRAMID_DEPTH, 
                         azimuth_fov=AZIMUTH_FOV, elevation_fov=ELEVATION_FOV):
    """
    Create a pyramid mesh representing the sonar FOV.
    
    Args:
        position: [3] camera/sonar position in world coordinates
        rotation_matrix: [3, 3] rotation from camera to world (columns are camera axes in world)
        depth: How far the pyramid extends
        azimuth_fov: Horizontal FOV in degrees
        elevation_fov: Vertical FOV in degrees
        
    Returns:
        open3d.geometry.TriangleMesh of the pyramid
    """
    # Convert FOV to radians
    half_az = math.radians(azimuth_fov / 2)
    half_el = math.radians(elevation_fov / 2)
    
    # Calculate base dimensions at the given depth
    # For sonar: +X is forward, +Y is right, +Z is down
    # We'll create pyramid in local frame then transform
    
    # Base corners in local sonar frame (origin at sonar, +X forward)
    # At distance 'depth' along +X axis
    width = 2 * depth * math.tan(half_az)   # horizontal span
    height = 2 * depth * math.tan(half_el)  # vertical span
    
    # 5 vertices: tip (origin) + 4 base corners
    # Local frame: X=forward, Y=right, Z=down
    vertices_local = np.array([
        [0, 0, 0],                                    # 0: tip (sonar position)
        [depth, -width/2, -height/2],                 # 1: base top-left
        [depth,  width/2, -height/2],                 # 2: base top-right
        [depth,  width/2,  height/2],                 # 3: base bottom-right
        [depth, -width/2,  height/2],                 # 4: base bottom-left
    ])
    
    # Triangular faces (tip to each edge of base + base quad as 2 triangles)
    faces = np.array([
        [0, 1, 2],  # top face
        [0, 2, 3],  # right face
        [0, 3, 4],  # bottom face
        [0, 4, 1],  # left face
        [1, 3, 2],  # base triangle 1
        [1, 4, 3],  # base triangle 2
    ])
    
    # Transform to world coordinates
    # Camera convention (COLMAP/OpenCV): +X right, +Y down, +Z forward
    # We need to convert from camera frame to world frame
    # rotation_matrix columns are camera X, Y, Z axes in world coordinates
    
    # Camera axes in world frame
    cam_x_world = rotation_matrix[:, 0]  # camera +X (right) in world
    cam_y_world = rotation_matrix[:, 1]  # camera +Y (down) in world  
    cam_z_world = rotation_matrix[:, 2]  # camera +Z (forward) in world
    
    # For the pyramid, we want:
    # - Pyramid +X (forward) = camera +Z (forward)
    # - Pyramid +Y (right) = camera +X (right)
    # - Pyramid +Z (down) = camera +Y (down)
    
    # Build transformation: pyramid_local -> world
    R_local_to_world = np.column_stack([cam_z_world, cam_x_world, cam_y_world])
    
    # Transform vertices
    vertices_world = (R_local_to_world @ vertices_local.T).T + position
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    return mesh


def main():
    print("=" * 60)
    print("GENERATING POSE PYRAMID VISUALIZATION")
    print(f"Pyramid depth: {PYRAMID_DEPTH}m")
    print(f"Azimuth FOV: {AZIMUTH_FOV}°")
    print(f"Elevation FOV: {ELEVATION_FOV}°")
    print("=" * 60)
    
    # Setup dataset arguments
    dataset_args = Namespace(
        source_path=DATASET_PATH,
        model_path=OUTPUT_DIR,
        images="images",
        resolution=2,
        white_background=False,
        data_device="cpu",
        eval=False,
        sh_degree=3,
        sonar_mode=True,
        sonar_images="sonar",
        sonar_azimuth_fov=AZIMUTH_FOV,
        sonar_elevation_fov=ELEVATION_FOV,
        sonar_range_min=0.1,
        sonar_range_max=30.0,
        sonar_intensity_threshold=0.01,
        gamma=2.2,
    )
    
    print("\nLoading scene...")
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians, shuffle=False)
    
    train_cameras = scene.getTrainCameras()
    print(f"Loaded {len(train_cameras)} camera poses")
    
    # Create combined mesh for all pyramids
    combined_mesh = o3d.geometry.TriangleMesh()
    
    print("\nGenerating pyramids...")
    for i, cam in enumerate(train_cameras):
        # Get camera pose
        # cam.R is world-to-camera rotation (3x3)
        # cam.T is world-to-camera translation (3,)
        # Camera position in world = -R^T @ T
        
        R_w2c = cam.R  # [3, 3] world-to-camera rotation
        T_w2c = cam.T  # [3] world-to-camera translation
        
        # Camera position in world coordinates
        R_c2w = R_w2c.T  # camera-to-world rotation
        position = -R_c2w @ T_w2c  # camera center in world
        
        # Create pyramid for this pose
        pyramid = create_sonar_pyramid(position, R_c2w, depth=PYRAMID_DEPTH)
        
        # Color the pyramid (use gradient based on timestamp for easy identification)
        # Normalize index to [0, 1] for color mapping
        t = i / max(1, len(train_cameras) - 1)
        # Color: blue -> cyan -> green -> yellow -> red
        if t < 0.25:
            r, g, b = 0, t * 4, 1
        elif t < 0.5:
            r, g, b = 0, 1, 1 - (t - 0.25) * 4
        elif t < 0.75:
            r, g, b = (t - 0.5) * 4, 1, 0
        else:
            r, g, b = 1, 1 - (t - 0.75) * 4, 0
        
        pyramid.paint_uniform_color([r, g, b])
        
        # Add to combined mesh
        combined_mesh += pyramid
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(train_cameras)} poses")
    
    print(f"\nTotal pyramids: {len(train_cameras)}")
    print(f"Combined mesh vertices: {len(combined_mesh.vertices)}")
    print(f"Combined mesh triangles: {len(combined_mesh.triangles)}")
    
    # Save the mesh
    output_path = os.path.join(OUTPUT_DIR, "pose_pyramids.ply")
    o3d.io.write_triangle_mesh(output_path, combined_mesh)
    print(f"\nSaved: {output_path}")
    
    # Also save a wireframe version (edges only) for cleaner overlay
    print("\nGenerating wireframe version...")
    wireframe_lines = []
    wireframe_points = []
    point_offset = 0
    
    for i, cam in enumerate(train_cameras):
        R_w2c = cam.R
        T_w2c = cam.T
        R_c2w = R_w2c.T
        position = -R_c2w @ T_w2c
        
        # Get pyramid vertices in world frame
        half_az = math.radians(AZIMUTH_FOV / 2)
        half_el = math.radians(ELEVATION_FOV / 2)
        width = 2 * PYRAMID_DEPTH * math.tan(half_az)
        height = 2 * PYRAMID_DEPTH * math.tan(half_el)
        
        vertices_local = np.array([
            [0, 0, 0],
            [PYRAMID_DEPTH, -width/2, -height/2],
            [PYRAMID_DEPTH,  width/2, -height/2],
            [PYRAMID_DEPTH,  width/2,  height/2],
            [PYRAMID_DEPTH, -width/2,  height/2],
        ])
        
        cam_z_world = R_c2w[:, 2]
        cam_x_world = R_c2w[:, 0]
        cam_y_world = R_c2w[:, 1]
        R_local_to_world = np.column_stack([cam_z_world, cam_x_world, cam_y_world])
        vertices_world = (R_local_to_world @ vertices_local.T).T + position
        
        wireframe_points.extend(vertices_world.tolist())
        
        # Edges: tip to each base corner, then base rectangle
        edges = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # tip to corners
            [1, 2], [2, 3], [3, 4], [4, 1],  # base rectangle
        ]
        for e in edges:
            wireframe_lines.append([e[0] + point_offset, e[1] + point_offset])
        
        point_offset += 5
    
    wireframe = o3d.geometry.LineSet()
    wireframe.points = o3d.utility.Vector3dVector(np.array(wireframe_points))
    wireframe.lines = o3d.utility.Vector2iVector(np.array(wireframe_lines))
    
    # Color wireframe by time (same color scheme)
    colors = []
    for i in range(len(train_cameras)):
        t = i / max(1, len(train_cameras) - 1)
        if t < 0.25:
            r, g, b = 0, t * 4, 1
        elif t < 0.5:
            r, g, b = 0, 1, 1 - (t - 0.25) * 4
        elif t < 0.75:
            r, g, b = (t - 0.5) * 4, 1, 0
        else:
            r, g, b = 1, 1 - (t - 0.75) * 4, 0
        # 8 edges per pyramid
        for _ in range(8):
            colors.append([r, g, b])
    
    wireframe.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    wireframe_path = os.path.join(OUTPUT_DIR, "pose_pyramids_wireframe.ply")
    o3d.io.write_line_set(wireframe_path, wireframe)
    print(f"Saved: {wireframe_path}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {output_path}")
    print(f"     (Solid pyramids - good for checking FOV coverage)")
    print(f"  2. {wireframe_path}")
    print(f"     (Wireframe - cleaner for overlay in Blender)")
    print(f"\nColors: Blue (early) -> Cyan -> Green -> Yellow -> Red (late)")


if __name__ == "__main__":
    main()
