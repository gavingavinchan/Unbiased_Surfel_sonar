#!/usr/bin/env python3
"""
Debug script: Compare mesh before and after sonar training with SINGLE frame.

NEW: Uses sonar backward projection to initialize Gaussians (not COLMAP points).

Outputs:
- sonar_init_points.ply: Initial point cloud from sonar backward projection
- mesh_before_training.ply: Mesh from sonar-initialized Gaussians (no training)
- mesh_after_1iter.ply: After 1 iteration of sonar forward projection + training
- pose_pyramid_wireframe.ply: Single pose pyramid for the frame being used
- gt_sonar_frame.png / rendered_sonar_frame.png: Visual comparison

This helps visualize the effect of sonar training on the mesh.
"""

import os
import sys
import torch
import random
import numpy as np
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argparse import Namespace
from scene import Scene, GaussianModel
from scene.dataset_readers import readColmapCameras, readColmapSceneInfo, getNerfppNorm
from gaussian_renderer import render_sonar, render
from utils.sonar_utils import (SonarConfig, SonarScaleFactor, SonarExtrinsic,
                                sonar_frame_to_points, sonar_frames_to_point_cloud)
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import l1_loss, ssim
from utils.mesh_utils import GaussianExtractor
import open3d as o3d
from PIL import Image


def create_pose_pyramid_wireframe(position, rotation_matrix, depth=0.5, 
                                   azimuth_fov=120.0, elevation_fov=20.0):
    """Create a wireframe pyramid for a single pose."""
    half_az = math.radians(azimuth_fov / 2)
    half_el = math.radians(elevation_fov / 2)
    width = 2 * depth * math.tan(half_az)
    height = 2 * depth * math.tan(half_el)
    
    vertices_local = np.array([
        [0, 0, 0],
        [depth, -width/2, -height/2],
        [depth,  width/2, -height/2],
        [depth,  width/2,  height/2],
        [depth, -width/2,  height/2],
    ])
    
    cam_z_world = rotation_matrix[:, 2]
    cam_x_world = rotation_matrix[:, 0]
    cam_y_world = rotation_matrix[:, 1]
    R_local_to_world = np.column_stack([cam_z_world, cam_x_world, cam_y_world])
    vertices_world = (R_local_to_world @ vertices_local.T).T + position
    
    edges = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    
    wireframe = o3d.geometry.LineSet()
    wireframe.points = o3d.utility.Vector3dVector(vertices_world)
    wireframe.lines = o3d.utility.Vector2iVector(np.array(edges))
    wireframe.paint_uniform_color([1.0, 0.0, 0.0])
    return wireframe


# Fix random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configuration
DATASET_PATH = "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs"
OUTPUT_DIR_BASE = "./output/debug_sonar_init"
TARGET_FRAME = "sonar_1765233595134"
PYRAMID_DEPTH = 0.5
USE_SONAR_INIT = True  # Use sonar backward projection for init (vs COLMAP)
ELEV_INIT_MODE = os.environ.get("ELEV_INIT_MODE", "zero").strip().lower()
if ELEV_INIT_MODE not in {"zero", "random"}:
    raise ValueError(f"Invalid ELEV_INIT_MODE='{ELEV_INIT_MODE}'. Expected one of: random, zero")

# Create unique output folder (increment version number)
def get_next_output_dir(base_path):
    """Find next available output directory with incrementing version."""
    version = 1
    while True:
        output_dir = f"{base_path}_v{version}"
        if not os.path.exists(output_dir):
            return output_dir
        version += 1

OUTPUT_DIR = get_next_output_dir(OUTPUT_DIR_BASE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DEBUG: Single Frame - SONAR-INITIALIZED Gaussians")
print(f"Seed: {SEED}")
print(f"Target frame: {TARGET_FRAME}")
print(f"Init mode: {'SONAR backward projection' if USE_SONAR_INIT else 'COLMAP points'}")
print(f"Elevation init mode: {ELEV_INIT_MODE}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

# Sonar config
sonar_config = SonarConfig(
    image_height=100,  # Will be updated from actual image
    image_width=128,
    azimuth_fov=120.0,
    elevation_fov=20.0,
    range_min=0.1,
    range_max=30.0,
    intensity_threshold=0.01,
    device="cuda"
)

# Dataset arguments
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
    sonar_azimuth_fov=120.0,
    sonar_elevation_fov=20.0,
    sonar_range_min=0.1,
    sonar_range_max=30.0,
    sonar_intensity_threshold=0.01,
    gamma=2.2,
)

# Pipeline args for mesh extraction
pipe_args = Namespace(
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False,
)

# Load scene (to get cameras and poses)
print("\nLoading scene...")
gaussians_dummy = GaussianModel(dataset_args.sh_degree)
scene = Scene(dataset_args, gaussians_dummy, shuffle=False)

train_cameras = scene.getTrainCameras()
if len(train_cameras) == 0:
    print("ERROR: No training cameras loaded!")
    sys.exit(1)

# Find target frame
viewpoint = None
for cam in train_cameras:
    if cam.image_name == TARGET_FRAME:
        viewpoint = cam
        break

if viewpoint is None:
    print(f"ERROR: Frame {TARGET_FRAME} not found!")
    sys.exit(1)

print(f"Using frame: {viewpoint.image_name}")

# Update sonar config with actual image size
sonar_config = SonarConfig(
    image_height=viewpoint.image_height,
    image_width=viewpoint.image_width,
    azimuth_fov=120.0,
    elevation_fov=20.0,
    range_min=0.1,
    range_max=30.0,
    intensity_threshold=0.01,
    device="cuda"
)

print(f"\nSonar config:")
print(f"  Image size: {sonar_config.image_width}x{sonar_config.image_height}")
print(f"  Azimuth FOV: {sonar_config.azimuth_fov}°")
print(f"  Range: {sonar_config.range_min}m - {sonar_config.range_max}m")

# ============================================================
# POSE PYRAMID
# ============================================================
print("\n" + "=" * 60)
print("POSE PYRAMID: Generating wireframe for single pose")
print("=" * 60)

R_w2c = viewpoint.R
T_w2c = viewpoint.T
R_c2w = R_w2c.T
position = -R_c2w @ T_w2c
print(f"Camera position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")

pyramid = create_pose_pyramid_wireframe(position, R_c2w, depth=PYRAMID_DEPTH)
pyramid_path = os.path.join(OUTPUT_DIR, "pose_pyramid_wireframe.ply")
o3d.io.write_line_set(pyramid_path, pyramid)
print(f"Saved: {pyramid_path}")

# ============================================================
# SONAR-BASED POINT CLOUD INITIALIZATION
# ============================================================
print("\n" + "=" * 60)
print("POINT CLOUD: Generating from sonar backward projection")
print("=" * 60)

if USE_SONAR_INIT:
    # Generate points from SINGLE frame
    # Use low threshold since sonar images are often dim
    frame_result = sonar_frame_to_points(
        viewpoint, sonar_config,
        intensity_threshold=0.01,  # Lowered from 0.05
        mask_top_rows=10,
        elevation_mode=ELEV_INIT_MODE,
        return_debug=False,
    )
    points, colors = frame_result[0], frame_result[1]
    
    print(f"Generated {len(points)} points from sonar frame")
    
    # Compute normals (pointing toward camera for now)
    normals = np.zeros_like(points)
    for i in range(len(points)):
        dir_to_cam = position - points[i]
        norm = np.linalg.norm(dir_to_cam)
        if norm > 1e-6:
            normals[i] = dir_to_cam / norm
    
    # Save point cloud for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    init_points_path = os.path.join(OUTPUT_DIR, "sonar_init_points.ply")
    o3d.io.write_point_cloud(init_points_path, pcd)
    print(f"Saved: {init_points_path}")
    
    # Create BasicPointCloud for Gaussian initialization
    basic_pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    
    # Compute spatial extent (for learning rate scaling)
    cameras_extent = getNerfppNorm(train_cameras)["radius"]
    print(f"Cameras extent (radius): {cameras_extent:.3f}")
    
    # Initialize fresh GaussianModel from sonar points
    gaussians = GaussianModel(dataset_args.sh_degree)
    gaussians.create_from_pcd(basic_pcd, cameras_extent)
    
else:
    # Use COLMAP points (existing behavior)
    gaussians = gaussians_dummy
    print(f"Using COLMAP points: {len(gaussians.get_xyz)} Gaussians")

print(f"Gaussian count: {len(gaussians.get_xyz)}")

# ============================================================
# MESH 1: Before training
# ============================================================
print("\n" + "=" * 60)
print("MESH 1: Before training (sonar-initialized)")
print("=" * 60)

bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

# Use multiple cameras for mesh reconstruction
NUM_CAMERAS_FOR_MESH = 50
mesh_cameras = train_cameras[:NUM_CAMERAS_FOR_MESH]
gaussExtractor = GaussianExtractor(gaussians, render, pipe_args, bg_color=bg_color)

print(f"Reconstructing from {NUM_CAMERAS_FOR_MESH} cameras...")
gaussExtractor.reconstruction(mesh_cameras)

print("Extracting mesh (before training)...")
depth_trunc = gaussExtractor.radius * 2.0
voxel_size = depth_trunc / 128
sdf_trunc = 5.0 * voxel_size

mesh_before = gaussExtractor.extract_mesh_bounded(
    voxel_size=voxel_size,
    sdf_trunc=sdf_trunc,
    depth_trunc=depth_trunc
)

mesh_before_path = os.path.join(OUTPUT_DIR, "mesh_before_training.ply")
o3d.io.write_triangle_mesh(mesh_before_path, mesh_before)
print(f"Saved: {mesh_before_path}")
print(f"  Vertices: {len(mesh_before.vertices)}")
print(f"  Triangles: {len(mesh_before.triangles)}")

# ============================================================
# TRAINING: 100 iterations
# ============================================================
NUM_ITERATIONS = 100
print("\n" + "=" * 60)
print(f"TRAINING: {NUM_ITERATIONS} iterations (forward projection + loss)")
print("=" * 60)

# Setup optimizer
gaussians.training_setup(Namespace(
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30000,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    percent_dense=0.01,
    lambda_dssim=0.2,
    densification_interval=100,
    opacity_reset_interval=3000,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_grad_threshold=0.0002,
))

sonar_scale_factor = SonarScaleFactor(init_value=1.0).cuda()
sonar_extrinsic = SonarExtrinsic(device="cuda")

optimizer = torch.optim.Adam([
    {'params': [sonar_scale_factor._log_scale], 'lr': 0.01, 'name': 'sonar_scale'}
])

# Training loop over multiple cameras
print(f"Training on {len(train_cameras)} cameras for {NUM_ITERATIONS} iterations...")

for iteration in range(1, NUM_ITERATIONS + 1):
    # Use the same viewpoint we initialized from (points are only visible from this view)
    viewpoint_cam = viewpoint

    # Get ground truth
    gt_image = viewpoint_cam.original_image.cuda()
    gt_image = gt_image.clone()
    gt_image[:, :10, :] = 0  # Mask top 10 rows

    # Forward projection
    # TEMPORARY: scale_factor=None to eliminate scale effects for debugging
    render_pkg = render_sonar(
        viewpoint_cam, gaussians, background,
        sonar_config=sonar_config,
        scale_factor=None,  # TEMPORARY: disabled for debugging
        sonar_extrinsic=None  # Skip camera-to-sonar transform (poses are already sonar poses)
    )

    rendered = render_pkg["render"]

    # Compute loss
    Ll1 = l1_loss(rendered, gt_image)
    ssim_val = ssim(rendered, gt_image)
    loss = 0.8 * Ll1 + 0.2 * (1 - ssim_val)

    # Backward pass
    loss.backward()

    # Update gaussians
    with torch.no_grad():
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update learning rate
        gaussians.update_learning_rate(iteration)

    # Print progress every 10 iterations
    if iteration % 10 == 0 or iteration == 1:
        print(f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, loss={loss.item():.6f}, scale={sonar_scale_factor.get_scale_value():.4f}")

print(f"\nFinal scale factor: {sonar_scale_factor.get_scale_value():.6f}")

# Save sonar images from target viewpoint
print("\nSaving sonar images from target frame...")
gt_image = viewpoint.original_image.cuda().clone()
gt_image[:, :10, :] = 0

with torch.no_grad():
    # TEMPORARY: scale_factor=None to eliminate scale effects for debugging
    render_pkg = render_sonar(
        viewpoint, gaussians, background,
        sonar_config=sonar_config,
        scale_factor=None,  # TEMPORARY: disabled for debugging
        sonar_extrinsic=None  # Skip camera-to-sonar transform (poses are already sonar poses)
    )
    rendered = render_pkg["render"]

gt_np = gt_image[0].cpu().numpy()
gt_np = (gt_np * 255).astype(np.uint8)
Image.fromarray(gt_np, mode='L').save(os.path.join(OUTPUT_DIR, "gt_sonar_frame.png"))

rendered_np = rendered[0].detach().cpu().numpy()
rendered_np = (np.clip(rendered_np, 0, 1) * 255).astype(np.uint8)
Image.fromarray(rendered_np, mode='L').save(os.path.join(OUTPUT_DIR, "rendered_sonar_frame.png"))

comparison = np.hstack([gt_np, rendered_np])
Image.fromarray(comparison, mode='L').save(os.path.join(OUTPUT_DIR, "comparison_gt_vs_rendered.png"))

# Save brightened comparison (scale intensity for better visibility)
def brighten_image(img_np, percentile=99, min_brightness=0.3):
    """Brighten image by normalizing to percentile and applying gamma."""
    img_float = img_np.astype(np.float32)
    # Normalize to percentile (avoid outliers)
    p_val = np.percentile(img_float[img_float > 0], percentile) if np.any(img_float > 0) else 1.0
    p_val = max(p_val, 1.0)  # Avoid division by zero
    img_norm = np.clip(img_float / p_val, 0, 1)
    # Apply gamma correction to brighten dark areas
    gamma = 0.5  # < 1 brightens
    img_bright = np.power(img_norm, gamma)
    # Ensure minimum brightness for non-zero pixels
    img_bright = np.clip(img_bright * 255, 0, 255).astype(np.uint8)
    return img_bright

gt_bright = brighten_image(gt_np)
rendered_bright = brighten_image(rendered_np)
comparison_bright = np.hstack([gt_bright, rendered_bright])
Image.fromarray(comparison_bright, mode='L').save(os.path.join(OUTPUT_DIR, "comparison_gt_vs_rendered_bright.png"))

print(f"  Saved: gt_sonar_frame.png, rendered_sonar_frame.png, comparison_gt_vs_rendered.png, comparison_gt_vs_rendered_bright.png")

# ============================================================
# MESH 2: After training
# ============================================================
print("\n" + "=" * 60)
print(f"MESH 2: After {NUM_ITERATIONS} iterations")
print("=" * 60)

print(f"Reconstructing from {NUM_CAMERAS_FOR_MESH} cameras (after {NUM_ITERATIONS} iter)...")
gaussExtractor2 = GaussianExtractor(gaussians, render, pipe_args, bg_color=bg_color)
gaussExtractor2.reconstruction(mesh_cameras)

print("Extracting mesh (after training)...")
mesh_after = gaussExtractor2.extract_mesh_bounded(
    voxel_size=voxel_size,
    sdf_trunc=sdf_trunc,
    depth_trunc=depth_trunc
)

mesh_after_path = os.path.join(OUTPUT_DIR, f"mesh_after_{NUM_ITERATIONS}iter.ply")
o3d.io.write_triangle_mesh(mesh_after_path, mesh_after)
print(f"Saved: {mesh_after_path}")
print(f"  Vertices: {len(mesh_after.vertices)}")
print(f"  Triangles: {len(mesh_after.triangles)}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nGenerated files:")
print(f"  - sonar_init_points.ply      (Initial points from sonar backward projection)")
print(f"  - pose_pyramid_wireframe.ply (Single pose pyramid)")
print(f"  - mesh_before_training.ply   (Mesh before training)")
print(f"  - mesh_after_{NUM_ITERATIONS}iter.ply        (Mesh after {NUM_ITERATIONS} iterations)")
print(f"  - gt_sonar_frame.png         (Ground truth sonar)")
print(f"  - rendered_sonar_frame.png   (Forward projected)")
print(f"  - comparison_gt_vs_rendered.png        (Side-by-side)")
print(f"  - comparison_gt_vs_rendered_bright.png (Side-by-side, brightened)")
