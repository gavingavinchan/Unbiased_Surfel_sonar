#!/usr/bin/env python3
"""
Debug script: Compare mesh before and after sonar training.

Outputs:
- mesh_before_training.ply: Initial state (COLMAP points, backward projection only)
- mesh_after_1iter.ply: After 1 iteration of sonar forward projection + training

This helps visualize the effect of sonar training on the mesh.
"""

import os
import sys
import torch
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argparse import Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render_sonar, render
from utils.sonar_utils import SonarConfig, SonarScaleFactor, SonarExtrinsic
from utils.loss_utils import l1_loss, ssim
from utils.mesh_utils import GaussianExtractor
import open3d as o3d

# Fix random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configuration
DATASET_PATH = "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs"
OUTPUT_DIR = "./output/debug_before_after"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DEBUG: Before/After Mesh Comparison")
print(f"Seed: {SEED}")
print(f"Output: {OUTPUT_DIR}")
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
    # Sonar mode
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

# Initialize Gaussians
print("\nLoading scene...")
gaussians = GaussianModel(dataset_args.sh_degree)
scene = Scene(dataset_args, gaussians, shuffle=False)

# Get one training camera
train_cameras = scene.getTrainCameras()
if len(train_cameras) == 0:
    print("ERROR: No training cameras loaded!")
    sys.exit(1)

# Select specific frame (closest available to requested sonar_1765233592638)
TARGET_FRAME = "sonar_1765233595134"
viewpoint = None
for cam in train_cameras:
    if cam.image_name == TARGET_FRAME:
        viewpoint = cam
        break

if viewpoint is None:
    print(f"ERROR: Frame {TARGET_FRAME} not found in training cameras!")
    print(f"Available frames: {[c.image_name for c in train_cameras[:10]]}...")
    sys.exit(1)

print(f"Using frame: {viewpoint.image_name}")
print(f"Gaussian count: {len(gaussians.get_xyz)}")

# Setup sonar config
image_height = viewpoint.image_height
image_width = viewpoint.image_width

sonar_config = SonarConfig(
    image_height=image_height,
    image_width=image_width,
    azimuth_fov=dataset_args.sonar_azimuth_fov,
    elevation_fov=dataset_args.sonar_elevation_fov,
    range_min=dataset_args.sonar_range_min,
    range_max=dataset_args.sonar_range_max,
)

sonar_scale_factor = SonarScaleFactor(init_value=1.0).cuda()
sonar_extrinsic = SonarExtrinsic(device="cuda")

print(f"\nSonar config:")
print(f"  Image size: {image_width}x{image_height}")
print(f"  Azimuth FOV: {sonar_config.azimuth_fov}°")
print(f"  Range: {sonar_config.range_min}m - {sonar_config.range_max}m")

# ============================================================
# MESH 1: Before training (backward projection only)
# ============================================================
print("\n" + "=" * 60)
print("MESH 1: Before training (initial COLMAP state)")
print("=" * 60)

# Create extractor for mesh generation
bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

# We need to use the standard render function for mesh extraction
# since GaussianExtractor expects the standard render signature
gaussExtractor = GaussianExtractor(gaussians, render, pipe_args, bg_color=bg_color)

# Reconstruct from all cameras (needed for TSDF fusion)
print("Reconstructing from training cameras...")
gaussExtractor.reconstruction(train_cameras[:50])  # Use subset for speed

# Extract mesh
print("Extracting mesh (before training)...")
depth_trunc = gaussExtractor.radius * 2.0
voxel_size = depth_trunc / 128  # mesh_res = 128
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
# MESH 2: After 1 iteration of sonar training
# ============================================================
print("\n" + "=" * 60)
print("MESH 2: After 1 iteration (forward projection + training)")
print("=" * 60)

# Setup optimizer (same as train.py)
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

# Add scale factor to optimizer
optimizer = torch.optim.Adam([
    {'params': [sonar_scale_factor._log_scale], 'lr': 0.01, 'name': 'sonar_scale'}
])

# Get ground truth
gt_image = viewpoint.original_image.cuda().clone()
# Apply mask (top 10 rows)
gt_image[:, :10, :] = 0

print(f"Running 1 training iteration...")

# Forward projection (sonar rendering)
render_pkg = render_sonar(
    viewpoint, gaussians, background,
    sonar_config=sonar_config,
    scale_factor=sonar_scale_factor,
    sonar_extrinsic=sonar_extrinsic
)

rendered = render_pkg["render"]

# Compute loss
Ll1 = l1_loss(rendered, gt_image)
ssim_val = ssim(rendered, gt_image)
loss = 0.8 * Ll1 + 0.2 * (1 - ssim_val)

print(f"  L1 loss: {Ll1.item():.6f}")
print(f"  SSIM: {ssim_val.item():.6f}")
print(f"  Total loss: {loss.item():.6f}")

# Backward pass
loss.backward()

# Update Gaussians and scale factor
gaussians.optimizer.step()
gaussians.optimizer.zero_grad(set_to_none=True)
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print(f"  Scale factor after update: {sonar_scale_factor.get_scale_value():.6f}")

# Extract mesh after training
print("\nReconstructing from training cameras (after 1 iter)...")
gaussExtractor2 = GaussianExtractor(gaussians, render, pipe_args, bg_color=bg_color)
gaussExtractor2.reconstruction(train_cameras[:50])

print("Extracting mesh (after training)...")
mesh_after = gaussExtractor2.extract_mesh_bounded(
    voxel_size=voxel_size,
    sdf_trunc=sdf_trunc,
    depth_trunc=depth_trunc
)

mesh_after_path = os.path.join(OUTPUT_DIR, "mesh_after_1iter.ply")
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
print(f"\nGenerated meshes:")
print(f"  1. {mesh_before_path}")
print(f"     (Initial state - COLMAP backward projection only)")
print(f"  2. {mesh_after_path}")
print(f"     (After 1 iteration - sonar forward projection + training)")
