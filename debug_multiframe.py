#!/usr/bin/env python3
"""
Debug script: Multi-frame sonar training with curriculum learning for scale factor.

Uses 5 sonar frames with curriculum learning:
- Stage 1: Fix surfels, learn scale factor only
- Stage 2: Fix scale factor, learn surfels only
- Stage 3: (Optional) Joint fine-tuning

This addresses the scale-surfel coupling problem where both can compensate for
each other in single-frame training. Multi-frame provides geometric constraints.

Outputs:
- sonar_init_points.ply: Initial point cloud from sonar backward projection (all frames)
- pose_pyramids_wireframe.ply: Wireframe pyramids for all training frames
- mesh_before_training.ply: Mesh from sonar-initialized Gaussians (no training)
- mesh_after_training.ply: Mesh after curriculum training
- comparison_frame_N.png: GT vs rendered for each training frame
"""

import os
import sys
import torch
import random
import numpy as np
import math
import matplotlib.pyplot as plt

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


def is_in_sonar_fov(xyz, camera, sonar_config, scale_factor, return_details=False):
    """
    Check if 3D points are within the sonar FOV of a given camera.

    Uses the EXACT same transform as render_sonar to ensure consistency.

    Args:
        xyz: [N, 3] tensor of 3D points in world coordinates
        camera: Camera object with world_view_transform
        sonar_config: SonarConfig with FOV and range parameters
        scale_factor: SonarScaleFactor for pose scaling
        return_details: If True, return dict with per-constraint masks

    Returns:
        [N] boolean tensor: True if point is within FOV
        (or dict if return_details=True)
    """
    N = xyz.shape[0]
    if N == 0:
        empty = torch.zeros(0, dtype=torch.bool, device=xyz.device)
        if return_details:
            return {"in_fov": empty, "in_azimuth": empty, "in_elevation": empty,
                    "in_range": empty, "in_front": empty}
        return empty

    # Match render_sonar's transform EXACTLY
    w2v = camera.world_view_transform.cuda()  # [4, 4]

    # Extract R and t (translation is in row 3, not column 3!)
    R_w2v = w2v[:3, :3]
    t_w2v = w2v[3, :3]

    # Apply scale factor to translation
    scale = scale_factor.scale
    t_w2v_scaled = scale * t_w2v

    # Transform points to sonar frame: p_sonar = p_world @ R.T + t_scaled
    # This matches render_sonar exactly
    points_sonar = (xyz @ R_w2v.T) + t_w2v_scaled  # [N, 3]

    # Camera/sonar frame: +X = right, +Y = down, +Z = forward
    right = points_sonar[:, 0]
    down = points_sonar[:, 1]
    forward = points_sonar[:, 2]

    # Compute azimuth (matches render_sonar: -atan2(right, forward))
    # We use abs() so sign doesn't matter for FOV check
    azimuth = torch.atan2(right, forward)  # [N]

    # Compute range (3D distance from sonar origin)
    range_vals = torch.sqrt(right**2 + down**2 + forward**2)  # [N]

    # Compute elevation (matches render_sonar: atan2(down, horiz_dist))
    horiz_dist = torch.sqrt(right**2 + forward**2)
    elevation = torch.atan2(down, horiz_dist)  # [N]

    # Check FOV constraints
    half_az_rad = math.radians(sonar_config.azimuth_fov / 2)
    half_el_rad = math.radians(sonar_config.elevation_fov / 2)

    in_azimuth = torch.abs(azimuth) <= half_az_rad
    in_elevation = torch.abs(elevation) <= half_el_rad
    in_range = (range_vals >= sonar_config.range_min) & (range_vals <= sonar_config.range_max)
    in_front = forward > 0  # Must be in front of sonar

    in_fov = in_azimuth & in_elevation & in_range & in_front

    if return_details:
        return {
            "in_fov": in_fov,
            "in_azimuth": in_azimuth,
            "in_elevation": in_elevation,
            "in_range": in_range,
            "in_front": in_front,
            "range_vals": range_vals,
            "azimuth_deg": torch.rad2deg(azimuth),
            "elevation_deg": torch.rad2deg(elevation),
        }
    return in_fov


def compute_fov_margin_debug(range_vals, azimuth, elevation, sonar_config):
    """
    Compute distance from each point to nearest FOV boundary.

    Args:
        range_vals: [N] distance from sonar origin
        azimuth: [N] horizontal angle (radians)
        elevation: [N] vertical angle (radians)
        sonar_config: SonarConfig with FOV limits

    Returns:
        [N] margin in world units (meters)
    """
    half_az_rad = math.radians(sonar_config.azimuth_fov / 2)
    half_el_rad = math.radians(sonar_config.elevation_fov / 2)

    # Angular margins (convert to linear distance at current range)
    az_margin = (half_az_rad - torch.abs(azimuth)) * range_vals
    el_margin = (half_el_rad - torch.abs(elevation)) * range_vals

    # Range margins
    range_margin_near = range_vals - sonar_config.range_min
    range_margin_far = sonar_config.range_max - range_vals

    # Minimum margin across all constraints
    margin = torch.min(torch.stack([
        az_margin, el_margin, range_margin_near, range_margin_far
    ], dim=0), dim=0).values

    return margin


def is_fully_in_sonar_fov(xyz, scaling, camera, sonar_config, scale_factor):
    """
    Check if surfels (center + size extent) are fully within the sonar FOV.

    A surfel is fully inside FOV if:
    1. Its center is within FOV (azimuth, elevation, range constraints)
    2. Its margin to FOV boundary exceeds its radius

    Args:
        xyz: [N, 3] tensor of surfel center positions
        scaling: [N, 2] tensor of surfel scaling (already activated, not log)
        camera: Camera object with world_view_transform
        sonar_config: SonarConfig with FOV and range parameters
        scale_factor: SonarScaleFactor for pose scaling

    Returns:
        [N] boolean tensor: True if surfel is fully within FOV
    """
    N = xyz.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=xyz.device)

    # Transform points to sonar frame (same as is_in_sonar_fov)
    w2v = camera.world_view_transform.cuda()
    R_w2v = w2v[:3, :3]
    t_w2v = w2v[3, :3]
    t_w2v_scaled = scale_factor.scale * t_w2v
    points_sonar = (xyz @ R_w2v.T) + t_w2v_scaled

    right = points_sonar[:, 0]
    down = points_sonar[:, 1]
    forward = points_sonar[:, 2]

    azimuth = torch.atan2(right, forward)
    range_vals = torch.sqrt(right**2 + down**2 + forward**2)
    horiz_dist = torch.sqrt(right**2 + forward**2)
    elevation = torch.atan2(down, horiz_dist)

    # Center-based FOV check
    half_az_rad = math.radians(sonar_config.azimuth_fov / 2)
    half_el_rad = math.radians(sonar_config.elevation_fov / 2)

    in_azimuth = torch.abs(azimuth) <= half_az_rad
    in_elevation = torch.abs(elevation) <= half_el_rad
    in_range = (range_vals >= sonar_config.range_min) & (range_vals <= sonar_config.range_max)
    in_front = forward > 0
    center_in_fov = in_azimuth & in_elevation & in_range & in_front

    # Size-aware check: margin must exceed surfel radius
    surfel_radius = scaling.max(dim=1).values  # [N]
    margin = compute_fov_margin_debug(range_vals, azimuth, elevation, sonar_config)

    fully_inside = center_in_fov & (margin > surfel_radius)
    return fully_inside


def prune_outside_fov(gaussians, training_frames, sonar_config, scale_factor,
                      require_all=False, check_size=True):
    """
    Prune Gaussians that are outside the FOV of training cameras.

    Args:
        gaussians: GaussianModel instance
        training_frames: List of camera objects
        sonar_config: SonarConfig
        scale_factor: SonarScaleFactor
        require_all: If True, prune if outside ALL cameras' FOV
                     If False, keep if visible from ANY camera (default)
        check_size: If True, also check that surfel size doesn't extend beyond FOV

    Returns:
        Number of points pruned
    """
    xyz = gaussians.get_xyz  # [N, 3]
    N = xyz.shape[0]

    if N == 0:
        return 0

    # Check visibility from each training frame
    visible_masks = []
    if check_size:
        # Size-aware check: surfel center AND extent must be within FOV
        scaling = gaussians.get_scaling  # [N, 2]
        for cam in training_frames:
            in_fov = is_fully_in_sonar_fov(xyz, scaling, cam, sonar_config, scale_factor)
            visible_masks.append(in_fov)
    else:
        # Center-only check (original behavior)
        for cam in training_frames:
            in_fov = is_in_sonar_fov(xyz, cam, sonar_config, scale_factor)
            visible_masks.append(in_fov)

    # Stack masks: [num_cameras, N]
    all_masks = torch.stack(visible_masks, dim=0)

    if require_all:
        # Prune if outside ALL cameras (very aggressive)
        visible_from_any = all_masks.any(dim=0)  # [N]
        prune_mask = ~visible_from_any
    else:
        # Keep if visible from at least one camera (conservative)
        visible_from_any = all_masks.any(dim=0)  # [N]
        prune_mask = ~visible_from_any

    num_to_prune = prune_mask.sum().item()

    if num_to_prune > 0:
        gaussians.prune_points(prune_mask)

    return num_to_prune


def create_pose_pyramid_wireframe(position, rotation_matrix, depth=0.5,
                                   azimuth_fov=120.0, elevation_fov=20.0, color=[1.0, 0.0, 0.0]):
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
    wireframe.paint_uniform_color(color)
    return wireframe


def brighten_image(img_np, percentile=99, gamma=0.5):
    """Brighten image by normalizing to percentile and applying gamma."""
    img_float = img_np.astype(np.float32)
    p_val = np.percentile(img_float[img_float > 0], percentile) if np.any(img_float > 0) else 1.0
    p_val = max(p_val, 1.0)
    img_norm = np.clip(img_float / p_val, 0, 1)
    img_bright = np.power(img_norm, gamma)
    img_bright = np.clip(img_bright * 255, 0, 255).astype(np.uint8)
    return img_bright


# Intensity threshold: pixels below this value (0-255 scale) are treated as black
INTENSITY_THRESHOLD = 10  # out of 255


def preprocess_gt_image(image_tensor, mask_top_rows=10, intensity_threshold=INTENSITY_THRESHOLD):
    """
    Preprocess ground truth sonar image for training/comparison.

    Args:
        image_tensor: [C, H, W] tensor in 0-1 range
        mask_top_rows: Number of top rows to mask (close range artifacts)
        intensity_threshold: Pixel values below this (0-255 scale) are set to 0

    Returns:
        Preprocessed image tensor
    """
    gt = image_tensor.cuda().clone()

    # Mask top rows (close range artifacts)
    if mask_top_rows > 0:
        gt[:, :mask_top_rows, :] = 0

    # Threshold low intensity pixels (noise filtering)
    # Convert threshold from 0-255 to 0-1 range
    threshold_normalized = intensity_threshold / 255.0
    gt[gt < threshold_normalized] = 0

    return gt


def extract_and_save_mesh(gaussians, mesh_cameras, pipe_args, bg_color,
                          output_dir, filename, depth_trunc=None, voxel_size=None, sdf_trunc=None):
    """Helper to extract and save mesh at a checkpoint."""
    print(f"  Extracting mesh: {filename}")
    extractor = GaussianExtractor(gaussians, render, pipe_args, bg_color=bg_color)
    extractor.reconstruction(mesh_cameras)

    if depth_trunc is None:
        depth_trunc = extractor.radius * 2.0
    if voxel_size is None:
        voxel_size = depth_trunc / 128
    if sdf_trunc is None:
        sdf_trunc = 5.0 * voxel_size

    mesh = extractor.extract_mesh_bounded(
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc
    )

    mesh_path = os.path.join(output_dir, filename)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"  Saved: {filename} (V={len(mesh.vertices)}, T={len(mesh.triangles)})")
    return mesh, depth_trunc, voxel_size, sdf_trunc


def select_diverse_frames(cameras, num_frames, seed=42):
    """
    Select frames with diverse viewpoints for better geometric constraints.
    Uses simple strategy: evenly spaced indices from sorted cameras.
    """
    random.seed(seed)
    n = len(cameras)
    if num_frames >= n:
        return list(range(n))

    # Evenly spaced selection
    step = n // num_frames
    indices = [i * step for i in range(num_frames)]
    return indices


def save_comparison_images(training_frames, gaussians, background, sonar_config,
                           scale_factor, output_dir, stage_name):
    """Save GT vs rendered comparison images for all training frames at a given stage."""
    print(f"\n  Saving comparison images for {stage_name}...")
    for i, cam in enumerate(training_frames):
        gt_image = preprocess_gt_image(cam.original_image)

        with torch.no_grad():
            render_pkg = render_sonar(
                cam, gaussians, background,
                sonar_config=sonar_config,
                scale_factor=scale_factor,
                sonar_extrinsic=None
            )
            rendered = render_pkg["render"]

        gt_np = (gt_image[0].cpu().numpy() * 255).astype(np.uint8)
        rendered_np = (np.clip(rendered[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8)

        # Brightened comparison
        comparison_bright = np.hstack([brighten_image(gt_np), brighten_image(rendered_np)])
        filename = f"comparison_{stage_name}_frame{i}.png"
        Image.fromarray(comparison_bright, mode='L').save(os.path.join(output_dir, filename))

    print(f"  Saved comparison images for {stage_name}")


def resolve_raw_sonar_path(cam, dataset_path, sonar_dir):
    base_path = os.path.join(dataset_path, sonar_dir, cam.image_name)
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidate = base_path + ext
        if os.path.exists(candidate):
            return candidate
    return None


def save_raw_comparison_images(training_frames, gaussians, background, sonar_config,
                               scale_factor, output_dir, stage_name, dataset_path, sonar_dir):
    """Save raw sonar vs rendered comparison images (no brighten, no masking)."""
    print(f"\n  Saving raw-frame comparisons for {stage_name}...")
    for i, cam in enumerate(training_frames):
        raw_path = resolve_raw_sonar_path(cam, dataset_path, sonar_dir)
        if raw_path is None:
            print(f"  WARNING: Raw image not found for {cam.image_name}")
            continue

        raw_image = Image.open(raw_path).convert("L")
        raw_np = np.array(raw_image)

        with torch.no_grad():
            render_pkg = render_sonar(
                cam, gaussians, background,
                sonar_config=sonar_config,
                scale_factor=scale_factor,
                sonar_extrinsic=None
            )
            rendered = render_pkg["render"]

        rendered_np = (np.clip(rendered[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        rendered_img = Image.fromarray(rendered_np, mode="L").resize(raw_image.size, Image.BILINEAR)
        rendered_resized = np.array(rendered_img)

        comparison_raw = np.hstack([raw_np, rendered_resized])
        filename = f"comparison_{stage_name}_raw_frame{i}.png"
        Image.fromarray(comparison_raw, mode="L").save(os.path.join(output_dir, filename))

    print(f"  Saved raw-frame comparisons for {stage_name}")


metric_step = 0
metric_iters = []
metric_scale = []
metric_loss = []
metric_stage = []


def record_metrics(loss_value, scale_value, stage_name):
    global metric_step
    metric_step += 1
    metric_iters.append(metric_step)
    metric_loss.append(loss_value)
    metric_scale.append(scale_value)
    metric_stage.append(stage_name)


def plot_training_metrics(output_dir, stage_boundaries):
    if not metric_iters:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(metric_iters, metric_scale, color="tab:blue", linewidth=1.5)
    axes[0].set_ylabel("Scale")
    axes[0].set_title("Scale Factor and Loss")

    axes[1].plot(metric_iters, metric_loss, color="tab:orange", linewidth=1.5)
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Iteration")

    for boundary, label in stage_boundaries:
        axes[0].axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
        axes[1].axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
        axes[0].text(boundary + 0.5, axes[0].get_ylim()[1], label, rotation=90,
                     va="top", ha="left", fontsize=8, color="gray")

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "scale_and_loss.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved training plot: {plot_path}")


# Fix random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =============================================================================
# Configuration
# =============================================================================
DATASET_PATH = "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs"
OUTPUT_DIR_BASE = "./output/debug_multiframe"
NUM_TRAINING_FRAMES = 1  # Number of frames to use for training (TODO: increase to 500 for full run)
PYRAMID_DEPTH = 0.5

# Curriculum learning parameters
STAGE1_ITERATIONS = 0   # Learn scale only (surfels frozen) - DISABLED, using known scale=0.65
STAGE2_ITERATIONS = 3000  # Learn surfels only (scale frozen)
STAGE3_ITERATIONS = 50   # Joint fine-tuning

# FOV-aware pruning: remove surfels that drift outside all training cameras' FOV
FOV_PRUNE_INTERVAL = 100  # Prune every N iterations (0 to disable)

# Create unique output folder
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
print("DEBUG: Multi-Frame Training with Curriculum Learning")
print("=" * 60)
print(f"Seed: {SEED}")
print(f"Num training frames: {NUM_TRAINING_FRAMES}")
print(f"Curriculum: Stage1={STAGE1_ITERATIONS} (scale), Stage2={STAGE2_ITERATIONS} (surfels), Stage3={STAGE3_ITERATIONS} (joint)")
print(f"FOV pruning interval: {FOV_PRUNE_INTERVAL} iterations")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

# Sonar config (will be updated with actual image size)
sonar_config = SonarConfig(
    image_height=100,
    image_width=128,
    azimuth_fov=120.0,
    elevation_fov=20.0,
    range_min=0.2,
    range_max=3.0,
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
    sonar_range_min=0.2,
    sonar_range_max=3.0,
    sonar_intensity_threshold=0.01,
    gamma=2.2,
)

# Pipeline args for mesh extraction
pipe_args = Namespace(
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False,
)

# =============================================================================
# Load Scene
# =============================================================================
print("\nLoading scene...")
gaussians_dummy = GaussianModel(dataset_args.sh_degree)
scene = Scene(dataset_args, gaussians_dummy, shuffle=False)

train_cameras = scene.getTrainCameras()
if len(train_cameras) == 0:
    print("ERROR: No training cameras loaded!")
    sys.exit(1)

print(f"Total cameras available: {len(train_cameras)}")

# Select diverse frames for training
frame_indices = select_diverse_frames(train_cameras, NUM_TRAINING_FRAMES, seed=SEED)
training_frames = [train_cameras[i] for i in frame_indices]

print(f"Selected {len(training_frames)} training frames:")
for i, cam in enumerate(training_frames):
    print(f"  [{i}] {cam.image_name}")

# Update sonar config with actual image size
sample_cam = training_frames[0]
sonar_config = SonarConfig(
    image_height=sample_cam.image_height,
    image_width=sample_cam.image_width,
    azimuth_fov=120.0,
    elevation_fov=20.0,
    range_min=0.2,
    range_max=3.0,
    intensity_threshold=0.01,
    device="cuda"
)

print(f"\nSonar config:")
print(f"  Image size: {sonar_config.image_width}x{sonar_config.image_height}")
print(f"  Azimuth FOV: {sonar_config.azimuth_fov}deg")
print(f"  Range: {sonar_config.range_min}m - {sonar_config.range_max}m")

# =============================================================================
# Generate Pose Pyramids for All Training Frames
# =============================================================================
print("\n" + "=" * 60)
print("POSE PYRAMIDS: Generating wireframes for training frames")
print("=" * 60)

combined_wireframe = o3d.geometry.LineSet()
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]  # Different colors for each frame

for i, cam in enumerate(training_frames):
    R_w2c = cam.R
    T_w2c = cam.T
    R_c2w = R_w2c.T
    position = -R_c2w @ T_w2c

    color = colors[i % len(colors)]
    pyramid = create_pose_pyramid_wireframe(position, R_c2w, depth=PYRAMID_DEPTH, color=color)
    combined_wireframe += pyramid
    print(f"  Frame {i}: pos=[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")

pyramid_path = os.path.join(OUTPUT_DIR, "pose_pyramids_wireframe.ply")
o3d.io.write_line_set(pyramid_path, combined_wireframe)
print(f"Saved: {pyramid_path}")

# =============================================================================
# Initialize Gaussians from Multi-Frame Backward Projection
# =============================================================================
print("\n" + "=" * 60)
print("POINT CLOUD: Generating from multi-frame backward projection")
print("=" * 60)

all_points = []
all_colors = []
all_normals = []

for i, cam in enumerate(training_frames):
    points, colors = sonar_frame_to_points(
        cam, sonar_config,
        intensity_threshold=INTENSITY_THRESHOLD / 255.0,  # Same threshold as training
        mask_top_rows=10
    )

    if len(points) == 0:
        print(f"  Frame {i}: 0 points (skipped)")
        continue

    # Compute normals pointing toward camera
    R_c2w = cam.R.T
    cam_pos = -R_c2w @ cam.T
    normals = np.zeros_like(points)
    for j in range(len(points)):
        dir_to_cam = cam_pos - points[j]
        norm = np.linalg.norm(dir_to_cam)
        if norm > 1e-6:
            normals[j] = dir_to_cam / norm

    all_points.append(points)
    all_colors.append(colors)
    all_normals.append(normals)
    print(f"  Frame {i}: {len(points)} points")

points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)
normals = np.concatenate(all_normals, axis=0)

print(f"Total points: {len(points)}")

# Diagnostic: Print range statistics of generated points
# Compute distance from each point to its source camera
print("\nDiagnostic: Point distance from source cameras")
point_idx = 0
for i, cam in enumerate(training_frames):
    n_pts = len(all_points[i]) if i < len(all_points) else 0
    if n_pts == 0:
        continue
    R_c2w = cam.R.T
    cam_pos = -R_c2w @ cam.T
    pts = all_points[i]
    distances = np.linalg.norm(pts - cam_pos, axis=1)
    print(f"  Frame {i}: min={distances.min():.2f}m, max={distances.max():.2f}m, mean={distances.mean():.2f}m")

# Save combined point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.normals = o3d.utility.Vector3dVector(normals)

init_points_path = os.path.join(OUTPUT_DIR, "sonar_init_points.ply")
o3d.io.write_point_cloud(init_points_path, pcd)
print(f"Saved: {init_points_path}")

# Create BasicPointCloud and initialize Gaussians
basic_pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
cameras_extent = getNerfppNorm(train_cameras)["radius"]
print(f"Cameras extent (radius): {cameras_extent:.3f}")

gaussians = GaussianModel(dataset_args.sh_degree)
gaussians.create_from_pcd(basic_pcd, cameras_extent)
print(f"Gaussian count: {len(gaussians.get_xyz)}")

# Diagnostic: Check initial FOV visibility with temporary scale factor
print("\nDiagnostic: Initial surfel FOV visibility")
temp_scale = SonarScaleFactor(init_value=0.65).cuda()  # Use calibrated scale
for i, cam in enumerate(training_frames):
    details = is_in_sonar_fov(gaussians.get_xyz, cam, sonar_config, temp_scale, return_details=True)
    in_fov = details["in_fov"]
    print(f"  Frame {i}: {in_fov.sum().item()}/{len(gaussians.get_xyz)} surfels in FOV")
    print(f"    - in_front: {details['in_front'].sum().item()}")
    print(f"    - in_azimuth: {details['in_azimuth'].sum().item()} (±{sonar_config.azimuth_fov/2:.0f}°)")
    print(f"    - in_elevation: {details['in_elevation'].sum().item()} (±{sonar_config.elevation_fov/2:.0f}°)")
    print(f"    - in_range: {details['in_range'].sum().item()} ({sonar_config.range_min:.1f}-{sonar_config.range_max:.1f}m)")
    # Show range distribution
    r = details["range_vals"]
    print(f"    - range stats: min={r.min().item():.2f}m, max={r.max().item():.2f}m, mean={r.mean().item():.2f}m")

# =============================================================================
# Mesh Before Training
# =============================================================================
print("\n" + "=" * 60)
print("MESH 1: Before training")
print("=" * 60)

bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

NUM_CAMERAS_FOR_MESH = 50
mesh_cameras = train_cameras[:NUM_CAMERAS_FOR_MESH]
gaussExtractor = GaussianExtractor(gaussians, render, pipe_args, bg_color=bg_color)

print(f"Reconstructing from {NUM_CAMERAS_FOR_MESH} cameras...")
gaussExtractor.reconstruction(mesh_cameras)

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
print(f"  Vertices: {len(mesh_before.vertices)}, Triangles: {len(mesh_before.triangles)}")

# Save comparison images before any training (using scale=1.0)
temp_scale = SonarScaleFactor(init_value=1.0).cuda()
save_comparison_images(training_frames, gaussians, background, sonar_config,
                       temp_scale, OUTPUT_DIR, "before_training")

# =============================================================================
# Setup Training
# =============================================================================
print("\n" + "=" * 60)
print("TRAINING SETUP")
print("=" * 60)

# Setup Gaussian optimizer
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

# Scale factor module
# Known scale factor from calibration cube in COLMAP (true value ~0.66)
# TODO: Fix scale factor learning - currently not converging to correct value
sonar_scale_factor = SonarScaleFactor(init_value=0.65).cuda()

# Separate optimizer for scale factor
scale_optimizer = torch.optim.Adam([
    {'params': [sonar_scale_factor._log_scale], 'lr': 0.01, 'name': 'sonar_scale'}
])

print(f"Initial scale factor: {sonar_scale_factor.get_scale_value():.6f}")

# =============================================================================
# Scale Sensitivity Test: Check if loss changes with scale perturbation
# =============================================================================
print("\n" + "=" * 60)
print("SCALE SENSITIVITY TEST")
print("=" * 60)

test_scales = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 2.0]
viewpoint_test = training_frames[0]
gt_test = preprocess_gt_image(viewpoint_test.original_image)

print("Testing loss at different scale values:")

# Debug: Check camera transform properties
w2c = viewpoint_test.world_view_transform
print(f"  Camera transform: device={w2c.device}, dtype={w2c.dtype}, requires_grad={w2c.requires_grad}")
print(f"  Full w2c matrix:\n{w2c.cpu().numpy()}")
t_w2v = w2c[:3, 3]
print(f"  t_w2v (col 3) = {t_w2v.cpu().numpy()}")

# Check other camera properties
print(f"  viewpoint.R:\n{viewpoint_test.R}")
print(f"  viewpoint.T: {viewpoint_test.T}")
cam_center = viewpoint_test.camera_center if hasattr(viewpoint_test, 'camera_center') else "N/A"
print(f"  viewpoint.camera_center: {cam_center}")

for test_scale in test_scales:
    test_sf = SonarScaleFactor(init_value=test_scale).cuda()
    t_scaled = test_sf.scale * t_w2v.cuda()
    print(f"  scale={test_scale:.1f}: t_scaled={t_scaled.detach().cpu().numpy()}")

    with torch.no_grad():
        render_pkg = render_sonar(
            viewpoint_test, gaussians, background,
            sonar_config=sonar_config,
            scale_factor=test_sf,
            sonar_extrinsic=None
        )
        rendered = render_pkg["render"]

    test_l1 = l1_loss(rendered, gt_test)
    test_ssim = ssim(rendered, gt_test)
    print(f"           L1={test_l1.item():.6f}, SSIM={test_ssim.item():.4f}")

print()

# =============================================================================
# Stage 1: Learn Scale Only (Surfels Frozen)
# =============================================================================
if STAGE1_ITERATIONS > 0:
    print("\n" + "=" * 60)
    print(f"STAGE 1: Learn scale factor only ({STAGE1_ITERATIONS} iterations)")
    print("=" * 60)

    for iteration in range(1, STAGE1_ITERATIONS + 1):
        # Cycle through training frames
        frame_idx = (iteration - 1) % len(training_frames)
        viewpoint_cam = training_frames[frame_idx]

        # Get ground truth (with intensity thresholding)
        gt_image = preprocess_gt_image(viewpoint_cam.original_image)

        # Forward projection WITH scale factor
        render_pkg = render_sonar(
            viewpoint_cam, gaussians, background,
            sonar_config=sonar_config,
            scale_factor=sonar_scale_factor,  # Scale factor enabled
            sonar_extrinsic=None
        )
        rendered = render_pkg["render"]

        # Debug: Check gradient flow on first iteration
        if iteration == 1:
            print(f"\n  DEBUG: Gradient flow check:")
            print(f"    scale._log_scale.requires_grad: {sonar_scale_factor._log_scale.requires_grad}")
            print(f"    scale.scale.requires_grad: {sonar_scale_factor.scale.requires_grad}")
            print(f"    rendered.requires_grad: {rendered.requires_grad}")
            print(f"    rendered.grad_fn: {rendered.grad_fn}")

        # Compute loss
        Ll1 = l1_loss(rendered, gt_image)
        ssim_val = ssim(rendered, gt_image)
        loss = 0.8 * Ll1 + 0.2 * (1 - ssim_val)

        if iteration == 1:
            print(f"    loss.requires_grad: {loss.requires_grad}")
            print(f"    loss.grad_fn: {loss.grad_fn}\n")

        # Backward - only scale factor gets gradients (surfels frozen)
        loss.backward()

        # Debug: Check scale factor gradient BEFORE optimizer step
        scale_grad = sonar_scale_factor._log_scale.grad
        grad_val = scale_grad.item() if scale_grad is not None else 0.0

        # Update scale factor only
        with torch.no_grad():
            scale_optimizer.step()
            scale_optimizer.zero_grad(set_to_none=True)
            # Zero out Gaussian gradients without stepping
            gaussians.optimizer.zero_grad(set_to_none=True)

        scale_value = sonar_scale_factor.get_scale_value()
        record_metrics(loss.item(), scale_value, "stage1")

        if iteration % 10 == 0 or iteration == 1:
            print(f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, scale={scale_value:.4f}, grad={grad_val:.6f}")

        # Extract mesh after iteration 1
        if iteration == 1:
            _, depth_trunc, voxel_size, sdf_trunc = extract_and_save_mesh(
                gaussians, mesh_cameras, pipe_args, bg_color,
                OUTPUT_DIR, "mesh_after_iter1.ply"
            )

    print(f"Stage 1 complete. Scale factor: {sonar_scale_factor.get_scale_value():.6f}")

    # Extract mesh after Stage 1
    extract_and_save_mesh(
        gaussians, mesh_cameras, pipe_args, bg_color,
        OUTPUT_DIR, "mesh_after_stage1.ply",
        depth_trunc=depth_trunc, voxel_size=voxel_size, sdf_trunc=sdf_trunc
    )

    # Save comparison images after Stage 1
    save_comparison_images(training_frames, gaussians, background, sonar_config,
                           sonar_scale_factor, OUTPUT_DIR, "after_stage1")

# =============================================================================
# Stage 2: Learn Surfels Only (Scale Frozen)
# =============================================================================
if STAGE2_ITERATIONS > 0:
    print("\n" + "=" * 60)
    print(f"STAGE 2: Learn surfels only ({STAGE2_ITERATIONS} iterations)")
    print("=" * 60)

    # Freeze scale factor
    sonar_scale_factor._log_scale.requires_grad = False
    frozen_scale = sonar_scale_factor.get_scale_value()
    print(f"Scale factor frozen at: {frozen_scale:.6f}")

    for iteration in range(1, STAGE2_ITERATIONS + 1):
        # Cycle through training frames
        frame_idx = (iteration - 1) % len(training_frames)
        viewpoint_cam = training_frames[frame_idx]

        # Get ground truth (with intensity thresholding)
        gt_image = preprocess_gt_image(viewpoint_cam.original_image)

        # Forward projection with frozen scale
        render_pkg = render_sonar(
            viewpoint_cam, gaussians, background,
            sonar_config=sonar_config,
            scale_factor=sonar_scale_factor,
            sonar_extrinsic=None
        )
        rendered = render_pkg["render"]

        # Compute loss
        Ll1 = l1_loss(rendered, gt_image)
        ssim_val = ssim(rendered, gt_image)
        loss = 0.8 * Ll1 + 0.2 * (1 - ssim_val)

        # Backward
        loss.backward()

        # Update surfels only
        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.update_learning_rate(iteration)

            # FOV-aware pruning: remove surfels that drifted outside all training FOVs
            if FOV_PRUNE_INTERVAL > 0 and iteration % FOV_PRUNE_INTERVAL == 0:
                num_pruned = prune_outside_fov(gaussians, training_frames, sonar_config, sonar_scale_factor)
                if num_pruned > 0:
                    print(f"  [FOV prune] Removed {num_pruned} surfels outside FOV, {len(gaussians.get_xyz)} remaining")

        scale_value = sonar_scale_factor.get_scale_value()
        record_metrics(loss.item(), scale_value, "stage2")

        if iteration % 10 == 0 or iteration == 1:
            print(f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}")

    print(f"Stage 2 complete. Surfels: {len(gaussians.get_xyz)}")

    # Extract mesh after Stage 2
    extract_and_save_mesh(
        gaussians, mesh_cameras, pipe_args, bg_color,
        OUTPUT_DIR, "mesh_after_stage2.ply",
        depth_trunc=depth_trunc, voxel_size=voxel_size, sdf_trunc=sdf_trunc
    )

    # Save comparison images after Stage 2
    save_comparison_images(training_frames, gaussians, background, sonar_config,
                           sonar_scale_factor, OUTPUT_DIR, "after_stage2")

# =============================================================================
# Stage 3: Joint Fine-tuning
# =============================================================================
if STAGE3_ITERATIONS > 0:
    print("\n" + "=" * 60)
    print(f"STAGE 3: Joint fine-tuning ({STAGE3_ITERATIONS} iterations)")
    print("=" * 60)

    # Keep scale frozen (using known calibrated value)
    # TODO: Re-enable scale learning once scale factor convergence is fixed
    sonar_scale_factor._log_scale.requires_grad = False

    for iteration in range(1, STAGE3_ITERATIONS + 1):
        frame_idx = (iteration - 1) % len(training_frames)
        viewpoint_cam = training_frames[frame_idx]

        gt_image = preprocess_gt_image(viewpoint_cam.original_image)

        render_pkg = render_sonar(
            viewpoint_cam, gaussians, background,
            sonar_config=sonar_config,
            scale_factor=sonar_scale_factor,
            sonar_extrinsic=None
        )
        rendered = render_pkg["render"]

        Ll1 = l1_loss(rendered, gt_image)
        ssim_val = ssim(rendered, gt_image)
        loss = 0.8 * Ll1 + 0.2 * (1 - ssim_val)

        loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            # Scale frozen - no optimizer step
            gaussians.update_learning_rate(STAGE2_ITERATIONS + iteration)

            # FOV-aware pruning
            if FOV_PRUNE_INTERVAL > 0 and iteration % FOV_PRUNE_INTERVAL == 0:
                num_pruned = prune_outside_fov(gaussians, training_frames, sonar_config, sonar_scale_factor)
                if num_pruned > 0:
                    print(f"  [FOV prune] Removed {num_pruned} surfels outside FOV, {len(gaussians.get_xyz)} remaining")

        scale_value = sonar_scale_factor.get_scale_value()
        record_metrics(loss.item(), scale_value, "stage3")

        if iteration % 10 == 0 or iteration == 1:
            print(f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}")

    print(f"Stage 3 complete. Surfels: {len(gaussians.get_xyz)}")

    # Final FOV diagnostic and forced prune before mesh extraction
    print("\n  Final FOV check before mesh extraction:")
    for i, cam in enumerate(training_frames):
        details = is_in_sonar_fov(gaussians.get_xyz, cam, sonar_config, sonar_scale_factor, return_details=True)
        in_fov = details["in_fov"]
        print(f"    Frame {i}: {in_fov.sum().item()}/{len(gaussians.get_xyz)} in FOV")
        if not in_fov.all():
            # Show stats for out-of-FOV surfels
            out_mask = ~in_fov
            r = details["range_vals"][out_mask]
            az = details["azimuth_deg"][out_mask]
            el = details["elevation_deg"][out_mask]
            if len(r) > 0:
                print(f"      Out-of-FOV: range=[{r.min().item():.2f}, {r.max().item():.2f}]m, "
                      f"az=[{az.min().item():.1f}, {az.max().item():.1f}]°, "
                      f"el=[{el.min().item():.1f}, {el.max().item():.1f}]°")

    # Force final prune
    num_pruned = prune_outside_fov(gaussians, training_frames, sonar_config, sonar_scale_factor)
    if num_pruned > 0:
        print(f"  [Final prune] Removed {num_pruned} surfels, {len(gaussians.get_xyz)} remaining")

    # Save final surfel positions as point cloud (for verification)
    final_xyz = gaussians.get_xyz.detach().cpu().numpy()
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_xyz)
    final_pcd_path = os.path.join(OUTPUT_DIR, "surfels_after_training.ply")
    o3d.io.write_point_cloud(final_pcd_path, final_pcd)
    print(f"  Saved surfel positions: {final_pcd_path} ({len(final_xyz)} points)")

    # Extract mesh after Stage 3
    extract_and_save_mesh(
        gaussians, mesh_cameras, pipe_args, bg_color,
        OUTPUT_DIR, "mesh_after_stage3.ply",
        depth_trunc=depth_trunc, voxel_size=voxel_size, sdf_trunc=sdf_trunc
    )

    # Save comparison images after Stage 3
    save_comparison_images(training_frames, gaussians, background, sonar_config,
                           sonar_scale_factor, OUTPUT_DIR, "after_stage3")
    save_raw_comparison_images(training_frames, gaussians, background, sonar_config,
                               sonar_scale_factor, OUTPUT_DIR, "after_stage3",
                               DATASET_PATH, dataset_args.sonar_images)

stage_boundaries = []
completed_iters = 0
if STAGE1_ITERATIONS > 0:
    completed_iters += STAGE1_ITERATIONS
    stage_boundaries.append((completed_iters, "Stage 1"))
if STAGE2_ITERATIONS > 0:
    completed_iters += STAGE2_ITERATIONS
    stage_boundaries.append((completed_iters, "Stage 2"))
if STAGE3_ITERATIONS > 0:
    completed_iters += STAGE3_ITERATIONS
    stage_boundaries.append((completed_iters, "Stage 3"))

plot_training_metrics(OUTPUT_DIR, stage_boundaries)

print(f"\nFinal scale factor: {sonar_scale_factor.get_scale_value():.6f}")

# =============================================================================
# Summary
# =============================================================================
total_iters = STAGE1_ITERATIONS + STAGE2_ITERATIONS + STAGE3_ITERATIONS
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nFinal scale factor: {sonar_scale_factor.get_scale_value():.6f}")
print(f"\nGenerated files:")
print(f"  - sonar_init_points.ply       (Combined points from {NUM_TRAINING_FRAMES} frames)")
print(f"  - pose_pyramids_wireframe.ply (Wireframes for training frames)")
print(f"  - mesh_before_training.ply    (Mesh before any training)")
print(f"  - mesh_after_iter1.ply        (Mesh after 1st iteration)")
print(f"  - mesh_after_stage1.ply       (Mesh after Stage 1: scale learning)")
print(f"  - mesh_after_stage2.ply       (Mesh after Stage 2: surfel learning)")
print(f"  - mesh_after_stage3.ply       (Mesh after Stage 3: joint fine-tuning)")
print(f"  - comparison_before_training_frameN.png (Before any training)")
print(f"  - comparison_after_stage1_frameN.png    (After scale learning)")
print(f"  - comparison_after_stage2_frameN.png    (After surfel learning)")
print(f"  - comparison_after_stage3_frameN.png    (After joint fine-tuning)")
print(f"  - comparison_after_stage3_raw_frameN.png (Raw sonar vs rendered)")
print(f"  - scale_and_loss.png                    (Scale and loss curves)")
