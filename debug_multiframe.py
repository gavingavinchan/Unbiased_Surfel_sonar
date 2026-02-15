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
- mesh_poisson_init.ply: Poisson mesh from initial point cloud
- mesh_poisson_after_stage1.ply: Poisson mesh after Stage 1
- mesh_poisson_after_stage2.ply: Poisson mesh after Stage 2
- mesh_poisson_after_stage3.ply: Poisson mesh after Stage 3
- mesh_poisson_after_iter1.ply: Poisson mesh after iter 1
- comparison_frame_N.png: GT vs rendered for each training frame
"""

import os
import sys
import atexit
import csv
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
from gaussian_renderer import render_sonar, render, quaternion_to_normal
from utils.sonar_utils import (SonarConfig, SonarScaleFactor, SonarExtrinsic,
                                sonar_frame_to_points, sonar_frames_to_point_cloud,
                                back_project_bins,
                                SONAR_CAMERA_FRAME_CONVENTION, SONAR_IMAGE_CONVENTION,
                                SONAR_MOUNT_TRANSLATION_CAM, SONAR_MOUNT_PITCH_DEG,
                                run_sonar_convention_asserts)
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import l1_loss, ssim
from utils.mesh_utils import GaussianExtractor
from utils.general_utils import inverse_sigmoid
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

    # Apply scale factor to translation and points
    scale = scale_factor.scale if scale_factor is not None else 1.0
    xyz_scaled = xyz * scale
    t_w2v_scaled = scale * t_w2v

    # Transform points to sonar frame: p_sonar = p_world_scaled @ R.T + t_scaled
    # This matches render_sonar exactly
    points_sonar = (xyz_scaled @ R_w2v.T) + t_w2v_scaled  # [N, 3]

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
    scale = scale_factor.scale if scale_factor is not None else 1.0
    xyz_scaled = xyz * scale
    t_w2v_scaled = scale * t_w2v
    points_sonar = (xyz_scaled @ R_w2v.T) + t_w2v_scaled


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

# Bright-pixel loss settings (top-k brightest GT pixels)
BRIGHT_PERCENTILE = 95.0
BRIGHT_WEIGHT = 0.5
BRIGHT_MIN_PIXELS = 32
LOSS_SMOOTH_WINDOW = 200
LOSS_LOG_FLUSH_INTERVAL = 100
GAUSSIAN_OPACITY_LR = 0.05
FIXED_OPACITY_TARGET = 0.999


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


def get_epoch_indices(num_frames, epoch_seed):
    indices = list(range(num_frames))
    random.Random(epoch_seed).shuffle(indices)
    return indices


def compute_bright_loss(rendered, gt_image, percentile=BRIGHT_PERCENTILE, min_pixels=BRIGHT_MIN_PIXELS):
    gt_gray = gt_image.mean(dim=0)
    diff_gray = (rendered - gt_image).abs().mean(dim=0)

    threshold = torch.quantile(gt_gray, percentile / 100.0)
    bright_mask = gt_gray >= threshold

    if bright_mask.sum() < min_pixels:
        bright_mask = gt_gray >= torch.quantile(gt_gray, 0.5)

    return diff_gray[bright_mask].mean()


def apply_opacity_policy(gaussians, fixed_opacity, fixed_target=FIXED_OPACITY_TARGET,
                         learnable_opacity_lr=GAUSSIAN_OPACITY_LR):
    """Apply and re-apply opacity policy while keeping optimizer group structure intact."""
    opacity_group = None
    for group in gaussians.optimizer.param_groups:
        if group.get("name") == "opacity":
            opacity_group = group
            break

    if opacity_group is None:
        raise RuntimeError("Gaussian optimizer is missing required 'opacity' param group")

    if fixed_opacity:
        target_activated = torch.full_like(gaussians._opacity.data, fixed_target)
        fixed_logits = inverse_sigmoid(target_activated)
        gaussians._opacity.data.copy_(fixed_logits)
        gaussians._opacity.requires_grad_(False)
        gaussians._opacity.grad = None
        opacity_group["lr"] = 0.0
    else:
        gaussians._opacity.requires_grad_(True)
        opacity_group["lr"] = learnable_opacity_lr


def save_training_checkpoint(checkpoint_path, gaussians, sonar_scale_factor, scale_optimizer,
                             iteration, stage_name, metadata=None):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    payload = {
        "gaussians_capture": gaussians.capture(),
        "iteration": int(iteration),
        "stage_name": str(stage_name),
        "sonar_scale_state_dict": sonar_scale_factor.state_dict(),
        "scale_optimizer_state_dict": scale_optimizer.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)
    print(f"[Checkpoint] Saved: {checkpoint_path} (iter={iteration}, stage={stage_name})")


def load_training_checkpoint(checkpoint_path, gaussians, gaussian_training_args,
                             sonar_scale_factor, scale_optimizer):
    try:
        payload = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cuda")

    if isinstance(payload, tuple) and len(payload) == 2:
        # Compatibility with legacy tuple checkpoints: (gaussians.capture(), iteration)
        model_args, iteration = payload
        gaussians.restore(model_args, gaussian_training_args)
        return int(iteration), {"format": "legacy_tuple"}

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}")

    model_args = payload.get("gaussians_capture")
    if model_args is None:
        raise RuntimeError(f"Missing 'gaussians_capture' in checkpoint: {checkpoint_path}")

    gaussians.restore(model_args, gaussian_training_args)

    scale_state = payload.get("sonar_scale_state_dict")
    if scale_state is not None:
        sonar_scale_factor.load_state_dict(scale_state)

    scale_optim_state = payload.get("scale_optimizer_state_dict")
    if scale_optim_state is not None:
        scale_optimizer.load_state_dict(scale_optim_state)

    iteration = int(payload.get("iteration", 0))
    metadata = payload.get("metadata", {})
    return iteration, metadata


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


LOG_FILE = None
LOSS_LOG_HANDLE = None
LOSS_LOG_PATH = None


def setup_logging(output_dir):
    global LOG_FILE
    log_path = os.path.join(output_dir, "run.log")
    LOG_FILE = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, LOG_FILE)
    sys.stderr = Tee(sys.stderr, LOG_FILE)
    print(f"Logging to: {log_path}")
    return log_path


def init_loss_log(output_dir):
    global LOSS_LOG_HANDLE, LOSS_LOG_PATH
    LOSS_LOG_PATH = os.path.join(output_dir, "loss_log.csv")
    LOSS_LOG_HANDLE = open(LOSS_LOG_PATH, "w")
    LOSS_LOG_HANDLE.write("iter,stage,L1,SSIM,base_loss,bright_loss,total_loss,scale,num_points\n")
    LOSS_LOG_HANDLE.flush()


def log_loss(iteration, stage_name, l1_value, ssim_value, base_loss, bright_loss, total_loss,
             scale_value, num_points):
    if LOSS_LOG_HANDLE is None:
        return

    LOSS_LOG_HANDLE.write(
        f"{iteration},{stage_name},{l1_value:.6f},{ssim_value:.6f},{base_loss:.6f},"
        f"{bright_loss:.6f},{total_loss:.6f},{scale_value:.6f},{num_points}\n"
    )

    if LOSS_LOG_FLUSH_INTERVAL > 0 and iteration % LOSS_LOG_FLUSH_INTERVAL == 0:
        LOSS_LOG_HANDLE.flush()


def close_logs():
    if LOSS_LOG_HANDLE is not None:
        LOSS_LOG_HANDLE.flush()
        LOSS_LOG_HANDLE.close()
    if LOG_FILE is not None:
        LOG_FILE.flush()
        LOG_FILE.close()


def print_sonar_diagnostics(diag, prefix=""):
    if not diag:
        return
    gain_mode = diag.get("attenuation_gain_mode", "unknown")
    enabled = diag.get("attenuation_enabled", False)
    effective_gain = diag.get("attenuation_effective_gain", 0.0)
    exp = diag.get("attenuation_exp", 0.0)
    r0 = diag.get("attenuation_r0", 0.0)
    eps = diag.get("attenuation_eps", 0.0)
    near_mean = diag.get("near_range_mean_intensity", 0.0)
    far_mean = diag.get("far_range_mean_intensity", 0.0)
    ratio = diag.get("far_over_near_ratio", 0.0)
    sat = diag.get("near_range_saturation_rate", 0.0)
    nan_inf = diag.get("nan_inf_count", 0)

    print(
        f"{prefix}attenuation enabled={enabled} mode={gain_mode} "
        f"gain={effective_gain:.6f} exp={exp:.3f} r0={r0:.3f} eps={eps:.1e}"
    )
    print(
        f"{prefix}near_mean={near_mean:.6f}, far_mean={far_mean:.6f}, "
        f"far/near={ratio:.6f}, near_sat={sat:.6f}, nan_inf={nan_inf}"
    )


def render_sonar_for_mesh(sonar_config, scale_factor, sonar_extrinsic=None):
    def _render(viewpoint_cam, gaussians, pipe, bg_color):
        return render_sonar(
            viewpoint_cam, gaussians, bg_color,
            sonar_config=sonar_config,
            scale_factor=scale_factor,
            sonar_extrinsic=sonar_extrinsic,
            **SONAR_RENDER_KWARGS,
        )
    return _render


def save_poisson_mesh(points, normals, output_dir, filename, opacities=None, scales=None):
    if points.size == 0:
        print("  Skipping Poisson mesh (no points)")
        return

    if scales is not None:
        scales = np.asarray(scales)

    if opacities is not None:
        opacities = np.asarray(opacities).reshape(-1)
        min_opacity = POISSON_MIN_OPACITY
        perc_opacity = np.quantile(opacities, POISSON_OPACITY_PERCENTILE)
        opacity_cutoff = max(min_opacity, perc_opacity)
        keep_mask = opacities >= opacity_cutoff
        points = points[keep_mask]
        normals = normals[keep_mask]
        if scales is not None:
            scales = scales[keep_mask]
        print(f"  Poisson filter: opacity >= {opacity_cutoff:.4f} (kept {keep_mask.sum()}/{len(keep_mask)})")

    if scales is not None:
        if scales.ndim == 2:
            scales = np.max(scales, axis=1)
        max_scale = np.quantile(scales, POISSON_SCALE_PERCENTILE)
        keep_mask = scales <= max_scale
        points = points[keep_mask]
        normals = normals[keep_mask]
        print(f"  Poisson filter: scale <= {max_scale:.4f} (kept {keep_mask.sum()}/{len(keep_mask)})")

    if points.size == 0:
        print("  Skipping Poisson mesh (filtered all points)")
        return

    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = o3d.utility.Vector3dVector(points)
    pcd_mesh.normals = o3d.utility.Vector3dVector(normals)
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_mesh, depth=POISSON_DEPTH
    )
    densities = np.asarray(densities)
    if densities.size > 0:
        cutoff = np.quantile(densities, POISSON_DENSITY_QUANTILE)
        poisson_mesh.remove_vertices_by_mask(densities < cutoff)
        print(f"  Removed low-density vertices below {cutoff:.6f}")
    poisson_mesh_path = os.path.join(output_dir, filename)
    o3d.io.write_triangle_mesh(poisson_mesh_path, poisson_mesh)
    print(f"  Saved: {poisson_mesh_path} (V={len(poisson_mesh.vertices)}, T={len(poisson_mesh.triangles)})")


def extract_and_save_mesh(gaussians, mesh_cameras, pipe_args, bg_color, sonar_config,
                          scale_factor, output_dir, filename, depth_trunc=None, voxel_size=None, sdf_trunc=None,
                          sonar_extrinsic=None):
    """Helper to extract and save mesh at a checkpoint."""
    print(f"  Extracting mesh: {filename}")
    render_fn = render_sonar_for_mesh(sonar_config, scale_factor, sonar_extrinsic=sonar_extrinsic)
    extractor = GaussianExtractor(gaussians, render_fn, pipe_args, bg_color=bg_color)
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
                sonar_extrinsic=None,
                **SONAR_RENDER_KWARGS,
            )
            rendered = render_pkg["render"]

        if i == 0:
            print_sonar_diagnostics(render_pkg.get("sonar_diagnostics"), prefix="    ")

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
    raw_render_kwargs = dict(SONAR_RENDER_KWARGS)
    if raw_render_kwargs.get("range_atten_auto_gain", False):
        # Keep raw comparison faithful; avoid per-frame auto-gain amplification.
        raw_render_kwargs["range_atten_auto_gain"] = False
        raw_render_kwargs["range_atten_gain"] = SONAR_RANGE_ATTEN_GAIN
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
                sonar_extrinsic=None,
                **raw_render_kwargs,
            )
            rendered = render_pkg["render"]

        rendered_np = (np.clip(rendered[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        rendered_img = Image.fromarray(rendered_np, mode="L").resize(raw_image.size, Image.BILINEAR)
        rendered_resized = np.array(rendered_img)

        comparison_raw = np.hstack([raw_np, rendered_resized])
        filename = f"comparison_{stage_name}_raw_frame{i}.png"
        Image.fromarray(comparison_raw, mode="L").save(os.path.join(output_dir, filename))

    print(f"  Saved raw-frame comparisons for {stage_name}")


def write_csv_rows(csv_path, fieldnames, rows):
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_frame_set(frame_set_name, frames, gaussians, background, sonar_config,
                       scale_factor, output_dir):
    if not frames:
        return None

    rows = []
    with torch.no_grad():
        for i, cam in enumerate(frames):
            gt_image = preprocess_gt_image(cam.original_image)
            render_pkg = render_sonar(
                cam, gaussians, background,
                sonar_config=sonar_config,
                scale_factor=scale_factor,
                sonar_extrinsic=None,
                **SONAR_RENDER_KWARGS,
            )
            rendered = render_pkg["render"]

            l1_val = l1_loss(rendered, gt_image)
            ssim_val = ssim(rendered, gt_image)
            base_loss = 0.8 * l1_val + 0.2 * (1 - ssim_val)
            bright_loss = compute_bright_loss(rendered, gt_image)
            total_loss = (1 - BRIGHT_WEIGHT) * base_loss + BRIGHT_WEIGHT * bright_loss

            rows.append({
                "frame_idx": i,
                "image_name": cam.image_name,
                "l1": float(l1_val.item()),
                "ssim": float(ssim_val.item()),
                "base_loss": float(base_loss.item()),
                "bright_loss": float(bright_loss.item()),
                "total_loss": float(total_loss.item()),
            })

    csv_path = os.path.join(output_dir, f"final_eval_{frame_set_name}_frames.csv")
    fieldnames = ["frame_idx", "image_name", "l1", "ssim", "base_loss", "bright_loss", "total_loss"]
    write_csv_rows(csv_path, fieldnames, rows)

    loss_vals = np.array([row["total_loss"] for row in rows], dtype=np.float64)
    ssim_vals = np.array([row["ssim"] for row in rows], dtype=np.float64)
    print(
        f"[Final Eval:{frame_set_name}] frames={len(rows)} "
        f"loss_mean={loss_vals.mean():.6f}, loss_std={loss_vals.std():.6f}, "
        f"loss_min={loss_vals.min():.6f}, loss_max={loss_vals.max():.6f}, "
        f"ssim_mean={ssim_vals.mean():.4f}, ssim_std={ssim_vals.std():.4f}"
    )

    worst_rows = sorted(rows, key=lambda row: row["total_loss"], reverse=True)[: min(5, len(rows))]
    print(f"  [Final Eval:{frame_set_name}] Worst frames by total loss:")
    for row in worst_rows:
        print(
            f"    frame={row['frame_idx']:3d} ({row['image_name']}): "
            f"loss={row['total_loss']:.6f}, ssim={row['ssim']:.4f}"
        )
    print(f"  [Final Eval:{frame_set_name}] Saved CSV: {csv_path}")

    return {
        "rows": rows,
        "csv_path": csv_path,
        "loss_mean": float(loss_vals.mean()),
        "loss_std": float(loss_vals.std()),
        "ssim_mean": float(ssim_vals.mean()),
        "ssim_std": float(ssim_vals.std()),
    }


def summarize_training_frame_visits(training_frames, frame_visit_counts, frame_loss_sums,
                                    frame_loss_counts, output_dir):
    if len(training_frames) == 0:
        return

    rows = []
    for i, cam in enumerate(training_frames):
        visit_count = int(frame_visit_counts[i])
        loss_count = int(frame_loss_counts[i])
        avg_loss = float(frame_loss_sums[i] / loss_count) if loss_count > 0 else float("nan")
        rows.append({
            "frame_idx": i,
            "image_name": cam.image_name,
            "visit_count": visit_count,
            "avg_training_loss": avg_loss,
        })

    csv_path = os.path.join(output_dir, "frame_training_visits.csv")
    write_csv_rows(csv_path, ["frame_idx", "image_name", "visit_count", "avg_training_loss"], rows)

    total_visits = int(frame_visit_counts.sum())
    max_visits = int(frame_visit_counts.max()) if total_visits > 0 else 0
    min_visits = int(frame_visit_counts.min()) if total_visits > 0 else 0
    zero_visit = int((frame_visit_counts == 0).sum())
    concentration = (max_visits / total_visits) if total_visits > 0 else 0.0
    print(
        f"[Frame Coverage] total_visits={total_visits}, min={min_visits}, max={max_visits}, "
        f"zero_visit_frames={zero_visit}, max_share={concentration:.4f}"
    )
    if zero_visit > 0:
        print(
            f"[Issue] {zero_visit} training frames received zero optimization visits; "
            "cross-view consistency estimates may be biased."
        )
    print(f"  [Frame Coverage] Saved CSV: {csv_path}")


def compute_multiview_support_metrics(gaussians, frames, sonar_config, scale_factor):
    if not frames:
        return None

    xyz = gaussians.get_xyz
    num_surfels = int(xyz.shape[0])
    if num_surfels == 0:
        return {
            "num_surfels": 0,
            "num_frames": len(frames),
            "support_ge_1_frac": 0.0,
            "support_ge_2_frac": 0.0,
            "support_ge_3_frac": 0.0,
            "support_mean": 0.0,
            "support_median": 0.0,
            "single_view_count": 0,
            "single_view_top_share": 0.0,
            "nearest_owner_top_share": 0.0,
            "visible_counts_per_frame": [0 for _ in frames],
            "single_view_owner_counts": [0 for _ in frames],
            "nearest_owner_counts": [0 for _ in frames],
        }

    with torch.no_grad():
        visibility_masks = []
        visible_ranges = []
        for cam in frames:
            details = is_in_sonar_fov(xyz, cam, sonar_config, scale_factor, return_details=True)
            in_fov = details["in_fov"]
            visibility_masks.append(in_fov)
            visible_ranges.append(
                torch.where(in_fov, details["range_vals"], torch.full_like(details["range_vals"], float("inf")))
            )

        visibility = torch.stack(visibility_masks, dim=0)
        support_counts = visibility.sum(dim=0)
        visible_counts_per_frame = visibility.sum(dim=1)

        support_ge_1 = (support_counts >= 1).float().mean().item()
        support_ge_2 = (support_counts >= 2).float().mean().item()
        support_ge_3 = (support_counts >= 3).float().mean().item()
        support_mean = support_counts.float().mean().item()
        support_median = support_counts.float().median().item()

        single_view_mask = support_counts == 1
        single_view_count = int(single_view_mask.sum().item())
        if single_view_count > 0:
            single_owners = visibility[:, single_view_mask].float().argmax(dim=0)
            single_view_owner_counts = torch.bincount(single_owners, minlength=len(frames)).cpu().numpy()
            single_view_top_share = float(single_view_owner_counts.max() / single_view_owner_counts.sum())
        else:
            single_view_owner_counts = np.zeros(len(frames), dtype=np.int64)
            single_view_top_share = 0.0

        range_stack = torch.stack(visible_ranges, dim=0)
        any_visible = visibility.any(dim=0)
        nearest_owner = range_stack.argmin(dim=0)
        nearest_visible = nearest_owner[any_visible]
        if nearest_visible.numel() > 0:
            nearest_owner_counts = torch.bincount(nearest_visible, minlength=len(frames)).cpu().numpy()
            nearest_owner_top_share = float(nearest_owner_counts.max() / nearest_owner_counts.sum())
        else:
            nearest_owner_counts = np.zeros(len(frames), dtype=np.int64)
            nearest_owner_top_share = 0.0

    return {
        "num_surfels": num_surfels,
        "num_frames": len(frames),
        "support_ge_1_frac": float(support_ge_1),
        "support_ge_2_frac": float(support_ge_2),
        "support_ge_3_frac": float(support_ge_3),
        "support_mean": float(support_mean),
        "support_median": float(support_median),
        "single_view_count": single_view_count,
        "single_view_top_share": single_view_top_share,
        "nearest_owner_top_share": nearest_owner_top_share,
        "visible_counts_per_frame": visible_counts_per_frame.cpu().numpy().tolist(),
        "single_view_owner_counts": single_view_owner_counts.tolist(),
        "nearest_owner_counts": nearest_owner_counts.tolist(),
    }


def report_support_metrics(label, metrics, frames, output_dir):
    if metrics is None:
        print(f"[Support:{label}] skipped (no frames)")
        return

    print(
        f"[Support:{label}] surfels={metrics['num_surfels']}, frames={metrics['num_frames']}, "
        f"support>=1={metrics['support_ge_1_frac']:.4f}, "
        f"support>=2={metrics['support_ge_2_frac']:.4f}, "
        f"support>=3={metrics['support_ge_3_frac']:.4f}, "
        f"mean={metrics['support_mean']:.3f}, median={metrics['support_median']:.3f}"
    )
    print(
        f"  [Support:{label}] single_view_count={metrics['single_view_count']}, "
        f"single_view_top_share={metrics['single_view_top_share']:.4f}, "
        f"nearest_owner_top_share={metrics['nearest_owner_top_share']:.4f}"
    )

    rows = []
    for i, cam in enumerate(frames):
        rows.append({
            "frame_idx": i,
            "image_name": cam.image_name,
            "visible_surfel_count": int(metrics["visible_counts_per_frame"][i]),
            "single_view_owner_count": int(metrics["single_view_owner_counts"][i]),
            "nearest_owner_count": int(metrics["nearest_owner_counts"][i]),
        })
    csv_path = os.path.join(output_dir, f"support_metrics_{label}.csv")
    write_csv_rows(
        csv_path,
        [
            "frame_idx",
            "image_name",
            "visible_surfel_count",
            "single_view_owner_count",
            "nearest_owner_count",
        ],
        rows,
    )
    print(f"  [Support:{label}] Saved CSV: {csv_path}")


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


def smooth_series(values, window, x_values=None):
    if not values:
        return np.array([]), np.array([])

    if x_values is None:
        x_values = list(range(1, len(values) + 1))

    window = max(1, min(window, len(values)))
    if window == 1:
        return np.array(values), np.array(x_values)

    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    x_smoothed = np.array(x_values[window - 1:])
    return smoothed, x_smoothed


def plot_training_metrics(output_dir, stage_boundaries):
    if not metric_iters:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(metric_iters, metric_scale, color="tab:blue", linewidth=1.5)
    axes[0].set_ylabel("Scale")
    axes[0].set_title("Scale Factor and Loss")

    axes[1].plot(metric_iters, metric_loss, color="tab:orange", linewidth=1.0, alpha=0.4, label="Loss")

    if LOSS_SMOOTH_WINDOW > 1:
        smoothed_loss, smoothed_iters = smooth_series(metric_loss, LOSS_SMOOTH_WINDOW, metric_iters)
        if smoothed_loss.size > 0:
            axes[1].plot(smoothed_iters, smoothed_loss, color="tab:red", linewidth=2.0,
                         label=f"Loss (MA {LOSS_SMOOTH_WINDOW})")

    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].legend(loc="upper right")

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
DATASET_PATHS = {
    "legacy": "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs",
    "r2": "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs_R2",
}
DATASET_KEY = os.environ.get("SONAR_DATASET", "r2")
if DATASET_KEY not in DATASET_PATHS:
    raise ValueError(f"Unknown dataset key '{DATASET_KEY}'. Options: {list(DATASET_PATHS)}")
DATASET_PATH = DATASET_PATHS[DATASET_KEY]

INIT_SCALE_FACTORS = {
    "legacy": 0.65,
    "r2": 0.6127,
}
INIT_SCALE_FACTOR = INIT_SCALE_FACTORS[DATASET_KEY]

OUTPUT_DIR_BASE = f"./output/debug_multiframe_{DATASET_KEY}"
OUTPUT_DIR_OVERRIDE = os.environ.get("SONAR_OUTPUT_DIR")
NUM_TRAINING_FRAMES_DEFAULT = 500  # Number of frames to use for training
PYRAMID_DEPTH = 0.5

def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def env_bool(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() not in ("0", "false", "no", "off")


def env_choice(name, default, choices):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    normalized = value.strip().lower()
    if normalized not in choices:
        options = ", ".join(sorted(choices))
        raise ValueError(f"Invalid {name}='{value}'. Expected one of: {options}")
    return normalized


# Curriculum learning parameters
STAGE1_ITERATIONS = 0   # Learn scale only (surfels frozen) - DISABLED, using known scale
STAGE2_ITERATIONS = 1000  # Learn surfels only (scale frozen)
STAGE3_ITERATIONS = 1   # Joint fine-tuning

# FOV-aware pruning: remove surfels that drift outside all training cameras' FOV
FOV_PRUNE_INTERVAL = 100  # Prune every N iterations (0 to disable)

POISSON_MESH = True
POISSON_DEPTH = 9
POISSON_DENSITY_QUANTILE = 0.02
POISSON_MIN_OPACITY = 0.05
POISSON_OPACITY_PERCENTILE = 0.2
POISSON_SCALE_PERCENTILE = 0.9

POISSON_DEPTH = env_int("POISSON_DEPTH", POISSON_DEPTH)
POISSON_DENSITY_QUANTILE = env_float("POISSON_DENSITY_QUANTILE", POISSON_DENSITY_QUANTILE)
POISSON_MIN_OPACITY = env_float("POISSON_MIN_OPACITY", POISSON_MIN_OPACITY)
POISSON_OPACITY_PERCENTILE = env_float("POISSON_OPACITY_PERCENTILE", POISSON_OPACITY_PERCENTILE)
POISSON_SCALE_PERCENTILE = env_float("POISSON_SCALE_PERCENTILE", POISSON_SCALE_PERCENTILE)

STAGE2_ITERATIONS = env_int("SONAR_STAGE2_ITERS", STAGE2_ITERATIONS)
STAGE3_ITERATIONS = env_int("SONAR_STAGE3_ITERS", STAGE3_ITERATIONS)
NUM_TRAINING_FRAMES = env_int("SONAR_NUM_FRAMES", NUM_TRAINING_FRAMES_DEFAULT)
SONAR_HOLDOUT_FRAMES = max(0, env_int("SONAR_HOLDOUT_FRAMES", 0))

SONAR_CONVENTION_ASSERTS = env_bool("SONAR_CONVENTION_ASSERTS", True)
SONAR_USE_RANGE_ATTEN = env_bool("SONAR_USE_RANGE_ATTEN", True)
SONAR_RANGE_ATTEN_EXP = env_float("SONAR_RANGE_ATTEN_EXP", 2.0)
SONAR_RANGE_ATTEN_GAIN = env_float("SONAR_RANGE_ATTEN_GAIN", 1.0)
SONAR_RANGE_ATTEN_R0 = env_float("SONAR_RANGE_ATTEN_R0", 0.35)
SONAR_RANGE_ATTEN_EPS = env_float("SONAR_RANGE_ATTEN_EPS", 1e-6)
SONAR_RANGE_ATTEN_AUTO_GAIN_ENV = os.environ.get("SONAR_RANGE_ATTEN_AUTO_GAIN")
SONAR_RANGE_ATTEN_AUTO_GAIN = env_bool("SONAR_RANGE_ATTEN_AUTO_GAIN", False)
ELEV_INIT_MODE = env_choice("ELEV_INIT_MODE", "random", {"random", "zero"})
SONAR_FIXED_OPACITY = env_bool("SONAR_FIXED_OPACITY", True)
SONAR_OPACITY_WARMUP_ITERS = max(0, env_int("SONAR_OPACITY_WARMUP_ITERS", 200))
SONAR_LOAD_CHECKPOINT = os.environ.get("SONAR_LOAD_CHECKPOINT", "").strip()
SONAR_SAVE_CHECKPOINT = os.environ.get("SONAR_SAVE_CHECKPOINT", "").strip()

# In learnable-opacity mode, default to auto attenuation gain unless explicitly overridden.
if (
    (not SONAR_FIXED_OPACITY)
    and SONAR_USE_RANGE_ATTEN
    and SONAR_RANGE_ATTEN_AUTO_GAIN_ENV in (None, "")
):
    SONAR_RANGE_ATTEN_AUTO_GAIN = True

if not SONAR_USE_RANGE_ATTEN:
    SONAR_ATTENUATION_MODE = "off"
elif SONAR_RANGE_ATTEN_AUTO_GAIN:
    SONAR_ATTENUATION_MODE = "auto"
else:
    SONAR_ATTENUATION_MODE = "manual"

SONAR_RENDER_KWARGS = {
    "use_range_attenuation": SONAR_USE_RANGE_ATTEN,
    "range_atten_exp": SONAR_RANGE_ATTEN_EXP,
    "range_atten_gain": SONAR_RANGE_ATTEN_GAIN,
    "range_atten_r0": SONAR_RANGE_ATTEN_R0,
    "range_atten_eps": SONAR_RANGE_ATTEN_EPS,
    "range_atten_auto_gain": SONAR_RANGE_ATTEN_AUTO_GAIN,
}

# Create output folder
if OUTPUT_DIR_OVERRIDE:
    OUTPUT_DIR = OUTPUT_DIR_OVERRIDE
else:
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

setup_logging(OUTPUT_DIR)
init_loss_log(OUTPUT_DIR)
atexit.register(close_logs)

print("=" * 60)
print("DEBUG: Multi-Frame Training with Curriculum Learning")
print("=" * 60)
print(f"Seed: {SEED}")
print(f"Dataset: {DATASET_KEY} ({DATASET_PATH})")
print(f"Init scale: {INIT_SCALE_FACTOR}")
print(f"Num training frames: {NUM_TRAINING_FRAMES}")
print(f"Holdout frames: {SONAR_HOLDOUT_FRAMES}")
print(f"Curriculum: Stage1={STAGE1_ITERATIONS} (scale), Stage2={STAGE2_ITERATIONS} (surfels), Stage3={STAGE3_ITERATIONS} (joint)")
if NUM_TRAINING_FRAMES > 1 and (STAGE2_ITERATIONS + STAGE3_ITERATIONS) < NUM_TRAINING_FRAMES:
    print(
        "[Warning] Stage2+Stage3 iterations are fewer than selected training frames; "
        "many frames may receive zero gradient updates in this run."
    )
print(f"FOV pruning interval: {FOV_PRUNE_INTERVAL} iterations")
print(f"Convention asserts: {SONAR_CONVENTION_ASSERTS}")
print(f"Camera/view convention: {SONAR_CAMERA_FRAME_CONVENTION}")
print(f"Sonar image convention: {SONAR_IMAGE_CONVENTION}")
print(f"Mount extrinsic (camera frame): translation={SONAR_MOUNT_TRANSLATION_CAM}, pitch_deg={SONAR_MOUNT_PITCH_DEG}")
print(f"[Stage 0] ELEV_INIT_MODE={ELEV_INIT_MODE}, SONAR_FIXED_OPACITY={int(SONAR_FIXED_OPACITY)}")
if SONAR_FIXED_OPACITY:
    print("[Stage 0] Opacity mode: FIXED (target=0.999)")
else:
    print(f"[Stage 0] Opacity mode: LEARNABLE (warmup fixed for first {SONAR_OPACITY_WARMUP_ITERS} iters)")
if SONAR_LOAD_CHECKPOINT:
    print(f"[Checkpoint] Resume from: {SONAR_LOAD_CHECKPOINT}")
if SONAR_SAVE_CHECKPOINT:
    print(f"[Checkpoint] Save at end: {SONAR_SAVE_CHECKPOINT}")
if SONAR_ATTENUATION_MODE == "off":
    print("Range attenuation: OFF (all attenuation parameters ignored)")
elif SONAR_ATTENUATION_MODE == "auto":
    print(
        f"Range attenuation: AUTO gain (seed={SONAR_RANGE_ATTEN_GAIN:.4f}, "
        f"exp={SONAR_RANGE_ATTEN_EXP:.3f}, r0={SONAR_RANGE_ATTEN_R0:.3f}, eps={SONAR_RANGE_ATTEN_EPS:.1e})"
    )
else:
    print(
        f"Range attenuation: MANUAL gain={SONAR_RANGE_ATTEN_GAIN:.4f}, "
        f"exp={SONAR_RANGE_ATTEN_EXP:.3f}, r0={SONAR_RANGE_ATTEN_R0:.3f}, eps={SONAR_RANGE_ATTEN_EPS:.1e}"
    )
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

holdout_frames = []
if SONAR_HOLDOUT_FRAMES > 0:
    selected_index_set = set(frame_indices)
    remaining_indices = [idx for idx in range(len(train_cameras)) if idx not in selected_index_set]
    if len(remaining_indices) == 0:
        print("[Holdout] Requested holdout frames but no remaining cameras are available")
    else:
        if SONAR_HOLDOUT_FRAMES > len(remaining_indices):
            print(
                f"[Holdout] Requested {SONAR_HOLDOUT_FRAMES} frames but only "
                f"{len(remaining_indices)} are available; clipping"
            )
        holdout_count = min(SONAR_HOLDOUT_FRAMES, len(remaining_indices))
        holdout_pool = [train_cameras[idx] for idx in remaining_indices]
        holdout_rel_indices = select_diverse_frames(holdout_pool, holdout_count, seed=SEED + 1000)
        holdout_indices = [remaining_indices[idx] for idx in holdout_rel_indices]
        holdout_frames = [train_cameras[idx] for idx in holdout_indices]

print(f"Selected {len(training_frames)} training frames:")
for i, cam in enumerate(training_frames):
    print(f"  [{i}] {cam.image_name}")

if holdout_frames:
    print(f"Selected {len(holdout_frames)} holdout frames:")
    for i, cam in enumerate(holdout_frames):
        print(f"  [H{i}] {cam.image_name}")

frame_visit_counts = np.zeros(len(training_frames), dtype=np.int64)
frame_loss_sums = np.zeros(len(training_frames), dtype=np.float64)
frame_loss_counts = np.zeros(len(training_frames), dtype=np.int64)

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

if SONAR_CONVENTION_ASSERTS:
    report = run_sonar_convention_asserts(sonar_config, sample_camera=sample_cam, device="cuda")
    print("  Convention checks: PASS")
    print(f"    azimuth left={report.azimuth_left_rad:.6f} rad, right={report.azimuth_right_rad:.6f} rad")
    print(f"    elevation + -> y={report.positive_elevation_y:.6f}, - -> y={report.negative_elevation_y:.6f}")
    print(
        f"    transform roundtrip max_abs={report.extrinsic_roundtrip_max_abs:.3e}, "
        f"layout max_abs={report.layout_roundtrip_max_abs:.3e}"
    )
else:
    print("  Convention checks: DISABLED (SONAR_CONVENTION_ASSERTS=0)")

probe_rows = torch.tensor([10, sonar_config.image_height // 2], device="cuda", dtype=torch.long)
probe_cols = torch.tensor([0, sonar_config.image_width - 1], device="cuda", dtype=torch.long)
probe_elev_bins = torch.tensor(
    [-sonar_config.half_elevation_rad, 0.0, sonar_config.half_elevation_rad],
    device="cuda",
)
probe_points = back_project_bins(
    frame_idx=0,
    rows=probe_rows,
    cols=probe_cols,
    elev_bins=probe_elev_bins,
    cameras=training_frames,
    sonar_config=sonar_config,
    scale_factor=None,
)
if probe_points.shape != (probe_rows.shape[0], probe_elev_bins.shape[0], 3):
    raise RuntimeError(
        f"back_project_bins contract failed: expected {(probe_rows.shape[0], probe_elev_bins.shape[0], 3)}, "
        f"got {tuple(probe_points.shape)}"
    )
print(f"  back_project_bins contract: PASS shape={tuple(probe_points.shape)}")

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

stage0_rng = np.random.default_rng(SEED)
stage0_point_count = 0
stage0_y_sum = 0.0
stage0_y_sumsq = 0.0
stage0_y_min = float("inf")
stage0_y_max = float("-inf")
stage0_elev_min = float("inf")
stage0_elev_max = float("-inf")

temp_scale_factor = SonarScaleFactor(init_value=INIT_SCALE_FACTOR).cuda()

for i, cam in enumerate(training_frames):
    frame_init = sonar_frame_to_points(
        cam, sonar_config,
        intensity_threshold=INTENSITY_THRESHOLD / 255.0,  # Same threshold as training
        mask_top_rows=10,
        scale_factor=temp_scale_factor.get_scale_value(),
        elevation_mode=ELEV_INIT_MODE,
        rng=stage0_rng,
        return_debug=True,
    )
    if len(frame_init) != 3:
        raise RuntimeError("sonar_frame_to_points(return_debug=True) must return (points, colors, debug)")
    points, colors = frame_init[0], frame_init[1]
    init_debug = frame_init[2]

    if init_debug["num_points"] > 0:
        stage0_point_count += init_debug["num_points"]
        stage0_y_sum += init_debug["y_cam_sum"]
        stage0_y_sumsq += init_debug["y_cam_sumsq"]
        stage0_y_min = min(stage0_y_min, init_debug["y_cam_min"])
        stage0_y_max = max(stage0_y_max, init_debug["y_cam_max"])
        stage0_elev_min = min(stage0_elev_min, init_debug["elevation_min_rad"])
        stage0_elev_max = max(stage0_elev_max, init_debug["elevation_max_rad"])

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

if stage0_point_count > 0:
    stage0_y_mean = stage0_y_sum / stage0_point_count
    stage0_y_var = max((stage0_y_sumsq / stage0_point_count) - (stage0_y_mean ** 2), 0.0)
    stage0_y_std = math.sqrt(stage0_y_var)
    print(
        f"[Stage 0] Init points: N={stage0_point_count}, "
        f"Y mean={stage0_y_mean:.4f}, std={stage0_y_std:.4f}, "
        f"range=[{stage0_y_min:.4f}, {stage0_y_max:.4f}]"
    )
    print(
        f"[Stage 0] Elevation samples: min={stage0_elev_min:.4f} rad ({math.degrees(stage0_elev_min):.2f} deg), "
        f"max={stage0_elev_max:.4f} rad ({math.degrees(stage0_elev_max):.2f} deg)"
    )
    if ELEV_INIT_MODE == "zero" and (abs(stage0_y_mean) > 1e-6 or stage0_y_std > 1e-7):
        raise RuntimeError(
            "Zero-mode legacy-parity contract failed: expected near-zero sonar-frame Y spread "
            f"but got mean={stage0_y_mean:.3e}, std={stage0_y_std:.3e}"
        )

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

if POISSON_MESH:
    print("\nPoisson reconstruction from initial point cloud...")
    save_poisson_mesh(points, normals, OUTPUT_DIR, "mesh_poisson_init.ply")


gaussians = GaussianModel(dataset_args.sh_degree)
gaussians.create_from_pcd(basic_pcd, cameras_extent)
print(f"Gaussian count: {len(gaussians.get_xyz)}")

# Diagnostic: Check initial FOV visibility with temporary scale factor
# ============================================================================

temp_scale = SonarScaleFactor(init_value=INIT_SCALE_FACTOR).cuda()  # Use calibrated scale

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

with torch.no_grad():
    header_render = render_sonar(
        training_frames[0],
        gaussians,
        background,
        sonar_config=sonar_config,
        scale_factor=temp_scale,
        sonar_extrinsic=None,
        **SONAR_RENDER_KWARGS,
    )
print("  Initial sonar render diagnostics:")
print_sonar_diagnostics(header_render.get("sonar_diagnostics"), prefix="    ")

NUM_CAMERAS_FOR_MESH = 50
mesh_cameras = train_cameras[:NUM_CAMERAS_FOR_MESH]
gaussExtractor = GaussianExtractor(
    gaussians,
    render_sonar_for_mesh(sonar_config, temp_scale, sonar_extrinsic=None),
    pipe_args,
    bg_color=bg_color
)

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

# Save comparison images before any training (using calibrated scale)
temp_scale = SonarScaleFactor(init_value=INIT_SCALE_FACTOR).cuda()
save_comparison_images(training_frames, gaussians, background, sonar_config,
                       temp_scale, OUTPUT_DIR, "before_training")

# =============================================================================
# Setup Training
# =============================================================================
print("\n" + "=" * 60)
print("TRAINING SETUP")
print("=" * 60)

# Setup Gaussian optimizer
gaussian_training_args = Namespace(
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30000,
    feature_lr=0.0025,
    opacity_lr=GAUSSIAN_OPACITY_LR,
    scaling_lr=0.005,
    rotation_lr=0.001,
    percent_dense=0.01,
    lambda_dssim=0.2,
    densification_interval=100,
    opacity_reset_interval=3000,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_grad_threshold=0.0002,
)
gaussians.training_setup(gaussian_training_args)

# Scale factor module
# Known scale factor from calibration cube in COLMAP (true value ~0.66)
# TODO: Fix scale factor learning - currently not converging to correct value
sonar_scale_factor = SonarScaleFactor(init_value=INIT_SCALE_FACTOR).cuda()

# Separate optimizer for scale factor
scale_optimizer = torch.optim.Adam([
    {'params': [sonar_scale_factor._log_scale], 'lr': 0.01, 'name': 'sonar_scale'}
])

training_iter_offset = 0

if SONAR_LOAD_CHECKPOINT:
    resumed_iter, resume_meta = load_training_checkpoint(
        SONAR_LOAD_CHECKPOINT,
        gaussians,
        gaussian_training_args,
        sonar_scale_factor,
        scale_optimizer,
    )
    training_iter_offset = resumed_iter
    print(f"[Checkpoint] Loaded: {SONAR_LOAD_CHECKPOINT} (iter={resumed_iter})")
    if resume_meta:
        print(f"[Checkpoint] Metadata: {resume_meta}")

opacity_policy_state = {"initialized": False, "fixed": False}


def effective_fixed_opacity(global_iter):
    if SONAR_FIXED_OPACITY:
        return True
    if SONAR_OPACITY_WARMUP_ITERS <= 0:
        return False
    return global_iter <= SONAR_OPACITY_WARMUP_ITERS


def sync_opacity_policy(global_iter, context, force=False):
    fixed_now = effective_fixed_opacity(global_iter)
    mode_changed = (not opacity_policy_state["initialized"]) or opacity_policy_state["fixed"] != fixed_now
    if force or mode_changed:
        apply_opacity_policy(
            gaussians,
            fixed_opacity=fixed_now,
            fixed_target=FIXED_OPACITY_TARGET,
            learnable_opacity_lr=GAUSSIAN_OPACITY_LR,
        )
        if mode_changed and fixed_now and not SONAR_FIXED_OPACITY:
            print(
                f"[Opacity] Warmup FIXED at iter {global_iter} ({context}); "
                f"will switch to LEARNABLE after iter {SONAR_OPACITY_WARMUP_ITERS}"
            )
        elif mode_changed and (not fixed_now) and (not SONAR_FIXED_OPACITY):
            print(f"[Opacity] Switched to LEARNABLE at iter {global_iter} ({context})")
        elif force and (not mode_changed):
            mode_name = "FIXED" if fixed_now else "LEARNABLE"
            print(f"[Opacity] Re-applied {mode_name} policy at iter {global_iter} ({context})")
        opacity_policy_state["initialized"] = True
        opacity_policy_state["fixed"] = fixed_now
    return fixed_now


sync_opacity_policy(max(training_iter_offset + 1, 1), "setup")

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
t_w2v = w2c[3, :3]
print(f"  t_w2v (row 3) = {t_w2v.cpu().numpy()}")

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
            sonar_extrinsic=None,
            **SONAR_RENDER_KWARGS,
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

    epoch_indices = get_epoch_indices(len(training_frames), SEED)
    for iteration in range(1, STAGE1_ITERATIONS + 1):
        # Shuffle frames per epoch
        if (iteration - 1) % len(training_frames) == 0:
            epoch_seed = SEED + (iteration - 1) // len(training_frames)
            epoch_indices = get_epoch_indices(len(training_frames), epoch_seed)

        frame_idx = epoch_indices[(iteration - 1) % len(training_frames)]
        viewpoint_cam = training_frames[frame_idx]
        frame_visit_counts[frame_idx] += 1


        # Get ground truth (with intensity thresholding)
        gt_image = preprocess_gt_image(viewpoint_cam.original_image)

        # Forward projection WITH scale factor
        render_pkg = render_sonar(
            viewpoint_cam, gaussians, background,
            sonar_config=sonar_config,
            scale_factor=sonar_scale_factor,  # Scale factor enabled
            sonar_extrinsic=None,
            **SONAR_RENDER_KWARGS,
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
        base_loss = 0.8 * Ll1 + 0.2 * (1 - ssim_val)
        bright_loss = compute_bright_loss(rendered, gt_image)
        loss = (1 - BRIGHT_WEIGHT) * base_loss + BRIGHT_WEIGHT * bright_loss
        frame_loss_sums[frame_idx] += float(loss.item())
        frame_loss_counts[frame_idx] += 1

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
        log_loss(
            metric_step,
            "stage1",
            Ll1.item(),
            ssim_val.item(),
            base_loss.item(),
            bright_loss.item(),
            loss.item(),
            scale_value,
            len(gaussians.get_xyz)
        )

        if iteration % 10 == 0 or iteration == 1:
            print(f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, scale={scale_value:.4f}, grad={grad_val:.6f}")

        # Extract mesh after iteration 1
        if iteration == 1:
            if POISSON_MESH:
                poisson_points = gaussians.get_xyz.detach()
                poisson_normals = quaternion_to_normal(gaussians.get_rotation.detach()).cpu().numpy()
                poisson_opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)
                poisson_scale = gaussians.get_scaling.detach().cpu().numpy()
                save_poisson_mesh(
                    poisson_points.cpu().numpy(),
                    poisson_normals,
                    OUTPUT_DIR,
                    "mesh_poisson_after_iter1.ply",
                    opacities=poisson_opacity,
                    scales=poisson_scale
                )
            _, depth_trunc, voxel_size, sdf_trunc = extract_and_save_mesh(
                gaussians, mesh_cameras, pipe_args, bg_color, sonar_config,
                sonar_scale_factor, OUTPUT_DIR, "mesh_after_iter1.ply",
                sonar_extrinsic=None
            )

    print(f"Stage 1 complete. Scale factor: {sonar_scale_factor.get_scale_value():.6f}")

    if POISSON_MESH:
        poisson_points = gaussians.get_xyz.detach()
        poisson_normals = quaternion_to_normal(gaussians.get_rotation.detach()).cpu().numpy()
        poisson_opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)
        poisson_scale = gaussians.get_scaling.detach().cpu().numpy()
        save_poisson_mesh(
            poisson_points.cpu().numpy(),
            poisson_normals,
            OUTPUT_DIR,
            "mesh_poisson_after_stage1.ply",
            opacities=poisson_opacity,
            scales=poisson_scale
        )

    # Extract mesh after Stage 1
    extract_and_save_mesh(
        gaussians, mesh_cameras, pipe_args, bg_color, sonar_config,
        sonar_scale_factor, OUTPUT_DIR, "mesh_after_stage1.ply",
        depth_trunc=depth_trunc, voxel_size=voxel_size, sdf_trunc=sdf_trunc,
        sonar_extrinsic=None
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

    epoch_indices = get_epoch_indices(len(training_frames), SEED)
    stage2_global_offset = training_iter_offset + STAGE1_ITERATIONS
    for iteration in range(1, STAGE2_ITERATIONS + 1):
        # Shuffle frames per epoch
        if (iteration - 1) % len(training_frames) == 0:
            epoch_seed = SEED + (iteration - 1) // len(training_frames)
            epoch_indices = get_epoch_indices(len(training_frames), epoch_seed)

        frame_idx = epoch_indices[(iteration - 1) % len(training_frames)]
        viewpoint_cam = training_frames[frame_idx]
        global_iter = stage2_global_offset + iteration
        sync_opacity_policy(global_iter, f"stage2/iter{iteration}")
        frame_visit_counts[frame_idx] += 1

        # Get ground truth (with intensity thresholding)
        gt_image = preprocess_gt_image(viewpoint_cam.original_image)

        # Forward projection with frozen scale
        render_pkg = render_sonar(
            viewpoint_cam, gaussians, background,
            sonar_config=sonar_config,
            scale_factor=sonar_scale_factor,
            sonar_extrinsic=None,
            **SONAR_RENDER_KWARGS,
        )
        rendered = render_pkg["render"]

        # Compute loss
        Ll1 = l1_loss(rendered, gt_image)
        ssim_val = ssim(rendered, gt_image)
        base_loss = 0.8 * Ll1 + 0.2 * (1 - ssim_val)
        bright_loss = compute_bright_loss(rendered, gt_image)
        loss = (1 - BRIGHT_WEIGHT) * base_loss + BRIGHT_WEIGHT * bright_loss
        frame_loss_sums[frame_idx] += float(loss.item())
        frame_loss_counts[frame_idx] += 1

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
                    sync_opacity_policy(global_iter, f"stage2/fov_prune@{iteration}", force=True)
                    print(f"  [FOV prune] Removed {num_pruned} surfels outside FOV, {len(gaussians.get_xyz)} remaining")

        scale_value = sonar_scale_factor.get_scale_value()
        record_metrics(loss.item(), scale_value, "stage2")
        log_loss(
            metric_step,
            "stage2",
            Ll1.item(),
            ssim_val.item(),
            base_loss.item(),
            bright_loss.item(),
            loss.item(),
            scale_value,
            len(gaussians.get_xyz)
        )

        if iteration % 10 == 0 or iteration == 1:
            print(f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}")

    print(f"Stage 2 complete. Surfels: {len(gaussians.get_xyz)}")

    if POISSON_MESH:
        poisson_points = gaussians.get_xyz.detach()
        poisson_normals = quaternion_to_normal(gaussians.get_rotation.detach()).cpu().numpy()
        poisson_opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)
        poisson_scale = gaussians.get_scaling.detach().cpu().numpy()
        save_poisson_mesh(
            poisson_points.cpu().numpy(),
            poisson_normals,
            OUTPUT_DIR,
            "mesh_poisson_after_stage2.ply",
            opacities=poisson_opacity,
            scales=poisson_scale
        )

    # Extract mesh after Stage 2
    extract_and_save_mesh(
        gaussians, mesh_cameras, pipe_args, bg_color, sonar_config,
        sonar_scale_factor, OUTPUT_DIR, "mesh_after_stage2.ply",
        depth_trunc=depth_trunc, voxel_size=voxel_size, sdf_trunc=sdf_trunc,
        sonar_extrinsic=None
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

    epoch_indices = get_epoch_indices(len(training_frames), SEED)
    stage3_global_offset = training_iter_offset + STAGE1_ITERATIONS + STAGE2_ITERATIONS
    for iteration in range(1, STAGE3_ITERATIONS + 1):
        # Shuffle frames per epoch
        if (iteration - 1) % len(training_frames) == 0:
            epoch_seed = SEED + (iteration - 1) // len(training_frames)
            epoch_indices = get_epoch_indices(len(training_frames), epoch_seed)

        frame_idx = epoch_indices[(iteration - 1) % len(training_frames)]
        viewpoint_cam = training_frames[frame_idx]
        global_iter = stage3_global_offset + iteration
        sync_opacity_policy(global_iter, f"stage3/iter{iteration}")
        frame_visit_counts[frame_idx] += 1


        gt_image = preprocess_gt_image(viewpoint_cam.original_image)

        render_pkg = render_sonar(
            viewpoint_cam, gaussians, background,
            sonar_config=sonar_config,
            scale_factor=sonar_scale_factor,
            sonar_extrinsic=None,
            **SONAR_RENDER_KWARGS,
        )
        rendered = render_pkg["render"]

        Ll1 = l1_loss(rendered, gt_image)
        ssim_val = ssim(rendered, gt_image)
        base_loss = 0.8 * Ll1 + 0.2 * (1 - ssim_val)
        bright_loss = compute_bright_loss(rendered, gt_image)
        loss = (1 - BRIGHT_WEIGHT) * base_loss + BRIGHT_WEIGHT * bright_loss
        frame_loss_sums[frame_idx] += float(loss.item())
        frame_loss_counts[frame_idx] += 1

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
                    sync_opacity_policy(global_iter, f"stage3/fov_prune@{iteration}", force=True)
                    print(f"  [FOV prune] Removed {num_pruned} surfels outside FOV, {len(gaussians.get_xyz)} remaining")

        scale_value = sonar_scale_factor.get_scale_value()
        record_metrics(loss.item(), scale_value, "stage3")
        log_loss(
            metric_step,
            "stage3",
            Ll1.item(),
            ssim_val.item(),
            base_loss.item(),
            bright_loss.item(),
            loss.item(),
            scale_value,
            len(gaussians.get_xyz)
        )

        if iteration % 10 == 0 or iteration == 1:
            print(f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}")

    print(f"Stage 3 complete. Surfels: {len(gaussians.get_xyz)}")

    # Save comparison images before final prune so render quality is evaluated
    # on the actual post-training state (pruning is for mesh cleanup).
    save_comparison_images(training_frames, gaussians, background, sonar_config,
                           sonar_scale_factor, OUTPUT_DIR, "after_stage3")
    save_raw_comparison_images(training_frames, gaussians, background, sonar_config,
                               sonar_scale_factor, OUTPUT_DIR, "after_stage3",
                               DATASET_PATH, dataset_args.sonar_images)

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
        final_global_iter = training_iter_offset + STAGE1_ITERATIONS + STAGE2_ITERATIONS + STAGE3_ITERATIONS
        sync_opacity_policy(final_global_iter, "final_prune", force=True)
        print(f"  [Final prune] Removed {num_pruned} surfels, {len(gaussians.get_xyz)} remaining")

    # Save final surfel positions as point cloud (for verification)
    final_xyz = gaussians.get_xyz.detach().cpu().numpy()
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_xyz)
    final_pcd_path = os.path.join(OUTPUT_DIR, "surfels_after_training.ply")
    o3d.io.write_point_cloud(final_pcd_path, final_pcd)
    print(f"  Saved surfel positions: {final_pcd_path} ({len(final_xyz)} points)")

    if POISSON_MESH:
        poisson_points = gaussians.get_xyz.detach()
        poisson_normals = quaternion_to_normal(gaussians.get_rotation.detach()).cpu().numpy()
        poisson_opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)
        poisson_scale = gaussians.get_scaling.detach().cpu().numpy()
        save_poisson_mesh(
            poisson_points.cpu().numpy(),
            poisson_normals,
            OUTPUT_DIR,
            "mesh_poisson_after_stage3.ply",
            opacities=poisson_opacity,
            scales=poisson_scale
        )

    # Extract mesh after Stage 3
    extract_and_save_mesh(
        gaussians, mesh_cameras, pipe_args, bg_color, sonar_config,
        sonar_scale_factor, OUTPUT_DIR, "mesh_after_stage3.ply",
        depth_trunc=depth_trunc, voxel_size=voxel_size, sdf_trunc=sdf_trunc,
        sonar_extrinsic=None
    )

print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

summarize_training_frame_visits(
    training_frames,
    frame_visit_counts,
    frame_loss_sums,
    frame_loss_counts,
    OUTPUT_DIR,
)

train_eval = evaluate_frame_set(
    "train",
    training_frames,
    gaussians,
    background,
    sonar_config,
    sonar_scale_factor,
    OUTPUT_DIR,
)

holdout_eval = None
if holdout_frames:
    holdout_eval = evaluate_frame_set(
        "holdout",
        holdout_frames,
        gaussians,
        background,
        sonar_config,
        sonar_scale_factor,
        OUTPUT_DIR,
    )

if train_eval is not None and holdout_eval is not None:
    loss_gap = holdout_eval["loss_mean"] - train_eval["loss_mean"]
    ssim_gap = train_eval["ssim_mean"] - holdout_eval["ssim_mean"]
    print(
        f"[Holdout Gap] loss_gap={loss_gap:+.6f} (holdout-train), "
        f"ssim_gap={ssim_gap:+.4f} (train-holdout)"
    )
    if holdout_eval["loss_mean"] > train_eval["loss_mean"] * 1.25:
        print(
            "[Issue] Holdout loss is >25% above train loss; "
            "cross-view generalization remains weak under current settings."
        )

train_support = compute_multiview_support_metrics(
    gaussians,
    training_frames,
    sonar_config,
    sonar_scale_factor,
)
report_support_metrics("train", train_support, training_frames, OUTPUT_DIR)

if holdout_frames:
    combined_frames = training_frames + holdout_frames
    combined_support = compute_multiview_support_metrics(
        gaussians,
        combined_frames,
        sonar_config,
        sonar_scale_factor,
    )
    report_support_metrics("train_plus_holdout", combined_support, combined_frames, OUTPUT_DIR)

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

if SONAR_SAVE_CHECKPOINT:
    save_training_checkpoint(
        SONAR_SAVE_CHECKPOINT,
        gaussians,
        sonar_scale_factor,
        scale_optimizer,
        iteration=training_iter_offset + total_iters,
        stage_name="final",
        metadata={
            "dataset_key": DATASET_KEY,
            "seed": SEED,
            "elev_init_mode": ELEV_INIT_MODE,
            "sonar_fixed_opacity": int(SONAR_FIXED_OPACITY),
            "stage1_iterations": STAGE1_ITERATIONS,
            "stage2_iterations": STAGE2_ITERATIONS,
            "stage3_iterations": STAGE3_ITERATIONS,
        },
    )

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
print(f"  - frame_training_visits.csv             (Per-frame optimizer visit coverage)")
print(f"  - final_eval_train_frames.csv           (Per-frame final train losses)")
if holdout_frames:
    print(f"  - final_eval_holdout_frames.csv         (Per-frame final holdout losses)")
print(f"  - support_metrics_train.csv             (Surfel support diagnostics)")
if holdout_frames:
    print(f"  - support_metrics_train_plus_holdout.csv (Support with holdout views)")
