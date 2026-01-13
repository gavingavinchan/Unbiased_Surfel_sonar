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
import atexit
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

# =============================================================================
# Peak-Aware Loss Parameters
# =============================================================================
# Log compression: J = log(1 + alpha * I_normalized)
LOSS_ALPHA = 10.0           # Log compression factor
LOSS_DELTA = 1e-6           # Normalization stability

# Charbonnier robust penalty: rho(x) = sqrt(x^2 + eps^2)
LOSS_EPSILON = 1e-3

# Peak mask: m(x) = sigmoid((J_gt - tau) / s)
LOSS_SIGMOID_S = 0.08       # Sigmoid temperature
LOSS_TAU_PERCENTILE = 0.97  # Threshold percentile for peak detection (log-space)

# One-sided negative penalty on background overshoot
LOSS_LAMBDA_NEG = 0.01

# DoG (Difference of Gaussians) blob loss - tuned for 1-2px dots
LOSS_DOG_SIGMAS = [0.8, 1.6, 3.2]
LOSS_DOG_K = 1.6
LOSS_DOG_WEIGHT_EPS = 1e-12
LOSS_LAMBDA_BLOB = 0.5

# KL distribution matching for peak recall
LOSS_KL_GAMMA = 15.0        # Softmax temperature (higher = more top-k like)
LOSS_KL_ETA = 1e-12         # Numerical stability
LOSS_LAMBDA_KL = 0.1

# Peak-mass constraint
LOSS_LAMBDA_MASS = 0.1

# Peak-support EMA (for pruning)
PEAK_SUPPORT_EMA = 0.02
PEAK_SUPPORT_MIN = 0.05
PEAK_SUPPORT_GRAD_MIN = 1e-4
PEAK_SUPPORT_UPDATE_INTERVAL = 10

# Logging

LOSS_SMOOTH_WINDOW = 200
LOSS_LOG_FLUSH_INTERVAL = 1


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


# =============================================================================
# Peak-Aware Loss Functions
# =============================================================================

def gaussian_blur_2d(x, sigma):
    """Apply 2D Gaussian blur to tensor [C, H, W] or [H, W]."""
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        squeeze_back = True
    elif x.dim() == 3:
        x = x.unsqueeze(0)  # [1, C, H, W]
        squeeze_back = False
    else:
        squeeze_back = False

    # Kernel size = 6*sigma rounded up to odd
    ksize = int(np.ceil(sigma * 6))
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)

    # Create 1D Gaussian kernel
    coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - ksize // 2
    kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable convolution
    kernel_h = kernel_1d.view(1, 1, 1, -1)
    kernel_v = kernel_1d.view(1, 1, -1, 1)

    # Pad and convolve
    pad_h = ksize // 2
    pad_v = ksize // 2

    C = x.shape[1]
    # Expand kernel for all channels
    kernel_h = kernel_h.expand(C, 1, 1, -1)
    kernel_v = kernel_v.expand(C, 1, -1, 1)

    x = torch.nn.functional.pad(x, (pad_h, pad_h, 0, 0), mode='reflect')
    x = torch.nn.functional.conv2d(x, kernel_h, groups=C)
    x = torch.nn.functional.pad(x, (0, 0, pad_v, pad_v), mode='reflect')
    x = torch.nn.functional.conv2d(x, kernel_v, groups=C)

    if squeeze_back:
        x = x.squeeze(0).squeeze(0)
    else:
        x = x.squeeze(0)

    return x


def compute_dog(J, sigma, k=LOSS_DOG_K):
    """Compute Difference of Gaussians at scale sigma."""
    G_sigma = gaussian_blur_2d(J, sigma)
    G_k_sigma = gaussian_blur_2d(J, k * sigma)
    return G_sigma - G_k_sigma


def compute_peak_aware_loss(rendered, gt_image, iteration, total_iterations):
    """
    Peak-aware loss with peak-only photometric term, weak background overshoot,
    DoG blob matching, masked KL, and peak-mass constraint.

    Args:
        rendered: [C, H, W] predicted image (0-1 range)
        gt_image: [C, H, W] ground truth image (0-1 range)
        iteration: current training iteration
        total_iterations: total iterations (unused, kept for call compatibility)

    Returns:
        total_loss, dict of individual loss components
    """
    # Work with grayscale (mean over channels)
    I_gt = gt_image.mean(dim=0)      # [H, W]
    I_pred = rendered.mean(dim=0)    # [H, W]

    # Per-frame normalization (detach quantile computation)
    with torch.no_grad():
        c = torch.quantile(I_gt, 0.99)

    I_gt_n = I_gt / (c + LOSS_DELTA)
    I_pred_n = I_pred / (c + LOSS_DELTA)

    # Log compression
    J_gt = torch.log1p(LOSS_ALPHA * I_gt_n)
    J_pred = torch.log1p(LOSS_ALPHA * I_pred_n)

    # Peak mask in log space (detached)
    with torch.no_grad():
        tau = torch.quantile(J_gt, LOSS_TAU_PERCENTILE)
        m = torch.sigmoid((J_gt - tau) / LOSS_SIGMOID_S)

    # Charbonnier penalty
    diff = J_pred - J_gt
    rho = torch.sqrt(diff**2 + LOSS_EPSILON**2)

    # Peak-only positive term
    L_pos = (m * rho).mean()

    # Very weak one-sided negative (overshoot only)
    relu_diff = torch.relu(diff)
    rho_neg = torch.sqrt(relu_diff**2 + LOSS_EPSILON**2)
    L_neg = ((1 - m) * rho_neg).mean()

    # DoG multi-scale blob loss (weighted by GT bandpass energy)
    L_blob_list = []
    for sigma in LOSS_DOG_SIGMAS:
        dog_gt = compute_dog(J_gt, sigma)
        dog_pred = compute_dog(J_pred, sigma)
        with torch.no_grad():
            denom = torch.quantile(dog_gt.abs(), 0.99) + LOSS_DOG_WEIGHT_EPS
            weight = torch.clamp(dog_gt.abs() / denom, 0.0, 1.0)
        L_blob_list.append((weight * (dog_pred - dog_gt).abs()).mean())
    L_blob = sum(L_blob_list) if L_blob_list else torch.zeros(1, device=rendered.device).squeeze()

    # Masked peak distribution KL
    with torch.no_grad():
        mask_flat = m.flatten()
        valid_region = mask_flat > 0.01
        num_valid = valid_region.sum().item()

    if num_valid < 10:
        L_KL = torch.zeros(1, device=rendered.device).squeeze()
    else:
        log_m = torch.log(mask_flat[valid_region].clamp(min=1e-10))
        logits_gt = LOSS_KL_GAMMA * J_gt.flatten()[valid_region] + log_m
        logits_pred = LOSS_KL_GAMMA * J_pred.flatten()[valid_region] + log_m

        with torch.no_grad():
            log_denom_gt = torch.logaddexp(
                torch.logsumexp(logits_gt, dim=0),
                torch.log(torch.tensor(LOSS_KL_ETA, device=logits_gt.device, dtype=logits_gt.dtype))
            )
            log_p = logits_gt - log_denom_gt
            p = torch.exp(log_p)

        log_denom_pred = torch.logaddexp(
            torch.logsumexp(logits_pred, dim=0),
            torch.log(torch.tensor(LOSS_KL_ETA, device=logits_pred.device, dtype=logits_pred.dtype))
        )
        log_q = logits_pred - log_denom_pred

        L_KL = (p * (log_p - log_q)).sum()

    # Peak-mass constraint
    M_gt = (m * J_gt).sum()
    M_pred = (m * J_pred).sum()
    L_mass = torch.abs(M_pred - M_gt)

    total_loss = (
        L_pos
        + LOSS_LAMBDA_NEG * L_neg
        + LOSS_LAMBDA_BLOB * L_blob
        + LOSS_LAMBDA_KL * L_KL
        + LOSS_LAMBDA_MASS * L_mass
    )

    mask_frac = (m > 0.5).float().mean()

    return total_loss, {
        'L_pos': L_pos.item(),
        'L_neg': L_neg.item(),
        'L_blob': L_blob.item(),
        'L_KL': L_KL.item(),
        'L_mass': L_mass.item(),
        'tau': tau.item(),
        'mask_frac': mask_frac.item(),
        'M_gt': M_gt.item(),
        'M_pred': M_pred.item()
    }, m


def project_sonar_pixels(gaussians, camera, sonar_config, scale_factor):
    w2c = camera.world_view_transform.cuda()
    R_w2v = w2c[:3, :3]
    t_w2v = w2c[3, :3]
    t_w2v_scaled = scale_factor.scale * t_w2v

    points_sonar = (gaussians.get_xyz @ R_w2v.T) + t_w2v_scaled
    right = points_sonar[:, 0]
    down = points_sonar[:, 1]
    forward = points_sonar[:, 2]

    azimuth = -torch.atan2(right, forward)
    range_vals = torch.sqrt(right**2 + down**2 + forward**2)
    elevation = torch.atan2(down, torch.sqrt(right**2 + forward**2))

    valid_azimuth = torch.abs(azimuth) <= sonar_config.half_azimuth_rad
    valid_elevation = torch.abs(elevation) <= sonar_config.half_elevation_rad
    valid_range = (range_vals >= sonar_config.range_min) & (range_vals <= sonar_config.range_max)
    center_in_fov = valid_azimuth & valid_elevation & valid_range & (forward > 0)

    scaling = gaussians.get_scaling
    surfel_radius = scaling.max(dim=1).values
    margin = compute_fov_margin_debug(range_vals, azimuth, elevation, sonar_config)
    in_fov = center_in_fov & (margin > surfel_radius)

    H = camera.image_height
    W = camera.image_width
    col = (-azimuth / sonar_config.half_azimuth_rad + 1) * (W / 2)
    row = (range_vals - sonar_config.range_min) / (sonar_config.range_max - sonar_config.range_min) * H

    col = torch.clamp(col, 0, W - 1)
    row = torch.clamp(row, 0, H - 1)

    return col, row, in_fov


def bilinear_sample_mask(mask, col, row):
    H, W = mask.shape
    col_floor = col.floor().long()
    row_floor = row.floor().long()
    col_ceil = (col_floor + 1).clamp(max=W - 1)
    row_ceil = (row_floor + 1).clamp(max=H - 1)

    col_frac = col - col_floor.float()
    row_frac = row - row_floor.float()

    w00 = (1 - col_frac) * (1 - row_frac)
    w01 = (1 - col_frac) * row_frac
    w10 = col_frac * (1 - row_frac)
    w11 = col_frac * row_frac

    v00 = mask[row_floor, col_floor]
    v01 = mask[row_ceil, col_floor]
    v10 = mask[row_floor, col_ceil]
    v11 = mask[row_ceil, col_ceil]

    return w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11


@torch.no_grad()
def update_peak_support(gaussians, camera, sonar_config, scale_factor, peak_mask):
    if not hasattr(gaussians, "peak_support"):
        gaussians.peak_support = torch.zeros(len(gaussians.get_xyz), device=peak_mask.device)

    col, row, in_fov = project_sonar_pixels(gaussians, camera, sonar_config, scale_factor)
    sampled = bilinear_sample_mask(peak_mask, col, row)

    update_mask = in_fov
    if update_mask.any():
        updated = (1 - PEAK_SUPPORT_EMA) * gaussians.peak_support + PEAK_SUPPORT_EMA * sampled
        gaussians.peak_support = torch.where(update_mask, updated, gaussians.peak_support)


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
    LOSS_LOG_HANDLE = open(LOSS_LOG_PATH, "w", buffering=1)
    LOSS_LOG_HANDLE.write(
        "iter,stage,L_pos,L_neg,L_blob,L_KL,L_mass,total_loss,mask_frac,M_gt,M_pred,scale,num_points\n"
    )
    LOSS_LOG_HANDLE.flush()


def log_loss(iteration, stage_name, loss_components, total_loss, scale_value, num_points):
    if LOSS_LOG_HANDLE is None:
        return

    LOSS_LOG_HANDLE.write(
        f"{iteration},{stage_name},{loss_components['L_pos']:.6f},{loss_components['L_neg']:.6f},"
        f"{loss_components['L_blob']:.6f},{loss_components['L_KL']:.6f},{loss_components['L_mass']:.6f},"
        f"{total_loss:.6f},{loss_components['mask_frac']:.6f},{loss_components['M_gt']:.6f},"
        f"{loss_components['M_pred']:.6f},{scale_value:.6f},{num_points}\n"
    )

    if LOSS_LOG_FLUSH_INTERVAL > 0 and iteration % LOSS_LOG_FLUSH_INTERVAL == 0:
        LOSS_LOG_HANDLE.flush()


def update_opacity_lr(gaussians, new_lr):
    for group in gaussians.optimizer.param_groups:
        if group.get("name") == "opacity":
            group["lr"] = new_lr


def print_collapse_stats(global_iter, loss_components, gaussians):
    with torch.no_grad():
        opacity = gaussians.get_opacity.squeeze()
        mean_opacity = opacity.mean().item()
        median_opacity = opacity.median().item()
        frac_low = (opacity < OPACITY_LOW_THRESHOLD).float().mean().item()
        mask_frac = loss_components.get("mask_frac", 0.0)
        mass_gt = loss_components.get("M_gt", 0.0)
        mass_pred = loss_components.get("M_pred", 0.0)

    print(
        f"  [Stats] Iter {global_iter}: opacity mean={mean_opacity:.4e}, median={median_opacity:.4e}, "
        f"frac<1e-3={frac_low:.3f}, M_gt={mass_gt:.4f}, M_pred={mass_pred:.4f}, "
        f"mask_frac={mask_frac:.3f}"
    )


def close_logs():
    if LOSS_LOG_HANDLE is not None:
        LOSS_LOG_HANDLE.flush()
        LOSS_LOG_HANDLE.close()
    if LOG_FILE is not None:
        LOG_FILE.flush()
        LOG_FILE.close()


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
DATASET_PATH = "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs"
OUTPUT_DIR_BASE = "./output/debug_multiframe"
NUM_TRAINING_FRAMES = 500  # Number of frames to use for training
PYRAMID_DEPTH = 0.5

# Curriculum learning parameters
STAGE1_ITERATIONS = 0   # Learn scale only (surfels frozen) - DISABLED, using known scale=0.65
STAGE2_ITERATIONS = 30000  # Learn surfels only (scale frozen)
STAGE3_ITERATIONS = 1   # Joint fine-tuning

# Stabilization phase to avoid collapse (disable prune/reset early)
STABILIZATION_ITERS = 5000  # 5k-10k recommended
OPACITY_LR_STABLE = 5e-3
OPACITY_LR_AFTER = 5e-2
OPACITY_RESET_INTERVAL = 1_000_000  # effectively off during debug

# FOV-aware pruning: remove surfels that drift outside all training cameras' FOV
FOV_PRUNE_INTERVAL = 100  # Prune every N iterations (0 to disable)
FOV_PRUNE_START = STABILIZATION_ITERS

# Opacity prune warmup (delay opacity pruning after stabilization)
OPACITY_PRUNE_WARMUP = 1000
OPACITY_PRUNE_START = FOV_PRUNE_START + OPACITY_PRUNE_WARMUP

# Debug stats
DEBUG_STATS_INTERVAL = 500
OPACITY_LOW_THRESHOLD = 1e-3

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

gaussians.peak_support = torch.zeros(len(gaussians.get_xyz), device="cuda")
gaussians.peak_support_min = None
gaussians.peak_support_grad_min = PEAK_SUPPORT_GRAD_MIN

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
    opacity_lr=OPACITY_LR_STABLE,
    scaling_lr=0.005,
    rotation_lr=0.001,
    percent_dense=0.01,
    lambda_dssim=0.2,
    densification_interval=100,
    opacity_reset_interval=OPACITY_RESET_INTERVAL,
    densify_from_iter=STABILIZATION_ITERS,
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

    epoch_indices = get_epoch_indices(len(training_frames), SEED)
    for iteration in range(1, STAGE1_ITERATIONS + 1):
        # Shuffle frames per epoch
        if (iteration - 1) % len(training_frames) == 0:
            epoch_seed = SEED + (iteration - 1) // len(training_frames)
            epoch_indices = get_epoch_indices(len(training_frames), epoch_seed)

        frame_idx = epoch_indices[(iteration - 1) % len(training_frames)]
        viewpoint_cam = training_frames[frame_idx]
        global_iter = iteration

        if global_iter == STABILIZATION_ITERS + 1:
            update_opacity_lr(gaussians, OPACITY_LR_AFTER)
            print(f"  [Stabilization] Opacity LR set to {OPACITY_LR_AFTER}")

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

        # Compute loss (peak-aware)
        total_iter_so_far = iteration  # Stage 1 iteration count
        loss, loss_components, peak_mask = compute_peak_aware_loss(
            rendered, gt_image, total_iter_so_far, STAGE1_ITERATIONS + STAGE2_ITERATIONS
        )

        if iteration == 1:
            print(f"    loss.requires_grad: {loss.requires_grad}")
            print(f"    loss.grad_fn: {loss.grad_fn}")
            print(
                f"    L_pos={loss_components['L_pos']:.4f}, L_neg={loss_components['L_neg']:.4f}, "
                f"L_blob={loss_components['L_blob']:.4f}, L_KL={loss_components['L_KL']:.4f}, "
                f"L_mass={loss_components['L_mass']:.4f}\n"
            )

        if PEAK_SUPPORT_UPDATE_INTERVAL > 0 and global_iter % PEAK_SUPPORT_UPDATE_INTERVAL == 0:
            update_peak_support(gaussians, viewpoint_cam, sonar_config, sonar_scale_factor, peak_mask.detach())

        # Backward - only scale factor gets gradients (surfels frozen)
        # Skip if no gradients (no visible surfels)
        if loss.requires_grad:
            loss.backward()
        else:
            if iteration % 100 == 0:
                print(f"  [Warning] Iter {iteration}: No visible surfels for frame, skipping backward")

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
        log_loss(metric_step, "stage1", loss_components, loss.item(), scale_value, len(gaussians.get_xyz))

        # Periodic memory cleanup
        if iteration % 100 == 0:
            torch.cuda.empty_cache()

        if DEBUG_STATS_INTERVAL > 0 and global_iter % DEBUG_STATS_INTERVAL == 0:
            print_collapse_stats(global_iter, loss_components, gaussians)

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"  Iter {iteration:3d}: L_pos={loss_components['L_pos']:.4f}, L_neg={loss_components['L_neg']:.4f}, "
                f"L_blob={loss_components['L_blob']:.4f}, L_KL={loss_components['L_KL']:.4f}, "
                f"L_mass={loss_components['L_mass']:.4f}, scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}"
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

    sonar_scale_factor._log_scale.requires_grad = False
    frozen_scale = sonar_scale_factor.get_scale_value()
    print(f"Scale factor frozen at: {frozen_scale:.6f}")

    epoch_indices = get_epoch_indices(len(training_frames), SEED)
    for iteration in range(1, STAGE2_ITERATIONS + 1):
        # Shuffle frames per epoch
        if (iteration - 1) % len(training_frames) == 0:
            epoch_seed = SEED + (iteration - 1) // len(training_frames)
            epoch_indices = get_epoch_indices(len(training_frames), epoch_seed)

        frame_idx = epoch_indices[(iteration - 1) % len(training_frames)]
        viewpoint_cam = training_frames[frame_idx]
        global_iter = STAGE1_ITERATIONS + iteration

        if global_iter == STABILIZATION_ITERS + 1:
            update_opacity_lr(gaussians, OPACITY_LR_AFTER)
            print(f"  [Stabilization] Opacity LR set to {OPACITY_LR_AFTER}")

        if global_iter == OPACITY_PRUNE_START:
            gaussians.peak_support_min = PEAK_SUPPORT_MIN
            print(f"  [Stabilization] Peak-support opacity pruning enabled at S_min={PEAK_SUPPORT_MIN}")

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

        # Compute loss (peak-aware)
        total_iter_so_far = STAGE1_ITERATIONS + iteration
        loss, loss_components, peak_mask = compute_peak_aware_loss(
            rendered, gt_image, total_iter_so_far, STAGE1_ITERATIONS + STAGE2_ITERATIONS
        )

        if PEAK_SUPPORT_UPDATE_INTERVAL > 0 and global_iter % PEAK_SUPPORT_UPDATE_INTERVAL == 0:
            update_peak_support(gaussians, viewpoint_cam, sonar_config, sonar_scale_factor, peak_mask.detach())

        # Backward - skip if no gradients (no visible surfels for this frame)
        if loss.requires_grad:
            loss.backward()
        else:
            if iteration % 100 == 0:
                print(f"  [Warning] Iter {iteration}: No visible surfels for frame, skipping backward")

        # Update surfels only
        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.update_learning_rate(iteration)

            # FOV-aware pruning after stabilization
            if global_iter >= FOV_PRUNE_START and FOV_PRUNE_INTERVAL > 0 and global_iter % FOV_PRUNE_INTERVAL == 0:
                num_pruned = prune_outside_fov(gaussians, training_frames, sonar_config, sonar_scale_factor)
                if num_pruned > 0:
                    print(f"  [FOV prune] Removed {num_pruned} surfels outside FOV, {len(gaussians.get_xyz)} remaining")

        scale_value = sonar_scale_factor.get_scale_value()
        record_metrics(loss.item(), scale_value, "stage2")
        log_loss(metric_step, "stage2", loss_components, loss.item(), scale_value, len(gaussians.get_xyz))

        # Periodic memory cleanup
        if iteration % 100 == 0:
            torch.cuda.empty_cache()

        if DEBUG_STATS_INTERVAL > 0 and global_iter % DEBUG_STATS_INTERVAL == 0:
            print_collapse_stats(global_iter, loss_components, gaussians)

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"  Iter {iteration:3d}: L_pos={loss_components['L_pos']:.4f}, L_neg={loss_components['L_neg']:.4f}, "
                f"L_blob={loss_components['L_blob']:.4f}, L_KL={loss_components['L_KL']:.4f}, "
                f"L_mass={loss_components['L_mass']:.4f}, scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}"
            )

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

    epoch_indices = get_epoch_indices(len(training_frames), SEED)
    for iteration in range(1, STAGE3_ITERATIONS + 1):
        # Shuffle frames per epoch
        if (iteration - 1) % len(training_frames) == 0:
            epoch_seed = SEED + (iteration - 1) // len(training_frames)
            epoch_indices = get_epoch_indices(len(training_frames), epoch_seed)

        frame_idx = epoch_indices[(iteration - 1) % len(training_frames)]
        viewpoint_cam = training_frames[frame_idx]
        global_iter = STAGE1_ITERATIONS + STAGE2_ITERATIONS + iteration

        if global_iter == OPACITY_PRUNE_START and gaussians.peak_support_min is None:
            gaussians.peak_support_min = PEAK_SUPPORT_MIN
            print(f"  [Stabilization] Peak-support opacity pruning enabled at S_min={PEAK_SUPPORT_MIN}")

        gt_image = preprocess_gt_image(viewpoint_cam.original_image)

        render_pkg = render_sonar(
            viewpoint_cam, gaussians, background,
            sonar_config=sonar_config,
            scale_factor=sonar_scale_factor,
            sonar_extrinsic=None
        )
        rendered = render_pkg["render"]

        # Compute loss (peak-aware)
        total_iter_so_far = STAGE1_ITERATIONS + STAGE2_ITERATIONS + iteration
        loss, loss_components, peak_mask = compute_peak_aware_loss(
            rendered, gt_image, total_iter_so_far, STAGE1_ITERATIONS + STAGE2_ITERATIONS + STAGE3_ITERATIONS
        )

        if PEAK_SUPPORT_UPDATE_INTERVAL > 0 and global_iter % PEAK_SUPPORT_UPDATE_INTERVAL == 0:
            update_peak_support(gaussians, viewpoint_cam, sonar_config, sonar_scale_factor, peak_mask.detach())

        # Backward - skip if no gradients (no visible surfels)
        if loss.requires_grad:
            loss.backward()
        else:
            if iteration % 100 == 0:
                print(f"  [Warning] Iter {iteration}: No visible surfels for frame, skipping backward")

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            # Scale frozen - no optimizer step
            gaussians.update_learning_rate(STAGE2_ITERATIONS + iteration)

            # FOV-aware pruning after stabilization
            if global_iter >= FOV_PRUNE_START and FOV_PRUNE_INTERVAL > 0 and global_iter % FOV_PRUNE_INTERVAL == 0:
                num_pruned = prune_outside_fov(gaussians, training_frames, sonar_config, sonar_scale_factor)
                if num_pruned > 0:
                    print(f"  [FOV prune] Removed {num_pruned} surfels outside FOV, {len(gaussians.get_xyz)} remaining")

        scale_value = sonar_scale_factor.get_scale_value()
        record_metrics(loss.item(), scale_value, "stage3")
        log_loss(metric_step, "stage3", loss_components, loss.item(), scale_value, len(gaussians.get_xyz))

        # Periodic memory cleanup
        if iteration % 100 == 0:
            torch.cuda.empty_cache()

        if DEBUG_STATS_INTERVAL > 0 and global_iter % DEBUG_STATS_INTERVAL == 0:
            print_collapse_stats(global_iter, loss_components, gaussians)

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"  Iter {iteration:3d}: L_pos={loss_components['L_pos']:.4f}, L_neg={loss_components['L_neg']:.4f}, "
                f"L_blob={loss_components['L_blob']:.4f}, L_KL={loss_components['L_KL']:.4f}, "
                f"L_mass={loss_components['L_mass']:.4f}, scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}"
            )

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
