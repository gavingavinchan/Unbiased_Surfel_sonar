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
import subprocess
from dataclasses import dataclass
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
from gaussian_renderer import render_sonar, render, quaternion_to_normal, sonar_project_points
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
from utils.elevation_stage1_helpers import (
    CHECKPOINT_SCHEMA_VERSION,
    RoundRobinSamplerState,
    assert_frame_keys_unique,
    compute_frame_stats,
    combine_pose_overlap_score,
    compute_active_frame_fingerprint,
    normalize_by_percentiles,
    pose_only_hard_gate,
    rank_overlap_candidates,
    resolve_effective_stage1_mode,
    resolve_temperatures,
    round_robin_sample,
    run_stage1_likelihood_step,
    resolve_stage1_resume_action,
    should_refresh_pixel_bank,
    remap_or_reset_pixel_logits,
    optimizer_rebuild_required,
)
from utils.elevation_chunk4_helpers import (
    CHECKPOINT_SCHEMA_VERSION as CHUNK4_CHECKPOINT_SCHEMA_VERSION,
    resolve_effective_chunk4_modes,
    mode_enables_weighted_coupling,
    mode_enables_hard_prune,
    associate_expected_points_to_surfels,
    reduce_coupling_loss,
    initialize_persistent_surfel_state,
    apply_densify_to_surfel_state,
    apply_prune_reorder_to_surfel_state,
    assert_surfel_id_integrity,
    update_support_buffers_by_id,
    compute_support_failure_mask,
    apply_prune_hysteresis,
    apply_new_surfel_grace,
    build_chunk4_checkpoint_payload,
    resolve_chunk4_resume_action,
)
from utils.visualization_utils import (
    build_frame_stem,
    build_surfel_glyph_mesh,
    compute_frame_surfel_membership,
    create_pose_wireframe,
    deterministic_glyph_indices,
    diagnostic_colors_from_metrics,
    ensure_visualizer_dirs,
    prepare_surfel_visualization_state,
    select_frame_surfel_indices,
    write_line_set,
    write_point_cloud,
    write_triangle_mesh,
    write_visualizer_manifest,
)
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


def prune_outside_fov(
    gaussians,
    training_frames,
    sonar_config,
    scale_factor,
    require_all=False,
    check_size=True,
    chunk4_runtime_state=None,
    chunk4_cfg=None,
    reason="fov",
    current_iter=0,
):
    """
    Prune Gaussians that are outside the FOV of training cameras.

    Args:
        gaussians: GaussianModel instance
        training_frames: List of camera objects
        sonar_config: SonarConfig
        scale_factor: SonarScaleFactor
        require_all: If True, keep only points visible from ALL training cameras
                     If False, keep points visible from ANY training camera (default)
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
        # Strict mode: points must be visible from every training camera
        visible_from_all = all_masks.all(dim=0)  # [N]
        prune_mask = ~visible_from_all
    else:
        # Conservative mode: keep if visible from at least one camera
        visible_from_any = all_masks.any(dim=0)  # [N]
        prune_mask = ~visible_from_any

    num_to_prune = prune_mask.sum().item()

    if num_to_prune > 0:
        if chunk4_runtime_state is not None:
            cfg = chunk4_cfg if chunk4_cfg is not None else ELEV_CHUNK4_CFG
            ensure_chunk4_state_capacity(
                chunk4_runtime_state,
                gaussians,
                cfg,
                current_iter=current_iter,
            )
            num_to_prune = apply_row_prune_with_chunk4_state(
                gaussians,
                chunk4_runtime_state,
                prune_mask,
                cfg=cfg,
                reason=reason,
            )
        else:
            gaussians.prune_points(prune_mask)

    return int(num_to_prune)


def create_pose_pyramid_wireframe(position, rotation_matrix, depth=0.5,
                                   azimuth_fov=120.0, elevation_fov=20.0, color=[1.0, 0.0, 0.0], mode="near"):
    """Create a wireframe pyramid for a single pose."""
    return create_pose_wireframe(
        position,
        rotation_matrix,
        depth=depth,
        azimuth_fov=azimuth_fov,
        elevation_fov=elevation_fov,
        color=color,
        mode=mode,
    )


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


def build_pose_overlap_table(training_frames, overlap_cfg):
    frame_keys = [str(cam.image_name) for cam in training_frames]
    assert_frame_keys_unique(frame_keys)

    positions = {}
    yaws_deg = {}
    for cam in training_frames:
        frame_key = str(cam.image_name)
        r_w2c = np.asarray(cam.R, dtype=np.float64)
        t_w2c = np.asarray(cam.T, dtype=np.float64)
        r_c2w = r_w2c.T

        position = -r_c2w @ t_w2c
        forward = r_c2w[:, 2]
        forward_xz = np.array([forward[0], forward[2]], dtype=np.float64)
        forward_xz_norm = np.linalg.norm(forward_xz)
        if forward_xz_norm > 1e-12:
            yaw_deg = math.degrees(math.atan2(forward_xz[0], forward_xz[1]))
        else:
            yaw_deg = 0.0

        positions[frame_key] = position
        yaws_deg[frame_key] = float(yaw_deg)

    baseline_eps = max(float(overlap_cfg.overlap_min_baseline), 1e-8)
    max_yaw_deg = float(overlap_cfg.overlap_max_yaw_deg)
    overlap_table = {}

    for key_a in frame_keys:
        pos_a = positions[key_a]
        yaw_a = yaws_deg[key_a]

        candidates = []
        pair_stats = {}
        for key_b in frame_keys:
            if key_b == key_a:
                continue

            pos_b = positions[key_b]
            yaw_b = yaws_deg[key_b]
            baseline_m = float(np.linalg.norm(pos_a - pos_b))

            yaw_delta = (yaw_a - yaw_b + 180.0) % 360.0 - 180.0
            yaw_deg = abs(float(yaw_delta))

            gate_ok = pose_only_hard_gate(
                baseline_m=baseline_m,
                yaw_deg=yaw_deg,
                min_baseline_m=overlap_cfg.overlap_min_baseline,
                max_yaw_deg=overlap_cfg.overlap_max_yaw_deg,
            )
            if max_yaw_deg > 0.0:
                yaw_score = max(0.0, 1.0 - (yaw_deg / max_yaw_deg))
            else:
                yaw_score = 1.0 if yaw_deg <= 1e-8 else 0.0
            baseline_score = baseline_m / (baseline_m + baseline_eps)
            score = combine_pose_overlap_score(
                yaw_score=yaw_score,
                baseline_score=baseline_score,
                w_yaw=overlap_cfg.overlap_score_w_yaw,
                w_base=overlap_cfg.overlap_score_w_base,
            )

            if gate_ok and score >= float(overlap_cfg.overlap_min_score):
                candidates.append((key_b, float(score)))
                pair_stats[key_b] = {
                    "baseline_m": baseline_m,
                    "yaw_deg": yaw_deg,
                }

        ranked = rank_overlap_candidates(candidates, overlap_cfg.overlap_topk_build)
        ranked_records = []
        for key_b, score in ranked:
            stats = pair_stats.get(key_b, {})
            ranked_records.append(
                {
                    "frame_key": key_b,
                    "score": float(score),
                    "baseline_m": float(stats.get("baseline_m", 0.0)),
                    "yaw_deg": float(stats.get("yaw_deg", 0.0)),
                }
            )

        overlap_table[key_a] = {
            "ranked": ranked_records,
            "topk_use": [entry["frame_key"] for entry in ranked_records[: overlap_cfg.overlap_topk_use]],
            "num_candidates": int(len(candidates)),
        }

    build_params = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "score_mode": overlap_cfg.overlap_score_mode,
        "topk_build": int(overlap_cfg.overlap_topk_build),
        "topk_use": int(overlap_cfg.overlap_topk_use),
        "min_baseline_m": float(overlap_cfg.overlap_min_baseline),
        "max_yaw_deg": float(overlap_cfg.overlap_max_yaw_deg),
        "min_score": float(overlap_cfg.overlap_min_score),
        "w_yaw": float(overlap_cfg.overlap_score_w_yaw),
        "w_base": float(overlap_cfg.overlap_score_w_base),
    }
    return overlap_table, build_params


def summarize_overlap_table(overlap_table, frame_keys, label):
    if not overlap_table:
        print(f"[Elevation Stage 1] overlap_table skipped ({label})")
        return
    counts = [len(overlap_table.get(k, {}).get("topk_use", [])) for k in frame_keys]
    min_neighbors = min(counts) if counts else 0
    max_neighbors = max(counts) if counts else 0
    mean_neighbors = (sum(counts) / len(counts)) if counts else 0.0
    print(
        f"[Elevation Stage 1] overlap_table ({label}): frames={len(frame_keys)}, "
        f"neighbors min/mean/max={min_neighbors}/{mean_neighbors:.2f}/{max_neighbors}"
    )


def serialize_sampler_state(sampler_state):
    return {
        "frame_keys": list(sampler_state.frame_keys),
        "cursor": int(sampler_state.cursor),
        "epoch": int(sampler_state.epoch),
    }


def restore_sampler_state(payload, active_frame_keys):
    if not isinstance(payload, dict):
        return RoundRobinSamplerState(frame_keys=list(active_frame_keys), cursor=0, epoch=0)

    loaded_keys = [str(key) for key in payload.get("frame_keys", [])]
    if loaded_keys and loaded_keys != list(active_frame_keys):
        raise ValueError("sampler frame_keys mismatch")

    cursor = max(0, int(payload.get("cursor", 0)))
    epoch = max(0, int(payload.get("epoch", 0)))
    return RoundRobinSamplerState(frame_keys=list(active_frame_keys), cursor=cursor, epoch=epoch)


def sample_gt(frame_gray, row, col):
    h, w = frame_gray.shape
    row_f = row.to(dtype=torch.float32)
    col_f = col.to(dtype=torch.float32)

    valid = (
        (row_f >= 0.0)
        & (row_f <= float(h - 1))
        & (col_f >= 0.0)
        & (col_f <= float(w - 1))
    )

    row0 = torch.floor(row_f).clamp(0, h - 1).to(dtype=torch.long)
    col0 = torch.floor(col_f).clamp(0, w - 1).to(dtype=torch.long)
    row1 = (row0 + 1).clamp(0, h - 1)
    col1 = (col0 + 1).clamp(0, w - 1)

    dr = (row_f - row0.to(dtype=torch.float32)).clamp(0.0, 1.0)
    dc = (col_f - col0.to(dtype=torch.float32)).clamp(0.0, 1.0)

    v00 = frame_gray[row0, col0]
    v01 = frame_gray[row0, col1]
    v10 = frame_gray[row1, col0]
    v11 = frame_gray[row1, col1]
    sampled = (
        (1.0 - dr) * (1.0 - dc) * v00
        + (1.0 - dr) * dc * v01
        + dr * (1.0 - dc) * v10
        + dr * dc * v11
    )
    return sampled, valid


def resolve_stage1_overlap_neighbors(frame_key, overlap_table, topk_use):
    entry = overlap_table.get(str(frame_key), {}) if isinstance(overlap_table, dict) else {}
    raw_neighbors = entry.get("topk_use", []) if isinstance(entry, dict) else []
    neighbors = []
    for key in raw_neighbors:
        key_str = str(key)
        if key_str not in neighbors:
            neighbors.append(key_str)
    limit = max(0, int(topk_use))
    if limit > 0:
        neighbors = neighbors[:limit]
    return neighbors


def build_stage1_multiview_loglik(
    *,
    frame_idx,
    frame_key,
    training_frames,
    frame_key_to_index,
    overlap_table,
    pixel_bank,
    gt_frame_cache,
    frame_stats_cache,
    elev_angle_bins,
    sonar_config,
    sonar_scale_factor,
    cfg,
):
    if str(cfg.lik_invalid_mode) != "neutral":
        raise ValueError(f"Unsupported ELEV_LIK_INVALID_MODE: {cfg.lik_invalid_mode}")

    bank_entry = pixel_bank[str(frame_key)]
    rows = bank_entry["rows"]
    cols = bank_entry["cols"]
    p = int(rows.shape[0])
    k = int(elev_angle_bins.shape[0])
    device = rows.device

    loglik = torch.full((p, k), float(cfg.lik_log_floor), dtype=torch.float32, device=device)
    support_mask = torch.zeros((p, k), dtype=torch.bool, device=device)
    if p <= 0 or k <= 0:
        return loglik, support_mask

    pts_bins = back_project_bins(
        frame_idx=int(frame_idx),
        rows=rows,
        cols=cols,
        elev_bins=elev_angle_bins,
        cameras=training_frames,
        sonar_config=sonar_config,
        scale_factor=sonar_scale_factor,
    )
    pts_flat = pts_bins.reshape(-1, 3)

    loglik_sum = torch.zeros_like(loglik)
    valid_w = torch.zeros_like(loglik)
    neighbors = resolve_stage1_overlap_neighbors(
        frame_key=frame_key,
        overlap_table=overlap_table,
        topk_use=cfg.overlap_topk_use,
    )

    for neighbor_key in neighbors:
        neighbor_idx = frame_key_to_index.get(neighbor_key)
        if neighbor_idx is None:
            continue
        if neighbor_key not in gt_frame_cache:
            continue
        if neighbor_key not in frame_stats_cache:
            continue

        neighbor_cam = training_frames[int(neighbor_idx)]
        proj = sonar_project_points(
            pts_flat,
            neighbor_cam,
            sonar_config,
            scale_factor=sonar_scale_factor,
        )

        sampled_i, sampled_valid = sample_gt(
            gt_frame_cache[neighbor_key]["gt_gray"],
            proj.row,
            proj.col,
        )

        sampled_i = sampled_i.reshape(p, k)
        sampled_valid = sampled_valid.reshape(p, k)
        proj_valid = proj.valid.reshape(p, k)
        valid_mask = proj_valid & sampled_valid
        if not bool(valid_mask.any().item()):
            continue

        frame_stats = frame_stats_cache[neighbor_key]
        norm_vals = normalize_by_percentiles(
            sampled_i,
            lo=frame_stats["p_lo"],
            hi=frame_stats["p_hi"],
            eps=cfg.lik_log_eps,
        )
        log_evidence = torch.log(norm_vals.clamp_min(cfg.lik_log_eps)).clamp(
            min=cfg.lik_log_floor,
            max=0.0,
        )

        reliability = float(frame_stats["reliability"]) if cfg.lik_use_frame_reliability else 1.0
        weighted_valid = valid_mask.to(dtype=torch.float32)
        if reliability < 1.0:
            weighted_valid = weighted_valid * reliability

        loglik_sum = loglik_sum + (weighted_valid * log_evidence)
        valid_w = valid_w + weighted_valid

    supported = valid_w > 0.0
    loglik = torch.where(
        supported,
        loglik_sum / valid_w.clamp_min(1e-8),
        torch.full_like(loglik_sum, float(cfg.lik_log_floor)),
    )
    support_mask = valid_w > float(cfg.lik_min_support)
    return loglik, support_mask


def select_topk_bright_pixels(gt_gray, k):
    h, w = gt_gray.shape
    flat = gt_gray.reshape(-1)
    valid_mask = torch.isfinite(flat) & (flat > 0)
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

    if valid_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=gt_gray.device), torch.empty(0, dtype=torch.long, device=gt_gray.device)

    valid_vals = flat[valid_idx]
    idx_np = valid_idx.detach().cpu().numpy()
    vals_np = valid_vals.detach().cpu().numpy()
    order = np.lexsort((idx_np, -vals_np))
    topk = min(max(1, int(k)), len(order))
    chosen_idx = torch.from_numpy(idx_np[order[:topk]]).to(device=gt_gray.device, dtype=torch.long)
    rows = chosen_idx // w
    cols = chosen_idx % w
    return rows, cols


def build_pixel_bank(training_frames, pixels_per_frame):
    pixel_bank = {}
    gt_frame_cache = {}

    for frame_idx, cam in enumerate(training_frames):
        frame_key = str(cam.image_name)
        gt_image = preprocess_gt_image(cam.original_image)
        gt_gray = gt_image.mean(dim=0)
        rows, cols = select_topk_bright_pixels(gt_gray, pixels_per_frame)

        if rows.numel() == 0:
            fallback_row = torch.tensor([gt_gray.shape[0] // 2], dtype=torch.long, device=gt_gray.device)
            fallback_col = torch.tensor([gt_gray.shape[1] // 2], dtype=torch.long, device=gt_gray.device)
            rows = fallback_row
            cols = fallback_col
            print(f"[Elevation Stage 1] frame={frame_key}: no valid pixels, using center fallback")

        pixel_bank[frame_key] = {
            "frame_key": frame_key,
            "frame_idx": int(frame_idx),
            "rows": rows,
            "cols": cols,
            "logits_key": frame_key,
        }
        gt_frame_cache[frame_key] = {
            "gt_gray": gt_gray,
        }

    return pixel_bank, gt_frame_cache


def build_frame_stats_cache(gt_frame_cache, cfg):
    frame_stats = {}
    for frame_key, frame_data in gt_frame_cache.items():
        gt_gray = frame_data["gt_gray"]
        valid_mask = gt_gray > 0
        frame_stats[frame_key] = compute_frame_stats(
            gt_frame=gt_gray,
            valid_mask=valid_mask,
            p_lo=cfg.lik_norm_p_lo,
            p_hi=cfg.lik_norm_p_hi,
            rel_floor=cfg.rel_floor,
            rel_valid_min=cfg.rel_valid_min,
            rel_valid_max=cfg.rel_valid_max,
            rel_dyn_min=cfg.rel_dyn_min,
            rel_dyn_max=cfg.rel_dyn_max,
        )
    return frame_stats


def serialize_pixel_bank(pixel_bank):
    out = {}
    for frame_key, entry in pixel_bank.items():
        out[str(frame_key)] = {
            "frame_key": str(entry["frame_key"]),
            "frame_idx": int(entry["frame_idx"]),
            "rows": entry["rows"].detach().cpu().tolist(),
            "cols": entry["cols"].detach().cpu().tolist(),
            "logits_key": str(entry["logits_key"]),
        }
    return out


def build_pixel_logits_registry(pixel_bank, bins, loaded_pixel_logits=None, mismatch_policy="strict"):
    policy = str(mismatch_policy)
    if policy not in {"strict", "reset_frame", "reset_all"}:
        raise ValueError(f"Invalid mismatch_policy: {mismatch_policy}")

    expected_keys = set(pixel_bank.keys())
    loaded = loaded_pixel_logits if isinstance(loaded_pixel_logits, dict) else {}
    loaded_keys = set(str(k) for k in loaded.keys())

    if policy == "strict" and loaded_keys and loaded_keys != expected_keys:
        raise ValueError("Pixel-logit frame-key mismatch under strict resume policy")

    registry = {}
    restored_count = 0
    reset_count = 0
    for frame_key, entry in pixel_bank.items():
        k = int(entry["rows"].numel())
        target_shape = (k, int(bins))
        init_tensor = torch.zeros(target_shape, device=entry["rows"].device, dtype=torch.float32)

        if policy != "reset_all" and frame_key in loaded:
            loaded_tensor = torch.as_tensor(loaded[frame_key], dtype=torch.float32, device=entry["rows"].device)
            if tuple(loaded_tensor.shape) == target_shape:
                init_tensor = loaded_tensor
                restored_count += 1
            elif policy == "strict":
                raise ValueError(
                    f"Pixel-logit shape mismatch for frame_key={frame_key}: "
                    f"checkpoint={tuple(loaded_tensor.shape)}, runtime={target_shape}"
                )
            else:
                reset_count += 1
        elif policy != "reset_all" and loaded and frame_key not in loaded:
            if policy == "strict":
                raise ValueError(f"Missing pixel-logit frame_key={frame_key} under strict resume policy")
            reset_count += 1

        registry[frame_key] = torch.nn.Parameter(init_tensor)

    return registry, restored_count, reset_count


def serialize_pixel_logits_registry(pixel_logits_registry):
    return {
        str(frame_key): param.detach().cpu()
        for frame_key, param in pixel_logits_registry.items()
    }


def build_optim_elev(pixel_logits_registry, lr, loaded_state=None, strict_optimizer_state=False):
    params = [param for _, param in sorted(pixel_logits_registry.items(), key=lambda kv: kv[0])]
    if not params:
        return None

    optim = torch.optim.Adam(
        [
            {
                "params": params,
                "lr": float(lr),
                "name": "elev_pixel_logits",
            }
        ]
    )

    if loaded_state is not None:
        try:
            optim.load_state_dict(loaded_state)
        except Exception as exc:
            if strict_optimizer_state:
                raise ValueError(f"Failed to restore optim_elev state under strict policy: {exc}") from exc
            print(f"[Elevation Stage 1] optim_elev restore skipped: {exc}")

    return optim


def build_pixel_bank_from_cache(training_frames, gt_frame_cache, pixels_per_frame):
    pixel_bank = {}

    for frame_idx, cam in enumerate(training_frames):
        frame_key = str(cam.image_name)
        if frame_key not in gt_frame_cache:
            raise KeyError(f"Missing gt_frame_cache entry for frame_key={frame_key}")
        gt_gray = gt_frame_cache[frame_key]["gt_gray"]
        rows, cols = select_topk_bright_pixels(gt_gray, pixels_per_frame)

        if rows.numel() == 0:
            fallback_row = torch.tensor([gt_gray.shape[0] // 2], dtype=torch.long, device=gt_gray.device)
            fallback_col = torch.tensor([gt_gray.shape[1] // 2], dtype=torch.long, device=gt_gray.device)
            rows = fallback_row
            cols = fallback_col
            print(f"[Elevation Stage 1] frame={frame_key}: no valid pixels on refresh, using center fallback")

        pixel_bank[frame_key] = {
            "frame_key": frame_key,
            "frame_idx": int(frame_idx),
            "rows": rows,
            "cols": cols,
            "logits_key": frame_key,
        }

    return pixel_bank


def maybe_refresh_pixel_bank_and_logits(
    iteration,
    training_frames,
    active_frame_keys,
    gt_frame_cache,
    pixel_bank,
    pixel_logits_registry,
    optim_elev,
    cfg,
):
    if not should_refresh_pixel_bank(iteration=iteration, refresh_interval=cfg.bank_refresh_interval):
        return pixel_bank, pixel_logits_registry, optim_elev, False

    refreshed_bank = build_pixel_bank_from_cache(
        training_frames=training_frames,
        gt_frame_cache=gt_frame_cache,
        pixels_per_frame=cfg.pixels_per_frame,
    )

    refreshed_tensors = {}
    shape_changed = False
    for frame_key in active_frame_keys:
        old_entry = pixel_bank[frame_key]
        new_entry = refreshed_bank[frame_key]
        old_logits = pixel_logits_registry[frame_key].detach()
        remapped = remap_or_reset_pixel_logits(
            old_rows=old_entry["rows"],
            old_cols=old_entry["cols"],
            old_logits=old_logits,
            new_rows=new_entry["rows"],
            new_cols=new_entry["cols"],
            remap_mode=cfg.bank_remap_mode,
            remap_max_dist=cfg.bank_remap_max_dist,
        )
        if optimizer_rebuild_required(old_logits, remapped):
            shape_changed = True
        refreshed_tensors[frame_key] = remapped

    if shape_changed:
        new_registry = {
            frame_key: torch.nn.Parameter(refreshed_tensors[frame_key])
            for frame_key in active_frame_keys
        }
        pixel_logits_registry = new_registry
        optim_elev = build_optim_elev(pixel_logits_registry, cfg.logit_lr)
    else:
        with torch.no_grad():
            for frame_key in active_frame_keys:
                pixel_logits_registry[frame_key].copy_(refreshed_tensors[frame_key])

    pixel_bank = refreshed_bank
    print(
        "[Elevation Stage 1] pixel bank refresh: "
        f"iter={iteration}, mode={cfg.bank_remap_mode}, "
        f"shape_changed={int(shape_changed)}"
    )
    return pixel_bank, pixel_logits_registry, optim_elev, True


def resolve_chunk4_coupling_weight(iteration, cfg):
    warmup = max(0, int(cfg.couple_warmup))
    if warmup <= 0:
        return float(cfg.couple_weight_end)
    t = max(0.0, min(float(iteration) / float(warmup), 1.0))
    return float(cfg.couple_weight_start) + t * (float(cfg.couple_weight_end) - float(cfg.couple_weight_start))


def compute_chunk4_coupling_for_frame(
    *,
    frame_idx,
    frame_key,
    training_frames,
    render_pkg,
    gaussians,
    sonar_config,
    sonar_scale_factor,
    pixel_bank,
    p_post_frame,
    elev_angle_bins,
    chunk4_cfg,
):
    zero = gaussians.get_xyz.new_tensor(0.0)
    out = {
        "loss": zero,
        "match_rate": 0.0,
        "match_count": 0,
        "expected_count": 0,
        "residual_mean": 0.0,
        "residual_p95": 0.0,
        "assoc_w_mean": 0.0,
    }

    if frame_key not in pixel_bank:
        return out
    if p_post_frame is None or p_post_frame.ndim != 2:
        return out

    bank_entry = pixel_bank[frame_key]
    rows = bank_entry["rows"]
    cols = bank_entry["cols"]
    if rows.numel() == 0 or cols.numel() == 0:
        return out

    p = min(int(rows.shape[0]), int(cols.shape[0]), int(p_post_frame.shape[0]))
    b = min(int(p_post_frame.shape[1]), int(elev_angle_bins.shape[0]))
    if p <= 0 or b <= 0:
        return out

    rows = rows[:p]
    cols = cols[:p]
    p_post = p_post_frame[:p, :b].detach().to(dtype=torch.float32)
    p_post = p_post / p_post.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    elev_bins = elev_angle_bins[:b].to(device=rows.device, dtype=torch.float32)

    pts_bins = back_project_bins(
        frame_idx=frame_idx,
        rows=rows,
        cols=cols,
        elev_bins=elev_bins,
        cameras=training_frames,
        sonar_config=sonar_config,
        scale_factor=sonar_scale_factor,
    )
    pts_expected = torch.sum(p_post.unsqueeze(-1) * pts_bins, dim=1)

    exp_proj = sonar_project_points(
        pts_expected,
        training_frames[frame_idx],
        sonar_config,
        scale_factor=sonar_scale_factor,
    )

    visible_mask = render_pkg["visibility_filter"].to(dtype=torch.bool)
    candidate_rows = torch.where(visible_mask)[0]
    max_candidates = int(chunk4_cfg.couple_max_candidates)
    if max_candidates > 0 and candidate_rows.numel() > max_candidates:
        pick = torch.linspace(
            0,
            candidate_rows.numel() - 1,
            steps=max_candidates,
            device=candidate_rows.device,
        ).round().to(dtype=torch.long)
        candidate_rows = candidate_rows[pick]
    if candidate_rows.numel() == 0:
        out["expected_count"] = int(p)
        return out

    surfel_xyz = gaussians.get_xyz[candidate_rows]
    surf_proj = sonar_project_points(
        surfel_xyz,
        training_frames[frame_idx],
        sonar_config,
        scale_factor=sonar_scale_factor,
    )

    surf_idx, assoc_w, match_valid = associate_expected_points_to_surfels(
        exp_row=exp_proj.row,
        exp_col=exp_proj.col,
        exp_depth=exp_proj.range_vals,
        exp_valid=exp_proj.valid,
        surf_row=surf_proj.row,
        surf_col=surf_proj.col,
        surf_depth=surf_proj.range_vals,
        surf_valid=surf_proj.valid,
        max_pix_err=chunk4_cfg.couple_max_pix_err,
        max_depth_err=chunk4_cfg.couple_max_depth_err,
        sigma_pix=chunk4_cfg.couple_sigma_pix,
        sigma_depth=chunk4_cfg.couple_sigma_depth,
        min_w=chunk4_cfg.couple_min_w,
    )

    loss_couple = reduce_coupling_loss(
        pts_expected=pts_expected,
        surfel_xyz=surfel_xyz,
        surf_idx=surf_idx,
        assoc_w=assoc_w,
        match_valid=match_valid,
        huber_delta=chunk4_cfg.couple_huber_delta,
    )
    if not bool(torch.isfinite(loss_couple).item()):
        loss_couple = zero

    match_count = int(match_valid.sum().item())
    expected_count = int(match_valid.shape[0])
    match_rate = float(match_count / max(1, expected_count))

    residual_mean = 0.0
    residual_p95 = 0.0
    assoc_w_mean = 0.0
    if match_count > 0:
        matched_idx = surf_idx[match_valid]
        residual = torch.norm(surfel_xyz[matched_idx] - pts_expected[match_valid], dim=-1)
        residual_detached = residual.detach()
        residual_mean = float(residual_detached.mean().item())
        residual_p95 = float(torch.quantile(residual_detached, 0.95).item())
        assoc_w_mean = float(assoc_w[match_valid].detach().mean().item())

    out.update(
        {
            "loss": loss_couple,
            "match_rate": match_rate,
            "match_count": match_count,
            "expected_count": expected_count,
            "residual_mean": residual_mean,
            "residual_p95": residual_p95,
            "assoc_w_mean": assoc_w_mean,
        }
    )
    return out


def camera_world_position_tensor(camera, device):
    r_w2c = torch.as_tensor(camera.R, device=device, dtype=torch.float32)
    t_w2c = torch.as_tensor(camera.T, device=device, dtype=torch.float32)
    r_c2w = r_w2c.transpose(0, 1)
    return -(r_c2w @ t_w2c)


def camera_forward_world_tensor(camera, device):
    r_w2c = torch.as_tensor(camera.R, device=device, dtype=torch.float32)
    r_c2w = r_w2c.transpose(0, 1)
    forward = r_c2w[:, 2]
    return forward / torch.norm(forward).clamp_min(1e-8)


def select_diverse_support_frames(sampled_frame_indices, training_frames, min_angle_deg, device):
    if not sampled_frame_indices:
        return []
    min_angle = max(0.0, float(min_angle_deg))
    if min_angle <= 0.0:
        return list(sampled_frame_indices)

    selected = []
    selected_dirs = []
    cos_thr = math.cos(math.radians(min_angle))
    for frame_idx in sampled_frame_indices:
        direction = camera_forward_world_tensor(training_frames[frame_idx], device=device)
        keep = True
        for prev in selected_dirs:
            cos_sim = torch.clamp(torch.dot(direction, prev), -1.0, 1.0)
            if float(cos_sim.item()) > cos_thr:
                keep = False
                break
        if keep:
            selected.append(frame_idx)
            selected_dirs.append(direction)
    if not selected:
        selected.append(sampled_frame_indices[0])
    return selected


def compute_chunk4_support_observations_for_frame(
    frame_idx,
    training_frames,
    gaussians,
    render_pkg,
    rendered,
    gt_image,
    sonar_config,
    sonar_scale_factor,
    support_residual_thresh,
):
    device = gaussians.get_xyz.device
    empty_idx = torch.empty((0,), dtype=torch.long, device=device)
    out = {
        "candidate_idx": empty_idx,
        "support_idx": empty_idx,
        "candidate_count": 0,
        "support_count": 0,
        "residual_mean": 0.0,
        "residual_p95": 0.0,
        "valid_projection_count": 0,
        "meaningful_gt_count": 0,
    }

    visible_mask = render_pkg.get("visibility_filter")
    if visible_mask is None:
        return out
    visible_mask = visible_mask.to(dtype=torch.bool)
    if not bool(visible_mask.any().item()):
        return out

    row_idx = torch.where(visible_mask)[0]
    xyz_visible = gaussians.get_xyz[row_idx]
    proj = sonar_project_points(
        xyz_visible,
        training_frames[frame_idx],
        sonar_config,
        scale_factor=sonar_scale_factor,
    )
    valid_proj = proj.valid & torch.isfinite(proj.row) & torch.isfinite(proj.col)
    out["valid_projection_count"] = int(valid_proj.sum().item())
    if not bool(valid_proj.any().item()):
        return out

    gt_gray = gt_image.mean(dim=0)
    rendered_gray = rendered.mean(dim=0)
    gt_sample, gt_valid = sample_gt(gt_gray, proj.row, proj.col)
    rendered_sample, rendered_valid = sample_gt(rendered_gray, proj.row, proj.col)
    meaningful_gt = gt_valid & rendered_valid & (gt_sample > 0.0)
    out["meaningful_gt_count"] = int(meaningful_gt.sum().item())

    candidate_mask = valid_proj & meaningful_gt
    if not bool(candidate_mask.any().item()):
        return out

    residual = torch.abs(rendered_sample - gt_sample)
    support_mask = candidate_mask & (residual <= float(support_residual_thresh))

    candidate_idx = row_idx[candidate_mask]
    support_idx = row_idx[support_mask]
    out["candidate_idx"] = candidate_idx
    out["support_idx"] = support_idx
    out["candidate_count"] = int(candidate_idx.shape[0])
    out["support_count"] = int(support_idx.shape[0])

    if out["support_count"] > 0:
        support_residual = residual[support_mask].detach()
        out["residual_mean"] = float(support_residual.mean().item())
        out["residual_p95"] = float(torch.quantile(support_residual, 0.95).item())

    return out


def init_chunk4_runtime_state(gaussians, cfg, init_iter=0):
    if not bool(cfg.support_use_persistent_ids):
        return None
    device = gaussians.get_xyz.device
    num_surfels = int(gaussians.get_xyz.shape[0])
    persistent_state = initialize_persistent_surfel_state(
        num_surfels=num_surfels,
        device=device,
        init_birth_iter=init_iter,
    )
    if bool(cfg.surfel_id_asserts):
        assert_surfel_id_integrity(persistent_state)
    next_surfel_id = int(persistent_state["next_surfel_id"])
    return {
        "persistent_state": persistent_state,
        "support_count_by_id": torch.zeros((next_surfel_id,), dtype=torch.float32, device=device),
        "diverse_candidate_count_by_id": torch.zeros((next_surfel_id,), dtype=torch.float32, device=device),
        "last_support_stats": {},
    }


def ensure_chunk4_state_capacity(chunk4_runtime_state, gaussians, cfg, current_iter):
    if chunk4_runtime_state is None:
        return None

    state = chunk4_runtime_state["persistent_state"]
    num_rows = int(gaussians.get_xyz.shape[0])
    known_rows = int(state["surfel_ids"].shape[0])
    if num_rows < known_rows:
        raise RuntimeError(
            "Chunk-4 persistent state lost row alignment: "
            f"model_rows={num_rows}, state_rows={known_rows}. "
            "Use Chunk-4 prune wrappers for topology edits."
        )
    if num_rows > known_rows:
        n_new = num_rows - known_rows
        state = apply_densify_to_surfel_state(state, n_new=n_new, current_iter=current_iter)
        chunk4_runtime_state["persistent_state"] = state
        support_count = chunk4_runtime_state["support_count_by_id"]
        diverse_count = chunk4_runtime_state["diverse_candidate_count_by_id"]
        device = support_count.device
        chunk4_runtime_state["support_count_by_id"] = torch.cat(
            [support_count, torch.zeros((n_new,), dtype=torch.float32, device=device)],
            dim=0,
        )
        chunk4_runtime_state["diverse_candidate_count_by_id"] = torch.cat(
            [diverse_count, torch.zeros((n_new,), dtype=torch.float32, device=device)],
            dim=0,
        )

    if bool(cfg.surfel_id_asserts):
        assert_surfel_id_integrity(chunk4_runtime_state["persistent_state"])
    return chunk4_runtime_state


def apply_row_prune_with_chunk4_state(gaussians, chunk4_runtime_state, row_prune_mask, cfg, reason):
    prune_mask = row_prune_mask.to(dtype=torch.bool)
    num_to_prune = int(prune_mask.sum().item())
    if num_to_prune <= 0:
        return 0

    keep_idx = torch.where(~prune_mask)[0]
    gaussians.prune_points(prune_mask)

    if chunk4_runtime_state is not None:
        state = chunk4_runtime_state["persistent_state"]
        state = apply_prune_reorder_to_surfel_state(state, keep_row_idx=keep_idx)
        chunk4_runtime_state["persistent_state"] = state
        if bool(cfg.surfel_id_asserts):
            assert_surfel_id_integrity(state)

    return num_to_prune


def update_chunk4_support_runtime(
    global_iter,
    sampled_frame_indices,
    iter_support_obs,
    training_frames,
    gaussians,
    chunk4_runtime_state,
    chunk4_cfg,
):
    zero_stats = {
        "match_frames": 0,
        "diverse_frames": 0,
        "candidate_obs": 0,
        "support_obs": 0,
        "support_ratio": 0.0,
        "residual_mean": 0.0,
        "residual_p95": 0.0,
        "active_support_ge2_frac": 0.0,
        "active_support_ge3_frac": 0.0,
        "active_support_median": 0.0,
        "grace_active_count": 0,
        "pruned_support_count": 0,
        "hard_prune_enabled": mode_enables_hard_prune(chunk4_cfg.effective_support_mode),
        "duplicate_active_ids": 0,
        "invalid_id_to_row": 0,
    }

    if chunk4_runtime_state is None:
        return zero_stats

    ensure_chunk4_state_capacity(chunk4_runtime_state, gaussians, chunk4_cfg, global_iter)
    state = chunk4_runtime_state["persistent_state"]
    num_rows = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device

    diverse_frames = select_diverse_support_frames(
        sampled_frame_indices,
        training_frames,
        min_angle_deg=chunk4_cfg.support_view_angle_min_deg,
        device=device,
    )

    row_candidate_counts = torch.zeros((num_rows,), dtype=torch.float32, device=device)
    row_support_counts = torch.zeros((num_rows,), dtype=torch.float32, device=device)
    residual_values = []
    match_frames = 0
    for frame_idx in diverse_frames:
        obs = iter_support_obs.get(frame_idx)
        if obs is None:
            continue
        cand_idx = obs["candidate_idx"]
        supp_idx = obs["support_idx"]
        if cand_idx.numel() > 0:
            row_candidate_counts[cand_idx] += 1.0
        if supp_idx.numel() > 0:
            row_support_counts[supp_idx] += 1.0
            residual_values.append(float(obs["residual_mean"]))
            match_frames += 1

    surfel_ids = state["surfel_ids"].to(dtype=torch.long)
    support_inc_by_id = torch.zeros_like(chunk4_runtime_state["support_count_by_id"])
    diverse_inc_by_id = torch.zeros_like(chunk4_runtime_state["diverse_candidate_count_by_id"])
    if surfel_ids.numel() > 0:
        support_inc_by_id[surfel_ids] = row_support_counts
        diverse_inc_by_id[surfel_ids] = row_candidate_counts

    chunk4_runtime_state["support_count_by_id"] = (
        chunk4_runtime_state["support_count_by_id"] + support_inc_by_id
    )
    chunk4_runtime_state["diverse_candidate_count_by_id"] = (
        chunk4_runtime_state["diverse_candidate_count_by_id"] + diverse_inc_by_id
    )

    row_support_ratio = torch.where(
        row_candidate_counts > 0,
        row_support_counts / row_candidate_counts.clamp_min(1.0),
        torch.zeros_like(row_candidate_counts),
    )
    state = update_support_buffers_by_id(
        state,
        row_support_raw=row_support_ratio,
        ema_decay=chunk4_cfg.support_ema_decay,
    )

    support_count_by_id = chunk4_runtime_state["support_count_by_id"]
    diverse_count_by_id = chunk4_runtime_state["diverse_candidate_count_by_id"]
    fail_mask_by_id, _ = compute_support_failure_mask(
        iteration=global_iter,
        support_count_by_id=support_count_by_id,
        diverse_candidate_count_by_id=diverse_count_by_id,
        warmup_iters=chunk4_cfg.support_warmup_iters,
        late_phase_start_iter=chunk4_cfg.support_late_phase_start_iter,
        min_ratio_mid=chunk4_cfg.support_min_ratio_mid,
        min_ratio_late=chunk4_cfg.support_min_ratio_late,
        min_count_mid=chunk4_cfg.support_min_count_mid,
        min_count_late=chunk4_cfg.support_min_count_late,
        use_ratio=chunk4_cfg.support_use_ratio,
    )
    fail_streak_by_id, prune_mask_by_id = apply_prune_hysteresis(
        fail_streak_by_id=state["fail_streak_by_id"],
        fail_mask_by_id=fail_mask_by_id,
        patience=chunk4_cfg.support_prune_patience,
    )
    state["fail_streak_by_id"] = fail_streak_by_id
    prune_mask_by_id = apply_new_surfel_grace(
        prune_mask_by_id=prune_mask_by_id,
        birth_iter_by_id=state["birth_iter_by_id"],
        current_iter=global_iter,
        grace_iters=chunk4_cfg.support_new_surfel_grace_iters,
        enabled=chunk4_cfg.support_use_new_surfel_grace,
    )

    pruned_support_count = 0
    if mode_enables_hard_prune(chunk4_cfg.effective_support_mode):
        active_prune_mask = prune_mask_by_id[state["surfel_ids"]]
        pruned_support_count = apply_row_prune_with_chunk4_state(
            gaussians,
            chunk4_runtime_state,
            active_prune_mask,
            cfg=chunk4_cfg,
            reason="support",
        )
        state = chunk4_runtime_state["persistent_state"]

    active_ids = state["surfel_ids"]
    active_support = support_count_by_id[active_ids] if active_ids.numel() > 0 else torch.zeros(0, device=device)
    ge2 = float((active_support >= 2.0).float().mean().item()) if active_support.numel() > 0 else 0.0
    ge3 = float((active_support >= 3.0).float().mean().item()) if active_support.numel() > 0 else 0.0
    median_support = float(active_support.median().item()) if active_support.numel() > 0 else 0.0

    age = global_iter - state["birth_iter_by_id"].to(dtype=torch.long)
    grace_active = age < int(chunk4_cfg.support_new_surfel_grace_iters)
    grace_active_count = int(grace_active[active_ids].sum().item()) if active_ids.numel() > 0 else 0

    duplicate_active_ids = 0
    invalid_id_to_row = 0
    if bool(chunk4_cfg.surfel_id_asserts):
        assert_surfel_id_integrity(state)
    if active_ids.numel() > 0:
        duplicate_active_ids = int(active_ids.numel() - torch.unique(active_ids).numel())
    expected_map = torch.full_like(state["id_to_row"], -1)
    if active_ids.numel() > 0:
        expected_map[active_ids] = torch.arange(active_ids.shape[0], device=active_ids.device, dtype=torch.long)
    invalid_id_to_row = int((state["id_to_row"] != expected_map).sum().item())

    candidate_obs = float(row_candidate_counts.sum().item())
    support_obs = float(row_support_counts.sum().item())
    support_ratio = float(support_obs / max(candidate_obs, 1.0))
    residual_mean = float(np.mean(residual_values)) if residual_values else 0.0
    residual_p95 = float(np.quantile(np.asarray(residual_values, dtype=np.float64), 0.95)) if residual_values else 0.0

    stats = {
        "match_frames": int(match_frames),
        "diverse_frames": int(len(diverse_frames)),
        "candidate_obs": int(candidate_obs),
        "support_obs": int(support_obs),
        "support_ratio": support_ratio,
        "residual_mean": residual_mean,
        "residual_p95": residual_p95,
        "active_support_ge2_frac": ge2,
        "active_support_ge3_frac": ge3,
        "active_support_median": median_support,
        "grace_active_count": grace_active_count,
        "pruned_support_count": int(pruned_support_count),
        "hard_prune_enabled": mode_enables_hard_prune(chunk4_cfg.effective_support_mode),
        "duplicate_active_ids": duplicate_active_ids,
        "invalid_id_to_row": invalid_id_to_row,
    }
    chunk4_runtime_state["last_support_stats"] = stats
    chunk4_runtime_state["persistent_state"] = state
    return stats


def build_chunk4_runtime_checkpoint_state(chunk4_runtime_state, active_frame_keys, cfg):
    if chunk4_runtime_state is None:
        return None

    state = chunk4_runtime_state["persistent_state"]
    support_scheduler_state = {
        "last_support_stats": dict(chunk4_runtime_state.get("last_support_stats", {})),
        "warmup_iters": int(cfg.support_warmup_iters),
        "late_phase_start_iter": int(cfg.support_late_phase_start_iter),
        "prune_patience": int(cfg.support_prune_patience),
    }
    return build_chunk4_checkpoint_payload(
        surfel_ids=state["surfel_ids"],
        next_surfel_id=state["next_surfel_id"],
        id_to_row=state["id_to_row"],
        ema_by_id=state["ema_by_id"],
        last_raw_by_id=state["last_raw_by_id"],
        birth_iter_by_id=state["birth_iter_by_id"],
        fail_streak_by_id=state["fail_streak_by_id"],
        active_frame_keys=active_frame_keys,
        couple_mode=cfg.effective_couple_mode,
        support_mode=cfg.effective_support_mode,
        support_scheduler_state=support_scheduler_state,
        support_count_by_id=chunk4_runtime_state["support_count_by_id"],
        diverse_candidate_count_by_id=chunk4_runtime_state["diverse_candidate_count_by_id"],
    )


def restore_chunk4_runtime_state_from_checkpoint(chunk4_payload, gaussians, cfg):
    if not isinstance(chunk4_payload, dict):
        raise ValueError("Chunk-4 checkpoint payload must be a dict")

    device = gaussians.get_xyz.device
    num_rows = int(gaussians.get_xyz.shape[0])

    def _load_tensor(name, dtype):
        value = chunk4_payload.get(name)
        if value is None:
            raise ValueError(f"Chunk-4 checkpoint missing '{name}'")
        return torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)

    surfel_ids = _load_tensor("surfel_ids", torch.long)
    next_surfel_id = int(chunk4_payload.get("next_surfel_id", -1))
    id_to_row = _load_tensor("id_to_row", torch.long)
    ema_by_id = _load_tensor("ema_by_id", torch.float32)
    last_raw_by_id = _load_tensor("last_raw_by_id", torch.float32)
    birth_iter_by_id = _load_tensor("birth_iter_by_id", torch.long)
    fail_streak_by_id = _load_tensor("fail_streak_by_id", torch.long)

    if surfel_ids.shape[0] != num_rows:
        raise ValueError(
            "Chunk-4 checkpoint surfel row mismatch: "
            f"checkpoint_rows={surfel_ids.shape[0]}, runtime_rows={num_rows}"
        )
    if next_surfel_id < 0:
        raise ValueError(f"Chunk-4 checkpoint has invalid next_surfel_id={next_surfel_id}")

    expected_len = int(next_surfel_id)
    id_tensors = {
        "id_to_row": id_to_row,
        "ema_by_id": ema_by_id,
        "last_raw_by_id": last_raw_by_id,
        "birth_iter_by_id": birth_iter_by_id,
        "fail_streak_by_id": fail_streak_by_id,
    }
    for name, tensor in id_tensors.items():
        if int(tensor.shape[0]) != expected_len:
            raise ValueError(
                "Chunk-4 checkpoint id-space length mismatch: "
                f"{name}={tensor.shape[0]}, expected={expected_len}"
            )

    support_count_by_id = chunk4_payload.get("support_count_by_id")
    if support_count_by_id is None:
        support_count_by_id = torch.zeros((expected_len,), dtype=torch.float32, device=device)
    else:
        support_count_by_id = torch.as_tensor(support_count_by_id, dtype=torch.float32, device=device).reshape(-1)
    diverse_candidate_count_by_id = chunk4_payload.get("diverse_candidate_count_by_id")
    if diverse_candidate_count_by_id is None:
        diverse_candidate_count_by_id = torch.zeros((expected_len,), dtype=torch.float32, device=device)
    else:
        diverse_candidate_count_by_id = torch.as_tensor(
            diverse_candidate_count_by_id,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)

    if int(support_count_by_id.shape[0]) != expected_len:
        raise ValueError(
            "Chunk-4 checkpoint support_count_by_id length mismatch: "
            f"got={support_count_by_id.shape[0]}, expected={expected_len}"
        )
    if int(diverse_candidate_count_by_id.shape[0]) != expected_len:
        raise ValueError(
            "Chunk-4 checkpoint diverse_candidate_count_by_id length mismatch: "
            f"got={diverse_candidate_count_by_id.shape[0]}, expected={expected_len}"
        )

    persistent_state = {
        "surfel_ids": surfel_ids,
        "next_surfel_id": next_surfel_id,
        "id_to_row": id_to_row,
        "ema_by_id": ema_by_id,
        "last_raw_by_id": last_raw_by_id,
        "birth_iter_by_id": birth_iter_by_id,
        "fail_streak_by_id": fail_streak_by_id,
    }
    if bool(cfg.surfel_id_asserts):
        assert_surfel_id_integrity(persistent_state)

    scheduler_state = chunk4_payload.get("support_scheduler_state")
    if isinstance(scheduler_state, dict):
        last_support_stats = dict(scheduler_state.get("last_support_stats", {}))
    else:
        last_support_stats = {}

    return {
        "persistent_state": persistent_state,
        "support_count_by_id": support_count_by_id,
        "diverse_candidate_count_by_id": diverse_candidate_count_by_id,
        "last_support_stats": last_support_stats,
    }


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
                             iteration, stage_name, metadata=None, stage1_runtime_state=None,
                             chunk4_runtime_state=None):
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
        "elevation_stage1_state": stage1_runtime_state or {},
        "elevation_chunk4_state": chunk4_runtime_state,
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
        return int(iteration), {"format": "legacy_tuple"}, None, None

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
    stage1_runtime_state = payload.get("elevation_stage1_state")
    chunk4_runtime_state = payload.get("elevation_chunk4_state")
    return iteration, metadata, stage1_runtime_state, chunk4_runtime_state


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                continue
        self.flush()

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                continue


LOG_FILE = None
LOSS_LOG_HANDLE = None
LOSS_LOG_PATH = None
ORIGINAL_STDOUT = sys.stdout
ORIGINAL_STDERR = sys.stderr


def setup_logging(output_dir):
    global LOG_FILE
    log_path = os.path.join(output_dir, "run.log")
    LOG_FILE = open(log_path, "w")
    sys.stdout = Tee(ORIGINAL_STDOUT, LOG_FILE)
    sys.stderr = Tee(ORIGINAL_STDERR, LOG_FILE)
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
    global LOG_FILE, LOSS_LOG_HANDLE

    sys.stdout = ORIGINAL_STDOUT
    sys.stderr = ORIGINAL_STDERR

    if LOSS_LOG_HANDLE is not None:
        try:
            LOSS_LOG_HANDLE.flush()
            LOSS_LOG_HANDLE.close()
        finally:
            LOSS_LOG_HANDLE = None
    if LOG_FILE is not None:
        try:
            LOG_FILE.flush()
            LOG_FILE.close()
        finally:
            LOG_FILE = None


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

    render_mode = diag.get("sonar_render_mode", "unknown")
    occlusion_mode = diag.get("sonar_occlusion_mode", "unknown")
    lambertian_mode = diag.get("lambertian_mode", "unknown")
    visible_ratio = float(diag.get("visible_surfel_ratio", 0.0))
    print(
        f"{prefix}render_mode={render_mode}, occlusion_mode={occlusion_mode}, "
        f"lambertian={lambertian_mode}, visible={visible_ratio:.6f}"
    )

    surfel_stats = diag.get("surfel_size_stats_world")
    if surfel_stats is not None:
        print(f"{prefix}surfel_size_stats_world={surfel_stats}")

    mass_loss = diag.get("occlusion_support_cap_mass_loss")
    if mass_loss is not None:
        print(f"{prefix}mass_loss={mass_loss}")


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


def select_frame_indices(cameras, num_frames, seed=42, mode="diverse"):
    if mode == "first":
        return list(range(min(num_frames, len(cameras))))
    if mode == "diverse":
        return select_diverse_frames(cameras, num_frames, seed=seed)
    raise ValueError(f"Unsupported frame selection mode: {mode}")


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


def get_repo_commit_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def relative_visualizer_path(path, visualizer_dir):
    return os.path.relpath(path, visualizer_dir).replace(os.sep, "/")


def build_visualizer_metadata_refs(output_dir, visualizer_dir, dataset_path):
    refs = {}
    candidate_paths = {
        "run_log": os.path.join(output_dir, "run.log"),
        "loss_log_csv": os.path.join(output_dir, "loss_log.csv"),
        "cfg_args": os.path.join(output_dir, "cfg_args"),
        "cameras_json": os.path.join(output_dir, "cameras.json"),
        "dataset_manifest": os.path.join(dataset_path, "manifest.json"),
        "dataset_settings": os.path.join(dataset_path, "DATASET_SETTINGS.md"),
        "dataset_consistency_gate": os.path.join(dataset_path, "consistency_gate.json"),
    }
    for key, candidate in candidate_paths.items():
        if os.path.exists(candidate):
            refs[key] = relative_visualizer_path(candidate, visualizer_dir)
    return refs


def get_gaussian_visualization_state(gaussians):
    return prepare_surfel_visualization_state(
        centers=gaussians.get_xyz.detach(),
        scales=gaussians._scaling.detach(),
        rotations=gaussians._rotation.detach(),
        opacity=gaussians.get_opacity.detach().squeeze(-1),
        scales_are_latent=True,
        rotations_are_normalized=False,
    )


def build_stage_glyph_colors(opacity, eq_radius):
    opacity = np.asarray(opacity, dtype=np.float64).reshape(-1)
    eq_radius = np.asarray(eq_radius, dtype=np.float64).reshape(-1)
    if opacity.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    opacity_span = max(opacity.max() - opacity.min(), 1e-8)
    radius_span = max(eq_radius.max() - eq_radius.min(), 1e-8)
    opacity_norm = np.clip((opacity - opacity.min()) / opacity_span, 0.0, 1.0)
    radius_norm = np.clip((eq_radius - eq_radius.min()) / radius_span, 0.0, 1.0)

    warm = np.array([0.93, 0.45, 0.18], dtype=np.float64)
    cool = np.array([0.20, 0.66, 0.95], dtype=np.float64)
    pale = np.array([0.96, 0.90, 0.72], dtype=np.float64)
    color = warm[None, :] * opacity_norm[:, None] + cool[None, :] * (1.0 - radius_norm[:, None])
    color += pale[None, :] * (0.5 * radius_norm[:, None])
    return np.clip(color / 1.5, 0.0, 1.0)


def export_sampled_glyph_artifacts(
    gaussians,
    visualizer_dir,
    glyph_prefix,
    *,
    include_centers_full=False,
):
    surfel_state = get_gaussian_visualization_state(gaussians)
    opacity = surfel_state["opacity"]
    eq_radius = surfel_state["equivalent_radius"]
    selected_idx = deterministic_glyph_indices(
        opacity,
        eq_radius,
        max_count=VISUALIZER_MAX_GLYPHS,
        opacity_percentile=VISUALIZER_OPACITY_PERCENTILE,
        eq_radius_percentile=VISUALIZER_EQ_RADIUS_PERCENTILE,
    )
    selected_colors = build_stage_glyph_colors(opacity[selected_idx], eq_radius[selected_idx])

    artifact = {
        "total_surfels": int(surfel_state["centers"].shape[0]),
        "sampled_surfels": int(selected_idx.shape[0]),
    }

    if include_centers_full:
        center_colors = np.repeat(np.clip(opacity[:, None], 0.0, 1.0), 3, axis=1)
        centers_path = os.path.join(visualizer_dir, f"glyphs_{glyph_prefix}_centers_full.ply")
        write_point_cloud(centers_path, surfel_state["centers"], center_colors)
        artifact["centers_full"] = relative_visualizer_path(centers_path, visualizer_dir)

    combined_path = os.path.join(visualizer_dir, f"glyphs_{glyph_prefix}_sampled.ply")
    front_path = os.path.join(visualizer_dir, f"glyphs_{glyph_prefix}_front_faces.ply")
    back_path = os.path.join(visualizer_dir, f"glyphs_{glyph_prefix}_back_faces.ply")

    if selected_idx.size == 0:
        write_point_cloud(combined_path, np.zeros((0, 3), dtype=np.float64))
        write_point_cloud(front_path, np.zeros((0, 3), dtype=np.float64))
        write_point_cloud(back_path, np.zeros((0, 3), dtype=np.float64))
    else:
        write_triangle_mesh(
            combined_path,
            build_surfel_glyph_mesh(
                surfel_state,
                indices=selected_idx,
                face_mode="double",
                base_colors=selected_colors,
                ellipse_segments=VISUALIZER_GLYPH_SEGMENTS,
                face_offset_scale=VISUALIZER_FACE_OFFSET_SCALE,
                normal_stem_scale=VISUALIZER_NORMAL_STEM_SCALE,
            ),
        )
        write_triangle_mesh(
            front_path,
            build_surfel_glyph_mesh(
                surfel_state,
                indices=selected_idx,
                face_mode="front",
                base_colors=selected_colors,
                ellipse_segments=VISUALIZER_GLYPH_SEGMENTS,
                face_offset_scale=VISUALIZER_FACE_OFFSET_SCALE,
                normal_stem_scale=VISUALIZER_NORMAL_STEM_SCALE,
            ),
        )
        write_triangle_mesh(
            back_path,
            build_surfel_glyph_mesh(
                surfel_state,
                indices=selected_idx,
                face_mode="back",
                base_colors=selected_colors,
                ellipse_segments=VISUALIZER_GLYPH_SEGMENTS,
                face_offset_scale=VISUALIZER_FACE_OFFSET_SCALE,
                normal_stem_scale=VISUALIZER_NORMAL_STEM_SCALE,
            ),
        )

    artifact["sampled_double_sided"] = relative_visualizer_path(combined_path, visualizer_dir)
    artifact["front_faces"] = relative_visualizer_path(front_path, visualizer_dir)
    artifact["back_faces"] = relative_visualizer_path(back_path, visualizer_dir)
    return artifact


def export_stage_visualizer_state(
    gaussians,
    visualizer_dir,
    *,
    stage_filename,
    stage_status,
    glyph_prefix=None,
    include_centers_full=False,
):
    state_path = os.path.join(visualizer_dir, stage_filename)
    gaussians.save_ply(state_path)
    artifact = {
        "status": stage_status,
        "state_ply": relative_visualizer_path(state_path, visualizer_dir),
    }
    if glyph_prefix is not None:
        artifact["glyphs"] = export_sampled_glyph_artifacts(
            gaussians,
            visualizer_dir,
            glyph_prefix,
            include_centers_full=include_centers_full,
        )
    return artifact


def export_frame_visualizer_artifacts(
    training_frames,
    gaussians,
    background,
    sonar_config,
    scale_factor,
    visualizer_dir,
    rendered_dir,
):
    frame_entries = []
    surfel_state = get_gaussian_visualization_state(gaussians)
    frame_width = max(3, len(str(max(len(training_frames) - 1, 0))))
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]

    for frame_idx, cam in enumerate(training_frames):
        stem = build_frame_stem(frame_idx, cam.image_name, width=frame_width)
        color = colors[frame_idx % len(colors)]
        r_c2w = cam.R.T
        position = -r_c2w @ cam.T

        wireframe_near_path = os.path.join(visualizer_dir, f"{stem}_wireframe_near.ply")
        wireframe_full_path = os.path.join(visualizer_dir, f"{stem}_wireframe_full_range.ply")
        write_line_set(
            wireframe_near_path,
            create_pose_wireframe(
                position,
                r_c2w,
                depth=PYRAMID_DEPTH,
                azimuth_fov=sonar_config.azimuth_fov,
                elevation_fov=sonar_config.elevation_fov,
                color=color,
                mode="near",
            ),
        )
        write_line_set(
            wireframe_full_path,
            create_pose_wireframe(
                position,
                r_c2w,
                depth=sonar_config.range_max,
                azimuth_fov=sonar_config.azimuth_fov,
                elevation_fov=sonar_config.elevation_fov,
                color=color,
                mode="full_range",
            ),
        )

        frame_diag = compute_frame_surfel_membership(
            surfel_state["centers"],
            surfel_state["scales"],
            surfel_state["rotations"],
            cam,
            sonar_config,
            scale_factor=scale_factor,
        )
        overlap_idx = np.flatnonzero(frame_diag["overlap_mask"])
        export_idx = select_frame_surfel_indices(
            frame_diag["center_in_fov"],
            frame_diag["overlap_mask"],
            frame_diag["fov_margin"],
            frame_diag["facing_score"],
            surfel_state["opacity"],
            surfel_state["equivalent_radius"],
            max_count=VISUALIZER_MAX_FRAME_GLYPHS,
        )
        surfels_path = os.path.join(visualizer_dir, f"{stem}_surfels_in_fov.ply")

        if export_idx.size == 0:
            write_point_cloud(surfels_path, np.zeros((0, 3), dtype=np.float64))
        else:
            base_colors = diagnostic_colors_from_metrics(
                frame_diag["facing_score"][export_idx],
                frame_diag["range_vals"][export_idx],
                sonar_config.range_min,
                sonar_config.range_max,
            )
            write_triangle_mesh(
                surfels_path,
                build_surfel_glyph_mesh(
                    surfel_state,
                    indices=export_idx,
                    face_mode="double",
                    base_colors=base_colors,
                    ellipse_segments=VISUALIZER_GLYPH_SEGMENTS,
                    face_offset_scale=VISUALIZER_FACE_OFFSET_SCALE,
                    normal_stem_scale=VISUALIZER_NORMAL_STEM_SCALE,
                ),
            )

        with torch.no_grad():
            render_pkg = render_sonar(
                cam,
                gaussians,
                background,
                sonar_config=sonar_config,
                scale_factor=scale_factor,
                sonar_extrinsic=None,
                **SONAR_RENDER_KWARGS,
            )
            rendered = render_pkg["render"]
        rendered_np = (np.clip(rendered[0].detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
        rendered_path = os.path.join(rendered_dir, f"{stem}.png")
        Image.fromarray(rendered_np, mode="L").save(rendered_path)

        frame_entries.append(
            {
                "frame_index": frame_idx,
                "image_name": cam.image_name,
                "stem": stem,
                "wireframe_near": relative_visualizer_path(wireframe_near_path, visualizer_dir),
                "wireframe_full_range": relative_visualizer_path(wireframe_full_path, visualizer_dir),
                "surfels_in_fov": relative_visualizer_path(surfels_path, visualizer_dir),
                "rendered_image": relative_visualizer_path(rendered_path, visualizer_dir),
                "surfel_center_in_fov_count": int(np.count_nonzero(frame_diag["center_in_fov"])),
                "surfel_overlap_count": int(overlap_idx.shape[0]),
                "surfel_export_count": int(export_idx.shape[0]),
            }
        )

    return frame_entries


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

# =============================================================================
# Configuration
# =============================================================================
SYNTHETIC_DATASET_KEYS = {"synthetic_a_clean", "synthetic_c_clean"}
DEFAULT_SYNTHETIC_A_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "synthetic_datasets",
    "synthetic_sphere_A_clean",
)
DEFAULT_SYNTHETIC_C_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "synthetic_datasets",
    "synthetic_cube_C_clean",
)

DATASET_PATHS = {
    "legacy": "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs",
    "r2": "/home/gavin/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs_R2",
    "synthetic_a_clean": DEFAULT_SYNTHETIC_A_PATH,
    "synthetic_c_clean": DEFAULT_SYNTHETIC_C_PATH,
}

DATASET_KEY = os.environ.get("SONAR_DATASET", "r2").strip().lower()
DATASET_PATH_OVERRIDE = os.environ.get("SONAR_DATASET_PATH", "").strip()

if DATASET_PATH_OVERRIDE:
    DATASET_PATH = os.path.abspath(os.path.expanduser(DATASET_PATH_OVERRIDE))
    if DATASET_KEY not in DATASET_PATHS:
        print(
            f"[Config] SONAR_DATASET_PATH override is set; allowing custom dataset key '{DATASET_KEY}' "
            f"with path '{DATASET_PATH}'"
        )
else:
    if DATASET_KEY not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset key '{DATASET_KEY}'. Options: {list(DATASET_PATHS)}")
    DATASET_PATH = DATASET_PATHS[DATASET_KEY]

INIT_SCALE_FACTORS = {
    "legacy": 0.65,
    "r2": 0.6127,
    "synthetic_a_clean": 1.0,
    "synthetic_c_clean": 1.0,
}

is_synthetic_key = DATASET_KEY in SYNTHETIC_DATASET_KEYS or DATASET_KEY.startswith("synthetic")
is_synthetic_path = "synthetic" in os.path.basename(DATASET_PATH).lower()
IS_SYNTHETIC_DATASET = is_synthetic_key or is_synthetic_path

INIT_SCALE_FACTOR_DEFAULT = INIT_SCALE_FACTORS.get(
    DATASET_KEY,
    1.0 if IS_SYNTHETIC_DATASET else INIT_SCALE_FACTORS["r2"],
)
init_scale_override = os.environ.get("SONAR_INIT_SCALE_FACTOR", "").strip()
if init_scale_override:
    INIT_SCALE_FACTOR = float(init_scale_override)
else:
    INIT_SCALE_FACTOR = INIT_SCALE_FACTOR_DEFAULT

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


def env_optional_int(name, default=None):
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


@dataclass(frozen=True)
class ElevationStage1Config:
    elevation_aware: bool
    stage1_mode: str
    effective_stage1_mode: str
    bins: int
    frames_per_iter: int
    frame_sampler: str
    overlap_topk_build: int
    overlap_topk_use: int
    overlap_min_baseline: float
    overlap_max_yaw_deg: float
    overlap_min_score: float
    overlap_score_mode: str
    overlap_score_w_yaw: float
    overlap_score_w_base: float
    pixels_per_frame: int
    logit_lr: float
    bank_refresh_interval: int
    bank_remap_mode: str
    bank_remap_max_dist: int
    resume_pixellogit_mismatch: str
    temp_start: float
    temp_end: float
    temp_post_mode: str
    temp_post_start: float
    temp_post_end: float
    anneal_iters: int
    anneal_iters_is_explicit: bool
    lik_tgt_temp: float
    lik_weight: float
    entropy_weight: float
    lik_norm_p_lo: float
    lik_norm_p_hi: float
    lik_log_eps: float
    lik_log_floor: float
    lik_min_support: float
    lik_use_frame_reliability: bool
    lik_invalid_mode: str
    rel_floor: float
    rel_valid_min: float
    rel_valid_max: float
    rel_dyn_min: float
    rel_dyn_max: float


def _require_config(name, condition, message):
    if not condition:
        raise ValueError(f"Invalid {name}: {message}")


def parse_elevation_stage1_config(stage2_iters):
    elevation_aware = env_bool("ELEVATION_AWARE", True)
    stage1_mode = env_choice("ELEV_STAGE1_MODE", "shadow", {"off", "shadow", "active"})
    effective_stage1_mode = resolve_effective_stage1_mode(
        elevation_aware=elevation_aware,
        requested_mode=stage1_mode,
    )

    bins = env_int("ELEV_BINS", 7)
    frames_per_iter = env_int("ELEV_FRAMES_PER_ITER", 3)
    frame_sampler = env_choice("ELEV_FRAME_SAMPLER", "round_robin", {"round_robin"})
    overlap_topk_build = env_int("ELEV_OVERLAP_TOPK_BUILD", 24)
    overlap_topk_use = env_int("ELEV_OVERLAP_TOPK_USE", 6)
    overlap_min_baseline = env_float("ELEV_OVERLAP_MIN_BASELINE", 0.06)
    overlap_max_yaw_deg = env_float("ELEV_OVERLAP_MAX_YAW_DEG", 40.0)
    overlap_min_score = env_float("ELEV_OVERLAP_MIN_SCORE", 0.30)
    overlap_score_mode = env_choice("ELEV_OVERLAP_SCORE_MODE", "pose_only", {"pose_only"})
    overlap_score_w_yaw = env_float("ELEV_OVERLAP_SCORE_W_YAW", 0.6)
    overlap_score_w_base = env_float("ELEV_OVERLAP_SCORE_W_BASE", 0.4)
    pixels_per_frame = env_int("ELEV_PIXELS_PER_FRAME", 2000)
    logit_lr = env_float("ELEV_LOGIT_LR", 2e-3)
    bank_refresh_interval = env_int("ELEV_BANK_REFRESH_INTERVAL", 0)
    bank_remap_mode = env_choice("ELEV_BANK_REMAP_MODE", "nearest", {"nearest", "reset"})
    bank_remap_max_dist = env_int("ELEV_BANK_REMAP_MAX_DIST", 6)
    resume_pixellogit_mismatch = env_choice(
        "ELEV_RESUME_PIXELLOGIT_MISMATCH",
        "strict",
        {"strict", "reset_frame", "reset_all"},
    )
    temp_start = env_float("ELEV_TEMP_START", 2.0)
    temp_end = env_float("ELEV_TEMP_END", 0.1)
    temp_post_mode = env_choice("ELEV_TEMP_POST_MODE", "shared", {"shared", "decoupled"})
    temp_post_start = env_float("ELEV_TEMP_POST_START", 2.0)
    temp_post_end = env_float("ELEV_TEMP_POST_END", 0.1)
    anneal_iters_raw = env_optional_int("ELEV_ANNEAL_ITERS", None)
    anneal_iters_is_explicit = anneal_iters_raw is not None
    anneal_iters = int(anneal_iters_raw) if anneal_iters_is_explicit else int(stage2_iters)
    lik_tgt_temp = env_float("ELEV_LIK_TGT_TEMP", 1.0)
    lik_weight = env_float("ELEV_LIK_WEIGHT", 1.0)
    entropy_weight = env_float("ELEV_ENTROPY_WEIGHT", 0.01)
    lik_norm_p_lo = env_float("ELEV_LIK_NORM_P_LO", 10.0)
    lik_norm_p_hi = env_float("ELEV_LIK_NORM_P_HI", 99.0)
    lik_log_eps = env_float("ELEV_LIK_LOG_EPS", 1e-3)
    lik_log_floor = env_float("ELEV_LIK_LOG_FLOOR", -6.9)
    lik_min_support = env_float("ELEV_LIK_MIN_SUPPORT", 1e-6)
    lik_use_frame_reliability = env_bool("ELEV_LIK_USE_FRAME_RELIABILITY", True)
    lik_invalid_mode = env_choice("ELEV_LIK_INVALID_MODE", "neutral", {"neutral"})
    rel_floor = env_float("ELEV_REL_FLOOR", 0.3)
    rel_valid_min = env_float("ELEV_REL_VALID_MIN", 0.03)
    rel_valid_max = env_float("ELEV_REL_VALID_MAX", 0.30)
    rel_dyn_min = env_float("ELEV_REL_DYN_MIN", 0.08)
    rel_dyn_max = env_float("ELEV_REL_DYN_MAX", 0.50)

    _require_config("ELEV_BINS", bins >= 2, "must be >= 2")
    _require_config("ELEV_FRAMES_PER_ITER", frames_per_iter >= 1, "must be >= 1")
    _require_config("ELEV_OVERLAP_TOPK_BUILD", overlap_topk_build >= 1, "must be >= 1")
    _require_config("ELEV_OVERLAP_TOPK_USE", overlap_topk_use >= 1, "must be >= 1")
    _require_config(
        "ELEV_OVERLAP_TOPK_USE",
        overlap_topk_use <= overlap_topk_build,
        "must be <= ELEV_OVERLAP_TOPK_BUILD",
    )
    _require_config("ELEV_OVERLAP_MIN_BASELINE", overlap_min_baseline >= 0.0, "must be >= 0")
    _require_config(
        "ELEV_OVERLAP_MAX_YAW_DEG",
        0.0 <= overlap_max_yaw_deg <= 180.0,
        "must be in [0, 180]",
    )
    _require_config("ELEV_OVERLAP_MIN_SCORE", overlap_min_score >= 0.0, "must be >= 0")
    _require_config("ELEV_OVERLAP_SCORE_W_YAW", overlap_score_w_yaw >= 0.0, "must be >= 0")
    _require_config("ELEV_OVERLAP_SCORE_W_BASE", overlap_score_w_base >= 0.0, "must be >= 0")
    _require_config(
        "ELEV_OVERLAP_SCORE_WEIGHTS",
        (overlap_score_w_yaw + overlap_score_w_base) > 0.0,
        "sum of yaw/base weights must be > 0",
    )
    _require_config("ELEV_PIXELS_PER_FRAME", pixels_per_frame >= 1, "must be >= 1")
    _require_config("ELEV_LOGIT_LR", logit_lr > 0.0, "must be > 0")
    _require_config("ELEV_BANK_REFRESH_INTERVAL", bank_refresh_interval >= 0, "must be >= 0")
    _require_config("ELEV_BANK_REMAP_MAX_DIST", bank_remap_max_dist >= 0, "must be >= 0")
    _require_config("ELEV_TEMP_START", temp_start > 0.0, "must be > 0")
    _require_config("ELEV_TEMP_END", temp_end > 0.0, "must be > 0")
    _require_config("ELEV_TEMP_POST_START", temp_post_start > 0.0, "must be > 0")
    _require_config("ELEV_TEMP_POST_END", temp_post_end > 0.0, "must be > 0")
    _require_config("ELEV_ANNEAL_ITERS", anneal_iters >= 0, "must be >= 0")
    _require_config("ELEV_LIK_TGT_TEMP", lik_tgt_temp > 0.0, "must be > 0")
    _require_config("ELEV_LIK_WEIGHT", lik_weight >= 0.0, "must be >= 0")
    _require_config("ELEV_ENTROPY_WEIGHT", entropy_weight >= 0.0, "must be >= 0")
    _require_config("ELEV_LIK_NORM_P_LO", 0.0 <= lik_norm_p_lo < 100.0, "must be in [0, 100)")
    _require_config("ELEV_LIK_NORM_P_HI", 0.0 < lik_norm_p_hi <= 100.0, "must be in (0, 100]")
    _require_config("ELEV_LIK_NORM_PERCENTILES", lik_norm_p_lo < lik_norm_p_hi, "must satisfy p_lo < p_hi")
    _require_config("ELEV_LIK_LOG_EPS", lik_log_eps > 0.0, "must be > 0")
    _require_config("ELEV_LIK_MIN_SUPPORT", lik_min_support > 0.0, "must be > 0")
    _require_config("ELEV_REL_FLOOR", 0.0 <= rel_floor <= 1.0, "must be in [0, 1]")
    _require_config(
        "ELEV_REL_VALID_BOUNDS",
        0.0 <= rel_valid_min <= rel_valid_max <= 1.0,
        "must satisfy 0 <= min <= max <= 1",
    )
    _require_config("ELEV_REL_DYN_BOUNDS", rel_dyn_min <= rel_dyn_max, "must satisfy min <= max")

    return ElevationStage1Config(
        elevation_aware=elevation_aware,
        stage1_mode=stage1_mode,
        effective_stage1_mode=effective_stage1_mode,
        bins=bins,
        frames_per_iter=frames_per_iter,
        frame_sampler=frame_sampler,
        overlap_topk_build=overlap_topk_build,
        overlap_topk_use=overlap_topk_use,
        overlap_min_baseline=overlap_min_baseline,
        overlap_max_yaw_deg=overlap_max_yaw_deg,
        overlap_min_score=overlap_min_score,
        overlap_score_mode=overlap_score_mode,
        overlap_score_w_yaw=overlap_score_w_yaw,
        overlap_score_w_base=overlap_score_w_base,
        pixels_per_frame=pixels_per_frame,
        logit_lr=logit_lr,
        bank_refresh_interval=bank_refresh_interval,
        bank_remap_mode=bank_remap_mode,
        bank_remap_max_dist=bank_remap_max_dist,
        resume_pixellogit_mismatch=resume_pixellogit_mismatch,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_post_mode=temp_post_mode,
        temp_post_start=temp_post_start,
        temp_post_end=temp_post_end,
        anneal_iters=anneal_iters,
        anneal_iters_is_explicit=anneal_iters_is_explicit,
        lik_tgt_temp=lik_tgt_temp,
        lik_weight=lik_weight,
        entropy_weight=entropy_weight,
        lik_norm_p_lo=lik_norm_p_lo,
        lik_norm_p_hi=lik_norm_p_hi,
        lik_log_eps=lik_log_eps,
        lik_log_floor=lik_log_floor,
        lik_min_support=lik_min_support,
        lik_use_frame_reliability=lik_use_frame_reliability,
        lik_invalid_mode=lik_invalid_mode,
        rel_floor=rel_floor,
        rel_valid_min=rel_valid_min,
        rel_valid_max=rel_valid_max,
        rel_dyn_min=rel_dyn_min,
        rel_dyn_max=rel_dyn_max,
    )


@dataclass(frozen=True)
class ElevationChunk4Config:
    couple_mode: str
    support_mode: str
    effective_couple_mode: str
    effective_support_mode: str
    couple_weight_start: float
    couple_weight_end: float
    couple_warmup: int
    couple_max_pix_err: float
    couple_max_depth_err: float
    couple_huber_delta: float
    couple_sigma_mode: str
    couple_sigma_pix: float
    couple_sigma_depth: float
    couple_min_w: float
    couple_max_candidates: int
    support_warmup_iters: int
    support_late_phase_start_iter: int
    support_use_persistent_ids: bool
    support_use_ratio: bool
    support_use_new_surfel_grace: bool
    support_new_surfel_grace_iters: int
    support_min_ratio_mid: float
    support_min_ratio_late: float
    support_min_count_mid: int
    support_min_count_late: int
    support_view_angle_min_deg: float
    support_residual_thresh: float
    support_ema_decay: float
    support_prune_patience: int
    support_resume_mismatch_policy: str
    surfel_id_asserts: bool


def _scaled_warmup_default(stage2_iters, frac, min_iters, max_iters):
    total = max(0, int(stage2_iters))
    if total <= 0:
        return 0
    scaled = int(round(float(frac) * float(total)))
    scaled = max(int(min_iters), scaled)
    scaled = min(int(max_iters), scaled)
    return min(total, scaled)


def parse_elevation_chunk4_config(elevation_aware, stage2_iters):
    couple_mode = env_choice("ELEV_COUPLE_MODE", "shadow", {"off", "shadow", "active"})
    support_mode = env_choice("ELEV_SUPPORT_MODE", "shadow", {"off", "shadow", "active"})
    effective_couple_mode, effective_support_mode = resolve_effective_chunk4_modes(
        elevation_aware=elevation_aware,
        requested_couple_mode=couple_mode,
        requested_support_mode=support_mode,
    )

    couple_warmup_default = _scaled_warmup_default(
        stage2_iters=stage2_iters,
        frac=0.30,
        min_iters=50,
        max_iters=400,
    )
    support_warmup_default = _scaled_warmup_default(
        stage2_iters=stage2_iters,
        frac=0.50,
        min_iters=100,
        max_iters=600,
    )
    if int(stage2_iters) > 0:
        support_late_default = max(
            support_warmup_default,
            _scaled_warmup_default(
                stage2_iters=stage2_iters,
                frac=0.80,
                min_iters=support_warmup_default,
                max_iters=max(int(stage2_iters), support_warmup_default),
            ),
        )
    else:
        support_late_default = 0

    couple_weight_start = env_float("ELEV_COUPLE_WEIGHT_START", 0.10)
    couple_weight_end = env_float("ELEV_COUPLE_WEIGHT_END", 0.50)
    couple_warmup = env_int("ELEV_COUPLE_WARMUP", couple_warmup_default)
    couple_max_pix_err = env_float("ELEV_COUPLE_MAX_PIX_ERR", 3.0)
    couple_max_depth_err = env_float("ELEV_COUPLE_MAX_DEPTH_ERR", 0.08)
    couple_huber_delta = env_float("ELEV_COUPLE_HUBER_DELTA", 0.03)
    couple_sigma_mode = env_choice("ELEV_COUPLE_SIGMA_MODE", "fixed", {"fixed"})
    couple_sigma_pix = env_float("ELEV_COUPLE_SIGMA_PIX", 2.0)
    couple_sigma_depth = env_float("ELEV_COUPLE_SIGMA_DEPTH", 0.05)
    couple_min_w = env_float("ELEV_COUPLE_MIN_W", 0.10)
    couple_max_candidates = env_int("ELEV_COUPLE_MAX_CANDIDATES", 2048)

    support_warmup_iters = env_int("ELEV_SUPPORT_WARMUP_ITERS", support_warmup_default)
    support_late_phase_start_iter = env_int("ELEV_SUPPORT_LATE_PHASE_START_ITERS", support_late_default)
    support_use_persistent_ids = env_bool("ELEV_SUPPORT_USE_PERSISTENT_IDS", True)
    support_use_ratio = env_bool("ELEV_SUPPORT_USE_RATIO", True)
    support_use_new_surfel_grace = env_bool("ELEV_SUPPORT_USE_NEW_SURFEL_GRACE", True)
    support_new_surfel_grace_iters = env_int("ELEV_SUPPORT_NEW_SURFEL_GRACE_ITERS", 1500)
    support_min_ratio_mid = env_float("ELEV_SUPPORT_MIN_RATIO_MID", 0.25)
    support_min_ratio_late = env_float("ELEV_SUPPORT_MIN_RATIO_LATE", 0.45)
    support_min_count_mid = env_int("ELEV_SUPPORT_MIN_COUNT_MID", 2)
    support_min_count_late = env_int("ELEV_SUPPORT_MIN_COUNT_LATE", 4)
    support_view_angle_min_deg = env_float("ELEV_SUPPORT_VIEW_ANGLE_MIN_DEG", 8.0)
    support_residual_thresh = env_float("ELEV_SUPPORT_RESIDUAL_THRESH", 0.20)
    support_ema_decay = env_float("ELEV_SUPPORT_EMA_DECAY", 0.90)
    support_prune_patience = env_int("ELEV_SUPPORT_PRUNE_PATIENCE", 4)
    support_resume_mismatch_policy = env_choice(
        "ELEV_CHUNK4_RESUME_MISMATCH",
        "strict",
        {"strict", "reset_chunk4", "reset_all"},
    )
    surfel_id_asserts = env_bool("ELEV_SURFEL_ID_ASSERTS", True)

    _require_config("ELEV_COUPLE_WEIGHT_START", couple_weight_start >= 0.0, "must be >= 0")
    _require_config("ELEV_COUPLE_WEIGHT_END", couple_weight_end >= 0.0, "must be >= 0")
    _require_config("ELEV_COUPLE_WARMUP", couple_warmup >= 0, "must be >= 0")
    _require_config("ELEV_COUPLE_MAX_PIX_ERR", couple_max_pix_err > 0.0, "must be > 0")
    _require_config("ELEV_COUPLE_MAX_DEPTH_ERR", couple_max_depth_err > 0.0, "must be > 0")
    _require_config("ELEV_COUPLE_HUBER_DELTA", couple_huber_delta > 0.0, "must be > 0")
    _require_config("ELEV_COUPLE_SIGMA_PIX", couple_sigma_pix > 0.0, "must be > 0")
    _require_config("ELEV_COUPLE_SIGMA_DEPTH", couple_sigma_depth > 0.0, "must be > 0")
    _require_config("ELEV_COUPLE_MIN_W", 0.0 <= couple_min_w <= 1.0, "must be in [0, 1]")
    _require_config("ELEV_COUPLE_MAX_CANDIDATES", couple_max_candidates >= 0, "must be >= 0")
    _require_config("ELEV_SUPPORT_WARMUP_ITERS", support_warmup_iters >= 0, "must be >= 0")
    _require_config(
        "ELEV_SUPPORT_LATE_PHASE_START_ITERS",
        support_late_phase_start_iter >= support_warmup_iters,
        "must be >= ELEV_SUPPORT_WARMUP_ITERS",
    )
    _require_config(
        "ELEV_SUPPORT_NEW_SURFEL_GRACE_ITERS",
        support_new_surfel_grace_iters >= 0,
        "must be >= 0",
    )
    _require_config("ELEV_SUPPORT_MIN_RATIO_MID", 0.0 <= support_min_ratio_mid <= 1.0, "must be in [0, 1]")
    _require_config("ELEV_SUPPORT_MIN_RATIO_LATE", 0.0 <= support_min_ratio_late <= 1.0, "must be in [0, 1]")
    _require_config(
        "ELEV_SUPPORT_MIN_RATIO_PHASES",
        support_min_ratio_mid <= support_min_ratio_late,
        "must satisfy mid <= late",
    )
    _require_config("ELEV_SUPPORT_MIN_COUNT_MID", support_min_count_mid >= 0, "must be >= 0")
    _require_config("ELEV_SUPPORT_MIN_COUNT_LATE", support_min_count_late >= 0, "must be >= 0")
    _require_config(
        "ELEV_SUPPORT_MIN_COUNT_PHASES",
        support_min_count_mid <= support_min_count_late,
        "must satisfy mid <= late",
    )
    _require_config("ELEV_SUPPORT_VIEW_ANGLE_MIN_DEG", support_view_angle_min_deg >= 0.0, "must be >= 0")
    _require_config("ELEV_SUPPORT_RESIDUAL_THRESH", support_residual_thresh > 0.0, "must be > 0")
    _require_config("ELEV_SUPPORT_EMA_DECAY", 0.0 <= support_ema_decay <= 1.0, "must be in [0, 1]")
    _require_config("ELEV_SUPPORT_PRUNE_PATIENCE", support_prune_patience >= 1, "must be >= 1")

    return ElevationChunk4Config(
        couple_mode=couple_mode,
        support_mode=support_mode,
        effective_couple_mode=effective_couple_mode,
        effective_support_mode=effective_support_mode,
        couple_weight_start=couple_weight_start,
        couple_weight_end=couple_weight_end,
        couple_warmup=couple_warmup,
        couple_max_pix_err=couple_max_pix_err,
        couple_max_depth_err=couple_max_depth_err,
        couple_huber_delta=couple_huber_delta,
        couple_sigma_mode=couple_sigma_mode,
        couple_sigma_pix=couple_sigma_pix,
        couple_sigma_depth=couple_sigma_depth,
        couple_min_w=couple_min_w,
        couple_max_candidates=couple_max_candidates,
        support_warmup_iters=support_warmup_iters,
        support_late_phase_start_iter=support_late_phase_start_iter,
        support_use_persistent_ids=support_use_persistent_ids,
        support_use_ratio=support_use_ratio,
        support_use_new_surfel_grace=support_use_new_surfel_grace,
        support_new_surfel_grace_iters=support_new_surfel_grace_iters,
        support_min_ratio_mid=support_min_ratio_mid,
        support_min_ratio_late=support_min_ratio_late,
        support_min_count_mid=support_min_count_mid,
        support_min_count_late=support_min_count_late,
        support_view_angle_min_deg=support_view_angle_min_deg,
        support_residual_thresh=support_residual_thresh,
        support_ema_decay=support_ema_decay,
        support_prune_patience=support_prune_patience,
        support_resume_mismatch_policy=support_resume_mismatch_policy,
        surfel_id_asserts=surfel_id_asserts,
    )


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
SONAR_FREEZE_SCALE = env_bool("SONAR_FREEZE_SCALE", IS_SYNTHETIC_DATASET)
ELEV_STAGE1_CFG = parse_elevation_stage1_config(STAGE2_ITERATIONS)
ELEV_CHUNK4_CFG = parse_elevation_chunk4_config(ELEV_STAGE1_CFG.elevation_aware, STAGE2_ITERATIONS)

if SONAR_FREEZE_SCALE and STAGE1_ITERATIONS > 0:
    STAGE1_ITERATIONS = 0

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
SONAR_SURFEL_STATS_EVERY = max(1, env_int("SONAR_SURFEL_STATS_EVERY", 50))
SONAR_LOAD_CHECKPOINT = os.environ.get("SONAR_LOAD_CHECKPOINT", "").strip()
SONAR_SAVE_CHECKPOINT = os.environ.get("SONAR_SAVE_CHECKPOINT", "").strip()
VISUALIZER_MAX_GLYPHS = max(1, env_int("SONAR_VIS_MAX_GLYPHS", 2000))
VISUALIZER_MAX_FRAME_GLYPHS = max(1, env_int("SONAR_VIS_FRAME_MAX_GLYPHS", 1500))
VISUALIZER_GLYPH_SEGMENTS = max(6, env_int("SONAR_VIS_GLYPH_SEGMENTS", 12))
VISUALIZER_OPACITY_PERCENTILE = min(100.0, max(0.0, env_float("SONAR_VIS_OPACITY_PERCENTILE", 0.0)))
VISUALIZER_EQ_RADIUS_PERCENTILE = min(100.0, max(0.0, env_float("SONAR_VIS_EQ_RADIUS_PERCENTILE", 100.0)))
VISUALIZER_FACE_OFFSET_SCALE = max(1e-4, env_float("SONAR_VIS_FACE_OFFSET_SCALE", 0.04))
VISUALIZER_NORMAL_STEM_SCALE = max(1e-4, env_float("SONAR_VIS_NORMAL_STEM_SCALE", 0.35))
SONAR_FRAME_SELECTION = env_choice("SONAR_FRAME_SELECTION", "diverse", {"diverse", "first"})

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

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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
    visualizer_dir, visualizer_rendered_dir = ensure_visualizer_dirs(OUTPUT_DIR)
    visualizer_stage_artifacts = {}
    visualizer_frame_artifacts = []

    setup_logging(OUTPUT_DIR)
    init_loss_log(OUTPUT_DIR)
    atexit.register(close_logs)

    print("=" * 60)
    print("DEBUG: Multi-Frame Training with Curriculum Learning")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"Dataset: {DATASET_KEY} ({DATASET_PATH})")
    print(f"Visualizer output: {visualizer_dir}")
    if DATASET_PATH_OVERRIDE:
        print(f"Dataset path override: {DATASET_PATH_OVERRIDE}")
    print(f"Synthetic dataset mode: {IS_SYNTHETIC_DATASET}")
    print(f"Init scale: {INIT_SCALE_FACTOR}")
    print(f"Scale frozen: {SONAR_FREEZE_SCALE}")
    print(f"Num training frames: {NUM_TRAINING_FRAMES}")
    print(f"Frame selection: {SONAR_FRAME_SELECTION}")
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
    if (not ELEV_STAGE1_CFG.elevation_aware) and ELEV_STAGE1_CFG.stage1_mode != "off":
        print(
            "[Elevation Stage 1] ELEVATION_AWARE=0 forces effective mode to off "
            f"(requested={ELEV_STAGE1_CFG.stage1_mode})"
        )
    if (not ELEV_STAGE1_CFG.elevation_aware) and (
        ELEV_CHUNK4_CFG.couple_mode != "off" or ELEV_CHUNK4_CFG.support_mode != "off"
    ):
        print(
            "[Elevation Chunk 4] ELEVATION_AWARE=0 forces coupling/support modes to off "
            f"(requested couple={ELEV_CHUNK4_CFG.couple_mode}, support={ELEV_CHUNK4_CFG.support_mode})"
        )
    anneal_source = "ELEV_ANNEAL_ITERS" if ELEV_STAGE1_CFG.anneal_iters_is_explicit else "SONAR_STAGE2_ITERS"
    print(
        "[Elevation Stage 1] "
        f"mode={ELEV_STAGE1_CFG.stage1_mode}, "
        f"effective_mode={ELEV_STAGE1_CFG.effective_stage1_mode}, "
        f"bins={ELEV_STAGE1_CFG.bins}, "
        f"frames_per_iter={ELEV_STAGE1_CFG.frames_per_iter}, "
        f"sampler={ELEV_STAGE1_CFG.frame_sampler}"
    )
    print(
        "[Elevation Stage 1] "
        f"anneal_horizon={ELEV_STAGE1_CFG.anneal_iters} (source={anneal_source}), "
        f"temp_model={ELEV_STAGE1_CFG.temp_start:.3f}->{ELEV_STAGE1_CFG.temp_end:.3f}, "
        f"temp_post_mode={ELEV_STAGE1_CFG.temp_post_mode}"
    )
    if ELEV_STAGE1_CFG.temp_post_mode == "decoupled":
        print(
            "[Elevation Stage 1] "
            f"temp_post={ELEV_STAGE1_CFG.temp_post_start:.3f}->{ELEV_STAGE1_CFG.temp_post_end:.3f}"
        )
    print(
        "[Elevation Stage 1] "
        f"lik_weight={ELEV_STAGE1_CFG.lik_weight:.4f}, "
        f"entropy_weight={ELEV_STAGE1_CFG.entropy_weight:.4f}, "
        f"resume_policy={ELEV_STAGE1_CFG.resume_pixellogit_mismatch}, "
        f"refresh_interval={ELEV_STAGE1_CFG.bank_refresh_interval}, "
        f"remap_mode={ELEV_STAGE1_CFG.bank_remap_mode}"
    )
    print(
        "[Elevation Chunk 4] "
        f"couple_mode={ELEV_CHUNK4_CFG.couple_mode}, "
        f"support_mode={ELEV_CHUNK4_CFG.support_mode}, "
        f"effective_couple={ELEV_CHUNK4_CFG.effective_couple_mode}, "
        f"effective_support={ELEV_CHUNK4_CFG.effective_support_mode}"
    )
    print(
        "[Elevation Chunk 4] "
        f"couple_weight={ELEV_CHUNK4_CFG.couple_weight_start:.3f}->{ELEV_CHUNK4_CFG.couple_weight_end:.3f}, "
        f"warmup={ELEV_CHUNK4_CFG.couple_warmup}, "
        f"gates=(pix<={ELEV_CHUNK4_CFG.couple_max_pix_err:.2f}, depth<={ELEV_CHUNK4_CFG.couple_max_depth_err:.3f}), "
        f"sigma=({ELEV_CHUNK4_CFG.couple_sigma_pix:.2f}, {ELEV_CHUNK4_CFG.couple_sigma_depth:.3f}), "
        f"huber_delta={ELEV_CHUNK4_CFG.couple_huber_delta:.3f}, min_w={ELEV_CHUNK4_CFG.couple_min_w:.2f}, "
        f"max_candidates={ELEV_CHUNK4_CFG.couple_max_candidates}"
    )
    if ELEV_STAGE1_CFG.effective_stage1_mode == "off" and ELEV_CHUNK4_CFG.effective_couple_mode != "off":
        print(
            "[Elevation Chunk 4] coupling diagnostics are requested but Stage-1 is off; "
            "coupling will remain inactive because no posterior cache is produced"
        )
    print(
        "[Elevation Chunk 4] support runtime wiring: "
        f"persistent_ids={int(ELEV_CHUNK4_CFG.support_use_persistent_ids)}, "
        f"hard_prune_active={int(mode_enables_hard_prune(ELEV_CHUNK4_CFG.effective_support_mode))}"
    )
    if ELEV_STAGE1_CFG.effective_stage1_mode == "off":
        print("[Elevation Stage 1] effective_mode=off -> sampler fallback=legacy single-frame shuffled path")
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
    frame_indices = select_frame_indices(
        train_cameras,
        NUM_TRAINING_FRAMES,
        seed=SEED,
        mode=SONAR_FRAME_SELECTION,
    )
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
            holdout_rel_indices = select_frame_indices(
                holdout_pool,
                holdout_count,
                seed=SEED + 1000,
                mode=SONAR_FRAME_SELECTION,
            )
            holdout_indices = [remaining_indices[idx] for idx in holdout_rel_indices]
            holdout_frames = [train_cameras[idx] for idx in holdout_indices]

    print(f"Selected {len(training_frames)} training frames:")
    for i, cam in enumerate(training_frames):
        print(f"  [{i}] {cam.image_name}")

    if holdout_frames:
        print(f"Selected {len(holdout_frames)} holdout frames:")
        for i, cam in enumerate(holdout_frames):
            print(f"  [H{i}] {cam.image_name}")

    active_frame_keys = [str(cam.image_name) for cam in training_frames]
    assert_frame_keys_unique(active_frame_keys)
    active_frame_fingerprint = compute_active_frame_fingerprint(active_frame_keys)
    print(f"[Elevation Stage 1] active_frame_fingerprint={active_frame_fingerprint}")
    frame_key_to_index = {frame_key: idx for idx, frame_key in enumerate(active_frame_keys)}
    elev_sampler_batch_size = min(ELEV_STAGE1_CFG.frames_per_iter, len(active_frame_keys))
    elev_sampler_state = RoundRobinSamplerState(frame_keys=list(active_frame_keys), cursor=0, epoch=0)

    if ELEV_STAGE1_CFG.effective_stage1_mode == "off":
        overlap_table = {}
        overlap_build_params = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "disabled": True,
            "reason": "effective_stage1_mode=off",
        }
        summarize_overlap_table(overlap_table, active_frame_keys, label="disabled")
    else:
        overlap_table, overlap_build_params = build_pose_overlap_table(training_frames, ELEV_STAGE1_CFG)
        summarize_overlap_table(overlap_table, active_frame_keys, label="built")

    if ELEV_STAGE1_CFG.effective_stage1_mode == "off":
        pixel_bank = {}
        gt_frame_cache = {}
        frame_stats_cache = {}
        pixel_logits_registry = {}
        print("[Elevation Stage 1] pixel_bank/logits disabled in off mode")
    else:
        pixel_bank, gt_frame_cache = build_pixel_bank(training_frames, ELEV_STAGE1_CFG.pixels_per_frame)
        frame_stats_cache = build_frame_stats_cache(gt_frame_cache, ELEV_STAGE1_CFG)
        pixel_logits_registry = {}
        print(
            "[Elevation Stage 1] pixel_bank built: "
            f"frames={len(pixel_bank)}, pixels/frame~{ELEV_STAGE1_CFG.pixels_per_frame}, "
            f"logits_bins={ELEV_STAGE1_CFG.bins}"
        )
        if SONAR_LOAD_CHECKPOINT:
            print(
                "[Elevation Stage 1] checkpoint load requested; "
                "deferring pixel-logit initialization until resume handling"
            )
        else:
            pixel_logits_registry, restored_logits_count, reset_logits_count = build_pixel_logits_registry(
                pixel_bank,
                bins=ELEV_STAGE1_CFG.bins,
                loaded_pixel_logits=None,
                mismatch_policy="reset_all",
            )
            if restored_logits_count or reset_logits_count:
                print(
                    "[Elevation Stage 1] pixel_logits init: "
                    f"restored={restored_logits_count}, reset={reset_logits_count}"
                )

    elevation_stage1_runtime_state = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "active_frame_keys": list(active_frame_keys),
        "active_frame_fingerprint": active_frame_fingerprint,
        "overlap_table": overlap_table,
        "overlap_build_params": overlap_build_params,
        "sampler_state": serialize_sampler_state(elev_sampler_state),
        "pixel_bank": serialize_pixel_bank(pixel_bank),
        "pixel_logits": serialize_pixel_logits_registry(pixel_logits_registry),
        "optim_elev_state": None,
    }

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
    visualizer_stage_artifacts["initial"] = export_stage_visualizer_state(
        gaussians,
        visualizer_dir,
        stage_filename="surfels_initial_state.ply",
        stage_status="executed",
        glyph_prefix="initial",
        include_centers_full=True,
    )

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
    optim_elev = build_optim_elev(pixel_logits_registry, ELEV_STAGE1_CFG.logit_lr)

    training_iter_offset = 0
    chunk4_resume_enabled = bool(ELEV_CHUNK4_CFG.support_use_persistent_ids)
    chunk4_runtime_state = None
    if chunk4_resume_enabled:
        chunk4_runtime_state = init_chunk4_runtime_state(gaussians, ELEV_CHUNK4_CFG, init_iter=0)
        if chunk4_runtime_state is not None:
            print(
                "[Elevation Chunk 4] runtime initialized: "
                f"rows={int(gaussians.get_xyz.shape[0])}, "
                f"schema={CHUNK4_CHECKPOINT_SCHEMA_VERSION}"
            )
    elif not bool(ELEV_CHUNK4_CFG.support_use_persistent_ids):
        print("[Elevation Chunk 4] runtime state skipped: persistent IDs disabled")

    if SONAR_LOAD_CHECKPOINT:
        resumed_iter, resume_meta, resume_stage1_state, resume_chunk4_state = load_training_checkpoint(
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

        if chunk4_resume_enabled:
            chunk4_runtime_state = init_chunk4_runtime_state(
                gaussians,
                ELEV_CHUNK4_CFG,
                init_iter=training_iter_offset,
            )
            if resume_chunk4_state is None:
                print("[Elevation Chunk 4] checkpoint has no state; using fresh runtime state")
            else:
                chunk4_resume_schema = str(resume_chunk4_state.get("checkpoint_schema_version", ""))
                chunk4_resume_fingerprint = str(resume_chunk4_state.get("active_frame_fingerprint", ""))
                chunk4_resume_fingerprint_matches = chunk4_resume_fingerprint == active_frame_fingerprint
                chunk4_resume_action = resolve_chunk4_resume_action(
                    checkpoint_schema_version=chunk4_resume_schema,
                    runtime_schema_version=CHUNK4_CHECKPOINT_SCHEMA_VERSION,
                    frame_fingerprint_matches=chunk4_resume_fingerprint_matches,
                    mismatch_policy=ELEV_CHUNK4_CFG.support_resume_mismatch_policy,
                )
                if chunk4_resume_action == "load":
                    try:
                        chunk4_runtime_state = restore_chunk4_runtime_state_from_checkpoint(
                            resume_chunk4_state,
                            gaussians,
                            ELEV_CHUNK4_CFG,
                        )
                        print(
                            "[Elevation Chunk 4] loaded runtime state: "
                            f"rows={int(chunk4_runtime_state['persistent_state']['surfel_ids'].shape[0])}, "
                            f"next_id={int(chunk4_runtime_state['persistent_state']['next_surfel_id'])}, "
                            f"fingerprint_match={int(chunk4_resume_fingerprint_matches)}"
                        )
                    except ValueError as exc:
                        if ELEV_CHUNK4_CFG.support_resume_mismatch_policy == "strict":
                            raise
                        chunk4_runtime_state = init_chunk4_runtime_state(
                            gaussians,
                            ELEV_CHUNK4_CFG,
                            init_iter=training_iter_offset,
                        )
                        print(
                            "[Elevation Chunk 4] reset runtime state after invalid payload: "
                            f"reason={exc}"
                        )
                elif chunk4_resume_action in {"reset_chunk4", "reset_all"}:
                    chunk4_runtime_state = init_chunk4_runtime_state(
                        gaussians,
                        ELEV_CHUNK4_CFG,
                        init_iter=training_iter_offset,
                    )
                    print(
                        f"[Elevation Chunk 4] resume action={chunk4_resume_action}; "
                        "using fresh runtime state"
                    )
                else:
                    print(f"[Elevation Chunk 4] resume action={chunk4_resume_action}; state skipped")
        elif resume_chunk4_state is not None:
            print("[Elevation Chunk 4] checkpoint state present but runtime restore is disabled; skipped")

        if resume_stage1_state and ELEV_STAGE1_CFG.effective_stage1_mode != "off":
            resume_schema = str(resume_stage1_state.get("checkpoint_schema_version", ""))
            resume_fingerprint = str(resume_stage1_state.get("active_frame_fingerprint", ""))
            fingerprint_matches = resume_fingerprint == active_frame_fingerprint
            resume_action = resolve_stage1_resume_action(
                checkpoint_schema_version=resume_schema,
                runtime_schema_version=CHECKPOINT_SCHEMA_VERSION,
                frame_fingerprint_matches=fingerprint_matches,
                mismatch_policy=ELEV_STAGE1_CFG.resume_pixellogit_mismatch,
            )
            if resume_action == "load":
                loaded_overlap = resume_stage1_state.get("overlap_table")
                loaded_params = resume_stage1_state.get("overlap_build_params")
                loaded_reason = str(loaded_params.get("reason", "")) if isinstance(loaded_params, dict) else ""
                loaded_disabled = loaded_reason == "effective_stage1_mode=off"
                has_overlap_entries = isinstance(loaded_overlap, dict) and len(loaded_overlap) > 0
                overlap_keys_match = isinstance(loaded_overlap, dict) and set(loaded_overlap.keys()) == set(active_frame_keys)
                if (
                    isinstance(loaded_overlap, dict)
                    and isinstance(loaded_params, dict)
                    and has_overlap_entries
                    and overlap_keys_match
                    and (not loaded_disabled)
                ):
                    overlap_table = loaded_overlap
                    overlap_build_params = loaded_params
                    summarize_overlap_table(overlap_table, active_frame_keys, label="restored")
                    try:
                        elev_sampler_state = restore_sampler_state(
                            resume_stage1_state.get("sampler_state"),
                            active_frame_keys,
                        )
                        print(
                            "[Elevation Stage 1] sampler restored: "
                            f"cursor={elev_sampler_state.cursor}, epoch={elev_sampler_state.epoch}, "
                            f"batch={elev_sampler_batch_size}"
                        )
                    except ValueError:
                        print(
                            "[Elevation Stage 1] checkpoint sampler state mismatch; "
                            "resetting sampler to cursor=0, epoch=0"
                        )
                        elev_sampler_state = RoundRobinSamplerState(
                            frame_keys=list(active_frame_keys),
                            cursor=0,
                            epoch=0,
                        )
                else:
                    reason = "incomplete"
                    if loaded_disabled:
                        reason = "saved_off_mode"
                    elif not has_overlap_entries:
                        reason = "empty_overlap_table"
                    elif not overlap_keys_match:
                        reason = "frame_key_mismatch"
                    print(
                        f"[Elevation Stage 1] checkpoint overlap state rejected ({reason}); "
                        "using freshly built overlap_table"
                    )
            else:
                print(
                    f"[Elevation Stage 1] resume action={resume_action}; "
                    "using freshly built overlap_table"
                )

            loaded_pixel_bank = resume_stage1_state.get("pixel_bank")
            loaded_pixel_logits = resume_stage1_state.get("pixel_logits")
            loaded_optim_elev_state = resume_stage1_state.get("optim_elev_state")

            if ELEV_STAGE1_CFG.resume_pixellogit_mismatch == "strict" and isinstance(loaded_pixel_bank, dict):
                if set(str(k) for k in loaded_pixel_bank.keys()) != set(pixel_bank.keys()):
                    raise ValueError("Pixel-bank frame-key mismatch under strict resume policy")

            pixel_logits_registry, restored_logits_count, reset_logits_count = build_pixel_logits_registry(
                pixel_bank,
                bins=ELEV_STAGE1_CFG.bins,
                loaded_pixel_logits=loaded_pixel_logits,
                mismatch_policy=ELEV_STAGE1_CFG.resume_pixellogit_mismatch,
            )
            optim_state_for_restore = loaded_optim_elev_state if resume_action == "load" else None
            optim_elev = build_optim_elev(
                pixel_logits_registry,
                ELEV_STAGE1_CFG.logit_lr,
                loaded_state=optim_state_for_restore,
                strict_optimizer_state=(ELEV_STAGE1_CFG.resume_pixellogit_mismatch == "strict"),
            )
            print(
                "[Elevation Stage 1] pixel-logit restore: "
                f"restored={restored_logits_count}, reset={reset_logits_count}"
            )
        elif resume_stage1_state and ELEV_STAGE1_CFG.effective_stage1_mode == "off":
            print("[Elevation Stage 1] effective_mode=off; ignoring checkpoint Stage-1 overlap state")
        elif ELEV_STAGE1_CFG.effective_stage1_mode != "off":
            print(
                "[Elevation Stage 1] checkpoint has no Stage-1 state; "
                "initializing fresh pixel logits"
            )

    if ELEV_STAGE1_CFG.effective_stage1_mode != "off" and not pixel_logits_registry:
        pixel_logits_registry, restored_logits_count, reset_logits_count = build_pixel_logits_registry(
            pixel_bank,
            bins=ELEV_STAGE1_CFG.bins,
            loaded_pixel_logits=None,
            mismatch_policy="reset_all",
        )
        optim_elev = build_optim_elev(pixel_logits_registry, ELEV_STAGE1_CFG.logit_lr)
        if restored_logits_count or reset_logits_count:
            print(
                "[Elevation Stage 1] pixel_logits init: "
                f"restored={restored_logits_count}, reset={reset_logits_count}"
            )

    elevation_stage1_runtime_state.update(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "active_frame_keys": list(active_frame_keys),
            "active_frame_fingerprint": active_frame_fingerprint,
            "overlap_table": overlap_table,
            "overlap_build_params": overlap_build_params,
            "sampler_state": serialize_sampler_state(elev_sampler_state),
            "pixel_bank": serialize_pixel_bank(pixel_bank),
            "pixel_logits": serialize_pixel_logits_registry(pixel_logits_registry),
            "optim_elev_state": optim_elev.state_dict() if optim_elev is not None else None,
        }
    )

    if SONAR_FREEZE_SCALE:
        with torch.no_grad():
            sonar_scale_factor._log_scale.fill_(math.log(INIT_SCALE_FACTOR))
        sonar_scale_factor._log_scale.requires_grad_(False)
        for group in scale_optimizer.param_groups:
            group["lr"] = 0.0
        print(f"[Scale] Frozen at configured value: {sonar_scale_factor.get_scale_value():.6f}")

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

        visualizer_stage_artifacts["stage1"] = export_stage_visualizer_state(
            gaussians,
            visualizer_dir,
            stage_filename="surfels_after_stage1.ply",
            stage_status="executed",
        )
    else:
        visualizer_stage_artifacts["stage1"] = export_stage_visualizer_state(
            gaussians,
            visualizer_dir,
            stage_filename="surfels_after_stage1.ply",
            stage_status="skipped_alias_to_initial",
        )

    elev_angle_bins = torch.linspace(
        -sonar_config.half_elevation_rad,
        sonar_config.half_elevation_rad,
        steps=ELEV_STAGE1_CFG.bins,
        device=gaussians.get_xyz.device,
        dtype=torch.float32,
    )
    cached_loglik = {}
    cached_support_mask = {}
    p_post = {}

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

        epoch_indices = []
        stage2_use_round_robin = ELEV_STAGE1_CFG.effective_stage1_mode != "off"
        if stage2_use_round_robin:
            print(
                "[Elevation Stage 1] Stage 2 sampler: round_robin "
                f"batch={elev_sampler_batch_size}, cursor={elev_sampler_state.cursor}, epoch={elev_sampler_state.epoch}"
            )
        else:
            epoch_indices = get_epoch_indices(len(training_frames), SEED)
        stage2_global_offset = training_iter_offset + STAGE1_ITERATIONS
        for iteration in range(1, STAGE2_ITERATIONS + 1):
            global_iter = stage2_global_offset + iteration
            if chunk4_runtime_state is not None:
                ensure_chunk4_state_capacity(
                    chunk4_runtime_state,
                    gaussians,
                    ELEV_CHUNK4_CFG,
                    current_iter=global_iter,
                )
            if stage2_use_round_robin:
                pixel_bank, pixel_logits_registry, optim_elev, _ = maybe_refresh_pixel_bank_and_logits(
                    iteration=global_iter,
                    training_frames=training_frames,
                    active_frame_keys=active_frame_keys,
                    gt_frame_cache=gt_frame_cache,
                    pixel_bank=pixel_bank,
                    pixel_logits_registry=pixel_logits_registry,
                    optim_elev=optim_elev,
                    cfg=ELEV_STAGE1_CFG,
                )
                sampled_frame_keys, elev_sampler_state = round_robin_sample(
                    elev_sampler_state,
                    elev_sampler_batch_size,
                )
                sampled_frame_indices = [frame_key_to_index[key] for key in sampled_frame_keys]
            else:
                # Shuffle frames per epoch
                if (iteration - 1) % len(training_frames) == 0:
                    epoch_seed = SEED + (iteration - 1) // len(training_frames)
                    epoch_indices = get_epoch_indices(len(training_frames), epoch_seed)

                frame_idx = epoch_indices[(iteration - 1) % len(training_frames)]
                sampled_frame_indices = [frame_idx]

            sync_opacity_policy(global_iter, f"stage2/iter{iteration}")

            temp_model_iter, temp_post_iter = resolve_temperatures(
                iteration=global_iter,
                temp_start=ELEV_STAGE1_CFG.temp_start,
                temp_end=ELEV_STAGE1_CFG.temp_end,
                temp_post_mode=ELEV_STAGE1_CFG.temp_post_mode,
                temp_post_start=ELEV_STAGE1_CFG.temp_post_start,
                temp_post_end=ELEV_STAGE1_CFG.temp_post_end,
                horizon=ELEV_STAGE1_CFG.anneal_iters,
            )
            couple_compute_enabled = (
                stage2_use_round_robin
                and ELEV_CHUNK4_CFG.effective_couple_mode in {"shadow", "active"}
            )
            couple_weight_iter = (
                resolve_chunk4_coupling_weight(global_iter, ELEV_CHUNK4_CFG)
                if mode_enables_weighted_coupling(ELEV_CHUNK4_CFG.effective_couple_mode)
                else 0.0
            )

            batch_l1 = []
            batch_ssim = []
            batch_base = []
            batch_bright = []
            batch_loss = []
            batch_loss_lik = []
            batch_loss_ent = []
            batch_stage1 = []
            batch_loss_couple = []
            batch_match_rate = []
            batch_residual_p95 = []
            batch_assoc_w = []
            iter_cached_loglik = {}
            iter_cached_support_mask = {}
            iter_p_post = {}
            iter_support_obs = {}
            for frame_idx in sampled_frame_indices:
                viewpoint_cam = training_frames[frame_idx]
                frame_key = str(viewpoint_cam.image_name)
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

                l1_i = l1_loss(rendered, gt_image)
                ssim_i = ssim(rendered, gt_image)
                base_i = 0.8 * l1_i + 0.2 * (1 - ssim_i)
                bright_i = compute_bright_loss(rendered, gt_image)
                photometric_i = (1 - BRIGHT_WEIGHT) * base_i + BRIGHT_WEIGHT * bright_i

                if stage2_use_round_robin:
                    logits_i = pixel_logits_registry[frame_key]
                    loglik_i, support_mask_i = build_stage1_multiview_loglik(
                        frame_idx=frame_idx,
                        frame_key=frame_key,
                        training_frames=training_frames,
                        frame_key_to_index=frame_key_to_index,
                        overlap_table=overlap_table,
                        pixel_bank=pixel_bank,
                        gt_frame_cache=gt_frame_cache,
                        frame_stats_cache=frame_stats_cache,
                        elev_angle_bins=elev_angle_bins,
                        sonar_config=sonar_config,
                        sonar_scale_factor=sonar_scale_factor,
                        cfg=ELEV_STAGE1_CFG,
                    )
                    stage1_out = run_stage1_likelihood_step(
                        logits=logits_i,
                        loglik=loglik_i,
                        support_mask=support_mask_i,
                        mode=ELEV_STAGE1_CFG.effective_stage1_mode,
                        lik_weight=ELEV_STAGE1_CFG.lik_weight,
                        entropy_weight=ELEV_STAGE1_CFG.entropy_weight,
                        temp_model=temp_model_iter,
                        temp_post=temp_post_iter,
                        lik_tgt_temp=ELEV_STAGE1_CFG.lik_tgt_temp,
                        min_support=ELEV_STAGE1_CFG.lik_min_support,
                    )

                    if stage1_out["cached_loglik"] is not None:
                        iter_cached_loglik[frame_key] = stage1_out["cached_loglik"]
                    if stage1_out["cached_support_mask"] is not None:
                        iter_cached_support_mask[frame_key] = stage1_out["cached_support_mask"]
                    if stage1_out["p_post"] is not None:
                        iter_p_post[frame_key] = stage1_out["p_post"]
                else:
                    zero_stage1 = photometric_i.new_tensor(0.0)
                    stage1_out = {
                        "loss_lik": zero_stage1,
                        "loss_ent": zero_stage1,
                        "stage1_total_loss": zero_stage1,
                    }

                coupling_stats = {
                    "loss": photometric_i.new_tensor(0.0),
                    "match_rate": 0.0,
                    "residual_p95": 0.0,
                    "assoc_w_mean": 0.0,
                }
                if couple_compute_enabled:
                    coupling_stats = compute_chunk4_coupling_for_frame(
                        frame_idx=frame_idx,
                        frame_key=frame_key,
                        training_frames=training_frames,
                        render_pkg=render_pkg,
                        gaussians=gaussians,
                        sonar_config=sonar_config,
                        sonar_scale_factor=sonar_scale_factor,
                        pixel_bank=pixel_bank,
                        p_post_frame=stage1_out.get("p_post"),
                        elev_angle_bins=elev_angle_bins,
                        chunk4_cfg=ELEV_CHUNK4_CFG,
                    )

                loss_couple_i = coupling_stats["loss"]
                loss_i = photometric_i + stage1_out["stage1_total_loss"] + (couple_weight_iter * loss_couple_i)

                frame_loss_sums[frame_idx] += float(loss_i.item())
                frame_loss_counts[frame_idx] += 1
                batch_l1.append(l1_i)
                batch_ssim.append(ssim_i)
                batch_base.append(base_i)
                batch_bright.append(bright_i)
                batch_loss.append(loss_i)
                batch_loss_lik.append(stage1_out["loss_lik"])
                batch_loss_ent.append(stage1_out["loss_ent"])
                batch_stage1.append(stage1_out["stage1_total_loss"])
                batch_loss_couple.append(loss_couple_i)
                batch_match_rate.append(float(coupling_stats["match_rate"]))
                batch_residual_p95.append(float(coupling_stats["residual_p95"]))
                batch_assoc_w.append(float(coupling_stats["assoc_w_mean"]))

                if chunk4_runtime_state is not None:
                    iter_support_obs[frame_idx] = compute_chunk4_support_observations_for_frame(
                        frame_idx=frame_idx,
                        training_frames=training_frames,
                        gaussians=gaussians,
                        render_pkg=render_pkg,
                        rendered=rendered,
                        gt_image=gt_image,
                        sonar_config=sonar_config,
                        sonar_scale_factor=sonar_scale_factor,
                        support_residual_thresh=ELEV_CHUNK4_CFG.support_residual_thresh,
                    )

            if not batch_loss:
                continue

            Ll1 = torch.stack(batch_l1).mean()
            ssim_val = torch.stack(batch_ssim).mean()
            base_loss = torch.stack(batch_base).mean()
            bright_loss = torch.stack(batch_bright).mean()
            loss = torch.stack(batch_loss).mean()
            loss_lik = torch.stack(batch_loss_lik).mean()
            loss_ent = torch.stack(batch_loss_ent).mean()
            loss_stage1 = torch.stack(batch_stage1).mean()
            loss_couple = torch.stack(batch_loss_couple).mean()
            couple_match_rate = float(np.mean(batch_match_rate)) if batch_match_rate else 0.0
            couple_residual_p95 = float(np.mean(batch_residual_p95)) if batch_residual_p95 else 0.0
            couple_assoc_w = float(np.mean(batch_assoc_w)) if batch_assoc_w else 0.0
            cached_loglik = iter_cached_loglik
            cached_support_mask = iter_cached_support_mask
            p_post = iter_p_post

            # Backward
            loss.backward()

            chunk4_support_stats = {
                "match_frames": 0,
                "diverse_frames": 0,
                "support_ratio": 0.0,
                "residual_p95": 0.0,
                "active_support_ge2_frac": 0.0,
                "active_support_ge3_frac": 0.0,
                "pruned_support_count": 0,
            }
            fov_pruned_count = 0

            # Update surfels only
            with torch.no_grad():
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if optim_elev is not None:
                    optim_elev.step()
                    optim_elev.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iteration)

                if chunk4_runtime_state is not None:
                    chunk4_support_stats = update_chunk4_support_runtime(
                        global_iter=global_iter,
                        sampled_frame_indices=sampled_frame_indices,
                        iter_support_obs=iter_support_obs,
                        training_frames=training_frames,
                        gaussians=gaussians,
                        chunk4_runtime_state=chunk4_runtime_state,
                        chunk4_cfg=ELEV_CHUNK4_CFG,
                    )

                # FOV-aware pruning: remove surfels that drifted outside all training FOVs
                if FOV_PRUNE_INTERVAL > 0 and iteration % FOV_PRUNE_INTERVAL == 0:
                    fov_pruned_count = prune_outside_fov(
                        gaussians,
                        training_frames,
                        sonar_config,
                        sonar_scale_factor,
                        chunk4_runtime_state=chunk4_runtime_state,
                        chunk4_cfg=ELEV_CHUNK4_CFG,
                        reason="fov_stage2",
                        current_iter=global_iter,
                    )
                    if fov_pruned_count > 0:
                        sync_opacity_policy(global_iter, f"stage2/fov_prune@{iteration}", force=True)
                        print(
                            f"  [FOV prune] Removed {fov_pruned_count} surfels outside FOV, "
                            f"{len(gaussians.get_xyz)} remaining"
                        )

                if chunk4_runtime_state is not None and bool(ELEV_CHUNK4_CFG.surfel_id_asserts):
                    assert_surfel_id_integrity(chunk4_runtime_state["persistent_state"])

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
                sampler_tail = ""
                if stage2_use_round_robin:
                    sampler_tail = (
                        f", batch={len(sampled_frame_indices)}, sampler_epoch={elev_sampler_state.epoch}, "
                        f"T_model={temp_model_iter:.3f}, T_post={temp_post_iter:.3f}, "
                        f"T_tgt={ELEV_STAGE1_CFG.lik_tgt_temp:.3f}"
                    )
                    if couple_compute_enabled:
                        sampler_tail += (
                            f", couple={loss_couple.item():.6f}, w_couple={couple_weight_iter:.3f}, "
                            f"match={couple_match_rate:.3f}, p95={couple_residual_p95:.3f}m, "
                            f"assoc_w={couple_assoc_w:.3f}"
                        )
                if chunk4_runtime_state is not None:
                    sampler_tail += (
                        f", sup_match={chunk4_support_stats['match_frames']}/{chunk4_support_stats['diverse_frames']}, "
                        f"sup_ratio={chunk4_support_stats['support_ratio']:.3f}, "
                        f"sup_p95={chunk4_support_stats['residual_p95']:.3f}, "
                        f"sup_ge2={chunk4_support_stats['active_support_ge2_frac']:.3f}, "
                        f"sup_ge3={chunk4_support_stats['active_support_ge3_frac']:.3f}, "
                        f"prune={int(chunk4_support_stats['pruned_support_count']) + int(fov_pruned_count)}"
                    )
                print(
                    f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, "
                    f"scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}, "
                    f"lik={loss_lik.item():.6f}, ent={loss_ent.item():.6f}, stage1={loss_stage1.item():.6f}{sampler_tail}"
                )

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
        visualizer_stage_artifacts["stage2"] = export_stage_visualizer_state(
            gaussians,
            visualizer_dir,
            stage_filename="surfels_after_stage2.ply",
            stage_status="executed",
            glyph_prefix="stage2",
        )
    else:
        visualizer_stage_artifacts["stage2"] = export_stage_visualizer_state(
            gaussians,
            visualizer_dir,
            stage_filename="surfels_after_stage2.ply",
            stage_status="skipped_alias_to_stage1",
            glyph_prefix="stage2",
        )

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

        epoch_indices = []
        stage3_use_round_robin = ELEV_STAGE1_CFG.effective_stage1_mode != "off"
        if stage3_use_round_robin:
            print(
                "[Elevation Stage 1] Stage 3 sampler: round_robin "
                f"batch={elev_sampler_batch_size}, cursor={elev_sampler_state.cursor}, epoch={elev_sampler_state.epoch}"
            )
        else:
            epoch_indices = get_epoch_indices(len(training_frames), SEED)
        stage3_global_offset = training_iter_offset + STAGE1_ITERATIONS + STAGE2_ITERATIONS
        for iteration in range(1, STAGE3_ITERATIONS + 1):
            global_iter = stage3_global_offset + iteration
            if chunk4_runtime_state is not None:
                ensure_chunk4_state_capacity(
                    chunk4_runtime_state,
                    gaussians,
                    ELEV_CHUNK4_CFG,
                    current_iter=global_iter,
                )
            if stage3_use_round_robin:
                pixel_bank, pixel_logits_registry, optim_elev, _ = maybe_refresh_pixel_bank_and_logits(
                    iteration=global_iter,
                    training_frames=training_frames,
                    active_frame_keys=active_frame_keys,
                    gt_frame_cache=gt_frame_cache,
                    pixel_bank=pixel_bank,
                    pixel_logits_registry=pixel_logits_registry,
                    optim_elev=optim_elev,
                    cfg=ELEV_STAGE1_CFG,
                )
                sampled_frame_keys, elev_sampler_state = round_robin_sample(
                    elev_sampler_state,
                    elev_sampler_batch_size,
                )
                sampled_frame_indices = [frame_key_to_index[key] for key in sampled_frame_keys]
            else:
                # Shuffle frames per epoch
                if (iteration - 1) % len(training_frames) == 0:
                    epoch_seed = SEED + (iteration - 1) // len(training_frames)
                    epoch_indices = get_epoch_indices(len(training_frames), epoch_seed)

                frame_idx = epoch_indices[(iteration - 1) % len(training_frames)]
                sampled_frame_indices = [frame_idx]

            sync_opacity_policy(global_iter, f"stage3/iter{iteration}")

            temp_model_iter, temp_post_iter = resolve_temperatures(
                iteration=global_iter,
                temp_start=ELEV_STAGE1_CFG.temp_start,
                temp_end=ELEV_STAGE1_CFG.temp_end,
                temp_post_mode=ELEV_STAGE1_CFG.temp_post_mode,
                temp_post_start=ELEV_STAGE1_CFG.temp_post_start,
                temp_post_end=ELEV_STAGE1_CFG.temp_post_end,
                horizon=ELEV_STAGE1_CFG.anneal_iters,
            )
            couple_compute_enabled = (
                stage3_use_round_robin
                and ELEV_CHUNK4_CFG.effective_couple_mode in {"shadow", "active"}
            )
            couple_weight_iter = (
                resolve_chunk4_coupling_weight(global_iter, ELEV_CHUNK4_CFG)
                if mode_enables_weighted_coupling(ELEV_CHUNK4_CFG.effective_couple_mode)
                else 0.0
            )

            batch_l1 = []
            batch_ssim = []
            batch_base = []
            batch_bright = []
            batch_loss = []
            batch_loss_lik = []
            batch_loss_ent = []
            batch_stage1 = []
            batch_loss_couple = []
            batch_match_rate = []
            batch_residual_p95 = []
            batch_assoc_w = []
            iter_cached_loglik = {}
            iter_cached_support_mask = {}
            iter_p_post = {}
            iter_support_obs = {}
            for frame_idx in sampled_frame_indices:
                viewpoint_cam = training_frames[frame_idx]
                frame_key = str(viewpoint_cam.image_name)
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

                l1_i = l1_loss(rendered, gt_image)
                ssim_i = ssim(rendered, gt_image)
                base_i = 0.8 * l1_i + 0.2 * (1 - ssim_i)
                bright_i = compute_bright_loss(rendered, gt_image)
                photometric_i = (1 - BRIGHT_WEIGHT) * base_i + BRIGHT_WEIGHT * bright_i

                if stage3_use_round_robin:
                    logits_i = pixel_logits_registry[frame_key]
                    loglik_i, support_mask_i = build_stage1_multiview_loglik(
                        frame_idx=frame_idx,
                        frame_key=frame_key,
                        training_frames=training_frames,
                        frame_key_to_index=frame_key_to_index,
                        overlap_table=overlap_table,
                        pixel_bank=pixel_bank,
                        gt_frame_cache=gt_frame_cache,
                        frame_stats_cache=frame_stats_cache,
                        elev_angle_bins=elev_angle_bins,
                        sonar_config=sonar_config,
                        sonar_scale_factor=sonar_scale_factor,
                        cfg=ELEV_STAGE1_CFG,
                    )
                    stage1_out = run_stage1_likelihood_step(
                        logits=logits_i,
                        loglik=loglik_i,
                        support_mask=support_mask_i,
                        mode=ELEV_STAGE1_CFG.effective_stage1_mode,
                        lik_weight=ELEV_STAGE1_CFG.lik_weight,
                        entropy_weight=ELEV_STAGE1_CFG.entropy_weight,
                        temp_model=temp_model_iter,
                        temp_post=temp_post_iter,
                        lik_tgt_temp=ELEV_STAGE1_CFG.lik_tgt_temp,
                        min_support=ELEV_STAGE1_CFG.lik_min_support,
                    )

                    if stage1_out["cached_loglik"] is not None:
                        iter_cached_loglik[frame_key] = stage1_out["cached_loglik"]
                    if stage1_out["cached_support_mask"] is not None:
                        iter_cached_support_mask[frame_key] = stage1_out["cached_support_mask"]
                    if stage1_out["p_post"] is not None:
                        iter_p_post[frame_key] = stage1_out["p_post"]
                else:
                    zero_stage1 = photometric_i.new_tensor(0.0)
                    stage1_out = {
                        "loss_lik": zero_stage1,
                        "loss_ent": zero_stage1,
                        "stage1_total_loss": zero_stage1,
                    }

                coupling_stats = {
                    "loss": photometric_i.new_tensor(0.0),
                    "match_rate": 0.0,
                    "residual_p95": 0.0,
                    "assoc_w_mean": 0.0,
                }
                if couple_compute_enabled:
                    coupling_stats = compute_chunk4_coupling_for_frame(
                        frame_idx=frame_idx,
                        frame_key=frame_key,
                        training_frames=training_frames,
                        render_pkg=render_pkg,
                        gaussians=gaussians,
                        sonar_config=sonar_config,
                        sonar_scale_factor=sonar_scale_factor,
                        pixel_bank=pixel_bank,
                        p_post_frame=stage1_out.get("p_post"),
                        elev_angle_bins=elev_angle_bins,
                        chunk4_cfg=ELEV_CHUNK4_CFG,
                    )

                loss_couple_i = coupling_stats["loss"]
                loss_i = photometric_i + stage1_out["stage1_total_loss"] + (couple_weight_iter * loss_couple_i)

                frame_loss_sums[frame_idx] += float(loss_i.item())
                frame_loss_counts[frame_idx] += 1
                batch_l1.append(l1_i)
                batch_ssim.append(ssim_i)
                batch_base.append(base_i)
                batch_bright.append(bright_i)
                batch_loss.append(loss_i)
                batch_loss_lik.append(stage1_out["loss_lik"])
                batch_loss_ent.append(stage1_out["loss_ent"])
                batch_stage1.append(stage1_out["stage1_total_loss"])
                batch_loss_couple.append(loss_couple_i)
                batch_match_rate.append(float(coupling_stats["match_rate"]))
                batch_residual_p95.append(float(coupling_stats["residual_p95"]))
                batch_assoc_w.append(float(coupling_stats["assoc_w_mean"]))

                if chunk4_runtime_state is not None:
                    iter_support_obs[frame_idx] = compute_chunk4_support_observations_for_frame(
                        frame_idx=frame_idx,
                        training_frames=training_frames,
                        gaussians=gaussians,
                        render_pkg=render_pkg,
                        rendered=rendered,
                        gt_image=gt_image,
                        sonar_config=sonar_config,
                        sonar_scale_factor=sonar_scale_factor,
                        support_residual_thresh=ELEV_CHUNK4_CFG.support_residual_thresh,
                    )

            if not batch_loss:
                continue

            Ll1 = torch.stack(batch_l1).mean()
            ssim_val = torch.stack(batch_ssim).mean()
            base_loss = torch.stack(batch_base).mean()
            bright_loss = torch.stack(batch_bright).mean()
            loss = torch.stack(batch_loss).mean()
            loss_lik = torch.stack(batch_loss_lik).mean()
            loss_ent = torch.stack(batch_loss_ent).mean()
            loss_stage1 = torch.stack(batch_stage1).mean()
            loss_couple = torch.stack(batch_loss_couple).mean()
            couple_match_rate = float(np.mean(batch_match_rate)) if batch_match_rate else 0.0
            couple_residual_p95 = float(np.mean(batch_residual_p95)) if batch_residual_p95 else 0.0
            couple_assoc_w = float(np.mean(batch_assoc_w)) if batch_assoc_w else 0.0
            cached_loglik = iter_cached_loglik
            cached_support_mask = iter_cached_support_mask
            p_post = iter_p_post

            loss.backward()

            chunk4_support_stats = {
                "match_frames": 0,
                "diverse_frames": 0,
                "support_ratio": 0.0,
                "residual_p95": 0.0,
                "active_support_ge2_frac": 0.0,
                "active_support_ge3_frac": 0.0,
                "pruned_support_count": 0,
            }
            fov_pruned_count = 0

            with torch.no_grad():
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if optim_elev is not None:
                    optim_elev.step()
                    optim_elev.zero_grad(set_to_none=True)
                # Scale frozen - no optimizer step
                gaussians.update_learning_rate(STAGE2_ITERATIONS + iteration)

                if chunk4_runtime_state is not None:
                    chunk4_support_stats = update_chunk4_support_runtime(
                        global_iter=global_iter,
                        sampled_frame_indices=sampled_frame_indices,
                        iter_support_obs=iter_support_obs,
                        training_frames=training_frames,
                        gaussians=gaussians,
                        chunk4_runtime_state=chunk4_runtime_state,
                        chunk4_cfg=ELEV_CHUNK4_CFG,
                    )

                # FOV-aware pruning
                if FOV_PRUNE_INTERVAL > 0 and iteration % FOV_PRUNE_INTERVAL == 0:
                    fov_pruned_count = prune_outside_fov(
                        gaussians,
                        training_frames,
                        sonar_config,
                        sonar_scale_factor,
                        chunk4_runtime_state=chunk4_runtime_state,
                        chunk4_cfg=ELEV_CHUNK4_CFG,
                        reason="fov_stage3",
                        current_iter=global_iter,
                    )
                    if fov_pruned_count > 0:
                        sync_opacity_policy(global_iter, f"stage3/fov_prune@{iteration}", force=True)
                        print(
                            f"  [FOV prune] Removed {fov_pruned_count} surfels outside FOV, "
                            f"{len(gaussians.get_xyz)} remaining"
                        )

                if chunk4_runtime_state is not None and bool(ELEV_CHUNK4_CFG.surfel_id_asserts):
                    assert_surfel_id_integrity(chunk4_runtime_state["persistent_state"])

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
                sampler_tail = ""
                if stage3_use_round_robin:
                    sampler_tail = (
                        f", batch={len(sampled_frame_indices)}, sampler_epoch={elev_sampler_state.epoch}, "
                        f"T_model={temp_model_iter:.3f}, T_post={temp_post_iter:.3f}, "
                        f"T_tgt={ELEV_STAGE1_CFG.lik_tgt_temp:.3f}"
                    )
                    if couple_compute_enabled:
                        sampler_tail += (
                            f", couple={loss_couple.item():.6f}, w_couple={couple_weight_iter:.3f}, "
                            f"match={couple_match_rate:.3f}, p95={couple_residual_p95:.3f}m, "
                            f"assoc_w={couple_assoc_w:.3f}"
                        )
                if chunk4_runtime_state is not None:
                    sampler_tail += (
                        f", sup_match={chunk4_support_stats['match_frames']}/{chunk4_support_stats['diverse_frames']}, "
                        f"sup_ratio={chunk4_support_stats['support_ratio']:.3f}, "
                        f"sup_p95={chunk4_support_stats['residual_p95']:.3f}, "
                        f"sup_ge2={chunk4_support_stats['active_support_ge2_frac']:.3f}, "
                        f"sup_ge3={chunk4_support_stats['active_support_ge3_frac']:.3f}, "
                        f"prune={int(chunk4_support_stats['pruned_support_count']) + int(fov_pruned_count)}"
                    )
                print(
                    f"  Iter {iteration:3d}: L1={Ll1.item():.6f}, SSIM={ssim_val.item():.4f}, "
                    f"scale={scale_value:.4f}, pts={len(gaussians.get_xyz)}, "
                    f"lik={loss_lik.item():.6f}, ent={loss_ent.item():.6f}, stage1={loss_stage1.item():.6f}{sampler_tail}"
                )

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
        final_global_iter = training_iter_offset + STAGE1_ITERATIONS + STAGE2_ITERATIONS + STAGE3_ITERATIONS
        num_pruned = prune_outside_fov(
            gaussians,
            training_frames,
            sonar_config,
            sonar_scale_factor,
            chunk4_runtime_state=chunk4_runtime_state,
            chunk4_cfg=ELEV_CHUNK4_CFG,
            reason="fov_final",
            current_iter=final_global_iter,
        )
        if num_pruned > 0:
            sync_opacity_policy(final_global_iter, "final_prune", force=True)
            print(f"  [Final prune] Removed {num_pruned} surfels, {len(gaussians.get_xyz)} remaining")
        if chunk4_runtime_state is not None and bool(ELEV_CHUNK4_CFG.surfel_id_asserts):
            assert_surfel_id_integrity(chunk4_runtime_state["persistent_state"])

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

    visualizer_stage_artifacts["stage3"] = export_stage_visualizer_state(
        gaussians,
        visualizer_dir,
        stage_filename="surfels_after_stage3.ply",
        stage_status="executed" if STAGE3_ITERATIONS > 0 else "skipped_alias_to_stage2",
        glyph_prefix="stage3",
    )
    visualizer_frame_artifacts = export_frame_visualizer_artifacts(
        training_frames,
        gaussians,
        background,
        sonar_config,
        sonar_scale_factor,
        visualizer_dir,
        visualizer_rendered_dir,
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
    elevation_stage1_runtime_state["sampler_state"] = serialize_sampler_state(elev_sampler_state)
    elevation_stage1_runtime_state["pixel_bank"] = serialize_pixel_bank(pixel_bank)
    elevation_stage1_runtime_state["pixel_logits"] = serialize_pixel_logits_registry(pixel_logits_registry)
    elevation_stage1_runtime_state["optim_elev_state"] = (
        optim_elev.state_dict() if optim_elev is not None else None
    )
    elevation_chunk4_runtime_state = build_chunk4_runtime_checkpoint_state(
        chunk4_runtime_state,
        active_frame_keys,
        ELEV_CHUNK4_CFG,
    )

    if cached_loglik:
        print(
            "[Elevation Stage 1] final cache snapshot: "
            f"cached_loglik_frames={len(cached_loglik)}, "
            f"cached_support_frames={len(cached_support_mask)}, "
            f"p_post_frames={len(p_post)}"
        )

    if SONAR_SAVE_CHECKPOINT:
        final_sonar_diag = {}
        if training_frames:
            with torch.no_grad():
                final_render_pkg = render_sonar(
                    training_frames[0],
                    gaussians,
                    background,
                    sonar_config=sonar_config,
                    scale_factor=sonar_scale_factor,
                    sonar_extrinsic=None,
                    **SONAR_RENDER_KWARGS,
                )
            final_sonar_diag = final_render_pkg.get("sonar_diagnostics") or {}

        save_training_checkpoint(
            SONAR_SAVE_CHECKPOINT,
            gaussians,
            sonar_scale_factor,
            scale_optimizer,
            iteration=training_iter_offset + total_iters,
            stage_name="final",
            metadata={
                "renderer_semantics_version": "v2",
                "normal_init_mode": "pcd_normals",
                "lambertian_transfer": os.environ.get("SONAR_LAMBERTIAN_MODE", "leaky").strip().lower(),
                "occlusion_model": "ray_binned",
                "render_sonar_contract_hash": os.environ.get("RENDER_SONAR_CONTRACT_HASH", "working_tree"),
                "sonar_render_mode": os.environ.get("SONAR_RENDER_MODE", "2dgs").strip().lower(),
                "sonar_occlusion_mode": os.environ.get("SONAR_OCCLUSION_MODE", "ray_binned").strip().lower(),
                "occlusion_space": "ray_binned",
                "occlusion_footprint_policy": "multi_bin_participation",
                "occlusion_support_cap_config": {
                    "k_sigma": float(os.environ.get("SONAR_OCCL_KSIGMA", "2.5")),
                    "weight_floor_rel": float(os.environ.get("SONAR_OCCL_WEIGHT_FLOOR_REL", "1e-3")),
                    "topk": int(os.environ.get("SONAR_OCCL_TOPK", "16")),
                },
                "occlusion_support_cap_mass_loss": {
                    "mean": float(final_sonar_diag.get("occlusion_support_cap_mass_loss", {}).get("mean", 0.0)),
                    "median": float(final_sonar_diag.get("occlusion_support_cap_mass_loss", {}).get("median", 0.0)),
                    "p95": float(final_sonar_diag.get("occlusion_support_cap_mass_loss", {}).get("p95", 0.0)),
                    "p99": float(final_sonar_diag.get("occlusion_support_cap_mass_loss", {}).get("p99", 0.0)),
                    "max": float(final_sonar_diag.get("occlusion_support_cap_mass_loss", {}).get("max", 0.0)),
                },
                "compat_reference_id": os.environ.get("SONAR_COMPAT_REFERENCE_ID", "v2_compat_default"),
                "elevation_bin_count": int(os.environ.get("SONAR_ELEV_BINS", "1")),
                "elevation_weight_mode": os.environ.get("SONAR_ELEV_WEIGHT_MODE", "uniform").strip().lower(),
                "surfel_size_stats_schema_version": "v1",
                "surfel_size_stats_final": final_sonar_diag.get("surfel_size_stats_world", {}),
                "surfel_size_stats_image_final": final_sonar_diag.get("surfel_size_stats_image", {}),
                "sigma_point_config": {
                    "kappa": float(os.environ.get("SONAR_SIGMA_KAPPA", "0.0")),
                    "alpha": float(os.environ.get("SONAR_SIGMA_ALPHA", "1.0")),
                    "beta": float(os.environ.get("SONAR_SIGMA_BETA", "2.0")),
                },
                "sigma_point_boundary_policy_version": os.environ.get(
                    "SONAR_SIGMA_BOUNDARY_POLICY_VERSION",
                    "v1",
                ),
                "sigma_point_fallback_fraction": float(final_sonar_diag.get("sigma_point_fallback_fraction", 0.0)),
                "dataset_key": DATASET_KEY,
                "dataset_path": DATASET_PATH,
                "seed": SEED,
                "elev_init_mode": ELEV_INIT_MODE,
                "elevation_aware": int(ELEV_STAGE1_CFG.elevation_aware),
                "elev_stage1_mode": ELEV_STAGE1_CFG.stage1_mode,
                "elev_stage1_effective_mode": ELEV_STAGE1_CFG.effective_stage1_mode,
                "elev_bins": ELEV_STAGE1_CFG.bins,
                "elev_frames_per_iter": ELEV_STAGE1_CFG.frames_per_iter,
                "elev_anneal_iters": ELEV_STAGE1_CFG.anneal_iters,
                "elev_temp_post_mode": ELEV_STAGE1_CFG.temp_post_mode,
                "elev_resume_mismatch_policy": ELEV_STAGE1_CFG.resume_pixellogit_mismatch,
                "elev_chunk4_couple_mode": ELEV_CHUNK4_CFG.couple_mode,
                "elev_chunk4_support_mode": ELEV_CHUNK4_CFG.support_mode,
                "elev_chunk4_effective_couple_mode": ELEV_CHUNK4_CFG.effective_couple_mode,
                "elev_chunk4_effective_support_mode": ELEV_CHUNK4_CFG.effective_support_mode,
                "active_frame_fingerprint": active_frame_fingerprint,
                "sonar_fixed_opacity": int(SONAR_FIXED_OPACITY),
                "sonar_freeze_scale": int(SONAR_FREEZE_SCALE),
                "stage1_iterations": STAGE1_ITERATIONS,
                "stage2_iterations": STAGE2_ITERATIONS,
                "stage3_iterations": STAGE3_ITERATIONS,
            },
            stage1_runtime_state=elevation_stage1_runtime_state,
            chunk4_runtime_state=elevation_chunk4_runtime_state,
        )

    visualizer_manifest = {
        "visualizer_root": ".",
        "rendered_root": "rendered",
        "git_commit_sha": get_repo_commit_sha(),
        "frame_artifacts": visualizer_frame_artifacts,
        "stage_artifacts": visualizer_stage_artifacts,
        "glyph_export_policy": {
            "max_sampled_glyphs": VISUALIZER_MAX_GLYPHS,
            "max_per_frame_glyphs": VISUALIZER_MAX_FRAME_GLYPHS,
            "opacity_percentile": VISUALIZER_OPACITY_PERCENTILE,
            "eq_radius_percentile": VISUALIZER_EQ_RADIUS_PERCENTILE,
            "ellipse_segments": VISUALIZER_GLYPH_SEGMENTS,
            "face_offset_scale": VISUALIZER_FACE_OFFSET_SCALE,
            "normal_stem_scale": VISUALIZER_NORMAL_STEM_SCALE,
            "per_frame_selection_priority": [
                "center_in_fov",
                "fov_margin_desc",
                "facing_score_desc",
                "opacity_desc",
                "eq_radius_asc",
            ],
        },
        "parameter_spaces": {
            "surfel_state_ply": {
                "scales": "latent_log",
                "rotations": "latent_quaternion",
            },
            "glyph_exports": {
                "scales": "activated_exp",
                "rotations": "normalized_quaternion",
            },
        },
        "wireframe_geometry": {
            "near": "legacy rectangular pose pyramid at forward depth",
            "full_range": "constant-range sonar FOV corners at sonar_config.range_max",
        },
        "frame_fov_membership_rule": {
            "description": "size-aware overlap export",
            "formula": "forward > 0 and compute_fov_margin(range, azimuth, elevation) + max(scale_u, scale_v) > 0",
            "surfel_radius": "max(activated_scale_u, activated_scale_v)",
        },
        "metadata_refs": build_visualizer_metadata_refs(OUTPUT_DIR, visualizer_dir, DATASET_PATH),
    }
    visualizer_manifest_path = os.path.join(visualizer_dir, "manifest.json")
    write_visualizer_manifest(visualizer_manifest_path, visualizer_manifest)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nFinal scale factor: {sonar_scale_factor.get_scale_value():.6f}")
    print(f"\nGenerated files:")
    print(f"  - sonar_init_points.ply       (Combined points from {NUM_TRAINING_FRAMES} frames)")
    print(f"  - pose_pyramids_wireframe.ply (Wireframes for training frames)")
    print(f"  - visualizer/manifest.json    (Visualizer artifact index)")
    print(f"  - visualizer/*.ply            (Stage glyphs + per-frame wireframes/FOV surfels)")
    print(f"  - visualizer/rendered/*.png   (Per-frame rendered sonar images)")
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


if __name__ == "__main__":
    main()
