#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import math
import os
from dataclasses import dataclass
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal, sonar_ranges_to_points, sonar_points_to_normals
from utils.sonar_utils import get_scaled_world_to_view_transform

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        ndc2world= viewpoint_camera.ndc2world,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        additive_mode=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap, converge = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "converge" : converge,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes. See Eq. 9 in Unbiased Depth paper
    surf_depth = torch.nan_to_num(allmap[5:6], 0, 0)
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets


# =============================================================================
# Sonar Forward Projection (Polar Geometry)
# =============================================================================


@dataclass
class SonarProjection:
    row: torch.Tensor
    col: torch.Tensor
    range_vals: torch.Tensor
    azimuth: torch.Tensor
    in_fov: torch.Tensor
    in_front: torch.Tensor
    in_bounds: torch.Tensor
    valid: torch.Tensor


def _transform_world_points_to_sonar_frame(points_world, viewpoint_camera, scale_factor=None, sonar_extrinsic=None):
    """Transform world points to sonar/view frame under row-major transform contract."""
    w2v = get_scaled_world_to_view_transform(
        viewpoint_camera,
        scale_factor=scale_factor,
        sonar_extrinsic=sonar_extrinsic,
    )
    R_w2v = w2v[:3, :3]
    t_w2v = w2v[3, :3]

    if scale_factor is not None:
        points_world_scaled = scale_factor.scale * points_world
    else:
        points_world_scaled = points_world

    points_view = points_world_scaled @ R_w2v.T + t_w2v
    return points_view, w2v


def sonar_project_points(points_world, viewpoint_camera, sonar_config, scale_factor=None, sonar_extrinsic=None):
    """
    Project world points into sonar image bins.

    Returns SonarProjection with explicit validity fields:
      - in_fov: azimuth/elevation/range constraints
      - in_front: forward > 0
      - in_bounds: pixel bounds check
      - valid: in_fov & in_front & in_bounds
    """
    points_sonar, _ = _transform_world_points_to_sonar_frame(
        points_world,
        viewpoint_camera,
        scale_factor=scale_factor,
        sonar_extrinsic=sonar_extrinsic,
    )

    right = points_sonar[:, 0]
    down = points_sonar[:, 1]
    forward = points_sonar[:, 2]

    azimuth = -torch.atan2(right, forward)
    range_vals = torch.sqrt(right**2 + down**2 + forward**2)
    horiz_dist = torch.sqrt(right**2 + forward**2)
    elevation = torch.atan2(down, horiz_dist.clamp_min(1e-8))

    in_fov = (
        (torch.abs(azimuth) <= sonar_config.half_azimuth_rad)
        & (torch.abs(elevation) <= sonar_config.half_elevation_rad)
        & (range_vals >= sonar_config.range_min)
        & (range_vals <= sonar_config.range_max)
    )
    in_front = forward > 0

    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    col = (-azimuth / sonar_config.half_azimuth_rad + 1) * (W / 2)
    row = (range_vals - sonar_config.range_min) / (sonar_config.range_max - sonar_config.range_min) * H
    in_bounds = (col >= 0) & (col <= W - 1) & (row >= 0) & (row <= H - 1)
    valid = in_fov & in_front & in_bounds

    return SonarProjection(
        row=row,
        col=col,
        range_vals=range_vals,
        azimuth=azimuth,
        in_fov=in_fov,
        in_front=in_front,
        in_bounds=in_bounds,
        valid=valid,
    )


def compute_sonar_range_attenuation(
    range_vals,
    use_range_attenuation=True,
    range_atten_exp=2.0,
    range_atten_gain=1.0,
    range_atten_r0=0.35,
    range_atten_eps=1e-6,
    range_atten_auto_gain=False,
):
    """Compute stabilized sonar range attenuation with deterministic precedence."""
    if not use_range_attenuation:
        attenuation = torch.ones_like(range_vals)
        return attenuation, {
            "enabled": False,
            "gain_mode": "off",
            "effective_gain": 1.0,
            "exp": float(range_atten_exp),
            "r0": float(range_atten_r0),
            "eps": float(range_atten_eps),
        }

    r0 = max(float(range_atten_r0), 0.0)
    eps = max(float(range_atten_eps), 1e-12)
    exp = float(range_atten_exp)
    gain_seed = float(range_atten_gain)

    r_eff = torch.clamp(range_vals, min=r0)
    atten_base = 1.0 / (torch.pow(r_eff, exp) + eps)

    if range_atten_auto_gain:
        if atten_base.numel() > 0:
            base_mean = atten_base.detach().mean().clamp_min(1e-8)
            effective_gain = float(gain_seed / base_mean.item())
        else:
            effective_gain = gain_seed
        gain_mode = "auto"
    else:
        effective_gain = gain_seed
        gain_mode = "manual"

    attenuation = atten_base * range_vals.new_tensor(effective_gain)
    return attenuation, {
        "enabled": True,
        "gain_mode": gain_mode,
        "effective_gain": effective_gain,
        "exp": exp,
        "r0": r0,
        "eps": eps,
    }


def compute_sonar_lambertian(dot_val, mode=None, alpha=0.01):
    """Apply configurable Lambertian transfer for sonar rendering."""
    lambertian_mode = (mode or os.getenv("SONAR_LAMBERTIAN_MODE", "leaky")).strip().lower()
    alpha_val = float(alpha)

    if lambertian_mode == "clamp0":
        return torch.clamp(dot_val, min=0.0), lambertian_mode
    if lambertian_mode == "leaky":
        return torch.where(dot_val >= 0, dot_val, alpha_val * dot_val), lambertian_mode
    if lambertian_mode == "elu":
        return torch.where(dot_val >= 0, dot_val, alpha_val * (torch.exp(dot_val) - 1.0)), lambertian_mode
    if lambertian_mode == "ste":
        hard = torch.clamp(dot_val, min=0.0)
        soft = torch.clamp(dot_val, min=alpha_val)
        return soft + (hard - soft).detach(), lambertian_mode

    raise ValueError(
        f"Unsupported SONAR_LAMBERTIAN_MODE '{lambertian_mode}'. "
        "Use one of: leaky, clamp0, elu, ste."
    )


def resolve_sonar_render_contract():
    """Parse renderer/occlusion runtime contract switches from environment."""
    render_mode = os.getenv("SONAR_RENDER_MODE", "2dgs").strip().lower()
    if render_mode == "legacy":
        raise ValueError("SONAR_RENDER_MODE=legacy is no longer supported.")
    if render_mode not in {"2dgs", "2dgs_nonlinear"}:
        raise ValueError(f"Unsupported SONAR_RENDER_MODE '{render_mode}'.")

    occlusion_mode = os.getenv("SONAR_OCCLUSION_MODE", "ray_binned").strip().lower()
    if occlusion_mode not in {"none", "ray_binned"}:
        raise ValueError(f"Unsupported SONAR_OCCLUSION_MODE '{occlusion_mode}'.")

    elev_bins = max(int(os.getenv("SONAR_ELEV_BINS", "1")), 1)
    elev_weight_mode = os.getenv("SONAR_ELEV_WEIGHT_MODE", "uniform").strip().lower()
    if elev_weight_mode not in {"uniform", "beam_pattern"}:
        raise ValueError(f"Unsupported SONAR_ELEV_WEIGHT_MODE '{elev_weight_mode}'.")

    occl_k_sigma = float(os.getenv("SONAR_OCCL_KSIGMA", "2.5"))
    occl_weight_floor_rel = float(os.getenv("SONAR_OCCL_WEIGHT_FLOOR_REL", "1e-3"))
    occl_topk = max(int(os.getenv("SONAR_OCCL_TOPK", "16")), 1)

    return {
        "render_mode": render_mode,
        "occlusion_mode": occlusion_mode,
        "elev_bins": elev_bins,
        "elev_weight_mode": elev_weight_mode,
        "occl_k_sigma": occl_k_sigma,
        "occl_weight_floor_rel": occl_weight_floor_rel,
        "occl_topk": occl_topk,
    }


def _sonar_config_values(sonar_config):
    image_width = float(getattr(sonar_config, "image_width", 256))
    image_height = float(getattr(sonar_config, "image_height", 200))

    half_azimuth_rad = getattr(sonar_config, "half_azimuth_rad", None)
    if half_azimuth_rad is None:
        half_azimuth_rad = math.radians(float(getattr(sonar_config, "azimuth_fov", 120.0)) * 0.5)

    half_elevation_rad = getattr(sonar_config, "half_elevation_rad", None)
    if half_elevation_rad is None:
        half_elevation_rad = math.radians(float(getattr(sonar_config, "elevation_fov", 20.0)) * 0.5)

    range_min = float(getattr(sonar_config, "range_min", 0.2))
    range_max = float(getattr(sonar_config, "range_max", 3.0))
    return {
        "image_width": image_width,
        "image_height": image_height,
        "half_azimuth_rad": float(half_azimuth_rad),
        "half_elevation_rad": float(half_elevation_rad),
        "range_min": range_min,
        "range_max": range_max,
    }


def _condition_sigma_2d(sigma_2d, min_var=0.1, max_var=400.0, cond_cap=100.0):
    sym = 0.5 * (sigma_2d + sigma_2d.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(sym)
    eigvals = torch.clamp(eigvals, min=min_var, max=max_var)
    if cond_cap is not None and cond_cap > 0:
        max_eval = eigvals.max(dim=-1, keepdim=True).values
        min_allowed = torch.clamp(max_eval / float(cond_cap), min=min_var)
        eigvals = torch.maximum(eigvals, min_allowed)
        eigvals = torch.clamp(eigvals, max=max_var)
    return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)


def _elevation_bin_weights(num_bins, mode, device, dtype):
    if num_bins <= 1:
        return torch.ones(1, device=device, dtype=dtype)
    if mode == "beam_pattern":
        coords = torch.linspace(-1.0, 1.0, steps=num_bins, device=device, dtype=dtype)
        weights = torch.cos(coords * (math.pi * 0.5)).clamp_min(0.0)
    else:
        weights = torch.ones(num_bins, device=device, dtype=dtype)
    return weights / weights.sum().clamp_min(1e-8)


def sigma2d_to_transmat_precomp(mu_2d, sigma_2d):
    sigma_conditioned = _condition_sigma_2d(sigma_2d)
    try:
        chol = torch.linalg.cholesky(sigma_conditioned)
    except RuntimeError:
        eigvals, eigvecs = torch.linalg.eigh(0.5 * (sigma_conditioned + sigma_conditioned.T))
        chol = eigvecs @ torch.diag(torch.sqrt(torch.clamp(eigvals, min=1e-6)))

    zero = sigma_2d.new_tensor(0.0)
    one = sigma_2d.new_tensor(1.0)
    row0 = torch.stack([chol[0, 0], chol[1, 0], mu_2d[0]])
    row1 = torch.stack([zero, chol[1, 1], mu_2d[1]])
    row2 = torch.stack([zero, zero, one])
    return torch.stack([row0, row1, row2], dim=0)


def transmat_precomp_to_sigma2d(transmat_precomp):
    l = torch.stack(
        [
            torch.stack([transmat_precomp[0, 0], transmat_precomp[1, 0]]),
            torch.stack([transmat_precomp[0, 1], transmat_precomp[1, 1]]),
        ],
        dim=0,
    )
    sigma_2d = l @ l.T
    return 0.5 * (sigma_2d + sigma_2d.T)


def _quat_to_rotation_matrices(quat_wxyz):
    q = quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(1e-8)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1)
    row1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1)
    row2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _project_points_to_sonar_batch(points_3d, sonar_config):
    cfg = _sonar_config_values(sonar_config)
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]

    azimuth = -torch.atan2(x, z)
    horiz_dist = torch.sqrt(x * x + z * z)
    elevation = torch.atan2(y, horiz_dist.clamp_min(1e-8))
    range_vals = torch.sqrt(x * x + y * y + z * z)

    half_az = range_vals.new_tensor(cfg["half_azimuth_rad"])
    half_el = range_vals.new_tensor(cfg["half_elevation_rad"])
    width = range_vals.new_tensor(cfg["image_width"])
    height = range_vals.new_tensor(cfg["image_height"])
    range_min = range_vals.new_tensor(cfg["range_min"])
    range_max = range_vals.new_tensor(cfg["range_max"])
    range_span = (range_max - range_min).clamp_min(1e-8)

    col = (-azimuth / half_az + 1.0) * (width * 0.5)
    row = (range_vals - range_min) / range_span * height

    in_front = z > 0
    in_range = (range_vals >= range_min) & (range_vals <= range_max)
    in_azimuth = torch.abs(azimuth) <= half_az
    in_elevation = torch.abs(elevation) <= half_el
    in_bounds = (col >= 0) & (col <= (width - 1.0)) & (row >= 0) & (row <= (height - 1.0))
    valid = in_front & in_range & in_azimuth & in_elevation & in_bounds

    return {
        "col": col,
        "row": row,
        "azimuth": azimuth,
        "elevation": elevation,
        "range": range_vals,
        "in_front": in_front,
        "in_range": in_range,
        "in_azimuth": in_azimuth,
        "in_elevation": in_elevation,
        "in_bounds": in_bounds,
        "valid": valid,
    }


def _ukf_sigma_point_weights(alpha, beta, kappa, device, dtype):
    n_dim = 2.0
    lam = alpha * alpha * (n_dim + kappa) - n_dim
    denom = max(n_dim + lam, 1e-8)
    mean_weights = torch.tensor(
        [lam / denom, 1.0 / (2.0 * denom), 1.0 / (2.0 * denom), 1.0 / (2.0 * denom), 1.0 / (2.0 * denom)],
        device=device,
        dtype=dtype,
    )
    cov_weights = mean_weights.clone()
    cov_weights[0] = cov_weights[0] + (1.0 - alpha * alpha + beta)
    return lam, mean_weights, cov_weights


def _jacobian_sigma_footprint_batch(mean_3d, scale_xy, quat_wxyz, sonar_config):
    cfg = _sonar_config_values(sonar_config)
    x = mean_3d[:, 0]
    y = mean_3d[:, 1]
    z = mean_3d[:, 2]

    width = mean_3d.new_tensor(cfg["image_width"])
    height = mean_3d.new_tensor(cfg["image_height"])
    half_az = mean_3d.new_tensor(cfg["half_azimuth_rad"])
    range_min = mean_3d.new_tensor(cfg["range_min"])
    range_max = mean_3d.new_tensor(cfg["range_max"])
    range_span = (range_max - range_min).clamp_min(1e-8)

    denom_az = (x * x + z * z).clamp_min(1e-8)
    dcol_daz = -(width / (2.0 * half_az.clamp_min(1e-8)))
    range_val = torch.sqrt(x * x + y * y + z * z).clamp_min(1e-8)
    drow_scale = height / range_span

    j = torch.zeros((mean_3d.shape[0], 2, 3), dtype=mean_3d.dtype, device=mean_3d.device)
    j[:, 0, 0] = dcol_daz * (-z / denom_az)
    j[:, 0, 2] = dcol_daz * (x / denom_az)
    j[:, 1, 0] = drow_scale * (x / range_val)
    j[:, 1, 1] = drow_scale * (y / range_val)
    j[:, 1, 2] = drow_scale * (z / range_val)

    rot = _quat_to_rotation_matrices(quat_wxyz)
    t1 = rot[:, :, 0] * scale_xy[:, 0:1]
    t2 = rot[:, :, 1] * scale_xy[:, 1:2]
    sigma_3d = t1.unsqueeze(-1) * t1.unsqueeze(-2) + t2.unsqueeze(-1) * t2.unsqueeze(-2)
    sigma_2d = j @ sigma_3d @ j.transpose(-1, -2)
    return _condition_sigma_2d(sigma_2d)


def _build_range_profiles_batch(row_centers, sigma_rows, num_rows, k_sigma=2.5):
    if row_centers.numel() == 0:
        return row_centers.new_zeros((0, int(num_rows)))

    row_coords = torch.arange(int(num_rows), device=row_centers.device, dtype=row_centers.dtype).unsqueeze(0)
    sigma_rows = sigma_rows.unsqueeze(1).clamp_min(0.5)
    row_dist = row_coords - row_centers.unsqueeze(1)
    support_mask = row_dist.abs() <= (float(k_sigma) * sigma_rows)
    row_weights = torch.exp(-0.5 * (row_dist / sigma_rows) ** 2) * support_mask.to(row_centers.dtype)

    empty_mask = row_weights.sum(dim=1) <= 0
    if empty_mask.any():
        row_weights[empty_mask] = 0.0
        nearest = torch.clamp(torch.round(row_centers[empty_mask]).long(), min=0, max=int(num_rows) - 1)
        row_weights[empty_mask, nearest] = 1.0

    return row_weights / row_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _cap_support_weights_batched(weights, topk=16, floor_rel=1e-3):
    if weights.numel() == 0:
        return {
            "weights": weights,
            "mass_lost": weights.new_zeros((weights.shape[0],)),
        }

    w = torch.clamp(weights, min=0.0)
    peak = w.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    keep = w >= (peak * float(floor_rel))

    if int(topk) > 0 and int(topk) < w.shape[1]:
        topk_idx = torch.topk(w, k=int(topk), dim=1, largest=True, sorted=False).indices
        topk_mask = torch.zeros_like(keep)
        topk_mask.scatter_(1, topk_idx, True)
        keep = keep & topk_mask

    retained_raw = torch.where(keep, w, torch.zeros_like(w))
    retained_sum = retained_raw.sum(dim=1, keepdim=True)
    normalized = torch.where(retained_sum > 0, retained_raw / retained_sum.clamp_min(1e-12), torch.zeros_like(retained_raw))
    mass_lost = (w.sum(dim=1) - retained_sum.squeeze(1)).clamp_min(0.0)
    return {
        "weights": normalized,
        "mass_lost": mass_lost,
    }


def _build_multibin_support_weights_batch(
    center_col,
    center_elev,
    sigma_2d,
    sonar_config,
    elev_bins,
    topk,
    floor_rel,
    k_sigma,
):
    if center_col.numel() == 0:
        empty = center_col.new_zeros((0, int(round(_sonar_config_values(sonar_config)["image_width"])) * max(int(elev_bins), 1)))
        return {"weights": empty, "mass_lost": center_col.new_zeros((0,))}

    cfg = _sonar_config_values(sonar_config)
    width = int(round(cfg["image_width"]))
    elev_bins = max(int(elev_bins), 1)

    col_coords = torch.arange(width, device=center_col.device, dtype=center_col.dtype).unsqueeze(0)
    col_sigma = torch.sqrt(torch.clamp(sigma_2d[:, 0, 0], min=0.25)).unsqueeze(1).clamp_min(0.5)
    col_dist = col_coords - center_col.unsqueeze(1)
    col_weights = torch.exp(-0.5 * (col_dist / col_sigma) ** 2)
    col_weights = col_weights * (col_dist.abs() <= (float(k_sigma) * col_sigma)).to(center_col.dtype)

    if elev_bins <= 1:
        elev_weights = center_col.new_ones((center_col.shape[0], 1))
    else:
        half_el = center_col.new_tensor(cfg["half_elevation_rad"])
        elev_sigma = center_col.new_full((center_col.shape[0], 1), max(float(cfg["half_elevation_rad"]) / max(elev_bins, 1), 1e-3))
        elev_coords = torch.linspace(-float(half_el.item()), float(half_el.item()), steps=elev_bins, device=center_col.device, dtype=center_col.dtype).unsqueeze(0)
        elev_dist = elev_coords - center_elev.unsqueeze(1)
        elev_weights = torch.exp(-0.5 * (elev_dist / elev_sigma) ** 2)
        elev_weights = elev_weights * (elev_dist.abs() <= (float(k_sigma) * elev_sigma)).to(center_col.dtype)

    flat_weights = (col_weights.unsqueeze(-1) * elev_weights.unsqueeze(1)).reshape(center_col.shape[0], width * elev_bins)
    return _cap_support_weights_batched(flat_weights, topk=topk, floor_rel=floor_rel)


def _project_sonar_footprints_batch(
    mean_3d,
    scale_xy,
    quat_wxyz,
    mode="2dgs",
    sigma_point_config=None,
    sonar_config=None,
    previous_fallback_used=None,
):
    if sonar_config is None:
        raise ValueError("sonar_config is required")

    mode_name = (mode or "2dgs").strip().lower()
    if mode_name not in {"2dgs", "2dgs_nonlinear"}:
        raise ValueError(f"Unsupported footprint mode '{mode_name}'")

    center_proj = _project_points_to_sonar_batch(mean_3d, sonar_config)
    center_hard_valid = center_proj["in_front"] & center_proj["in_range"]
    center_in_bounds = center_proj["in_bounds"]

    mu_2d = torch.stack([center_proj["col"], center_proj["row"]], dim=-1)
    jac_sigma = _jacobian_sigma_footprint_batch(mean_3d, scale_xy, quat_wxyz, sonar_config)

    skipped = ~center_hard_valid
    fallback_used = torch.zeros_like(skipped)
    sigma_valid_count = torch.zeros(mean_3d.shape[0], dtype=torch.long, device=mean_3d.device)
    sigma_invalid_reason_counts = {
        "azimuth": torch.zeros(mean_3d.shape[0], dtype=torch.long, device=mean_3d.device),
        "range": torch.zeros(mean_3d.shape[0], dtype=torch.long, device=mean_3d.device),
        "elevation": torch.zeros(mean_3d.shape[0], dtype=torch.long, device=mean_3d.device),
        "in_front": torch.zeros(mean_3d.shape[0], dtype=torch.long, device=mean_3d.device),
    }
    edge_band_fraction = mean_3d.new_zeros((mean_3d.shape[0],))
    effective_mode = ["skip" if bool(skip.item()) else mode_name for skip in skipped]
    sigma_2d = jac_sigma.clone()

    if mode_name == "2dgs_nonlinear" and mean_3d.shape[0] > 0:
        sigma_cfg = sigma_point_config or {}
        alpha = float(sigma_cfg.get("alpha", 1.0))
        beta = float(sigma_cfg.get("beta", 2.0))
        kappa = float(sigma_cfg.get("kappa", 0.0))
        lam, mean_weights_base, cov_weights_base = _ukf_sigma_point_weights(alpha, beta, kappa, mean_3d.device, mean_3d.dtype)
        spread = math.sqrt(max(2.0 + lam, 1e-8))

        rot = _quat_to_rotation_matrices(quat_wxyz)
        t1 = rot[:, :, 0] * scale_xy[:, 0:1]
        t2 = rot[:, :, 1] * scale_xy[:, 1:2]
        sigma_points = torch.stack(
            [
                mean_3d,
                mean_3d + spread * t1,
                mean_3d + spread * t2,
                mean_3d - spread * t1,
                mean_3d - spread * t2,
            ],
            dim=1,
        )
        sigma_proj = _project_points_to_sonar_batch(sigma_points, sonar_config)
        hard_valid = (
            sigma_proj["in_front"]
            & sigma_proj["in_range"]
            & sigma_proj["in_azimuth"]
            & sigma_proj["in_elevation"]
        )
        sigma_valid_count = hard_valid.sum(dim=1)
        sigma_invalid_reason_counts = {
            "azimuth": (~sigma_proj["in_azimuth"]).sum(dim=1),
            "range": (~sigma_proj["in_range"]).sum(dim=1),
            "elevation": (~sigma_proj["in_elevation"]).sum(dim=1),
            "in_front": (~sigma_proj["in_front"]).sum(dim=1),
        }

        cfg = _sonar_config_values(sonar_config)
        edge_dist = torch.stack(
            [
                sigma_proj["col"],
                sigma_proj["row"],
                sigma_proj["col"].new_tensor(cfg["image_width"] - 1.0) - sigma_proj["col"],
                sigma_proj["row"].new_tensor(cfg["image_height"] - 1.0) - sigma_proj["row"],
            ],
            dim=0,
        ).min(dim=0).values
        edge_band_fraction = torch.where(
            hard_valid.any(dim=1),
            (edge_dist.lt(4.0) & hard_valid).float().sum(dim=1) / hard_valid.float().sum(dim=1).clamp_min(1.0),
            torch.zeros_like(edge_dist[:, 0]),
        )

        taper = torch.sigmoid((edge_dist - 4.0) / 2.0)
        mean_weights = mean_weights_base.view(1, 5) * taper * hard_valid.to(mean_3d.dtype)
        cov_weights = cov_weights_base.view(1, 5) * taper * hard_valid.to(mean_3d.dtype)
        mean_sum = mean_weights.sum(dim=1, keepdim=True)
        prev_fallback = previous_fallback_used
        if prev_fallback is None:
            prev_fallback = torch.zeros_like(skipped)
        else:
            prev_fallback = prev_fallback.to(device=mean_3d.device, dtype=torch.bool)
            if prev_fallback.shape != skipped.shape:
                raise ValueError(
                    f"previous_fallback_used shape {tuple(prev_fallback.shape)} does not match {tuple(skipped.shape)}"
                )

        center_valid_for_fit = (~skipped) & center_in_bounds & (mean_sum.squeeze(1) > 0)
        fallback_enter = (~skipped) & ((~center_in_bounds) | (sigma_valid_count < 3) | (mean_sum.squeeze(1) <= 0))
        fallback_exit = center_valid_for_fit & (sigma_valid_count >= 4)
        fallback_used = torch.where(prev_fallback, ~fallback_exit, fallback_enter)
        
        if (~fallback_used).any():
            valid_idx = torch.where(~fallback_used)[0]
            pts_2d = torch.stack([sigma_proj["col"], sigma_proj["row"]], dim=-1)[valid_idx]
            cur_mean_weights = mean_weights[valid_idx]
            cur_cov_weights = cov_weights[valid_idx]

            cur_mean_weights = cur_mean_weights / cur_mean_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            mu_valid = torch.sum(cur_mean_weights.unsqueeze(-1) * pts_2d, dim=1)

            cur_cov_weights = cur_cov_weights / cur_cov_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            centered = pts_2d - mu_valid.unsqueeze(1)
            sigma_valid = torch.sum(
                cur_cov_weights.unsqueeze(-1).unsqueeze(-1)
                * (centered.unsqueeze(-1) * centered.unsqueeze(-2)),
                dim=1,
            )
            sigma_2d[valid_idx] = _condition_sigma_2d(sigma_valid)
            mu_2d[valid_idx] = mu_valid

        effective_mode = []
        for i in range(mean_3d.shape[0]):
            if bool(skipped[i].item()):
                effective_mode.append("skip")
            elif bool(fallback_used[i].item()):
                effective_mode.append("2dgs")
            else:
                effective_mode.append("2dgs_nonlinear")

    radii = torch.sqrt(torch.linalg.eigvalsh(sigma_2d).max(dim=-1).values.clamp_min(1e-8))
    zero_mu = torch.zeros_like(mu_2d)
    zero_sigma = torch.eye(2, dtype=sigma_2d.dtype, device=sigma_2d.device).unsqueeze(0).expand_as(sigma_2d) * 0.1
    mu_2d = torch.where(skipped.unsqueeze(-1), zero_mu, mu_2d)
    sigma_2d = torch.where(skipped.unsqueeze(-1).unsqueeze(-1), zero_sigma, sigma_2d)

    return {
        "mu_2d": mu_2d,
        "sigma_2d": sigma_2d,
        "effective_mode": effective_mode,
        "fallback_used": fallback_used,
        "sigma_valid_count": sigma_valid_count,
        "sigma_invalid_reason_counts": sigma_invalid_reason_counts,
        "edge_band_fraction": edge_band_fraction,
        "skipped": skipped,
        "radii": radii,
    }


def project_sonar_footprint(
    mean_3d,
    scale_xy,
    quat_wxyz,
    mode="2dgs",
    sigma_point_config=None,
    sonar_config=None,
    previous_fallback_used=None,
):
    batch = _project_sonar_footprints_batch(
        mean_3d=mean_3d.unsqueeze(0),
        scale_xy=scale_xy.unsqueeze(0),
        quat_wxyz=quat_wxyz.unsqueeze(0),
        mode=mode,
        sigma_point_config=sigma_point_config,
        sonar_config=sonar_config,
        previous_fallback_used=None if previous_fallback_used is None else torch.tensor([bool(previous_fallback_used)]),
    )
    return {
        "mu_2d": batch["mu_2d"][0],
        "sigma_2d": batch["sigma_2d"][0],
        "effective_mode": batch["effective_mode"][0],
        "fallback_used": bool(batch["fallback_used"][0].item()),
        "sigma_valid_count": int(batch["sigma_valid_count"][0].item()),
        "sigma_invalid_reason_counts": {
            "azimuth": int(batch["sigma_invalid_reason_counts"]["azimuth"][0].item()),
            "range": int(batch["sigma_invalid_reason_counts"]["range"][0].item()),
            "elevation": int(batch["sigma_invalid_reason_counts"]["elevation"][0].item()),
            "in_front": int(batch["sigma_invalid_reason_counts"]["in_front"][0].item()),
        },
        "edge_band_fraction": float(batch["edge_band_fraction"][0].item()),
        "skipped": bool(batch["skipped"][0].item()),
    }


def cap_support_weights(weights, topk=16, floor_rel=1e-3):
    if weights.numel() == 0:
        zero = torch.zeros_like(weights)
        mass = weights.new_tensor(0.0)
        return {
            "weights": zero,
            "mass_loss": {
                "mass_lost": mass,
                "mean": mass,
                "median": mass,
                "p95": mass,
                "p99": mass,
                "max": mass,
            },
        }

    w = torch.clamp(weights, min=0.0)
    peak = torch.max(w).clamp_min(1e-12)
    keep = w >= (peak * float(floor_rel))

    if int(topk) > 0 and int(topk) < w.numel():
        topk_idx = torch.topk(w, k=int(topk), largest=True, sorted=False).indices
        topk_mask = torch.zeros_like(keep)
        topk_mask[topk_idx] = True
        keep = keep & topk_mask

    retained_raw = torch.where(keep, w, torch.zeros_like(w))
    retained_sum = retained_raw.sum()
    mass_lost = (w.sum() - retained_sum).clamp_min(0.0)
    normalized = torch.where(retained_sum > 0, retained_raw / retained_sum.clamp_min(1e-12), torch.zeros_like(retained_raw))

    mass_vec = mass_lost.reshape(1)
    p95 = torch.quantile(mass_vec, 0.95)
    p99 = torch.quantile(mass_vec, 0.99)

    return {
        "weights": normalized,
        "mass_loss": {
            "mass_lost": mass_lost,
            "mean": mass_vec.mean(),
            "median": mass_vec.median(),
            "p95": p95,
            "p99": p99,
            "max": mass_vec.max(),
        },
    }


def compose_ray_binned_occlusion(ray_ids, range_vals, alpha_vals, value_vals, num_rays, **kwargs):
    del kwargs
    if range_vals.numel() == 0:
        empty = torch.zeros_like(value_vals)
        return {
            "event_returns": empty,
            "ray_returns": value_vals.new_zeros(int(num_rays)),
            "final_transmittance": value_vals.new_ones(int(num_rays)),
        }

    sort_stride = float(range_vals.detach().max().item()) + 1.0 if range_vals.numel() > 0 else 1.0
    sort_key = ray_ids.to(dtype=torch.float64) * sort_stride + range_vals.to(dtype=torch.float64)
    order = torch.argsort(sort_key)
    ray_sorted = ray_ids[order]
    alpha_sorted = torch.clamp(alpha_vals[order], min=0.0, max=1.0)
    value_sorted = value_vals[order]
    one_minus_alpha = (1.0 - alpha_sorted).clamp_min(1e-8)
    log_one_minus_alpha = torch.log(one_minus_alpha)

    new_ray = torch.ones_like(ray_sorted, dtype=torch.bool)
    new_ray[1:] = ray_sorted[1:] != ray_sorted[:-1]
    segment_ids = torch.cumsum(new_ray.to(torch.long), dim=0) - 1
    segment_starts = torch.nonzero(new_ray, as_tuple=False).squeeze(-1)

    inclusive_log = torch.cumsum(log_one_minus_alpha, dim=0)
    segment_prefix_log = value_sorted.new_zeros(int(segment_ids[-1].item()) + 1)
    if segment_starts.numel() > 1:
        segment_prefix_log[1:] = inclusive_log[segment_starts[1:] - 1]
    exclusive_log = (inclusive_log - log_one_minus_alpha) - segment_prefix_log[segment_ids]
    trans_before = torch.exp(exclusive_log)
    event_sorted = trans_before * value_sorted

    event_returns = torch.zeros_like(value_vals).scatter(0, order, event_sorted)
    ray_returns = value_vals.new_zeros(int(num_rays)).scatter_add(0, ray_sorted, event_sorted)

    final_transmittance = value_vals.new_ones(int(num_rays))
    segment_ends = torch.empty_like(segment_starts)
    if segment_starts.numel() > 1:
        segment_ends[:-1] = segment_starts[1:] - 1
    segment_ends[-1] = ray_sorted.numel() - 1
    final_log = inclusive_log[segment_ends] - segment_prefix_log
    final_transmittance[ray_sorted[segment_starts]] = torch.exp(final_log)

    return {
        "event_returns": event_returns,
        "ray_returns": ray_returns,
        "final_transmittance": final_transmittance,
    }


def marginalize_elevation_bins(returns_aer, elev_weights=None):
    if elev_weights is None:
        num_elev = returns_aer.shape[1]
        elev_weights = returns_aer.new_full((num_elev,), 1.0 / max(num_elev, 1))
    return torch.einsum("aer,e->ar", returns_aer, elev_weights)


def _sigma2d_to_transmat_precomp_batch(mu_2d, sigma_2d):
    sigma_conditioned = _condition_sigma_2d(sigma_2d)
    try:
        chol = torch.linalg.cholesky(sigma_conditioned)
    except RuntimeError:
        eigvals, eigvecs = torch.linalg.eigh(0.5 * (sigma_conditioned + sigma_conditioned.transpose(-1, -2)))
        chol = eigvecs @ torch.diag_embed(torch.sqrt(torch.clamp(eigvals, min=1e-6)))

    zero = mu_2d.new_zeros(mu_2d.shape[0])
    one = mu_2d.new_ones(mu_2d.shape[0])
    row0 = torch.stack([chol[:, 0, 0], chol[:, 1, 0], mu_2d[:, 0]], dim=-1)
    row1 = torch.stack([zero, chol[:, 1, 1], mu_2d[:, 1]], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


def _can_use_sonar_cuda_rasterizer(device):
    return (
        device.type == "cuda"
        and os.getenv("SONAR_USE_CUDA_RASTERIZER", "1").strip().lower() not in {"0", "false", "no"}
        and "GaussianRasterizer" in globals()
        and "GaussianRasterizationSettings" in globals()
    )


def _make_sonar_rasterizer_settings(image_height, image_width, device, dtype):
    eye = torch.eye(4, device=device, dtype=dtype)
    return GaussianRasterizationSettings(
        image_height=int(image_height),
        image_width=int(image_width),
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device, dtype=dtype),
        ndc2world=eye,
        scale_modifier=1.0,
        viewmatrix=eye,
        projmatrix=eye,
        sh_degree=0,
        campos=torch.zeros(3, device=device, dtype=dtype),
        prefiltered=False,
        additive_mode=True,
        debug=False,
    )


def _rasterize_sonar_event_volume(
    vis_mu,
    vis_sigma,
    vis_range,
    event_surfel_idx,
    event_ray_ids,
    event_returns,
    event_share,
    image_width,
    image_height,
    elev_bins,
):
    device = vis_mu.device
    dtype = vis_mu.dtype
    returns_aer = torch.zeros(image_width, elev_bins, image_height, device=device, dtype=dtype)
    range_aer = torch.zeros_like(returns_aer)
    support_aer = torch.zeros_like(returns_aer)
    surfel_radii = torch.zeros(vis_mu.shape[0], device=device, dtype=dtype)

    linear_opacity = float(os.getenv("SONAR_RASTER_LINEAR_OPACITY", "0.0625"))
    linear_opacity = min(max(linear_opacity, 1.0 / 255.0), 0.99)
    raster_settings = _make_sonar_rasterizer_settings(image_height, image_width, device, dtype)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    zero_plane = torch.zeros(image_width, image_height, device=device, dtype=dtype)
    returns_bins = []
    range_bins = []
    support_bins = []

    for elev_idx in range(int(elev_bins)):
        cur_mask = torch.remainder(event_ray_ids, int(elev_bins)) == elev_idx
        if not bool(cur_mask.any().item()):
            returns_bins.append(zero_plane)
            range_bins.append(zero_plane)
            support_bins.append(zero_plane)
            continue

        cur_surfel_idx = event_surfel_idx[cur_mask]
        cur_range = vis_range[cur_surfel_idx].clamp_min(0.21)
        cur_mu = vis_mu[cur_surfel_idx]
        cur_sigma = vis_sigma[cur_surfel_idx]
        cur_transmat = _sigma2d_to_transmat_precomp_batch(cur_mu, cur_sigma).reshape(-1, 9).contiguous()
        cur_means3d = torch.stack([
            torch.zeros_like(cur_range),
            torch.zeros_like(cur_range),
            cur_range,
        ], dim=-1).contiguous()
        cur_means2d = torch.zeros_like(cur_means3d)
        cur_opacity = torch.full((cur_surfel_idx.shape[0], 1), linear_opacity, device=device, dtype=dtype)
        cur_color = torch.stack(
            [
                event_returns[cur_mask],
                event_returns[cur_mask] * cur_range,
                event_share[cur_mask],
            ],
            dim=-1,
        ) / linear_opacity

        color, event_radii, _, _ = rasterizer(
            means3D=cur_means3d,
            means2D=cur_means2d,
            opacities=cur_opacity,
            colors_precomp=cur_color,
            cov3D_precomp=cur_transmat,
        )
        returns_bins.append(torch.nan_to_num(color[0].transpose(0, 1), nan=0.0, posinf=0.0, neginf=0.0))
        range_bins.append(torch.nan_to_num(color[1].transpose(0, 1), nan=0.0, posinf=0.0, neginf=0.0))
        support_bins.append(torch.nan_to_num(color[2].transpose(0, 1), nan=0.0, posinf=0.0, neginf=0.0))

        if hasattr(surfel_radii, "scatter_reduce_"):
            local_radii = torch.zeros_like(surfel_radii)
            local_radii.scatter_reduce_(0, cur_surfel_idx, event_radii.to(dtype), reduce="amax", include_self=True)
            surfel_radii = torch.maximum(surfel_radii, local_radii)

    if returns_bins:
        returns_aer = torch.stack(returns_bins, dim=1)
        range_aer = torch.stack(range_bins, dim=1)
        support_aer = torch.stack(support_bins, dim=1)

    return returns_aer, range_aer, support_aer, surfel_radii


def _summary_stats(values):
    if values.numel() == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    vals = values.detach().reshape(-1).float()
    return {
        "mean": float(vals.mean().item()),
        "median": float(vals.median().item()),
        "p01": float(torch.quantile(vals, 0.01).item()),
        "p05": float(torch.quantile(vals, 0.05).item()),
        "p95": float(torch.quantile(vals, 0.95).item()),
        "p99": float(torch.quantile(vals, 0.99).item()),
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
    }

def compute_fov_margin(range_vals, azimuth, elevation, sonar_config):
    """
    Compute distance from each point to nearest FOV boundary.

    Used for size-aware FOV checking: a surfel is fully inside FOV only if
    its center is inside AND margin > surfel_radius.

    Args:
        range_vals: [N] distance from sonar origin
        azimuth: [N] horizontal angle (radians)
        elevation: [N] vertical angle (radians)
        sonar_config: SonarConfig with FOV limits

    Returns:
        [N] margin in world units (meters)
    """
    # Angular margins (convert to linear distance at current range)
    # This approximation is accurate for small angles
    az_margin = (sonar_config.half_azimuth_rad - torch.abs(azimuth)) * range_vals
    el_margin = (sonar_config.half_elevation_rad - torch.abs(elevation)) * range_vals

    # Range margins
    range_margin_near = range_vals - sonar_config.range_min
    range_margin_far = sonar_config.range_max - range_vals

    # Minimum margin across all constraints
    margin = torch.min(torch.stack([
        az_margin, el_margin, range_margin_near, range_margin_far
    ], dim=0), dim=0).values

    return margin


def render_sonar(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    sonar_config,
    scale_factor=None,
    sonar_extrinsic=None,
    scaling_modifier=1.0,
    use_range_attenuation=True,
    range_atten_exp=2.0,
    range_atten_gain=1.0,
    range_atten_r0=0.35,
    range_atten_eps=1e-6,
    range_atten_auto_gain=False,
):
    """
    Render the scene using sonar polar projection.
    
    Unlike camera rendering which uses pinhole projection, sonar uses:
    - Polar coordinates (azimuth, range) instead of (x, y, depth)
    - 20-degree elevation beam spread (sum surfels within elevation arc)
    - Lambertian-like intensity model based on surface normal vs sonar direction
    
    Args:
        viewpoint_camera: View object with pose information (camera pose)
        pc: GaussianModel containing surfel positions and properties
        bg_color: Background color tensor [3] on GPU
        sonar_config: SonarConfig instance with sonar parameters
        scale_factor: Optional SonarScaleFactor for pose scaling
        sonar_extrinsic: Optional SonarExtrinsic for camera-to-sonar transform
        scaling_modifier: Scaling modifier for surfel sizes (default 1.0)
        use_range_attenuation: Enable range attenuation path
        range_atten_exp: Attenuation exponent p in 1/(r^p)
        range_atten_gain: Manual gain, or auto-gain seed when auto mode is on
        range_atten_r0: Near-range floor before exponentiation
        range_atten_eps: Numeric epsilon added to attenuation denominator
        range_atten_auto_gain: Auto-calibrate gain from current frame attenuation statistics
        
    Returns:
        Dictionary containing:
        - render: Rendered sonar intensity image [1, H, W]
        - surf_range: Surface range map [1, H, W]
        - surf_normal: Surface normals [3, H, W]
        - visibility_filter: Boolean mask of visible surfels
        - viewspace_points: Screen-space point positions for gradients
    """
    device = pc.get_xyz.device
    runtime_contract = resolve_sonar_render_contract()

    means3D = pc.get_xyz
    viewspace_points = means3D + 0.0
    try:
        viewspace_points.retain_grad()
    except Exception:
        pass

    points_sonar, w2v = _transform_world_points_to_sonar_frame(
        viewspace_points,
        viewpoint_camera,
        scale_factor=scale_factor,
        sonar_extrinsic=sonar_extrinsic,
    )

    if scale_factor is not None:
        means3D_scaled = scale_factor.scale * viewspace_points
    else:
        means3D_scaled = viewspace_points

    R_w2v = w2v[:3, :3]
    t_w2v_scaled = w2v[3, :3]
    R_v2w = R_w2v.T
    sonar_origin_scaled = -R_v2w @ t_w2v_scaled

    N = viewspace_points.shape[0]

    right = points_sonar[:, 0]
    down = points_sonar[:, 1]
    forward = points_sonar[:, 2]

    projection = sonar_project_points(
        viewspace_points,
        viewpoint_camera,
        sonar_config,
        scale_factor=scale_factor,
        sonar_extrinsic=sonar_extrinsic,
    )
    azimuth = projection.azimuth
    range_vals = projection.range_vals
    horiz_dist = torch.sqrt(right * right + forward * forward)
    elevation = torch.atan2(down, horiz_dist.clamp_min(1e-8))
    in_fov = projection.valid

    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    col = torch.clamp(projection.col, 0, W - 1)
    row = torch.clamp(projection.row, 0, H - 1)

    footprint_config = type(
        "SonarRuntimeConfig",
        (),
        {
            "image_width": W,
            "image_height": H,
            "half_azimuth_rad": getattr(sonar_config, "half_azimuth_rad"),
            "half_elevation_rad": getattr(sonar_config, "half_elevation_rad"),
            "range_min": getattr(sonar_config, "range_min"),
            "range_max": getattr(sonar_config, "range_max"),
        },
    )()

    rotations = pc.get_rotation
    normals_world = quaternion_to_normal(rotations)

    diff_to_sonar = sonar_origin_scaled.unsqueeze(0) - means3D_scaled
    dist_to_sonar = torch.norm(diff_to_sonar, dim=-1, keepdim=True) + 1e-8
    dir_to_sonar = diff_to_sonar / dist_to_sonar

    dot_val = torch.sum(normals_world * dir_to_sonar, dim=-1)
    lambertian, lambertian_mode = compute_sonar_lambertian(dot_val)

    opacity = pc.get_opacity.squeeze(-1)
    base_intensity = opacity * lambertian

    attenuation, attenuation_diag = compute_sonar_range_attenuation(
        range_vals,
        use_range_attenuation=use_range_attenuation,
        range_atten_exp=range_atten_exp,
        range_atten_gain=range_atten_gain,
        range_atten_r0=range_atten_r0,
        range_atten_eps=range_atten_eps,
        range_atten_auto_gain=range_atten_auto_gain,
    )
    intensity = base_intensity * attenuation

    returns_aer = torch.zeros(W, int(runtime_contract["elev_bins"]), H, device=device, dtype=viewspace_points.dtype)
    range_aer = torch.zeros_like(returns_aer)
    support_aer = torch.zeros_like(returns_aer)

    if hasattr(pc, "get_scaling"):
        scaling_xy = pc.get_scaling * float(scaling_modifier)
    else:
        scaling_xy = torch.ones((N, 2), dtype=viewspace_points.dtype, device=device)

    sigma_point_config = {
        "kappa": float(os.getenv("SONAR_SIGMA_KAPPA", "0.0")),
        "alpha": float(os.getenv("SONAR_SIGMA_ALPHA", "1.0")),
        "beta": float(os.getenv("SONAR_SIGMA_BETA", "2.0")),
    }

    previous_fallback_used = getattr(viewpoint_camera, "_sonar_sigma_fallback_state", None)
    if previous_fallback_used is not None:
        if not isinstance(previous_fallback_used, torch.Tensor) or previous_fallback_used.shape != (N,):
            previous_fallback_used = None

    footprints = _project_sonar_footprints_batch(
        mean_3d=points_sonar,
        scale_xy=scaling_xy,
        quat_wxyz=rotations,
        mode=runtime_contract["render_mode"],
        sigma_point_config=sigma_point_config,
        sonar_config=footprint_config,
        previous_fallback_used=previous_fallback_used,
    )
    setattr(viewpoint_camera, "_sonar_sigma_fallback_state", footprints["fallback_used"].detach())
    radii = footprints["radii"]
    sigma_fallback_count = int(footprints["fallback_used"].sum().item())
    sigma_invalid_az = int(footprints["sigma_invalid_reason_counts"]["azimuth"].sum().item())
    sigma_invalid_range = int(footprints["sigma_invalid_reason_counts"]["range"].sum().item())
    sigma_invalid_elev = int(footprints["sigma_invalid_reason_counts"]["elevation"].sum().item())
    sigma_invalid_front = int(footprints["sigma_invalid_reason_counts"]["in_front"].sum().item())

    visible_mask = in_fov & (~footprints["skipped"])
    visible_idx = torch.where(visible_mask)[0]
    mass_loss_tensor = viewspace_points.new_zeros((0,))

    if visible_idx.numel() > 0:
        vis_mu = footprints["mu_2d"][visible_idx]
        vis_sigma = footprints["sigma_2d"][visible_idx]
        vis_elev = elevation[visible_idx]
        vis_range = range_vals[visible_idx]
        vis_intensity = intensity[visible_idx]
        vis_opacity = opacity[visible_idx]

        surfel_batch_size = 2048
        num_ray_bins = W * int(runtime_contract["elev_bins"])
        event_surfel_parts = []
        event_ray_parts = []
        event_alpha_parts = []
        event_value_parts = []
        event_range_parts = []
        event_share_parts = []
        mass_loss_parts = []

        for start in range(0, int(visible_idx.shape[0]), surfel_batch_size):
            end = min(start + surfel_batch_size, int(visible_idx.shape[0]))
            cur_mu = vis_mu[start:end]
            cur_sigma = vis_sigma[start:end]
            cur_elev = vis_elev[start:end]
            cur_range = vis_range[start:end]
            cur_intensity = vis_intensity[start:end]
            cur_opacity = vis_opacity[start:end]

            support = _build_multibin_support_weights_batch(
                center_col=cur_mu[:, 0],
                center_elev=cur_elev,
                sigma_2d=cur_sigma,
                sonar_config=footprint_config,
                elev_bins=int(runtime_contract["elev_bins"]),
                topk=int(runtime_contract["occl_topk"]),
                floor_rel=float(runtime_contract["occl_weight_floor_rel"]),
                k_sigma=float(runtime_contract["occl_k_sigma"]),
            )
            mass_loss_parts.append(support["mass_lost"])

            nz = torch.nonzero(support["weights"] > 0, as_tuple=False)
            if nz.numel() == 0:
                continue

            local_surfel = nz[:, 0]
            flat_bin = nz[:, 1].long()
            share = support["weights"][local_surfel, flat_bin]
            event_surfel_parts.append(local_surfel + start)
            event_ray_parts.append(flat_bin)
            event_alpha_parts.append(torch.clamp(cur_opacity[local_surfel] * share, min=0.0, max=1.0))
            event_value_parts.append(cur_intensity[local_surfel] * share)
            event_range_parts.append(cur_range[local_surfel])
            event_share_parts.append(share)

        if mass_loss_parts:
            mass_loss_tensor = torch.cat(mass_loss_parts, dim=0)

        if event_ray_parts:
            event_surfel_idx = torch.cat(event_surfel_parts, dim=0)
            event_ray_ids = torch.cat(event_ray_parts, dim=0)
            event_alpha = torch.cat(event_alpha_parts, dim=0)
            event_value = torch.cat(event_value_parts, dim=0)
            event_range = torch.cat(event_range_parts, dim=0)
            event_share = torch.cat(event_share_parts, dim=0)

            if runtime_contract["occlusion_mode"] == "ray_binned":
                composed = compose_ray_binned_occlusion(
                    ray_ids=event_ray_ids,
                    range_vals=event_range,
                    alpha_vals=event_alpha,
                    value_vals=event_value,
                    num_rays=num_ray_bins,
                )
                event_returns = composed["event_returns"]
            else:
                event_returns = event_value
            event_returns = torch.nan_to_num(event_returns, nan=0.0, posinf=0.0, neginf=0.0)

            if _can_use_sonar_cuda_rasterizer(device):
                raster_returns, raster_range, raster_support, raster_radii = _rasterize_sonar_event_volume(
                    vis_mu=vis_mu,
                    vis_sigma=vis_sigma,
                    vis_range=vis_range,
                    event_surfel_idx=event_surfel_idx,
                    event_ray_ids=event_ray_ids,
                    event_returns=event_returns,
                    event_share=event_share,
                    image_width=W,
                    image_height=H,
                    elev_bins=int(runtime_contract["elev_bins"]),
                )
                returns_aer = raster_returns
                range_aer = raster_range
                support_aer = raster_support
                if bool((raster_radii > 0).any().item()):
                    radii[visible_idx] = torch.maximum(radii[visible_idx], raster_radii)
            else:
                row_profiles = _build_range_profiles_batch(
                    row_centers=vis_mu[:, 1],
                    sigma_rows=torch.sqrt(torch.clamp(vis_sigma[:, 1, 1], min=0.25)),
                    num_rows=H,
                    k_sigma=float(runtime_contract["occl_k_sigma"]),
                )
                returns_flat = torch.zeros(W * int(runtime_contract["elev_bins"]) * H, device=device, dtype=viewspace_points.dtype)
                range_flat = torch.zeros_like(returns_flat)
                support_flat = torch.zeros_like(returns_flat)
                row_coords = torch.arange(H, device=device, dtype=torch.long).unsqueeze(0)
                event_batch_size = 32768

                for start in range(0, int(event_ray_ids.shape[0]), event_batch_size):
                    end = min(start + event_batch_size, int(event_ray_ids.shape[0]))
                    cur_surfel = event_surfel_idx[start:end]
                    cur_row_profiles = row_profiles[cur_surfel]
                    cur_flat_idx = event_ray_ids[start:end].unsqueeze(1) * H + row_coords
                    cur_returns = event_returns[start:end].unsqueeze(1) * cur_row_profiles
                    cur_range = cur_returns * event_range[start:end].unsqueeze(1)
                    cur_support = event_share[start:end].unsqueeze(1) * cur_row_profiles

                    returns_flat = returns_flat.scatter_add(0, cur_flat_idx.reshape(-1), cur_returns.reshape(-1))
                    range_flat = range_flat.scatter_add(0, cur_flat_idx.reshape(-1), cur_range.reshape(-1))
                    support_flat = support_flat.scatter_add(0, cur_flat_idx.reshape(-1), cur_support.reshape(-1))

                returns_aer = returns_flat.view(W, int(runtime_contract["elev_bins"]), H)
                range_aer = range_flat.view(W, int(runtime_contract["elev_bins"]), H)
                support_aer = support_flat.view(W, int(runtime_contract["elev_bins"]), H)

    returns_aer = torch.nan_to_num(returns_aer, nan=0.0, posinf=0.0, neginf=0.0)
    range_aer = torch.nan_to_num(range_aer, nan=0.0, posinf=0.0, neginf=0.0)
    support_aer = torch.nan_to_num(support_aer, nan=0.0, posinf=0.0, neginf=0.0)

    elev_weights = _elevation_bin_weights(
        int(runtime_contract["elev_bins"]),
        runtime_contract["elev_weight_mode"],
        device=device,
        dtype=viewspace_points.dtype,
    )
    rendered_ar = marginalize_elevation_bins(returns_aer, elev_weights=elev_weights)
    range_num_ar = marginalize_elevation_bins(range_aer, elev_weights=elev_weights)
    support_ar = marginalize_elevation_bins(support_aer, elev_weights=elev_weights)

    rendered_image = rendered_ar.transpose(0, 1).unsqueeze(0)
    range_image = range_num_ar.transpose(0, 1).unsqueeze(0)
    weight_sum = support_ar.transpose(0, 1)

    range_image = torch.where(
        weight_sum.unsqueeze(0) > 1e-6,
        range_image / weight_sum.unsqueeze(0),
        torch.zeros_like(range_image),
    )
    rendered_image = torch.nan_to_num(rendered_image, nan=0.0, posinf=0.0, neginf=0.0)
    range_image = torch.nan_to_num(range_image, nan=0.0, posinf=0.0, neginf=0.0)

    rendered_image = torch.clamp(
        rendered_image + 1e-6 * (weight_sum.unsqueeze(0) - weight_sum.detach().unsqueeze(0)),
        0,
        1,
    )
    mask_top_rows = 10
    if mask_top_rows > 0:
        mask = torch.ones_like(rendered_image)
        mask[:, :mask_top_rows, :] = 0
        rendered_image = rendered_image * mask
        range_image = range_image * mask

    rendered_image = rendered_image.expand(3, -1, -1)

    surf_normal = sonar_points_to_normals(
        sonar_ranges_to_points(viewpoint_camera, range_image, sonar_config, scale_factor),
        range_image,
    ).permute(2, 0, 1)

    if mass_loss_tensor.numel() > 0:
        mass_loss_mean = float(mass_loss_tensor.mean().detach().item())
        mass_loss_median = float(mass_loss_tensor.median().detach().item())
        mass_loss_p95 = float(torch.quantile(mass_loss_tensor, 0.95).detach().item())
        mass_loss_p99 = float(torch.quantile(mass_loss_tensor, 0.99).detach().item())
        mass_loss_max = float(mass_loss_tensor.max().detach().item())
        mass_loss_total = float(mass_loss_tensor.sum().detach().item())
    else:
        mass_loss_mean = 0.0
        mass_loss_median = 0.0
        mass_loss_p95 = 0.0
        mass_loss_p99 = 0.0
        mass_loss_max = 0.0
        mass_loss_total = 0.0

    range_span = max(float(sonar_config.range_max - sonar_config.range_min), 1e-6)
    near_thresh = float(sonar_config.range_min + 0.2 * range_span)
    far_thresh = float(sonar_config.range_min + 0.8 * range_span)
    near_mask = in_fov & (range_vals <= near_thresh)
    far_mask = in_fov & (range_vals >= far_thresh)
    near_mean = float(intensity[near_mask].detach().mean().item()) if near_mask.any() else 0.0
    far_mean = float(intensity[far_mask].detach().mean().item()) if far_mask.any() else 0.0
    saturation_rate = float((rendered_image > 0.95).float().mean().item())
    nan_inf_count = int((~torch.isfinite(rendered_image)).sum().item() + (~torch.isfinite(range_image)).sum().item())

    world_s1 = scaling_xy[:, 0]
    world_s2 = scaling_xy[:, 1]
    world_req = torch.sqrt((world_s1 * world_s2).clamp_min(1e-12))
    if in_fov.any():
        in_fov_s1 = world_s1[in_fov]
        in_fov_s2 = world_s2[in_fov]
        in_fov_req = world_req[in_fov]
        in_fov_radii = radii[in_fov]
    else:
        in_fov_s1 = world_s1[:0]
        in_fov_s2 = world_s2[:0]
        in_fov_req = world_req[:0]
        in_fov_radii = radii[:0]

    surfel_stats_every = max(int(os.getenv("SONAR_SURFEL_STATS_EVERY", "50")), 1)

    sonar_diagnostics = {
        "attenuation_enabled": bool(attenuation_diag["enabled"]),
        "attenuation_gain_mode": attenuation_diag["gain_mode"],
        "attenuation_effective_gain": float(attenuation_diag["effective_gain"]),
        "attenuation_exp": float(attenuation_diag["exp"]),
        "attenuation_r0": float(attenuation_diag["r0"]),
        "attenuation_eps": float(attenuation_diag["eps"]),
        "visible_surfel_ratio": float(in_fov.float().mean().item()) if N > 0 else 0.0,
        "near_range_mean_intensity": near_mean,
        "far_range_mean_intensity": far_mean,
        "far_over_near_ratio": far_mean / max(near_mean, 1e-8),
        "near_range_saturation_rate": saturation_rate,
        "nan_inf_count": nan_inf_count,
        "lambertian_mode": lambertian_mode,
        "lambertian_negative_fraction": float((dot_val < 0).float().mean().item()) if N > 0 else 0.0,
        "sonar_render_mode": runtime_contract["render_mode"],
        "sonar_occlusion_mode": runtime_contract["occlusion_mode"],
        "elevation_bin_count": int(runtime_contract["elev_bins"]),
        "elevation_weight_mode": runtime_contract["elev_weight_mode"],
        "surfel_size_stats_schema_version": "v1",
        "surfel_size_stats_world": {
            "s1": _summary_stats(world_s1),
            "s2": _summary_stats(world_s2),
            "r_eq": _summary_stats(world_req),
        },
        "surfel_size_stats_image": {
            "radii_px": _summary_stats(radii),
        },
        "surfel_size_stats_in_fov": {
            "world": {
                "s1": _summary_stats(in_fov_s1),
                "s2": _summary_stats(in_fov_s2),
                "r_eq": _summary_stats(in_fov_req),
            },
            "image": {
                "radii_px": _summary_stats(in_fov_radii),
            },
        },
        "surfel_size_stats_every": surfel_stats_every,
        "sigma_point_fallback_fraction": float(sigma_fallback_count / max(N, 1)),
        "sigma_point_invalid_reason_counts": {
            "azimuth": int(sigma_invalid_az),
            "range": int(sigma_invalid_range),
            "elevation": int(sigma_invalid_elev),
            "in_front": int(sigma_invalid_front),
        },
        "occlusion_support_cap_config": {
            "k_sigma": float(runtime_contract["occl_k_sigma"]),
            "weight_floor_rel": float(runtime_contract["occl_weight_floor_rel"]),
            "topk": int(runtime_contract["occl_topk"]),
        },
        "occlusion_support_cap_mass_loss": {
            "mean": mass_loss_mean,
            "median": mass_loss_median,
            "p95": mass_loss_p95,
            "p99": mass_loss_p99,
            "max": mass_loss_max,
            "mass_lost": mass_loss_total,
        },
    }

    return {
        "render": rendered_image,
        "viewspace_points": viewspace_points,
        "visibility_filter": in_fov,
        "radii": radii,
        "converge": torch.tensor(0.0, device=device),
        "rend_alpha": (weight_sum > 0).float().unsqueeze(0),
        "rend_normal": surf_normal,
        "rend_dist": torch.zeros(1, H, W, device=device),
        "surf_depth": range_image,
        "surf_normal": surf_normal,
        "sonar_diagnostics": sonar_diagnostics,
    }


def quaternion_to_normal(quaternions):
    """
    Convert rotation quaternions to surfel normal vectors.
    
    For 2D Gaussians (surfels), the normal is the local Z-axis of the surfel,
    which is obtained by rotating the unit Z vector [0, 0, 1] by the quaternion.
    
    Args:
        quaternions: Rotation quaternions [N, 4] in (w, x, y, z) format
        
    Returns:
        normals: Unit normal vectors [N, 3]
    """
    # Normalize quaternions
    q = F.normalize(quaternions, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Rotate [0, 0, 1] by quaternion
    # n = q * [0,0,1] * q^-1 
    # Simplified formula for rotating z-axis:
    normal_x = 2 * (x * z + w * y)
    normal_y = 2 * (y * z - w * x)
    normal_z = 1 - 2 * (x * x + y * y)
    
    return torch.stack([normal_x, normal_y, normal_z], dim=-1)
