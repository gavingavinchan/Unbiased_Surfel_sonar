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

    # Get surfel positions and transform to sonar frame under row-major contract.
    means3D = pc.get_xyz  # [N, 3]
    points_sonar, w2v = _transform_world_points_to_sonar_frame(
        means3D,
        viewpoint_camera,
        scale_factor=scale_factor,
        sonar_extrinsic=sonar_extrinsic,
    )

    if scale_factor is not None:
        means3D_scaled = scale_factor.scale * means3D
    else:
        means3D_scaled = means3D

    R_w2v = w2v[:3, :3]
    t_w2v_scaled = w2v[3, :3]
    R_v2w = R_w2v.T
    sonar_origin_scaled = -R_v2w @ t_w2v_scaled

    N = means3D.shape[0]
    
    # Create screenspace points tensor for gradient tracking
    screenspace_points = torch.zeros_like(means3D, requires_grad=True, device=device)
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Compute polar coordinates in camera/view frame.
    right = points_sonar[:, 0]   # +X in camera frame
    down = points_sonar[:, 1]    # +Y in camera frame  
    forward = points_sonar[:, 2] # +Z in camera frame

    projection = sonar_project_points(
        means3D,
        viewpoint_camera,
        sonar_config,
        scale_factor=scale_factor,
        sonar_extrinsic=sonar_extrinsic,
    )
    azimuth = projection.azimuth
    range_vals = projection.range_vals
    horiz_dist = torch.sqrt(right**2 + forward**2)
    elevation = torch.atan2(down, horiz_dist.clamp_min(1e-8))
    # Rendering uses center-based visibility. Size-aware full-extent FOV constraints
    # are handled by debug-side pruning to avoid suppressing boundary bins in images.
    in_fov = projection.valid
    
    # Convert polar to pixel coordinates for valid surfels
    # Column: maps azimuth to [0, width] with flipped direction
    # Convention: positive azimuth → left (low col), negative azimuth → right (high col)
    # Row: maps range [range_min, range_max] to [0, height]
    # Use actual image dimensions from viewpoint (may be downscaled)
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    col = torch.clamp(projection.col, 0, W - 1)
    row = torch.clamp(projection.row, 0, H - 1)
    
    # Get surfel normals in world space (from rotation quaternions)
    rotations = pc.get_rotation  # [N, 4] quaternions
    normals_world = quaternion_to_normal(rotations)  # [N, 3]
    
    # Compute direction from surfel to sonar origin (for Lambertian intensity)
    # Add eps to avoid nan from normalizing zero vectors when surfel is at sonar origin
    diff_to_sonar = sonar_origin_scaled.unsqueeze(0) - means3D_scaled  # [N, 3]
    dist_to_sonar = torch.norm(diff_to_sonar, dim=-1, keepdim=True) + 1e-8  # [N, 1]
    dir_to_sonar = diff_to_sonar / dist_to_sonar  # [N, 3]
    
    # Lambertian intensity: I = max(0, n · d)
    lambertian = torch.clamp(torch.sum(normals_world * dir_to_sonar, dim=-1), min=0)  # [N]
    
    # Base intensity from surfel opacity and Lambertian response.
    opacity = pc.get_opacity.squeeze(-1)  # [N]
    base_intensity = opacity * lambertian  # [N]

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
    
    # Initialize output images (use actual viewpoint dimensions)
    out_H = viewpoint_camera.image_height
    out_W = viewpoint_camera.image_width
    rendered_image = torch.zeros(1, out_H, out_W, device=device)
    range_image = torch.zeros(1, out_H, out_W, device=device)
    weight_sum = torch.zeros(out_H, out_W, device=device)
    
    # Splat each surfel to its pixel bin using differentiable scatter operations
    # This uses vectorized scatter_add for gradient flow
    
    if in_fov.any():
        valid_idx = torch.where(in_fov)[0]
        
        valid_col = col[valid_idx]
        valid_row = row[valid_idx]
        valid_intensity = intensity[valid_idx]
        valid_range = range_vals[valid_idx]
        
        # Bilinear splatting coordinates
        col_floor = valid_col.floor().long()
        col_ceil = (col_floor + 1).clamp(max=out_W-1)
        row_floor = valid_row.floor().long()
        row_ceil = (row_floor + 1).clamp(max=out_H-1)
        
        col_frac = valid_col - col_floor.float()
        row_frac = valid_row - row_floor.float()
        
        # Compute weights for 4 neighbors (differentiable)
        w00 = (1 - col_frac) * (1 - row_frac)  # top-left
        w01 = (1 - col_frac) * row_frac        # bottom-left
        w10 = col_frac * (1 - row_frac)        # top-right
        w11 = col_frac * row_frac              # bottom-right
        
        # Convert 2D indices to 1D for scatter_add
        # index = row * width + col
        idx_00 = row_floor * out_W + col_floor
        idx_01 = row_ceil * out_W + col_floor
        idx_10 = row_floor * out_W + col_ceil
        idx_11 = row_ceil * out_W + col_ceil
        
        # Flatten output tensors for scatter_add
        rendered_flat = rendered_image.view(-1)
        range_flat = range_image.view(-1)
        weight_flat = weight_sum.view(-1)
        
        # Weighted intensity contributions (differentiable through weights and intensity)
        contrib_00 = w00 * valid_intensity
        contrib_01 = w01 * valid_intensity
        contrib_10 = w10 * valid_intensity
        contrib_11 = w11 * valid_intensity
        
        # Use scatter_add for differentiable accumulation
        rendered_flat.scatter_add_(0, idx_00, contrib_00)
        rendered_flat.scatter_add_(0, idx_01, contrib_01)
        rendered_flat.scatter_add_(0, idx_10, contrib_10)
        rendered_flat.scatter_add_(0, idx_11, contrib_11)
        
        # Weighted range contributions
        range_contrib_00 = w00 * valid_intensity * valid_range
        range_contrib_01 = w01 * valid_intensity * valid_range
        range_contrib_10 = w10 * valid_intensity * valid_range
        range_contrib_11 = w11 * valid_intensity * valid_range
        
        range_flat.scatter_add_(0, idx_00, range_contrib_00)
        range_flat.scatter_add_(0, idx_01, range_contrib_01)
        range_flat.scatter_add_(0, idx_10, range_contrib_10)
        range_flat.scatter_add_(0, idx_11, range_contrib_11)
        
        # Weight accumulation for normalization
        weight_flat.scatter_add_(0, idx_00, contrib_00.detach())
        weight_flat.scatter_add_(0, idx_01, contrib_01.detach())
        weight_flat.scatter_add_(0, idx_10, contrib_10.detach())
        weight_flat.scatter_add_(0, idx_11, contrib_11.detach())
        
        # Reshape back
        rendered_image = rendered_flat.view(1, out_H, out_W)
        range_image = range_flat.view(1, out_H, out_W)
        weight_sum = weight_flat.view(out_H, out_W)
    
    # Normalize range by intensity weight
    range_image = torch.where(
        weight_sum.unsqueeze(0) > 1e-6,
        range_image / weight_sum.unsqueeze(0),
        torch.zeros_like(range_image)
    )
    
    # Clamp intensity to [0, 1]
    rendered_image = torch.clamp(rendered_image, 0, 1)
    
    # Mask out top rows (closest range bins often have artifacts)
    # Use differentiable masking instead of in-place assignment to preserve gradients
    mask_top_rows = 10
    if mask_top_rows > 0:
        mask = torch.ones_like(rendered_image)
        mask[:, :mask_top_rows, :] = 0
        rendered_image = rendered_image * mask
        range_image = range_image * mask
    
    # Expand to 3 channels for compatibility with RGB loss functions
    rendered_image = rendered_image.expand(3, -1, -1)  # [3, H, W]
    
    # Compute surface normals from range image
    surf_normal = sonar_points_to_normals(
        sonar_ranges_to_points(viewpoint_camera, range_image, sonar_config, scale_factor),
        range_image
    ).permute(2, 0, 1)  # [3, H, W]

    range_span = max(float(sonar_config.range_max - sonar_config.range_min), 1e-6)
    near_thresh = float(sonar_config.range_min + 0.2 * range_span)
    far_thresh = float(sonar_config.range_min + 0.8 * range_span)
    near_mask = in_fov & (range_vals <= near_thresh)
    far_mask = in_fov & (range_vals >= far_thresh)
    near_mean = float(intensity[near_mask].detach().mean().item()) if near_mask.any() else 0.0
    far_mean = float(intensity[far_mask].detach().mean().item()) if far_mask.any() else 0.0
    saturation_rate = float((rendered_image > 0.95).float().mean().item())
    nan_inf_count = int((~torch.isfinite(rendered_image)).sum().item() + (~torch.isfinite(range_image)).sum().item())
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
    }
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": in_fov,
        "radii": torch.zeros(N, device=device),  # Placeholder for compatibility
        "converge": torch.tensor(0.0, device=device),  # Placeholder
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
