import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

# =============================================================================
# Camera (Pinhole) Projection Functions
# =============================================================================

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth, sonar_mode=False, sonar_config=None, scale_factor=None):
    """
    Convert depth/range map to surface normals.
    
    Args:
        view: Camera/sonar view object (provides pose transforms)
        depth: Depth map [1, H, W] for camera, or range image [1, H, W] for sonar
        sonar_mode: If True, use sonar polar geometry instead of pinhole
        sonar_config: SonarConfig instance (required if sonar_mode=True)
        scale_factor: SonarScaleFactor instance for pose scaling (optional)
        
    Returns:
        Normal map [H, W, 3] in world coordinates
    """
    if sonar_mode:
        if sonar_config is None:
            raise ValueError("sonar_config required when sonar_mode=True")
        points = sonar_ranges_to_points(view, depth, sonar_config, scale_factor)
        return sonar_points_to_normals(points, depth)
    else:
        # Original camera path
        points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
        output = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output[1:-1, 1:-1, :] = normal_map
        return output


# =============================================================================
# Sonar (Polar) Projection Functions
# =============================================================================

def sonar_ranges_to_points(view, range_image, sonar_config, scale_factor=None):
    """
    Convert sonar range image to 3D world-space points.
    
    Sonar image coordinate system:
    - Columns (horizontal): Azimuth angle
      - Center column = 0 degrees (forward)
      - Left columns = positive azimuth
      - Right columns = negative azimuth
    - Rows (vertical): Range
      - Top row = range_min (closest)
      - Bottom row = range_max (farthest)
    
    Sonar frame convention:
    - +X = forward (boresight direction)
    - +Y = right
    - +Z = down
    - Azimuth θ: +X direction = negative azimuth, -Y direction = positive azimuth
    
    Args:
        view: Sonar view object (provides world_view_transform for pose)
        range_image: Range values [1, H, W] in meters
        sonar_config: SonarConfig instance with sonar parameters
        scale_factor: Optional SonarScaleFactor to scale pose translation
        
    Returns:
        points: World-space 3D points [H, W, 3]
    """
    # Get range image shape
    if range_image.dim() == 3:
        range_image = range_image.squeeze(0)  # [H, W]
    H, W = range_image.shape
    
    # Get azimuth angles for each column
    # Center column = 0 degrees, left = negative, right = positive
    azimuth_grid = sonar_config.azimuth_grid[:W]  # [W]
    
    # Expand to full grid
    azimuth = azimuth_grid[None, :].expand(H, W)  # [H, W]
    
    # Range values come directly from the image (already in meters)
    r = range_image  # [H, W]
    
    # Convert polar to Cartesian in sonar frame
    # Assumption: elevation = 0 (flat fan beam)
    # Convention: +X = negative azimuth, so we negate y component
    # x = r * cos(θ) (forward)
    # y = -r * sin(θ) (right, negated because positive azimuth = left)
    # z = 0 (no elevation info)
    x_s = r * torch.cos(azimuth)
    y_s = -r * torch.sin(azimuth)  # Negate to flip azimuth direction
    z_s = torch.zeros_like(r)
    
    # Stack to get points in sonar frame [H, W, 3]
    points_sonar = torch.stack([x_s, y_s, z_s], dim=-1)
    
    # Get sonar-to-world transform from view WITHOUT using torch.inverse()
    # world_view_transform is [R|t] where camera_pos = -R^T @ t
    w2v = view.world_view_transform  # [4, 4]
    R_w2v = w2v[:3, :3]  # rotation world-to-view
    t_w2v = w2v[:3, 3]   # translation
    
    # Apply scale factor to translation if provided
    if scale_factor is not None:
        t_w2v_scaled = scale_factor.scale * t_w2v
    else:
        t_w2v_scaled = t_w2v
    
    # Compute view-to-world transform without torch.inverse()
    # For [R|t], the inverse is [R^T | -R^T @ t]
    R_v2w = R_w2v.T  # transpose = inverse for orthogonal rotation
    t_v2w = -R_v2w @ t_w2v_scaled  # camera/sonar origin in world coords
    
    # Transform points to world: p_world = R_v2w @ p_sonar + t_v2w
    # Reshape for batch matrix multiply: [H*W, 3] @ [3, 3].T + [3]
    points_flat = points_sonar.reshape(-1, 3)  # [H*W, 3]
    points_world_flat = points_flat @ R_v2w.T + t_v2w  # [H*W, 3]
    
    # Reshape back to [H, W, 3]
    points_world = points_world_flat.reshape(H, W, 3)
    
    return points_world


def sonar_points_to_normals(points, range_image, intensity_threshold=0.01):
    """
    Compute surface normals from sonar 3D points using finite differences.
    
    Normals are computed along the range (row) and azimuth (column) directions
    using central differences, then combined via cross product.
    
    Args:
        points: World-space 3D points [H, W, 3]
        range_image: Original range image [1, H, W] or [H, W] for validity masking
        intensity_threshold: Threshold for valid pixels (default 0.01)
        
    Returns:
        normals: Unit normal vectors [H, W, 3], invalid pixels get [0, 0, 0]
    """
    H, W, _ = points.shape
    normals = torch.zeros_like(points)
    
    # Create validity mask from range image
    if range_image.dim() == 3:
        range_image = range_image.squeeze(0)
    valid_mask = range_image > intensity_threshold  # [H, W]
    
    # Compute finite differences along range (rows) and azimuth (columns)
    # Using central differences for better accuracy
    
    # dp/dr: derivative along range (row direction)
    # points[i+1, j] - points[i-1, j]
    dp_dr = points[2:, 1:-1] - points[:-2, 1:-1]  # [H-2, W-2, 3]
    
    # dp/dtheta: derivative along azimuth (column direction)
    # points[i, j+1] - points[i, j-1]
    dp_dtheta = points[1:-1, 2:] - points[1:-1, :-2]  # [H-2, W-2, 3]
    
    # Cross product to get normal (order determines direction)
    # We want normals pointing toward the sonar (into the scene from viewer's perspective)
    n = torch.cross(dp_dtheta, dp_dr, dim=-1)  # [H-2, W-2, 3]
    
    # Normalize
    n = F.normalize(n, dim=-1)
    
    # Check validity: center pixel and all 4 neighbors must be valid
    valid_center = valid_mask[1:-1, 1:-1]
    valid_up = valid_mask[:-2, 1:-1]
    valid_down = valid_mask[2:, 1:-1]
    valid_left = valid_mask[1:-1, :-2]
    valid_right = valid_mask[1:-1, 2:]
    
    full_valid = valid_center & valid_up & valid_down & valid_left & valid_right
    
    # Zero out invalid normals
    n = n * full_valid[..., None].float()
    
    # Place in output
    normals[1:-1, 1:-1] = n
    
    return normals


def get_sonar_valid_mask(intensity_image, threshold=0.01):
    """
    Create a validity mask from sonar intensity image.
    
    Black pixels (intensity near 0) indicate no sonar return.
    
    Args:
        intensity_image: Sonar intensity image [1, H, W] or [H, W], values in [0, 1]
        threshold: Intensity threshold (default 0.01)
        
    Returns:
        valid_mask: Boolean mask [H, W] where True = valid return
    """
    if intensity_image.dim() == 3:
        intensity_image = intensity_image.squeeze(0)
    return intensity_image > threshold