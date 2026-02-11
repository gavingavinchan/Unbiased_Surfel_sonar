#
# Sonar utilities for 2D Gaussian Splatting
# Implements sonar-specific projection and scale factor learning
#

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Dict


SONAR_CAMERA_FRAME_CONVENTION = "+X right, +Y down, +Z forward"
SONAR_IMAGE_CONVENTION = "columns: left=+azimuth right=-azimuth; rows: top=near bottom=far"
SONAR_MOUNT_TRANSLATION_CAM = (0.0, -0.10, -0.08)
SONAR_MOUNT_PITCH_DEG = 5.0


@dataclass
class ConventionCheckReport:
    azimuth_left_rad: float
    azimuth_right_rad: float
    positive_elevation_y: float
    negative_elevation_y: float
    extrinsic_roundtrip_max_abs: float
    layout_roundtrip_max_abs: float


def sonar_polar_to_points(azimuth: torch.Tensor, elevation: torch.Tensor, range_vals: torch.Tensor) -> torch.Tensor:
    """
    Convert sonar polar bins into camera/view-frame 3D points.

    Canonical convention:
      - +X right, +Y down, +Z forward
      - left image columns correspond to positive azimuth
      - positive elevation maps to +Y (down)
    """
    x = -range_vals * torch.sin(azimuth) * torch.cos(elevation)
    y = range_vals * torch.sin(elevation)
    z = range_vals * torch.cos(azimuth) * torch.cos(elevation)
    return torch.stack([x, y, z], dim=-1)


def get_scaled_world_to_view_transform(viewpoint_camera, scale_factor=None, sonar_extrinsic=None) -> torch.Tensor:
    """
    Build world->view transform using repository row-major layout.

    Layout contract:
      - rotation in [:3, :3]
      - translation in [3, :3]
      - first three entries of [:3, 3] expected to be 0
    """
    w2v = viewpoint_camera.world_view_transform.clone()
    if scale_factor is not None:
        w2v[3, :3] = scale_factor.scale * w2v[3, :3]
    if sonar_extrinsic is not None:
        w2v = sonar_extrinsic(w2v)
    return w2v


def view_points_to_world(points_view: torch.Tensor, w2v: torch.Tensor, scale_factor=None) -> torch.Tensor:
    """Convert view-frame points to world-frame points under row-major transform semantics."""
    R_w2v = w2v[:3, :3]
    t_w2v = w2v[3, :3]
    points_world_scaled = (points_view - t_w2v.unsqueeze(0)) @ R_w2v
    if scale_factor is not None:
        points_world = points_world_scaled / scale_factor.scale
    else:
        points_world = points_world_scaled
    return points_world


def back_project_bins(
    frame_idx: int,
    rows: torch.Tensor,
    cols: torch.Tensor,
    elev_bins: torch.Tensor,
    *,
    cameras,
    sonar_config,
    scale_factor,
    sonar_extrinsic=None,
) -> torch.Tensor:
    """
    Back-project selected sonar bins into world-frame points.

    Contract:
      - rows/cols are [P]
      - elev_bins is [K] (radians, +elevation -> +Y)
      - output is [P, K, 3] in world coordinates (COLMAP scale)
    """
    if frame_idx < 0 or frame_idx >= len(cameras):
        raise IndexError(f"frame_idx={frame_idx} out of range for {len(cameras)} cameras")

    if rows.dim() != 1 or cols.dim() != 1:
        raise ValueError("rows and cols must be 1D tensors")
    if rows.shape[0] != cols.shape[0]:
        raise ValueError(f"rows/cols length mismatch: {rows.shape[0]} vs {cols.shape[0]}")
    if elev_bins.dim() != 1:
        raise ValueError("elev_bins must be a 1D tensor")

    device = rows.device
    camera = cameras[frame_idx]
    rows_f = rows.to(dtype=torch.float32)
    cols_f = cols.to(dtype=torch.float32)
    elev_bins = elev_bins.to(device=device, dtype=torch.float32)

    azimuth, range_vals = sonar_config.pixel_to_polar(cols_f, rows_f)
    P = rows.shape[0]
    K = elev_bins.shape[0]

    azimuth_pk = azimuth[:, None].expand(P, K)
    range_pk = range_vals[:, None].expand(P, K)
    elevation_pk = elev_bins[None, :].expand(P, K)

    points_view = sonar_polar_to_points(
        azimuth_pk.reshape(-1),
        elevation_pk.reshape(-1),
        range_pk.reshape(-1),
    )

    w2v = get_scaled_world_to_view_transform(camera, scale_factor=scale_factor, sonar_extrinsic=sonar_extrinsic)
    points_world = view_points_to_world(points_view, w2v, scale_factor=scale_factor)
    return points_world.reshape(P, K, 3)


def _raise_convention_error(name: str, details: str):
    raise RuntimeError(
        f"[SONAR_CONVENTION_ASSERT] {name} failed: {details}. "
        "Set SONAR_CONVENTION_ASSERTS=0 to bypass (not recommended)."
    )


def assert_azimuth_sign_convention(sonar_config, atol: float = 1e-6) -> Dict[str, float]:
    """Validate left=+azimuth, right=-azimuth image convention."""
    device = sonar_config.azimuth_grid.device
    row_mid = torch.tensor([sonar_config.image_height / 2.0], device=device)
    col_left = torch.tensor([0.0], device=device)
    col_right = torch.tensor([float(sonar_config.image_width - 1)], device=device)

    az_left, _ = sonar_config.pixel_to_polar(col_left, row_mid)
    az_right, _ = sonar_config.pixel_to_polar(col_right, row_mid)
    left_ok = az_left.item() > atol
    right_ok = az_right.item() < -atol

    if not (left_ok and right_ok):
        _raise_convention_error(
            "azimuth_sign",
            f"expected left>0 and right<0 but got left={az_left.item():.6f}, right={az_right.item():.6f}",
        )
    return {
        "azimuth_left_rad": az_left.item(),
        "azimuth_right_rad": az_right.item(),
    }


def assert_elevation_sign_convention(device: str = "cuda", atol: float = 1e-6) -> Dict[str, float]:
    """Validate +elevation -> +Y and -elevation -> -Y in camera/view frame."""
    elev = torch.tensor([0.1, -0.1], device=device)
    azimuth = torch.zeros_like(elev)
    range_vals = torch.ones_like(elev)
    points = sonar_polar_to_points(azimuth, elev, range_vals)

    pos_y = points[0, 1].item()
    neg_y = points[1, 1].item()
    if not (pos_y > atol and neg_y < -atol):
        _raise_convention_error(
            "elevation_sign",
            f"expected +elev->+Y and -elev->-Y but got +elev Y={pos_y:.6f}, -elev Y={neg_y:.6f}",
        )
    return {
        "positive_elevation_y": pos_y,
        "negative_elevation_y": neg_y,
    }


def assert_transform_roundtrip(sample_camera=None, device: str = "cuda", atol: float = 1e-5) -> Dict[str, float]:
    """Validate row-major transform contract and camera->sonar->camera roundtrip."""
    T_c2s = get_camera_to_sonar_transform(device=device)
    if torch.max(torch.abs(T_c2s[:3, 3])).item() > atol:
        _raise_convention_error(
            "extrinsic_layout",
            f"translation must be stored in row 3 but max(abs(T[:3,3]))={torch.max(torch.abs(T_c2s[:3, 3])).item():.6e}",
        )

    T_s2c = torch.inverse(T_c2s)
    pts = torch.tensor(
        [[0.20, -0.10, 1.30], [-0.30, 0.05, 2.00], [0.05, 0.20, 0.60]],
        device=device,
        dtype=T_c2s.dtype,
    )
    ones = torch.ones(pts.shape[0], 1, device=device, dtype=T_c2s.dtype)
    pts_h = torch.cat([pts, ones], dim=1)
    pts_rt = ((pts_h @ T_c2s) @ T_s2c)[:, :3]
    extrinsic_roundtrip_max_abs = torch.max(torch.abs(pts_rt - pts)).item()
    if extrinsic_roundtrip_max_abs > atol:
        _raise_convention_error(
            "extrinsic_roundtrip",
            f"max abs error {extrinsic_roundtrip_max_abs:.6e} exceeds atol={atol}",
        )

    layout_roundtrip_max_abs = 0.0
    if sample_camera is not None:
        w2v = sample_camera.world_view_transform
        if torch.max(torch.abs(w2v[:3, 3])).item() > atol:
            _raise_convention_error(
                "world_view_layout",
                f"expected translation in row 3 but max(abs(T[:3,3]))={torch.max(torch.abs(w2v[:3, 3])).item():.6e}",
            )

        R_w2v = w2v[:3, :3]
        t_w2v = w2v[3, :3]
        pts_cam = pts.to(device=w2v.device, dtype=w2v.dtype)
        # Validate transform roundtrip under renderer semantics:
        # p_view = p_world @ R.T + t; p_world = (p_view - t) @ R
        pts_view = pts_cam @ R_w2v.T + t_w2v
        pts_recovered = (pts_view - t_w2v) @ R_w2v
        layout_roundtrip_max_abs = torch.max(torch.abs(pts_recovered - pts_cam)).item()
        if layout_roundtrip_max_abs > atol:
            _raise_convention_error(
                "world_view_transform_contract",
                f"renderer-semantic roundtrip differs by {layout_roundtrip_max_abs:.6e}",
            )

    return {
        "extrinsic_roundtrip_max_abs": extrinsic_roundtrip_max_abs,
        "layout_roundtrip_max_abs": layout_roundtrip_max_abs,
    }


def run_sonar_convention_asserts(sonar_config, sample_camera=None, device: str = "cuda", atol: float = 1e-5) -> ConventionCheckReport:
    """Run fail-fast sonar convention checks and return diagnostics."""
    azimuth_report = assert_azimuth_sign_convention(sonar_config, atol=atol)
    elevation_report = assert_elevation_sign_convention(device=device, atol=atol)
    transform_report = assert_transform_roundtrip(sample_camera=sample_camera, device=device, atol=atol)
    return ConventionCheckReport(
        azimuth_left_rad=azimuth_report["azimuth_left_rad"],
        azimuth_right_rad=azimuth_report["azimuth_right_rad"],
        positive_elevation_y=elevation_report["positive_elevation_y"],
        negative_elevation_y=elevation_report["negative_elevation_y"],
        extrinsic_roundtrip_max_abs=transform_report["extrinsic_roundtrip_max_abs"],
        layout_roundtrip_max_abs=transform_report["layout_roundtrip_max_abs"],
    )


class SonarScaleFactor(nn.Module):
    """
    Learnable scale factor to align COLMAP arbitrary-scale poses with sonar metric range.
    
    COLMAP produces poses with arbitrary scale (e.g., scene might be 10x too small).
    Sonar ranges are in real meters. This module learns a single scale factor `s` such that:
        scaled_translation = s * colmap_translation
    
    The scale factor converges because:
    - If scale is too small: rendered ranges will be too small, loss will push scale up
    - If scale is too large: rendered ranges will be too large, loss will push scale down
    - Sonar's physical range measurements provide the ground truth reference
    """
    
    def __init__(self, init_value: float = 1.0):
        """
        Args:
            init_value: Initial scale factor value (default 1.0)
        """
        super().__init__()
        # Use log scale internally for numerical stability and to ensure positive values
        # scale = exp(log_scale), so log_scale = log(init_value)
        self._log_scale = nn.Parameter(torch.tensor(math.log(init_value)))
    
    @property
    def scale(self) -> torch.Tensor:
        """Get the actual scale factor (always positive due to exp)."""
        return torch.exp(self._log_scale)
    
    def forward(self, translation: torch.Tensor) -> torch.Tensor:
        """
        Apply scale factor to translation vector(s).
        
        Args:
            translation: Translation vector(s) of shape [..., 3] or [3]
            
        Returns:
            Scaled translation of same shape
        """
        return self.scale * translation
    
    def get_scale_value(self) -> float:
        """Get current scale factor as a Python float for logging."""
        return self.scale.item()
    
    def get_log_scale_grad(self) -> float:
        """Get gradient of log_scale for monitoring convergence."""
        if self._log_scale.grad is not None:
            return self._log_scale.grad.item()
        return 0.0


class SonarConfig:
    """
    Configuration class for sonar parameters.
    Precomputes grids and mappings for efficient projection.
    """
    
    def __init__(
        self,
        image_width: int = 256,
        image_height: int = 200,
        azimuth_fov: float = 120.0,      # degrees
        elevation_fov: float = 20.0,     # degrees  
        range_min: float = 0.2,          # meters
        range_max: float = 3.0,          # meters
        intensity_threshold: float = 0.01,
        device: str = "cuda"
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.azimuth_fov = azimuth_fov
        self.elevation_fov = elevation_fov
        self.range_min = range_min
        self.range_max = range_max
        self.intensity_threshold = intensity_threshold
        self.device = device
        
        # Convert FOV to radians
        self.azimuth_fov_rad = math.radians(azimuth_fov)
        self.elevation_fov_rad = math.radians(elevation_fov)
        self.half_azimuth_rad = self.azimuth_fov_rad / 2  # ±60 degrees
        self.half_elevation_rad = self.elevation_fov_rad / 2  # ±10 degrees
        
        # Precompute azimuth angles for each column
        # Center column (128) = 0 degrees
        # Left (col 0) = +60 degrees (positive azimuth), Right (col 255) = -60 degrees (negative azimuth)
        # Convention: +X direction in world/sonar frame = negative azimuth
        cols = torch.arange(image_width, dtype=torch.float32, device=device)
        self.azimuth_grid = -(cols - image_width / 2) / (image_width / 2) * self.half_azimuth_rad
        
        # Precompute range values for each row
        # Top row (0) = range_min, Bottom row (199) = range_max
        rows = torch.arange(image_height, dtype=torch.float32, device=device)
        self.range_grid = range_min + (rows / image_height) * (range_max - range_min)
        
        # Create meshgrid for full image
        self.azimuth_mesh, self.range_mesh = torch.meshgrid(
            self.azimuth_grid, self.range_grid, indexing='xy'
        )
        # Transpose to get shape [H, W] where H=rows (range), W=cols (azimuth)
        self.azimuth_mesh = self.azimuth_mesh.T  # [H, W]
        self.range_mesh = self.range_mesh.T      # [H, W]
    
    def pixel_to_polar(self, col: torch.Tensor, row: torch.Tensor):
        """
        Convert pixel coordinates to polar (azimuth, range).
        
        Convention: +X direction = negative azimuth
        - Left side of image (col 0) = positive azimuth
        - Right side of image (col 255) = negative azimuth
        
        Args:
            col: Column indices (can be float for sub-pixel)
            row: Row indices (can be float for sub-pixel)
            
        Returns:
            azimuth: Azimuth angles in radians
            range_val: Range values in meters
        """
        # Negate to flip azimuth direction: left = positive, right = negative
        azimuth = -(col - self.image_width / 2) / (self.image_width / 2) * self.half_azimuth_rad
        range_val = self.range_min + (row / self.image_height) * (self.range_max - self.range_min)
        return azimuth, range_val
    
    def polar_to_pixel(self, azimuth: torch.Tensor, range_val: torch.Tensor):
        """
        Convert polar (azimuth, range) to pixel coordinates.
        
        Convention: +X direction = negative azimuth
        - Positive azimuth → left side of image (lower col)
        - Negative azimuth → right side of image (higher col)
        
        Args:
            azimuth: Azimuth angles in radians
            range_val: Range values in meters
            
        Returns:
            col: Column indices (float)
            row: Row indices (float)
        """
        # Negate azimuth to flip direction: positive azimuth → left (lower col)
        col = (-azimuth / self.half_azimuth_rad + 1) * (self.image_width / 2)
        row = (range_val - self.range_min) / (self.range_max - self.range_min) * self.image_height
        return col, row
    
    def is_in_fov(self, azimuth: torch.Tensor, elevation: torch.Tensor, range_val: torch.Tensor):
        """
        Check if points are within sonar field of view.
        
        Args:
            azimuth: Azimuth angles in radians
            elevation: Elevation angles in radians
            range_val: Range values in meters
            
        Returns:
            Boolean mask of valid points
        """
        valid_azimuth = torch.abs(azimuth) <= self.half_azimuth_rad
        valid_elevation = torch.abs(elevation) <= self.half_elevation_rad
        valid_range = (range_val >= self.range_min) & (range_val <= self.range_max)
        return valid_azimuth & valid_elevation & valid_range


def build_sonar_config(args) -> SonarConfig:
    """
    Build SonarConfig from command-line arguments.
    
    Args:
        args: Parsed arguments containing sonar parameters
        
    Returns:
        SonarConfig instance
    """
    return SonarConfig(
        image_width=256,   # Sonoptix Echo default
        image_height=200,  # Sonoptix Echo default
        azimuth_fov=args.sonar_azimuth_fov,
        elevation_fov=args.sonar_elevation_fov,
        range_min=args.sonar_range_min,
        range_max=args.sonar_range_max,
        intensity_threshold=args.sonar_intensity_threshold,
        device="cuda"
    )


# =============================================================================
# Camera-to-Sonar Extrinsic Transform
# =============================================================================

def get_camera_to_sonar_transform(device="cuda"):
    """
    Get row-major camera->sonar extrinsic transform.

    Contract:
      - Translation is stored in row 3 (T[3, :3]).
      - Intended for row-vector transforms: p_out = p_in @ R.T + t.
      - Mount tuple in camera frame is SONAR_MOUNT_TRANSLATION_CAM.

    Returns:
        T_cam_to_sonar: [4, 4] row-major homogeneous transform.
    """
    dtype = torch.float32
    translation_cam = torch.tensor(SONAR_MOUNT_TRANSLATION_CAM, device=device, dtype=dtype)
    
    # Pitch down = +rotation around camera X in camera convention
    pitch_rad = math.radians(SONAR_MOUNT_PITCH_DEG)
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    
    # Rotation matrix for pitch (around X-axis)
    R_pitch = torch.tensor([
        [1.0,    0.0,     0.0],
        [0.0,  cos_p, -sin_p],
        [0.0,  sin_p,  cos_p]
    ], device=device, dtype=dtype)
    
    # Also need to transform from camera convention to sonar convention
    # Camera: +Z forward, +X right, +Y down
    # Sonar:  +X forward, +Y right, +Z down
    # This is a permutation: sonar_X = cam_Z, sonar_Y = cam_X, sonar_Z = cam_Y
    R_convention = torch.tensor([
        [0.0, 0.0, 1.0],  # sonar_X = cam_Z
        [1.0, 0.0, 0.0],  # sonar_Y = cam_X
        [0.0, 1.0, 0.0]   # sonar_Z = cam_Y
    ], device=device, dtype=dtype)
    
    # Combined rotation: apply pitch in camera frame, then convert convention.
    R_cam_to_sonar = R_convention @ R_pitch

    # t = -R * p_mount_cam for camera->sonar point transform.
    t_cam_to_sonar = -R_cam_to_sonar @ translation_cam
    
    # Build row-major homogeneous transform (translation in row 3).
    T_cam_to_sonar = torch.eye(4, device=device, dtype=dtype)
    T_cam_to_sonar[:3, :3] = R_cam_to_sonar
    T_cam_to_sonar[3, :3] = t_cam_to_sonar
    
    return T_cam_to_sonar


def apply_camera_to_sonar_extrinsic(camera_pose_w2c, device="cuda"):
    """
    Transform a camera pose (world-to-camera) to sonar pose (world-to-sonar).
    
    Given row-major world-to-camera transform T_w2c, compute world-to-sonar as:
        T_w2s = T_w2c @ T_c2s
    
    Args:
        camera_pose_w2c: 4x4 world-to-camera transformation matrix
        device: Device for computation
        
    Returns:
        sonar_pose_w2s: 4x4 world-to-sonar transformation matrix
    """
    T_c2s = get_camera_to_sonar_transform(device=camera_pose_w2c.device).to(dtype=camera_pose_w2c.dtype)
    T_w2s = camera_pose_w2c @ T_c2s
    return T_w2s


class SonarExtrinsic(nn.Module):
    """
    Camera-to-sonar extrinsic calibration.
    
    The sonar is mounted with SONAR_MOUNT_TRANSLATION_CAM and SONAR_MOUNT_PITCH_DEG.
    This module stores the fixed extrinsic transform and can apply it to camera poses.
    """
    
    def __init__(self, device="cuda"):
        super().__init__()
        # Register as buffer (not trainable, but moves with model)
        T_c2s = get_camera_to_sonar_transform(device)
        self.register_buffer('T_cam_to_sonar', T_c2s)
    
    def forward(self, camera_pose_w2c):
        """
        Transform world-to-camera pose to world-to-sonar pose.
        
        Args:
            camera_pose_w2c: [4, 4] world-to-camera transformation
            
        Returns:
            [4, 4] world-to-sonar transformation
        """
        return camera_pose_w2c @ self.T_cam_to_sonar
    
    def inverse_transform(self, sonar_pose_w2s):
        """
        Transform world-to-sonar pose back to world-to-camera pose.
        
        Args:
            sonar_pose_w2s: [4, 4] world-to-sonar transformation
            
        Returns:
            [4, 4] world-to-camera transformation
        """
        T_s2c = torch.inverse(self.T_cam_to_sonar)
        return sonar_pose_w2s @ T_s2c


# =============================================================================
# Sonar-Based Point Cloud Generation
# =============================================================================

def sonar_frame_to_points(camera, sonar_config, intensity_threshold=0.05, mask_top_rows=10,
                          scale_factor=1.0):
    """
    Generate 3D points from a single sonar frame via backward projection.
    
    For each valid sonar pixel (intensity > threshold):
    - Convert (col, row) -> (azimuth, range)
    - Assume elevation = 0 (center of beam)
    - Convert to 3D in sonar frame (metric)
    - Transform to world frame using camera pose (COLMAP scale)
    - Convert to COLMAP scale for downstream consistency
    
    Args:
        camera: Camera object with R, T, and original_image
        sonar_config: SonarConfig with FOV and range parameters
        intensity_threshold: Minimum intensity to consider valid (0-1)
        mask_top_rows: Skip top N rows (closest range, often artifacts)
        scale_factor: COLMAP->metric scale; points returned in COLMAP scale
        
    Returns:
        points: [N, 3] numpy array of 3D points in world coordinates (COLMAP scale)
        colors: [N, 3] numpy array of RGB colors (grayscale from intensity)
    """
    import numpy as np
    
    # Get sonar image
    image = camera.original_image  # [3, H, W] or [1, H, W]
    if image.shape[0] == 3:
        intensity = image[0]  # Take first channel
    else:
        intensity = image.squeeze(0)
    
    # Convert to numpy
    if hasattr(intensity, 'numpy'):
        intensity = intensity.numpy()
    
    H, W = intensity.shape
    
    # Create mask for valid pixels
    valid_mask = intensity > intensity_threshold
    
    # Mask out top rows (artifacts at close range)
    if mask_top_rows > 0:
        valid_mask[:mask_top_rows, :] = False
    
    # Get indices of valid pixels
    rows, cols = np.where(valid_mask)
    
    if len(rows) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    # Convert pixel coords to polar (azimuth, range)
    # Azimuth: center column = 0, left = positive, right = negative
    half_az_rad = math.radians(sonar_config.azimuth_fov / 2)
    azimuth = -(cols - W / 2) / (W / 2) * half_az_rad  # radians
    
    # Range: top row = range_min, bottom row = range_max (metric)
    range_vals_metric = sonar_config.range_min + (rows / H) * (sonar_config.range_max - sonar_config.range_min)
    
    # Convert to 3D in sonar/camera frame (metric)
    # Assuming elevation = 0 (center of beam)
    # Camera frame: +Z forward, +X right, +Y down
    # Azimuth convention: right side of image = negative azimuth, left = positive
    # So: x = -r * sin(az) to get positive x for right side (negative azimuth)
    x_cam = -range_vals_metric * np.sin(azimuth)  # lateral (flipped to match +X = right)
    y_cam = np.zeros_like(range_vals_metric)      # elevation = 0
    z_cam = range_vals_metric * np.cos(azimuth)   # forward (depth)
    
    points_cam_metric = np.stack([x_cam, y_cam, z_cam], axis=1)  # [N, 3] metric
    
    # Transform to world coordinates
    # Camera pose: R is world-to-camera rotation, T is world-to-camera translation
    # point_world = R^T @ (point_cam - T) ... wait, that's not right
    # Actually: point_cam = R @ point_world + T
    # So: point_world = R^T @ point_cam - R^T @ T = R^T @ (point_cam - T)
    # But T is not subtracted from point_cam, it's: point_world = R^T @ point_cam + camera_center
    # where camera_center = -R^T @ T
    
    R_w2c = camera.R  # [3, 3]
    T_w2c = camera.T  # [3]
    R_c2w = R_w2c.T
    camera_center_colmap = -R_c2w @ T_w2c
    camera_center_metric = camera_center_colmap * scale_factor
    
    # Transform points: point_world_metric = R_c2w @ point_cam_metric + camera_center_metric
    points_world_metric = (R_c2w @ points_cam_metric.T).T + camera_center_metric  # [N, 3] metric
    points_world = points_world_metric / scale_factor  # back to COLMAP scale
    
    # Get colors from intensity (grayscale -> RGB)
    intensities = intensity[rows, cols]
    colors = np.stack([intensities, intensities, intensities], axis=1)  # [N, 3]
    
    return points_world, colors


def sonar_frames_to_point_cloud(cameras, sonar_config, intensity_threshold=0.05, 
                                 mask_top_rows=10, max_points_per_frame=5000,
                                 voxel_downsample=None, scale_factor=1.0):
    """
    Generate combined 3D point cloud from multiple sonar frames.
    
    Args:
        cameras: List of camera objects with poses and images
        sonar_config: SonarConfig with FOV and range parameters
        intensity_threshold: Minimum intensity to consider valid (0-1)
        mask_top_rows: Skip top N rows (artifacts at close range)
        max_points_per_frame: Randomly sample if more points than this (None = no limit)
        voxel_downsample: Voxel size for downsampling final cloud (None = no downsample)
        
    Returns:
        points: [N, 3] numpy array of 3D points in world coordinates
        colors: [N, 3] numpy array of RGB colors
    """
    import numpy as np
    
    all_points = []
    all_colors = []
    
    for i, cam in enumerate(cameras):
        points, colors = sonar_frame_to_points(
            cam, sonar_config, 
            intensity_threshold=intensity_threshold,
            mask_top_rows=mask_top_rows,
            scale_factor=scale_factor
        )
        
        if len(points) == 0:
            continue
        
        # Optionally limit points per frame
        if max_points_per_frame is not None and len(points) > max_points_per_frame:
            indices = np.random.choice(len(points), max_points_per_frame, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        all_points.append(points)
        all_colors.append(colors)
    
    if len(all_points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    
    # Optionally voxel downsample
    if voxel_downsample is not None:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
        except ImportError:
            pass  # Skip downsampling if open3d not available
    
    return points, colors
