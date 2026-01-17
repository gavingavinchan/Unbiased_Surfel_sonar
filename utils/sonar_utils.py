#
# Sonar utilities for 2D Gaussian Splatting
# Implements sonar-specific projection and scale factor learning
#

import torch
import torch.nn as nn
import math


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
    Get the transformation matrix from camera frame to sonar frame.
    
    The sonar is mounted:
    - 10cm above the camera (translation in -Y direction in camera frame)
    - Pitched down 5 degrees (rotation around X-axis)
    
    Camera frame convention (OpenCV/COLMAP):
    - +X = right
    - +Y = down
    - +Z = forward (optical axis)
    
    Sonar frame convention:
    - +X = forward (boresight)
    - +Y = right
    - +Z = down
    
    Returns:
        T_cam_to_sonar: 4x4 homogeneous transformation matrix
    """
    # Translation: sonar is 10cm above camera (in camera frame: -Y direction)
    # In camera frame coordinates: [0, -0.1, 0] (up is -Y)
    translation = torch.tensor([0.0, -0.1, 0.0], device=device)
    
    # Rotation: sonar is pitched down 5 degrees relative to camera
    # Pitch down = rotation around X-axis by +5 degrees (in camera frame)
    pitch_rad = math.radians(5.0)
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    
    # Rotation matrix for pitch (around X-axis)
    R_pitch = torch.tensor([
        [1.0,    0.0,     0.0],
        [0.0,  cos_p, -sin_p],
        [0.0,  sin_p,  cos_p]
    ], device=device)
    
    # Also need to transform from camera convention to sonar convention
    # Camera: +Z forward, +X right, +Y down
    # Sonar:  +X forward, +Y right, +Z down
    # This is a permutation: sonar_X = cam_Z, sonar_Y = cam_X, sonar_Z = cam_Y
    R_convention = torch.tensor([
        [0.0, 0.0, 1.0],  # sonar_X = cam_Z
        [1.0, 0.0, 0.0],  # sonar_Y = cam_X
        [0.0, 1.0, 0.0]   # sonar_Z = cam_Y
    ], device=device)
    
    # Combined rotation: first apply pitch, then convention change
    R_cam_to_sonar = R_convention @ R_pitch
    
    # Build 4x4 homogeneous transform
    T_cam_to_sonar = torch.eye(4, device=device)
    T_cam_to_sonar[:3, :3] = R_cam_to_sonar
    T_cam_to_sonar[:3, 3] = R_convention @ translation  # Transform translation to sonar frame
    
    return T_cam_to_sonar


def apply_camera_to_sonar_extrinsic(camera_pose_w2c, device="cuda"):
    """
    Transform a camera pose (world-to-camera) to sonar pose (world-to-sonar).
    
    Given the world-to-camera transform T_w2c, compute world-to-sonar as:
        T_w2s = T_c2s @ T_w2c
    
    Args:
        camera_pose_w2c: 4x4 world-to-camera transformation matrix
        device: Device for computation
        
    Returns:
        sonar_pose_w2s: 4x4 world-to-sonar transformation matrix
    """
    T_c2s = get_camera_to_sonar_transform(device)
    T_w2s = T_c2s @ camera_pose_w2c
    return T_w2s


class SonarExtrinsic(nn.Module):
    """
    Camera-to-sonar extrinsic calibration.
    
    The sonar is mounted 10cm above the camera and pitched down 5 degrees.
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
        return self.T_cam_to_sonar @ camera_pose_w2c
    
    def inverse_transform(self, sonar_pose_w2s):
        """
        Transform world-to-sonar pose back to world-to-camera pose.
        
        Args:
            sonar_pose_w2s: [4, 4] world-to-sonar transformation
            
        Returns:
            [4, 4] world-to-camera transformation
        """
        T_s2c = torch.inverse(self.T_cam_to_sonar)
        return T_s2c @ sonar_pose_w2s


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
