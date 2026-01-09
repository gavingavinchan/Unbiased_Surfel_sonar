#!/usr/bin/env python3
"""
Debug script: Train on 2 specific frames for 1 iteration and save rendered output.
"""

import os
import sys
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# Fixed seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# The 2 selected frames (from seed=42 selection) - without .png extension
SELECTED_FRAMES = [
    'sonar_1765233540734',
    'sonar_1765233425925',
]

print(f"=" * 60)
print(f"DEBUG: Training on 2 frames with seed={SEED}")
print(f"Selected frames:")
for f in SELECTED_FRAMES:
    print(f"  {f}")
print(f"=" * 60)

# Import after seeding
from scene import Scene, GaussianModel
from gaussian_renderer import render
from gaussian_renderer import render_sonar
from utils.sonar_utils import SonarScaleFactor, SonarConfig, SonarExtrinsic
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import Namespace

# Setup paths
source_path = os.path.expanduser('~/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs')
output_path = './output/debug_2frames'
os.makedirs(output_path, exist_ok=True)

# Create minimal args
dataset_args = Namespace(
    source_path=source_path,
    model_path=output_path,
    images='images',
    resolution=2,
    white_background=False,
    data_device='cpu',
    eval=False,
    sh_degree=3,
    sonar_mode=True,
    sonar_images='sonar',
    # Sonar params
    sonar_azimuth_fov=120.0,
    sonar_elevation_fov=20.0,
    sonar_range_min=0.1,
    sonar_range_max=30.0,
    sonar_intensity_threshold=0.01,
    sonar_scale_init=1.0,
    sonar_scale_lr=0.01,
    gamma=1.0,
)

pipe_args = Namespace(
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False,
)

# Override the scene loader to only use our 2 frames
from scene.dataset_readers import sceneLoadTypeCallbacks, readColmapSceneInfo

original_callback = sceneLoadTypeCallbacks["Colmap"]

def filtered_callback(path, images, eval, llffhold=8, sonar_mode=False, sonar_images='sonar'):
    """Wrapper that filters to only selected frames."""
    scene_info = original_callback(path, images, eval, llffhold, sonar_mode, sonar_images)
    
    # Filter train cameras to only our selected frames
    original_train = scene_info.train_cameras
    filtered_train = [cam for cam in original_train if cam.image_name in SELECTED_FRAMES]
    
    print(f"\nFiltered from {len(original_train)} to {len(filtered_train)} training cameras")
    for cam in filtered_train:
        print(f"  Using: {cam.image_name}")
    
    # Create new scene_info with filtered cameras
    from scene.dataset_readers import SceneInfo
    return SceneInfo(
        point_cloud=scene_info.point_cloud,
        train_cameras=filtered_train,
        test_cameras=[],  # No test cameras for debug
        nerf_normalization=scene_info.nerf_normalization,
        ply_path=scene_info.ply_path,
    )

sceneLoadTypeCallbacks["Colmap"] = filtered_callback

# Load scene
print("\nLoading scene...")
gaussians = GaussianModel(dataset_args.sh_degree)
scene = Scene(dataset_args, gaussians)
print(f"Loaded {len(scene.getTrainCameras())} training cameras")
print(f"Gaussian count: {gaussians.get_xyz.shape[0]}")

# Setup sonar rendering
sonar_config = SonarConfig(
    azimuth_fov=dataset_args.sonar_azimuth_fov,
    elevation_fov=dataset_args.sonar_elevation_fov,
    range_min=dataset_args.sonar_range_min,
    range_max=dataset_args.sonar_range_max,
    image_width=scene.getTrainCameras()[0].image_width,
    image_height=scene.getTrainCameras()[0].image_height,
)
sonar_scale_factor = SonarScaleFactor(init_value=dataset_args.sonar_scale_init).cuda()
sonar_extrinsic = SonarExtrinsic(device="cuda")

print(f"\nSonar config:")
print(f"  Image size: {sonar_config.image_width}x{sonar_config.image_height}")
print(f"  Azimuth FOV: {sonar_config.azimuth_fov}°")
print(f"  Range: {sonar_config.range_min}m - {sonar_config.range_max}m")
print(f"  Initial scale: {sonar_scale_factor.get_scale_value()}")

# Render each frame
bg_color = torch.zeros(3, device='cuda')

print(f"\n" + "=" * 60)
print("RENDERING (1 iteration, no optimization)")
print("=" * 60)

for i, viewpoint in enumerate(scene.getTrainCameras()):
    print(f"\nFrame {i+1}: {viewpoint.image_name}")
    print(f"  Image size: {viewpoint.image_width}x{viewpoint.image_height}")
    
    # Get ground truth and apply mask
    gt_image = viewpoint.original_image.cuda().clone()
    
    # Mask top 10 rows (closest range bins often have artifacts)
    mask_top_rows = 10
    gt_image[:, :mask_top_rows, :] = 0
    
    print(f"  GT image shape: {gt_image.shape}")
    print(f"  GT image range: [{gt_image.min():.4f}, {gt_image.max():.4f}] (after masking top {mask_top_rows} rows)")
    
    # Render with sonar projection
    render_pkg = render_sonar(viewpoint, gaussians, bg_color, sonar_config, sonar_scale_factor, sonar_extrinsic)
    rendered = render_pkg["render"]
    
    print(f"  Rendered shape: {rendered.shape}")
    print(f"  Rendered range: [{rendered.min():.4f}, {rendered.max():.4f}]")
    
    # Check how many surfels are in FOV
    if "visibility_filter" in render_pkg:
        visible = render_pkg["visibility_filter"].sum().item()
        print(f"  Visible surfels: {visible}")
    
    # Save images (take first channel since it's grayscale replicated to RGB)
    gt_np = (gt_image[0].cpu().numpy() * 255).astype(np.uint8)
    rendered_np = (rendered[0].cpu().detach().numpy() * 255).astype(np.uint8)
    
    gt_pil = Image.fromarray(gt_np, mode='L')
    rendered_pil = Image.fromarray(rendered_np, mode='L')
    
    gt_path = os.path.join(output_path, f'gt_{i}_{viewpoint.image_name}.png')
    rendered_path = os.path.join(output_path, f'rendered_{i}_{viewpoint.image_name}.png')
    
    gt_pil.save(gt_path)
    rendered_pil.save(rendered_path)
    
    print(f"  Saved: {gt_path}")
    print(f"  Saved: {rendered_path}")

print(f"\n" + "=" * 60)
print(f"DEBUG COMPLETE")
print(f"Output saved to: {output_path}")
print(f"=" * 60)
