# Sonar Support Modifications Log

This document tracks all modifications made to Unbiased_Surfel to support sonar data processing.

---

## Overview

**Goal**: Enable 2D Gaussian Splatting (2DGS) training on sonar imagery by interpolating poses from a synchronized camera trajectory.

**Dataset Context**:
- BlueROV underwater platform with synchronized camera and sonar
- Camera frames processed through COLMAP with accurate 6-DoF poses
- Sonar frames need poses derived from camera trajectory via interpolation

---

## Modifications

### 1. Pose Interpolation Script (Implemented)

**File**: `scripts/interpolate_sonar_poses.py`

**What it does**:
- Reads camera poses from COLMAP `images.bin`
- Interpolates poses for sonar frames based on timestamp proximity
- Uses linear interpolation for translation, SLERP for rotation quaternions
- Outputs new COLMAP-compatible `images.bin` for sonar frames

**Reasoning**:
- Sonar and camera have different capture rates and timestamps
- Camera trajectory provides ground-truth poses from COLMAP SfM
- Interpolation leverages rigid mounting assumption (camera-sonar extrinsics fixed)

**Key Design Decisions**:

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Interpolation method | Linear + SLERP | Linear is sufficient for translation; SLERP properly handles quaternion interpolation on SO(3) manifold |
| Time threshold | ±100ms | Sonar runs at ~15.6 Hz (~64ms intervals). 100ms ensures we have nearby camera poses for reliable interpolation |
| Frames outside threshold | Discarded | Extrapolation is unreliable; better to use fewer high-quality poses than many uncertain ones |

**Filtering Statistics** (for session_2025-12-08_16-35-13):
- Total sonar frames: 4,444
- Valid frames (within ±100ms): 2,595 (58%)
- Rejected frames: 1,849 (42%) - mostly from 90-second gap at recording start

---

### 2. Sonar Forward and Backward Projection (Implemented)

**Files**: `gaussian_renderer/__init__.py`, `utils/point_utils.py`, `utils/sonar_utils.py`

**What it does**:
- **Forward projection**: Renders 3D surfels to sonar image using polar geometry
  - Transforms surfel positions to sonar frame
  - Converts to polar coordinates (azimuth, elevation, range)
  - Maps to pixel coordinates (azimuth → column, range → row)
  - Applies Lambertian intensity model based on surfel normal vs sonar direction
  
- **Backward projection**: Converts sonar range image to 3D world points
  - Converts pixel coordinates to polar (azimuth, range)
  - Converts polar to Cartesian in sonar frame
  - Transforms to world coordinates using (scaled) pose

**Sonar Image Coordinate Convention** (Sonoptix Echo):
```
         ← positive azimuth    negative azimuth →
              +60°      0°      -60°
               |        |        |
    row 0   ───┬────────┬────────┬───  range_min (closest)
               │        │        │
               │   SONAR IMAGE   │
               │   256 x 200     │
               │        │        │
    row 199 ───┴────────┴────────┴───  range_max (farthest)
              col 0   col 128  col 255

Convention: +X direction (forward) = negative azimuth
            +Y direction (right)  = maps to right side of image (high col)
```

**Key Design Decisions**:

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Elevation handling | Sum all surfels within 20° arc | Sonar integrates acoustic returns over elevation beam spread |
| Occlusion model | Block only along exact same ray | Different elevations both contribute due to beam spread |
| Intensity model | Lambertian: I = max(0, n·d) | Simple approximation; surfels facing sonar are brighter |
| Implementation | Splatting (surfel → pixel) | Consistent with 2DGS; maintains differentiability |

---

### 3. Learnable Scale Factor (Implemented)

**Files**: `utils/sonar_utils.py`, `train.py`

**What it does**:
- Learns a single scale factor `s` to align COLMAP's arbitrary-scale poses with sonar's metric range
- Applied to pose translations: `scaled_translation = s * colmap_translation`
- Optimized jointly with surfels using Adam optimizer

**Reasoning**:
- COLMAP SfM produces poses with arbitrary scale (classic monocular ambiguity)
- Sonar measures physical range in meters
- A single global scale factor is sufficient to align them

**Implementation Details**:
- Uses log-parameterization internally: `scale = exp(log_scale)`
- This ensures scale stays positive and has stable gradients
- Logged to TensorBoard: `sonar/scale_factor` and `sonar/scale_factor_grad`

---

### 4. Sonar Mode Toggle (Implemented)

**Files**: `arguments/__init__.py`, `scene/dataset_readers.py`, `scene/__init__.py`

**What it does**:
- Adds `--sonar_mode` flag to enable sonar-specific behavior
- Adds `--sonar_images` parameter to specify sonar image folder (default: "sonar")

**Reasoning**:
- Preserves original codebase functionality for standard camera datasets
- Easy toggle between camera and sonar modes without code changes
- Allows same codebase to handle both modalities

---

## Known Limitations / Future Work

### Sonar Intrinsics (IMPLEMENTED)

**Current State**: ✅ Fully implemented with proper sonar polar geometry

**Implementation**:
- Sonar projection uses polar coordinates (azimuth, range) instead of pinhole (x, y, depth)
- `render_sonar()` function in `gaussian_renderer/__init__.py` handles forward projection
- `sonar_ranges_to_points()` in `utils/point_utils.py` handles backward projection
- Sonar parameters configurable via CLI flags:
  - `--sonar_azimuth_fov`: Horizontal FOV in degrees (default: 120°)
  - `--sonar_elevation_fov`: Elevation beam spread (default: 20°)
  - `--sonar_range_min`: Minimum range in meters (default: 0.2m)
  - `--sonar_range_max`: Maximum range in meters (default: 3.0m)

---

### Camera-to-Sonar Extrinsic Transform (IMPLEMENTED)

**Current State**: ✅ Fully implemented via `SonarExtrinsic` class

**Implementation**:
- Sonar is mounted **10cm above** the camera
- Sonar is **pitched down 5 degrees** relative to camera
- `SonarExtrinsic` class in `utils/sonar_utils.py` handles the transformation
- Transform automatically applied during rendering when `--sonar_mode` is enabled

**Extrinsic Transform (camera → sonar)**:
```
Translation: (0, -0.1, 0) in camera frame  # 10cm up (Y-down convention)
Rotation: 5° pitch down around X-axis
```

---

### Learnable Scale Factor (IMPLEMENTED)

**Problem**: COLMAP produces poses with arbitrary scale (structure-from-motion ambiguity). Sonar measures ranges in real meters.

**Solution**: `SonarScaleFactor` class learns a single scalar during training:
- Applied to pose translations: `scaled_t = scale * colmap_t`
- Uses log-parameterization for numerical stability and positivity
- Logged to TensorBoard for monitoring convergence

**Configuration**:
- `--sonar_scale_init`: Initial scale factor (default: 1.0)
- `--sonar_scale_lr`: Learning rate for scale factor (default: 0.01)

---

### 4. Top Row Masking (Implemented)

**Files Modified**:
- `gaussian_renderer/__init__.py` - Masks rendered output
- `train.py` - Masks ground truth before loss computation

**What it does**:
- Zeros out the top 10 rows (closest range bins) of both rendered and ground truth images
- Applied consistently in training and rendering

**Reasoning**:
- The closest range bins in sonar images often contain:
  - Direct path reflections/artifacts from the sonar housing
  - Saturation from nearby strong reflectors
  - Invalid data at minimum range (before pulse settles)
- These artifacts don't represent actual scene geometry
- Masking prevents the model from trying to fit these artifacts

**Configuration**:
- Currently hardcoded to 10 rows
- TODO: Make configurable via `--sonar_mask_top_rows` parameter

---

## Dataset Details

**Sonar Dataset Location**: `~/ros2_ws/outputs/session_2025-12-08_16-35-13_sonar_data_for_2dgs/`

```
session_2025-12-08_16-35-13_sonar_data_for_2dgs/
├── sonar/                    # 4,444 sonar images (256x200 px)
│   ├── sonar_<timestamp>.png
│   └── ...
└── sparse/0/                 # COLMAP model (from camera)
    ├── cameras.bin           # Camera intrinsics
    ├── images.bin            # Camera poses (6,206 frames)
    └── points3D.bin          # 3D point cloud (243,562 points)
```

**Timestamp Format**: Unix epoch milliseconds (e.g., `sonar_1765233318418.png`)

**Frame Rates**:
- Camera: Variable (from COLMAP reconstruction)
- Sonar: ~15.6 Hz (64ms average interval)

**Timestamp Ranges**:
- Sonar: 1765233318418 - 1765233602561
- Camera: 1765233408026 - 1765233638119
- Note: Sonar starts ~90 seconds before camera data begins

---

---

## Usage

### Step 1: Generate Interpolated Sonar Poses

Run the interpolation script to create a sonar-specific COLMAP model:

```bash
python scripts/interpolate_sonar_poses.py \
    --camera_model /path/to/camera/sparse/0 \
    --sonar_images /path/to/sonar/images \
    --output_dir /path/to/output/sparse_sonar/0 \
    --threshold_ms 100 \
    --max_frames 500 \
    --seed 42
```

Options:
- `--threshold_ms 100` - Discard sonar frames without camera poses within ±100ms
- `--max_frames 500` - Randomly sample 500 frames (useful for laptops with limited VRAM)
- `--seed 42` - Random seed for reproducible sampling

This creates a new `sparse_sonar/0/` directory with:
- `images.bin` - Interpolated sonar poses
- `cameras.bin` - Copied from camera model (intrinsics)
- `points3D.bin` - Copied from camera model (3D points)

### Step 2: Train in Sonar Mode

```bash
python train.py \
    -s /path/to/sonar/dataset \
    --sonar_mode \
    --sonar_images sonar \
    -m ./output/sonar_model \
    --data_device cpu \
    --resolution 2
```

Note: The dataset should have:
- `sparse/0/` containing the interpolated sonar poses (copy from `sparse_sonar/0/`)
- `sonar/` containing the sonar images

---

## Changelog

| Date | Modification | Author |
|------|--------------|--------|
| 2026-01-09 | Added top 10 row masking for sonar images (closest range bins often have artifacts) | - |
| 2026-01-09 | Implemented full sonar projection pipeline: forward (surfel→pixel) and backward (pixel→3D) with polar geometry | - |
| 2026-01-09 | Added learnable scale factor (SonarScaleFactor) to align COLMAP arbitrary scale with sonar metric range | - |
| 2026-01-09 | Added camera-to-sonar extrinsic transform (SonarExtrinsic): 10cm offset, 5deg pitch down | - |
| 2026-01-09 | Added sonar calibration parameters to arguments/__init__.py (Sonoptix Echo: 256x200, 120° HFOV, 20° elev) | - |
| 2026-01-09 | Added TensorBoard logging for scale factor convergence monitoring | - |
| 2026-01-09 | Created utils/sonar_utils.py with SonarScaleFactor, SonarConfig, SonarExtrinsic classes | - |
| 2026-01-09 | Added render_sonar() function to gaussian_renderer for polar coordinate splatting | - |
| 2026-01-09 | Added sonar_ranges_to_points() and sonar_points_to_normals() to utils/point_utils.py | - |
| 2026-01-08 | Added mathematical derivation of ray-surfel intersection to architecture_diagram.md | - |
| 2026-01-06 | Added --max_frames option to interpolation script for random sampling (laptop-friendly) | - |
| 2026-01-06 | Implemented pose interpolation script with SLERP and ±100ms filtering | - |
| 2026-01-06 | Added --sonar_mode and --sonar_images CLI flags | - |
| 2026-01-06 | Updated dataset reader to support sonar image folder | - |
| 2026-01-06 | Initial planning and documentation | - |

