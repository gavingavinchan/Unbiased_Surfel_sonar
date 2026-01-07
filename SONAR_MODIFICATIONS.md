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

### 2. Sonar Mode Toggle (Implemented)

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

### Sonar Intrinsics (TODO)

**Current State**: Using camera intrinsics for sonar frames

**Why This Is Wrong**:
- Sonar has different field of view, resolution (256x200), and imaging geometry
- Acoustic imaging != optical imaging (range-azimuth vs pinhole projection)

**Why We're Keeping It For Now**:
- Want to verify pipeline works end-to-end first
- Intrinsics can be corrected once basic functionality confirmed
- Incremental approach reduces debugging complexity

**Future Fix**: Define proper sonar intrinsics or create sonar-specific camera model

---

### Camera-to-Sonar Extrinsic Transform (TODO)

**Current State**: Sonar poses are directly interpolated from camera poses (assumes co-located)

**Why This Is Wrong**:
- Sonar is mounted **10cm above** the camera
- Sonar is **pitched down 5 degrees** relative to camera

**Extrinsic Transform (camera → sonar)**:
```
Translation: (0, -0.1, 0) in camera frame  # 10cm up (Y-down convention)
Rotation: 5° pitch down around X-axis
```

**Future Fix**: Apply this transform in `interpolate_sonar_poses.py` after interpolating camera poses:
```python
# After interpolating camera pose (R_cam, t_cam):
T_cam_to_sonar = compute_extrinsic(translation=[0, -0.1, 0], pitch_deg=-5)
T_sonar = T_cam @ T_cam_to_sonar
```

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
| 2026-01-06 | Added --max_frames option to interpolation script for random sampling (laptop-friendly) | - |
| 2026-01-06 | Implemented pose interpolation script with SLERP and ±100ms filtering | - |
| 2026-01-06 | Added --sonar_mode and --sonar_images CLI flags | - |
| 2026-01-06 | Updated dataset reader to support sonar image folder | - |
| 2026-01-06 | Initial planning and documentation | - |

