# Dataset Preparation Guide

*Written by opus4.5; this file was previously untracked in git.*

This guide explains how to prepare a dataset for sonar-based surfel reconstruction.

## Overview

The pipeline requires:
1. Camera trajectory from COLMAP (with timestamped filenames)
2. Raw sonar images (with matching timestamps)
3. Pose interpolation to align sonar frames with camera trajectory

## Raw Input Data

### 1. Camera Trajectory from COLMAP

Run COLMAP on your camera images to obtain poses:

```
camera_colmap_output/
└── sparse/0/
    ├── images.bin       # Camera extrinsics (poses)
    ├── cameras.bin      # Camera intrinsics
    └── points3D.bin     # 3D points (optional)
```

**Requirements:**
- Camera model must be `SIMPLE_PINHOLE` or `PINHOLE` (undistorted images only)
- Image filenames must follow the pattern: `camera_{TIMESTAMP}.png`

### 2. Raw Sonar Images

```
sonar_raw/
├── sonar_1765233408009.png
├── sonar_1765233408073.png
├── sonar_1765233408326.png
└── ...
```

**Requirements:**
- Filename pattern: `sonar_{TIMESTAMP}.png`
- Image size: **256 x 200 pixels**
- Format: PNG, grayscale (single-channel intensity)
- Coordinate system: Polar projection
  - Columns (0-255): Azimuth angle (left = +60°, right = -60°)
  - Rows (0-199): Range (top = 0.2m, bottom = 3.0m)

## Naming Convention (Critical)

Timestamps must be in **milliseconds** and synchronized between camera and sonar:

| Sensor | Filename Pattern | Example |
|--------|------------------|---------|
| Camera | `camera_{TIMESTAMP}.png` | `camera_1765233408026.png` |
| Sonar  | `sonar_{TIMESTAMP}.png`  | `sonar_1765233408009.png` |

The interpolation script matches sonar frames to camera frames by finding the nearest timestamps.

## Step 1: Run Pose Interpolation

The interpolation script generates sonar poses by interpolating from the camera trajectory:

```bash
python scripts/interpolate_sonar_poses.py \
    --camera_model /path/to/camera_colmap_output/sparse/0 \
    --sonar_images /path/to/sonar_raw \
    --output_dir /path/to/sonar_dataset/sparse/0 \
    --threshold_ms 100 \
    --max_frames 500 \
    --seed 42
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--camera_model` | Path to camera COLMAP model (sparse/0 directory) | Required |
| `--sonar_images` | Path to raw sonar images directory | Required |
| `--output_dir` | Output directory for sonar COLMAP model | Required |
| `--threshold_ms` | Maximum time gap to nearest camera pose (ms) | 100 |
| `--max_frames` | Random sample N frames (-1 for all) | -1 |
| `--seed` | Random seed for reproducible sampling | 42 |

### What the Script Does

1. Reads camera poses from `images.bin`
2. Extracts timestamps from camera filenames (`camera_TIMESTAMP.png`)
3. For each sonar frame:
   - Finds bracketing camera poses by timestamp
   - Interpolates rotation using SLERP (spherical linear interpolation)
   - Interpolates translation using LERP (linear interpolation)
   - Rejects frames where nearest camera pose exceeds threshold
4. Writes new `images.bin` with sonar filenames and interpolated poses
5. Copies `cameras.bin` and `points3D.bin` from camera model

## Step 2: Organize Final Dataset

After running the interpolation script, organize your dataset:

```
sonar_dataset/
├── sparse/0/
│   ├── images.bin      # Interpolated sonar poses (from script)
│   ├── cameras.bin     # Copied from camera model
│   └── points3D.bin    # Copied from camera model (optional)
└── sonar/
    ├── sonar_1765233408009.png
    ├── sonar_1765233408073.png
    └── ...
```

Copy or symlink the sonar images to the `sonar/` subdirectory:

```bash
# Option 1: Copy
cp /path/to/sonar_raw/*.png /path/to/sonar_dataset/sonar/

# Option 2: Symlink (saves disk space)
ln -s /path/to/sonar_raw/*.png /path/to/sonar_dataset/sonar/
```

## Step 3: Run Training

```bash
python debug_multiframe.py
```

Or with the standard training script:

```bash
python train.py -s /path/to/sonar_dataset --sonar_mode --sonar_images sonar
```

## Sonar Configuration

The default sonar parameters (in `utils/sonar_utils.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image width | 256 | Azimuth resolution |
| Image height | 200 | Range resolution |
| Azimuth FOV | 120° | Horizontal field of view (±60°) |
| Elevation FOV | 20° | Vertical field of view (±10°) |
| Range min | 0.2 m | Minimum range |
| Range max | 3.0 m | Maximum range |

## Sonar Extrinsic Offset

The sonar is mounted offset from the camera:

| Axis | Offset | Description |
|------|--------|-------------|
| X | 0 cm | No lateral offset |
| Y | -10 cm | 10 cm above camera (+Y is down) |
| Z | -8 cm | 8 cm behind camera (+Z is forward) |
| Pitch | +5° | Pitched 5° downward |

This offset is applied automatically during training via `SonarExtrinsic`.

## Troubleshooting

### "No valid sonar frames" error
- Check that timestamps overlap between camera and sonar
- Increase `--threshold_ms` if camera framerate is low

### Poor reconstruction quality
- Ensure sonar images are undistorted
- Verify timestamp synchronization between sensors
- Check that camera poses from COLMAP are accurate

### Scale mismatch
- The `SonarScaleFactor` learns to align COLMAP's arbitrary scale with metric sonar ranges
- If scale doesn't converge, check that range values in sonar images are correct

## Example: Full Pipeline

```bash
# 1. Run COLMAP on camera images (external step)
colmap automatic_reconstructor \
    --workspace_path /data/camera_colmap \
    --image_path /data/camera_images

# 2. Interpolate sonar poses
python scripts/interpolate_sonar_poses.py \
    --camera_model /data/camera_colmap/sparse/0 \
    --sonar_images /data/sonar_raw \
    --output_dir /data/sonar_dataset/sparse/0 \
    --threshold_ms 100 \
    --max_frames 500

# 3. Link sonar images
mkdir -p /data/sonar_dataset/sonar
ln -s /data/sonar_raw/*.png /data/sonar_dataset/sonar/

# 4. Train
cd /path/to/Unbiased_Surfel_sonar
python debug_multiframe.py  # Edit DATA_PATH in script first
```
