# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is loaded at the start of every session; keep it concise to avoid bloating context length.

## Project Overview

This is **Unbiased Surfel** extended with **sonar imaging support** for underwater 3D reconstruction. The base project implements 2D Gaussian Splatting (surfels) with unbiased depth estimation for high-accuracy mesh extraction. The sonar extension adapts this to polar coordinate geometry for BlueROV underwater platforms.

**Current Focus**: Development is centered on `debug_multiframe.py` (multi-frame sonar training) and `scripts/poisson_tuner_gui.py` (mesh tuning GUI). Other modules are not a priority unless they affect these.

## Development Commands

### Environment Setup
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar
```

### Training

```bash
# Standard camera training
python train.py -s <path_to_dataset>

# Sonar mode training
python train.py -s <path_to_dataset> --sonar_mode --sonar_images sonar

# Key sonar hyperparameters
python train.py -s <path> --sonar_mode \
    --sonar_azimuth_fov 120.0 \
    --sonar_elevation_fov 20.0 \
    --sonar_range_min 0.2 \
    --sonar_range_max 3.0 \
    --sonar_scale_init 1.0 \
    --sonar_scale_lr 0.01
```

### Multi-Frame Debug Training (Recommended for Sonar)

The primary sonar training script with curriculum learning:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && SONAR_DATASET=r2 python debug_multiframe.py 2>&1
```

Configure via environment variables:
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_OUTPUT_DIR=./output/my_run \
SONAR_DATASET=r2 \
SONAR_STAGE2_ITERS=2000 \
python debug_multiframe.py 2>&1
```

### GUI Poisson Mesh Tuner

Interactive GUI for tuning Poisson reconstruction parameters:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && python scripts/poisson_tuner_gui.py
```

Features:
- Real-time mesh visualization with Open3D
- Tunable parameters: depth, density quantile, opacity/scale percentiles
- Can launch debug_multiframe.py with configured parameters
- Live log viewer and mesh loading

### Mesh Extraction
```bash
python render.py -m <trained_model_path> -s <dataset_path>
```

## Architecture

### Core Components

**GaussianModel** (`scene/gaussian_model.py`): Manages surfel parameters
- `_xyz`: positions [N,3]
- `_rotation`: quaternions [N,4] - normal extracted via `quaternion_to_normal()`
- `_scaling`, `_opacity`, `_features_dc/rest`: appearance properties

**Scene** (`scene/__init__.py`): Dataset loading and camera management
- Loads COLMAP or Blender format
- Sonar mode: reads from `sonar/` folder with `--sonar_mode`

**Rendering** (`gaussian_renderer/__init__.py`):
- `render()`: Standard pinhole camera projection
- `render_sonar()`: Polar coordinate projection for sonar
- `compute_fov_margin()`: Size-aware FOV boundary checking
- `quaternion_to_normal()`: Extract surfel normal from quaternion

### Sonar-Specific Components

#### `utils/sonar_utils.py`

**SonarConfig**: Calibration parameters
- 256x200 image, 120° azimuth FOV, 20° elevation FOV
- Range: 0.2m to 3.0m
- Precomputes azimuth/range grids

**SonarScaleFactor**: Learnable scale for pose alignment
- Log-parameterized for stability
- Aligns arbitrary-scale COLMAP poses with metric sonar ranges
- `get_log_scale_grad()` for monitoring convergence

**SonarExtrinsic**: Camera-to-sonar transform
- Camera-frame translation = `[0.0, -0.10, -0.08]` m (8cm back, 10cm up in camera Y-down convention), pitch = +5° down about X-axis
- `get_camera_to_sonar_transform()`, `apply_camera_to_sonar_extrinsic()`

**Projection Functions**:
- `sonar_frame_to_points()`: Backward projection from single sonar frame
- `sonar_frames_to_point_cloud()`: Combined point cloud from multiple frames
- `build_sonar_config()`: Factory function from arguments

#### `utils/point_utils.py`

- `depth_to_normal()`: Enhanced with `sonar_mode` parameter
- `sonar_ranges_to_points()`: Converts sonar range image to 3D world points
- `sonar_points_to_normals()`: Surface normals from sonar 3D points using finite differences

### Data Flow

**Forward Projection** (render_sonar):
```
3D surfels → scale factor → world-to-sonar transform → polar coords (azimuth, range) → pixel coords → bilinear splat
```

**Backward Projection** (sonar_frame_to_points):
```
pixel (col, row) → polar (azimuth, range) → 3D camera frame → world coords
```

**Coordinate Conventions**:
- Camera frame: +X right, +Y down, +Z forward
- Sonar image: columns = azimuth (left=+, right=-), rows = range (top=near, bottom=far)
- Azimuth: `-(col - W/2) / (W/2) * half_fov_rad`

### Training Loop

#### Standard Training (train.py)

1. Random camera selection
2. Render via `render()` or `render_sonar()`
3. Loss computation: photometric (SSIM+L1) + normal consistency + depth regularization
4. Sonar mode: disables normal/distance loss, masks top 10 rows
5. Separate optimizer step for sonar scale factor
6. TensorBoard logging: scale factor, gradients, rendered mean range
7. Densification/pruning of surfels
8. Save checkpoints at iterations 7000, 30000

#### Curriculum Learning (debug_multiframe.py)

Three-stage training for stable scale factor convergence:

1. **Stage 1**: Fix surfels, learn scale factor only
2. **Stage 2**: Fix scale factor, learn surfels
3. **Stage 3**: Optional joint fine-tuning

Additional features:
- **Bright-pixel loss**: Weighted loss on top-k brightest pixels (configurable percentile)
- **FOV-aware pruning**: Removes surfels outside all training cameras' FOVs
- **Automatic Poisson mesh generation** at checkpoints
- **Real-time visualization**: Loss plots, raw frame comparisons

## Key Implementation Notes

### Size-Aware FOV Constraints

Implemented in `debug_multiframe.py` (not in `render_sonar` itself). Surfels are checked with their radius extent, not just center point:
```python
fov_margin = compute_fov_margin(means3D, sonar_config, ...)
in_fov = (fov_margin > surfel_radius)  # ensures entire surfel is visible
```
`render_sonar` provides `compute_fov_margin()` for distance-to-boundary calculation; size-aware pruning is done at the debug script level. Prevents boundary artifacts where large surfels partly outside FOV receive invalid gradients.

### Gradient Flow in render_sonar
- Uses `scatter_add_` for differentiable accumulation
- Top-row masking uses multiplicative mask (not in-place assignment) to preserve gradients
- If `in_fov.any()` is False (no visible points), output has no gradients

### Backward Projection Fix
The x-coordinate requires sign flip to match camera conventions:
```python
x_cam = -range_vals * np.sin(azimuth)  # NOT positive
```

### Multi-View Training
Points initialized from one sonar frame are only visible from that viewpoint. For single-frame debugging, train on the same viewpoint used for initialization.

## Environment Variables

Used by `debug_multiframe.py` and the GUI. Defaults change often; check the script for current values.

| Variable | Description |
|----------|-------------|
| `SONAR_OUTPUT_DIR` | Output directory override |
| `SONAR_DATASET` | Dataset key (e.g., `r2`, `legacy`) |
| `SONAR_STAGE2_ITERS` | Iterations for stage 2 |
| `POISSON_DEPTH` | Octree depth for Poisson reconstruction |
| `POISSON_DENSITY_QUANTILE` | Density cutoff for mesh filtering |
| `POISSON_MIN_OPACITY` | Minimum opacity filter |
| `POISSON_OPACITY_PERCENTILE` | Opacity percentile threshold |
| `POISSON_SCALE_PERCENTILE` | Scale percentile threshold |
| `BRIGHT_PERCENTILE` | Bright pixel threshold for loss |
| `BRIGHT_WEIGHT` | Weight for bright-pixel loss |
| `BRIGHT_MIN_PIXELS` | Minimum bright pixels for loss |

## Known Issues & Workarounds

| Issue | Status | Notes |
|-------|--------|-------|
| Scale factor convergence | Workaround | Should converge to ~0.66 but tends to 1.0. Currently frozen at 0.65 via curriculum learning |
| Top row artifacts | Mitigated | Masking top 10 rows in render and losses |
| Mesh extends beyond FOV | Open | Mesh extraction uses pinhole cameras; much worse for R2 dataset, legacy is fine-ish |
| Scale-surfel coupling | Mitigated | Curriculum learning (scale-first, then surfels) prevents degeneracy |

## File References

### Core
- Main training: `train.py`
- Multi-frame debug training: `debug_multiframe.py`
- Sonar renderer: `gaussian_renderer/__init__.py` (`render_sonar`, `compute_fov_margin`)
- Sonar utilities: `utils/sonar_utils.py`
- Point projections: `utils/point_utils.py`
- Mesh extraction: `utils/mesh_utils.py`

### Scripts
- GUI Poisson tuner: `scripts/poisson_tuner_gui.py`
- Pose interpolation: `scripts/interpolate_sonar_poses.py`

### Scene & Data
- Scene loading: `scene/__init__.py`
- Dataset readers: `scene/dataset_readers.py`
- Model parameters: `arguments/__init__.py`

### Documentation
- Planning & progress: `plans/` (PLAN_* files, progress_overview.md, scientific_progress.md)
- Plans guide: `plans/README.md` (plan format and naming)
- Snapshots: `snapshots/` (SNAPSHOT_* files; current-state notes meant to reduce repeated forensics)
- Snapshots guide: `snapshots/README.md` (snapshot format and naming)
- Reference docs: `docs/` (DATASET_PREPARATION.md, R2_DATASET_ISSUES.md, unbiased_surfel_installation_notes.md)
- Archived docs: `docs/old/` (superseded by plans/)

## Documentation Workflow

### Git Commit Format

Include the LLM name in the commit title to track which model was used:

```
<description> (opus4.5)
```

Examples:
- `Add size-aware FOV constraints for surfels (opus4.5)`
- `Fix scale factor gradient flow (sonnet3.5)`

### Progress Documents (Update Before Git Commits)

**`plans/progress_overview.md`**: High-level project state
- Architecture diagrams (mermaid flowcharts)
- Branch summaries and diff stats
- Approach ledger (what was tried, effort, outcome)
- Effort heatmap by subsystem
- Decision timeline and stuck points

**`plans/scientific_progress.md`**: Mathematical/algorithmic documentation
- Equations with LaTeX notation
- Algorithm pseudocode
- Parameter values and formulas
- Limitations and open issues

Both should be updated before git commits to keep project history coherent.
Plans are updated as decisions evolve, then committed with the implementation.
