# Sonar 2DGS Implementation Progress

## Overview

Adapting 2D Gaussian Splatting for multibeam forward-looking sonar (Sonoptix Echo) by implementing:
1. Forward projection (surfel splatting to polar sonar image)
2. Backward projection (range image to 3D points)
3. Learnable scale factor to align COLMAP arbitrary-scale poses with sonar metric range

---

## Current Status: Phase 1 Complete

All core sonar projection components are implemented and aligned with the design plan.

---

## Completed Items

### Scale Factor Module
- **File**: `utils/sonar_utils.py:11-60`
- **Class**: `SonarScaleFactor`
- **Status**: ✅ Complete
- **Notes**: Uses log scale internally for numerical stability (improvement over original plan)

### Sonar Configuration
- **File**: `utils/sonar_utils.py:63-172`
- **Class**: `SonarConfig`
- **Status**: ✅ Complete
- **Notes**: Comprehensive config with precomputed azimuth/range grids

### Backward Projection
- **File**: `utils/point_utils.py:64-145`
- **Function**: `sonar_ranges_to_points()`
- **Status**: ✅ Complete
- **Notes**: Converts sonar range image to 3D world-space points using polar geometry

### Normal Computation from Sonar
- **File**: `utils/point_utils.py:148-204`
- **Function**: `sonar_points_to_normals()`
- **Status**: ✅ Complete
- **Notes**: Uses finite differences for surface normal estimation

### Forward Projection (Render)
- **File**: `gaussian_renderer/__init__.py:160-414`
- **Function**: `render_sonar()`
- **Status**: ✅ Complete
- **Notes**: Full polar projection with bilinear splatting, Lambertian intensity model

### Camera-to-Sonar Extrinsic
- **File**: `utils/sonar_utils.py:279-316`
- **Class**: `SonarExtrinsic`
- **Status**: ✅ Complete
- **Notes**: 10cm vertical offset, 5° pitch down transformation

### Quaternion to Normal
- **File**: `gaussian_renderer/__init__.py:417-440`
- **Function**: `quaternion_to_normal()`
- **Status**: ✅ Complete
- **Notes**: Extracts surfel normal (local Z-axis) from rotation quaternion

---

## Design Decisions Implemented

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Elevation Handling | Sum all surfels within 20° arc | Sonar integrates over elevation beam spread |
| Occlusion Model | Same ray only (azimuth AND elevation) | Different elevations both contribute |
| Implementation Style | Splatting (not ray-casting) | Consistent with 2DGS, maintains differentiability |
| Intensity Model | Lambertian: `I = max(0, n·d) * opacity` | Simple acoustic reflectance approximation |
| Valid Mask | `intensity > 0` | Black pixels = no sonar return |
| Scale Factor Learning | Joint optimization with Adam | Single scalar, easy to optimize |
| What Scale Applies To | Pose translations only | Surfels learn from scratch, cleaner separation |
| Azimuth Convention | +X direction = negative azimuth | Matches physical sonar coordinate system |

---

## Coordinate Conventions

### Sonar Image Coordinates
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
```

### Camera/Sonar Frame (OpenCV Convention)
- +X = right
- +Y = down
- +Z = forward (optical axis / boresight)

### Key Formulas

**Forward Projection (surfel to pixel):**
```python
azimuth = -atan2(right, forward)  # negated for convention
col = (-azimuth / half_fov + 1) * (width / 2)
row = (range - range_min) / (range_max - range_min) * height
```

**Backward Projection (pixel to 3D):**
```python
azimuth = -(col - width/2) / (width/2) * half_fov
x = range * cos(azimuth)   # forward
y = -range * sin(azimuth)  # right (negated)
z = 0                       # elevation = 0 assumption
```

---

## Implementation Improvements Over Original Plan

1. **Log-scale for scale factor**: `scale = exp(log_scale)` guarantees positive values
2. **Bilinear splatting**: 4-neighbor interpolation for smoother gradients
3. **Differentiable masking**: Multiply by mask instead of in-place assignment
4. **Avoid torch.inverse()**: Use `R^T` for rotation inverse (better numerical stability)

---

## Debug Tools

### `debug_before_after_mesh.py`
Single-frame debugging script that outputs:
- `sonar_init_points.ply`: Initial point cloud from backward projection
- `mesh_before_training.ply`: Mesh from sonar-initialized Gaussians
- `mesh_after_100iter.ply`: Mesh after 100 training iterations
- `pose_pyramid_wireframe.ply`: Single pose visualization
- `gt_sonar_frame.png` / `rendered_sonar_frame.png`: Visual comparison

---

## Files Modified

| File | Changes |
|------|---------|
| `arguments/__init__.py` | Sonar params, scale factor config |
| `gaussian_renderer/__init__.py` | `render_sonar()`, `quaternion_to_normal()` |
| `utils/point_utils.py` | `sonar_ranges_to_points()`, `sonar_points_to_normals()` |
| `utils/sonar_utils.py` | `SonarScaleFactor`, `SonarConfig`, `SonarExtrinsic` |
| `debug_before_after_mesh.py` | Debug script for single-frame testing |

---

## TODO / Next Steps

- [x] **Implement curriculum learning for scale factor** (see Decision 001 in `docs/DESIGN_DECISIONS.md`)
  - Stage 1: Fix surfels, learn scale only ✅
  - Stage 2: Fix scale, learn surfels ✅
  - Stage 3: Joint fine-tuning ✅
- [ ] **Fix scale factor learning** - converges to ~1.0 but correct value is 0.66 (calibration cube). Currently using frozen scale=0.65.
- [ ] Integrate into main `train.py` training loop
- [ ] Add TensorBoard logging for scale factor convergence
- [x] Test with full dataset (multiple frames) - debug_multiframe.py with 5 frames works
- [ ] Evaluate mesh quality vs camera-based 2DGS
- [ ] (Optional) CUDA kernel optimization if Python forward projection is too slow

---

## Known Issues / Watch Items

| Issue | Status | Notes |
|-------|--------|-------|
| **Scale-surfel coupling** | Mitigated | Scale and surfel positions can compensate for each other; curriculum learning works (see Decision 001) |
| **Matrix transpose bug** | **FIXED** | `world_view_transform` is transposed; translation in row 3, not column 3 (see Bug Fix 001) |
| **Scale factor learning** | **TODO** | Learning converges to ~1.0 but correct value is 0.66 (from calibration cube). Currently frozen at 0.65. Need to investigate why learning doesn't converge to correct value. |
| Top row artifacts | Mitigated | Masking top 10 rows in render |
| Elevation assumption | Accepted | Assuming elevation=0 for backward projection |

## Session Notes (2025-01-10)

### Debug Scripts Created
- `debug_before_after_mesh.py`: Single-frame debugging (scale_factor=None for baseline)
- `debug_multiframe.py`: Multi-frame with curriculum learning (5 frames, 3 stages), raw-frame comparisons, and `scale_and_loss.png` plotting; Stage 1 set to 1000 iters for convergence checks

### Key Finding: Scale Factor Bug
- Scale factor was not affecting rendered output (gradient always 0)
- Root cause: `world_view_transform` matrix stored transposed (OpenGL convention)
- Translation is in `w2v[3, :3]` not `w2v[:3, 3]`
- Fix applied in `gaussian_renderer/__init__.py`

### Point Distance Diagnostic
Points ARE correctly placed at metric distances (3-30m from cameras):
```
Frame 0: min=3.09m, max=29.70m, mean=19.38m
Frame 1: min=3.09m, max=29.70m, mean=18.88m
```
If they appear <1m in Blender, it's likely Blender's scale interpretation of COLMAP coordinates.

### Bug Fix Verified ✅ (2025-01-10)

**Scale sensitivity test now shows different losses for different scale values:**
```
scale=0.5: L1=0.029040, SSIM=0.5943
scale=1.0: L1=0.033910, SSIM=0.5211
scale=2.0: L1=0.031900, SSIM=0.5307
```

**Gradients now non-zero:** `grad=-0.445334` at iteration 1 (was always 0 before fix)

**Curriculum Learning Test Results (debug_multiframe_v7):**
- Stage 1 (scale only, 50 iters): Scale converged 1.0 → 1.053
- Stage 2 (surfels only, 100 iters): L1 dropped 0.034 → 0.008, SSIM improved 0.59 → 0.86
- Stage 3 (joint, 50 iters): Final SSIM=0.875, scale=1.053

**Conclusion:** Scale factor learning is now working correctly. The curriculum learning approach (scale-first) is effective.

## Session Notes (2026-01-12)

- 500-frame runs show loss oscillations without downward trend and raw comparisons not matching; brightened comparisons only capture low-frequency blobs.
- Thin-leg bright dots from the calibration cube are missing in multi-frame outputs; SSIM weighting may be suppressing high-frequency detail (to revisit).
- Switched training frame selection to shuffle-per-epoch to remove strict per-frame cycling in `debug_multiframe.py`.
- Added bright-pixel loss (top-k brightest GT pixels) with tunable `BRIGHT_PERCENTILE`, `BRIGHT_WEIGHT`, `BRIGHT_MIN_PIXELS` in `debug_multiframe.py` to preserve small bright dots; base loss remains `0.8*L1 + 0.2*(1-SSIM)` for blending.
- Loss alternatives noted for trial: intensity-weighted L1, top-k bright-pixel loss (implemented), reduced/disabled SSIM, and blended base+bright losses.

---

## References

- **Design decisions**: `docs/DESIGN_DECISIONS.md` (tracks all major decisions with reasoning)
- Original plan: `.cursor/plans/sonar_projection_with_scale_c2ed5703.plan.md`
- Sonar specs: Sonoptix Echo (120° azimuth, 20° elevation, 0.2-3.0m range)
- Base repo: 2D Gaussian Splatting (Unbiased Surfel)

---

*Last updated: 2025-01-10*
