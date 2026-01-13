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

## Session Notes (2026-01-12 continued) - Peak-Aware Loss Design

### Problem
Thin-leg bright dots from calibration cube are missing in outputs. Need a better loss function that preserves small bright features.

### Proposed Loss (from external LLM consultation)

**Log-compressed intensity** (per-frame 99th percentile normalization):
```
c = percentile(I_gt, 99)  # per-frame, detached
I_n = I / (c + delta)
J = log(1 + alpha * I_n)   # alpha=10
```

**Peak-aware weighted photometric loss (L_focal)**:
- Charbonnier penalty: `rho(x) = sqrt(x^2 + eps^2)`, eps=1e-3
- Peak weight: `w(x) = 1 + beta * sigmoid((J_gt - tau) / s)`
- tau = 97-99th percentile of J_gt, s=0.08, beta=20 (ramp from 0)
- `L_focal = E[w(x) * rho(J_pred - J_gt)]`

**Multi-scale blob loss (DoG)**:
- Scales: Sigma = {0.8, 1.6, 3.2} for 1-2px dots (or {1, 2, 4} for 2-3px)
- k = 1.6
- `L_blob = sum_sigma |DoG_sigma(J_pred) - DoG_sigma(J_gt)|_1`

**Peak-recall via distribution matching (KL)**:
- Softmax heatmaps: `p(x) = exp(gamma * J_gt) / sum`, gamma in [5,20]
- `L_KL = sum p(x) * log(p(x)/q(x))`
- Helps when model misses small dots entirely

**Total**: `L = L_focal + lambda_blob * L_blob + lambda_KL * L_KL`
- lambda_blob = 0.5, lambda_KL = 0.1

**Skip L_size**: Renderer uses fixed 2x2 bilinear splats regardless of 3D scale.

### Sonar Data Analysis (for loss tuning)

**Q1: Pixel radius of leg dots?**
- Measured: median ~1.6px, range 0.6-6.5px
- User correction: actual leg dots are **2-3 pixels**
- Implementation: Sigma = {0.8, 1.6, 3.2} (smaller scales for tighter matching)

**Q2: Forward model - additive or alpha compositing?**
- **Additive intensity** via `scatter_add_()` - intensities summed directly

**Q3: Frames normalized?**
- **Not normalized** - raw from sensor
- Max intensity varies 81-112, 99th percentile 37-42
- Per-frame 99th percentile normalization recommended

**Q4: Resolution alignment?**
- **Same resolution** - render_sonar outputs 200x256 (same as raw)

**Q5: Range-dependent gain/attenuation?**
- Initially appeared to have range-dependent intensity
- **Actually scene geometry, not sensor artifact**:
  - Near range (0.2-0.5m): Transducer backscatter/noise (masked with top 10 rows)
  - Mid range (0.76-2.45m): Empty water - nothing to reflect
  - Far range (2.45-3.0m): Actual scene content (floor, cube legs)
- **No range weighting correction needed**

**Q6: Saturation/clamping?**
- **No saturation** - max values 87-115, nowhere near 255

### Implementation (2026-01-12)

Implemented peak-aware loss in `debug_multiframe.py`:

**Parameters:**
```python
LOSS_ALPHA = 10.0           # Log compression
LOSS_DELTA = 1e-6           # Normalization stability
LOSS_EPSILON = 1e-3         # Charbonnier smoothing
LOSS_BETA_MAX = 20.0        # Peak weight boost (ramped 0→20)
LOSS_BETA_RAMP_ITERS = 2000
LOSS_SIGMOID_S = 0.08       # Sigmoid temperature
LOSS_TAU_PERCENTILE = 0.97  # Peak threshold
LOSS_DOG_SIGMAS = [0.8, 1.6, 3.2]  # DoG scales for 1-2px dots
LOSS_DOG_K = 1.6
LOSS_LAMBDA_BLOB = 0.5
LOSS_KL_GAMMA = 15.0        # KL softmax temperature
LOSS_KL_ETA = 1e-12
LOSS_LAMBDA_KL = 0.1
```

**Functions added:**
- `gaussian_blur_2d(x, sigma)` - Separable Gaussian blur
- `compute_dog(J, sigma, k)` - Difference of Gaussians
- `compute_peak_aware_loss(rendered, gt_image, iteration, total_iterations)` - Main loss

**Loss composition:**
- L_focal: Peak-weighted Charbonnier loss in log-intensity space
- L_blob: Multi-scale DoG blob matching
- L_KL: Masked peak distribution matching (KL divergence)
- Total: L = L_focal + 0.5*L_blob + 0.1*L_KL

### Revised Loss Function (2026-01-12)

Updated loss to address collapse issues. New formulation:

**Loss terms:**
- `L_pos = mean(m * charbonnier(J_pred - J_gt))` - Peak-only photometric (no 1+βm baseline)
- `L_neg = mean((1-m) * charbonnier(relu(J_pred - J_gt)))` - One-sided background overshoot penalty
- `L_blob` - GT-weighted DoG blob matching
- `L_KL` - Masked peak distribution KL divergence
- `L_mass = |M_pred - M_gt|` where `M = sum(m * J)` - Peak mass constraint

**Total:** `L = L_pos + 0.01*L_neg + 0.5*L_blob + 0.1*L_KL + 0.1*L_mass`

**Anti-collapse measures:**
- Stabilization phase (5k iters): disable pruning, reduce opacity LR to 5e-3
- Peak-gated opacity pruning: only prune if `(opacity < min) AND (peak_support < 0.05) AND (grad < 1e-4)`
- Peak support EMA updated every 10 iters via bilinear sampling from peak mask
- `torch.cuda.empty_cache()` every 100 iters to prevent OOM

### Training Results (v44, 2026-01-12)

**Configuration:** 500 frames, 30k iterations Stage 2, scale fixed at 0.65

**Loss convergence:**
| Iter | total_loss | L_mass | M_pred/M_gt |
|------|------------|--------|-------------|
| 1 | 1464 | 14476 | 12.9x |
| 5000 | 53.7 | 470 | 0.53x |
| 20000 | 11.5 | 9.4 | 0.96x |
| 30000 | 0.57 | 0.15 | 1.00x |

**Opacity stats at end:** mean=0.062, median=6e-5, 66% surfels have opacity < 1e-3

**Observations:**
1. Loss converged well (1464 → 0.57)
2. Peak mass preserved (M_pred ≈ M_gt at end)
3. First signs of vertical cube legs appearing in mesh
4. **Issues:**
   - Vertical legs in wrong positions (not in square formation)
   - No horizontal legs visible (likely physics - parallel to sonar beam)
   - Too many false positives in rendered sonar frames
   - 66% of surfels effectively "dead" but not pruned

**Next steps to try:**
1. Check `sonar_init_points.ply` - if cube wrong there, issue is upstream (poses/projection)
2. Increase `LOSS_LAMBDA_NEG` from 0.01 to 0.05 to reduce false positives
3. Reduce `PEAK_SUPPORT_MIN` from 0.05 to 0.02 to prune more dead surfels

---

## References

- **Design decisions**: `docs/DESIGN_DECISIONS.md` (tracks all major decisions with reasoning)
- Original plan: `.cursor/plans/sonar_projection_with_scale_c2ed5703.plan.md`
- Sonar specs: Sonoptix Echo (120° azimuth, 20° elevation, 0.2-3.0m range)
- Base repo: 2D Gaussian Splatting (Unbiased Surfel)

---

*Last updated: 2026-01-13*
