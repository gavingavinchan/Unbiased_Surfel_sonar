# Plan: Training Pipeline Overview

**Date/Time:** 2026-01-28 (documentation)
**Git Commit:** 88b210c
**Type:** Reference documentation (not an implementation plan)

## Overview

This document explains how the sonar Gaussian splatting training works, specifically how back projection and forward projection fit into the training loop.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INITIALIZATION (Before Training)             │
│                                                                  │
│   Sonar Images (2D)  ───► BACK PROJECTION ───► 3D Point Cloud   │
│      [256×200]                                    (Surfels)      │
│                                                                  │
│   Uses: sonar_frame_to_points() from sonar_utils.py             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP (Iterative)                   │
│                                                                  │
│   3D Surfels ───► FORWARD PROJECTION ───► Rendered Image (2D)   │
│                         │                      [256×200]         │
│                         │                          │             │
│                         │                          ▼             │
│                   render_sonar()          Compare with GT Image  │
│                                                    │             │
│                                                    ▼             │
│                                             Compute Loss         │
│                                             (L1 + SSIM +         │
│                                              Bright-pixel)       │
│                                                    │             │
│                                                    ▼             │
│                                            loss.backward()       │
│                                                    │             │
│                                                    ▼             │
│                                            Update Surfels        │
│                                            (via gradients)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Back Projection → Point Cloud Initialization

**When**: Once, before training starts

**Purpose**: Converts 2D sonar images to 3D points to initialize the surfels

**Location**: `sonar_frame_to_points()` in `utils/sonar_utils.py:323-415`

### Coordinate Conversion

```python
# For each training frame:
pixel (col, row)
    → polar (azimuth, range)
    → 3D camera frame (x, y, z)
    → world coordinates
```

### Key Equations

```python
# Pixel to polar
azimuth = -(col - W/2) / (W/2) * half_fov_rad
range = range_min + (row / H) * (range_max - range_min)

# Polar to 3D camera frame
x_cam = -range * sin(azimuth)  # Note: sign flip for coordinate convention!
y_cam = 0                       # Elevation assumed flat
z_cam = range * cos(azimuth)

# Camera frame to world
p_world = R_c2w @ p_cam + camera_center
```

### Coordinate Conventions

**Sonar Image:**
- Column 0 = +60° azimuth (left), Column 255 = -60° (right)
- Row 0 = range_min (0.2m, close), Row 199 = range_max (3.0m, far)

**Camera Frame (OpenCV/COLMAP):**
- +X = right, +Y = down, +Z = forward

### Key Implementation Detail

The x-coordinate requires a sign flip (`-range * sin(azimuth)`) because:
- Positive azimuth = left side of image
- Left in image = negative X in camera frame (after coordinate convention)

---

## 2. Forward Projection → Rendered Images

**When**: Every training iteration

**Purpose**: Projects 3D surfels to 2D sonar image for loss computation

**Location**: `render_sonar()` in `gaussian_renderer/__init__.py:193-470`

### Pipeline

```
3D surfels (world coords)
    → scale by scale_factor (COLMAP→metric)
    → world-to-camera transform
    → polar coords (azimuth, range, elevation)
    → FOV check (size-aware)
    → pixel coords (col, row)
    → bilinear splat with Lambertian intensity
    → rendered image [256×200]
```

### Key Transformations

**1. World to Sonar Frame:**
```python
# Apply scale factor and camera transform
t_w2c_metric = scale_factor.scale * t_w2c  # COLMAP→metric translation
means3D_scaled = scale_factor.scale * means3D
points_sonar = (means3D_scaled @ R_w2c.T) + t_w2c_metric
```

**2. Cartesian to Polar:**
```python
azimuth = -torch.atan2(right, forward)
range_vals = torch.sqrt(right**2 + down**2 + forward**2)
elevation = torch.atan2(down, horiz_dist)
```

**3. Polar to Pixel:**
```python
col = (-azimuth / half_azimuth_rad + 1) * (W / 2)
row = (range_vals - range_min) / (range_max - range_min) * H
```

**4. Intensity (Lambertian Model):**
```python
normals = quaternion_to_normal(rotations)
lambertian = clamp(dot(normals, dir_to_sonar), min=0)
intensity = opacity * lambertian
```

### Key Difference from Back Projection

Forward projection is **fully differentiable** - gradients flow backwards through all operations via PyTorch autograd.

### Size-Aware FOV Checking

Surfels are checked with their radius extent, not just center point:
```python
in_fov = center_in_fov & (margin > surfel_radius)
```
This prevents boundary artifacts where large surfels partly outside FOV receive invalid gradients.

---

## 3. Loss Computation & Gradient Flow

### Losses Used

| Loss | Weight | Purpose |
|------|--------|---------|
| L1 | 0.8 × 0.5 | Pixel-wise absolute difference |
| 1 - SSIM | 0.2 × 0.5 | Structural similarity |
| Bright-pixel | 0.5 | Focus on top 5% brightest pixels (strong returns) |

```python
base_loss = 0.8 * L1 + 0.2 * (1 - SSIM)
total_loss = 0.5 * base_loss + 0.5 * bright_loss
```

### Gradient Flow Path

```
Loss
 └→ Rendered Image (bilinear weights, scatter_add_)
     └→ Pixel coordinates (col, row)
         └→ Polar coords (azimuth, range)
             └→ Surfel positions (xyz)
             └→ Scale factor (_log_scale)
     └→ Intensity (Lambertian)
         └→ Surfel opacity
         └→ Surfel rotation (normals via quaternion)
```

### Critical Implementation Details

**Bilinear Splatting**: Uses `scatter_add_` which is fully differentiable
```python
rendered_flat.scatter_add_(0, idx, weight * intensity)
```

**Top-Row Masking**: Uses multiplicative mask (not in-place) to preserve gradients
```python
mask[:, :10, :] = 0
rendered = rendered * mask  # Gradients still flow
```

**Weight Detachment**: Range normalization uses detached weights to prevent normalization from dominating gradients

---

## 4. Curriculum Learning (3 Stages)

### The Problem: Scale-Surfel Coupling

Scale factor and surfel positions are coupled:
- For any scale `s`, surfels can adjust positions to compensate
- Joint optimization cannot distinguish correct solution
- Degenerate case: scale→1.0 with surfels at wrong positions

### Solution: Sequential Training

| Stage | Iterations | What's Learned | What's Frozen |
|-------|------------|----------------|---------------|
| **1** | 0 (disabled) | Scale factor | Surfels |
| **2** | 1000 | Surfels | Scale factor |
| **3** | 1 (disabled) | Both (planned) | — |

### Stage 1: Learn Scale Only (DISABLED)

**Why disabled**: Scale tends to converge to 1.0 instead of correct ~0.66. Using pre-calibrated scale (0.6127 for R2).

```python
# Only scale optimizer steps
loss.backward()
scale_optimizer.step()
gaussians.optimizer.zero_grad()  # Don't step, just clear
```

### Stage 2: Learn Surfels Only (ACTIVE)

**Current main training stage**

```python
sonar_scale_factor._log_scale.requires_grad = False  # Freeze scale

loss.backward()
gaussians.optimizer.step()  # Update surfels
```

Additional features:
- FOV-aware pruning (removes surfels outside all training FOVs)
- Periodic mesh extraction and visualization

### Stage 3: Joint Fine-tuning (PLANNED)

Would enable joint refinement with careful learning rate balance. Currently disabled due to scale convergence issues.

---

## 5. Scale Factor Mechanism

**Class**: `SonarScaleFactor` in `utils/sonar_utils.py:11-60`

### Log-Space Parameterization

```python
class SonarScaleFactor(nn.Module):
    def __init__(self, init_value=1.0):
        self._log_scale = nn.Parameter(torch.tensor(math.log(init_value)))

    @property
    def scale(self):
        return torch.exp(self._log_scale)  # Always positive
```

**Why log-space?**
- Ensures positivity (scale > 0 always)
- Linear gradients in log space improve optimization
- Prevents scale collapse to 0 or blow-up to ∞

### Usage in Rendering

Scale affects both camera translation and surfel positions:
```python
t_w2c_metric = scale_factor.scale * t_w2c      # Camera translation
means3D_scaled = scale_factor.scale * means3D  # Surfel positions
```

---

## 6. Summary Table

| Component | Role | Differentiable? | When |
|-----------|------|-----------------|------|
| **Back projection** | Initialize surfels from sonar images | No (numpy) | Once at startup |
| **Forward projection** | Render surfels to compare with GT | Yes (torch) | Every iteration |
| **Loss** | L1 + SSIM + bright-pixel | Yes | Every iteration |
| **Scale factor** | Align COLMAP ↔ metric units | Frozen (broken) | — |
| **Surfel optimizer** | Update xyz, rotation, opacity, scale | Yes | Stage 2+ |

---

## Key Insight

**Back projection gives you a starting point, forward projection lets you refine it via gradient descent.**

The entire system is designed around this duality:
1. Back projection provides reasonable geometry from real sonar data
2. Forward projection creates a differentiable rendering path
3. Loss measures how well current surfels explain the observations
4. Gradients update surfels to better match observations

---

## File References

| File | Key Functions |
|------|---------------|
| `utils/sonar_utils.py` | `sonar_frame_to_points()`, `SonarScaleFactor` |
| `gaussian_renderer/__init__.py` | `render_sonar()`, `compute_fov_margin()` |
| `debug_multiframe.py` | Training loop, curriculum stages |
| `utils/loss_utils.py` | `l1_loss()`, `ssim()` |
