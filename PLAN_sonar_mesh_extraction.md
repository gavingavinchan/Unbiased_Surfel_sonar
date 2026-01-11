# Plan: FOV-Aware Surfel Size Constraints

**Date**: 2025-01-11
**Author**: Claude Opus 4.5
**Status**: Draft

---

## Problem Statement

Current FOV pruning only checks if surfel **centers** are within sonar FOV. However, surfels have **size** (scaling parameters `_scaling [N, 2]`) that defines their 3D extent. A surfel with center inside FOV but large size may extend outside the FOV boundary.

**Current behavior**:
- `is_in_sonar_fov()` checks center position only
- `prune_outside_fov()` removes surfels with centers outside FOV
- Surfel size is not considered

**Goal**: Ensure surfels are **fully contained** within FOV, including their size extent.

---

## Background: Surfel Scaling

### Storage
- `_scaling` tensor: `[N, 2]` (2D Gaussian: sx, sy)
- Stored in log-space, activated via `exp(_scaling)`
- Actual size: `scaling = exp(_scaling)`

### Key Locations
| What | File | Line |
|------|------|------|
| Storage | `gaussian_model.py` | 99-100 |
| Initialization | `gaussian_model.py` | 137-138 |
| Size clamping | `train.py` | 262 |
| Pruning large surfels | `gaussian_model.py` | 402 |

### Size Limits
- **Max size**: `0.1 × scene_extent` (clamped after densification)
- **Pruning threshold**: `0.1 × extent` triggers pruning

### Sonar Mode Note
In `render_sonar()`, **surfel scaling is NOT used** for polar projection. Each surfel projects to a 2×2 pixel bilinear splat regardless of size. However, the 3D extent still matters for:
1. Mesh extraction (TSDF sees 3D positions)
2. Physical correctness (surfel should represent local surface patch)

---

## Solution Options

### Option A: Prune Surfels Extending Beyond FOV

**Approach**: After center-based FOV check, additionally prune surfels whose 3D extent crosses FOV boundary.

```python
def is_fully_in_sonar_fov(xyz, scaling, camera, sonar_config, scale_factor):
    """Check if surfel center AND extent are within FOV."""
    # Get surfel size (max of sx, sy as radius)
    surfel_radius = scaling.max(dim=1).values  # [N]

    # Check center
    center_in_fov = is_in_sonar_fov(xyz, camera, sonar_config, scale_factor)

    # Check if surfel could extend to FOV boundary
    # Compute margin to each FOV boundary (azimuth, elevation, range)
    margin_to_boundary = compute_fov_margin(xyz, camera, sonar_config, scale_factor)

    # Surfel is fully inside if margin > surfel_radius
    fully_inside = (margin_to_boundary > surfel_radius) & center_in_fov

    return fully_inside
```

| Pros | Cons |
|------|------|
| Clean solution | May remove too many surfels near FOV edges |
| Simple logic | Need to compute distance to FOV boundary |

### Option B: Clamp Surfel Size at FOV Boundary

**Approach**: Don't prune, but clamp surfel size so it doesn't extend beyond FOV.

```python
def clamp_scaling_to_fov(gaussians, camera, sonar_config, scale_factor):
    """Reduce surfel size if it would extend beyond FOV."""
    xyz = gaussians.get_xyz
    scaling = gaussians.get_scaling

    # Compute max allowed size based on distance to FOV boundary
    margin = compute_fov_margin(xyz, camera, sonar_config, scale_factor)

    # Clamp scaling to not exceed margin
    clamped_scaling = torch.min(scaling, margin.unsqueeze(-1))
    gaussians._scaling = inverse_scaling_activation(clamped_scaling)
```

| Pros | Cons |
|------|------|
| Preserves surfels | More complex implementation |
| Gradual degradation at edges | May distort surfel shapes |

### Option C: Mask Gradients for Boundary Surfels

**Approach**: During backward pass, zero out gradients for surfels extending beyond FOV.

```python
# In render_sonar(), before returning:
if training:
    fully_inside = is_fully_in_sonar_fov(...)
    # Mask viewspace_points gradients for boundary surfels
    # These surfels won't receive gradient updates
```

| Pros | Cons |
|------|------|
| Doesn't remove surfels | Complex gradient manipulation |
| Soft constraint | May cause training instability |

---

## Recommended Approach: Option A (Prune Extending Surfels)

This is the cleanest solution and aligns with the existing pruning infrastructure.

### Implementation Plan

#### Step 1: Add FOV Margin Computation

**File**: `debug_multiframe.py`

```python
def compute_fov_margin(xyz, camera, sonar_config, scale_factor):
    """
    Compute distance from each point to the nearest FOV boundary.

    Returns [N] tensor of distances in world units.
    """
    # Transform to sonar frame (same as is_in_sonar_fov)
    w2v = camera.world_view_transform.cuda()
    R_w2v = w2v[:3, :3]
    t_w2v = w2v[3, :3]
    t_w2v_scaled = scale_factor.scale * t_w2v
    points_sonar = (xyz @ R_w2v.T) + t_w2v_scaled

    right = points_sonar[:, 0]
    down = points_sonar[:, 1]
    forward = points_sonar[:, 2]

    # Compute current angles and range
    range_vals = torch.sqrt(right**2 + down**2 + forward**2)
    azimuth = torch.atan2(right, forward)
    horiz_dist = torch.sqrt(right**2 + forward**2)
    elevation = torch.atan2(down, horiz_dist)

    # Compute margin to each boundary
    half_az = math.radians(sonar_config.azimuth_fov / 2)
    half_el = math.radians(sonar_config.elevation_fov / 2)

    # Angular margins (convert to approximate linear distance at current range)
    az_margin = (half_az - torch.abs(azimuth)) * range_vals
    el_margin = (half_el - torch.abs(elevation)) * range_vals

    # Range margins
    range_margin_near = range_vals - sonar_config.range_min
    range_margin_far = sonar_config.range_max - range_vals

    # Minimum margin across all constraints
    margin = torch.min(torch.stack([
        az_margin, el_margin, range_margin_near, range_margin_far
    ], dim=0), dim=0).values

    return margin
```

#### Step 2: Update FOV Check to Include Size

**File**: `debug_multiframe.py`

```python
def is_fully_in_sonar_fov(xyz, scaling, camera, sonar_config, scale_factor):
    """
    Check if surfel center AND extent are within FOV.

    Args:
        xyz: [N, 3] surfel center positions
        scaling: [N, 2] surfel scaling (already activated, not log)
        camera: Camera object
        sonar_config: SonarConfig
        scale_factor: SonarScaleFactor

    Returns:
        [N] boolean tensor
    """
    # Get surfel "radius" (max of sx, sy)
    surfel_radius = scaling.max(dim=1).values  # [N]

    # Check center is in FOV
    center_ok = is_in_sonar_fov(xyz, camera, sonar_config, scale_factor)

    # Compute margin to FOV boundary
    margin = compute_fov_margin(xyz, camera, sonar_config, scale_factor)

    # Surfel fully inside if margin > radius
    fully_inside = center_ok & (margin > surfel_radius)

    return fully_inside
```

#### Step 3: Update Pruning Function

**File**: `debug_multiframe.py`

```python
def prune_outside_fov(gaussians, training_frames, sonar_config, scale_factor,
                      check_size=True):
    """
    Prune surfels outside FOV, optionally including size check.

    Args:
        check_size: If True, also prune surfels whose size extends beyond FOV
    """
    xyz = gaussians.get_xyz

    if check_size:
        scaling = gaussians.get_scaling  # [N, 2], activated
        visible_masks = []
        for cam in training_frames:
            in_fov = is_fully_in_sonar_fov(xyz, scaling, cam, sonar_config, scale_factor)
            visible_masks.append(in_fov)
    else:
        # Original center-only check
        visible_masks = []
        for cam in training_frames:
            in_fov = is_in_sonar_fov(xyz, cam, sonar_config, scale_factor)
            visible_masks.append(in_fov)

    all_masks = torch.stack(visible_masks, dim=0)
    visible_from_any = all_masks.any(dim=0)
    prune_mask = ~visible_from_any

    num_to_prune = prune_mask.sum().item()
    if num_to_prune > 0:
        gaussians.prune_points(prune_mask)

    return num_to_prune
```

#### Step 4: Add Diagnostic Output

Show how many surfels fail center-only vs size-aware checks:

```python
# In diagnostic section
for cam in training_frames:
    center_ok = is_in_sonar_fov(xyz, cam, sonar_config, scale_factor)
    fully_ok = is_fully_in_sonar_fov(xyz, scaling, cam, sonar_config, scale_factor)
    print(f"  Frame {i}: center_ok={center_ok.sum()}, fully_ok={fully_ok.sum()}")
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `debug_multiframe.py` | Add `compute_fov_margin()`, `is_fully_in_sonar_fov()`, update `prune_outside_fov()` |

---

## Verification

1. **Run**: `python debug_multiframe.py`

2. **Check diagnostic output**:
   - Compare "center_ok" vs "fully_ok" counts
   - Should see difference for large surfels near FOV edges

3. **Load in Blender**:
   - `surfels_after_training.ply` should have no large surfels at FOV boundaries
   - Mesh should have cleaner edges at FOV boundaries

4. **Quantitative**:
   - For each remaining surfel, verify: `margin > max(sx, sy)`

---

## Notes

- **Margin computation approximation**: Angular margin converted to linear using `angle × range`. This is an approximation that's accurate for small angles.

- **Surfel radius definition**: Using `max(sx, sy)` as conservative estimate. Could also use `sqrt(sx² + sy²)` for 2D diagonal.

- **Multi-frame consideration**: A surfel only needs to be fully inside ONE training frame's FOV (union of FOVs).

---

## Alternative: Size Clamping Instead of Pruning

If pruning removes too many surfels, could clamp size instead:

```python
def clamp_scaling_to_fov_margin(gaussians, training_frames, sonar_config, scale_factor):
    """Reduce surfel size if extending beyond FOV."""
    xyz = gaussians.get_xyz
    scaling = gaussians.get_scaling

    # Compute max margin across all training frames
    max_margin = torch.zeros(len(xyz), device=xyz.device)
    for cam in training_frames:
        margin = compute_fov_margin(xyz, cam, sonar_config, scale_factor)
        max_margin = torch.max(max_margin, margin)

    # Clamp scaling to margin
    max_scaling = max_margin.unsqueeze(-1).expand_as(scaling)
    clamped = torch.min(scaling, max_scaling)

    # Update (need to convert back to log space)
    gaussians._scaling.data = torch.log(clamped + 1e-8)
```

This preserves surfels but constrains their size.
