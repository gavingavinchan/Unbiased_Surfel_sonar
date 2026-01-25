# Progress Notes

## 2025-01-11: FOV-Aware Surfel Size Constraints

### Summary
Implemented size-aware FOV constraints so that backward projection from a sonar frame only affects surfels whose center + size extent are fully within that frame's FOV.

### Changes Made

**gaussian_renderer/__init__.py**
- Added `compute_fov_margin()` helper to compute distance from each point to nearest FOV boundary
- Updated `render_sonar()` FOV check to include surfel size: `in_fov = center_in_fov & (margin > surfel_radius)`

**debug_multiframe.py**
- Added `compute_fov_margin_debug()` helper
- Added `is_fully_in_sonar_fov()` function for size-aware FOV checking
- Updated `prune_outside_fov()` with `check_size=True` parameter

### Results (debug_multiframe_v26)

**Surfels:** 2214/2214 (100%) in FOV
- Range: 0.222 to 2.204m
- Azimuth: -59.0° to 58.6° (FOV: ±60°)
- Elevation: -8.6° to 8.6° (FOV: ±10°)

**Mesh:** 17,371/17,602 (98.7%) in FOV
- Range: 0.215 to 2.236m
- Azimuth: -49.5° to 61.1° (FOV: ±60°)
- Elevation: -18.8° to 14.0° (FOV: ±10°)

### Observations
- `mesh_after_stage2.ply` and `mesh_after_stage3.ply` look good
- Mesh extends ~1.3% beyond FOV in elevation direction (-18.8° to +14.0° vs ±10° limit)
- This is due to TSDF/marching cubes interpolation creating surface slightly beyond surfel positions
- `mesh_before_training.ply` still has the same problem (to investigate)

### Open Questions
- Why does `mesh_before_training` have mesh outside FOV even though surfels are constrained?
