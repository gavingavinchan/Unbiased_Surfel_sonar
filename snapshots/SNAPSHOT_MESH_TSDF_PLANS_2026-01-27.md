# Snapshot: Mesh TSDF Plans vs Original

**Date/Time:** 2026-01-27 02:05:34 CST
**Git Commit:** 44d6382d6ada1221d2a61e777cb2ac6a53d6ce59
**Branch:** debug-multiframe-r2

## Status
- Not implementing mesh changes now; priority is to improve training stability/quality first.

## Side-by-Side: Original Camera TSDF vs Sonar-Native TSDF Plan

### Original Camera TSDF (Current Baseline)
- **Inputs**: Trained Gaussians + camera poses.
- **Rendering**: `render(...)` (pinhole projection).
- **Depth meaning**: `surf_depth` is camera-space Z along pinhole rays.
- **Integration**: Open3D `ScalableTSDFVolume` with `PinholeCameraIntrinsic` from `viewpoint_cam.projection_matrix` (`utils/mesh_utils.py::to_cam_open3d`).
- **Consistency**: Depth and ray model match (pinhole), so meshes stay bounded by camera frustums and `depth_trunc`.

### Sonar-Native TSDF (Planned)
- **Inputs**: Trained Gaussians + sonar poses + `SonarConfig` (+ `SonarExtrinsic`).
- **Rendering**: `render_sonar(...)` to produce `surf_depth` (range map) and `rend_alpha` (hit mask).
- **Depth meaning**: `surf_depth` is range along sonar rays in polar grid.
- **Integration**: Custom TSDF integration along sonar rays (azimuth/elevation) instead of pinhole rays.
- **Consistency**: Depth and ray model match (sonar), so meshes stay within sonar FOV/range even with many overlapping frames.

## Sonar-Native TSDF Pipeline Plan (Gaussian-Only)

1) **Ray model per frame**
   - Use the exact transform path from `render_sonar` (scaled translation + sonar extrinsic).
   - Build per-pixel ray directions from `SonarConfig` (azimuth/elevation grid).

2) **Render sonar depth from Gaussians**
   - Call `render_sonar(...)` to get `surf_depth` and `rend_alpha`.
   - Valid pixels: `rend_alpha > 0` (or a small weight threshold).

3) **TSDF volume bounds + resolution**
   - Bound by union of sonar frustums or conservative AABB from trajectory and `range_max`.
   - Voxel size from sonar resolution:
     - `delta_az = azimuth_fov_rad / W`
     - `delta_r = (range_max - range_min) / H`
     - `voxel_size = max(delta_r, range_ref * delta_az)` with `range_ref = range_max`.
   - `sdf_trunc = 5 * voxel_size` (align with existing TSDF defaults).

4) **TSDF integration along sonar rays**
   - For each valid pixel: `p = origin + depth * dir`.
   - Update TSDF in a band around `p` along the ray.
   - Start with uniform weights; optionally scale by `weight_sum` later.

5) **Mesh extraction + postprocess**
   - Marching cubes on TSDF grid.
   - Optional `post_process_mesh` to remove floaters.

## Notes on Alpha (Sonar Render)
- `render_sonar` returns `rend_alpha = (weight_sum > 0)`.
- This is a hit mask derived from intensity contributions; it is not currently used for TSDF extraction.
