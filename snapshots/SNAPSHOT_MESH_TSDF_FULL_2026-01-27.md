# Snapshot: Full Mesh TSDF Notes (Original + Sonar Plan)

**Date/Time:** 2026-01-27 02:09:43 CST
**Git Commit:** 44d6382d6ada1221d2a61e777cb2ac6a53d6ce59
**Branch:** debug-multiframe-r2

## Status
- Not implementing mesh changes now; priority is to improve training stability/quality first.

## Original Camera TSDF Pipeline (Full Answer, 6 Parts)

Here’s the **original camera-only TSDF mesh extraction pipeline** (as in the 0d41037 state). It uses the trained surfels only; no raw images after training.

**1) Input representation (trained surfels)**
- The trained scene is stored in a `GaussianModel`: positions, scales, rotations, opacity, and color features.
- Mesh extraction does not re-read image pixels; it uses the trained Gaussians.

**2) Render depth and color from surfels (per camera)**
- `GaussianExtractor.reconstruction(viewpoint_stack)` loops through cameras.
- For each camera, it calls the renderer (`render(...)` with pinhole projection).
- The renderer produces:
  - `render` (RGB image)
  - `surf_depth` (per-pixel depth along the pinhole ray)
  - other buffers (alpha, normals)
- These RGB + depth maps are stored in memory.

**3) Estimate bounds**
- `estimate_bounding_sphere()` computes a scene center + radius from camera poses.
- This is used to pick `depth_trunc` and voxel size for TSDF.

**4) TSDF fusion (Open3D)**
- `extract_mesh_bounded(...)` constructs a `ScalableTSDFVolume`.
- For each camera:
  - Builds a pinhole intrinsic from `viewpoint_cam.projection_matrix`.
  - Builds an Open3D `RGBDImage` from the rendered RGB + depth.
  - Optionally masks depth with `gt_alpha_mask`.
  - Integrates the RGBD into the TSDF volume with the camera extrinsic.
- Because the depth maps were produced with **the same pinhole model**, the rays match.

**5) Mesh extraction**
- After integrating all frames, `volume.extract_triangle_mesh()` generates the mesh.

**6) Why it works in the original camera pipeline**
- Rendered depth is in **camera space** and rays are **pinhole**.
- TSDF integration assumes **pinhole**, so the geometry is consistent and bounded by camera frustums.

That’s the full surfels → render depth → TSDF → mesh pipeline in the original setup.

## Sonar-Native TSDF Plan (Full Answer)

Here’s a concrete outline for a **Gaussian-only, sonar-projection TSDF** mesh extraction. No raw sonar pixels at mesh time.

**Goal**
- Fuse depth rendered from trained Gaussians along **sonar rays** (azimuth/elevation) so the TSDF volume is bounded by sonar FOV/range.

**High-level steps**
1) **Render sonar depth from Gaussians**
   - Use `render_sonar(...)` to get `surf_depth` (already in sonar range space).
   - This is per-pixel depth in the sonar image grid (azimuth × range).

2) **Precompute sonar ray directions**
   - From `SonarConfig` (azimuth/elevation FOV, image size), build a ray direction for each pixel.
   - The ray origin is the sonar origin in world coords (camera pose + sonar extrinsic).
   - Directions are in world coords using the camera’s `world_view_transform` and optional sonar extrinsic.

3) **TSDF integration (custom, sonar-native)**
   - Build a TSDF volume in world space (voxel grid).
   - For each frame and each pixel with valid depth:
     - The surface point is `p = origin + depth * dir`.
     - Update voxels along the ray within a truncation band around `p` (standard TSDF update).
   - Accumulate a weighted TSDF volume across frames.

4) **Extract mesh**
   - Run marching cubes on the TSDF volume.
   - Optional: restrict the volume to the union of sonar frustums for efficiency.

**Where it lives**
- New module, e.g. `utils/sonar_tsdf.py` with:
  - `build_sonar_rays(sonar_config, camera, sonar_extrinsic)`
  - `integrate_tsdf(volume, rays, depth_map, trunc, voxel_size)`
  - `extract_mesh(volume)`
- In `debug_multiframe.py`, replace `GaussianExtractor.extract_mesh_bounded(...)` with this sonar TSDF pipeline.

**Key inputs**
- Trained Gaussians (for `render_sonar` depth).
- Camera poses + sonar extrinsic (for ray directions).
- SonarConfig (FOV/range).

**Expected benefits**
- Mesh bounded by sonar FOV/range by construction.
- Scales to 500+ frames with overlap; overlap only improves TSDF stability.

## Alpha Explanation (Full Answer)

- In `render_sonar`, `rend_alpha` is a **binary hit mask**: `(weight_sum > 0)` where `weight_sum` accumulates intensity contributions of surfels per pixel.
- `range_image` is computed as an **intensity-weighted average** of ranges, then normalized by `weight_sum`.
- `rend_alpha` is currently used only for visualization/logging (e.g., `train.py`, `utils/image_utils.py`). It is **not used** in mesh extraction, and it doesn’t gate the TSDF.
- In the original camera renderer, `rend_alpha` is accumulated opacity used to normalize depth and weight normals; in sonar it’s just a hit mask derived from weight accumulation.
