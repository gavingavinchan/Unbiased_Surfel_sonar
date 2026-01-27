# Snapshot: Mesh Extraction FOV Discussion

**Date/Time:** 2026-01-26 23:13:35 CST
**Git Commit:** 44d6382d6ada1221d2a61e777cb2ac6a53d6ce59
**Branch:** debug-multiframe-r2

## Context
- User confirmed `sonar_init_points.ply` and `surfels_after_training.ply` look correct in Blender.
- User observed all `mesh_*` outputs exceed the sonar FOV.
- Clarified that mesh extraction is post-training; raw sonar pixels are not used as mesh input.

## Current Behavior (Why Mesh Exceeds FOV)
- TSDF extraction uses Open3D pinhole intrinsics in `utils/mesh_utils.py` via `to_cam_open3d(...)` and integrates depth maps rendered from Gaussians.
- The sonar renderer is polar, but TSDF integration assumes pinhole rays, so geometry can be projected outside the sonar frustum.
- Poisson meshes operate on surfel-derived points/normals and can extrapolate/close surfaces beyond observed bounds.

## Key Clarification
- Any mention of per-pixel integration refers to **rendered depth pixels** from Gaussians, not raw sonar intensities.
- The intended pipeline should still derive meshes from trained Gaussians; bypassing Gaussians would only be for debugging.
