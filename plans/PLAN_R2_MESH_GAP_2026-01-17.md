# Plan: R2 Mesh Missing Despite Surfels

**Date/Time:** 2026-01-17 02:40:11 CST  
**Git Commit:** 9f0ea511e2803f060a2bbf457415bbef5d1dded7

## Symptom
- `surfels_after_training.ply` aligns with `sonar_init_points.ply`, but mesh misses the center region.

## Hypotheses
1. Mesh extraction uses pinhole assumptions and discards sonar depth returns.
2. `sonar_ranges_to_points()` uses translation column (`w2v[:3,3]`) instead of row (`w2v[3,:3]`) and misses scale + extrinsic, leading to wrong TSDF input.
3. TSDF integration expects pinhole intrinsics; sonar range-to-point conversion might be inconsistent.

## Plan
1. Audit `sonar_ranges_to_points()` transform (translation row/col, scale, extrinsic).
2. Fix translation row usage to match `render_sonar` convention.
3. Thread `scale_factor` and optional `sonar_extrinsic` into sonar-to-world conversion for mesh extraction.
4. Re-run `debug_multiframe.py` on R2 and compare `mesh_*` outputs.
5. If mesh still missing, add a point-cloud meshing fallback (Poisson/ball-pivot) using `sonar_init_points.ply`.

## Warning
- Prior WIP fixes (commit `37d763c`) were unreliable; validate each change.
