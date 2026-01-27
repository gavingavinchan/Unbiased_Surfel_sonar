# Snapshot: R2 Single-Frame Debug

**Date/Time:** 2026-01-26 22:52:29 CST
**Git Commit:** 44d6382d6ada1221d2a61e777cb2ac6a53d6ce59
**Branch:** debug-multiframe-r2

## What Was Run
- Command(s) run:
  - `source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && SONAR_DATASET=r2 SONAR_NUM_FRAMES=1 python debug_multiframe.py 2>&1`
- Output directory:
  - `output/debug_multiframe_r2_v7`

## Key Observations
- Single-frame training selected `sonar_1765233318609` and initialized 1124 points; after FOV pruning at iter 90, 913 surfels remain.
- `sonar_init_points.ply` and `surfels_after_training.ply` align well in Blender, but all `mesh_*` outputs extend beyond the sonar FOV.
- TSDF meshes (`mesh_before_training.ply`, `mesh_after_stage2.ply`, `mesh_after_stage3.ply`) are empty (0 vertices), while Poisson meshes succeed (`mesh_poisson_after_stage3.ply` ~3798 V / 7428 T).
- Scale sensitivity test shows flat loss beyond ~0.9; debug print shows `t_w2v` column 3 is zero, so scale does not affect translation in that test.

## Interpretation (Mesh FOV)
- Not only Poisson: Poisson can extrapolate/close surfaces, but TSDF extraction uses pinhole Open3D cameras with no sonar FOV clipping in `utils/mesh_utils.py` and integrates depth maps directly, so meshes can extend outside the sonar FOV as well.

## Fixes Made During This Run
- Added `SONAR_NUM_FRAMES` env override to control frame count in `debug_multiframe.py`.
- Fixed Poisson filtering alignment bug in `debug_multiframe.py`: now masks `scales` when `opacities` filter reduces the point set.

## Files Generated (Selected)
- `output/debug_multiframe_r2_v7/sonar_init_points.ply`
- `output/debug_multiframe_r2_v7/surfels_after_training.ply`
- `output/debug_multiframe_r2_v7/mesh_poisson_after_stage3.ply`
- `output/debug_multiframe_r2_v7/scale_and_loss.png`
- `output/debug_multiframe_r2_v7/comparison_after_stage3_raw_frame0.png`
