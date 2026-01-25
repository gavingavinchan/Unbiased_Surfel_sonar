# R2 Dataset Issues / Differences Observed

## 1) Dataset structure differs (not just poses)
- Legacy dataset contains extra folders: `sparse_camera_backup/`, `sparse_sonar/`, `sparse_sonar_500/`.
- R2 dataset only contains `sonar/` and `sparse/`.

## 2) COLMAP intrinsics changed
R2 is not a poses-only update; intrinsics changed too:

- **Legacy** `cameras.bin`:
  - Camera ID: 2935
  - Model: `PINHOLE`
  - Size: `2008x1093`
  - Params: `fx=974.0936, fy=974.0936, cx=1004.0, cy=546.5`

- **R2** `cameras.bin`:
  - Camera ID: 1
  - Model: `PINHOLE`
  - Size: `1803x1013`
  - Params: `fx=1182.5371, fy=1182.5371, cx=901.5, cy=506.5`

This affects FOV values used by `Scene`/`Camera` and can shift pose normalization + scale.

## 3) Camera count mismatch
- Legacy `cameras.bin` contains **10,060** cameras (only ID 2935 referenced by images).
- R2 `cameras.bin` contains **1** camera (ID 1).

This is a large structural change in the COLMAP model.

## 4) Pose scale/offset differences
From `sparse/0/images.bin` camera centers:

- Legacy center range: `[18.97, 2.80, 11.97]`
- R2 center range: `[12.16, 3.31, 7.69]`
- Legacy center norm mean/std: `3.3059 / 1.7638`
- R2 center norm mean/std: `3.8904 / 1.0888`

Best-fit similarity transform (legacy → R2) on common names:
- **Scale ≈ 0.816**
- **Translation ≈ [ +1.07, +0.11, −0.14 ]**
- **Residual mean ≈ 1.61** (not just a clean similarity)

This aligns with the R2 mesh looking **smaller and offset**.

## 5) Frame identity mismatch (not the same frames)
- Only **44** common image names across both datasets.
- The sonar timestamp ranges differ:
  - Legacy: starts at `1765233408009`
  - R2: starts at `1765233318609`

So R2 is not just “same experiment with better poses” — the frame list differs.

## 6) Interpolation script mismatch (camera_ prefix)
`scripts/interpolate_sonar_poses.py` expects camera filenames like `camera_*.png`.
Both datasets have `sonar_*.png` in `images.bin`, so the script extracts **zero** camera timestamps and fails with `IndexError` (empty list). This may indicate interpolation steps were not aligned with the expected naming.

## 7) Sonar extrinsic offset not applied in base
`render_sonar()` only applies `sonar_extrinsic` if explicitly passed. On the base commit, `debug_multiframe.py` doesn’t apply it, so the sonar is assumed co-located with the camera. If R2 poses are camera poses, this introduces an offset in the rendered/initialized geometry.

## 8) Scale factor handling in base
At `9f0ea51`, `render_sonar()` scales only the **translation**, not the **surfel positions**. For R2 (different scale), this can shrink geometry and cause the “few blobs in 1m space” symptom. (This was fixed in later WIP commit `37d763c`.)

## 9) Pose orientation mismatch (same names, different rotations)
For the 44 frames with common names:
- Rotation difference between legacy and R2 averages **~19°** (min ~6°, max ~37°)
- This suggests R2 is not just a scaled/translated version; orientation changed too.

## 10) Camera center magnitude ratios vary widely
For common frames, the ratio of camera center norms (R2 / legacy):
- Min ~0.37, max ~2.50, mean ~1.40
- Large variance implies non-uniform scale differences, not a single clean scale.

## 11) Sonar image size is the same (not the culprit)
Both datasets have sonar images sized `256x200`, so image resolution is not the cause.
