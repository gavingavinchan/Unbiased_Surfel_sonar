# Snapshot: R2 GUI Tuner Status

**Date/Time:** 2026-01-26 02:31:57 CST
**Git Commit:** 72affbd7f7a0056fbfa4207befb5ee3733c19c82
**Branch:** debug-multiframe-r2

## What Works (Verified)
- Command(s) run:
  - `source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && python scripts/poisson_tuner_gui.py`
- Expected outputs:
  - `output/poisson_tuner/mesh_poisson_after_stage3.ply` (auto-loaded; GUI falls back to earlier Poisson meshes if needed)

## What’s Broken / Unclear
- R2 mesh quality still poor after training; output looks like small blobs (user expects ~8m scale).
- TSDF mesh often misses the center even when `surfels_after_training.ply` aligns with `sonar_init_points.ply`.
- Poisson mesh tends to be overly dense/too much geometry everywhere.
- Legacy dataset is "fine-ish"; failures are much worse with R2.
- GUI had prior issues: mesh not loading (path/output mismatch) and dark scene; fixed by forcing repo-root output dir and adding brightness slider.
- Brightness slider was added very recently and needs re-verification.

## Current Knobs
- Env vars used:
  - `SONAR_OUTPUT_DIR=./output/poisson_tuner`
  - `SONAR_STAGE2_ITERS=<int>`
  - `POISSON_DEPTH=<int>`
  - `POISSON_DENSITY_QUANTILE=<float>`
  - `POISSON_MIN_OPACITY=<float>`
  - `POISSON_OPACITY_PERCENTILE=<float>`
  - `POISSON_SCALE_PERCENTILE=<float>`

## Next Actions
1. Re-verify GUI brightness slider behavior.
2. Continue R2 mesh quality investigation with GUI tuning.
3. Decide if/when to merge `debug-multiframe-r2` into `master` once R2 is working.
