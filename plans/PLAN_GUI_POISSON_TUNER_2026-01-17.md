# Plan: GUI Poisson Tuner Mode

**Date/Time:** 2026-01-17 04:22:14 CST  
**Git Commit:** d0f5b48d41b7492d3ea4372658a27f5581d2d1c7

## Goals
- Keep the current `debug_multiframe.py` workflow unchanged as “Mode 1”.
- Add a separate GUI-based “Mode 2” for fast Poisson parameter tuning.
- Use a fixed output folder: `./output/poisson_tuner/`.
- No extra image saving; only existing mesh outputs.

## Plan
1. Add minimal env overrides to `debug_multiframe.py` for the GUI mode:
   - `OUTPUT_DIR` override (fixed folder).
   - `STAGE2_ITERATIONS` override.
   - `POISSON_*` overrides.
2. Add a new GUI script (`scripts/poisson_tuner_gui.py`) using Open3D:
   - Sliders for Poisson params and Stage 2 iterations.
   - Manual “Run” button.
   - Load/display `mesh_poisson_after_stage3.ply` by default.
3. GUI runs `debug_multiframe.py` with env vars and refreshes the mesh.
4. Keep all existing outputs unchanged for direct CLI runs.

## Notes
- GUI mode is optional; direct CLI usage remains intact.
- Outputs always overwrite in `./output/poisson_tuner/` during GUI runs.
