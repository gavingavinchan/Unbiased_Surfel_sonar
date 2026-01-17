# Plan: R2 Scale Fix for debug_multiframe

**Date/Time:** 2026-01-17 02:10:28 CST  
**Git Commit:** 9f0ea511e2803f060a2bbf457415bbef5d1dded7

## Goals
- Make R2 dataset render/mesh scale consistent with metric sonar ranges.
- Keep fixes scoped to `debug_multiframe.py` dependencies.

## Warning
- The WIP commit (`37d763c`) contains partial fixes that **did not work in testing**. Treat all changes from that commit as untrusted and re-validate each change carefully.

## Plan
1. Audit current scale usage in `render_sonar`, `sonar_frame_to_points`, and FOV checks.
2. Apply consistent scaling so surfel positions and camera translations share the same scale.
3. Keep initial point generation, rendering, and visibility tests in the same metric/colmap scale.
4. Run a quick R2 debug_multiframe sanity check.
5. Adjust if mesh scale remains off.
