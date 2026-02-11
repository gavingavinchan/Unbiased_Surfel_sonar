# Plan: Elevation-Aware Chunk 1 Execution

**Date:** 2026-02-10  
**Status:** Draft for review (opus4.6)  
**Scope:** Chunk 1 only (safety rails + geometry contracts)

---

## Goal

Implement the Chunk 1 foundation in a way that is deterministic, testable, and low-risk before adding Stage 0/1 optimization behavior.

Primary outcome:
- Coordinate/sign conventions are enforced and visible at runtime.
- Projection/back-projection interfaces are explicit and stable.
- Range attenuation is integrated with diagnostics and default behavior aligned to raw sonar assumptions.

---

## Source of Truth

- Main implementation contract: `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md`
- Execution strategy: `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md`

If any mismatch appears, follow the detailed plan.

---

## In-Scope Files (Chunk 1)

- `utils/sonar_utils.py`
- `gaussian_renderer/__init__.py`
- `debug_multiframe.py`

Optional only if needed for clean config plumbing:
- `arguments/__init__.py`

Out of scope for Chunk 1:
- Pixel-bank/logit optimization
- Coupling loss and support pruning
- Normals ramp and Stage 2 densification

---

## Implementation Method (How)

1. Add contracts first, then wire usage.
   - Define `SonarProjection` typed return object.
   - Implement/standardize `sonar_project_points(...)` output fields including `valid`.
   - Implement/standardize `back_project_bins(...)` shape and convention behavior.

2. Enforce conventions with fail-fast checks.
   - Add assertion helpers for azimuth sign, elevation sign, and transform roundtrip.
   - Gate with `SONAR_CONVENTION_ASSERTS=1` default in debug path.
   - Fail early with actionable error text.

3. Integrate attenuation with deterministic precedence.
   - Implement stabilized attenuation in `render_sonar()`:
     - `I = lambert * gain / (max(r, r0)^p + eps)`
   - Respect precedence contract:
     - attenuation off -> ignore attenuation params,
     - attenuation on + auto-gain on -> auto mode,
     - attenuation on + auto-gain off -> manual gain.

4. Add run-header observability.
   - Print active convention summary.
   - Print active mount tuple/extrinsic constants.
   - Print attenuation mode and effective parameters.

5. Keep behavior gated and backward-safe.
   - New checks/diagnostics should be opt-out only via explicit flags.
   - No silent behavior changes outside Chunk 1 scope.

---

## Step-by-Step Work Order

1. Add `SonarProjection` and `sonar_project_points` contract-complete return path.
2. Add/align elevation-aware projection helpers and transform layout contract handling.
3. Add convention assertion utilities and startup invocation in debug path.
4. Add attenuation path + parameter precedence + diagnostics fields.
5. Add run-header logging for conventions/extrinsics/attenuation.
6. Run Chunk 1 validation gate.
7. Run checkpoint save/resume smoke for compatibility.
8. Record results and prepare for commit boundary.

---

## Validation Gate (Chunk 1)

All must pass:

1. Convention checks:
   - azimuth sign mapping correct (`left=+`, `right=-`),
   - elevation sign mapping correct (`+elev -> +Y`),
   - transform roundtrip consistency within tolerance.

2. Attenuation sanity:
   - with attenuation enabled, farther equal-lambert points produce lower intensity than nearer points.

3. Runtime stability:
   - no NaN/Inf introduced in sonar render path during short smoke run.

4. Resume compatibility (Chunk-level gate):
   - checkpoint save/load succeeds,
   - short continuation run starts without state-contract errors.

---

## Deliverables

- Code changes in Chunk 1 files.
- Clear startup logs showing active conventions and attenuation config.
- Passing gate results written in run notes.
- Ready-to-commit Chunk 1 boundary if all checks pass.

---

## Commit Boundary Rule

Chunk 1 is commit-ready only when:
- all Chunk 1 gate checks pass,
- resume smoke passes,
- no open blocker remains in Chunk 1 scope.

Commit message format (per CLAUDE.md / AGENTS.md convention):
- `<description> (<model-name>)` — use the name of the model that performed the implementation work (e.g., `opus4.6`, `gpt-5.3-codex`).

---

## Chunk 1 Test Catalog (Runnable Now)

These tests are in-scope for Chunk 1 only (contracts + assertions + attenuation + compatibility smoke), and do not require Stage 0/1 optimization logic.

1. **Syntax/Import Compile Gate**
   - Command:
     - `python -m py_compile utils/sonar_utils.py gaussian_renderer/__init__.py debug_multiframe.py`
   - Purpose: fail fast on syntax/runtime import breakage in Chunk 1 touched files.

2. **Convention Assertion Gate (Unit-level)**
   - Command (conda env):
     - run `run_sonar_convention_asserts(...)` with CPU SonarConfig and a sample camera transform.
   - Checks:
     - azimuth sign (`left=+`, `right=-`),
     - elevation sign (`+elev -> +Y`, `-elev -> -Y`),
     - transform roundtrip consistency,
     - row-major translation layout contract.

3. **`back_project_bins(...)` Contract Test**
   - Command (conda env):
     - call `back_project_bins(frame_idx, rows, cols, elev_bins, ...)` on a minimal dummy camera list.
   - Checks:
     - output shape is `[P, K, 3]`,
     - elevation sign behavior visible in Y output,
     - deterministic tensor/device behavior.

4. **`sonar_project_points(...)` Contract Test**
   - Command (conda env):
     - project known synthetic world points with identity camera transform.
   - Checks:
     - typed return fields exist,
     - azimuth sign direction is correct,
     - `valid = in_fov & in_front & in_bounds` behavior is consistent.

5. **Attenuation Sanity (Physics Check)**
   - Command (conda env):
     - call `compute_sonar_range_attenuation(...)` for near/far ranges.
   - Checks:
     - with attenuation ON, near attenuation factor > far attenuation factor,
     - no NaN/Inf in attenuation outputs.

6. **Attenuation Precedence Contract Test**
   - Command (conda env):
     - evaluate helper with:
       - attenuation OFF,
       - attenuation ON + auto gain ON,
       - attenuation ON + auto gain OFF.
   - Checks:
     - mode resolves to `off|auto|manual` correctly,
     - effective gain behaves per precedence spec.

7. **`render_sonar(...)` Runtime Stability Smoke (CPU micro-scene)**
   - Command (conda env):
     - run render on a small dummy PC + identity camera.
   - Checks:
     - render and range maps are finite,
     - `sonar_diagnostics.nan_inf_count == 0`,
     - diagnostics dictionary fields are populated.

8. **Debug Startup Header/Assertions Smoke**
   - Command:
     - short `debug_multiframe.py` startup execution with reduced workload settings.
   - Checks:
     - run header prints convention/extrinsic/attenuation summary,
     - `SONAR_CONVENTION_ASSERTS=1` triggers startup checks,
     - `back_project_bins` contract check prints pass status.

9. **Chunk 1 Gate: Short Training Stability Smoke**
   - Command:
     - short run (`SONAR_STAGE2_ITERS` small, fixed seed) to cover active sonar render path.
   - Checks:
     - no NaN/Inf in render path,
     - attenuation diagnostics remain finite,
     - startup checks pass.

10. **Checkpoint Save/Load Continuation Smoke (Compatibility)**
    - Command:
      - save checkpoint, reload, continue short run.
    - Checks:
      - state loads without contract errors,
      - continuation starts and executes sonar render path successfully.

---

## Tests Already Run and Passing (Current Implementation Session)

Recorded here for traceability. No additional tests were executed in response to this note request.

1. **Syntax compile passed**
   - `python -m py_compile utils/sonar_utils.py gaussian_renderer/__init__.py debug_multiframe.py`

2. **Convention asserts passed (CPU SonarConfig)**
   - `run_sonar_convention_asserts(...)` returned success for azimuth/elevation/roundtrip checks.

3. **Attenuation sanity passed**
   - `compute_sonar_range_attenuation(...)` confirmed near attenuation > far attenuation.

4. **`render_sonar` CPU smoke passed**
   - dummy scene render completed with finite outputs and `nan_inf_count = 0`.

5. **`back_project_bins` contract smoke passed**
   - output shape matched expected `[P, K, 3]` and elevation sign in Y was correct.

6. **`sonar_project_points` contract smoke passed**
   - typed return fields valid; azimuth sign direction matched convention.

7. **Attenuation precedence smoke passed**
   - helper produced correct `off/auto/manual` mode behavior and effective gain output.

### Not Yet Run (Still Pending)

- Full Chunk 1 runtime gate inside `debug_multiframe.py` short training run.
- Checkpoint save/load continuation smoke for Chunk-level compatibility.

---

## Additional Chunk 1 Tests (opus4.6 review, no Chunk 2 dependency)

These tests exercise Chunk 1 contracts more deeply than the unit-level checks above. They validate end-to-end geometric consistency and provide baseline data for later chunks.

11. **Projection Roundtrip Test**
     - For a grid of pixels across the sonar image, call `back_project_bins` → `sonar_project_points` and verify recovered `(row, col)` matches the original within tolerance for each elevation bin.
     - This is the strongest geometric consistency test — validates the entire forward/backward pipeline end-to-end, not just at a single synthetic point.
     - Checks:
      - evaluate error only on points marked `valid` by forward projection,
      - roundtrip pixel error < 0.5 px for all valid bins,
      - no systematic bias (mean error near zero),
      - p95 pixel error reported and within tolerance,
      - edge-of-FOV pixels also recover correctly.

12. **Elevation Arc Visualization (PLY export)**
    - Call `back_project_bins` on a sparse grid of pixels (e.g., 5x5 across the image) with `K=7` bins.
    - Export the `[P, K, 3]` points as a colored PLY (color by elevation bin index).
    - Visually confirm arcs fan out correctly in 3D and elevation spread matches the 20° FOV.
    - This is a one-time visual sanity check, not an automated gate.

13. **Surfel Visibility Census (Multi-View Overlap Baseline)**
    - Use `sonar_project_points` to project all current surfels into every training frame.
    - Compute per-surfel visibility count (how many frames each surfel is `valid` in).
    - Report: min/median/max visibility, histogram of visibility counts, percentage of surfels visible in 0/1/2-4/5+ frames.
    - Purpose: baseline data for tuning `ELEV_OVERLAP_*` defaults in Chunk 3 and validating that the dataset has sufficient multi-view coverage.

14. **Attenuation Intensity-vs-Range Profile**
     - Run a short training with attenuation ON (default) and render a few frames.
     - Plot rendered intensity vs range (scatter or binned mean) to verify `1/r²` falloff looks physical.
     - Compare with attenuation OFF to confirm the difference is visible and directionally correct.
     - Add a synthetic controlled-scene check (fixed lambertian/opacity setup) so attenuation behavior is unambiguous.
     - Purpose: visual/controlled confirmation that attenuation is meaningfully affecting the intensity model before committing to longer runs.

---

## Chunk 1 Execution Results (2026-02-11)

All tests in the Chunk 1 catalog were executed; results are summarized below.

### Pass Summary

- **Passed:** 14 / 14
- **Consolidated report:** `output/chunk1_tests/chunk1_all_results_2026-02-11.json`

### Result by Test ID

1. `T1` Syntax/Import Compile Gate — **PASS**
2. `T2` Convention Assertion Gate — **PASS**
3. `T3` `back_project_bins(...)` Contract — **PASS**
4. `T4` `sonar_project_points(...)` Contract — **PASS**
5. `T5` Attenuation Sanity — **PASS**
6. `T6` Attenuation Precedence Contract — **PASS**
7. `T7` `render_sonar(...)` Runtime Stability Smoke — **PASS**
8. `T8` Debug Startup Header/Assertions Smoke — **PASS**
9. `T9` Short Training Stability Smoke — **PASS**
10. `T10` Checkpoint Save/Load Continuation Smoke — **PASS**
11. `T11` Projection Roundtrip Test — **PASS**
12. `T12` Elevation Arc Visualization Export — **PASS**
13. `T13` Surfel Visibility Census — **PASS**
14. `T14` Attenuation Intensity-vs-Range Profile — **PASS**

### Artifacts and Logs

- Contract/unit harness summary: `output/chunk1_tests/chunk1_tests_2_7_11_14_summary.json`
- Visibility + attenuation profile summary: `output/chunk1_tests/chunk1_tests_13_14_summary.json`
- Startup/training smoke log: `output/chunk1_tests/test8_debug_startup.log`
- Exit-check smoke log: `output/chunk1_tests/test8_9_debug_exitcheck.log`
- Checkpoint compatibility summary: `output/chunk1_tests/test10_checkpoint_summary.json`
- Elevation arc visualization PLY: `output/chunk1_tests/test12_elevation_arcs.ply`

### Notes

- `debug_multiframe.py` logs show checks passed and run reached `COMPLETE`, but process exits with code `120` due an unraisable shutdown exception (`sys.unraisablehook`) after completion; this did not invalidate Chunk 1 gate signals in-log.
- For `T10`, a manual checkpoint capture/restore continuation smoke (GaussianModel `capture()`/`restore()`) was used because a direct short `train.py` sonar attempt hit an unrelated in-place autograd error before checkpoint write.
