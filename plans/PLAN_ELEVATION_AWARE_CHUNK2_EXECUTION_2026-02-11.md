# Plan: Elevation-Aware Chunk 2 Execution

**Date:** 2026-02-11  
**Status:** Draft for review (gpt-5.3-codex)  
**Scope:** Chunk 2 only (Stage 0 behavior)

---

## Goal

Implement Stage 0 behavior in a way that is deterministic, backward-safe, and easy to validate before Stage 1 likelihood logic is introduced.

Primary outcome:
- Elevation-aware initialization is active with `ELEV_INIT_MODE=random` as default.
- `ELEV_INIT_MODE=zero` remains available as the legacy-compatible fallback.
- `SONAR_FIXED_OPACITY=1` freezes opacity behavior in the sonar training path by default.
- Stage 0 default migration is explicit: `sonar_frame_to_points()` defaults to random elevation repo-wide, while callers pass explicit mode for readability and ablation control.

---

## Source of Truth

- Main implementation contract: `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md`
- Execution strategy and chunk gate: `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md`
- Chunk 1 reference style/results: `plans/PLAN_ELEVATION_AWARE_CHUNK1_EXECUTION_2026-02-10.md`

If a mismatch appears, follow the detailed plan first, then the execution plan.

---

## In-Scope Files (Chunk 2)

- `debug_multiframe.py`
- `utils/sonar_utils.py`

Optional only if needed for clean flag plumbing and defaults:
- `arguments/__init__.py`
- `debug_before_after_mesh.py` (explicit init-mode pass-through to keep behavior intentional under new default)

Out of scope for Chunk 2:
- Stage 1 overlap/pixel-bank/logit likelihood logic
- Coupling/support pruning and persistent surfel ID lifecycle
- Normals ramp and Stage 2 densification hooks

---

## Implementation Method (How)

1. Implement Stage 0 initialization modes as an explicit contract.
   - `ELEV_INIT_MODE=random` (default): sample elevation per valid pixel uniformly in `[-half_elevation_rad, +half_elevation_rad]`.
   - `ELEV_INIT_MODE=zero`: force zero-elevation initialization for ablation/legacy-like behavior.
   - Migration decision: helper-level default in `sonar_frame_to_points()` is random.
   - Call-site policy: pass mode explicitly at known initialization call sites to avoid silent behavior drift.

2. Keep Stage 0 deterministic and inspectable.
   - Respect fixed seeds where already present.
   - Use deterministic sampling ownership in random mode: seed RNG from `SEED` before Stage 0 init, then sample in deterministic valid-pixel traversal order.
   - Add startup logging for active init mode and summary stats (`y_mean`, `y_std`, min/max elevation sample).

3. Add fixed-opacity behavior in the sonar path with clear ownership.
   - `SONAR_FIXED_OPACITY=1` default in debug path.
   - Ensure opacity values are set to fixed mode target at initialization and excluded from further optimization updates.
   - Preserve opt-out path (`SONAR_FIXED_OPACITY=0`) for ablation.

4. Preserve backward safety.
   - Zero mode should pass the zero-mode legacy-parity contract (Point 2).
   - New behavior remains behind explicit flags and defaults from the detailed plan.

5. Keep checkpoint/resume compatibility intact.
   - No hidden state assumptions in Stage 0 setup.
   - Save/load continuation remains stable with both init modes.

---

## Step-by-Step Work Order

1. Add or finalize env/config parsing for `ELEV_INIT_MODE` and `SONAR_FIXED_OPACITY` in one centralized runtime config section.
2. Implement Stage 0 elevation initialization branch (`random` and `zero`) at the initialization entrypoint used by `debug_multiframe.py`.
3. Update known initialization call sites to pass explicit mode under the new helper default (at minimum `debug_multiframe.py`; extend to other debug init scripts if touched in this chunk).
4. Add lightweight Stage 0 diagnostics (mode and Y-spread summary) to run header/startup logs.
5. Implement fixed-opacity toggle in the sonar training setup:
    - force fixed opacity values in fixed mode,
    - freeze/exclude opacity parameter updates,
    - log effective opacity mode.
6. Add small contract checks for invalid `ELEV_INIT_MODE` values (fail fast with actionable message).
7. Run Chunk 2 validation gate tests.
8. Run checkpoint save/resume continuation smoke for Stage 0 config combinations.
9. Record outcomes and prepare commit boundary only after gate pass.

---

## Validation Gate (Chunk 2)

All must pass:

1. Init-only smoke run succeeds.
2. `ELEV_INIT_MODE=random` shows non-zero elevation/Y spread.
3. `ELEV_INIT_MODE=zero` passes the zero-mode legacy-parity contract (Point 2).
4. Fixed-opacity mode confirms opacity freeze contract (finite near-1 target, optimizer-structure-safe freeze, no drift).
5. Runtime stability: no NaN/Inf introduced during short Stage 0 startup/smoke run.
6. Resume compatibility (chunk-level gate): checkpoint save/load + short continuation succeeds.
7. Invalid-mode fail-fast works: invalid `ELEV_INIT_MODE` exits non-zero with actionable message and no silent fallback.

---

## Deliverables

- Stage 0 init mode implementation in Chunk 2 files.
- Fixed-opacity mode implementation and startup observability.
- Passing gate results with notes/artifacts for random vs zero and fixed-opacity on/off checks.
- Ready-to-commit Chunk 2 boundary if all checks pass.

---

## Commit Boundary Rule

Chunk 2 is commit-ready only when:
- all Chunk 2 gate checks pass,
- resume smoke passes,
- no open blocker remains in Chunk 2 scope.

Commit message format (per repo convention):
- `<description> (<model-name>)`.

---

## Chunk 2 Test Catalog (Planned)

These tests target Stage 0 behavior only and intentionally avoid Stage 1/2 logic.

1. **Syntax/Import Compile Gate**
   - Command:
     - `python -m py_compile utils/sonar_utils.py debug_multiframe.py`
   - Purpose: fail fast on syntax/import breakage in Chunk 2 touched files.

2. **Config Default Contract Test**
    - Check defaults resolve to:
      - `ELEV_INIT_MODE=random`
      - `SONAR_FIXED_OPACITY=1`
    - Purpose: enforce Stage 0 default behavior contract.

2a. **Helper Default Migration Contract Test**
   - Call `sonar_frame_to_points()` without explicit `elevation_mode`.
   - Checks:
     - default path uses random elevation sampling,
     - output shows non-zero Y spread under fixed seed,
     - explicit `elevation_mode=zero` still restores zero-mode behavior.

2b. **Invalid `ELEV_INIT_MODE` Fail-Fast Test**
   - Set `ELEV_INIT_MODE=invalid_value`.
   - Checks:
     - script exits with non-zero status,
     - actionable error message is printed,
     - no partial initialization or silent fallback occurs.

3. **Random Init Spread Contract Test**
   - Run Stage 0 init with `ELEV_INIT_MODE=random` and fixed seed.
   - Checks:
     - non-trivial Y spread (`std(Y) > 0`),
     - both sign directions present when expected by beam range,
     - finite outputs only.

4. **Zero Init Legacy-Parity Contract Test**
   - Run Stage 0 init with `ELEV_INIT_MODE=zero`.
   - Checks:
      - initialization points satisfy the zero-mode legacy-parity contract,
      - no extra elevation spread is introduced.

5. **Init Determinism Test (Random Mode)**
   - Same seed => same initialized points/statistics.
   - Different seed => changed random-elevation assignment.

6. **Fixed-Opacity Freeze Contract Test (ON)**
   - Run setup with `SONAR_FIXED_OPACITY=1`.
   - Checks:
      - opacity tensor is set to the fixed target (finite near-1 activated value),
      - opacity optimizer group remains structurally present and uses freeze policy,
      - one optimization step does not drift opacity values.

7. **Opacity Learnability Contract Test (OFF)**
   - Run setup with `SONAR_FIXED_OPACITY=0`.
   - Checks:
     - opacity remains trainable,
     - optimizer can update opacity in a controlled micro-step.

8. **Init-Only Runtime Smoke (Random/Zero)**
   - Two short runs with reduced workload:
     - run A: `ELEV_INIT_MODE=random`
     - run B: `ELEV_INIT_MODE=zero`
   - Checks:
     - startup completes,
     - init diagnostics printed,
     - no NaN/Inf in early render/loss values.

9. **Chunk 2 Gate: Fixed-Opacity Runtime Smoke**
   - Short run with `SONAR_FIXED_OPACITY=1`.
   - Checks:
     - training executes,
     - logged opacity mode is fixed,
     - opacity values remain stable during the smoke window.

10. **Checkpoint Save/Load Continuation Smoke**
    - Save checkpoint after short run, reload, continue for short window.
    - Execute for at least one Stage 0 config pair (`random+fixed`, `zero+fixed`).
    - Checks:
      - no resume contract break,
      - continuation starts cleanly.

---

## Visual Review Tests (User-Requested Manual Checks)

These are included because visual/3D quality cannot be fully validated automatically.

11. **Stage 0 Init Geometry Visualization (Random vs Zero)**
    - Export two PLY point sets from identical frame/seed context:
      - `random` init,
      - `zero` init.
    - Suggested artifact paths:
      - `output/chunk2_tests/test11_init_random.ply`
      - `output/chunk2_tests/test11_init_zero.ply`
    - Manual checklist:
      - random init shows visible elevation thickness/arc spread,
      - zero init collapses near the zero-elevation sheet,
      - no mirrored/sign-flipped geometry artifacts.

12. **Early Render Comparison Panel (Mode Ablation)**
    - Save matched rendered frames (same seed/frame IDs) for:
      - `random + fixed-opacity`,
      - `zero + fixed-opacity`,
      - optional `random + learnable-opacity` ablation.
    - Suggested artifact paths:
      - `output/chunk2_tests/test12_render_random_fixed.png`
      - `output/chunk2_tests/test12_render_zero_fixed.png`
      - `output/chunk2_tests/test12_render_random_learnable.png`
    - Manual checklist:
      - random mode does not introduce obvious geometric tearing,
      - zero mode remains legacy-like,
      - fixed-opacity mode does not create obvious instability in first-iteration render.

---

## Notes

- Visual tests (11-12) are manual-review-required evidence for Chunk 2 acceptance; pass/fail is decided by human reviewer verdict and must be recorded explicitly.
- Automated gates remain mandatory; visual checks complement them for geometry sanity.

---

## End-of-Plan Clarifications (2026-02-11)

### Point 1: Visual Test Authority and Pass/Fail Ownership

- Visual tests 11-12 are executed by the implementation agent, but final pass/fail is decided by the human reviewer.
- Reviewer verdict is authoritative for these manual checks.
- Record verdict explicitly in run notes, for example:
  - `Visual review verdict: T11=PASS, T12=FAIL (reason: <brief reason>)`
- Chunk 2 acceptance rule for visual checks:
  - if reviewer marks either test as FAIL, Chunk 2 is not gate-passed/commit-ready,
  - if reviewer marks required visual tests as PASS, visual evidence requirement is satisfied.

### Point 2: Zero-Mode Legacy-Parity Contract (Suggested)

To replace vague "legacy-like" wording with objective thresholds, use this contract for `ELEV_INIT_MODE=zero`:

1. Reference setup
   - Fixed seed (`SEED=42`), fixed frame subset, fixed thresholds.
   - Build a one-time legacy zero reference artifact before Chunk 2 changes.

2. Pass criteria (`ELEV_INIT_MODE=zero` vs legacy reference)
   - Point count must match exactly.
   - `y_abs_max <= 1e-6` and `y_std <= 1e-7`.
   - `mean_l2(new_xyz, legacy_xyz) <= 1e-5` meters.
   - `p99_l2(new_xyz, legacy_xyz) <= 1e-4` meters.
   - No NaN/Inf values.

3. Fallback when point ordering differs
   - If exact point-wise ordering is not stable, compare invariant stats instead:
     - identical point count,
     - range distribution deltas within tolerance (example: `|mean_range_delta| <= 1e-4`, `|p99_range_delta| <= 1e-3`),
     - near/far min/max consistency within tolerance.

4. Gate wording recommendation
   - Replace "legacy-like" and "parity-close" with "passes zero-mode legacy-parity contract".
   - Treat this as hard pass/fail in Chunk 2 gate reporting.

### Point 3: Implementation Specificity Gaps (opus4.6 review, 2026-02-11)

The plan is consistent with the source-of-truth documents (detailed plan, execution plan, Chunk 1 reference) on scope, defaults, and gates. The following gaps should be resolved before coding begins.

#### 3a. Elevation-aware initialization mechanism

`sonar_frame_to_points()` in `utils/sonar_utils.py` currently hardcodes `elevation = 0`. The plan should specify:

- Extend `sonar_frame_to_points()` with an `elevation_mode` parameter (`"zero"` | `"random"`).
- `"zero"` mode: keep `y_cam = 0` (current behavior, no change).
- `"random"` mode: sample elevation per valid pixel uniformly from `[-half_elevation_rad, +half_elevation_rad]`, then compute `y_cam = range * sin(elevation)` and adjust `x_cam`/`z_cam` by `cos(elevation)`.
- Migration rule: helper default is `"random"` for future runs; call sites should still pass explicit mode for intent clarity.
- Wire `ELEV_INIT_MODE` env var in `debug_multiframe.py` (around the initialization loop at ~line 1012) and pass it through to `sonar_frame_to_points()`.

#### 3b. Fixed-opacity freezing mechanism

"Freeze/exclude opacity parameter updates" (step 4, line 77) needs a concrete mechanism:

- After `gaussians.training_setup()` in `debug_multiframe.py`, when `SONAR_FIXED_OPACITY=1`:
  - Set `gaussians._opacity.data` to a finite near-1 target in activated space (example: `inverse_sigmoid(0.999)`).
  - Set `gaussians._opacity.requires_grad = False`.
  - Keep optimizer group structure intact (including `opacity`) and enforce freeze via optimizer/grad policy (for example opacity LR = 0 and no grad updates).
  - Re-apply freeze policy after prune/densify/optimizer-rebuild events that recreate parameters.
- When `SONAR_FIXED_OPACITY=0`: leave opacity in optimizer as normal (learnable).

#### 3c. Gate 4 (fixed-opacity freeze) concrete checks

Replace the vague "no optimizer-driven drift" criterion with:

- `gaussians._opacity.requires_grad == False` when fixed mode is on.
- After N training iterations (e.g., 10), `max(abs(opacity_tN - opacity_t0)) < 1e-6`.
- `opacity` group remains present (repo-compatibility) and has freeze policy active (for example LR = 0).

#### 3d. Diagnostic logging format

Step 3 (startup diagnostics) should log the following at initialization:

```
[Stage 0] ELEV_INIT_MODE={mode}, SONAR_FIXED_OPACITY={value}
[Stage 0] Init points: N={count}, Y mean={y_mean:.4f}, std={y_std:.4f}, range=[{y_min:.4f}, {y_max:.4f}]
[Stage 0] Opacity mode: {"FIXED (target=0.999)" | "LEARNABLE"}
```

For `ELEV_INIT_MODE=zero`, the Y stats should confirm near-zero spread (`std < 1e-6`).

#### 3e. Env-var parsing and fail-fast (resolved in catalog)

Step 5 of the work order requires fail-fast on invalid `ELEV_INIT_MODE` values. This is now covered by Test 2b in the main test catalog:

```
Test 2b: Invalid ELEV_INIT_MODE Fail-Fast Test
- Set ELEV_INIT_MODE=invalid_value
- Confirm: script exits with non-zero status and prints an actionable error message
- Confirm: no partial initialization or silent fallback occurs
```

#### 3f. Fixed-opacity implementation safety note (gpt-5.3-codex review, 2026-02-11)

Resolved by 3b authoritative mechanism above.

Safety rationale retained for traceability:

- Do not use `inverse_sigmoid(1.0)` directly; this is singular (`log(x/(1-x))`) and yields `+inf` at `x=1.0`.
  - Reference: `utils/general_utils.py` (`inverse_sigmoid`).
- Avoid removing the optimizer `opacity` param group entirely in this repo flow.
  - `GaussianModel.prune_points()` and optimizer-maintenance paths assume an `opacity` group exists and rebind tensors by group name.
  - References: `scene/gaussian_model.py` (`prune_points`, `_prune_optimizer`, `densification_postfix`).
- One-time `requires_grad=False` is not enough if topology edits occur.
  - Prune/rebuild paths recreate parameters with `requires_grad_(True)`, which can silently unfreeze opacity.

No additional contract text is needed here; 3b/3c are now the source for fixed-opacity behavior and gate checks.

### Point 4: Pre-Implementation Cleanup Status (Resolved)

- 4a resolved: fixed-opacity mechanism conflict removed (3b is authoritative, 3f is rationale-only).
- 4b resolved: Gate/check text now assumes optimizer structure is kept and freeze/no-drift is validated.
- 4c resolved: `Test 2b` promoted into the main test catalog.
- 4d resolved: visual tests are now explicitly manual-review-required acceptance evidence.
- 4e resolved: deterministic random-init sampling rule documented in implementation method.

---

## Point 5: Chunk 2 Execution Log + Handoff (gpt-5.3-codex, 2026-02-11)

### 5a. Implemented code changes (completed)

- `utils/sonar_utils.py`
  - `sonar_frame_to_points()` now supports `elevation_mode` (`random`/`zero`), deterministic `rng`, and `return_debug` diagnostics.
  - Helper default migrated to `elevation_mode="random"`.
  - Added fail-fast validation for invalid mode values.
  - `sonar_frames_to_point_cloud()` now passes through elevation mode/rng.
- `debug_multiframe.py`
  - Added centralized env parsing and defaults:
    - `ELEV_INIT_MODE` default `random` (validated)
    - `SONAR_FIXED_OPACITY` default `1`
    - `SONAR_LOAD_CHECKPOINT`, `SONAR_SAVE_CHECKPOINT`
    - `SONAR_OPACITY_WARMUP_ITERS`
    - `SONAR_STAGE3_ITERS`
  - Stage 0 init now explicitly passes `ELEV_INIT_MODE`, uses deterministic RNG seeded from `SEED`, and logs required diagnostics (`N`, `Y mean/std/min/max`, elevation min/max).
  - Added zero-mode contract guard (`std/mean` near zero for sonar-frame Y).
  - Implemented fixed-opacity policy (`inverse_sigmoid(0.999)`, `requires_grad=False`, opacity group LR=0, group kept intact).
  - Added policy re-application hooks after prune/rebind events.
  - Added checkpoint save/load helpers and resume wiring (`gaussians.capture/restore`, scale module + optimizer states).
  - Added learnable-opacity warm-start behavior:
    - if `SONAR_FIXED_OPACITY=0`, opacity can remain fixed for first `SONAR_OPACITY_WARMUP_ITERS`, then auto-switch to learnable.
  - Added auto attenuation default for learnable-opacity mode when user does not explicitly set `SONAR_RANGE_ATTEN_AUTO_GAIN`.
  - Raw comparison export now disables auto-gain to keep raw output faithful and avoid inflated brightness artifacts.
  - Moved `after_stage3` comparison exports to pre-final-prune state so image quality review is not biased by cleanup prune.
  - Added warning when selected frame count exceeds total training iterations (`Stage2+Stage3 < NumFrames`).
- `gaussian_renderer/__init__.py`
  - Sonar render visibility switched to center-based validity (`projection.valid`) to avoid boundary over-suppression in rendered images.
- `debug_before_after_mesh.py`
  - Added explicit `ELEV_INIT_MODE` parse/pass-through and validation for intentional behavior under new helper default.

### 5b. Automated validation status (executed)

- Compile gate passed for touched files (`py_compile`).
- Init contracts passed:
  - random mode has non-zero spread,
  - zero mode remains near-zero spread,
  - helper default path uses random,
  - deterministic under same seed / changes with different seed.
- Fail-fast contracts passed:
  - invalid `ELEV_INIT_MODE` exits non-zero with actionable message,
  - invalid helper mode raises `ValueError`.
- Fixed-opacity contracts passed via micro-tests and runtime:
  - no drift under fixed mode,
  - opacity group retained,
  - learnable mode remains trainable when enabled.
- Checkpoint/resume smoke passed for required config pairs:
  - `random+fixed` save/load/continue,
  - `zero+fixed` save/load/continue.

### 5c. Manual visual review record (reviewer verdicts)

Tag-to-artifact map used in review:

- `1A`: `output/chunk2_smoke_random_fixed/sonar_init_points.ply`
- `1B`: `output/chunk2_smoke_zero_fixed/sonar_init_points.ply`
- `1C`: `output/chunk2_smoke_random_fixed/comparison_after_stage3_frame0.png`
- `1D`: `output/chunk2_smoke_zero_fixed/comparison_after_stage3_frame0.png`
- `1E`: `output/chunk2_smoke_random_learnable/comparison_after_stage3_frame0.png`
- `3A`: `output/chunk2_smoke_random_fixed_v3/comparison_after_stage3_frame0.png`
- `3B`: `output/chunk2_smoke_zero_fixed_v3/comparison_after_stage3_frame0.png`
- `3C`: `output/chunk2_smoke_random_learnable_v3/comparison_after_stage3_raw_frame0.png`
- `3D`: `output/chunk2_smoke_random_learnable_v3/comparison_after_stage3_frame0.png`
- `6A`: `output/chunk2_multiframe_random_learnable_warmup2_auto_v2/comparison_after_stage3_raw_frame0.png`
- `6B`: `output/chunk2_multiframe_random_learnable_warmup2_auto_v2/comparison_after_stage3_frame0.png`
- `6C`: `output/chunk2_multiframe_random_fixed_regression/comparison_after_stage3_frame0.png`

- `T11`: PASS
  - `1A`/`1B` reviewer note: zero-mode is random-mode geometry projected to elevation zero (expected).
- `T12` required views:
  - initial run: FAIL/marginal due to bottom-row dropout and sparsity,
  - after renderer/export fixes:
    - `3A`: PASS (conditional; sparse spots only at very bottom row),
    - `3B`: PASS (conditional; similar to `3A`).
- Optional ablation view (`random + learnable`, raw):
  - `3C`: FAIL,
  - `3D`: improved.
- Additional reviewer notes (later pass):
  - `6A`: too bright,
  - `6B`: acceptable (expected brightened),
  - `6C`: matches original brightness baseline,
  - non-frame0 views in both folders remain poor; reviewer did not yet review `.ply` manually.

### 5d. Multi-frame evaluation correction (important)

- Early visual checks used short/1-frame-like smoke settings and were not sufficient for multi-frame behavior claims.
- Additional true multi-frame runs were executed (`SONAR_NUM_FRAMES=8`) with longer Stage2/Stage3 windows.
- Observed behavior: frame 0 is often best; other frames remain weak in both fixed and learnable variants under current setup.

### 5e. Clarified evaluation objective (agreed)

- Distinguish two targets:
  1. per-frame fidelity (single-frame match can be very good),
  2. cross-view consistency (true multi-frame goal; often harder and may reduce single-frame fit quality).
- End-goal requires overlap for 3D consistency, but good single-frame reprojection does not imply good multi-frame consistency.

### 5f. Recommended target-2 (cross-view consistency) tracking for next session

- Add holdout-view evaluation (train subset A, report loss/SSIM on unseen subset B).
- Add surfel multi-view support metrics (`support>=2`, `support>=3`, median support).
- Add per-frame final loss table + variance to detect frame dominance.
- Add optional surfel dominance metric (single-frame ownership concentration).

### 5g. Artifacts and run folders to hand off

- Chunk 2 smoke / visual artifacts:
  - `output/chunk2_smoke_random_fixed/`
  - `output/chunk2_smoke_zero_fixed/`
  - `output/chunk2_smoke_random_learnable/`
  - `output/chunk2_smoke_random_fixed_v3/`
  - `output/chunk2_smoke_zero_fixed_v3/`
  - `output/chunk2_smoke_random_learnable_v3/`
- Checkpoint/resume artifacts:
  - `output/chunk2_resume/random_fixed_ckpt.pth`
  - `output/chunk2_resume/zero_fixed_ckpt.pth`
  - logs under `output/chunk2_resume/*/run.log`
- Multi-frame follow-up runs:
  - `output/chunk2_multiframe_random_learnable_warmup4_auto_v3/`
  - `output/chunk2_multiframe_random_fixed_v3/`

### 5h. Current gate-readiness summary

- Chunk 2 core implementation and automated gates: substantially complete.
- Manual visuals: required checks accepted with conditions; optional learnable ablation still problematic.
- Remaining risk for next session: robust cross-view consistency quality across non-frame0 views.

### 5h.1 Outstanding manual review items (not yet completed)

- `.ply` manual quality verdict is still pending (reviewer explicitly noted it was not checked yet).
- Pending `.ply` review targets (minimum):
  - `output/chunk2_multiframe_random_learnable_warmup4_auto_v3/sonar_init_points.ply`
  - `output/chunk2_multiframe_random_learnable_warmup4_auto_v3/mesh_poisson_after_stage3.ply`
  - `output/chunk2_multiframe_random_fixed_v3/sonar_init_points.ply`
  - `output/chunk2_multiframe_random_fixed_v3/mesh_poisson_after_stage3.ply`

### 5i. Session state snapshot for takeover

- Branch: `debug-multiframe-r2`
- Last committed checkpoint in this session:
  - `63deba1` `WIP chunk2 stage-0 elevation init and fixed-opacity policy (gpt-5.3-codex)`
- Current uncommitted files at handoff:
  - `debug_multiframe.py`
  - `gaussian_renderer/__init__.py`
  - `plans/PLAN_ELEVATION_AWARE_CHUNK2_EXECUTION_2026-02-11.md`
- Repro commands used for later-stage multi-frame checks:
  - learnable+warmup+auto (8 frames):
    - `SONAR_OUTPUT_DIR=./output/chunk2_multiframe_random_learnable_warmup4_auto_v3 SONAR_NUM_FRAMES=8 SONAR_STAGE2_ITERS=24 SONAR_STAGE3_ITERS=8 ELEV_INIT_MODE=random SONAR_FIXED_OPACITY=0 SONAR_OPACITY_WARMUP_ITERS=4 python debug_multiframe.py`
  - fixed baseline (8 frames):
    - `SONAR_OUTPUT_DIR=./output/chunk2_multiframe_random_fixed_v3 SONAR_NUM_FRAMES=8 SONAR_STAGE2_ITERS=24 SONAR_STAGE3_ITERS=8 ELEV_INIT_MODE=random SONAR_FIXED_OPACITY=1 python debug_multiframe.py`
- Known recurring runtime note:
  - occasional `Exception ignored in sys.unraisablehook` at process teardown; non-blocking in completed runs, but worth cleanup if reproducibility tooling is tightened.

---

## Point 6: Continuation Worklog + Newly Raised Issues (gpt-5.3-codex, 2026-02-11)

### 6a. Continuation implementation (in progress)

- `debug_multiframe.py`
  - Added cross-view tracking instrumentation from Point 5f recommendations:
    - optional holdout split via `SONAR_HOLDOUT_FRAMES`,
    - final per-frame evaluation CSVs (`final_eval_train_frames.csv`, `final_eval_holdout_frames.csv`),
    - frame coverage/visit accounting (`frame_training_visits.csv`),
    - surfel multi-view support diagnostics (`support_metrics_train.csv`, `support_metrics_train_plus_holdout.csv`).
  - Added final run summary logging for:
    - train vs holdout loss/SSIM gap,
    - support fractions (`support>=2`, `support>=3`),
    - simple dominance proxies (`single_view_top_share`, `nearest_owner_top_share`).
  - Fixed checkpoint continuation accounting bug:
    - final checkpoint save now stores `iteration=training_iter_offset + total_iters` instead of `total_iters` only.

### 6b. Issues raised during continuation

1. **Generalization risk remains likely unresolved**
   - Existing notes already show non-frame0 weakness; new holdout diagnostics are now instrumented but not yet fully re-baselined across representative multi-frame runs.

2. **Resume semantics are still coarse-grained**
   - Checkpoint restore now preserves Gaussian + scale states and iteration offset, but does not restore stage-local scheduler context beyond the scalar offset.
   - For strict stage-resume fidelity, explicit stage cursor/state restoration may still be needed in a future chunk.

3. **`sys.unraisablehook` teardown warning remains open**
   - Still seen intermittently at process shutdown; no root-cause fix has been implemented yet.

### 6c. Holdout re-baseline runs executed (8 train + 2 holdout)

Run A (fixed opacity baseline):

- Command:
  - `SONAR_OUTPUT_DIR=./output/chunk2_multiframe_random_fixed_holdout_v1 SONAR_DATASET=r2 SONAR_NUM_FRAMES=8 SONAR_HOLDOUT_FRAMES=2 SONAR_STAGE2_ITERS=24 SONAR_STAGE3_ITERS=8 ELEV_INIT_MODE=random SONAR_FIXED_OPACITY=1 python debug_multiframe.py`
- Key results:
  - train: `loss_mean=0.034048`, `ssim_mean=0.8679`
  - holdout: `loss_mean=0.048536`, `ssim_mean=0.8603`
  - gap: `loss_gap=+0.014487` (holdout-train), `ssim_gap=+0.0076` (train-holdout)
  - support(train): `support>=2=0.4614`, `support>=3=0.0449`, `median=1.0`
  - dominance proxy: `single_view_top_share=0.5588` (train), `0.7249` (train+holdout)

Run B (learnable opacity + warmup):

- Command:
  - `SONAR_OUTPUT_DIR=./output/chunk2_multiframe_random_learnable_holdout_v1 SONAR_DATASET=r2 SONAR_NUM_FRAMES=8 SONAR_HOLDOUT_FRAMES=2 SONAR_STAGE2_ITERS=24 SONAR_STAGE3_ITERS=8 ELEV_INIT_MODE=random SONAR_FIXED_OPACITY=0 SONAR_OPACITY_WARMUP_ITERS=4 python debug_multiframe.py`
- Key results:
  - train: `loss_mean=0.045582`, `ssim_mean=0.8332`
  - holdout: `loss_mean=0.059386`, `ssim_mean=0.8454`
  - gap: `loss_gap=+0.013804` (holdout-train), `ssim_gap=-0.0122` (train-holdout)
  - support(train): `support>=2=0.4625`, `support>=3=0.0444`, `median=1.0`
  - dominance proxy: `single_view_top_share=0.5607` (train), `0.7257` (train+holdout)

Artifacts:

- `output/chunk2_multiframe_random_fixed_holdout_v1/`
- `output/chunk2_multiframe_random_learnable_holdout_v1/`

### 6d. Additional issues raised from holdout re-baseline

1. **Cross-view support remains shallow**
   - In both modes, only ~4.4-4.5% of surfels have `support>=3` and median support is 1, indicating weak multi-view reinforcement.

2. **Frame ownership concentration remains high**
   - `single_view_top_share` is ~0.56 on train frames and ~0.73 when including holdout frames, suggesting persistent frame dominance.

3. **Learnable-opacity ablation underperforms fixed baseline at this budget**
   - Higher train loss and lower train SSIM than fixed mode at identical iteration budget.

4. **TSDF mesh extraction path still returns empty meshes (`V=0`) in this setup**
   - Open3D warns during writes for `mesh_before_training.ply`, `mesh_after_stage2.ply`, and `mesh_after_stage3.ply`; Poisson meshes are still produced.
   - Scope note: this is tracked as a non-blocking issue for Chunk 2 gate decisions because Chunk 2 acceptance is Stage-0/training-behavior focused.

### 6e. Higher-budget holdout follow-up (next-step execution)

Executed a higher-budget rerun to test whether the 8+2 holdout gaps/support improve with more optimization:

- Budget:
  - `SONAR_STAGE2_ITERS=96`, `SONAR_STAGE3_ITERS=32` (vs prior 24/8)

Run C (fixed opacity, higher budget):

- Command:
  - `SONAR_OUTPUT_DIR=./output/chunk2_multiframe_random_fixed_holdout_hb1 SONAR_DATASET=r2 SONAR_NUM_FRAMES=8 SONAR_HOLDOUT_FRAMES=2 SONAR_STAGE2_ITERS=96 SONAR_STAGE3_ITERS=32 ELEV_INIT_MODE=random SONAR_FIXED_OPACITY=1 python debug_multiframe.py`
- Key results:
  - train: `loss_mean=0.032021`, `ssim_mean=0.8775`
  - holdout: `loss_mean=0.048669`, `ssim_mean=0.8601`
  - gap: `loss_gap=+0.016648` (`~1.52x` holdout/train), `ssim_gap=+0.0175`
  - support(train): `support>=2=0.4615`, `support>=3=0.0446`, `median=1.0`

Run D (learnable opacity + warmup, higher budget):

- Command:
  - `SONAR_OUTPUT_DIR=./output/chunk2_multiframe_random_learnable_holdout_hb1 SONAR_DATASET=r2 SONAR_NUM_FRAMES=8 SONAR_HOLDOUT_FRAMES=2 SONAR_STAGE2_ITERS=96 SONAR_STAGE3_ITERS=32 ELEV_INIT_MODE=random SONAR_FIXED_OPACITY=0 SONAR_OPACITY_WARMUP_ITERS=16 python debug_multiframe.py`
- Key results:
  - train: `loss_mean=0.039490`, `ssim_mean=0.8455`
  - holdout: `loss_mean=0.060202`, `ssim_mean=0.8448`
  - gap: `loss_gap=+0.020712` (`~1.52x` holdout/train), `ssim_gap=+0.0006`
  - support(train): `support>=2=0.4630`, `support>=3=0.0439`, `median=1.0`

Artifacts:

- `output/chunk2_multiframe_random_fixed_holdout_hb1/`
- `output/chunk2_multiframe_random_learnable_holdout_hb1/`

### 6f. Issue update after higher-budget run

- Increasing iteration budget improved in-set train metrics modestly, but **did not improve cross-view support depth** (`support>=3` still ~4.4%) and **did not reduce holdout ratio risk** (still ~1.5x).
- Learnable-opacity mode remains weaker than fixed-opacity mode under both short and higher budgets.

### 6g. Manual review tag map (surfel/training focused)

For concise reviewer references (no mesh-gate dependency for Chunk 2):

- `HBF-I`: `output/chunk2_multiframe_random_fixed_holdout_hb1/sonar_init_points.ply`
- `HBF-S`: `output/chunk2_multiframe_random_fixed_holdout_hb1/surfels_after_training.ply`
- `HBF-F2`: `output/chunk2_multiframe_random_fixed_holdout_hb1/comparison_after_stage3_frame2.png`
- `HBF-F3`: `output/chunk2_multiframe_random_fixed_holdout_hb1/comparison_after_stage3_frame3.png`
- `HBF-R2`: `output/chunk2_multiframe_random_fixed_holdout_hb1/comparison_after_stage3_raw_frame2.png`
- `HBF-R3`: `output/chunk2_multiframe_random_fixed_holdout_hb1/comparison_after_stage3_raw_frame3.png`

- `HBL-I`: `output/chunk2_multiframe_random_learnable_holdout_hb1/sonar_init_points.ply`
- `HBL-S`: `output/chunk2_multiframe_random_learnable_holdout_hb1/surfels_after_training.ply`
- `HBL-F2`: `output/chunk2_multiframe_random_learnable_holdout_hb1/comparison_after_stage3_frame2.png`
- `HBL-F3`: `output/chunk2_multiframe_random_learnable_holdout_hb1/comparison_after_stage3_frame3.png`
- `HBL-R2`: `output/chunk2_multiframe_random_learnable_holdout_hb1/comparison_after_stage3_raw_frame2.png`
- `HBL-R3`: `output/chunk2_multiframe_random_learnable_holdout_hb1/comparison_after_stage3_raw_frame3.png`

- `HBF-TCSV`: `output/chunk2_multiframe_random_fixed_holdout_hb1/final_eval_train_frames.csv`
- `HBF-HCSV`: `output/chunk2_multiframe_random_fixed_holdout_hb1/final_eval_holdout_frames.csv`
- `HBF-SUP`: `output/chunk2_multiframe_random_fixed_holdout_hb1/support_metrics_train.csv`
- `HBF-VIS`: `output/chunk2_multiframe_random_fixed_holdout_hb1/frame_training_visits.csv`

- `HBL-TCSV`: `output/chunk2_multiframe_random_learnable_holdout_hb1/final_eval_train_frames.csv`
- `HBL-HCSV`: `output/chunk2_multiframe_random_learnable_holdout_hb1/final_eval_holdout_frames.csv`
- `HBL-SUP`: `output/chunk2_multiframe_random_learnable_holdout_hb1/support_metrics_train.csv`
- `HBL-VIS`: `output/chunk2_multiframe_random_learnable_holdout_hb1/frame_training_visits.csv`

### 6h. Expanded review checklist tags (everything to check)

Use this full tag set for reviewer notes.

Geometry (required):

- `HBF-I`: `output/chunk2_multiframe_random_fixed_holdout_hb1/sonar_init_points.ply`
- `HBF-S`: `output/chunk2_multiframe_random_fixed_holdout_hb1/surfels_after_training.ply`
- `HBL-I`: `output/chunk2_multiframe_random_learnable_holdout_hb1/sonar_init_points.ply`
- `HBL-S`: `output/chunk2_multiframe_random_learnable_holdout_hb1/surfels_after_training.ply`

Rendered comparisons (all training frames, stage3):

- `HBF-F0`..`HBF-F7`: `output/chunk2_multiframe_random_fixed_holdout_hb1/comparison_after_stage3_frame{0..7}.png`
- `HBL-F0`..`HBL-F7`: `output/chunk2_multiframe_random_learnable_holdout_hb1/comparison_after_stage3_frame{0..7}.png`

Raw-vs-render comparisons (all training frames, stage3 raw):

- `HBF-R0`..`HBF-R7`: `output/chunk2_multiframe_random_fixed_holdout_hb1/comparison_after_stage3_raw_frame{0..7}.png`
- `HBL-R0`..`HBL-R7`: `output/chunk2_multiframe_random_learnable_holdout_hb1/comparison_after_stage3_raw_frame{0..7}.png`

CSV diagnostics (required):

- `HBF-TCSV`: `output/chunk2_multiframe_random_fixed_holdout_hb1/final_eval_train_frames.csv`
- `HBF-HCSV`: `output/chunk2_multiframe_random_fixed_holdout_hb1/final_eval_holdout_frames.csv`
- `HBF-SUP`: `output/chunk2_multiframe_random_fixed_holdout_hb1/support_metrics_train.csv`
- `HBF-VIS`: `output/chunk2_multiframe_random_fixed_holdout_hb1/frame_training_visits.csv`

- `HBL-TCSV`: `output/chunk2_multiframe_random_learnable_holdout_hb1/final_eval_train_frames.csv`
- `HBL-HCSV`: `output/chunk2_multiframe_random_learnable_holdout_hb1/final_eval_holdout_frames.csv`
- `HBL-SUP`: `output/chunk2_multiframe_random_learnable_holdout_hb1/support_metrics_train.csv`
- `HBL-VIS`: `output/chunk2_multiframe_random_learnable_holdout_hb1/frame_training_visits.csv`

Suggested reviewer verdict line format:

- `HBF-I=PASS, HBF-S=PASS, HBF-F2=FAIL(<reason>), HBF-R2=PASS, HBF-TCSV=PASS, ...`

### 6i. Human manual review notes (user review)

Reviewer notes captured from tagged artifact review:

- `HBF-I/HBF-S`: after-training surfel cloud appears tighter than init cloud; reviewer notes too few frames for confident 3D-structure judgment.
- `HBF-F*` and `HBF-R*`: reviewer sees little to no visible change from prior behavior; frame-0 best-case may be due to weak overlap with other frames (hypothesis).
- `HBL-*` vs `HBF-*`: reviewer sees very similar behavior overall.
- `HBL-F2/HBL-F3`: frames containing the calibration cube (object of interest) still do not reproject cube quality well; persistent weakness remains.

Interpretation for Chunk-2 follow-up (training-focused):

- Manual review is consistent with metric findings (limited cross-view support depth and persistent frame-dominance signals).
- Remaining priority is not Stage-0 init correctness (already stabilized), but cross-view training effectiveness on overlap-critical frames (notably frame2/frame3 in this run setup).

### 6j. Quantitative cross-check against reviewer hypotheses

- Frame-0 overlap hypothesis is strongly supported by support CSVs:
  - fixed (`HBF-SUP`): frame0 `visible_surfel_count=479`, `single_view_owner_count=479`, `nearest_owner_count=479`.
  - learnable (`HBL-SUP`): frame0 `visible_surfel_count=465`, `single_view_owner_count=465`, `nearest_owner_count=465`.
  - Interpretation: in this run, frame0-visible surfels are effectively single-view-owned by frame0.

- Frames 2 and 3 remain the worst-loss frames in both modes (`HBF-TCSV`, `HBL-TCSV`):
  - fixed: frame2 `total_loss=0.079433`, frame3 `0.072710`.
  - learnable: frame2 `total_loss=0.082253`, frame3 `0.071277`.
  - Interpretation: reviewer-observed cube reprojection weakness in frame2/frame3 matches scalar diagnostics.

### 6k. Strategic decision: stop optimizing overlap quality inside Chunk 2

Decision (agreed):

- Do not spend more effort trying to "perfect" overlap/cross-view quality within Chunk 2.
- Treat Chunk 2 as Stage-0 stabilization work (init + opacity behavior), then move to later chunks for overlap-quality fixes.

Rationale from source plans:

- Execution plan defines Chunk 2 scope narrowly as Stage-0 behavior (`random/zero` init + fixed opacity), while overlap table/sampler belongs to Chunk 3 and coupling/support belongs to Chunk 4.
- Detailed plan defines Stage 0 as `init_only` and puts multi-view overlap likelihood in Stage 1.
- High-level plan states the multi-view support/diversity retention policy and mandatory belief-to-geometry coupling as the mechanisms intended to solve weak overlap/frame-dominance behavior.

Practical implication:

- Chunk 2 acceptance should remain based on its own gate contracts.
- Cross-view failure modes observed here (frame dominance, weak frame2/3 cube reprojection, low `support>=3`) are tracked as expected unresolved items to be addressed primarily in Chunk 3/4 implementation.

Chunk 2 baseline numbers for Chunk 3/4 to beat (from 6c/6e higher-budget runs):

| Metric | Fixed (Run C, primary baseline) | Learnable (Run D, secondary ablation) |
|--------|----------------------------------|----------------------------------------|
| `support>=2` | 0.4615 | 0.4630 |
| `support>=3` | 0.0446 | 0.0439 |
| `median_support` | 1.0 | 1.0 |
| `single_view_top_share` (train) | 0.5606 | 0.5643 |
| `single_view_top_share` (train+holdout) | 0.7249 | 0.7252 |
| `holdout_loss / train_loss` | 1.52x | 1.52x |
| `train_loss_mean` | 0.032021 | 0.039490 |
| `holdout_loss_mean` | 0.048669 | 0.060202 |
| `train_ssim_mean` | 0.8775 | 0.8455 |
| `holdout_ssim_mean` | 0.8601 | 0.8448 |

Chunk-specific expectations (to avoid overloading Chunk 3):

- Chunk 3 target emphasis (overlap/sampler/likelihood): reduce frame dominance (`single_view_top_share`), improve holdout generalization trend, and keep losses finite/stable under overlap-aware sampling.
- Chunk 4 target emphasis (coupling/support): increase multi-view retention depth (`support>=3`), raise `median_support` above 1.0, and improve overlap-critical frame quality (notably frame2/frame3 in this setup).

Evaluation policy notes:

- Use fixed-opacity (Run C) as the primary comparator because Chunk 2 default behavior is fixed opacity.
- Keep learnable-opacity (Run D) as a secondary ablation; improvements there are informative but not the primary gate signal.
- Holdout has only 2 frames in this protocol; treat holdout ratio as directional evidence and always pair it with per-frame holdout CSV rows and tagged visual checks.

These are the cross-view quality baselines for Chunk 3/4 progression. (opus4.6 review, refined by gpt-5.3-codex, 2026-02-11)
