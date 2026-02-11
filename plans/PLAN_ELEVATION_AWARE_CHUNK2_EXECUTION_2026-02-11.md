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
