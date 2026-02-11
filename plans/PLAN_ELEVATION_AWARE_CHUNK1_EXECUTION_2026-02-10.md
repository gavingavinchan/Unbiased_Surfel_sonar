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
