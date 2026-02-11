# Plan: Elevation-Aware Training Implementation Execution

**Date:** 2026-02-10  
**Status:** Ready to implement  
**Owner:** OpenCode (gpt-5.3-codex)

---

## Purpose

Define the practical execution strategy for implementing the elevation-aware training work described in:

- `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md` (**sole source of truth**)
- `plans/PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md` (high-level context)

This execution plan answers:

1. Should implementation be done all at once? (**No**)
2. Should tests be run between chunks? (**Yes, mandatory**)
3. When should commits happen? (**At validated chunk boundaries**)

---

## Implementation Strategy

Do **not** implement everything in one pass. Implement in risk-ordered chunks with validation gates between chunks.

### Why chunked delivery

- Isolates failures (convention/sign bugs vs likelihood bugs vs coupling bugs).
- Keeps each step runnable and debuggable.
- Reduces risk of hidden regressions in a large refactor.
- Allows mesh-quality checkpoints after each meaningful capability addition.

---

## Chunk Plan (Execution Order)

### Chunk 1: Safety rails + geometry contracts

Scope:

- Convention assertions and run-header convention logging.
- Projection/back-projection helper contracts (`SonarProjection`, `back_project_bins` path).
- Range attenuation path integration and diagnostics wiring.

Goal:

- Lock coordinate and projection correctness before adding optimization complexity.

### Chunk 2: Stage 0 behavior

Scope:

- Elevation-aware initialization (`random` default, `zero` fallback).
- Fixed-opacity toggle in sonar path (`SONAR_FIXED_OPACITY=1` default behavior).

Goal:

- Ensure initialization and physics toggles are stable and backwards-safe.

### Chunk 3: Stage 1 likelihood core

Scope:

- Overlap table and per-iteration frame sampler.
- Pixel bank/logit registry and optimizer.
- Frame stats cache/reliability path.
- Robust normalized amplitude likelihood + temperature annealing.

Goal:

- Get the belief layer stable first (without coupling added yet).

### Chunk 4: Belief-to-geometry enforcement

Scope:

- Mandatory coupling loss (expected point -> surfel association).
- Persistent surfel IDs and ID-keyed support buffers.
- Multi-view support schedule + retention/pruning logic.

Goal:

- Ensure improved elevation belief actually moves geometry and improves surfel retention quality.

### Chunk 5: Late-stage refinements

Scope:

- Normals ramp and expected-elevation normals path.
- Optional Stage 2 densification hook (default disabled).

Goal:

- Add late stability/quality improvements without destabilizing core training.

---

## Validation Gates Between Chunks (Mandatory)

Each chunk must pass its gate before moving to the next chunk.

### Gate after Chunk 1

- Convention checks pass:
  - azimuth sign mapping,
  - elevation sign mapping,
  - transform roundtrip consistency.
- Attenuation sanity check passes:
  - attenuation ON gives lower intensity for farther range (all else equal).

### Gate after Chunk 2

- Init-only smoke run succeeds.
- `ELEV_INIT_MODE=random` shows non-zero elevation/Y spread.
- `ELEV_INIT_MODE=zero` reproduces legacy-like behavior.
- Fixed-opacity mode confirms opacity params are frozen.

### Gate after Chunk 3

- Short training run with reduced workload succeeds.
- `loss_lik` / entropy terms are finite (no NaN/Inf).
- Invalid projection handling is neutral (not over-penalizing).
- Entropy trend is directionally decreasing over short horizon.

### Gate after Chunk 4

- Short run with coupling enabled succeeds.
- Coupling match rate and residual metrics are sensible.
- ID integrity checks pass across topology changes (no support-state drift).
- Support/pruning behavior follows configured warmup and thresholds.

### Gate after Chunk 5

- Normals ramp activates on configured iterations.
- Expected-elevation normals path does not destabilize training.
- Optional Stage 2 hook can be toggled on/off safely (off by default).

### Resume gate after every chunk

- Save checkpoint.
- Reload checkpoint.
- Continue training for a short continuation window.
- Confirm no state-contract breakage (`pixel_logits`, `optim_elev`, support buffers, surfel IDs).

---

## Commit Policy

Commit at **validated chunk boundaries** only.

### Commit criteria

A chunk is commit-ready only if:

- Code compiles/runs.
- Chunk-specific validation gate passed.
- Resume gate passed.
- No known blocker left inside the chunk scope.

### Planned commit cadence

1. conventions/asserts + projection contracts + attenuation
2. elevation-aware init + fixed-opacity toggle
3. Stage 1 likelihood/annealing core
4. coupling + persistent surfel IDs + support/pruning
5. normals ramp + optional Stage 2 hook

### Commit-message format

Use repository convention with model marker (per CLAUDE.md / AGENTS.md):

- `<description> (<model-name>)` — use the name of the model that performed the implementation work.

Examples:

- `Add sonar convention asserts and attenuation diagnostics (opus4.6)`
- `Implement Stage-1 elevation likelihood and annealing (gpt-5.3-codex)`

### Pre-commit documentation updates

Before each commit, update:

- `plans/progress_overview.md`
- `plans/scientific_progress.md`

---

## Implementation Tactics (How to Implement)

These tactics define engineering style and rollout behavior for this plan. They do not change the detailed plan scope; they reduce integration risk.

- Use a shadow-mode rollout first: compute Stage-1 likelihood/coupling paths and log diagnostics before adding them to `loss`; enable weights only after sanity checks pass.
- Centralize config parsing in one typed runtime config object in `debug_multiframe.py`; avoid scattered `os.getenv` calls in deep helpers.
- Keep all new behavior behind explicit gates (`ELEVATION_AWARE`, stage gates, feature flags) so baseline behavior is still reproducible.
- Use explicit state ownership for new runtime state (`pixel_bank`, `pixel_logits`, `optim_elev`, `frame_stats`, `surfel_ids`, support buffers) instead of ad-hoc globals.
- Keep core math helpers pure and side-effect-free (`back_project_bins`, `sonar_project_points`, masked-softmax, reliability, association) so they can be unit-checked in isolation.
- Precompute run-static structures in no-grad mode (`overlap_table`, `frame_stats`) once per run for v1; do not recompute in hot loops unless frame set changes.
- Make detach boundaries explicit: detached evidence target vs learnable logits prediction; avoid in-place tensor edits on values participating in autograd.
- Aggregate loss terms exactly once in one block at the end of Stage-1 assembly to prevent accidental double-counting.
- Add baseline parity checks for off-mode: when elevation-aware features are disabled, outputs should match legacy behavior within tolerance.
- Version checkpoint schema for new state payloads so resume mismatch causes deterministic, explicit failures.
- Instrument before optimizing runtime: verify correctness/consistency metrics first, then optimize vectorization/caching/memory.

---

## Execution Notes

- Keep Stage 2 densification disabled by default during initial stabilization.
- Prefer small, repeatable short runs for gates (fixed seed, reduced per-iter load).
- Treat mesh quality as the primary success criterion; scalar losses are supporting diagnostics.
- Before starting each chunk, create a short chunk-specific implementation plan and save it as a markdown file in `plans/` (one file per chunk) so intent and scope are explicit before coding.
