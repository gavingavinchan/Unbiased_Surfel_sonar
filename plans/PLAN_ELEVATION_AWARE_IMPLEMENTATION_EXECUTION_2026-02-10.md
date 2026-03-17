# Plan: Elevation-Aware Training Implementation Execution

**Date:** 2026-02-10  
**Status:** Chunk 1/2 implemented; Chunk 3 partially implemented (infrastructure complete, core likelihood contract parity pending); Chunk 4 implemented but blocked; Chunk 4.5 (renderer remediation) interposed before Chunk 5; Chunk 5 pending on post-v2 re-gating  
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

Terminology/intent lock across subsidiary plans:

- Wording drift is acceptable only when implementation intent is unchanged.
- In particular, frame identity references (`frame index` vs `frame_key`) are non-material only when they preserve the same contract: stable per-frame identity, deterministic checkpoint keying, and deterministic resume mapping.
- If wording drift changes behavior or makes intent ambiguous, stop and reconcile in this execution plan and the detailed plan before implementation.

---

## Implementation Strategy

Do **not** implement everything in one pass. Implement in risk-ordered chunks with validation gates between chunks.

### Current implementation state (2026-03-12)

- Chunk 1 is implemented and validated.
- Chunk 2 is implemented and validated.
- Chunk 3 is partially implemented: Stage-1 infrastructure (mode gating, frame-keyed pixel-logit state, checkpoint schema/fingerprint handling, refresh/remap plumbing, helper tests) is in code.
- Chunk 3 core contract parity remains open: overlap-neighbor multi-view likelihood assembly via `back_project_bins` + projection-validity evidence in the training loop is not fully integrated yet.
- Chunk 4 was implemented and extensively probed, but its gate posture remained blocked on synthetic cube behavior and unresolved closeout issues.
- After Chunk-4 investigation, Chunk 4.5 (renderer remediation) was introduced as an explicit prerequisite tranche before Chunk 5.
- Chunk 4.5 (renderer remediation) is implemented in code, but post-v2 active-path validation and synthetic re-baselining are still open; treat pre-v2 Chunk-4 metrics as historical when renderer semantics differ.
- Chunk 5 remains pending and should be evaluated only against post-v2 post-Chunk-4 evidence.
- Dataset-C synthetic results indicate a likely Chunk-2 quality ceiling for cube-like shape recovery, so future chunks must include explicit synthetic dataset validation in their gates.
- Recorded qualitative baseline artifact (exp3): `output/debug_multiframe_synth_c_exp3/input.ply` is cube-like, while `output/debug_multiframe_synth_c_exp3/surfels_after_training.ply` shows FOV-bounded torus-like streaking aligned with the revolution axis.

### Why chunked delivery

- Isolates failures (convention/sign bugs vs likelihood bugs vs coupling bugs).
- Keeps each step runnable and debuggable.
- Reduces risk of hidden regressions in a large refactor.
- Allows mesh-quality checkpoints after each meaningful capability addition.

### General execution note (user direction)

- Implementation details are flexible as long as verification is strong, explicit, and reproducible.
- For each chunk, publish the exact tests to be run (commands, expected checks, artifact paths) before gate review.
- Synthetic datasets are the primary validation path for automated checks; manual visual review is supplementary and used only where automated metrics are insufficient.
- Continue execution when plan reviews align across active reviewers (OpenCode + opus track); if a contract-level disagreement appears, record it and resolve before code changes proceed.

### Development methodology (TDD requirement)

- Use test-driven development for Chunk 3/4/4.5/5 work: define or update executable tests first, then implement code to satisfy those tests.
- Each implementation task must map to at least one pre-declared verification item (unit/contract test, smoke test, synthetic gate check, or resume check).
- Do not mark a chunk item complete until its mapped tests pass and artifacts are recorded.
- When behavior changes, update tests and acceptance criteria in the same chunk plan before merging.

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
- Prove the likelihood layer is numerically stable on both real and synthetic validation runs.

### Chunk 4: Belief-to-geometry enforcement

Scope:

- Mandatory coupling loss (expected point -> surfel association).
- Persistent surfel IDs and ID-keyed support buffers.
- Multi-view support schedule + retention/pruning logic.

Goal:

- Ensure improved elevation belief actually moves geometry and improves surfel retention quality.
- Specifically target geometry-smearing failure modes seen in synthetic cube-style reconstructions.

### Chunk 4.5: Renderer remediation (post-Chunk-4, pre-Chunk-5)

Implemented per `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md`. Execution steps R0–R6 in that plan define the full scope, test catalog, and exit criteria.

**Naming note (2026-03-16):** This tranche was originally referred to as "renderer remediation gate" or "renderer remediation tranche" in earlier plan revisions and in the renderer fix plan itself. It is now canonically **Chunk 4.5** across the codebase.

Scope (six confirmed defects + re-baseline):

- **R1 — Normal init + Lambertian gradient safety.** `create_from_pcd` now consumes `pcd.normals` into quaternions (camera-facing at init). Leaky Lambertian (`SONAR_LAMBERTIAN_MODE=leaky`, default `alpha=0.01`) replaces hard `clamp(min=0)` to eliminate the dead-neuron gradient trap. Ablation modes gated behind `SONAR_LAMBERTIAN_MODE`. Surfel size telemetry and dead-zone diagnostics added.
- **R2a — Ray-binned acoustic occlusion.** Per-`(azimuth_bin, elevation_bin)` front-to-back range-ordered compositing with multi-bin participation and support capping (`SONAR_OCCLUSION_MODE=ray_binned`). Elevation marginalization produces the final sonar image: `I[a,r] = sum_e w_e * R[a,e,r]`. Controls: `SONAR_ELEV_BINS`, `SONAR_ELEV_WEIGHT_MODE`, `SONAR_OCCL_KSIGMA`, `SONAR_OCCL_WEIGHT_FLOOR_REL`, `SONAR_OCCL_TOPK`. Mass-loss telemetry required per gate.
- **R2b — 2DGS Jacobian footprint rendering.** Replaces the legacy 4-pixel bilinear `scatter_add_` path with proper 2D Gaussian splatting via `transMat_precomp` into the existing CUDA rasterizer. Polar-projection Jacobian computes `Sigma_2D = J * Sigma_3D * J^T`. Mode: `SONAR_RENDER_MODE=2dgs`.
- **R2-alt — Sigma-point (nonlinear) footprint mode.** 5-point UKF-style projection captures second-order curvature the Jacobian misses for large/oblique surfels. Boundary-aware fitting (K-based fallback, soft border weighting, hysteresis, covariance conditioning). Mode: `SONAR_RENDER_MODE=2dgs_nonlinear`. Both footprint modes are required.
- **R3 — Densification signal wiring.** `viewspace_points`/`radii` carry meaningful rasterizer-produced values under both footprint modes with `ray_binned` occlusion. Densification itself remains Chunk-5 scope (default disabled).
- **R4 — Consistency fixes.** FOV prune `require_all` logic corrected. Coordinate convention divergence between `sonar_utils.py` and `point_utils.py` resolved. Legacy bilinear render path removed (`SONAR_RENDER_MODE=legacy` no longer callable).
- **R5 — Plan hierarchy patches.** Upstream plans annotated with v2 renderer contracts and pre/post-v2 evidence separation.
- **R6 — Synthetic re-baseline.** All pre-v2 synthetic gate claims become historical-only. Post-v2 reruns (C4-S1 through C4-S4, C4-T11, C4-T17) with renderer semantic fingerprint metadata. Comparator rule: cross-run deltas valid only when fingerprints match.

Goal:

- Eliminate six renderer-level defects (random normals, Lambertian dead zone, missing occlusion, point splatting, broken densification signals, consistency bugs) that contaminated all prior photometric training and synthetic gates.
- Replace the legacy bilinear accumulation path with physically motivated ray-space occlusion + proper 2DGS footprint rendering.
- Re-establish a trustworthy renderer baseline with explicit semantic versioning before Chunk-5 evaluation.
- Prevent Chunk-5 from inheriting pre-v2 renderer defects or incomparable synthetic baselines.

### Chunk 5: Late-stage refinements

Scope:

- Normals ramp and expected-elevation normals path.
- Optional Stage 2 densification hook (default disabled).

Goal:

- Add late stability/quality improvements without destabilizing core training.
- Preserve or improve synthetic-gate behavior while adding late refinements.

---

## Validation Gates Between Chunks (Mandatory)

Each chunk must pass its gate before moving to the next chunk.

### Synthetic validation policy for remaining chunks

- For Chunk 3/4/4.5/5 gates, synthetic dataset validation is mandatory in addition to legacy real-data checks.
- Do not hardcode synthetic dataset names in this plan; use the active inventory and commands documented in `docs/SYNTHETIC_DATASET_GUIDE.md`.
- Once renderer-v2 semantics are introduced, pre-v2 synthetic gate claims become historical-only for cross-run comparison purposes.
- Cross-run metric deltas are valid only when renderer semantic fingerprints match.
- For every synthetic run used in a gate, record:
  - command/config,
  - output artifact paths,
  - evaluator metrics,
  - delta versus the most recent matching baseline (including renderer-semantic match).
- Default material-regression rule for Chunk 3/4/4.5/5 synthetic gates:
  - any relative degradation greater than `10%` versus the active baseline in key evaluator error metrics (`mean_*_error_m`, `p95_*_error_m`, `center_error_m`) is treated as material.
  - material regression blocks progression unless an explicit written waiver (with rationale) is approved for the chunk.

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
- Synthetic gate smoke run(s) from `docs/SYNTHETIC_DATASET_GUIDE.md` complete end-to-end (generator/gate runner and/or training+evaluator path as appropriate).
- Synthetic evaluator metrics are finite and satisfy the default material-regression rule versus Chunk-2 baselines (or a documented waiver is approved before continuing).

### Gate after Chunk 4

- Short run with coupling enabled succeeds.
- Coupling match rate and residual metrics are sensible.
- ID integrity checks pass across topology changes (no support-state drift).
- Support/pruning behavior follows configured warmup and thresholds.
- Synthetic validation includes shape-diversity checks from the active synthetic inventory (for example smooth-shape and edge/corner-shape representatives listed in `docs/SYNTHETIC_DATASET_GUIDE.md`).
- Synthetic geometry artifacts associated with Chunk-2 limitations (streaking/toroidal smearing under in-FOV ambiguity) are explicitly reviewed and reported as improved/unchanged/regressed.
- Quantitative synthetic metrics show directional improvement from Chunk-2 baselines for at least the known-problem shape class, or a documented blocker is recorded before proceeding.

### Gate after Chunk 4.5 (mandatory before Chunk 5)

Full test catalog and exit criteria defined in `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md`. Summary gate checklist:

- **Contract tests pass** (`RB-T01`..`RB-T10`, `RB-T15`, `RB-T16`, `RB-T18`..`RB-T22`): normal init ingestion, Lambertian gradient safety, ray-binned occlusion core, no cross-ray occlusion, Gaussian footprint from scale, densification signals, FOV prune semantics, convention roundtrip, nonlinear footprint fidelity, range-order semantics, elevation marginalization, `Sigma_2D->T` conversion, multi-bin participation, support-cap mass accounting, legacy mode removal.
- **Smoke/integration tests pass** (`RB-T11`..`RB-T14`, `RB-T17`): shadow-mode debug run, active-mode debug run, off-mode v2 compatibility, checkpoint continuity, surfel-size telemetry.
- **Both footprint modes validated:** `SONAR_RENDER_MODE=2dgs` (Jacobian) and `2dgs_nonlinear` (sigma-point) pass their respective tests.
- **Active renderer path is gradient-connected end-to-end** (no `grad_fn` disconnects on synthetic guard runs).
- **Active renderer path produces finite rendered outputs** and finite evaluator metrics on synthetic smoke/gate reruns.
- **`SONAR_OCCLUSION_MODE=ray_binned`** is validated and used for all active/gating runs.
- **Renderer semantic fingerprints recorded** for all post-v2 comparison runs, including: `renderer_semantics_version`, `normal_init_mode`, `lambertian_transfer`, `occlusion_model`, `sonar_render_mode`, `sonar_occlusion_mode`, `render_sonar_contract_hash`, `sigma_point_config` (when applicable).
- **Post-v2 synthetic reruns completed** (C4-S1, C4-S2, C4-S3, C4-S4, C4-T11, C4-T17) and separated from pre-v2 historical evidence.
- **Chunk-4 blocker posture re-evaluated** under post-v2 evidence before Chunk 5 begins.
- **Upstream plan documents patched** and cross-linked (R5 deliverable).

### Gate after Chunk 5

- Normals ramp activates on configured iterations.
- Expected-elevation normals path does not destabilize training.
- Optional Stage 2 hook can be toggled on/off safely (off by default).
- Full synthetic gate rerun(s) from `docs/SYNTHETIC_DATASET_GUIDE.md` pass stability/reproducibility checks used for the synthetic program.
- Synthetic metric deltas versus the post-v2 post-Chunk-4 baseline are recorded; late refinements must not reintroduce previously mitigated geometric artifacts.

### Resume gate after every chunk

- Save checkpoint.
- Reload checkpoint.
- Continue training for a short continuation window.
- Confirm no state-contract breakage (`pixel_logits`, `optim_elev`, support buffers, surfel IDs).
- Include at least one synthetic continuation check (resume on a synthetic-config run) for Chunk 3/4/4.5/5.

### Manual visual test policy

- For any chunk, add manual visual tests whenever needed to validate geometry/mesh/render quality that cannot be judged reliably from scalar metrics alone.
- Manual visual checks are allowed in addition to automated gates and should be recorded with artifact paths and a brief pass/fail note.

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
5. Chunk 4.5: renderer remediation (R1–R4: normal init, leaky Lambertian, ray-binned occlusion, 2DGS footprints, densification signals, consistency fixes) + synthetic re-baseline (R5–R6)
6. normals ramp + optional Stage 2 hook

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

- Use a TDD flow for each chunked feature: write or update failing tests/contracts first, implement the minimal code to pass, then refactor while keeping tests green.
- Use a shadow-mode rollout first: compute Stage-1 likelihood/coupling paths and log diagnostics before adding them to `loss`; enable weights only after sanity checks pass.
- Centralize config parsing in one typed runtime config object in `debug_multiframe.py`; avoid scattered `os.getenv` calls in deep helpers.
- Keep all new behavior behind explicit gates (`ELEVATION_AWARE`, stage gates, feature flags) so baseline behavior is still reproducible.
- Use explicit state ownership for new runtime state (`pixel_bank`, `pixel_logits`, `optim_elev`, `frame_stats`, `surfel_ids`, support buffers) instead of ad-hoc globals.
- Keep core math helpers pure and side-effect-free (`back_project_bins`, `sonar_project_points`, masked-softmax, reliability, association) so they can be unit-checked in isolation.
- Precompute run-static structures in no-grad mode (`overlap_table`, `frame_stats`) once per run for v1; do not recompute in hot loops unless frame set changes.
- Make detach boundaries explicit: detached evidence target vs learnable logits prediction; avoid in-place tensor edits on values participating in autograd.
- Aggregate loss terms exactly once in one block at the end of Stage-1 assembly to prevent accidental double-counting.
- Add compatibility checks for off-mode against the frozen v2 compatibility reference rather than pre-v2 legacy bilinear outputs.
- Version checkpoint schema for new state payloads so resume mismatch causes deterministic, explicit failures.
- Instrument before optimizing runtime: verify correctness/consistency metrics first, then optimize vectorization/caching/memory.

---

## Execution Notes

- Keep Stage 2 densification disabled by default during initial stabilization.
- Prefer small, repeatable short runs for gates (fixed seed, reduced per-iter load).
- Treat mesh quality as the primary success criterion; scalar losses are supporting diagnostics.
- Before starting each chunk, create a short chunk-specific implementation plan and save it as a markdown file in `plans/` (one file per chunk) so intent and scope are explicit before coding.
- For future subsidiary plans (Chunk 3+), include a dedicated synthetic test matrix section that references `docs/SYNTHETIC_DATASET_GUIDE.md` for active datasets and defines run commands, artifacts, baselines, and pass/fail criteria for that chunk.
