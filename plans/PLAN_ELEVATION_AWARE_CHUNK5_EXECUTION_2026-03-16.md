# Plan: Elevation-Aware Chunk 5 Execution

**Date:** 2026-03-16  
**Status:** Draft; prerequisite-gated pending post-v2 re-gating (gpt-5.4)  
**Scope:** Chunk 5 only (late normals refinement + optional Stage 2 densification)

## Consensus Marker (Implementation Unlock)

- Consensus status: `pending`
- Last reviewed date: `2026-03-16`
- Approver(s): `pending`
- Implementation unlocked: `no`

---

## Goal

Implement the late refinement tranche only after the elevation-belief and renderer baselines are trustworthy.

Primary outcomes:
- normals regularization ramps from the early weak regime to the late stronger regime without destabilizing training,
- expected-elevation geometry is used for normals only when posterior support is valid/confident,
- optional Stage 2 densification can be toggled on safely, remains off by default, and preserves Chunk-4 persistent-ID/support contracts,
- Chunk-5 runs improve or preserve post-v2 synthetic geometry quality and do not reintroduce previously observed cube-class streaking/toroidal artifacts.

Chunk-5 is a refinement tranche, not a rescue tranche:
- it must not be used to compensate for an open Chunk-3 Stage-1 parity gap,
- it must not be used to compensate for an open Chunk-4.5 renderer-v2 validation/re-baseline gap,
- it must not be interpreted against pre-v2 historical baselines.

---

## Source of Truth

- Main implementation contract:
  - `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md`
- Chunk sequencing and gate policy:
  - `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md`
- Decision ledger / renderer interposition note:
  - `plans/PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md`
- Program flow snapshot:
  - `plans/PLAN_ELEVATION_AWARE_TRAINING_FLOW_2026-02-24.md`
- Prior chunk references:
  - `plans/PLAN_ELEVATION_AWARE_CHUNK2_EXECUTION_2026-02-11.md`
  - `plans/PLAN_ELEVATION_AWARE_CHUNK3_EXECUTION_2026-02-16.md`
  - `plans/PLAN_ELEVATION_AWARE_CHUNK4_EXECUTION_2026-02-23.md`
  - `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md`
- Active synthetic inventory and commands:
  - `docs/SYNTHETIC_DATASET_GUIDE.md`

If any mismatch appears, higher-level upstream plans supersede this Chunk-5 execution note. In practice:
- follow `plans/PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md` and `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md` for algorithm intent and stage semantics,
- follow `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md` for sequencing and gate policy,
- treat this document as the Chunk-5 refinement/rollout contract only where it does not conflict with those upstream plans.

---

## Readiness Gate Before Chunk 5 Starts

Current upstream posture says Chunk 5 is still blocked. This plan is the execution template to use once the following prerequisites are green or explicitly waived in writing:

1. **Chunk-3 parity closure or waiver**
   - The Stage-1 likelihood path must either match the overlap-neighbor `back_project_bins` multi-view evidence contract, or an explicit waiver must record why Chunk-5 evaluation is still meaningful.

2. **Chunk-4.5 mandatory gate close**
   - Active renderer-v2 path is gradient-connected end-to-end.
   - Active renderer-v2 path produces finite rendered outputs.
   - Post-v2 synthetic reruns are complete and recorded with renderer semantic fingerprints.

3. **Chunk-4 blocker posture re-evaluated under post-v2 evidence**
   - Chunk-4 results must be re-read against the renderer-v2 baseline before Chunk-5 refinements are judged.

4. **Post-v2 comparator selection fixed**
   - Chunk-5 baseline metrics must come from the most recent matching post-v2 post-Chunk-4 baseline group only.

Implementation-unlock rule:
- Chunk-5 code work may prepare tests/helpers in advance,
- but no Chunk-5 gate claim is valid until all four readiness items above are satisfied or waived explicitly.

---

## Terminology Alignment (Critical)

`debug_multiframe.py` already uses curriculum Stage 1/2/3 labels. This chunk plan uses elevation-stage terminology from the detailed plan:

- Elevation Stage 0 = initialization
- Elevation Stage 1 = bin-likelihood + coupling
- Elevation Stage 2 = optional densification

To avoid naming collision:
- keep existing curriculum stage labels unchanged in code/logs,
- refer to Chunk-5 work as `late normals` and `Elevation Stage 2 densification`,
- do not overload curriculum `Stage 2` with elevation `Stage 2` terminology.

---

## In-Scope Files (Chunk 5)

Primary implementation target:
- `debug_multiframe.py`

Likely helper modules:
- `utils/elevation_chunk5_helpers.py` (recommended new import-safe helper module for late normals and densification helpers)
- `utils/elevation_stage1_helpers.py` (reuse for posterior/entropy access when appropriate)
- `utils/point_utils.py` (expected-elevation point path for normals)
- `utils/sonar_utils.py` (only if minor helper parity is needed)

Hard dependency contracts that must remain consistent:
- `gaussian_renderer/__init__.py` (renderer-v2 `viewspace_points` / `radii` behavior already validated in Chunk 4.5)
- `scene/gaussian_model.py` (densify/split/clone/prune lifecycle)
- Chunk-4 persistent-ID/support bookkeeping inside `debug_multiframe.py`

Optional only if needed for clean config exposure:
- `arguments/__init__.py`

Out of scope for Chunk 5:
- renderer-v2 defect remediation that belongs to Chunk 4.5,
- redesign of Stage-1 belief contracts that belongs to Chunk 3,
- redefinition of Chunk-4 coupling/support policy beyond densify-entry integration,
- new synthetic datasets or evaluator semantics.

---

## Chunk-5 Contracts to Implement

### 1. Normals ramp schedule contract

- Keep the fixed-gate schedule from the detailed plan:
  - early regime: `iter < ELEV_NORMAL_RAMP_START_ITER`
  - ramp regime: `iter in [ELEV_NORMAL_RAMP_START_ITER, ELEV_NORMAL_RAMP_END_ITER]`
  - late regime: `iter > ELEV_NORMAL_RAMP_END_ITER`
- `w_normal` must interpolate deterministically from `ELEV_NORMAL_WEIGHT_EARLY` to `ELEV_NORMAL_WEIGHT_LATE`.
- Canonical v1 interpolation formula:

```python
if iter < ELEV_NORMAL_RAMP_START_ITER:
    w_normal = ELEV_NORMAL_WEIGHT_EARLY
elif iter > ELEV_NORMAL_RAMP_END_ITER:
    w_normal = ELEV_NORMAL_WEIGHT_LATE
else:
    progress = (iter - ELEV_NORMAL_RAMP_START_ITER) / max(
        1,
        ELEV_NORMAL_RAMP_END_ITER - ELEV_NORMAL_RAMP_START_ITER,
    )
    w_normal = ELEV_NORMAL_WEIGHT_EARLY + progress * (
        ELEV_NORMAL_WEIGHT_LATE - ELEV_NORMAL_WEIGHT_EARLY
    )
```

- Budget-scaling note: defaults assume the usual longer run budget; short synthetic runs should use env overrides rather than silently changing the contract in code.
- Off-mode compatibility rule: if `ELEVATION_AWARE=0`, effective Chunk-5 late-normal behavior is disabled.

### 2. Expected-elevation normals contract

- Expected-elevation geometry becomes eligible at `iter >= ELEV_NORMAL_ELEV_START_ITER`.
- Use the Stage-1 posterior from the same iteration:
  - `e_exp = sum_k p_post[k] * elev_bins[k]`.
- Feed expected elevation only into the normals-geometry path; do not rewrite the core back-projection/init contracts.
- `utils/point_utils.py` path must accept per-pixel elevation for normals generation while preserving `None -> legacy zero-elevation` behavior.
- Minimum implementation detail to keep this chunk unambiguous:
  1. compute expected elevation per supported pixel,
  2. build expected-elevation 3D points in the canonical camera/view convention,
  3. derive local finite-difference normals from neighboring expected-elevation points,
  4. normalize with epsilon-safe handling,
  5. supervise matched surfel normals only where the confidence gate passes.
- Sparse-bank implementation note:
  - current Stage-1 anchors in `debug_multiframe.py` come from a sparse bright-pixel bank rather than a dense per-pixel posterior field,
  - therefore Chunk-5 v1 must not assume the required four image-grid neighbors are already present in `pixel_bank`,
  - instead, Chunk-5 normal computation must explicitly materialize a deterministic local 4-neighborhood closure for each anchor pixel (either by augmenting the queried pixel set before posterior evaluation or by an equivalent on-demand helper that evaluates those exact four neighbors under the same Stage-1 evidence contract),
  - nearest-neighbor substitution inside the sparse bank is not an acceptable silent fallback for v1.

Recommended v1 finite-difference pattern:

```python
# neighbor displacements in azimuth/range directions
dp_az = pts_right - pts_left
dp_rg = pts_down - pts_up
n_fd = torch.cross(dp_az, dp_rg, dim=-1)
n_fd = n_fd / n_fd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
```

- Neighbor selection must use the exact image-grid 4-neighborhood around each anchor pixel after the explicit local-closure step above; do not switch to a sparse-bank surrogate neighborhood in v1.

### 3. Normals confidence gate contract

- Apply expected-elevation normals only where the posterior is both supported and confident.
- Minimum v1 confidence contract:
  - supported-bin mask is non-empty,
  - entropy-based confidence mask is finite and deterministic,
  - unsupported / low-confidence pixels fall back to the pre-Chunk-5 normals path.
- Canonical v1 confidence formula:

```python
entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-8)), dim=-1)
max_entropy = math.log(K)
confident = entropy < (ELEV_NORMAL_CONFIDENCE_THRESH * max_entropy)
```

- Default threshold for first-pass runs: `ELEV_NORMAL_CONFIDENCE_THRESH=0.5`.
- Confidence gating must be logged explicitly so late-normal coverage is observable.

### 4. Normal-loss aggregation contract

- Keep one unified total-loss block.
- Chunk-5 must not double-count normal terms.
- The late-normal contribution must be easy to disable for ablation without disturbing Chunk-3/4 loss terms.
- Current `debug_multiframe.py` aggregation only combines photometric loss, Stage-1 likelihood/entropy, and Chunk-4 coupling. Chunk-5 therefore introduces one new explicit normal-supervision term; it must not assume an existing normal-loss block is already present in the debug script.
- If no pixels pass the confidence gate, Chunk-5 normal refinement contributes zero for that step and training remains finite.
- V1 default neighborhood source for finite-difference normals is the local image-grid 4-neighborhood around each anchor pixel: `(row, col-1)`, `(row, col+1)`, `(row-1, col)`, `(row+1, col)`.
- Because the current Stage-1 anchor set is sparse, v1 implementation must explicitly request those four image-grid neighbors for Chunk-5 normal computation rather than assuming they already exist in the Stage-1 bank.
- If any required neighbor is invalid, unsupported, or unavailable even after that explicit local-closure step, skip Chunk-5 normal supervision for that anchor pixel rather than switching neighbor strategy implicitly.
- Recommended v1 supervision loss for matched/confident surfels is cosine distance with sign-ambiguity handling:

```python
loss_normal = 1.0 - torch.abs((n_quat * n_expected).sum(dim=-1))
loss_normal = loss_normal.mean()
```

- Normal mode gate should mirror the shadow-first rollout used in prior chunks:
  - `ELEV_NORMAL_MODE=off|shadow|active`
  - `off`: do not compute or apply Chunk-5 normal supervision,
  - `shadow`: compute/log `loss_normal` without adding it to total loss,
  - `active`: add `w_normal * loss_normal` exactly once in the unified loss block.

### 5. Densification enable contract

- `ELEV_DENSIFY=0` remains the default.
- Explicit precedence policy:
  - if `ELEVATION_AWARE=0`, force `ELEV_NORMAL_MODE=off`, `ELEV_DENSIFY=0`, and `ELEV_DENSIFY_MODE=off`,
  - if `ELEV_DENSIFY=0`, effective densify mode is `off` regardless of `ELEV_DENSIFY_MODE`,
  - `ELEV_DENSIFY_MODE` is consulted only when `ELEV_DENSIFY=1`.
- Densification becomes eligible only when all are true:
  - `ELEV_DENSIFY=1`,
  - `iter >= ELEV_STAGE2_START_ITER`,
  - `iter % ELEV_DENSIFY_INTERVAL == 0`,
  - renderer-v2 active contract is in use for the run,
  - `SONAR_OCCLUSION_MODE=ray_binned`,
  - Chunk-4 persistent-ID/support bookkeeping is active.
- Densification rollout order is mandatory:
  - `off` baseline,
  - `shadow` diagnostics only,
  - `active` surfel creation.

Recommended new mode flag for safe rollout:
- `ELEV_DENSIFY_MODE=off|shadow|active`

Mode semantics:
- `off`: no Stage-2 candidate scoring or surfel creation.
- `shadow`: score/report candidates and peak bins, but do not create surfels.
- `active`: create surfels and integrate them into support/optimizer state.

### 6. Densification candidate-selection contract

- Candidate anchors come from persistent high-error bright pixels, not arbitrary pixels.
- Minimum v1 candidate gate:
  - pixel is valid after existing masks,
  - GT return is meaningful/bright,
  - render residual indicates under-explained structure,
  - candidate is not already well explained by nearby supported surfels.
- Candidate selection must be deterministic under fixed seed and deterministic frame order.
- V1 persistence policy for candidate history is explicit:
  - maintain a per-frame, per-pixel high-error tracker only when densification is enabled,
  - persist that tracker in checkpoints so resume behavior is deterministic,
  - key it by stable frame identity plus pixel coordinates,
  - apply the same strict/reset mismatch policy as other frame-keyed Chunk-3/4 state.

### 7. Arc scoring / peak-selection contract

- For each selected candidate pixel, score elevation bins using multi-view agreement across overlap neighbors.
- Use the active Stage-1 projection validity and reliability rules; invalid projections remain neutral, not punitive.
- Keep densify placement scoring logically separate from the primary Stage-1 belief path: Stage-1 likelihood / posterior machinery remains the main optimization driver, while Stage-2 densify peak selection uses a derived multi-view agreement score for placement only.
- Minimum v1 placement rule:
   - one-surface case: spawn at the strongest peak bin,
   - optional multi-peak mode: allow more than one spawn only when peaks are clearly separated and pass explicit thresholds.
- If the score profile is flat/unsupported, skip densification for that candidate and log the skip reason.
- Recommended first-pass scoring rule for the separate placement score:

```python
score_e = sum_b reliability_b * valid_b * gt_intensity_b
```

- Because the higher-level plans supersede this document, the exact formula above is a recommended v1 implementation, not a mandatory law. Alternative multi-view agreement scores are acceptable if they preserve the upstream architecture:
  - Stage-1 likelihood remains the primary GT-anchored belief path,
  - optional Stage-2 densify uses a separate agreement score rather than becoming a new global loss,
  - validity/reliability neutrality is preserved,
  - the resulting placement behavior is documented and tested.
- Densification rate must be capped per trigger event for safety (`ELEV_DENSIFY_MAX_PER_EVENT`).

### 8. New-surfel initialization contract

- New surfels created by densification must initialize with:
  - position at the selected arc-peak world point,
  - rotation from expected/local normal when available, otherwise deterministic camera-facing fallback,
  - scale initialized by existing surfel-init policy or a documented local neighborhood heuristic,
  - opacity respecting `SONAR_FIXED_OPACITY` mode.
- Every new surfel must receive a fresh persistent ID and Chunk-4 support buffers must be extended in the same transaction.
- New surfels inherit the Chunk-4 grace-window contract before hard prune checks apply.

### 9. Sonar densification-signal contract

- Chunk 4.5 validated that `viewspace_points` and `radii` are no longer placeholder-only outputs; Chunk 5 must define how they are consumed.
- For sonar runs, the actionable contract is:
  - `viewspace_points` is the per-surfel differentiable position proxy returned by `render_sonar`, row-aligned with active surfels,
  - `viewspace_points.grad[row]` is the signal used by `add_densification_stats()` for that surfel,
  - `radii[row]` is the renderer-produced image-space footprint radius in sonar pixel units for that surfel,
  - `max_radii2D` remains interpreted in those sonar pixel units.
- Chunk-5 code must rely on the gradient/radius semantics above, not on any assumption that `viewspace_points` values themselves equal literal `(col, row)` coordinates.

### 10. Resume / checkpoint continuity contract

- Save/load/continue must preserve:
  - late-normal schedule state derivable from iteration counters,
  - densification mode and counters,
  - newly allocated surfel IDs and support buffers,
  - any densification bookkeeping required for deterministic continuation.
- Resume must not silently change densification mode or normal-ramp regime.
- Chunk-5 checkpoint state should be carried in a dedicated sibling payload (for example `elevation_chunk5_state`) parallel to the existing Stage-1 and Chunk-4 payloads, rather than replacing their schema markers.
- Chunk-5 schema marker should be explicit inside that dedicated payload for resume diagnostics:

```python
chunk5_checkpoint_state = {
    "checkpoint_schema_version": "chunk5_normals_densify_v1",
    ...,
}
```

- Top-level checkpoint format should therefore add `elevation_chunk5_state`, not repurpose `elevation_stage1_state` or `elevation_chunk4_state`.
- Do not persist `w_normal` as standalone state; derive it from iteration on resume.

### 11. Diagnostics contract

- Normals diagnostics:
  - active `w_normal`,
  - expected-elevation normals coverage,
  - confidence-mask coverage,
  - normal-loss scalar(s),
  - finite-count / skipped-count summaries.
- Densification diagnostics:
  - candidate count,
  - supported-candidate count,
  - spawned surfel count,
  - skipped-by-reason counts,
  - high-error tracker coverage / reset counts,
  - post-densify ID/support integrity stats,
  - surfel-count delta and prune-count delta after grace-aware checks.

---

## Quantitative Gate Thresholds (Chunk 5)

These thresholds make the Chunk-5 gate objective. If any value is changed for a run, the gate report must record the override and rationale.

- `GATE_FINITE_NAN_INF = 0`
- `GATE_OFFMODE_REL_LOSS_DELTA_MAX = 0.05`
- `GATE_OFFMODE_ABS_SSIM_DELTA_MAX = 0.01`
- `GATE_NORMAL_CONF_COVERAGE_MIN = 1e-4` on at least one representative active run
- `GATE_MAX_DUPLICATE_ACTIVE_IDS = 0`
- `GATE_MAX_INVALID_ID_TO_ROW = 0`
- default material-regression rule versus matching post-v2 post-Chunk-4 baseline:
  - any relative degradation `> 10%` in key evaluator error metrics (`mean_*_error_m`, `p95_*_error_m`, `center_error_m`) blocks progression unless explicitly waived.

---

## Implementation Method (TDD-First, Low-Risk Rollout)

1. Keep Chunk-5 split into two sub-tranches:
   - **5A:** late normals only, with densification forced off.
   - **5B:** optional densification, enabled only after 5A stability is demonstrated.

2. Start with fast failing tests for pure schedule/mask/selection helpers before editing training-loop behavior.

3. Roll out every new path in shadow-first mode:
   - late normals shadow = compute expected-elevation normals diagnostics before increasing loss weight,
   - densify shadow = compute/report candidates and peak bins before creating surfels.

4. Preserve off-mode parity:
   - `ELEV_DENSIFY_MODE=off` and late normals disabled must remain close to the post-v2 post-Chunk-4 baseline.

5. Keep defaults conservative:
   - normals enabled only through the existing ramp schedule,
   - densification disabled by default,
   - multi-peak densification optional and not required for the first green gate unless explicitly enabled.
   - reduced-budget smoke/gate runs must use explicit env overrides when they need to exercise Chunk-5 paths; do not rely on long-run default iteration thresholds in short synthetic runs.

6. Recommended test/helper layout:
   - `utils/elevation_chunk5_helpers.py`
   - `tests/test_elevation_chunk5_normals_contracts.py`
   - `tests/test_elevation_chunk5_densify_contracts.py`
   - `tests/test_elevation_chunk5_checkpoint_contracts.py`
   - `tests/test_elevation_chunk5_smoke_modes.py`

---

## Step-by-Step Work Order

1. Reconfirm prerequisite posture and record the selected post-v2 post-Chunk-4 comparator artifacts for Chunk-5 evaluation.
2. Write failing helper tests for normal-ramp interpolation, confidence masking, and expected-elevation geometry handoff.
3. Implement/import-safe helpers for late-normal schedule and confidence gating; make those tests green.
4. Wire expected-elevation normals path into `utils/point_utils.py` / `debug_multiframe.py` with densification still forced off.
5. Run late-normals shadow smoke, then active late-normals smoke on reduced workload.
6. Add failing tests for densify mode gates, candidate ranking, peak selection, and persistent-ID/support extension on spawn.
7. Implement `ELEV_DENSIFY_MODE=off|shadow|active` with shadow diagnostics only; make gate tests green.
8. Implement active surfel creation from arc-peak world points and integrate it with Chunk-4 ID/support lifecycle.
9. Run controlled densification smokes where candidate thresholds are tuned to guarantee at least one spawn event.
10. Run real-data reduced-workload continuation test with Chunk-5 features enabled according to the rollout order.
11. Run synthetic matrix and compare only against matching renderer-fingerprint baselines.
12. Publish Chunk-5 gate report with pass/fail per test ID, material-regression check, and visual artifact verdict.

---

## Proposed Test Catalog (Chunk 5)

### Fast contract / unit tests

| ID | Focus | Pass condition |
|---|---|---|
| `C5-T01` | Normal ramp schedule | `w_normal` follows early/ramp/late contract exactly at boundary iterations |
| `C5-T02` | Expected-elevation normals handoff | `elevation=None` preserves legacy path; explicit elevation changes points/normals deterministically |
| `C5-T03` | Confidence mask contract | unsupported / high-entropy pixels are excluded; supported confident pixels are included |
| `C5-T04` | Off-mode parity gate | `ELEVATION_AWARE=0` or late-normal disabled path leaves Chunk-4-style behavior unchanged |
| `C5-T05` | Densify mode gates | `off` creates nothing, `shadow` reports candidates only, `active` permits creation |
| `C5-T06` | Arc peak selection | single-peak placement picks argmax bin; flat/unsupported profile skips deterministically |
| `C5-T07` | New-surfel lifecycle | spawned surfels receive fresh IDs and support buffers grow consistently |
| `C5-T08` | Densification signal semantics | `viewspace_points.grad` and `radii` are consumed without assuming literal `(col,row)` coordinates |
| `C5-T09` | Resume continuity | save/load preserves mode, surfel IDs, and deterministic continuation behavior |

### Smoke / integration tests

| ID | Focus | Pass condition |
|---|---|---|
| `C5-T10` | Late-normals shadow smoke | diagnostics finite; no loss aggregation break |
| `C5-T11` | Late-normals active smoke | reduced-workload run completes with finite losses and non-zero confidence coverage |
| `C5-T12` | Densify shadow smoke | candidate/peak diagnostics emitted; zero topology change |
| `C5-T13` | Densify active smoke | at least one controlled spawn event succeeds and ID integrity remains clean |
| `C5-T14` | Resume continuation smoke | checkpoint + continuation succeeds with Chunk-5 features enabled |

Recommended fast-test command order:

```bash
python -m py_compile debug_multiframe.py utils/elevation_chunk5_helpers.py
pytest tests/test_elevation_chunk5_normals_contracts.py -q
pytest tests/test_elevation_chunk5_densify_contracts.py -q
pytest tests/test_elevation_chunk5_smoke_modes.py -q
```

---

## Runtime Config Contract (Chunk 5 Defaults)

Carry forward prior chunk defaults and add these Chunk-5-specific controls:

- `ELEV_NORMAL_MODE=shadow`
- `ELEV_NORMAL_RAMP_START_ITER=4000`
- `ELEV_NORMAL_RAMP_END_ITER=8000`
- `ELEV_NORMAL_WEIGHT_EARLY=0.01`
- `ELEV_NORMAL_WEIGHT_LATE=0.10`
- `ELEV_NORMAL_ELEV_START_ITER=4000`
- `ELEV_NORMAL_CONFIDENCE_THRESH=0.5`
- `ELEV_DENSIFY=0`
- `ELEV_DENSIFY_MODE=off`
- `ELEV_STAGE2_START_ITER=12000`
- `ELEV_DENSIFY_INTERVAL=1500`
- `ELEV_DENSIFY_MIN_INTENSITY=0.15`
- `ELEV_DENSIFY_RESIDUAL_THRESH=0.10`
- `ELEV_DENSIFY_MAX_PER_EVENT=500`
- `ELEV_DENSIFY_MULTI_VIEW_MIN_SCORE=0.3`

Short-run note:
- synthetic smoke/gate runs with much shorter iteration budgets should override iteration-gated thresholds explicitly rather than implicitly changing chunk semantics.
- if a short-run command is intended to exercise late normals or densification, it must set those override env vars directly in the command block so the exercised Chunk-5 path is unambiguous in artifacts/logs.

---

## Validation Gate (Chunk 5)

All items are required to close Chunk 5:

1. **Late normals stability**
    - active late-normal run completes with no NaN/Inf,
    - expected-elevation normals path does not disconnect gradients or break loss aggregation,
    - confidence-mask coverage is finite and `>= GATE_NORMAL_CONF_COVERAGE_MIN` on at least one representative active run.

2. **Off/default safety**
    - default `ELEV_DENSIFY=0` / `ELEV_DENSIFY_MODE=off` produces no unintended topology changes,
    - off-mode parity remains within:
      - `GATE_OFFMODE_REL_LOSS_DELTA_MAX = 0.05`,
      - `GATE_OFFMODE_ABS_SSIM_DELTA_MAX = 0.01`.

3. **Densification safety**
   - `shadow` mode produces diagnostics only,
   - `active` mode can add surfels in a controlled run,
   - new surfels integrate cleanly with persistent IDs, support buffers, and prune grace.

4. **Resume continuity**
    - save/reload/continue succeeds for at least one normals-enabled run and one densify-enabled run,
    - no support-state or ID drift appears after continuation,
    - densification high-error tracker state restores deterministically when densification is enabled.

5. **Synthetic gate stability**
   - active synthetic gate runs complete end-to-end,
   - evaluator metrics are finite,
   - deltas versus the matching post-v2 post-Chunk-4 baseline obey the default material-regression rule (`>10%` relative degradation on key error metrics blocks progression unless waived explicitly).

6. **Artifact non-regression**
   - Chunk-5 must not reintroduce previously mitigated cube-class toroidal/streaking artifacts,
   - if densification improves edge/corner recovery, record it explicitly; if unchanged, record unchanged; if worse, block or waive explicitly.

---

## Synthetic Test Matrix (Chunk 5 Mandatory)

Use the active inventory and commands from `docs/SYNTHETIC_DATASET_GUIDE.md`.

| ID | Dataset class | Purpose | Required artifacts |
|---|---|---|---|
| `C5-S1` | Smooth representative (`A_clean`) | Non-regression guard for late normals with densify off | gate summary JSON/MD + evaluator JSON |
| `C5-S2` | Edge/corner representative (`C_clean`) | Primary late-normals gate on known-problem geometry class | gate summary JSON/MD + evaluator JSON + visual notes |
| `C5-S3` | `C_clean` direct run, densify off vs on | Isolate Stage-2 densification effect under matching renderer fingerprints | logs + `surfels_after_training.ply` + evaluator JSON |
| `C5-S4` | Synthetic continuation run | Resume contract with Chunk-5 features enabled | checkpoint + continuation log + evaluator JSON |

Recommended execution order:
1. `C5-S1` with late normals active and densify off.
2. `C5-S2` with late normals active and densify off.
3. `C5-S3` only after `C5-S1` and `C5-S2` are stable.
4. `C5-S4` after at least one successful `C5-S3` run.

Comparator policy:
- compare only against the latest matching post-v2 post-Chunk-4 baseline group,
- include renderer semantic fingerprints in every gate summary,
- treat pre-v2 evidence as historical only.

Recommended runnable commands (match the selected post-v2 renderer fingerprint; examples below assume the active v2 default tuple):

1. `C5-S1` smooth-shape guard with late normals active and densify off:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
ELEV_NORMAL_MODE=active ELEV_DENSIFY=0 ELEV_DENSIFY_MODE=off \
ELEV_NORMAL_RAMP_START_ITER=1 ELEV_NORMAL_RAMP_END_ITER=200 ELEV_NORMAL_ELEV_START_ITER=1 \
SONAR_RENDER_MODE=2dgs SONAR_OCCLUSION_MODE=ray_binned SONAR_LAMBERTIAN_MODE=leaky \
python scripts/run_synthetic_a_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_sphere_A_clean \
  --pose-mode sonar_equivalent \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --overwrite-runs
```

2. `C5-S2` edge/corner guard with late normals active and densify off:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
ELEV_NORMAL_MODE=active ELEV_DENSIFY=0 ELEV_DENSIFY_MODE=off \
ELEV_NORMAL_RAMP_START_ITER=1 ELEV_NORMAL_RAMP_END_ITER=200 ELEV_NORMAL_ELEV_START_ITER=1 \
SONAR_RENDER_MODE=2dgs SONAR_OCCLUSION_MODE=ray_binned SONAR_LAMBERTIAN_MODE=leaky \
python scripts/run_synthetic_c_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --pose-mode sonar_equivalent \
  --pose-policy multi_band \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --overwrite-runs
```

3. `C5-S3` direct cube run for densify-off vs densify-on comparison:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=/home/gavin/Unbiased_Surfel_sonar/synthetic_datasets/synthetic_cube_C_clean \
SONAR_OUTPUT_DIR=./output/debug_multiframe_synth_c_chunk5 \
SONAR_NUM_FRAMES=500 SONAR_STAGE2_ITERS=1000 SONAR_STAGE3_ITERS=1 SONAR_FREEZE_SCALE=1 \
ELEV_NORMAL_MODE=active ELEV_DENSIFY=1 ELEV_DENSIFY_MODE=shadow \
ELEV_NORMAL_RAMP_START_ITER=1 ELEV_NORMAL_RAMP_END_ITER=200 ELEV_NORMAL_ELEV_START_ITER=1 \
ELEV_STAGE2_START_ITER=200 ELEV_DENSIFY_INTERVAL=200 \
SONAR_RENDER_MODE=2dgs SONAR_OCCLUSION_MODE=ray_binned SONAR_LAMBERTIAN_MODE=leaky \
python debug_multiframe.py
```

If the comparison is promoted from shadow to active densification on the same reduced budget, keep the same explicit iteration overrides and change only `ELEV_DENSIFY_MODE=active` so the topology-change delta remains attributable.

Then evaluate with:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/eval_synthetic_cube.py \
  --reconstruction ./output/debug_multiframe_synth_c_chunk5/surfels_after_training.ply \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --output-dir ./output/debug_multiframe_synth_c_chunk5/eval_surfel \
  --fit-mode both
```

---

## Manual Visual Review (Required)

Because Chunk 5 targets geometry/normal quality, manual review remains mandatory alongside automated gates.

Required visual checks:
1. late-normal render/mesh comparison versus the selected post-v2 baseline,
2. `C_clean` surfel cloud / mesh check for reintroduced torus-like or streak-like artifacts,
3. densify-on versus densify-off comparison for overpopulation, noisy floaters, or edge/corner recovery.

Record artifact paths and explicit reviewer verdicts (`PASS|FAIL`) in the gate report.

---

## Deliverables

- Chunk-5 execution plan and test catalog.
- Late-normal implementation with confidence-gated expected-elevation path.
- Optional Stage-2 densification mode gate and controlled active path.
- Synthetic + resume gate evidence recorded against matching post-v2 baselines.

---

## Commit Boundary Rule

Chunk 5 is commit-ready only when:
- prerequisite gate posture is explicitly cleared or waived,
- all Chunk-5 gate checks pass,
- resume gate passes,
- no open blocker remains inside Chunk-5 scope.

Before commit, update:
- `plans/progress_overview.md`
- `plans/scientific_progress.md`

Commit message format:
- `<description> (<model-name>)`

---

## Notes

- This plan intentionally keeps densification optional and late. The first success criterion for Chunk 5 is that late normals improve or preserve geometry without destabilization.
- If late normals are stable but densification remains too risky, Chunk 5 may be split into `5A` (normals) and `5B` (densification) for implementation order, but the gate report must say so explicitly.
- If any Chunk-5 implementation decision changes the detailed-plan algorithm contract, patch `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md` before code changes proceed.
- Follow-on investigation, diagnostic rerun guidance, and zero-signal analysis were moved to `plans/PLAN_ELEVATION_AWARE_CHUNK5_5_EXECUTION_2026-03-30.md` so this document stays focused on the main Chunk-5 implementation contract.

---
