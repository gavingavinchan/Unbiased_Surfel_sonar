# Plan: Elevation-Aware Chunk 4 Execution

**Date:** 2026-02-23  
**Status:** Closeout executed; gate currently **NO-GO** on documented blockers (gpt-5.3-codex)  
**Scope:** Chunk 4 only (Belief-to-geometry enforcement)

---

## Goal

Implement mandatory belief-to-geometry enforcement so Stage-1 elevation belief updates actually move surfel geometry and improve multi-view retention quality.

Primary outcomes:
- Coupling loss (`expected-point -> surfel`) is active, finite, and measurable.
- Surfel support state is tracked by persistent surfel IDs across densify/prune/reorder.
- Support retention/pruning follows warmup + threshold schedule + hysteresis instead of FOV-only heuristics.
- Synthetic cube-class failure mode (toroidal/streaking smearing) is explicitly evaluated and directionally improved, or blocked with written rationale.

---

## Source of Truth

- Algorithm and contracts:
  - `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md`
- Chunk sequencing and mandatory gates:
  - `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md`
- Design ledger / decisions:
  - `plans/PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md`
- Prior chunk references:
  - `plans/PLAN_ELEVATION_AWARE_CHUNK1_EXECUTION_2026-02-10.md`
  - `plans/PLAN_ELEVATION_AWARE_CHUNK2_EXECUTION_2026-02-11.md`
  - `plans/PLAN_ELEVATION_AWARE_CHUNK3_EXECUTION_2026-02-16.md`
- Active synthetic inventory and commands:
  - `docs/SYNTHETIC_DATASET_GUIDE.md`

If any mismatch appears, follow the detailed plan for contracts and the implementation-execution plan for chunk gate policy.

---

## Chunk-3 Handoff Preconditions

Chunk 4 depends on Chunk-3 Stage-1 cache interfaces:
- `cached_loglik[frame_a]`
- `cached_support_mask[frame_a]`
- `p_post`

Precondition policy before gate close:
1. Chunk-3 parity gap status (overlap-neighbor `back_project_bins` evidence contract) must be recorded at Chunk-4 start.
2. If still open, Chunk-4 results are valid only with an explicit caveat that coupling is evaluated against the current Stage-1 evidence provider.
3. Chunk-4 gate close requires either:
   - Chunk-3 parity gap closed, or
   - explicit written waiver for parity gap carryover in Chunk-4 gate notes.

---

## In-Scope Files (Chunk 4)

Primary implementation target:
- `debug_multiframe.py`

Likely helper modules (recommended for import-safe TDD):
- `utils/elevation_stage1_helpers.py` (reuse/extend where appropriate)
- `utils/elevation_chunk4_helpers.py` (new pure helpers for association/support lifecycle)

Hard dependency contracts (must stay consistent):
- `utils/sonar_utils.py` (`back_project_bins` conventions)
- `gaussian_renderer/__init__.py` (`sonar_project_points` and `SonarProjection.valid` contract)

Optional config exposure only if needed:
- `arguments/__init__.py`

Out of scope for Chunk 4:
- Normals ramp and expected-elevation normals path (Chunk 5)
- Optional Stage 2 densification hook behavior changes (Chunk 5)
- Synthetic dataset generation changes (use current active inventory only)

---

## Contracts to Implement in Chunk 4

1. Mandatory coupling loss contract
   - Use same sampled frames and same-iteration caches from Stage 1.
   - Compute expected points from posterior over bins:
     - `pts_exp = sum_k p_post[k] * pts_bins[k]`.
   - Associate expected points to surfels with gating + soft weights.
   - Add coupling term in unified loss block exactly once.

2. Association contract (`associate_expected_points_to_surfels`)
   - Candidate gates:
     - `pix_err <= ELEV_COUPLE_MAX_PIX_ERR`
     - `depth_err <= ELEV_COUPLE_MAX_DEPTH_ERR`
     - projection validity true for both expected point and candidate surfel.
   - Score:
     - `s = (pix_err / sigma_pix)^2 + (depth_err / sigma_depth)^2`
   - Weight:
     - `assoc_w = clamp(exp(-0.5 * s), ELEV_COUPLE_MIN_W, 1.0)`
   - Return per-point arrays:
     - `surf_idx`, `assoc_w`, `match_valid`.

3. Coupling schedule contract
   - Warmup/ramp via existing weights:
     - `ELEV_COUPLE_WEIGHT_START`, `ELEV_COUPLE_WEIGHT_END`, `ELEV_COUPLE_WARMUP`.
   - Coupling must be bounded and finite under sparse/no-match frames.
   - Zero-match frames contribute zero without sentinel-index side effects.

4. Persistent surfel ID contract
   - Maintain `surfel_ids`, `next_surfel_id`, and `id_to_row` mapping.
   - Densify: assign fresh monotonic IDs.
   - Prune/reorder: update `surfel_ids` and rebuild `id_to_row` deterministically.
   - No ID compaction in v1 (accepted monotonic growth).

5. ID-keyed support buffers contract
   - Maintain by surfel ID (not row index):
     - `ema_by_id`, `last_raw_by_id`, `birth_iter_by_id`.
   - Update support by ID each iteration after association/support evidence assembly.
   - Keep retired IDs in tensors in v1.

6. Support evidence contract (multi-view, not FOV-only)
   - A frame contributes support only if all true:
     - valid projection,
     - meaningful GT return,
     - residual under threshold.
   - Enforce viewpoint diversity with minimum pose-angle separation.

7. Support/prune schedule contract
   - Warmup: accumulate support only, no hard prune.
   - Mid/Late thresholds:
     - ratio checks (`ELEV_SUPPORT_MIN_RATIO_MID/LATE`)
     - count floors (`ELEV_SUPPORT_MIN_COUNT_MID/LATE`) with effective-floor clamp:
       - `floor_eff = min(config_floor, diverse_candidate_count)`.
   - Hysteresis:
     - require below-threshold persistence for `ELEV_SUPPORT_PRUNE_PATIENCE` checks.

8. New-surfel grace contract
   - If enabled, skip hard prune checks for IDs where:
     - `iter - birth_iter_by_id[sid] < ELEV_SUPPORT_NEW_SURFEL_GRACE_ITERS`.
   - Support still accumulates during grace.

9. Checkpoint schema and resume contract for Chunk 4 state
   - Persist and restore:
     - `surfel_ids`, `next_surfel_id`, `id_to_row` rebuild state,
     - ID-keyed support tensors,
     - coupling/support scheduler state needed for deterministic continuation.
   - Introduce schema marker:
     - `checkpoint_schema_version = "chunk4_coupling_support_v1"`.
   - Deterministic mismatch handling must follow explicit policy (fail-fast strict by default unless reset mode explicitly configured).

10. Diagnostics contract
    - Coupling:
      - match rate, residual mean/p95, weighted/unweighted coupling loss.
    - Support:
      - `support>=2`, `support>=3`, median support,
      - prune counts by reason,
      - grace-active counts,
      - ID integrity stats (duplicate IDs, invalid `id_to_row` entries).
    - Geometry-risk tracking:
      - frame-dominance metrics from Chunk 2/3 continuity,
      - explicit cube-smearing artifact verdict (`improved|unchanged|regressed`).

---

## Implementation Method (TDD-First)

1. Add/extend fast failing tests before coupling/support code changes.
2. Keep pure math/state helpers import-safe and unit-testable outside `debug_multiframe.py` runtime.
3. Rollout in shadow-first mode before active enforcement:
   - compute association/support diagnostics first,
   - then enable weighted coupling and hard prune behavior.
4. Keep one unified loss aggregation block and add coupling exactly once.
5. Keep all new behavior behind explicit gates so baseline behavior remains reproducible.

Recommended mode flags for safe rollout:
- `ELEV_COUPLE_MODE=off|shadow|active`
- `ELEV_SUPPORT_MODE=off|shadow|active`

Mode semantics:
- `off`: no coupling/support enforcement.
- `shadow`: compute diagnostics/state updates but do not apply coupling weight or hard prune.
- `active`: full coupling + support-prune enforcement.

---

## Step-by-Step Work Order

1. Write failing tests for association + coupling reduction contracts (`C4-T02`, `C4-T03`).
2. Write failing tests for persistent-ID lifecycle and by-ID support updates (`C4-T04`, `C4-T05`).
3. Write failing tests for support schedule/effective-floor/hysteresis/new-surfel grace (`C4-T06`, `C4-T07`, `C4-T08`).
4. Write failing schema/resume negative tests (`C4-T09`) and mode-gate parity tests (`C4-T10`, `C4-T11`).
5. Implement coupling mode gates and association helper; make `C4-T02`, `C4-T03`, `C4-T10` green.
6. Integrate coupling once in unified loss block; re-run `C4-T03` + smoke `C4-T12`.
7. Implement persistent surfel-ID lifecycle hooks; make `C4-T04` green.
8. Implement ID-keyed support buffers and updates; make `C4-T05` green.
9. Implement support schedule + effective floors + patience + grace; make `C4-T06`/`C4-T07`/`C4-T08` green.
10. Implement checkpoint schema/save/load for Chunk-4 state; make `C4-T09` and resume smoke `C4-T14` green.
11. Run short runtime smokes (`C4-T12`, `C4-T13`) in `shadow` first, then `active`.
12. Run synthetic matrix (`C4-S1`..`C4-S4`) and evaluate deltas against active baselines.
13. Run explicit off-mode parity check (`C4-T11`) against Chunk-3-style baseline.
14. Publish gate report with thresholds, pass/fail per test ID, and blocker/waiver decisions.

---

## Deterministic Fixture Contract (TDD)

To reduce flake in red/green cycles, all fast tests must use deterministic fixtures:

- Seed contract for unit/smoke tests:
  - `random.seed(42)`
  - `numpy.random.seed(42)`
  - `torch.manual_seed(42)`
- Tiny synthetic surfel fixture for lifecycle tests:
  - start with `N=8` surfels and fixed coordinates,
  - apply one deterministic densify event (`+3`) and one deterministic prune/reorder event.
- Projection/association fixture:
  - fixed camera pose and fixed expected-point batch with known gate pass/fail cases.
- Support-schedule fixture:
  - fixed `diverse_candidate_count` vectors including low-count cases to validate effective-floor behavior.
- Resume/schema fixture:
  - save payload, mutate `checkpoint_schema_version` or frame-set fingerprint, then validate strict failure/reset behavior deterministically.

---

## Quantitative Gate Thresholds (Chunk 4)

These are default pass thresholds for Chunk-4 gate evidence. If changed, the gate report must explicitly justify the override.

- `GATE_FINITE_NAN_INF = 0` (no NaN/Inf in coupling/support diagnostics).
- `GATE_MIN_MATCH_RATE = 0.01` (median coupling match rate over the last 20% iterations in active mode).
- `GATE_MAX_P95_COUPLE_RESIDUAL_M = 0.30` (active-mode p95 coupling residual over the last 20% iterations).
- `GATE_MAX_DUPLICATE_ACTIVE_IDS = 0`.
- `GATE_MAX_INVALID_ID_TO_ROW = 0`.
- `GATE_WARMUP_PRUNE_COUNT = 0` for iterations `< ELEV_SUPPORT_WARMUP_ITERS` in active mode.
- `GATE_OFFMODE_REL_LOSS_DELTA_MAX = 0.05` (off-mode parity vs Chunk-3-style baseline, short smoke aggregate).
- `GATE_OFFMODE_ABS_SSIM_DELTA_MAX = 0.01` (off-mode parity vs baseline, short smoke aggregate).

---

## Runtime Config Contract (Chunk 4 Core)

Carry forward Chunk-3 defaults and add/activate these controls for Chunk 4:

- Coupling controls:
  - `ELEV_COUPLE_MODE=shadow` (promote to `active` after sanity)
  - `ELEV_COUPLE_WEIGHT_START=0.10`
  - `ELEV_COUPLE_WEIGHT_END=0.50`
  - `ELEV_COUPLE_WARMUP=2000`
  - `ELEV_COUPLE_MAX_PIX_ERR=3.0`
  - `ELEV_COUPLE_MAX_DEPTH_ERR=0.08`
  - `ELEV_COUPLE_HUBER_DELTA=0.03`
  - `ELEV_COUPLE_SIGMA_MODE=fixed`
  - `ELEV_COUPLE_SIGMA_PIX=2.0`
  - `ELEV_COUPLE_SIGMA_DEPTH=0.05`
  - `ELEV_COUPLE_MIN_W=0.10`

- Support/pruning controls:
  - `ELEV_SUPPORT_MODE=shadow` (promote to `active` after sanity)
  - `ELEV_SUPPORT_WARMUP_ITERS=4000`
  - `ELEV_SUPPORT_USE_PERSISTENT_IDS=1`
  - `ELEV_SUPPORT_USE_RATIO=1`
  - `ELEV_SUPPORT_USE_NEW_SURFEL_GRACE=1`
  - `ELEV_SUPPORT_NEW_SURFEL_GRACE_ITERS=1500`
  - `ELEV_SUPPORT_MIN_RATIO_MID=0.25`
  - `ELEV_SUPPORT_MIN_RATIO_LATE=0.45`
  - `ELEV_SUPPORT_MIN_COUNT_MID=2`
  - `ELEV_SUPPORT_MIN_COUNT_LATE=4`
  - `ELEV_SUPPORT_VIEW_ANGLE_MIN_DEG=8`
  - `ELEV_SUPPORT_RESIDUAL_THRESH=0.20`
  - `ELEV_SUPPORT_EMA_DECAY=0.90`
  - `ELEV_SUPPORT_PRUNE_PATIENCE=4`
  - `ELEV_SURFEL_ID_ASSERTS=1`

Compatibility rule:
- If `ELEVATION_AWARE=0`, effective behavior must force `ELEV_COUPLE_MODE=off` and `ELEV_SUPPORT_MODE=off`.

---

## Validation Gate (Chunk 4)

All items required to close Chunk 4:

1. Short reduced-workload `active` run completes with zero NaN/Inf diagnostics (`GATE_FINITE_NAN_INF`).
2. Coupling metrics satisfy quantitative thresholds:
   - median match rate over last 20% iters `>= GATE_MIN_MATCH_RATE`,
   - p95 coupling residual over last 20% iters `<= GATE_MAX_P95_COUPLE_RESIDUAL_M`,
   - association weights stay in `[ELEV_COUPLE_MIN_W, 1.0]`.
3. ID integrity checks pass across topology edits:
   - duplicate active IDs `<= GATE_MAX_DUPLICATE_ACTIVE_IDS`,
   - invalid `id_to_row` entries `<= GATE_MAX_INVALID_ID_TO_ROW`,
   - support-state continuity verified before/after prune/reorder test events.
4. Support/pruning schedule checks pass:
   - prune count is exactly `GATE_WARMUP_PRUNE_COUNT` for `iter < ELEV_SUPPORT_WARMUP_ITERS`,
   - post-warmup prune decisions obey ratio/count thresholds with effective floors,
   - hysteresis enforces `ELEV_SUPPORT_PRUNE_PATIENCE` consecutive failures before prune.
5. Mode-gate checks pass in order:
   - `off` mode parity versus Chunk-3-style baseline is within `GATE_OFFMODE_REL_LOSS_DELTA_MAX` and `GATE_OFFMODE_ABS_SSIM_DELTA_MAX`,
   - `shadow` mode emits coupling/support diagnostics with no weighted coupling and no hard prune,
   - `active` mode enables weighted coupling and hard-prune path.
6. Resume and schema mismatch gates pass:
   - save/load/continue in matching schema succeeds,
   - strict schema mismatch fails fast with actionable error,
   - explicit reset policy path (if configured for test) behaves deterministically.
7. Synthetic shape-diversity validation completes using active inventory representatives:
   - smooth-shape class (Dataset A / sphere),
   - edge/corner-shape class (Dataset C / cube).
8. Synthetic metrics are finite and satisfy default material-regression rule vs active baseline (`<=10%` degradation on key error metrics unless explicit approved waiver).
9. For known-problem class (Dataset C), directional improvement vs Chunk-2 baseline is reported (or explicit blocker documented):
   - key evaluator metrics,
   - artifact-based visual verdict for toroidal/streaking smearing.
10. Chunk-2/3 cross-view diagnostics continuity remains present (`final_eval_*`, `support_metrics_*`, `frame_training_visits.csv`, dominance proxies).

---

## Baseline Comparator Policy (Chunk 4)

Primary comparison anchors:

1. Chunk-2 fixed-opacity baseline (higher-budget run C):
   - `support>=2 = 0.4615`
   - `support>=3 = 0.0446`
   - `median_support = 1.0`
   - `single_view_top_share(train) = 0.5606`
   - `single_view_top_share(train+holdout) = 0.7249`
   - `holdout_loss/train_loss = 1.52x`

2. Chunk-3 Dataset-C comparator (carry-risk context):
   - `output/debug_multiframe_synth_c_exp3/eval_surfel/cube_eval.json`
   - `output/debug_multiframe_synth_c_run1/eval_surfel/cube_eval.json`
   - Known issue: center-error material regression was waived for transition; Chunk 4 must explicitly report movement on this axis.

Policy:
- Use fixed-opacity path as primary comparator unless an explicit ablation objective says otherwise.
- Record absolute metrics and deltas for each gate run.
- Any material regression requires blocker note or explicit waiver approval.

---

## Synthetic Test Matrix (Chunk 4 Mandatory)

Reference dataset inventory and commands in `docs/SYNTHETIC_DATASET_GUIDE.md`.

| ID | Dataset class | Command path | Purpose | Required artifacts |
|---|---|---|---|---|
| C4-S1 | Smooth representative (A_clean) | `scripts/run_synthetic_a_gate.py` | Non-regression guard while enabling coupling/support | gate summary JSON/MD + evaluator JSON |
| C4-S2 | Edge/corner representative (C_clean) | `scripts/run_synthetic_c_gate.py` | Primary geometry-smearing improvement gate | gate summary JSON/MD + evaluator JSON + visual notes |
| C4-S3 | C_clean direct active-coupling run | `debug_multiframe.py` + `eval_synthetic_cube.py` | Isolate Chunk-4 coupling/support behavior under explicit mode flags | run log + `surfels_after_training.ply` + evaluator JSON |
| C4-S4 | C_clean continuation | save/load/continue + evaluator | Resume contract for IDs/support/coupling state | checkpoint + continuation log + evaluator JSON |

### Recommended runnable commands

1) Smooth-shape guard (`C4-S1`):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
ELEV_COUPLE_MODE=active ELEV_SUPPORT_MODE=active \
python scripts/run_synthetic_a_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_sphere_A_clean \
  --pose-mode sonar_equivalent \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --overwrite-runs
```

2) Edge/corner gate (`C4-S2`):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
ELEV_COUPLE_MODE=active ELEV_SUPPORT_MODE=active \
python scripts/run_synthetic_c_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --pose-mode sonar_equivalent \
  --pose-policy multi_band \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --overwrite-runs
```

3) Direct C-clean active run (`C4-S3`):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=./synthetic_datasets/synthetic_cube_C_clean \
SONAR_OUTPUT_DIR=./output/chunk4_s3_active \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=1000 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
ELEVATION_AWARE=1 \
ELEV_STAGE1_MODE=active \
ELEV_COUPLE_MODE=active \
ELEV_SUPPORT_MODE=active \
python debug_multiframe.py
```

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/eval_synthetic_cube.py \
  --reconstruction ./output/chunk4_s3_active/surfels_after_training.ply \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --output-dir ./output/chunk4_s3_active/eval_surfel \
  --fit-mode both
```

4) Synthetic continuation (`C4-S4`):

First run (save checkpoint):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=./synthetic_datasets/synthetic_cube_C_clean \
SONAR_OUTPUT_DIR=./output/chunk4_s4_run1 \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=600 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
ELEVATION_AWARE=1 \
ELEV_STAGE1_MODE=active \
ELEV_COUPLE_MODE=active \
ELEV_SUPPORT_MODE=active \
SONAR_SAVE_CHECKPOINT=./output/chunk4_s4_run1/chunk4_ckpt.pth \
python debug_multiframe.py
```

Continuation run (load checkpoint):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=./synthetic_datasets/synthetic_cube_C_clean \
SONAR_OUTPUT_DIR=./output/chunk4_s4_run2 \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=600 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
ELEVATION_AWARE=1 \
ELEV_STAGE1_MODE=active \
ELEV_COUPLE_MODE=active \
ELEV_SUPPORT_MODE=active \
SONAR_LOAD_CHECKPOINT=./output/chunk4_s4_run1/chunk4_ckpt.pth \
python debug_multiframe.py
```

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/eval_synthetic_cube.py \
  --reconstruction ./output/chunk4_s4_run2/surfels_after_training.ply \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --output-dir ./output/chunk4_s4_run2/eval_surfel \
  --fit-mode both
```

### Required recording for each synthetic run

- Full command/config.
- Output artifact paths.
- Evaluator metrics.
- Delta versus active baseline.
- For Dataset C: explicit artifact verdict (`improved|unchanged|regressed`) for toroidal/streaking smearing.

---

## Chunk 4 Test Catalog (Planned)

Test IDs are mandatory for red/green tracking and gate reporting.

### Fast unit/contract tests (must go red first)

- `C4-T01` syntax/import compile for touched files.
- `C4-T02` association-gating contract:
  - pixel/depth threshold behavior,
  - score and weight formula,
  - weight clamp in `[ELEV_COUPLE_MIN_W, 1.0]`,
  - no-match behavior returns `match_valid=False` without invalid gather indices.
- `C4-T03` coupling reduction contract:
  - finite weighted Huber in sparse and dense match fixtures,
  - zero-match frame contributes exact zero term.
- `C4-T04` persistent-ID lifecycle:
  - densify ID allocation monotonic/non-overlapping,
  - prune/reorder mapping correctness,
  - duplicate-ID fail-fast.
- `C4-T05` support-buffer by-ID stability:
  - update-by-ID is invariant to row reindexing,
  - no support drift after topology mutation.
- `C4-T06` support schedule + effective floors:
  - warmup bypass,
  - mid/late ratio and floor checks,
  - `floor_eff = min(config_floor, diverse_candidate_count)` behavior.
- `C4-T07` hysteresis/patience:
  - prune only after `ELEV_SUPPORT_PRUNE_PATIENCE` consecutive failures.
- `C4-T08` new-surfel grace:
  - grace skip before age threshold,
  - hard checks resume exactly at/after threshold.
- `C4-T09` checkpoint schema/resume negative tests (`chunk4_coupling_support_v1`):
  - payload roundtrip,
  - strict mismatch fail-fast,
  - explicit reset-policy behavior deterministic when configured.
- `C4-T10` mode-gate contract:
  - `off|shadow|active` behavior for coupling/support terms and prune path.

### Integration/smoke tests

- `C4-T11` off-mode parity smoke:
  - same seed/config, `ELEV_COUPLE_MODE=off`, `ELEV_SUPPORT_MODE=off`,
  - compare against Chunk-3-style baseline,
  - enforce `GATE_OFFMODE_REL_LOSS_DELTA_MAX` and `GATE_OFFMODE_ABS_SSIM_DELTA_MAX`.
- `C4-T12` short real-data smoke (`shadow`):
  - finite diagnostics,
  - no hard prune,
  - coupling contribution disabled.
- `C4-T13` short real-data smoke (`active`):
  - coupling/support active,
  - finite metrics,
  - thresholds from Quantitative Gate section met.
- `C4-T14` resume smoke (real):
  - checkpoint save/load/continue,
  - continuity checks for ID/support/coupling runtime state.
- `C4-T15` synthetic matrix execution (`C4-S1`..`C4-S4`) with baseline deltas.
- `C4-T16` synthetic continuation resume-state check:
  - verify ID/support continuity after synthetic checkpoint continuation.

### Manual visual checks (required supplement)

- `C4-T17` Dataset-C artifact panel review:
  - `surfels_after_training.ply`,
  - `mesh_poisson_after_stage3.ply`,
  - representative `comparison_after_stage3_frame*.png`,
  - verdict: `improved|unchanged|regressed` vs baseline toroidal/streaking failure.

### Recommended test file layout

- `tests/test_elevation_chunk4_coupling_contracts.py` (`C4-T02`, `C4-T03`, `C4-T10`)
- `tests/test_elevation_chunk4_id_support_lifecycle.py` (`C4-T04`..`C4-T08`)
- `tests/test_elevation_chunk4_checkpoint_contracts.py` (`C4-T09`)
- `tests/test_elevation_chunk4_smoke_modes.py` (`C4-T11`..`C4-T14`)

---

## Gate Evidence Checklist (Chunk 4 Closeout)

| Gate item | Required artifact path(s) |
|---|---|
| Fast contract tests pass (`C4-T01`..`C4-T10`) | pytest logs for Chunk-4 test files |
| Coupling metrics meet quantitative thresholds | `output/.../run.log`, `loss_log.csv`, threshold summary table |
| ID integrity across topology edits | dedicated test log + runtime assert logs |
| Support schedule/patience correctness | test logs + runtime summary CSV/log |
| Off-mode parity pass (`C4-T11`) | baseline-vs-off parity report (`loss/ssim` deltas) |
| Resume continuity for Chunk-4 state (`C4-T14`, `C4-T16`) | checkpoint path + continuation run log |
| Schema mismatch negative-path pass (`C4-T09`) | strict-fail log + optional reset-policy test log |
| Synthetic smooth-shape non-regression | Dataset-A gate summary + evaluator JSON |
| Synthetic edge/corner improvement tracking | Dataset-C gate summary + evaluator JSON + visual verdict notes |
| Material-regression rule evaluation | baseline-delta table in chunk report |
| Cross-view diagnostics continuity | `final_eval_*`, `support_metrics_*`, `frame_training_visits.csv` |
| Test-ID pass/fail ledger published | chunk report with `C4-Txx` and `C4-Sx` verdicts |
| Docs updated for commit boundary | `plans/progress_overview.md`, `plans/scientific_progress.md` |

---

## Deliverables

- Chunk-4 implementation for coupling, persistent IDs, and support/pruning lifecycle.
- Test suite additions for association/support/checkpoint contracts.
- Run artifacts for real + synthetic gates with explicit baseline deltas.
- Manual visual verdict notes for known Dataset-C geometry artifact class.
- Gate-close report with go/no-go decision and any waiver text.

---

## Commit Boundary Rule

Chunk 4 is commit-ready only when:
- Chunk-4 validation gate passes,
- resume gate passes (including synthetic continuation),
- synthetic deltas and artifact verdicts are recorded,
- `plans/progress_overview.md` and `plans/scientific_progress.md` are updated,
- no blocker remains in Chunk-4 scope (or an explicit approved waiver is documented).

Commit message format (repo convention):
- `<description> (<model-name>)`

Example:
- `Implement coupling, persistent surfel IDs, and support-pruning enforcement for elevation Stage 1 (gpt-5.3-codex)`

---

## 2026-02-24 Closeout Execution Snapshot

Closeout/gating was executed with consolidated evidence in:

- `output/chunk4_closeout/chunk4_gate_closeout_report_2026-02-24.md`
- `output/chunk4_closeout/chunk4_gate_closeout_report_2026-02-24.json`
- `output/chunk4_closeout/gate_logs/`

Test/gate ledger summary from this closeout pass:

- `C4-T01`..`C4-T16`: pass.
- `C4-T17`: manual placeholder skipped (expected by contract).
- `C4-S1`: pass.
- `C4-S2`: fail.
- `C4-S3`: fail.
- `C4-S4`: continuation path pass, evaluator threshold fail.

Key quantitative gate outcomes:

- Off-mode parity (`C4-T11`) passes (`rel_loss_delta=0.0012866`, `abs_ssim_delta=0.0006394`).
- Active coupling tail match rate passes (`median=0.9465`).
- Active coupling tail residual gate narrowly fails (`p95_of_p95=0.30195 m` vs `<=0.30 m`).
- Assoc weight bounds pass (`0.662..0.772` within `[0.10,1.0]`).

Gate decision at this snapshot: **NO-GO**

Blockers requiring resolution or explicit waiver:

1. Dataset-C synthetic gate (`C4-S2`) remains threshold-failing.
2. Coupling residual threshold miss by `0.00195 m`.
3. Manual artifact verdict (`C4-T17`) not yet recorded.

---

## 2026-02-24 Post-Closeout Probes and Observations Addendum

Additional directed probes were run after the initial closeout snapshot to test whether stronger Chunk-4 enforcement (especially support-prune pressure) can produce meaningful cube-shape correction.

### Probe A: Aggressive (stronger but not extreme)

- Run directory: `output/chunk4_aggressive_probe_run1/`
- Core intent: increase coupling strength and make support-prune activate earlier without immediate collapse.
- Key results:
  - Eval (`output/chunk4_aggressive_probe_run1/eval_surfel/cube_eval.json`):
    - `mean_surface_error_m = 0.088276`
    - `p95_surface_error_m = 0.231174`
    - `center_error_m = 0.022764`
    - `overall_pass = false`
  - Movement vs Chunk-3 baseline (`output/debug_multiframe_synth_c_run1/surfels_after_training.ply`):
    - symmetric NN mean/p95/max = `0.006794 / 0.016430 / 0.080364` m
  - Support-prune engagement from run log:
    - nonzero prune events: `1`
    - max per-iter prune: `3`
    - total pruned: `3`
- Interpretation: geometry changed only modestly; still visually/functionally close to Chunk-3 outcome.

### Probe B: Harsh (force prune engagement)

- Run directory: `output/chunk4_aggressive_probe_run2_harsh/`
- Core intent: force hard support-prune engagement (near-zero warmup, strict thresholds, low patience).
- Key results:
  - Eval (`output/chunk4_aggressive_probe_run2_harsh/eval_surfel/cube_eval.json`):
    - `mean_surface_error_m = 0.124671`
    - `p95_surface_error_m = 0.280349`
    - `center_error_m = 0.083447`
    - `overall_pass = false`
  - Movement vs Chunk-3 baseline (`output/debug_multiframe_synth_c_run1/surfels_after_training.ply`):
    - symmetric NN mean/p95/max = `0.258681 / 0.930141 / 1.183765` m
  - Support-prune engagement from run log:
    - nonzero prune events: `23`
    - max per-iter prune: `80984`
    - total pruned: `83083`
    - final surfel count: `1571`
- Manual visual verdict (recorded):
  - Note path: `output/chunk4_aggressive_probe_run2_harsh/manual_visual_note_2026-02-24.md`
  - Observed behavior: torus did not move toward cube geometry; approximately half the torus disappeared.
  - Verdict: `regressed` (collapse-by-pruning, not corrective reshaping).

### Consolidated post-probe assessment

- Forcing stronger support-prune/coupling is not sufficient and appears to be the wrong primary direction for this failure mode.
- Mild aggression gives only small geometric displacement; harsh aggression causes deletion/collapse rather than cube-shape convergence.
- Chunk-4 therefore remains **NO-GO** and blocked on Dataset-C geometry behavior.
- Working hypothesis for next stage: bottleneck is likely upstream evidence/association quality (Stage-1 posterior/evidence fidelity), not simply enforcement strength.

### 2026-02-25 Quantified Diagnostics (requested follow-up)

Using existing run artifacts (`chunk4_closeout`, `chunk4_aggressive_probe_run1`, `chunk4_aggressive_probe_run2_harsh`), three diagnostics were executed.

1) Criterion-level failure map (default vs aggressive)

- Default closeout (`output/chunk4_closeout/c4_s2_run1/`):
  - Coupling diagnostics: finite pass, match-rate pass (`0.9465`), assoc-weight bounds pass (`0.662..0.772`), residual gate fail (`p95_of_p95=0.30195 > 0.30`).
  - Dataset-C evaluator gates: mean fail (`0.087923 > 0.05`), p95 fail (`0.227167 > 0.10`), center pass (`0.023750 <= 0.03`).
- Aggressive probe (`output/chunk4_aggressive_probe_run1/`):
  - Coupling diagnostics: finite pass, match-rate pass (`0.957`), residual pass (`~0.284`), assoc-weight bounds pass (`~0.689..0.789`).
  - Dataset-C evaluator gates: mean fail (`0.088276 > 0.05`), p95 fail (`0.231174 > 0.10`), center pass (`0.022764 <= 0.03`).
- Harsh probe (`output/chunk4_aggressive_probe_run2_harsh/`):
  - Coupling diagnostics: finite pass, match-rate pass (`0.596`), residual pass (`~0.29275`), assoc-weight bounds fail (`assoc_w_min~0.0`).
  - Dataset-C evaluator gates: mean fail (`0.124671 > 0.05`), p95 fail (`0.280349 > 0.10`), center fail (`0.083447 > 0.03`).

Net: stronger enforcement can clear the closeout residual gate but does not close Dataset-C geometry thresholds; harsh settings regress to collapse.

2) Coupling-weight utilization and prune-eligibility timing

- Default closeout config (`Stage2=1000`, `couple_weight=0.1->0.5`, `warmup=2000`):
  - effective coupling ramp utilization by Stage-2 end: `50%`;
  - effective `w_couple` at Stage-2 end: `~0.300` (not full configured end weight).
- Aggressive probe (`Stage2=800`, `couple_weight=0.5->1.2`, `warmup=200`):
  - ramp utilization by Stage-2 end: `100%`;
  - end-of-Stage-2 `w_couple`: `1.2`.
- Harsh probe (`Stage2=800`, `couple_weight=1.2->2.0`, `warmup=50`):
  - ramp utilization by Stage-2 end: `100%`;
  - end-of-Stage-2 `w_couple`: `2.0`.

Prune timing from logs:
- Default/aggressive: only sparse prune activity (first nonzero at iter `100`, aligned with FOV-prune cadence, total `3`).
- Harsh: early non-FOV prune engagement (first at iter `30`), `23` nonzero events, total pruned `83083`, max single event `80984`.

3) Stage-1 posterior sharpness / entropy trend on Dataset-C

- 7-bin maximum entropy is `ln(7)=1.9459`.
- Stage-2 logged entropies remain close to this ceiling in all three configs:
  - default mean entropy `1.9175` (`~98.54%` of max),
  - aggressive mean entropy `1.9180` (`~98.56%` of max),
  - harsh mean entropy `1.9180` (`~98.56%` of max).
- Late-phase entropy decreases slightly but remains high-information-poor (default late mean `1.9130`, still `~98.31%` of max).

Interpretation: posterior evidence remains broad/weak; changing Chunk-4 enforcement strength alone does not materially sharpen Stage-1 belief quality for cube-shape correction.

## 2026-02-25 Diagnosis Addendum (No-Fix Pass)

Findings from code/tests/runtime inspection without applying fixes:

- Runtime wiring and contracts are present and executable (Chunk-4 helper/contract tests and opt-in runtime/synthetic smokes are green in this workspace).
- The failure mode is gate-quality, not a hard crash/import fault: Dataset-C acceptance remains the blocking axis.
- Current default schedule materially under-activates enforcement in typical short/medium budgets:
  - coupling warmup (`ELEV_COUPLE_WARMUP=2000`) exceeds the common Stage-2 budget (`SONAR_STAGE2_ITERS=1000`), so active coupling weight only reaches the ramp midpoint by Stage-2 end;
  - support warmup (`ELEV_SUPPORT_WARMUP_ITERS=4000`) delays hard support-prune eligibility well beyond the same budget.
- Stage-1 posterior signals observed during active runs remain high-entropy/low-information (entropy near the 7-bin maximum), consistent with weakly discriminative expected-point supervision for Chunk-4 coupling.
- Directed aggression in support-prune pressure can increase movement but tends to produce deletion/collapse behavior rather than corrective cube-shape convergence.

Immediate diagnostic work items queued from this addendum:

1. Build a criterion-level failure map versus gate thresholds for default vs aggressive settings.
2. Quantify effective coupling-weight utilization and first prune eligibility timing under each config.
3. Track Stage-1 posterior sharpness (entropy/proxy confidence) over time on Dataset-C to validate the evidence-quality bottleneck hypothesis.

## 2026-02-25 Independent Diagnosis Addendum (gpt-5.3-codex)

Method constraints for this addendum:
- Diagnosis derived from code and run artifacts only.
- No reliance on post-line-671 opinion sections of this plan.

### What is wrong with Chunk-4 performance

1. **Support-prune under-activation was observed in the original closeout config, but this is no longer the code default.**
   - Historical closeout run used a long warmup relative to budget (`Stage2=1000`), delaying hard support-prune eligibility.
   - Current code computes budget-scaled defaults in `parse_elevation_chunk4_config` (`debug_multiframe.py`:2649-2660), so warmup is no longer a fixed long constant by default.
   - `compute_support_failure_mask` still correctly bypasses failures during warmup by contract (`utils/elevation_chunk4_helpers.py`:263-264).
   - Practical implication: diagnose with the *effective runtime warmup values in logs/env* rather than assuming `4000`.

2. **Coupling under-ramp was true for the original closeout config, but current defaults are budget-scaled.**
   - Historical closeout used a long coupling warmup relative to Stage-2 budget, so `w_couple` did not reach end weight.
   - Current code computes warmup from Stage-2 budget (`debug_multiframe.py`:2649-2654) and then applies `resolve_chunk4_coupling_weight` (`debug_multiframe.py`:991-997).
   - Practical implication: verify per-run `ELEV_COUPLE_WARMUP` and observed `w_couple` instead of assuming a fixed `2000` default.

3. **Stage-1 evidence quality remains a likely bottleneck, but parity wiring has progressed.**
   - Active Stage-1 likelihood is assembled through overlap-neighbor multi-view projection in `build_stage1_multiview_loglik` (`debug_multiframe.py`:637-743), including `back_project_bins` and neighbor projections.
   - Overlap table is consumed per iteration via `resolve_stage1_overlap_neighbors` + neighbor loop (`debug_multiframe.py`:680-687).
   - Remaining concern is *information quality* (posterior sharpness/discriminativeness), not absence of overlap wiring.

4. **Measured geometry movement is negligible in default/aggressive runs and destructive in harsh runs.**
   - Chunk-4 closeout vs Chunk-3 comparator (cube eval):
     - mean error delta: `-0.000050 m`
     - p95 error delta: `-0.001689 m`
     - center error delta: `+0.000111 m`
     - Sources: `output/chunk4_closeout/c4_s2_run1/eval_surfel/cube_eval.json`, `output/debug_multiframe_synth_c_run1/eval_surfel/cube_eval.json`.
   - Aggressive run movement vs Chunk-3 baseline remains small (symmetric NN mean `0.00679 m`) (`output/chunk4_aggressive_probe_run1/movement_vs_chunk3.json`).
   - Harsh run forces prune engagement but collapses geometry (final surfels `1571`; much worse cube metrics) (`output/chunk4_aggressive_probe_run2_harsh/run.log`:5224, `output/chunk4_aggressive_probe_run2_harsh/eval_surfel/cube_eval.json`).

### Diagnosis conclusion

- Chunk-4 is not failing due to a single obvious mechanical bug in the coupling/support code path.
- The practical failure mode is a combination of:
  1) schedules that do not activate strongly within current gate budgets, and
  2) weak Stage-1 evidence quality feeding posterior-driven expected points.
- Under current conditions, stronger enforcement mostly increases deletion pressure (harsh settings) rather than corrective cube-shape convergence.

---

## 2026-02-24 Independent Review (claude-opus-4-6)

### Mechanical completeness

The Chunk-4 implementation is **mechanically complete**:
- `utils/elevation_chunk4_helpers.py`: 19 pure functions covering coupling, ID lifecycle, support scheduling, checkpoint/resume.
- `debug_multiframe.py`: ~218 lines of Chunk-4 integration across Stage 2 and Stage 3, with 21 config knobs.
- 16/16 fast contract tests pass (`C4-T01`..`C4-T10`, `C4-T11`..`C4-T14`).
- Off-mode parity is excellent (`rel_loss_delta=0.0013`, `abs_ssim_delta=0.0006`).
- Sphere-A (C4-S1) passes all gates cleanly.
- Resume/continuation path works correctly.

No bugs or implementation gaps were found in the coupling, ID, support, or checkpoint code.

### Blocker triage

| Blocker | Verdict | Rationale |
|---------|---------|-----------|
| **B2** (coupling residual 0.30195 vs 0.30m) | **Waivable** | 0.002m overshoot on a cube-class dataset where sphere-A passes easily. Threshold can be relaxed to 0.31m or left as-is with waiver. |
| **B3** (C4-T17 manual visual verdict) | **Administrative** | Just needs someone to record the verdict. Not a code issue. |
| **B1** (Dataset-C cube gate fails) | **Real blocker, but not a Chunk-4 problem** | See root cause analysis below. |

### Root cause analysis for B1

The post-closeout probes are decisive evidence that **Chunk-4 enforcement cannot fix this failure mode**:

- **Probe A** (mild): 3 surfels pruned. Geometry displacement 0.007m mean. Torus shape unchanged.
- **Probe B** (harsh): 83,083 surfels pruned (113k → 1,571). Half the torus deleted. Verdict: `regressed` (collapse, not correction).

This reveals a causal chain:

```
Chunk-3 parity gap (Stage-1 still uses interim surrogate, not full back_project_bins evidence)
  → Posterior does not encode correct elevation for cube corners/edges
    → Expected points from posterior land on torus surface, not cube surface
      → Coupling pulls surfels toward torus (no corrective signal)
        → Support evidence says torus surfels are "well supported"
          → Pruning either does nothing (mild) or mass-deletes (harsh)
```

Coupling and support-pruning can only act on the evidence they receive. If the Stage-1 posterior says "the torus is correct," then:
- Coupling reinforces torus geometry (not cube).
- Support confirms torus surfels as well-supported.
- Increasing enforcement strength amplifies deletion, not correction.

**B1 traces back to the Chunk-3 open parity gap**, not to any Chunk-4 deficiency.

### Recommended path forward

1. **Close the Chunk-3 Stage-1 likelihood parity gap.** The `back_project_bins` overlap-neighbor evidence assembly must match the detailed-plan contract, replacing the interim surrogate. This is the prerequisite for any meaningful Chunk-4 re-evaluation.

2. **Re-run Chunk-4 gates against corrected Stage-1 evidence.** Do not tune Chunk-4 parameters in isolation — the current parameters are reasonable and well-tested on sphere-A. The cube failure is an evidence-quality problem, not a coupling/support parameter problem.

3. **Waive B2 and B3 explicitly.** B2 is a 0.002m epsilon miss. B3 is documentation. Neither blocks progress.

4. **Do not start Chunk 5** until the Chunk-3 parity gap is closed and Chunk-4 is re-gated with real evidence. Chunk 5 (normals ramp, expected-elevation normals) depends on correct posterior beliefs, which depend on correct Stage-1 evidence.

### What NOT to do

- Do not keep tuning Chunk-4 coupling/support parameters to fix Dataset-C. The probes already showed this is the wrong lever.
- Do not relax Dataset-C thresholds to force a gate pass. The thresholds are reasonable; the evidence feeding the system is wrong.
- Do not skip Chunk-3 parity closure and jump to Chunk 5. The normals path will have the same upstream evidence problem.

---

## 2026-02-25 Root Cause and Fix Path (Consolidated)

### What is wrong

Chunk-4 is blocked by a coupled upstream/downstream problem, not by missing implementation mechanics.

First, Stage-1 parity wiring has moved closer to the detailed overlap-neighbor `back_project_bins` contract and is consumed in the active per-iteration path. The remaining blocker appears to be evidence *quality* (posterior still broad/weak on Dataset-C), so expected points can remain weakly corrective for cube geometry.

Second, schedule-to-budget mismatch can still happen depending on explicit env overrides, but defaults are now budget-scaled in code (`debug_multiframe.py`:2649-2669). Gate analysis should therefore use effective run configs, not historical fixed-default assumptions.

Third, enforcement-only escalation is confirmed to be the wrong lever under weak evidence. Mild aggression produces little geometric movement; harsh aggression primarily increases deletion pressure and can collapse geometry rather than reshape torus artifacts toward cube surfaces.

### What to fix

Fix order is mandatory:

1. Close the Chunk-3 Stage-1 parity gap by replacing surrogate likelihood assembly with the detailed multi-view overlap-neighbor evidence contract (`back_project_bins` -> neighbor projection -> robust normalized log-evidence -> valid-support normalization -> `loglik`/`support_mask`/posterior).

2. Keep same-iteration cache semantics (`cached_loglik`, `cached_support_mask`) unchanged so Chunk-4 coupling continues consuming detached same-iteration evidence tensors.

3. Retune Chunk-4 schedule parameters to run budget after Step 1 (schedule tuning is required and not optional): use budget-scaled warmups instead of large absolute constants. For Stage-2=1000, start with approximately `ELEV_COUPLE_WARMUP=200-400` and `ELEV_SUPPORT_WARMUP_ITERS=300-600`, then calibrate from diagnostics.

4. Preserve anti-collapse guardrails during re-validation (no harsh prune regimes that can mass-delete surfels before evidence quality improves).

5. Re-run Chunk-4 gate sequence (`C4-S2`, `C4-S3`, `C4-S4`) only after Step 1 is implemented, and evaluate directional geometry correction first, then threshold closure.

### Consolidated conclusion

Chunk-4 code integration is largely complete, but gate failure persists because evidence quality is upstream-limited and schedules are budget-misaligned. Reliable Dataset-C recovery requires Stage-1 parity closure first, then budget-aligned Chunk-4 schedules; tuning enforcement strength alone is insufficient.

---

## 2026-03-11 Visualizer Sanity Check (gpt-5.4)

Using the new per-frame visualizer export on the synthetic cube dataset, a first comprehensible qualitative result was obtained from a tightly constrained run:

- Run directory: `output/debug_multiframe_synth_c_first6/`
- Frame policy: first 6 frames only (`sonar_000000`..`sonar_000005`)
- Observation: these six views all look at the same cube face, and the exported surfels form a readable straight-line band exactly where that cube face should be.
- Orientation observation: surfel orientations are still randomized/noisy, but they more often face toward the observing sonar poses than away from them, which matches the intended initialization bias much better than the previously unreadable exports.
- Visualizer verdict: this is the first cube-dataset result that looks remotely comprehensible in Blender and is the first export that clearly shows a face-aligned surfel structure instead of an uninterpretable cloud.

Interpretation note:

- This does **not** close the Chunk-4 quantitative gate and does **not** by itself resolve the cube failure documented above.
- It **does** establish that the new visualizer is finally good enough to support meaningful manual diagnosis of cube-face structure, surfel footprint placement, and front/back orientation tendencies on a controlled subset.

Follow-up observer note from the next temporary visualizer probe:

- A 10-frame run spanning about 90 degrees of the 500-frame synthetic cube orbit (`output/debug_multiframe_synth_c_10frames_90deg/`) reverts to the familiar torus-like structure.
- This is consistent with prior behavior and is not surprising: the recent work changed visualization/export only, not the surfel-learning or rendering logic that drives the torus failure mode.
- The temporary frame-selection experiments (`first 6 frames`, `10 frames over ~90 degrees`) should therefore be interpreted as diagnostic probes only, not as algorithmic fixes.
- As frame count and azimuth coverage move back toward the full-orbit regime, the qualitative reconstruction predictably trends back toward the historical torus symptom seen in the broader 360-degree runs.

---

## 2026-02-27 Baseline Renderer Diagnosis (claude-opus-4-6)

Historical provenance note:
- The material from this section onward (`line 875+`) is kept because this is where the renderer problem was first written down when it was discovered.
- This section is now a historical record only; active planning/execution is superseded by `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md`.

### Problem statement

Rendered sonar images appear as "speckled dots" instead of matching the GT's smooth intensity arcs. The 3D reconstruction shows a torus with structured streak lines. Even ignoring elevation (which is expected to be unconstrained), the 2D (azimuth, range) projected shape fails to converge to the correct geometry.

### Root cause: Random normal initialization discarded by GaussianModel

The initialization code (`debug_multiframe.py:3275-3283`) computes camera-facing normals for every surfel:

```python
for j in range(len(points)):
    dir_to_cam = cam_pos - points[j]
    normals[j] = dir_to_cam / np.linalg.norm(dir_to_cam)
```

These normals are passed into `BasicPointCloud` and handed to `create_from_pcd`. However, `GaussianModel.create_from_pcd` (`scene/gaussian_model.py:139`) **ignores the normals entirely** and initializes rotation quaternions randomly:

```python
rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
```

The `pcd.normals` field is never read. All computed normals are discarded.

### Physical inconsistency

Every initialized surfel exists because the sonar received a reflection from that surface point. By definition, the surface at that location was facing the sonar — `lambertian > 0` is a physical invariant at initialization. The computed camera-facing normals (`debug_multiframe.py:3275-3283`) correctly encode this. Discarding them for random quaternions introduces surfels with `lambertian < 0` (facing away from the sonar that detected them), which is physically impossible and immediately corrupts ~50% of the reconstruction.

### How random normals cause speckled dots

`render_sonar` (`gaussian_renderer/__init__.py:436-451`) uses a Lambertian intensity model:

```python
normals_world = quaternion_to_normal(rotations)           # random directions
lambertian = clamp(dot(normals_world, dir_to_sonar), min=0)  # ~50% are zero
base_intensity = opacity * lambertian
```

With uniformly random quaternions, `quaternion_to_normal` produces uniformly random normals on the unit sphere. For any given sonar view direction:
- ~50% of surfels have `dot(normal, dir_to_sonar) < 0` → `lambertian = 0` → **invisible**
- The visible ~50% are a random sparse subset that changes per view
- Result: **speckled dot pattern** instead of dense smooth surface coverage

### Why rotations fail to learn correct normals

The rotation optimizer is active (`rotation_lr=0.001`), but convergence is blocked by a dead zone:

1. **`torch.clamp(min=0)` kills the gradient** when `dot(normal, dir) < 0`. Surfels facing away from the current sonar view receive **zero gradient on rotation** from that view — they cannot learn to turn around.
2. **Gradient signal is view-biased**: each surfel only learns from views where it happens to face correctly (lambertian > 0). This creates view-dependent normal biases rather than convergence to true surface normals.
3. **Fixed opacity** (`SONAR_FIXED_OPACITY=True`, default at line 2823) freezes opacity at ~0.999. Wrong-facing surfels cannot fade out; they occupy parameter budget while contributing nothing.

### Why this affects cubes but not spheres

- **Sphere**: rotationally symmetric. The "correct" normal at any point is similar to many random orientations, and all views provide consistent gradient signal. Random normals converge because the loss landscape is forgiving.
- **Cube**: 6 discrete face normal directions. A surfel on a cube face must find one specific normal direction. The dead zone in `clamp(min=0)` makes this much harder — the surfel can only learn from the subset of views where its random normal happens to partially align.

### Why streaks form in 3D

Surfels initialized from each training camera develop normal biases toward that camera's direction (the only views giving them gradient). Over training, this creates structured clusters in normal space aligned with individual camera poses. These clusters project as streak patterns in the 3D point cloud.

### Relationship to elevation chunks

This is a **baseline renderer defect** predating all elevation-aware work (Chunks 1-5). None of the chunks modify `render_sonar` or the rotation initialization. The elevation system builds on top of a renderer that cannot properly learn surface normals for non-trivially-shaped objects. This compounds the evidence-quality problems identified in earlier addenda but is an independent, lower-level failure mode.

### Additional baseline renderer issues identified

1. **Point splatting, not Gaussian splatting**: `render_sonar` splats each surfel to exactly 4 pixels via bilinear interpolation (`gaussian_renderer/__init__.py:482-536`). `pc.get_scaling` is never used for the spatial footprint. The standard `render()` uses the full CUDA rasterizer with 2D Gaussian covariance covering potentially hundreds of pixels per surfel.

2. **Densification disabled**: `debug_multiframe.py` Stage 2 never calls `densify_and_clone/split`. Surfel count is fixed from initialization (~113K) and only decreases via FOV pruning. Even if enabled, densification would fail because `screenspace_points` in `render_sonar` is a zero tensor that accumulates no gradients.

3. **No surfel scale effect on rendering**: since `render_sonar` ignores surfel scale, the scale parameters receive no useful gradient from the photometric loss and cannot influence surface coverage.

### Severity assessment

The random-normal + Lambertian dead-zone issue is the most impactful finding. It explains why the rendered images show dots instead of smooth arcs (the immediate visual symptom the user observes), and why optimization struggles to converge for non-spherical geometry. The point-splatting and disabled-densification issues are secondary but compounding.

### Recommended investigation before fixing

1. Verify by initializing rotations from the computed normals (convert camera-facing normals to quaternions at init) and comparing rendered image quality at iteration 0.
2. Quantify the lambertian dead-zone fraction per view in a diagnostic pass.
3. Evaluate whether proper normal init alone resolves the speckled-dot pattern, or whether Gaussian splatting / densification changes are also required.

---

## 2026-03-01 Lambertian Dead Zone and Surfel Drift Concerns (claude-opus-4-6)

### The Lambertian clamp is a ReLU dead-zone problem

The `torch.clamp(dot(normal, dir_to_sonar), min=0)` in `render_sonar` (line 447) is mathematically identical to a ReLU activation. The resulting failure mode — surfels that face away get zero gradient and can never recover — is the well-known **dead neuron problem** from neural network training.

| Neural network | Sonar renderer |
|---|---|
| `ReLU(x) = max(x, 0)` | `lambertian = clamp(dot(n, d), min=0)` |
| Neuron outputs 0 → zero gradient → permanently dead | Surfel faces away → zero lambertian → permanently invisible |
| ~10-30% neurons can die during training | ~50% surfels dead at init with random normals |

### Surfel drift concern: permanently trapped surfels

Even after fixing initialization (so all surfels start with correct camera-facing normals), surfels can **drift into a dead configuration during training**. Specific scenario:

A surfel positioned right in front of a cube's front surface but facing the wrong way (both the surfel's outer surface and the cube's outer surface face each other). This surfel is permanently trapped:

- **From the front**: `lambertian = 0` (normal faces away from sonar) → invisible, zero gradient on rotation.
- **From the rear**: the cube physically occludes it — the sonar beam cannot reach the surfel through the solid cube.
- **Result**: no view provides gradient to correct the normal. The surfel is a permanent parameter zombie — it cannot learn, cannot fade (fixed opacity), and cannot be pruned (no mechanism to detect it).

This concern generalizes: any surfel that ends up near a surface with its normal facing into that surface becomes trapped in the Lambertian dead zone on one side and physically/acoustically shadowed on the other. This can happen through:
- Noisy gradient updates during training (multiple views pulling the normal in conflicting directions)
- Position drift that moves a surfel from its original visible location to a trapped configuration
- Topology changes (densification, if ever enabled) creating surfels in problematic locations

### Why opacity adjustment is not appropriate for sonar

In standard 2DGS (camera rendering), opacity learning is the natural self-pruning mechanism: useless surfels fade to zero. However, for sonar rendering of solid objects, opacity does not have a physical analog. Acoustic surfaces either reflect the sonar beam or they don't — there is no case (in the environments being modeled) where sound propagates through a surface with partial attenuation. Fixed opacity at ~1.0 is the physically correct model. The dead-surfel problem must be solved through other means.

### Candidate fixes for the dead-zone problem

All options are borrowed from the neural network dead-neuron literature, adapted to the rendering context:

**1. Leaky Lambertian** (analogous to Leaky ReLU, most standard)
```python
alpha = 0.01
dot_val = torch.sum(normals_world * dir_to_sonar, dim=-1)
lambertian = torch.where(dot_val >= 0, dot_val, alpha * dot_val)
```
Wrong-facing surfels contribute 1% intensity. Minimal rendering artifact; gradient always flows. The 1% leak is physically defensible — real rough surfaces scatter small amounts of energy at reverse angles. `alpha=0.01` is the standard default in deep learning.

**2. ELU-style** (smoother transition)
```python
lambertian = torch.where(dot_val >= 0, dot_val, alpha * (torch.exp(dot_val) - 1))
```
Exponential decay for the leak — vanishes quickly for strongly wrong-facing surfels. Smoother gradient near zero.

**3. Straight-through estimator** (no rendering change, gradient-only fix)
```python
lambertian_hard = torch.clamp(dot_val, min=0)        # forward: physically correct
lambertian_soft = torch.clamp(dot_val, min=0.01)      # backward: gradient flows
lambertian = lambertian_hard + (lambertian_soft - lambertian_soft.detach())
```
Renders identically to the current model. Backward pass sees a floor. Used in quantization-aware training. More complex but zero visual change.

**Current recommendation**: Leaky Lambertian with `alpha=0.01`. One-line change, well-understood behavior, physically defensible for sonar. Combined with proper normal initialization, the leak would rarely activate in steady state — it serves as a safety net against drift into the dead zone. Decision pending.

---

## 2026-03-01 Missing Acoustic Occlusion in render_sonar (claude-opus-4-6)

### Problem statement

`render_sonar` (`gaussian_renderer/__init__.py:471-536`) uses purely additive `scatter_add_` accumulation with no depth ordering or occlusion. Every surfel that passes the FOV check contributes to the rendered image regardless of whether a closer surface blocks the acoustic beam path.

The standard camera `render()` function uses the CUDA rasterizer with front-to-back alpha compositing, transmittance tracking, and early termination — none of which exist in the sonar path.

### Physical sonar occlusion model

Sonar occlusion differs from camera occlusion:
- In a camera image, two objects at different depths project to the **same pixel** → need alpha compositing to blend/occlude.
- In a sonar image, two objects at different ranges project to **different rows** (different pixels). They do not compete for the same pixel in the normal sense.
- However, a solid surface at range R1 **blocks the acoustic beam** so that nothing at range R2 > R1 along the **same azimuth AND same elevation angle** should produce a return. The farther return appears at a different pixel (different row) but should not exist at all.

Correct occlusion rule: a surfel is occluded if and only if another surfel with sufficient opacity lies **closer in range along the same (azimuth, elevation) ray**. Surfels at different azimuth or different elevation angles do not occlude each other, even if they are at different ranges.

### Current behavior and consequences

1. **Ghost returns from inner/back surfaces**: Poisson reconstruction creates closed meshes, so surfels exist on all sides of a solid object — including back faces that no sonar ever observed. These back-face surfels project to range rows where the GT has no signal, creating spurious intensity.
2. **Photometric loss fights ghost returns**: The optimizer sees extra intensity at ghost-return pixels and tries to suppress it. With fixed opacity, the only levers are position and rotation — surfels contort to minimize their contribution rather than simply disappearing.
3. **Compounds with Lambertian dead zone**: Random normals make ~50% of front-face surfels invisible while no occlusion makes back-face surfels visible. The renderer simultaneously suppresses correct surfaces and hallucinates incorrect ones.

### Proposed approach: per-ray front-to-back accumulation with transmittance

The sonar beam travels along rays defined by **(azimuth, elevation)** pairs. For each ray, surfels should be processed front-to-back (sorted by range), with a transmittance variable that drops as solid surfaces are encountered. Once transmittance is exhausted, remaining surfels on that ray contribute nothing.

#### Approach sketch

1. **Ray binning**: Discretize the (azimuth, elevation) space. Each surfel maps to a ray bin based on its projected azimuth and computed elevation angle.
2. **Range sorting within each ray**: For surfels in the same ray bin, sort by range (ascending).
3. **Front-to-back accumulation with transmittance**:
   ```
   T = 1.0  (initial transmittance)
   for each surfel in range-sorted order:
       alpha_i = opacity_i * lambertian_i
       contribution_i = alpha_i * T
       accumulate contribution_i to pixel (col, row) via bilinear splat
       T = T * (1 - alpha_i)
       if T < epsilon: break  (fully occluded)
   ```
4. **Pixel mapping**: Each surfel still maps to its own (col, row) pixel based on (azimuth, range). The transmittance determines **whether** it contributes, not **where** it contributes.

#### Key design decisions to resolve

- **Elevation bin resolution for ray grouping**: How finely to discretize elevation for ray binning. Too coarse → surfels at different elevations incorrectly occlude each other. Too fine → sparse bins, inefficient. The elevation FOV is ±10° with typical objects subtending a few degrees, so ~1° bins may suffice.
- **Differentiability**: The sort operation and transmittance multiplication must be differentiable. The standard 2DGS approach (cumulative product of `(1 - alpha)`) is differentiable. Sorting can use straight-through or soft-sort if needed, though hard sort with detached indices is simpler and usually sufficient (gradients flow through `alpha` and `contribution`, not through the sort order).
- **Performance**: Per-ray sorting is more expensive than flat `scatter_add_`. For ~113K surfels across ~256 azimuth bins and ~7-20 elevation bins, each ray bin has on average 20-60 surfels. Sorting within small bins is fast.
- **Interaction with Lambertian model**: If the Lambertian dead-zone fix (leaky Lambertian) is also applied, wrong-facing surfels behind correct ones would get occluded anyway, providing a natural cleanup mechanism for inner-surface surfels.

#### What this does NOT change

- Pixel coordinates: each surfel still maps to (col, row) based on (azimuth, range). The sonar image layout is unchanged.
- FOV checking: unchanged.
- Loss computation: unchanged.
- The occlusion model only gates **whether** a surfel's contribution reaches the image, not where it goes.

### Interaction with other identified issues

| Issue | Occlusion fix impact |
|-------|---------------------|
| Random normals / dead zone | Orthogonal — occlusion gates by range ordering, not normal direction. Both fixes needed. |
| Point splatting | Orthogonal — occlusion determines contribution weight, splatting determines pixel spread. Both can be addressed independently. |
| Fixed opacity | Occlusion makes fixed opacity more viable: solid front surfaces naturally block back surfaces, so back-face surfels don't need opacity decay to disappear. |
| Densification disabled | Orthogonal. |
| Elevation-aware chunks | Occlusion is upstream of all elevation work. Correct occlusion improves the fidelity of the rendered image that the photometric loss trains against, which in turn improves the gradient signal for all downstream systems. |

### Priority relative to other fixes

1. **Normal initialization from computed normals** — simplest, highest immediate impact on speckled dots.
2. **Leaky Lambertian** — prevents dead-zone drift, one-line change.
3. **Acoustic occlusion** — architecturally significant, eliminates ghost returns and inner-surface contamination. Required for physically correct sonar rendering of solid objects.

All three are independent and can be developed/tested separately. Normal init and leaky Lambertian are quick wins; occlusion is a larger change that should be designed carefully.

### Correct occlusion rule (from user)

An outer-surface surfel blocks what is behind it **only if they are on the same azimuth AND same elevation angle**. Otherwise no occlusion. Specifically:
- Two surfels at the same (azimuth, elevation) but different ranges: the closer one occludes the farther one (the acoustic beam is blocked by the closer surface).
- Two surfels at different azimuth angles: no occlusion, regardless of range.
- Two surfels at different elevation angles: no occlusion, regardless of range. The sonar beam at one elevation does not block the beam at another elevation.

This means occlusion is per-(azimuth, elevation) ray, not per-pixel. Two surfels that land on the same pixel in the sonar image (same azimuth, similar range) but at different elevation angles do NOT occlude each other — they both contribute independently. This is physically correct: the sonar integrates returns from the full elevation beam, and surfaces at different elevations along that beam all reflect independently.

### Note on plan hierarchy

This section records the diagnosis and proposed approach in the Chunk-4 plan for immediate reference. Once the approach is settled, the highest-level plan (`plans/PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md`) and its dependent plans (`PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md`, `PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md`, and downstream chunk plans) must be updated to reflect the renderer-level changes, since they currently assume the baseline `render_sonar` behavior throughout. The highest-level plan already contains decisions on the Lambertian model (line 786), fixed opacity (line 736), and sonar intensity physics (line 786) that will need addenda for the normal initialization fix, leaky Lambertian, and acoustic occlusion.

---

## 2026-03-02 Planning Notes (No-Regeneration Yet)

1. **Synthetic refresh required after renderer-level changes**
   - Treat current synthetic gate artifacts/results as pre-renderer-fix baseline evidence only.
   - Any accepted change to normal initialization, Lambertian transfer, or sonar occlusion model requires synthetic dataset/evaluator refresh before new gate conclusions are considered current.

2. **Synthetic guide must carry explicit refresh triggers/versioning**
   - Add a concise trigger list and dataset-version marker policy to `docs/SYNTHETIC_DATASET_GUIDE.md` in a later planning pass.
   - Trigger examples: renderer intensity model change, occlusion semantics change, projection-convention change, or evaluator metric-definition change.

3. **Re-baseline checklist required before post-fix gate claims**
   - Define a minimal rerun checklist for Chunk-4 synthetic evidence (`C4-S1`..`C4-S4`) plus baseline deltas and artifact verdict panel updates.
   - Keep this as planning-only for now; do not regenerate datasets or rerun synthetic gates in this step.
