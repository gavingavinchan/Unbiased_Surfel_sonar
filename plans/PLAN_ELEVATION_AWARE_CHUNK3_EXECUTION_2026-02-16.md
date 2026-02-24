# Plan: Elevation-Aware Chunk 3 Execution

**Date:** 2026-02-16  
**Status:** Partially implemented; gate not closed (gpt-5.3-codex)  
**Scope:** Chunk 3 only (Elevation Stage 1 likelihood core)

## Consensus Marker (Implementation Unlock)

- Consensus status: `approved`
- Last reviewed date: `2026-02-18`
- Approver(s): `user + OpenCode` (opus review incorporated via `scratchpad.md`)
- Implementation unlocked: `yes`

---

## Implementation Snapshot (2026-02-23, code-verified)

- Implemented in code (`debug_multiframe.py` + `utils/elevation_stage1_helpers.py`): `ELEV_STAGE1_MODE`, frame-key identity/uniqueness checks, active-frame fingerprint + schema resume checks, frame-keyed pixel-bank/logit registry, `optim_elev` ownership, refresh/remap controls, cache handoff tensors (`cached_loglik`, `cached_support_mask`, `p_post`), and Stage-1 checkpoint payload wiring.
- Fast contract suite passes in conda env:
  - `python -m py_compile debug_multiframe.py utils/elevation_stage1_helpers.py`
  - `pytest tests/test_elevation_stage1_core_contracts.py tests/test_elevation_stage1_checkpoint_contracts.py tests/test_elevation_stage1_smoke_modes.py -q`
  - current result: `37 passed`.
- Remaining Chunk-3 parity gap (intent unchanged): Stage-1 likelihood in the training loop still uses an interim per-frame surrogate path and does not yet consume overlap-neighbor multi-view evidence via `back_project_bins` + projection-validity gating exactly as specified in the detailed plan.
- Gate-close implication: Chunk 3 remains open until this parity gap is closed or explicitly waived with recorded rationale.

---

## Goal

Implement Elevation Stage 1 likelihood infrastructure in a deterministic, numerically stable, and checkpoint-safe way before enabling mandatory coupling/support logic in Chunk 4.

Primary outcomes:
- Pose-overlap-driven multi-frame sampling is active and reproducible.
- Pixel-bank logits are learnable, persisted, restored, and mismatch-checked.
- Frame reliability and robust normalized amplitude likelihood are finite and stable.
- Temperature annealing and entropy behavior are observable on real and synthetic runs.
- Chunk 4 interface tensors (`cached_loglik`, `cached_support_mask`, `p_post`) are produced in Chunk 3 so later coupling can consume them without refactoring the Stage 1 core loop.

---

## Terminology Alignment (Critical)

Current code in `debug_multiframe.py` already uses curriculum stage names:
- curriculum Stage 1 = scale-only,
- curriculum Stage 2 = surfel-only,
- curriculum Stage 3 = joint fine-tune.

This plan uses the detailed-plan terminology:
- Elevation Stage 0 = initialization,
- Elevation Stage 1 = bin-likelihood optimization,
- Elevation Stage 2 = optional densification.

To avoid naming collision:
- Keep existing curriculum stage labels unchanged in code/logs.
- Prefix new logic/metrics with `elev_*` or explicit `Elevation Stage 1` text.
- Integrate Elevation Stage 1 loop into optimization phases that update surfels (curriculum Stage 2/3), while keeping curriculum Stage 1 (scale-only) behavior unchanged.

---

## Source of Truth

- Main implementation contract: `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md`
- Execution strategy and chunk gates: `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md`
- Prior chunk references (implemented):
  - `plans/PLAN_ELEVATION_AWARE_CHUNK1_EXECUTION_2026-02-10.md`
  - `plans/PLAN_ELEVATION_AWARE_CHUNK2_EXECUTION_2026-02-11.md`
- Active synthetic inventory and commands:
  - `docs/SYNTHETIC_DATASET_GUIDE.md`

If any mismatch appears, follow the detailed training plan for algorithm contracts and follow the execution plan for chunk sequencing/scope boundaries.

Intent lock for terminology/doc drift:
- Treat frame-index vs `frame_key` wording differences as non-material only when intent remains identical: stable per-frame identity, deterministic checkpoint keying, and deterministic resume mapping.
- If wording drift changes algorithm behavior or leaves identity intent ambiguous, stop and reconcile in source-of-truth plans before implementation.

Chunk-staging clarification:
- Coupling implementation remains scheduled for Chunk 4 per the execution plan.
- Chunk 3 must still produce the Stage-1 posterior/cache handoff tensors required by Chunk 4.

---

## Review Issues and Disposition (2026-02-16)

Cross-review outcome after plan/code reconciliation:

1. **Coupling scope mismatch (detailed plan vs chunk staging)**
   - Disposition: staged-delivery clarification added in Source-of-Truth section; coupling remains Chunk 4 scope in this execution plan.

2. **Missing Stage-1 enable/shadow gate**
   - Disposition: ACTION REQUIRED in this plan.
   - Fix: add `ELEV_STAGE1_MODE=off|shadow|active` runtime contract and validation behavior.

3. **Resume continuity vs checkpoint schema**
   - Disposition: ACTION REQUIRED in this plan.
   - Fix: add explicit checkpoint schema versioning (`checkpoint_schema_version`) and deterministic compatibility handling for Stage-1 runtime payloads.

4. **Frame identity ambiguity for pixel-logit registry**
   - Disposition: ACTION REQUIRED in this plan.
   - Fix: define stable frame key contract (`frame_key = camera.image_name`) for pixel bank/logits/checkpoint payloads.

5. **Overlap-table frame-set compatibility guard**
   - Disposition: ACTION REQUIRED in this plan.
   - Fix: persist active frame key list plus deterministic frame-set fingerprint/hash for resume validation.

6. **Anneal fallback horizon concern**
   - Disposition: current contract retained for Chunk 3 (`ELEV_ANNEAL_ITERS` else `SONAR_STAGE2_ITERS`).
   - Action: treat as tuning knob, not a scope/blocker item.

7. **Entropy sign convention ambiguity**
   - Disposition: ACTION REQUIRED in this plan.
   - Fix: explicitly define entropy term as minimizing entropy with positive weight.

8. **Synthetic command incompleteness (S3/S4)**
   - Disposition: ACTION REQUIRED in this plan.
   - Fix: add concrete runnable S3 and S4 command sequences.

9. **Commit-boundary docs omission**
   - Disposition: ACTION REQUIRED in this plan.
   - Fix: include `plans/progress_overview.md` and `plans/scientific_progress.md` update requirement before commit.

---

## In-Scope Files (Chunk 3)

Primary implementation target:
- `debug_multiframe.py`

Hard runtime dependencies (must match existing contracts):
- `utils/sonar_utils.py` (`back_project_bins` path/contracts)
- `gaussian_renderer/__init__.py` (`sonar_project_points`/`SonarProjection` contract)

Optional edits only if required for clean flag exposure or helper placement:
- `arguments/__init__.py`

Out of scope for Chunk 3:
- Coupling loss and association (`loss_couple`, persistent surfel IDs, support/pruning) -> Chunk 4
- Normals ramp / expected-elevation normals -> Chunk 5
- Optional Stage 2 densification hooks -> Chunk 5

---

## Chunk 3 Contracts to Implement

1. `overlap_table` contract
   - Build once from active training frames.
   - Pose-only score (`ELEV_OVERLAP_SCORE_MODE=pose_only`) using baseline + yaw gates.
   - Deterministic sort and top-k retention.
   - Persist table and build params in checkpoint state.

2. Frame sampler contract
   - Deterministic shuffled round-robin (`ELEV_FRAME_SAMPLER=round_robin` default).
   - Sample `A=min(ELEV_FRAMES_PER_ITER, N_active)` per iteration.
   - Persist sampler cursor/epoch state for resume continuity.

3. Pixel bank + logits registry contract
    - Stable frame identity contract: `frame_key = camera.image_name` (string), used for all Stage-1 registry/checkpoint keys.
    - Assert `frame_key` uniqueness over active training frames at Stage-1 startup; fail fast on collisions to avoid silent registry/checkpoint corruption.
    - `pixel_bank[frame_key]` stores metadata (`rows`, `cols`, `logits_key`, `frame_key`).
    - Learnable logits live in a registry keyed by `frame_key` (not local sampled index).
    - `optim_elev` owns pixel-logit params only.
    - Checkpoint includes bank metadata, logit weights, optimizer state.

4. `frame_stats` cache and reliability contract
   - Build once at Elevation Stage 1 startup from fixed GT frames and masks.
   - Compute robust percentiles (`p_lo`, `p_hi`), dynamic range, valid ratio, reliability.
   - Use deterministic fallback stats for low-valid frames.

5. Stage 1 likelihood contract
   - Robust normalized amplitude evidence with neutral invalid handling.
   - Per-bin support normalization and support-mask-aware softmax.
   - Detached evidence target for CE + learnable logits prediction branch.
   - Shared temperature annealing path (`T_model`, `T_post`) by default.

6. Chunk 4 handoff interface contract (implemented now)
   - Build `cached_loglik[frame_a]` per iteration (detached).
   - Build `cached_support_mask[frame_a]` per iteration (detached).
   - Compute `p_post` (posterior from logits + loglik) for entropy in Chunk 3 and coupling input compatibility in Chunk 4.
   - Caches are same-iteration only and must be rebuilt each iteration.

7. Sampling helper contracts (explicit)
   - `sample_gt(frame_idx, row, col)` uses bilinear interpolation on preprocessed GT frame tensor.
   - Out-of-bounds samples are marked invalid via projection validity mask and contribute neutral evidence (not penalties).
   - `normalize_by_percentiles(I, lo, hi)` clamps to `[0, 1]` using `max(hi-lo, eps)` denominator.

8. Pixel-bank selection contract (explicit)
   - Use GT-intensity ranking after existing preprocessing/masking.
   - Default mode: top-k brightest valid pixels with `K=ELEV_PIXELS_PER_FRAME`.
   - Deterministic tie-break order: row-major index.
   - This is independent from `BRIGHT_PERCENTILE` loss term, but may reuse the same preprocessed intensity tensor.

9. Annealing contract (explicit)
   - Default schedule for `anneal(iter, start, end)` is linear interpolation with clamped progress in `[0,1]`.
   - Anneal horizon is `ELEV_ANNEAL_ITERS` when set; otherwise fallback is `SONAR_STAGE2_ITERS` for current curriculum compatibility.
    - Concrete mapping: `progress = clamp(iter / max(1, anneal_horizon), 0, 1)`.
    - Alternate schedules are out of scope for Chunk 3.

10. Loss aggregation contract (Chunk 4-ready)
     - Aggregate Stage 1 terms in one dedicated block.
     - Reserve explicit slot for `loss_couple` with zero/disabled placeholder in Chunk 3 so Chunk 4 can enable coupling without restructuring the total-loss block.
     - Chunk-3 placeholder contract: define `loss_couple = 0.0` and `w_couple = 0.0` (or equivalent no-op tensor scalar) in the unified total-loss block.

11. Stage-1 mode gate contract (explicit)
    - `ELEV_STAGE1_MODE` supports `off|shadow|active`.
    - `off`: Stage-1 overlap/sampler/pixel-bank/likelihood/caches are disabled; baseline Chunk-2-style losses run.
    - `shadow`: Stage-1 tensors/diagnostics are computed and logged, but Stage-1 weighted loss terms are not added to total loss.
    - `active`: full Stage-1 weighted loss terms are applied.
    - Resume contract: checkpoint must restore mode and required Stage-1 runtime state; incompatible state follows `ELEV_RESUME_PIXELLOGIT_MISMATCH` (`strict|reset_frame|reset_all`).

12. Resume frame-set compatibility contract (explicit)
     - Persist `active_frame_keys` (ordered list of `frame_key`) and `active_frame_fingerprint` for overlap/pixel-bank state.
     - Fingerprint algorithm (fixed for Chunk 3):
       - `payload = "\n".join(active_frame_keys).encode("utf-8")`
       - `active_frame_fingerprint = sha256(payload).hexdigest()`
     - On resume, if frame-set fingerprint mismatches checkpoint payload, follow `ELEV_RESUME_PIXELLOGIT_MISMATCH` (`strict|reset_frame|reset_all`).
     - Persist `checkpoint_schema_version` for Stage-1 payloads (`overlap_table`, sampler state, `pixel_bank`, `pixel_logits`, `optim_elev`, frame-key metadata).
     - Schema value for Chunk 3 is fixed to: `checkpoint_schema_version = "chunk3_stage1_v1"`.
     - On resume, schema incompatibility must fail fast by default (`strict`) and allow explicit reset behavior only via `ELEV_RESUME_PIXELLOGIT_MISMATCH`.

13. Entropy objective sign contract (explicit)
     - Entropy is defined as `H(p_post) = -sum_k p_post[k] * log(p_post[k])` per pixel.
     - Chunk-3 default sharpness behavior minimizes entropy: aggregate `loss_ent = mean(H)` over valid pixels and add `+ ELEV_ENTROPY_WEIGHT * loss_ent` to total loss.
     - Positive `ELEV_ENTROPY_WEIGHT` therefore sharpens posterior distributions.
     - Erratum note: if any prior pseudocode shows accumulation with `(-H)` and then adds it with positive weight, treat that as sign-inverted for sharpening intent; this contract is authoritative for Chunk 3.

14. Pixel-bank refresh/remap contract (explicit)
    - `ELEV_BANK_REFRESH_INTERVAL=0` means refresh is disabled (default Chunk-3 path).
    - If `ELEV_BANK_REFRESH_INTERVAL>0`, refresh triggers on `iter % ELEV_BANK_REFRESH_INTERVAL == 0`.
    - `ELEV_BANK_REMAP_MODE=nearest`:
      - remap each refreshed pixel to nearest old pixel by L1 image distance in the same frame,
      - if nearest distance > `ELEV_BANK_REMAP_MAX_DIST`, reset that pixel logits to zeros.
    - `ELEV_BANK_REMAP_MODE=reset`: reset all refreshed pixel logits to zeros.
    - Any refresh that changes parameter shapes must rebuild `optim_elev` param groups deterministically.

---

## Implementation Method (How)

1. Start with explicit Stage-1 mode gating.
   - Use `ELEV_STAGE1_MODE=shadow` for first-pass instrumentation.
   - Compute overlap/sampler/pixel-bank/likelihood and log diagnostics before activating weighted Stage-1 losses.
   - Promote to `ELEV_STAGE1_MODE=active` only after finite/stability checks pass.

2. Centralize runtime config parsing.
   - Parse/validate Stage 1 env vars once in one typed config block.
   - Fail fast on invalid enums/ranges.

3. Keep runtime state explicit and owned.
   - `overlap_table`, sampler state, `pixel_bank`, `pixel_logits`, `optim_elev`, `frame_stats`, per-iter caches.
   - Avoid in-place edits on autograd tensors.

4. Preserve baseline-safe behavior when disabled.
    - With Elevation Stage 1 disabled, behavior remains close to Chunk 2 baseline.
    - Compatibility gate: `ELEVATION_AWARE=0` forces Stage-1 behavior off (equivalent to `ELEV_STAGE1_MODE=off`) regardless of mode setting.

5. Preserve cross-view diagnostics continuity from Chunk 2.
   - Keep holdout split, per-frame CSVs, support metrics, and frame-visit accounting active.

---

## Step-by-Step Work Order

1. Run a TDD preflight import-safety pass before Stage-1 feature work:
   - move Stage-1 pure math/state helpers into `utils/elevation_stage1_helpers.py`,
   - keep script orchestration in `debug_multiframe.py` under `main()` and `if __name__ == "__main__": main()`,
   - verify that importing helper modules (and, where test-required, importing `debug_multiframe.py`) does not start training setup.
2. Write failing fast tests for Stage-1 contracts before implementation changes:
   - overlap score formula fixture + deterministic ranking checks,
   - decoupled-vs-shared temperature schedule checks,
   - refresh/remap contract checks for `ELEV_BANK_REFRESH_INTERVAL>0` and optimizer param-group rebuild.
3. Add/validate centralized config parsing for all Chunk 3 controls and defaults (including `ELEVATION_AWARE` and `ELEV_STAGE1_MODE`).
4. Implement `overlap_table` build path, deterministic scoring, and checkpoint payload.
5. Implement deterministic frame sampler with persistent cursor/epoch state.
6. Implement pixel-bank builder with explicit top-k bright-pixel selection contract and stable `frame_key` identity (including startup uniqueness assertion).
7. Implement pixel-logit registry and `optim_elev` lifecycle keyed by `frame_key`.
8. Implement Stage 1 checkpoint schema additions and mismatch policy handling, including `checkpoint_schema_version` save/load checks.
9. Implement `frame_stats` cache + reliability fallback path.
10. Implement `sample_gt` and `normalize_by_percentiles` helper paths (or explicit wrappers to existing equivalents).
11. Implement robust normalized amplitude likelihood (`loss_lik`) and explicitly signed entropy term (`loss_ent`) using `p_post`.
12. Implement per-iteration `cached_loglik` and `cached_support_mask` outputs for Chunk 4 compatibility.
13. Implement linear temperature schedule and logging (`T_model`, `T_post`, `T_tgt`) with explicit `ELEV_ANNEAL_ITERS`/fallback horizon.
14. Implement unified loss aggregation block with Chunk 4 coupling slot placeholder.
15. Run automated contract tests and short real-data smoke.
16. Run synthetic matrix (A_clean + C_clean), including continuation run.
17. Run full resume gate (real + synthetic) and verify state-contract continuity.
18. Record metrics/artifacts and finalize Chunk 3 gate verdict.

---

## TDD Testability Prerequisite (Before Step 1)

`debug_multiframe.py` executes training setup at import time, so pure unit tests cannot safely import Stage-1 helper logic if it is embedded directly in that script.

Import-safe definition for this chunk:

- importing a module defines functions/classes/constants only,
- import does not trigger dataset loading, output-directory creation, logger setup, training-loop execution, or process exit,
- executable behavior remains under explicit runtime entrypoints (`main()` / CLI path).

Required precondition for Chunk 3 TDD:

1. Place Stage-1 pure helper logic in import-safe units (module-level functions/classes) that do not start training on import.
2. Keep script-side orchestration in `debug_multiframe.py`, but test math/state helpers from the import-safe module.
3. Ensure each new Stage-1 contract item maps to at least one fast pytest case and one integration/smoke verification where applicable.

Planning consensus guardrail (recorded):

- Pre-consensus rule was planning-document edits only.
- Consensus is now recorded in the marker above; implementation is unlocked under this approved plan.

Canonical helper module path for Chunk 3:

- `utils/elevation_stage1_helpers.py`
- Unit tests should import Stage-1 pure helpers from this module directly.

Recommended helper surface in `utils/elevation_stage1_helpers.py`:

- overlap-table builder and deterministic ranking helper,
- deterministic round-robin sampler state and step function,
- percentile normalization helper,
- support-mask-aware masked softmax helper,
- frame-stats/reliability builder,
- linear temperature schedule helper,
- frame-set fingerprint/schema compatibility helper,
- checkpoint payload serializer/loader for Stage-1 runtime state.

---

## Pre-Implementation Test Layout and Commands

Define tests first, then implement to green:

Determinism contract for fast/smoke checks:
- Set fixed seeds for `random`, `numpy`, and `torch` in test setup and smoke commands.
- Use deterministic sampler path/config for frame-selection checks.
- Record seed values in test logs/artifacts for reproducibility triage.

1. **Fast contract tests (CPU, < 1 min)**
   - Target files (recommended):
     - `tests/test_elevation_stage1_core_contracts.py`
     - `tests/test_elevation_stage1_checkpoint_contracts.py`
   - Scope: overlap/sampler/pixel-bank keying/normalization/masked-softmax/anneal/checkpoint schema+mismatch behavior.

2. **Script-level smoke tests (short runtime, deterministic env)**
   - Target file (recommended):
     - `tests/test_elevation_stage1_smoke_modes.py`
   - Scope: `off|shadow|active`, finite diagnostics, cache/posterior shape checks, and no crash under short runs.

3. **Synthetic gate tests (long-running acceptance)**
   - Commands in Synthetic Matrix section (`S1`-`S4`) are mandatory acceptance checks, not per-commit unit checks.

Recommended command order for local/dev CI gate:

```bash
python -m py_compile debug_multiframe.py utils/sonar_utils.py gaussian_renderer/__init__.py
pytest tests/test_elevation_stage1_core_contracts.py -q
pytest tests/test_elevation_stage1_checkpoint_contracts.py -q
pytest tests/test_elevation_stage1_smoke_modes.py -q
```

---

## Runtime Config Contract (Chunk 3 Core Defaults)

Use the following defaults unless explicitly overridden:

- `ELEVATION_AWARE=1`
- `ELEV_BINS=7`
- `ELEV_STAGE1_MODE=shadow` (`off|shadow|active`)
- `ELEV_FRAMES_PER_ITER=3`
- `ELEV_FRAME_SAMPLER=round_robin`
- `ELEV_OVERLAP_TOPK_BUILD=24`
- `ELEV_OVERLAP_TOPK_USE=6`
- `ELEV_OVERLAP_MIN_BASELINE=0.06`
- `ELEV_OVERLAP_MAX_YAW_DEG=40`
- `ELEV_OVERLAP_MIN_SCORE=0.30`
- `ELEV_OVERLAP_SCORE_MODE=pose_only`
- `ELEV_OVERLAP_SCORE_W_YAW=0.6`
- `ELEV_OVERLAP_SCORE_W_BASE=0.4`
- `ELEV_PIXELS_PER_FRAME=2000`
- `ELEV_LOGIT_LR=2e-3`
- `ELEV_BANK_REFRESH_INTERVAL=0`
- `ELEV_BANK_REMAP_MODE=nearest`
- `ELEV_BANK_REMAP_MAX_DIST=6`
- `ELEV_RESUME_PIXELLOGIT_MISMATCH=strict` (`strict|reset_frame|reset_all`)
- `ELEV_TEMP_START=2.0`
- `ELEV_TEMP_END=0.1`
- `ELEV_TEMP_POST_MODE=shared`
- `ELEV_TEMP_POST_START=2.0`
- `ELEV_TEMP_POST_END=0.1`
- `ELEV_ANNEAL_ITERS` unset by default (fallback to `SONAR_STAGE2_ITERS`)
- `ELEV_LIK_TGT_TEMP=1.0`
- `ELEV_LIK_WEIGHT=1.0`
- `ELEV_ENTROPY_WEIGHT=0.01`
- `ELEV_LIK_NORM_P_LO=10`
- `ELEV_LIK_NORM_P_HI=99`
- `ELEV_LIK_LOG_EPS=1e-3`
- `ELEV_LIK_LOG_FLOOR=-6.9`
- `ELEV_LIK_MIN_SUPPORT=1e-6`
- `ELEV_LIK_USE_FRAME_RELIABILITY=1`
- `ELEV_LIK_INVALID_MODE=neutral` (fixed in v1)
- `ELEV_REL_FLOOR=0.3`
- `ELEV_REL_VALID_MIN=0.03`
- `ELEV_REL_VALID_MAX=0.30`
- `ELEV_REL_DYN_MIN=0.08`
- `ELEV_REL_DYN_MAX=0.50`

Notes:
- Parse `ELEV_TEMP_POST_START` and `ELEV_TEMP_POST_END` always, but consume them only when `ELEV_TEMP_POST_MODE=decoupled`.
- In default `ELEV_TEMP_POST_MODE=shared`, `T_post` reuses the shared schedule and ignores decoupled-only endpoints.
- Stage-1 enable policy is controlled by `ELEV_STAGE1_MODE`; `shadow` computes diagnostics without adding Stage-1 weighted loss terms.
- Top-level compatibility gate: `ELEVATION_AWARE=0` disables Stage-1 paths and forces effective behavior to `off`.
- `ELEV_RESUME_PIXELLOGIT_MISMATCH` is the single Chunk-3 mismatch policy knob for Stage-1 checkpoint compatibility classes (frame-set fingerprint, per-frame pixel-logit shape/key mismatches, and schema-version incompatibility).

---

## Validation Gate (Chunk 3)

All must pass before moving to Chunk 4:

1. Short reduced-workload run completes in target training path.
2. `loss_lik` and entropy term are finite (no NaN/Inf) when `ELEV_STAGE1_MODE=shadow|active`.
3. Invalid projection handling remains neutral and does not create penalty spikes.
4. Entropy trend is directionally decreasing over short horizon, defined as:
   - `mean_entropy(last_10pct_iters) < mean_entropy(first_10pct_iters)`
   - evaluated in `ELEV_STAGE1_MODE=shadow|active` runs.
5. `cached_loglik`, `cached_support_mask`, and `p_post` are produced with expected shapes and finite values.
6. Holdout and support diagnostics from Chunk 2 remain produced (`final_eval_*`, `support_metrics_*`, `frame_training_visits.csv`).
7. Cross-view directional checks versus Chunk 2 primary baseline (fixed-opacity Run C) are reported:
   - `single_view_top_share` (train) trend,
   - `single_view_top_share` (train+holdout) trend,
   - holdout/train loss ratio trend.
   - These are directional checks for Chunk 3 (reporting required, hard threshold gating deferred to Chunk 4 unless a severe regression is judged blocking).
8. Synthetic matrix (`S1`-`S4`) completes end-to-end.
9. Synthetic evaluator metrics are finite and non-regressing versus recorded baselines.
   - Material regression uses the quantitative default in the Baseline/pass-fail policy section.
10. Resume gate passes (real + synthetic continuation):
    - save checkpoint,
    - reload,
    - continue short run,
    - verify `pixel_bank`, `pixel_logits`, `optim_elev`, overlap table, and sampler-state contracts.
11. `ELEV_STAGE1_MODE` contract is validated:
    - `off` preserves baseline behavior,
    - `shadow` produces finite Stage-1 diagnostics without loss-weight activation,
    - `active` enables Stage-1 weighted terms with finite values.
12. `checkpoint_schema_version` contract is validated:
    - checkpoint payload contains expected schema version field,
    - resume with matching schema succeeds,
    - resume with incompatible schema follows `ELEV_RESUME_PIXELLOGIT_MISMATCH` (`strict|reset_frame|reset_all`) deterministically.

---

## Chunk 2 Baseline Values (Primary Comparator)

Use these Chunk 2 higher-budget fixed-opacity metrics as the primary directional baseline for Chunk 3 progression:

- `support>=2 = 0.4615`
- `support>=3 = 0.0446`
- `median_support = 1.0`
- `single_view_top_share (train) = 0.5606`
- `single_view_top_share (train+holdout) = 0.7249`
- `holdout_loss / train_loss = 1.52x`
- `train_loss_mean = 0.032021`
- `holdout_loss_mean = 0.048669`
- `train_ssim_mean = 0.8775`
- `holdout_ssim_mean = 0.8601`

Policy:
- Chunk 3 is expected to improve directionally on frame-dominance and generalization trend.
- If a regression is observed, record exact deltas and blocker analysis before proceeding.

---

## Synthetic Test Matrix (Chunk 3 Mandatory)

Reference inventory and command source: `docs/SYNTHETIC_DATASET_GUIDE.md`.

### Matrix definition

| ID | Dataset class | Primary command path | Purpose | Required artifacts |
|---|---|---|---|---|
| S1 | `A_clean` sphere | `run_synthetic_a_gate.py` | Stage 1 numerical stability and sphere non-regression | gate summary JSON/MD + evaluator JSON |
| S2 | `C_clean` cube | `run_synthetic_c_gate.py` | known-problem shape class artifact tracking | gate summary JSON/MD + evaluator JSON |
| S3 | `A_clean` sphere reproducibility | repeat S1 with second output dir | low-drift check under fixed seed | metric-delta record |
| S4 | `C_clean` cube continuation | checkpoint resume continuation run | Stage 1 resume/state continuity on hard shape case | checkpoint + continuation logs + evaluator JSON |

### Recommended runnable commands

1) Sphere gate (`S1`):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/run_synthetic_a_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_sphere_A_clean \
  --pose-mode sonar_equivalent \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --overwrite-runs
```

2) Cube gate (`S2`):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/run_synthetic_c_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --pose-mode sonar_equivalent \
  --pose-policy multi_band \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --overwrite-runs
```

3) Reproducibility (`S3`) using a distinct run prefix/output namespace:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/run_synthetic_a_gate.py \
  --dataset-root ./synthetic_datasets/synthetic_sphere_A_clean \
  --pose-mode sonar_equivalent \
  --num-frames 500 \
  --stage2-iters 1000 \
  --stage3-iters 1 \
  --output-root ./output/chunk3_s3 \
  --run-prefix debug_multiframe_synth_s3 \
  --overwrite-runs
```

Compare `final_eval_train_frames.csv` aggregate means and evaluator JSON deltas versus `S1`.

4) Synthetic continuation (`S4`) with explicit save/load checkpoint flow:

First run (save checkpoint):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=./synthetic_datasets/synthetic_cube_C_clean \
SONAR_OUTPUT_DIR=./output/chunk3_s4_run1 \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=600 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
ELEVATION_AWARE=1 \
ELEV_STAGE1_MODE=shadow \
SONAR_SAVE_CHECKPOINT=./output/chunk3_s4_run1/chunk3_s4_ckpt.pth \
python debug_multiframe.py
```

Continuation run (load checkpoint and continue):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=./synthetic_datasets/synthetic_cube_C_clean \
SONAR_OUTPUT_DIR=./output/chunk3_s4_run2 \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=600 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
ELEVATION_AWARE=1 \
ELEV_STAGE1_MODE=shadow \
SONAR_LOAD_CHECKPOINT=./output/chunk3_s4_run1/chunk3_s4_ckpt.pth \
python debug_multiframe.py
```

Evaluate continuation output:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
python scripts/eval_synthetic_cube.py \
  --reconstruction ./output/chunk3_s4_run2/surfels_after_training.ply \
  --dataset-root ./synthetic_datasets/synthetic_cube_C_clean \
  --output-dir ./output/chunk3_s4_run2/eval_surfel \
  --fit-mode both
```

### Baseline and pass/fail policy

- Baseline source priority:
  1. Most recent Chunk 2 synthetic artifacts for matching dataset/config.
  2. If missing, produce and record a Chunk 2-equivalent baseline before accepting Chunk 3.
- Required checks:
  - evaluator metrics are finite,
  - gate scripts complete and emit summary artifacts,
  - no material regression versus baseline.
- Material regression definition (Chunk-3 default, required unless explicitly waived in written gate notes):
  - for S1 and S2 evaluator metrics, any relative degradation greater than `10%` versus baseline in key error metrics (`mean_*_error_m`, `p95_*_error_m`, `center_error_m`) is a material regression.
  - material regression blocks Chunk-3 go/no-go unless an explicit waiver is documented with rationale and approved for this chunk.
- Required qualitative review for Dataset C:
  - classify known toroidal/streaking artifact as `improved|unchanged|regressed`,
  - cite artifact paths (surfel PLY + comparison PNGs).

---

## Chunk 3 Test Catalog (Planned)

1. Syntax/import compile gate for touched files.
2. Overlap-table contract test:
   - validate `pose_only` score formula (`w_yaw * yaw_score + w_base * baseline_score`) with fixed pose fixture,
   - deterministic ranking and top-k,
   - hard-gate filtering,
   - checkpoint payload roundtrip.
3. Frame sampler determinism/coverage test:
   - fixed-seed reproducibility,
   - full active-frame coverage across epochs.
4. Pixel-bank/logit registry test:
   - shape/key consistency,
   - optimizer ownership correctness,
   - mismatch-policy behavior.
5. Frame-stats reliability test:
   - percentile stats finite,
   - low-valid fallback equals reliability floor.
6. `sample_gt` + percentile normalization contract test:
   - bilinear interpolation behavior,
   - out-of-bounds neutrality via validity mask,
   - normalization clamp behavior finite.
7. Likelihood numerics test:
   - all-masked rows finite in masked-softmax,
   - `loss_lik` finite,
   - neutral invalid mode does not inject false penalties.
8. Posterior/cache interface test:
   - `p_post`, `cached_loglik`, `cached_support_mask` shapes/finite checks.
9. Entropy/annealing trend smoke:
   - linear schedule follows expected values,
   - `ELEV_TEMP_POST_MODE=shared` reuses model schedule and `ELEV_TEMP_POST_MODE=decoupled` follows decoupled endpoints,
   - entropy directionally decreases.
10. Short real-data training smoke with diagnostics continuity.
    - Fallback policy: if the configured real dataset path is unavailable, run the same smoke contract on `SONAR_DATASET=synthetic_a_clean` and explicitly record the fallback in artifacts/notes.
11. Synthetic matrix execution (`S1`-`S4`) with metric logging.
12. Checkpoint save/load continuation smoke (real + synthetic).
13. Checkpoint schema-version compatibility test:
     - verify `checkpoint_schema_version` persistence on save/load,
     - verify mismatch behavior is deterministic under configured policy,
     - verify `active_frame_fingerprint` exactly matches the contract algorithm (`sha256("\n".join(active_frame_keys))`).

14. `ELEVATION_AWARE` hard-override parity test:
    - set `ELEVATION_AWARE=0` with `ELEV_STAGE1_MODE=active`,
    - verify effective behavior is `off` and Stage-1 weighted terms remain disabled.

15. Frame-key uniqueness and identity test:
    - construct duplicate `camera.image_name` scenario,
    - verify Stage-1 startup fails fast with actionable error,
    - verify checkpoint payload keys are `frame_key` strings (not sampled index).

16. Pixel-logit gradient-ownership test:
    - verify `optim_elev` param groups include only pixel logits,
    - verify detached CE target path does not backprop through evidence branch,
    - verify logits branch receives finite gradients.

17. Anneal-horizon fallback test:
    - unset `ELEV_ANNEAL_ITERS`, set `SONAR_STAGE2_ITERS` small fixed value,
    - verify `progress = clamp(iter / max(1, horizon), 0, 1)` contract exactly.

18. Entropy sign regression test:
    - synthetic posterior fixture with known entropy,
    - verify `loss_ent = mean(H)` and total aggregation uses `+ ELEV_ENTROPY_WEIGHT * loss_ent`.

19. All-masked-row masked-softmax test:
    - input rows with zero support,
    - verify finite outputs and zero-prob rows (no NaN/Inf).

20. Off-mode baseline parity smoke (directional):
    - compare short off-mode run to Chunk-2-style baseline under same seed/config,
    - require no material divergence in core losses/support diagnostics beyond small tolerance.

21. Import-safety acceptance test:
    - import `utils/elevation_stage1_helpers.py` in a fast pytest without side effects,
    - verify import does not trigger training setup (dataset load/output dir creation/logger setup/training loop),
    - when `debug_multiframe.py` is refactored behind `main()`, verify module import remains side-effect free.

22. Pixel-bank refresh/remap contract test (Contract 14, refresh-enabled fixture):
    - run with `ELEV_BANK_REFRESH_INTERVAL>0` to trigger refresh path,
    - verify `ELEV_BANK_REMAP_MODE=nearest` remaps to nearest prior pixels and resets logits when distance exceeds `ELEV_BANK_REMAP_MAX_DIST`,
    - verify `ELEV_BANK_REMAP_MODE=reset` zeroes refreshed logits,
    - verify optimizer param groups are rebuilt deterministically when parameter shapes change.

### Anti-Flake Policy for Chunk-3 Gates

- Directional trend checks (for example entropy decrease) use windowed/aggregate comparison, not strict per-iteration monotonicity.
- Repro checks use fixed seeds and deterministic sampler path; if drift appears, record deterministic repro command and exact artifact deltas.
- Any test relying on heavy synthetic runs is an acceptance gate artifact check, not a fast unit gate.

---

## Gate Evidence Checklist (Chunk 3 Closeout)

| Gate item | Required artifact path(s) | Status |
|---|---|---|
| Fast contract tests pass | `tests/test_elevation_stage1_core_contracts.py` output log; `tests/test_elevation_stage1_checkpoint_contracts.py` output log | done (`37 passed` total with smoke suite) |
| Overlap-score formula fixture pass | `tests/test_elevation_stage1_core_contracts.py` (overlap score case output) | done |
| Smoke mode tests pass (`off|shadow|active`) | `tests/test_elevation_stage1_smoke_modes.py` output log | done |
| Shared vs decoupled post-temperature schedule pass | `tests/test_elevation_stage1_core_contracts.py` (temperature schedule case output) | done |
| Stage-1 numerics finite (`loss_lik`, `loss_ent`) | `output/.../loss_log.csv`; `output/.../run.log` | done (see S1/S2/S3/S4 run logs and summaries) |
| Posterior/cache interface present and finite | `output/.../run.log` (shape prints) or dedicated artifact notes | done |
| Chunk-2 diagnostics continuity preserved | `output/.../final_eval_train_frames.csv`; `output/.../support_metrics_train.csv`; `output/.../frame_training_visits.csv` | done |
| Synthetic S1/S2/S3/S4 completed | `output/.../*gate_summary.json`; `output/.../*gate_summary.md`; continuation run logs | done (S2/S4 threshold failures explicitly recorded) |
| Material-regression check passed (`<=10%` default) | baseline-delta note in chunk report; evaluator JSON deltas | open (Dataset-C center metric regression; waiver/risk note active) |
| Resume gate passed (real + synthetic continuation) | checkpoint file path(s); continuation `run.log`; short resume summary note | done |
| Schema/frame-set compatibility validated | checkpoint metadata dump or test log proving `checkpoint_schema_version` and `active_frame_fingerprint` handling | done |
| Refresh/remap contract validated | test log for refresh-enabled fixture (`nearest`/`reset`, remap distance reset, optimizer rebuild) | done |
| Docs updated for commit boundary | `plans/progress_overview.md`; `plans/scientific_progress.md` | done (status updates recorded 2026-02-23) |

---

## Deliverables

- Chunk 3 implementation in in-scope files.
- Elevation Stage 1 diagnostics (`loss_lik`, entropy, temperatures, reliability, invalid rate, cache/posterior stats).
- Continued Chunk 2 diagnostics (`final_eval_*`, `support_metrics_*`, frame visit CSV).
- Synthetic matrix records with commands, metrics, artifacts, and baseline deltas.
- Resume-gate evidence including synthetic continuation.
- Chunk 3 gate verdict and explicit handoff interface notes for Chunk 4.

---

## Commit Boundary Rule

Chunk 3 is commit-ready only when:
- all Chunk 3 gate checks pass,
- synthetic matrix passes with documented baselines/deltas,
- resume gate passes,
- `plans/progress_overview.md` and `plans/scientific_progress.md` are updated for this chunk,
- no blocker remains in Chunk 3 scope.

Commit message format (repo convention):
- `<description> (<model-name>)`

Example:
- `Implement elevation Stage-1 overlap, pixel-bank likelihood, and annealing core (gpt-5.3-codex)`

---

## Intermediate Plan Update (2026-02-21)

Purpose: record post-implementation status, immediate blockers, and the shortest path to a defensible Chunk-3 gate decision.

### Current issues (must be documented before closeout)

1. **S2 (C_clean) baseline-comparison evidence gap in gate narrative**
   - Current scratchpad/logs report S2 threshold failure, but the Chunk-3 plan requires explicit comparison against Chunk-2 baselines and artifact classification (`improved|unchanged|regressed`).

2. **Material regression on C_clean center error versus Chunk-2 baseline**
   - Comparator baseline artifact: `output/debug_multiframe_synth_c_exp3/eval_surfel/cube_eval.json`
   - Chunk-3 S2 artifact: `output/debug_multiframe_synth_c_gate_summary.json` (run1 metrics)
   - Recomputed deltas:
     - `mean_surface_error_m`: `0.08538 -> 0.08797` (`+3.04%`)
     - `p95_surface_error_m`: `0.20835 -> 0.22886` (`+9.84%`)
     - `center_error_m`: `0.00654 -> 0.02364` (`+261.18%`, **material regression** by plan default `>10%`)
   - Required classification for Dataset C artifact state: **`regressed`** (unless superseded by new evidence).

3. **Chunk-3 gate decision is currently under-specified**
   - Given the material-regression rule, Chunk-3 cannot be considered cleanly closed without either:
     - mitigation that reduces the regression below material threshold, or
     - an explicit written waiver with rationale and approval.

### Intermediate execution plan (next actions)

1. **Complete evidence narrative update (documentation-first)**
   - Append a compact S2 baseline-delta table and explicit C artifact classification to gate notes.
   - Cite artifact paths directly:
     - `output/debug_multiframe_synth_c_gate_summary.json`
     - `output/debug_multiframe_synth_c_run1/eval_surfel/cube_eval.json`
     - `output/debug_multiframe_synth_c_exp3/eval_surfel/cube_eval.json`

2. **Run minimal isolation experiment for root-cause directionality**
   - Execute short, same-seed C_clean parity runs under:
     - `ELEV_STAGE1_MODE=off`
     - `ELEV_STAGE1_MODE=shadow`
   - Keep all other relevant settings fixed.
   - Interpretation:
     - if `off` recovers Chunk-2-like center error and `shadow` does not, suspect Stage-1 plumbing influence;
     - if both are similar, suspect baseline/config drift or known C-shape instability unrelated to active Stage-1 weighting.

3. **Record explicit go/no-go outcome for Chunk 3**
   - **Go** only if regression is mitigated or formally waived with rationale.
   - **No-go** if material regression remains and no waiver is approved.

### Proposed command skeleton for Step 2 (parity isolation)

Use identical command body except Stage-1 mode and output directory:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=./synthetic_datasets/synthetic_cube_C_clean \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=600 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
ELEVATION_AWARE=1 \
ELEV_STAGE1_MODE=off \
SONAR_OUTPUT_DIR=./output/chunk3_c_parity_off \
python debug_multiframe.py
```

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=./synthetic_datasets/synthetic_cube_C_clean \
SONAR_NUM_FRAMES=500 \
SONAR_STAGE2_ITERS=600 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
ELEVATION_AWARE=1 \
ELEV_STAGE1_MODE=shadow \
SONAR_OUTPUT_DIR=./output/chunk3_c_parity_shadow \
python debug_multiframe.py
```

Then evaluate both outputs with `scripts/eval_synthetic_cube.py` and compare against the same Chunk-2 comparator used above.

---

## Statistical Addendum for Chunk-4 Handoff (2026-02-22)

Purpose: move key evidence out of scratch notes and into the execution record before Chunk-4 start.

### A) Baseline-relative finding remains true

- Comparator baseline (`Chunk-2` style): `output/debug_multiframe_synth_c_exp3/eval_surfel/cube_eval.json`
- Current S2 reference (`Chunk-3`): `output/debug_multiframe_synth_c_run1/eval_surfel/cube_eval.json`
- Delta recap:
  - `mean_surface_error_m`: `0.08538 -> 0.08797` (`+3.04%`)
  - `p95_surface_error_m`: `0.20835 -> 0.22886` (`+9.84%`)
  - `center_error_m`: `0.00654 -> 0.02364` (`+261.18%`, material regression under current rule)

### B) Off-vs-shadow seed sweep (directionality isolation)

Short budget (`SONAR_STAGE2_ITERS=250`) with seeds `42, 101, 202`:

- Artifacts:
  - `output/chunk3_seed_sweep/seed42_off/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep/seed42_shadow/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep/seed101_off/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep/seed101_shadow/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep/seed202_off/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep/seed202_shadow/eval_surfel/cube_eval.json`
- Center-error aggregate:
  - off: mean `0.02802`, std `0.00619`, min `0.01934`, max `0.03332`
  - shadow: mean `0.02841`, std `0.00668`, min `0.01910`, max `0.03445`
- Paired deltas (`shadow-off`): `-0.00024`, `+0.00303`, `-0.00162` m (mixed sign).

Full budget parity (`SONAR_STAGE2_ITERS=1000`) with seeds `77, 303, 404`:

- Artifacts:
  - `output/chunk3_seed_sweep_full/seed77_off/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed77_shadow/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed303_off/eval_surfel_surfels/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed303_shadow/eval_surfel_surfels/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed404_off/eval_surfel_surfels/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed404_shadow/eval_surfel_surfels/cube_eval.json`
- Center-error aggregate:
  - off: mean `0.01850`, std `0.01278`, min `0.00850`, max `0.03655`
  - shadow: mean `0.02061`, std `0.01199`, min `0.01033`, max `0.03743`
- Paired deltas (`shadow-off`): `+0.00089`, `-0.00013`, `+0.00557` m (mixed sign).

Interpretation from B:
- Stage-1 `shadow` mode is not showing a consistent directional worsening versus `off` under tested seeds.
- Observed C-clean center offset behavior is better described as seed-sensitive variance plus known cube-shape difficulty than as a clear `shadow`-specific defect.

### C) Comparator hygiene note (important)

For cross-run comparability, use evaluator input `surfels_after_training.ply` consistently.

- A subagent run initially evaluated `mesh_after_stage3.ply` for seeds `303/404`, which produced much larger center errors and is not comparable to prior surfel-based gate values.
- Corrected comparable artifacts are the `eval_surfel_surfels` JSONs listed above.

### D) Chunk-4 transition assessment

- This addendum does **not** erase the baseline-relative material-regression flag on S2 (`run1` vs `exp3`).
- It does provide evidence that Stage-1 plumbing in `shadow` is not a dominant new regression mechanism by itself.
- Recommended transition policy:
  1. proceed to Chunk 4 with explicit waiver/risk note for C-clean center metric,
  2. keep multi-seed paired (`off`/`shadow` or `off`/`active`) checks as a standing regression monitor,
  3. judge Chunk-4 success on whether coupling/support logic reduces center-error variance and cross-view dominance trends.

### E) Consolidated test-result ledger (no omissions)

1. Fast contract tests (rerun in conda env)
- `python -m py_compile debug_multiframe.py` -> pass
- `pytest tests/test_elevation_stage1_core_contracts.py -q` -> `17 passed`
- `pytest tests/test_elevation_stage1_checkpoint_contracts.py -q` -> `10 passed`
- `pytest tests/test_elevation_stage1_smoke_modes.py -q` -> `10 passed`

2. Synthetic matrix core gates
- `S1` (`output/debug_multiframe_synth_gate_summary.json`) -> `overall_pass=true`
  - run1: mean/p95/center = `0.0103008 / 0.0320440 / 0.0163423`
  - run2: mean/p95/center = `0.0103251 / 0.0320196 / 0.0163662`
- `S2` (`output/debug_multiframe_synth_c_gate_summary.json`) -> `overall_pass=false`
  - run1: mean/p95/center = `0.0879734 / 0.2288554 / 0.0236388`
  - run2: mean/p95/center = `0.0879689 / 0.2288552 / 0.0236388`
- `S3` (`output/chunk3_s3/debug_multiframe_synth_s3_gate_summary.json`) -> `overall_pass=true`
  - run1: mean/p95/center = `0.0103082 / 0.0320086 / 0.0163682`
  - run2: mean/p95/center = `0.0103158 / 0.0320081 / 0.0163716`
- `S4` continuation evaluator (`output/chunk3_s4_run2/eval_surfel/cube_eval.json`) -> `overall_pass=false`
  - mean/p95/center = `0.0876985 / 0.2243599 / 0.0296999`
  - checks: `mean_pass=false`, `p95_pass=false`, `center_pass=true`

3. Resume/schema continuity evidence
- checkpoint: `output/chunk3_s4_run1/chunk3_s4_ckpt.pth`
- resume log: `output/chunk3_s4_run2/run.log` shows loaded iter `601`, sampler restore (`cursor=303, epoch=3`), pixel-logit restore (`restored=500, reset=0`)
- state payload includes `checkpoint_schema_version=chunk3_stage1_v1` and active-frame fingerprint

4. Off-vs-shadow statistical checks (additional)
- short budget (`SONAR_STAGE2_ITERS=250`, seeds `42/101/202`) center aggregates:
  - off mean/std = `0.02802 / 0.00619`
  - shadow mean/std = `0.02841 / 0.00668`
  - paired deltas (`shadow-off`) = `-0.00024`, `+0.00303`, `-0.00162`
- full budget (`SONAR_STAGE2_ITERS=1000`, seeds `77/303/404`) center aggregates:
  - off mean/std = `0.01850 / 0.01278`
  - shadow mean/std = `0.02061 / 0.01199`
  - paired deltas (`shadow-off`) = `+0.00089`, `-0.00013`, `+0.00557`

5. Comparator hygiene correction (tracked)
- non-comparable mesh-based evals for seeds `303/404` remain at:
  - `output/chunk3_seed_sweep_full/seed303_off/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed303_shadow/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed404_off/eval_surfel/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed404_shadow/eval_surfel/cube_eval.json`
- raw mesh-based values (recorded for traceability only):
  - seed303 off/shadow center = `0.376317 / 0.339563`
  - seed404 off/shadow center = `0.418896 / 0.414915`
- canonical surfel-based corrected evals (used for comparisons) are:
  - `output/chunk3_seed_sweep_full/seed303_off/eval_surfel_surfels/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed303_shadow/eval_surfel_surfels/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed404_off/eval_surfel_surfels/cube_eval.json`
  - `output/chunk3_seed_sweep_full/seed404_shadow/eval_surfel_surfels/cube_eval.json`

6. Manual visual QA snapshot before Chunk-4 transition
- Reviewer observations (compared to last manual Chunk-2 review):
  - `output/debug_multiframe_synth_c_run1/surfels_after_training.ply`: torus-like structure and vertical-line pattern persist.
  - `output/debug_multiframe_synth_c_run1/mesh_after_stage3.ply`: still blob-like / hard to interpret.
  - `output/debug_multiframe_synth_c_run1/mesh_poisson_after_stage3.ply`: improved; follows surfels and forms regular torus.
  - `output/debug_multiframe_synth_c_run1/comparison_after_stage3_frame0.png` and `output/debug_multiframe_synth_c_run1/comparison_after_stage3_frame499.png`: low return signal and noticeably weaker than the reviewer remembers from Chunk-2.
  - Mid-trajectory frames (roughly away from extremes, e.g. around central frame ranges) are visually closer to prior Chunk-2 behavior.
- Supporting sparsity context (raw synthetic C-clean frames):
  - source: `synthetic_datasets/synthetic_cube_C_clean/sonar/*.png`
  - nonzero-pixel fraction across frames is low (mean `~1.80%`, min `~0.62%`, max `~4.08%`), consistent with low-signal appearance in many views.
- Transition interpretation:
  - no new visual failure mode was identified specific to Chunk-3 Stage-1 plumbing;
  - residual torus/bias and mesh-blob issues remain aligned with known Dataset-C and extraction limitations;
  - proceed to Chunk 4 with explicit risk note and use coupling/support outcomes as the primary improvement criterion.

7. Explicit Chunk-3 waiver / risk note (for closeout decision)
- **Waiver scope**: Dataset-C (`S2`) center-error regression vs Chunk-2 comparator is accepted as a temporary carry risk into Chunk 4.
- **Why waiver is requested**:
  - baseline-relative material regression remains (`0.00654 -> 0.02364` m in `center_error_m`),
  - but off-vs-shadow seed sweeps show mixed signed deltas (no consistent Stage-1-plumbing-specific degradation),
  - and visual review indicates edge-frame weakness (near frame 0/499) while mid-trajectory behavior remains closer to prior Chunk-2 quality.
- **Risk statement**:
  - Chunk-3 is considered functionally complete on Stage-1 infrastructure and resume contracts,
  - with known Dataset-C geometric bias/variance risk still open.
- **Required Chunk-4 acceptance focus tied to this waiver**:
  1. reduce Dataset-C center-error variance across seeds and reduce edge-frame weakness,
  2. improve cross-view dominance/support trends (not just single-run point metrics),
  3. keep comparator hygiene fixed (`surfels_after_training.ply` for cross-run metric comparability).
