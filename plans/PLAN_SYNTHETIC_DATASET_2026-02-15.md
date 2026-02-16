# Plan: Synthetic Sonar Dataset Program (High-Level)

**Date:** 2026-02-15  
**Status:** Dataset A implemented/validated; Dataset C implemented and gated, but acceptance currently failing on reconstruction thresholds (gpt-5.3-codex)

---

## Motivation

This is the most practical way to reduce ambiguity and speed up iteration.

- Real sonar data mixes many unknowns (capture quality, pose noise, environment, and optimization behavior), so root causes are hard to isolate.
- In practice, it has been near-impossible to make LLMs consistently infer the intended real-world structure from noisy real captures alone.
- Analytic synthetic shapes (sphere/cube) are mathematically defined and therefore easy to describe, verify, and reason about for both humans and LLMs.
- Synthetic datasets provide exact geometry and exact poses, turning subjective debugging into measurable validation.

---

## Goals

- Build a synthetic sonar benchmark ladder with known ground truth.
- Start with Dataset A (sphere in vacuum) as the canonical correctness test.
- Add Dataset C (cube in vacuum) as the shape-complexity follow-up using A-like gates.
- Add automatic quantitative evaluation to gate progress.
- Expand complexity only after passing Dataset A criteria.

---

## Dataset Roadmap

### Dataset A (Primary): Sphere in Vacuum

- Single perfect sphere.
- No background geometry.
- Orbiting poses with small random perturbations.
- Dataset layout matches R2 format so training can switch by dataset path/key only.
- Expected reconstruction: sphere with low radial error.

### Dataset C: Cube in Vacuum

- Single perfect cube (axis-aligned canonical default).
- No background geometry.
- Multi-band orbiting poses with small random perturbations (not single-orbit concentrated).
- Pose diversity requirement inherited from Dataset A lessons: include multiple elevation bands and/or random-shell viewpoints to avoid equatorial coverage bias.
- Dataset layout matches R2 format so training can switch by dataset path/key only.
- Goal: verify planar-face reconstruction and edge/corner behavior without background confounders.
- Expected reconstruction: six planar faces and stable cube center with low surface-distance error.

### Dataset B: Sphere + Plane Background

- Add simple seabed/plane.
- Goal: verify target/background separation.

### Dataset D (Optional): Harder Sonar Effects

- Add controlled artifacts (speckle/dropout/multipath proxy).
- Goal: robustness stress test.

---

## Success Criteria

Dataset A is considered successful when all are true:

1. `debug_multiframe.py` runs end-to-end without special-case code edits.
2. Reconstruction is visually spherical and centered correctly.
3. Quantitative thresholds pass:
   - mean radial error <= 0.05 m,
   - p95 radial error <= 0.10 m,
   - center error <= 0.03 m.
4. Repeat run with same seed yields near-identical metrics.

Dataset C is considered successful when all are true:

1. `debug_multiframe.py` runs end-to-end without special-case code edits.
2. Reconstruction is visually cube-like (flat faces + corners) and centered correctly.
3. Quantitative thresholds pass (cube-surface distance analogue of Dataset A metrics):
   - mean surface distance <= 0.05 m,
   - p95 surface distance <= 0.10 m,
   - center error <= 0.03 m.
4. Repeat run with same seed yields near-identical metrics.

---

## Strategy

1. Standardize synthetic dataset contract (format + manifest + seeds).
2. Implement Dataset A generator and evaluation harness.
3. Integrate synthetic datasets into existing training entry points.
4. Run A_clean and lock acceptance gate (defer A_noisy until clean passes).
5. Expand to C, then B, then D.
6. Versioning policy: commit generators/manifests/eval scripts, but keep generated dataset binaries out of git.
7. Dataset self-documentation policy: each generated dataset includes a human-readable settings file (markdown) plus machine-readable manifest.

---

## Current State (2026-02-15 Snapshot)

### Implemented

- Dataset A generator implemented: `scripts/generate_synthetic_sonar_dataset.py`
  - Pose mode support added: `sonar_equivalent` and `camera_with_extrinsic`
- Sphere evaluator implemented: `scripts/eval_synthetic_sphere.py`
  - Fit mode support includes `both` (reports `least_squares` + `gt_trimmed`)
- Synthetic integration in debug training implemented: `debug_multiframe.py`
  - `SONAR_DATASET=synthetic_a_clean`
  - `SONAR_DATASET_PATH` override
  - synthetic default scale `1.0`
  - synthetic default scale freeze enabled
- One-command gate runner added: `scripts/run_synthetic_a_gate.py`
- Guide added: `docs/SYNTHETIC_DATASET_GUIDE.md`
- Dataset binaries excluded from git via `.gitignore` (`synthetic_datasets/`)

### Validation Outcomes

- Backward projection consistency gate passes on A_clean:
  - median pixel error ~= 0
  - mean radial residual ~= 0.0298 m
  - p95 radial residual ~= 0.0689 m
- `debug_multiframe.py` synthetic run completes end-to-end and keeps final scale fixed at `1.000000`.
- Two repeated runs with same seed show near-identical metrics (tight drift).
- Evaluator threshold pass achieved on final surfel cloud for both repeated runs.
- `camera_with_extrinsic` pose mode smoke run passes consistency gate with low residuals and low fitted-center error.

### Planned Next Synthetic Milestone

- Dataset C (cube in vacuum) is now planned as the immediate post-A shape test.
- Intent is to keep the same operational gate structure as Dataset A (generator -> consistency gate -> training run(s) -> quantitative evaluator -> repeatability check), with cube-specific geometry metrics.
- Dataset C pose generation should default to multi-band coverage (not a single rough orbit), reflecting the Dataset A coverage lesson below.

### Important Notes for Next Session

- Synthetic images are intentionally written as 3-channel grayscale PNGs to match current SSIM/training channel expectations.
- Evaluator default fit mode is `gt_trimmed` to make center estimation robust to sparse outlier surfels while still reporting full-cloud GT radial stats.
- Canonical Dataset A acceptance gate uses sonar poses (`--pose-mode sonar_equivalent`).
- Camera-to-sonar extrinsic path remains available as an optional diagnostic via `--pose-mode camera_with_extrinsic`.
- Dataset A note: current training poses were mostly a single rough orbit, so FOV coverage concentrated in an equatorial band and could bias initialized surfel centers toward a cylindrical shell.
- Policy for Dataset C (and future synthetic refreshes): use multi-orbit/multi-band or random-shell viewpoints within bounded radius while keeping rough center-looking orientation.

## Current State Update (2026-02-16, Handoff Snapshot)

### Dataset C Implementation Delivered

- Generator expanded for cube variants (`C_clean`, `C_noisy`) with cube geometry and shape-aware consistency metrics in `scripts/generate_synthetic_sonar_dataset.py`.
- Cube evaluator added in `scripts/eval_synthetic_cube.py` (surface-distance stats, center error, histogram artifact).
- One-command Dataset C gate runner added in `scripts/run_synthetic_c_gate.py` (generate -> consistency -> train x2 -> eval x2 -> drift check).
- Training entry integration added in `debug_multiframe.py` (`SONAR_DATASET=synthetic_c_clean`, synthetic scale defaults retained).
- Documentation refreshed in `docs/SYNTHETIC_DATASET_GUIDE.md` for Dataset C workflow.

### Dataset C Gate Outcome (Canonical)

- Canonical gate mode: `--pose-mode sonar_equivalent`, multi-band coverage policy.
- Gate summary artifacts:
  - `output/debug_multiframe_synth_c_gate_summary.json`
  - `output/debug_multiframe_synth_c_gate_summary.md`
- Result: overall gate **fails** on reconstruction thresholds despite passing consistency and reproducibility.
  - Consistency gate: pass (`mean=0.098358 m`, `p95=0.244904 m`, `median pixel error ~= 0`).
  - Run1 eval: fail (`mean=0.090046 m`, `p95=0.233642 m`, `center=0.010489 m`).
  - Run2 eval: fail (`mean=0.090045 m`, `p95=0.233642 m`, `center=0.010489 m`).
  - Drift check: pass (near-zero deltas).

### Additional Tuning Attempts (Post-Gate)

- `debug_multiframe_synth_c_exp1`: range attenuation off + zero elevation init + longer stage budget; still fails (`mean=0.085301`, `p95=0.213949`).
- `debug_multiframe_synth_c_exp2`: learnable opacity ablation + longer budget; still fails (`mean=0.089113`, `p95=0.230656`).
- `debug_multiframe_synth_c_exp3`: very long training run (`Stage2=10000`, `Stage3=1000`) evaluated at `output/debug_multiframe_synth_c_exp3/eval_surfel/cube_eval.json`; still fails (`mean=0.085379`, `p95=0.208353`, `center=0.006545`).
- Combined readout across gate/experiments: best mean is exp1 (`0.085301`) and best p95 is exp3 (`0.208353`), both still far from acceptance thresholds (`<=0.05`, `<=0.10`).

### Working Hypothesis for Next Session

- Current behavior suggests a Chunk-2 quality ceiling for Dataset C under present implementation scope.
- With exp3 now measured and still failing, this is treated as a confirmed blocker under current Chunk-2-only implementation scope.
- Likely next unblock is Chunk 3/4 work (overlap/likelihood core and belief-to-geometry coupling) from `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md`.

---

## Companion Detailed Plan

Execution-level steps, file contracts, and validation gates are in:

- `plans/PLAN_SYNTHETIC_DATASET_EXECUTION_2026-02-15.md`
