# Plan: Synthetic Sonar Dataset Program (High-Level)

**Date:** 2026-02-15  
**Status:** Dataset A implemented/validated; extrinsic-path and automation extensions added (gpt-5.3-codex)

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

---

## Strategy

1. Standardize synthetic dataset contract (format + manifest + seeds).
2. Implement Dataset A generator and evaluation harness.
3. Integrate synthetic datasets into existing training entry points.
4. Run A_clean and lock acceptance gate (defer A_noisy until clean passes).
5. Expand to B, then D.
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

### Important Notes for Next Session

- Synthetic images are intentionally written as 3-channel grayscale PNGs to match current SSIM/training channel expectations.
- Evaluator default fit mode is `gt_trimmed` to make center estimation robust to sparse outlier surfels while still reporting full-cloud GT radial stats.
- Canonical Dataset A acceptance gate uses sonar poses (`--pose-mode sonar_equivalent`).
- Camera-to-sonar extrinsic path remains available as an optional diagnostic via `--pose-mode camera_with_extrinsic`.
- Open follow-up: current training poses are mostly a single rough orbit, so FOV coverage concentrates in an equatorial band and can bias initialized surfel centers toward a cylindrical shell. Future pose sampling should add multi-orbit or random-shell viewpoints (within a bounded radius and still roughly looking at the object center).

---

## Companion Detailed Plan

Execution-level steps, file contracts, and validation gates are in:

- `plans/PLAN_SYNTHETIC_DATASET_EXECUTION_2026-02-15.md`
