# Plan: Synthetic Sonar Dataset Program (High-Level)

**Date:** 2026-02-15  
**Status:** Active planning (gpt-5.3-codex)

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

---

## Companion Detailed Plan

Execution-level steps, file contracts, and validation gates are in:

- `plans/PLAN_SYNTHETIC_DATASET_EXECUTION_2026-02-15.md`
