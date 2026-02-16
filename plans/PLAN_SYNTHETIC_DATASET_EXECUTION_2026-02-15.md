# Plan: Synthetic Sonar Dataset Program (Detailed Execution)

**Date:** 2026-02-15  
**Status:** Implemented for Dataset A with validation artifacts and gate automation extensions (gpt-5.3-codex)  
**Depends on:** `plans/PLAN_SYNTHETIC_DATASET_2026-02-15.md`

---

## Scope

Implement Dataset A (sphere vacuum) end-to-end:

- synthetic data generation,
- training integration,
- quantitative evaluation,
- acceptance gate.

Datasets B and D are out of scope for initial implementation. `A_noisy` is also deferred until `A_clean` passes.

---

## Implementation Snapshot (2026-02-15)

### Code Delivered

- Added: `scripts/generate_synthetic_sonar_dataset.py`
- Added: `scripts/eval_synthetic_sphere.py`
- Added: `scripts/run_synthetic_a_gate.py`
- Updated: `debug_multiframe.py` (synthetic key/path + default frozen scale for synthetic)
- Added: `docs/SYNTHETIC_DATASET_GUIDE.md`
- Updated: `.gitignore` to ignore `synthetic_datasets/`
- Updated: generator/evaluator contracts for extrinsic pose mode + dual-fit reporting

### Step Status

1. Step 1 (generator skeleton): **Done**
2. Step 2 (pose synthesis + COLMAP export): **Done**
3. Step 3 (elevation-integrated forward simulation): **Done**
4. Step 3b (backward-projection consistency gate): **Done**
5. Step 4 (debug training integration): **Done**
6. Step 5 (quantitative evaluator): **Done**
7. Step 6 (acceptance/repeatability runs): **Done for A_clean**
8. Follow-up: extrinsic-aware generator mode + automated gate runner: **Done (smoke validated)**

### Validation Artifacts and Metrics

- Dataset root used:
  - `synthetic_datasets/synthetic_sphere_A_clean`
- Consistency gate artifact:
  - `synthetic_datasets/synthetic_sphere_A_clean/consistency_gate.json`
  - Result: `pass=true`, mean residual `0.0298 m`, p95 residual `0.0689 m`
- Training run outputs:
  - `output/debug_multiframe_synth_run1`
  - `output/debug_multiframe_synth_run2`
  - Both completed and report final scale `1.000000`
- Evaluation outputs:
  - `output/debug_multiframe_synth_run1/eval_surfel/sphere_eval.json`
  - `output/debug_multiframe_synth_run2/eval_surfel/sphere_eval.json`
  - Both pass thresholds (mean/p95/center)
- Reproducibility checks:
  - Repeated training metrics show negligible drift (loss/SSIM/radial/center)
  - Re-generated dataset (same seed) produced identical COLMAP files, identical sonar image hashes, and identical gate outputs (except generation timestamp field)

### Implementation Decisions Recorded

1. Sonar image channel contract
   - Generator writes sonar PNGs as 3-channel grayscale RGB.
   - Reason: existing SSIM/loss path in training expects 3 channels.

2. Sphere fit policy in evaluator
   - Evaluator default fit mode is `gt_trimmed`.
   - Reason: robust center estimate in presence of sparse outlier surfels; full-cloud GT radial metrics are still reported separately.

3. Extrinsic-path coverage status
   - Generator now supports both:
     - `sonar_equivalent` (canonical Dataset A gate behavior; sonar-pose contract),
     - `camera_with_extrinsic` (camera poses exported so runtime camera->sonar extrinsic path is exercised).
   - `camera_with_extrinsic` smoke run passes consistency gate with low residuals and low fitted-center error.

### Recommended Next Work

1. Run full two-training-run acceptance gate using `scripts/run_synthetic_a_gate.py` in `sonar_equivalent` mode and archive summary artifacts.
2. Keep `camera_with_extrinsic` as optional diagnostic only (not canonical acceptance gate).
3. Start Dataset B (sphere + plane) only after freezing extrinsic diagnostic policy for CI-style checks.
4. Improve pose diversity for later runs: avoid single-orbit-only sampling that over-concentrates FOV near the sphere equator and can produce cylindrical surfel-center distributions; add multi-orbit and random-shell viewpoints (bounded radius, still roughly center-looking).

---

## In-Scope Files

- New: `scripts/generate_synthetic_sonar_dataset.py`
- New: `scripts/eval_synthetic_sphere.py`
- Update: `debug_multiframe.py` (synthetic dataset path support)
- Optional doc: `docs/SYNTHETIC_DATASET_GUIDE.md`

### Version Control Policy

- Commit generator/evaluation code, manifests, and plan/docs.
- Do not commit generated sonar image datasets or other large binary dataset artifacts.
- Reproducibility contract is code + manifest + seed, not binary snapshots in git.

---

## Dataset A Contract

### Geometry and Sensor Defaults

- Sphere center: `(0.0, 0.0, 0.0)` meters.
- Sphere radius: `0.8` meter (not 1.0; see geometry note below).
- Sonar image size: `256x200`.
- Azimuth FOV: `120 deg`.
- Elevation FOV: `20 deg`.
- Range limits: `[0.2, 3.0]` meters.

### Pose Policy

- Orbit radius default: `2.0` meters.
- Full 360-degree azimuth coverage.
- Elevation sweep in `[-12 deg, +12 deg]`.
- Small jitter:
  - translation sigma `0.02 m`,
  - rotation sigma `1.5 deg`.
- Orientation policy: look-at sphere center with jitter constraints.

### Output Layout

Dataset root example:

- `synthetic_sphere_A_clean/sparse/0/images.txt`
- `synthetic_sphere_A_clean/sparse/0/cameras.txt`
- `synthetic_sphere_A_clean/sparse/0/points3D.txt`
- `synthetic_sphere_A_clean/sonar/sonar_000000.png`
- `synthetic_sphere_A_clean/manifest.json`
- `synthetic_sphere_A_clean/DATASET_SETTINGS.md`

(`A_noisy` variant deferred until `A_clean` acceptance gate passes.)

Layout contract:

- Synthetic dataset directory/file structure must match the current R2 sonar dataset shape (`sparse/0/*` + `sonar/*.png`) so `debug_multiframe.py` can swap between R2 and synthetic by dataset key/path only.

### Manifest Contract

`manifest.json` must include:

- dataset id and variant (`A_clean` or `A_noisy`),
- geometry parameters,
- sonar model parameters,
- pose generation parameters,
- seed and derived seeds,
- generation timestamp,
- optional noise model settings.

`DATASET_SETTINGS.md` must include a human-readable summary of the same generation settings (geometry, sonar params, pose policy, noise policy, seed, script version, and generation timestamp).

---

## Work Plan

### Step 1: Build generator skeleton

Implement CLI in `scripts/generate_synthetic_sonar_dataset.py`:

- inputs: output dir, frame count, seed, clean/noisy toggle,
- deterministic RNG handling,
- manifest writing,
- dataset settings markdown (`DATASET_SETTINGS.md`) writing,
- folder creation and COLMAP text writer stubs.

Completion check:
- script runs and writes valid folder tree + manifest + `DATASET_SETTINGS.md`.

### Step 2: Implement pose synthesis + COLMAP export

- implement orbit + look-at + jitter pose generation,
- convert to COLMAP-compatible `qvec` and `tvec`,
- write `images.txt` and `cameras.txt`,
- write minimal `points3D.txt` placeholder accepted by loader.

Completion check:
- `readColmapSceneInfo(..., sonar_mode=True)` loads generated dataset without error.

### Step 3: Implement sonar frame simulation

**Critical: sonar forward model with elevation integration.**

A sonar pixel at (azimuth_col, range_row) is NOT a single ray. It represents the integrated acoustic return across all elevation angles within the beam at that (azimuth, range). The forward model must:

1. For each azimuth column θ (from pixel-to-angle mapping):
   - Sweep elevation φ across `[-half_elev_fov, +half_elev_fov]` with N samples (e.g., 64).
   - For each (θ, φ), compute ray direction in sonar frame:
     - `dir = (sin(θ)*cos(φ), sin(φ), cos(θ)*cos(φ))` (or per repo convention).
    - Transform ray to world frame using pose.
    - Intersect ray with sphere analytically (quadratic solve).
    - If hit, keep only the nearest positive root (front/outer visible surface) and deposit intensity at the corresponding range bin.

2. Accumulate all elevation hits into the (azimuth_col, range_row) image.

3. Intensity model for `A_clean`: binary (hit deposits 1.0 per elevation sample that hits). This produces a natural brightness gradient — pixels with more elevation hits are brighter (center of sphere arc is brighter than edges). Use deterministic global normalization (not per-frame):
   - `img_float = hit_count / N_elev_samples`,
   - `img_uint8 = round(clamp(img_float, 0, 1) * 255)`.

This elevation integration is what gives sonar images their characteristic appearance — a sphere projects as a bright arc in (azimuth, range) space, not a circle.

Completion check:
- images are generated and show expected arc pattern with orbit-consistent motion,
- visual spot-check: center columns should be brighter than edge columns for the sphere.

### Step 3b: Backward projection consistency gate

Before any training, validate that the existing backward projection recovers the known geometry:

- Run `sonar_frame_to_points()` on the synthetic images with the synthetic poses.
- For each recovered 3D point, compute distance to the ground-truth sphere surface.
- Pass criteria:
  - no NaN/Inf in recovered points,
  - reprojection round-trip consistency in the source frame (median pixel error <= 1 px on valid points),
  - radial residual sanity for asymmetric forward/backward model: mean <= 0.15 m, p95 <= 0.30 m,
  - recovered point cloud is visually spherical/equatorial-band consistent (manual spot-check).

This validates forward/backward projection consistency. If this step fails, the generator or the projection code has a convention mismatch and no amount of training will produce correct results.

Completion check:
- `sonar_frame_to_points()` output passes round-trip and residual sanity thresholds for Dataset A.

### Step 4: Integrate with debug training flow

Update `debug_multiframe.py`:

- allow synthetic dataset key/path with explicit contract:
  - `SONAR_DATASET=synthetic_a_clean` resolves to default generated location,
  - `SONAR_DATASET_PATH=<abs_or_rel_path>` overrides location directly,
- set `INIT_SCALE_FACTOR=1.0` for synthetic dataset presets,
- freeze sonar scale-factor optimization by default for synthetic runs (scale is treated as known-correct in this phase),
- maintain existing behavior for legacy/r2.

**Camera-to-sonar extrinsic handling:** The pipeline applies a camera-to-sonar transform (8cm back, 10cm up, 5 deg pitch). Synthetic poses must be *camera* poses (not sonar poses), and the orbit geometry must account for this offset so the sonar sensor ends up at the intended orbit radius. At 2.0m orbit this is small (~5cm effective shift) but must be documented and consistent. The generator should either:
- generate camera poses such that the *sonar* (after extrinsic) is at the orbit radius, or
- generate sonar poses directly and set the extrinsic to identity for synthetic runs.

Recommend the first approach (camera poses + real extrinsic) since it also tests the extrinsic path.

Completion check:
- one-command run works on synthetic dataset path,
- effective scale remains fixed at 1.0 during the run.

### Step 5: Implement quantitative evaluator

Create `scripts/eval_synthetic_sphere.py` to:

- read reconstruction PLY + dataset manifest,
- compute radial residuals to GT sphere,
- optionally fit sphere and report fitted center/radius,
- write `sphere_eval.json` and residual histogram image,
- emit pass/fail against thresholds.

Completion check:
- evaluator runs on a reconstruction output and writes artifacts.

### Step 6: Run acceptance gate

Required runs:

1. Train on `A_clean` and evaluate.
2. Re-run same config/seed and compare metric drift.

(`A_noisy` run deferred until `A_clean` gate passes.)

Gate criteria:

- no NaN/Inf during run,
- visual sphere quality acceptable,
- threshold pass on `A_clean`,
- scale factor is frozen at the configured synthetic default (1.0),
- reproducibility within tight tolerance.

---

## Validation Checklist

1. Generator reproducibility (same seed => same frames/manifests).
2. Generated dataset includes `DATASET_SETTINGS.md` and it matches `manifest.json` values.
3. Loader compatibility (`readColmapSceneInfo` in sonar mode).
4. Backward projection consistency (Step 3b): round-trip pixel and residual sanity thresholds pass.
5. Training smoke success on `A_clean`.
6. Scale factor is frozen at 1.0 for synthetic runs in this phase.
7. Evaluation artifact generation.
8. Threshold gate pass.

---

## Geometry Note (opus4.6 review, 2026-02-15)

Sphere radius 1.0m at orbit radius 2.0m places the far surface at exactly 3.0m — the range max. Any pose jitter pushes returns beyond the range limit, causing silent clipping artifacts. Radius reduced to 0.8m so the far surface is at 2.8m, leaving 0.2m of margin for jitter and the camera-to-sonar offset.

---

## Risks and Mitigations

1. Pose convention mistakes
   - Add small pose sanity visualization and projection spot-check.
2. Simulator mismatch with training renderer
   - Start with simple assumptions and compare one-frame rendered-vs-GT patterns.
   - **Step 3b (backward projection gate) is the primary defense against this.** If forward and backward projections are inconsistent, this gate catches it before any training time is wasted.
3. Thresholds too strict/loose initially
   - Keep thresholds explicit in evaluator config and tune after first baselines.
4. Camera-to-sonar extrinsic mismatch
   - Synthetic poses are camera poses; the pipeline applies the extrinsic internally. The generator must account for this offset in orbit geometry. Spot-check by verifying that `sonar_frame_to_points()` output is centered on the sphere, not offset by ~10cm.
5. Elevation integration model fidelity
    - The forward model integrates over elevation but the backward projection (`sonar_frame_to_points()`) picks a single elevation (zero or random). This asymmetry is expected and correct — it mirrors real sonar behavior. But it means backward-projected points will not perfectly tile the sphere surface; they will cluster near the equatorial band. This is not a bug.

---

## Clarifications Captured (user + gpt-5.3-codex, 2026-02-15)

1. Scale factor policy for synthetic Dataset A
   - Keep scale factor frozen and set to the known-correct value by default (`1.0`) during this implementation.
   - Scale-learning behavior on synthetic data is a useful future experiment, but is not part of this phase.

2. Sphere intersection policy
   - Use only the physically visible front/outer surface for intensity deposition (nearest positive quadratic root).
   - Do not accumulate back-surface intersections.

3. Dataset compatibility policy
   - Synthetic dataset format/structure should mirror R2 so the existing training flow can do a simple dataset swap.

---

## Deliverables

- Synthetic Dataset A clean folder (noisy deferred).
- Generator script with elevation-integrated forward model.
- Per-dataset settings markdown (`DATASET_SETTINGS.md`) generated alongside `manifest.json`.
- Sphere evaluation script.
- `debug_multiframe.py` synthetic path integration.
- Backward projection consistency gate artifacts (Step 3b).
- Metrics/artifacts proving Dataset A gate status.
