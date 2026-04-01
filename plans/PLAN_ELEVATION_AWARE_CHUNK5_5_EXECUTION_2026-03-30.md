# Plan: Elevation-Aware Chunk 5.5 Execution

**Date:** 2026-03-30  
**Status:** Draft diagnostic follow-on to Chunk 5 (gpt-5.4)  
**Scope:** Chunk-5 zero-signal investigation, gate-by-gate rerun instrumentation, and execution guidance before any further late-normal claims

**Navigation:** Section 2 (Progress Log) begins at line 125 if you want to jump directly to executed work and findings.

---

## Section 1: Plan

## Goal

Establish why Chunk-5 late normals are producing effectively zero usable signal in the current sonar path, and make the next rerun diagnostic enough that we can say exactly where anchors are being lost in the chain:

- support,
- confidence,
- 4-neighbor readiness,
- finite normal formation,
- surfel match.

Chunk 5.5 is a diagnostic and unblock tranche, not a close-the-gate tranche.

Its job is to answer:

> On a controlled rerun, where exactly does Chunk-5 supervision die, and is the blocker the center posterior, the neighbor posterior, the entropy threshold, the local-geometry construction, or the surfel matching step?

---

## Relationship To Chunk 5

- `plans/PLAN_ELEVATION_AWARE_CHUNK5_EXECUTION_2026-03-16.md` remains the main Chunk-5 implementation and validation contract.
- This document contains the migrated investigation addendum and the immediate execution plan for the zero-signal blocker identified after that plan was written.
- No Chunk-5 success claim should be made from a run that still shows zero or near-zero signal through the gate stack documented here.

---

## Immediate Rule For The Next Rerun

Before changing thresholds or algorithms, the first rerun must log what each gate is actually doing.

Minimum required per-step or per-report diagnostics for the first controlled rerun:

- anchor count,
- center-supported count / fraction,
- center-confident count / fraction,
- interior-anchor count / fraction,
- all-4-neighbors-confident count / fraction,
- finite-normal count / fraction,
- matched-surfel count / fraction,
- skipped-by-reason counts for each gate transition,
- center-posterior entropy summary,
- neighbor-posterior entropy summary,
- applied normal-loss count and scalar.

The first diagnostic question is not "does active mode help?"

It is:

> Out of all anchor pixels, how many survive each gate, and where is the collapse happening?

---

## Recommended Immediate Work Order

1. Rerun the controlled Chunk-5 synthetic test with explicit short-run overrides.
2. Add or expose gate-by-gate logging for `support -> confidence -> 4-neighbor -> finite -> match` before making algorithm changes.
3. Record whether the collapse happens mostly at:
   - center confidence,
   - neighbor confidence,
   - finite-difference geometry,
   - or association to visible surfels.
4. Only after that evidence exists, decide whether the first fix should target:
   - confidence threshold calibration,
   - center-vs-neighbor posterior parity,
   - neighborhood-closure construction,
   - or surfel matching tolerances.

### Next Execution Plan

This next pass should stay inside the existing Chunk-5 design, but it should no longer preserve the current 4-neighbor pixel finite-difference normal construction. That stencil does not have a convincing physical interpretation for sonar and should be removed from the active execution path.

For now, only **Phase A** is committed execution work. **Phase B** is optional and should remain deferred unless the Phase-A reruns show a clear reason to define a replacement local-geometry path.

Guiding rule:

- prefer the smallest change that can make the existing Chunk-5 path produce meaningful nonzero signal,
- remove the 4-neighbor pixel finite-difference normal construction instead of spending more time validating it,
- treat new mechanisms as deferred until the current path has been given a fair minimal-complexity attempt,
- do not implement the surfel-neighbor loss in this tranche.

Execution order:

1. **Phase A: keep the new gate logging, but revise the Chunk-5 local-geometry path so it no longer depends on the 4-neighbor pixel finite-difference stencil.**
   - Remove the current `left/right/up/down -> cross product` geometry construction from the active Chunk-5 execution path.
   - Preserve the gate artifact requirements so the rerun still reports where supervision is being lost.

2. **Phase A: after that removal, rerun the same controlled cube diagnostic family.**
   - Use the same dataset, frame subset, renderer tuple, and short-run posture already recorded in Section 2 unless the replacement geometry path requires one clearly documented change.
   - Continue to record the diagnostics that remain meaningful after stencil removal: anchors, support, center confidence, skip reasons, center entropy, neighbor entropy when still applicable, and any downstream counts that are still exercised.

3. **Phase A: run a few controlled reruns after removal and inspect what still survives.**
   - The purpose is not yet to restore full normal supervision.
   - The purpose is to determine what the remaining Chunk-5 path still tells us once the physically unconvincing stencil is gone.

4. **Phase B (optional, deferred): only if Phase A indicates the simplified Chunk-5 path is still worth salvaging, define one minimal replacement local-geometry path.**
   - Do not start Phase B automatically.
   - Only enter Phase B if the Phase-A reruns suggest that the remaining path is informative enough to justify one minimal replacement.

5. **After Phase A, decide whether to stop, continue with optional Phase B, or conclude that Chunk-5 needs deeper redesign.**
   - If Phase A already shows the remaining path is not useful, record that and stop without adding replacement machinery.
   - If Phase A suggests the path is still worth simplifying further, Phase B may define one minimal replacement.

Explicit non-goals for the next pass:

- do not implement surfel-neighbor normal supervision,
- do not add new densification behavior,
- do not preserve the current 4-neighbor finite-difference path merely for continuity with the earlier implementation,
- do not broaden the effort into a new feature branch before the current path is tested under minimal changes.

Success criteria for the next pass:

- successful removal of the 4-neighbor finite-difference stencil from the active Chunk-5 path,
- controlled reruns that preserve the diagnostic visibility of the remaining gate stack,
- a documented conclusion about what still survives after stencil removal,
- a documented decision on whether optional Phase B is justified at all.

---

## New-Session LLM Prompt

Use the following prompt to start a fresh session for the Chunk-5.5 diagnostic pass:

```text
You are continuing the Chunk-5 late-normal investigation in /home/gavin/Unbiased_Surfel_sonar.

Read these first:
- plans/PLAN_ELEVATION_AWARE_CHUNK5_EXECUTION_2026-03-16.md
- plans/PLAN_ELEVATION_AWARE_CHUNK5_5_EXECUTION_2026-03-30.md
- docs/CHUNK5_NORMALS_EXPLAINER.md

Primary goal:
- Determine exactly where the Chunk-5 normal-supervision pipeline is collapsing in the current controlled synthetic rerun.

The first step is mandatory:
- rerun the controlled test,
- and log what each gate is actually doing for the Chunk-5 path:
  - support,
  - confidence,
  - 4-neighbor readiness,
  - finite normal formation,
  - surfel match.

Do not start by changing thresholds blindly.
Do not assume the root cause is only the entropy threshold.

Required deliverables from the first rerun:
- the exact rerun command used,
- the output path,
- per-gate counts/fractions,
- skipped-by-reason counts,
- center entropy summary,
- neighbor entropy summary,
- a short conclusion stating where the main collapse happens.

Important implementation question to resolve from code and logs:
- Are center anchors using a stronger posterior path than queried neighbors, and is that asymmetry the main reason `conf_cov` stays at zero?

Only after the rerun and gate breakdown are recorded should you propose or implement the first fix.
If you instrument code, keep the changes minimal and focused on diagnostics.
```

---

## Section 2: Progress Log

### Migrated Investigation Addendum (from Chunk-5 plan)

This addendum records the most relevant takeaways from the post-backprojection-fix investigation of the cube run `output/cube_20frames_30k_azimuth45_fixedpos_backprojfix`, including the useful parts of the prior scratchpad analysis, the user's corrections, and a fresh code/artifact audit.

### Run Context

- Run inspected: `output/cube_20frames_30k_azimuth45_fixedpos_backprojfix`
- Training posture: `Stage1=0`, `Stage2=30000`, `Stage3=1`, scale frozen at `1.0`
- Chunk-5 posture in that run: `normal_mode=shadow`, `effective_normal=shadow`, `densify=0`, `effective_densify=off`
- Opacity posture in that run: fixed to `0.999`; opacity gradients disabled

### What Was Correct In The Earlier Investigation

- The initialization path in `debug_multiframe.py` seeds surfel rotations from camera-facing directions (`cam_pos - point`) rather than from finite-difference surface normals.
- `utils/point_utils.py:sonar_points_to_normals()` is a real alternative normal estimator and remains unused in the current initialization path.
- Chunk 5 already contains substantial infrastructure, not just a placeholder:
  - expected-elevation estimation from Stage-1 posteriors,
  - finite-difference expected normals from local pixel neighborhoods,
  - surfel/expected-point association,
  - sign-agnostic cosine normal loss,
  - optional densification hooks with geometry-informed spawn orientation,
  - checkpoint/runtime-state support.
- The user's correction is important: `shadow` mode is not the same thing as "Chunk 5 does not exist." In `shadow`, diagnostics are computed, but the training objective and spawning behavior are not modified.
- The user's point about surfel attrition is also valid: `2953 -> 583` by itself is not evidence of failure. Fewer surfels can be better if they are well placed and well oriented.

### Fresh Diagnosis: The Main Blocker Is Effective Zero Chunk-5 Signal

- The dominant current problem is not merely that Chunk 5 was run in `shadow` mode. The deeper issue is that Chunk 5 produced no usable confident normal matches in this run.
- Evidence from `run.log`:
  - after the Chunk-5 start point, `cov` rises to `1.0000`, meaning the sparse bank is geometrically covered,
  - but `conf_cov` stays `0.0000`,
  - `finite=0.0`, `match=0.0`, and `normal=0.000000` remain flat through the rest of training.
- This means that even if the same run had been switched from `shadow` to `active` without changing the confidence regime, Chunk 5 still would have contributed essentially no normal-loss gradient.
- The likely reason is the confidence gate itself:
  - `compute_confidence_mask()` gates on posterior entropy,
  - with `7` elevation bins and `ELEV_NORMAL_CONFIDENCE_THRESH=0.5`, the entropy cutoff is about `0.5 * log(7) ~= 0.97`,
  - but the run logs show posterior entropies around `1.86-1.93`, so the gate rejects everything.
- Therefore the immediate Chunk-5 execution requirement is: make `conf_cov`, `finite`, and `match` become nonzero under a controlled synthetic run before treating late normals as truly active.

### Rotation Initialization Is Imperfect, But Not The First Fix

- The earlier scratchpad was right to inspect rotation seeding, but the fresh artifact check indicates that initialization is not the first-order blocker.
- Measured on exported surfel states from the inspected run:
  - initial surfels are materially better aligned to nearest cube-face normals than final Stage-2 surfels,
  - initial `|cos(normal, nearest_face_normal)|`: mean `0.705`, median `0.783`,
  - Stage-2 `|cos(normal, nearest_face_normal)|`: mean `0.521`, median `0.502`.
- On face-interior surfels only, the same pattern persists:
  - initial mean `0.717`, median `0.798`,
  - Stage-2 mean `0.552`, median `0.586`.
- So the current system is not simply starting bad and staying bad. It is starting somewhat better and then degrading because the optimizer has no effective late-normal corrective signal.

### Geometry Also Degrades During Optimization

- The floaters are not just an initialization artifact.
- Using the known cube geometry from `synthetic_datasets/synthetic_cube_C_azimuth45_fixedpos/DATASET_SETTINGS.md`:
  - initial mean absolute distance to cube surface: about `3.45 cm`,
  - Stage-2 mean absolute distance to cube surface: about `7.58 cm`.
- Surfel counts within `5 cm` of the surface degrade sharply:
  - initial: `2199 / 2953`,
  - Stage 2: `241 / 583`.
- This means training plus pruning is allowing the representation to drift into a worse geometric configuration when Chunk 5 is not effectively constraining normals.

### Large Surfels Matter, But They Look Secondary

- Oversized surfels are a real contributor to the visible bad-segment failure mode.
- In the inspected run, the number of surfels with radius `>= 0.1 m` grows from `30` initially to `66` after Stage 2, and the maximum radius grows to about `0.39 m`.
- Larger surfels are more likely to be poorly oriented than smaller ones.
- However, scale control does not look like the primary root cause, because even non-giant surviving surfels still show weak orientation quality after Stage 2.
- Practical reading: scale control and pruning should be treated as a second-stage cleanup after Chunk 5 is producing real normal supervision.

### Important Correction To An Earlier Intuition About Compensation

- In this inspected run, opacity is fixed at `0.999` and not learnable.
- So the renderer cannot explain away bad orientation by trading it off against opacity in this configuration.
- The remaining escape valves are surfel movement, scaling changes, and pruning/retention dynamics.

### Support And Ownership Readout

- Support is not catastrophically low overall, but ownership remains diffuse.
- In `support_metrics_train.csv`, most frames have `single_view_owner_count` equal to `0`, while `nearest_owner_count` is nonzero but still modest.
- This supports the interpretation that the model can remain broadly visible across views without establishing crisp view-linked geometric ownership strong enough to rescue orientation on its own.

### Updated Chunk-5 Execution Priority

The next-execution priority should be:

1. **Make Chunk 5 produce nonzero confident matches.**
   - Treat this as the immediate gate.
   - In the first controlled rerun, require nonzero `conf_cov`, `finite`, and `match` in `run.log`.
   - The most likely knobs are `ELEV_NORMAL_CONFIDENCE_THRESH`, the normal start iteration, and any other settings that keep the posterior from being rejected as too uncertain.

2. **Then run true late-normal supervision in `active` mode.**
   - `ELEV_NORMAL_MODE=active` is necessary, but it is not sufficient by itself.
   - Use an explicit Stage-2 budget large enough that the normal schedule actually operates over a meaningful window.

3. **Only after late normals are genuinely active, revisit scale-control and pruning policy.**
   - Large-surface dominance, continuity gaps, and floater cleanup should be re-measured after Step 2.

4. **Densification remains optional and should follow a normal-signal proof.**
   - The user's suggestion is adopted here: Chunk-5 densification is a useful exploration mechanism, but it should not be treated as a substitute for effective supervision of existing surfels.

### Longer-Horizon Direction: Beyond Pure Gradient Descent

The user's RL-like exploration idea is relevant future work and should be recorded explicitly as out-of-scope-for-initial-close but in-scope-for-follow-on design:

- Current Chunk-5 machinery can **spawn** surfels at persistent high-error locations.
- Current code does **not** yet have an explicit mechanism to:
  - reorient existing surfels from geometric evidence outside normal gradient descent,
  - migrate surfel centers toward expected surface points,
  - cull surfels based directly on normal-quality inconsistency,
  - split/merge surfels using local orientation coherence.
- A later exploration tranche may test policies such as:
  - perturb-and-keep orientation search,
  - geometry-guided reorientation snaps,
  - normal-quality culling,
  - migration toward expected surface support,
  - local split/merge based on orientation consistency.

### Practical Gate Update For Chunk 5

Before declaring any future Chunk-5 run meaningful, require all of the following in the run artifacts:

- `ELEV_NORMAL_MODE=active` when evaluating real late-normal effect,
- nonzero `conf_cov`,
- nonzero `finite`,
- nonzero `match`,
- nonzero applied normal term in the objective,
- explicit comparison of geometry/orientation metrics against initialization and against the chosen post-fix baseline.

If those conditions are not met, the run should be treated as a Chunk-5 diagnostics run, not as an exercised late-normal refinement result.

---

### Controlled Rerun Findings (2026-03-30)

Executed a short controlled diagnostic rerun after adding focused Chunk-5 gate logging to `debug_multiframe.py`.

### Command Used

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate unbiased_surfel_sonar && \
SONAR_DATASET=synthetic_c_clean \
SONAR_DATASET_PATH=/home/gavin/Unbiased_Surfel_sonar/synthetic_datasets/synthetic_cube_C_azimuth45_fixedpos \
SONAR_OUTPUT_DIR=./output/chunk5_5_diag_cube20_short_active_v1 \
SONAR_NUM_FRAMES=20 \
SONAR_FRAME_INDICES=0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475 \
SONAR_STAGE2_ITERS=40 \
SONAR_STAGE3_ITERS=1 \
SONAR_FREEZE_SCALE=1 \
ELEV_STAGE1_MODE=shadow \
ELEV_COUPLE_MODE=shadow \
ELEV_SUPPORT_MODE=shadow \
ELEV_NORMAL_MODE=active \
ELEV_DENSIFY=0 \
ELEV_DENSIFY_MODE=off \
ELEV_NORMAL_RAMP_START_ITER=1 \
ELEV_NORMAL_RAMP_END_ITER=20 \
ELEV_NORMAL_ELEV_START_ITER=1 \
ELEV_NORMAL_CONFIDENCE_THRESH=0.5 \
SONAR_RENDER_MODE=2dgs \
SONAR_OCCLUSION_MODE=ray_binned \
SONAR_LAMBERTIAN_MODE=leaky \
python debug_multiframe.py
```

### Output Path

- Run directory: `output/chunk5_5_diag_cube20_short_active_v1/`
- Main log: `output/chunk5_5_diag_cube20_short_active_v1/run.log`
- Gate log: `output/chunk5_5_diag_cube20_short_active_v1/chunk5_gate_log.csv`

### Added Diagnostic Artifact

- `chunk5_gate_log.csv` now records, per training step:
  - anchor count,
  - center-supported count / fraction,
  - center-confident count / fraction,
  - interior-anchor count / fraction,
  - all-4-neighbors-confident count / fraction,
  - finite-normal count / fraction,
  - matched-surfel count / fraction,
  - skipped-by-reason counts,
  - center entropy summary,
  - neighbor entropy summary,
  - applied normal-loss count and scalar.

### Gate Breakdown Summary

- Support is not the blocker in this rerun:
  - `center_supported_frac = 1.0000` on every logged step.
- The main collapse starts at center confidence:
  - anchors per step: `394` to `496`,
  - center-confident count: `0` to `16`,
  - center-confident fraction: `0.0000` to `0.0374`.
- The surviving center-confident anchors then collapse almost entirely at the 4-neighbor gate:
  - all-4-neighbors-confident count is usually `0`,
  - only a few steps reach `1` survivor (`iter=2, 8, 15, 22, 28, 35`).
- Finite normal formation is not the primary blocker in this rerun:
  - whenever `all_neighbors_confident_count = 1`, `finite_count = 1`.
- Surfel matching is also not the primary blocker in this rerun:
  - whenever `finite_count = 1`, `match_count = 1`.
- Applied normal supervision is therefore effectively negligible:
  - `applied_count = 0` on most steps,
  - `applied_count = 1` on only `6 / 40` Stage-2 steps.

### Entropy Summary

- With `7` bins, `max_entropy = log(7) = 1.945910`.
- Center entropy from supported anchors remains near-maximal:
  - mean: `1.767848` to `1.905852`,
  - median: usually `1.945910`,
  - p95: `1.945910` on all logged steps.
- Neighbor entropy is also high:
  - mean: `1.722179` to `1.871454` on queried-neighbor steps,
  - median: usually `1.945910`,
  - p95: `1.945910` on all queried-neighbor steps.

### Conclusion From The Rerun

The current controlled rerun shows that the Chunk-5 pipeline is collapsing primarily at the confidence gate, first at the center pixel and then at the neighbor-confidence closure gate.

- `support -> confidence` is the dominant loss of supervision,
- `4-neighbor readiness` is the secondary collapse,
- `finite normal formation` is not the main blocker once an anchor survives readiness,
- `surfel match` is not the main blocker once a finite normal exists.

This rerun produced tiny intermittent nonzero signal, but still effectively zero usable supervision for practical Chunk-5 refinement.

### Center-vs-Neighbor Posterior Asymmetry Update

- The code still contains a real asymmetry:
  - center anchors use `logits / temp_post + loglik`,
  - queried neighbors use evidence-only `masked_softmax(query_loglik, ...)`.
- However, under this rerun posture (`ELEV_STAGE1_MODE=shadow`), the Stage-1 logits are not trained and remain near their zero initialization.
- Therefore, this rerun does not support the claim that center-vs-neighbor asymmetry is the dominant blocker by itself.
- The stronger immediate reading is that both center and neighbor posteriors remain too diffuse under the current confidence calibration.

### Immediate Follow-On Direction

The immediate follow-on direction is to simplify the current Chunk-5 path by removing the sonar-pixel 4-neighbor finite-difference normal construction, then retest the remaining path with minimal changes before deciding whether any further replacement is warranted.

- First: remove the current finite-difference stencil from the active Chunk-5 path.
- Second: run a few controlled reruns and inspect what remains informative.
- Third: decide whether optional Phase B is worth doing.
- Only after those checks should the project decide whether Chunk-5 needs a deeper redesign.

---

## Section 3: Reassessment and New Direction (2026-04-01)

### Why Chunk-5's Pixel-Level Finite Differences Are Fundamentally Wrong For Sonar

The camera pipeline's finite-difference normals work because the depth map is a dense, regular grid where neighboring pixels correspond to neighboring points on a smooth surface. One pixel step = a small, well-defined step along the surface.

Sonar breaks this assumption. A sonar pixel encodes (azimuth, range), but the **elevation dimension is collapsed** -- the sonar integrates all returns across the elevation beam. Two pixels that are adjacent in the sonar image (e.g., one pixel apart in azimuth) might correspond to surfels that are far apart in 3D, separated in the elevation dimension that the image doesn't resolve.

Doing finite differences on neighboring sonar pixels therefore computes the normal of the **collapsed projection**, not the actual 3D surface. The cross product of `pts_right - pts_left` and `pts_down - pts_up` is computing tangent vectors across points that may not be neighbors in 3D at all. The result may be geometrically meaningless.

Chunk-5 tries to fix this by estimating elevation posteriors for each pixel, but:
- The posteriors are nearly uniform (entropy ~1.8 out of max 1.95), so the expected elevation is just the mean of all bins.
- Finite differences on 5 "mean elevation" points give the normal of the mean surface, not the actual surface.
- Even if posteriors were sharp, one-pixel spacing in polar coordinates represents very different physical distances depending on range -- the stencil has no consistent physical scale.

The finite-difference stencil was borrowed from a context (dense regular depth grid) where it makes physical sense, and applied to a context (sparse posterior-estimated elevations in polar coordinates with collapsed elevation) where it does not.

Even if we relaxed the confidence gate enough to produce nonzero signal, the underlying normals would be unreliable because of this geometric mismatch. The problem is not just the gate calibration -- it is the approach.

### Deferred Alternative Note: Normal Consistency From Surfel Neighbors

This section is retained as a research note, not as the current implementation plan.

Current status:

- keep this idea on pause,
- do not implement it in the next Chunk-5.5 pass,
- only revisit it if the existing Chunk-5 path still fails after the minimal calibration-oriented execution plan in Section 1.

**Important framing**: all normal supervision approaches -- whether the camera path's depth-derived loss, Chunk-5's posterior-based loss, or the surfel-neighbor approach below -- are **loss terms** that feed into the training loop's `total_loss` and refine surfels via standard gradient descent through `total_loss.backward()` and the Adam optimizer. They are not separate optimization procedures or post-processing steps.

#### Motivation

Current test runs show that surfel positions (centers) converge to reasonable locations much more successfully than surfel orientations. The positions are good; the normals are bad. This suggests using the good positions to fix the bad normals.

The idea: for each surfel, find its K nearest neighboring surfels in 3D. Fit a local tangent plane to those neighbor positions. The normal of that plane is what the surfel's quaternion should agree with. Penalize disagreement as a loss term.

This is the same philosophy as the camera normal consistency loss -- "your claimed normal should match the local geometry" -- but instead of using rendered depth pixels (which don't work in sonar), it uses actual surfel centers in 3D.

#### Why this is better than Chunk-5 for sonar

- **No posteriors, no confidence gates, no pixel bank.** Works directly from surfel positions that already exist and are already well-optimized.
- **No pixel-grid assumption.** Operates in 3D, so polar coordinate distortion and elevation collapse don't matter.
- **Dense.** Every surfel with enough neighbors gets a signal, not just sparse anchor pixels that survive 7 gates.
- **Self-consistency.** Same principle as the camera path: the surfel's orientation should be consistent with the geometry defined by its neighborhood.

#### The computation

Each training iteration (or every N iterations):

1. **Find neighbors**: for each surfel, find its K nearest surfels by 3D Euclidean distance on `gaussians.get_xyz`.
2. **Fit local plane**: take the K neighbor positions, compute the normal of the best-fit plane via PCA (the eigenvector corresponding to the smallest eigenvalue of the covariance matrix of the neighbor positions).
3. **Compare**: compute `1 - |dot(surfel_quaternion_normal, fitted_plane_normal)|` (sign-agnostic cosine distance, same formula as Chunk-5's `compute_normal_supervision_loss`).
4. **Add to total loss**: `total_loss += lambda_neighbor_normal * neighbor_normal_loss.mean()`.

Gradients flow backward through the plane fit to the neighbor positions and through the dot product to the surfel's quaternion. The Adam optimizer updates `_rotation` accordingly.

#### Design decisions

**K (neighbor count)**: Too small and the plane fit is noisy (3 points define a plane exactly but are sensitive to outliers). Too large and you smooth over real geometry (a sharp edge between two faces gets an averaged normal). K=10-20 is a reasonable starting range.

**KNN frequency**: KNN on ~1000-3000 surfels is cheap. The codebase already has `simple_knn` (used for initialization). Could run every iteration, or cache the neighbor indices and refresh every 50-100 iterations. Surfel positions don't change drastically between iterations, so caching should be fine.

**Quality weighting**: If the plane fit has high residual (neighbor points are scattered in 3D, not roughly coplanar), the fitted normal is unreliable. The loss can be weighted by plane-fit quality -- e.g., the ratio of the smallest eigenvalue to the second-smallest eigenvalue. Low ratio = well-defined plane = high confidence. This is a soft weighting, not a hard gate like Chunk-5.

**Minimum neighbor count**: If a surfel is isolated (fewer than some minimum neighbors within a reasonable radius), skip it. This should be rare -- most surfels are clustered on surfaces.

**When to enable**: Surfel positions need to be reasonable before their neighborhood geometry is meaningful. Enable after Stage-1 (scale factor converged, positions roughly correct), similar to how the camera path enables its normal loss after iteration 7000.

**Sign ambiguity**: PCA gives a plane normal but not a direction (could point either way). The sign-agnostic loss `1 - |dot|` handles this -- it doesn't care which side of the plane the surfel normal points to.

#### Differentiability

The KNN lookup is non-differentiable (it's a discrete neighbor selection), but everything downstream is:
- The covariance matrix of neighbor positions is a smooth function of the positions.
- The eigenvector (plane normal) is a smooth function of the covariance matrix (away from degenerate cases).
- The cosine loss is a smooth function of the plane normal and the surfel's quaternion normal.

So gradients flow to: (a) the target surfel's quaternion (the main signal -- rotates the surfel to match local geometry), and (b) the neighbor positions (secondary signal -- positions adjust to be more consistent with each other's orientations).

### Future Direction: Exploration Beyond Gradient Descent

Out of scope for the current work, but noted for a future tranche.

Gradient descent nudges surfel orientations by small increments each iteration. This can be insufficient when a surfel is badly oriented -- the loss landscape may be flat or have local minima, and the surfel needs a large discrete reorientation rather than a small gradient step.

Possible non-gradient mechanisms to investigate later:

- **Snap to local geometry**: periodically hard-reset a surfel's quaternion to match the plane fitted to its neighbors, keep if loss improves.
- **Perturb and evaluate**: sample random rotations near the current one, render, keep the best.
- **Normal-quality culling**: kill surfels whose normals persistently disagree with their neighbors, respawn with geometry-implied orientation.

These would complement the surfel-neighbor normal loss term, not replace it.
