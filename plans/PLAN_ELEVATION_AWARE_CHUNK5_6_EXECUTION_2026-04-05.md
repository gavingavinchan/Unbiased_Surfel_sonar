# Plan: Elevation-Aware Chunk 5.6 Execution

**Date:** 2026-04-05  
**Status:** Draft recovery tranche for sonar regularizer restoration (gpt-5.4)  
**Git Commit:** `ffe7126cf08723196e5929e6063451c807af49e1`  
**Scope:** Restore working sonar normal consistency and sonar unbiased-depth regularization for `debug_multiframe.py` and the code it depends on, with normal consistency treated as the first required milestone.

---

## Goal

Chunk 4 is currently usable enough to keep as the active geometry-support baseline. Chunk 5 and 5.5 did useful diagnosis, but they did not restore a working normal-consistency signal for sonar.

Chunk 5.6 is the recovery tranche that changes the strategy:

- stop treating the sparse posterior + pixel-neighbor finite-difference path as the primary route to sonar normal supervision,
- restore the original **renderer-connected** regularization path from commit `0d41037` as faithfully as practical in sonar form,
- make sonar normal consistency produce nonzero, applied, geometry-improving gradient first,
- then restore sonar unbiased-depth / depth-regularization on top of that repaired renderer contract.

Primary outcomes:

- sonar `render_sonar()` returns the regularization maps needed by training with real semantics rather than placeholders,
- opacity is learnable again in the active `debug_multiframe.py` sonar path rather than being fixed to `0.999`,
- the active Chunk-5 trainer-side normal path is rewritten away from the currently hard-disabled sparse-posterior geometry route,
- the active sonar training loop receives a real normal-consistency term and adds it exactly once before backprop,
- the normal term measurably improves surfel orientation quality instead of going fully silent,
- a sonar hard-depth / unbiased-depth path is restored only after the normal path is proven alive,
- Chunk-5 sparse expected-elevation normals remain diagnostic history, not the default active route.

---

## Why Chunk 5 / 5.5 Did Not Close The Problem

Current code and plan evidence support the following reading:

1. Chunk 5's sparse expected-elevation normal path is no longer the best primary path for sonar.
   - The center-confidence gate was often the first collapse.
   - The old pixel-neighbor closure gate then destroyed almost all remaining anchors.
   - Chunk 5.5 correctly concluded that sonar pixel-neighbor finite differences are not physically trustworthy once elevation has been collapsed by the beam.

2. The current sonar renderer does not expose the right regularization outputs.
   - `rend_alpha` is currently a binary occupancy mask, not accumulated opacity.
   - `rend_normal` is currently set equal to `surf_normal`, not a distinct surfel-orientation field.
   - `rend_dist` is currently zero.
   - `converge` is currently zero.
   - `surf_depth` is currently a support-weighted mean range image, not a hard or unbiased depth estimate.

3. The active Chunk-5 trainer-side normal path is also explicitly disconnected.
   - In the current `debug_multiframe.py` path, `_compute_chunk5_local_expected_geometry()` marks the geometry route as disabled and returns no expected normals.
   - `compute_chunk5_normal_for_frame()` therefore cannot be repaired just by swapping renderer outputs underneath it; the trainer-side target construction has to be rewritten.

4. The sonar training loops therefore cannot be expected to reproduce the camera baseline behavior from `0d41037`.

5. Fixed-opacity sonar training is now considered part of the blocker posture.
   - In the active `debug_multiframe.py` path, opacity is currently fixed to `0.999` by default.
   - That removes an important part of the original optimization behavior and can prevent the renderer from learning a sensible surface/transmittance profile.
   - Chunk 5.6 therefore restores learnable opacity for the active sonar path and treats opacity behavior as part of the validation contract, not as a side issue.
   - Existing code does contain dormant learnable-opacity machinery, but it has not been trusted or validated recently; Chunk 5.6 must treat it as suspect until gradient flow and optimizer updates are empirically confirmed.

Chunk 5.6 therefore treats the main failure as an **architecture disconnect** from the original regularizer path, not just a threshold-tuning problem.

---

## Source Of Truth

- Camera baseline reference commit:
  - `0d41037` (a README-only edit; used as a snapshot of the pre-sonar camera path code, not as a meaningful code change itself)
- Existing execution lineage:
  - `plans/PLAN_ELEVATION_AWARE_CHUNK4_EXECUTION_2026-02-23.md`
  - `plans/PLAN_ELEVATION_AWARE_CHUNK5_EXECUTION_2026-03-16.md`
  - `plans/PLAN_ELEVATION_AWARE_CHUNK5_5_EXECUTION_2026-03-30.md`
- Training/algorithm context:
  - `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md`
  - `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md`
  - `plans/PLAN_ELEVATION_AWARE_TRAINING_FLOW_2026-02-24.md`
- Current explanatory comparison to camera path:
  - `docs/CHUNK5_NORMALS_EXPLAINER.md`

If there is a conflict between the old Chunk-5 sparse-posterior route and the recovery strategy here, Chunk 5.6 takes precedence for future sonar normal/unbiased-depth work.

---

## Scope

Primary implementation targets:

- `gaussian_renderer/__init__.py`
- `debug_multiframe.py`

Likely helper touch points:

- `utils/elevation_chunk5_helpers.py`
- `utils/point_utils.py`
- `tests/test_elevation_chunk5_*`
- `tests/test_renderer_baseline_*`

Out of scope for Chunk 5.6:

- new densification behavior,
- reviving the removed sonar pixel-neighbor finite-difference normal path,
- changing Chunk-4 coupling/support policy except where needed for compatible diagnostics,
- new dataset generation or evaluator redesign.

---

## Core Decision

Chunk 5.6 will restore the **same regularizer structure** used by the original camera code path and adapt only the geometry-specific pieces that sonar requires.

This restoration must happen in two places together:

- the sonar renderer contract in `render_sonar()`, and
- the active trainer-side normal/depth target construction in `debug_multiframe.py`.

Restoring only one side is not sufficient because the current Chunk-5 trainer path is explicitly geometry-disabled.

What must be preserved from `0d41037`:

- a renderer-connected `rend_normal` field representing claimed surfel orientation,
- a renderer-connected `surf_*` geometry target representing macroscopic surface structure,
- learnable opacity participating in the forward model rather than being globally frozen,
- a normal loss added directly to the active training objective,
- a hard/unbiased depth path and a depth regularizer derived from per-ray compositing, not from ad hoc post-hoc logging.

What must be adapted for sonar:

- the camera rasterizer produced dense aux channels directly, while the sonar path currently uses a custom event-volume composition path,
- the camera path derived `surf_normal` from hard `surf_depth`; sonar should do the same conceptually, but with sonar backprojection geometry,
- changing sonar `surf_depth` from today's support-weighted mean range to a hard-depth estimate will also change `surf_normal` semantics, because `surf_normal` is derived from whatever `surf_depth` field the renderer emits,
- the hard-depth rule and distortion signal must be reconstructed from ray-binned sonar compositing rather than copied line-by-line from the pinhole rasterizer.
- the current `debug_multiframe.py` path passes `sonar_extrinsic=None` throughout; Chunk-5.6 v1 may target that active path first, but any claim of general sonar parity must either extend the backprojection helpers for non-`None` extrinsics or explicitly state that the restored contract is only validated for the no-extrinsic path.

Required Chunk-5.6 interpretation for sonar:

- `rend_normal`: dense rendered field from alpha/occlusion-weighted surfel quaternion normals,
- `surf_depth`: hard front-surface or median-transmittance range estimate per ray,
- `surf_normal`: target field computed from `surf_depth` through the existing sonar point/backprojection normal path, not from sparse posterior anchors,
- `rend_dist`: depth-spread regularizer tied to the actual sonar compositing events.

**`rend_normal` accumulation for sonar (resolved default for Chunk-5.6 v1, informed by opus4.6 review, 2026-04-05):**

The camera path produces `rend_normal` via the CUDA rasterizer's built-in alpha-compositing of per-surfel normals. The sonar path has no CUDA rasterizer — it uses `scatter_add_` for pixel accumulation. Chunk 5.6 v1 should default to:

- `rend_normal[pixel] = sum(transmittance_before_event_i * alpha_i * surfel_normal_i)` over composited events for that pixel,
- left unnormalized, matching the camera path,
- implemented using the same per-event compositing weights as the active ray-binned occlusion path rather than a plain `scatter_add_` over raw support weights.

This default keeps the semantics aligned with the original comparison against `surf_normal * detach(rend_alpha)`. Alternative accumulation rules should be treated as explicit deviations and justified in the closeout note.

**Implementation requirement — expose compositing state, not just final images (gpt-5.4 review, 2026-04-05):**

The current `compose_ray_binned_occlusion()` path returns `event_returns`, `ray_returns`, and `final_transmittance`, which is enough for rendered intensity but not enough by itself to faithfully reconstruct camera-style `rend_normal`, hard-depth, and distortion from the same compositing semantics.

Chunk 5.6 should therefore add an explicit sub-step to expose or retain enough per-event compositing state for downstream regularizer maps.

Minimum useful returned state for v1:

- transmittance-before-event,
- per-event alpha after support sharing,
- sorted event range values,
- sorted ray ids,
- per-ray segment boundaries,
- the event sort order or another equivalent mapping back to the caller-side event arrays.

`event_surfel_idx` does not have to be returned by `compose_ray_binned_occlusion()` itself as long as `render_sonar()` preserves access to the caller-side event-to-surfel mapping when applying the returned order.

Without this, the renderer-contract goals below are underspecified.

---

## Chunk-5.6 Contracts To Implement

### 1. Sonar renderer regularization-output contract

`render_sonar()` must stop returning placeholder regularization outputs.

Required outputs:

- `rend_alpha`: true accumulated opacity / support mass after ray-binned composition,
- `rend_normal`: surfel-orientation normal field accumulated from `quaternion_to_normal(rotations)`, distinct from `surf_normal`,
- `surf_depth`: hard or unbiased surface range estimate, not support-weighted mean range,
- `surf_normal`: surface-geometry target normal field derived from a physically meaningful 3D construction,
- `rend_dist`: nonzero depth-regularization signal derived from the same per-ray events,
- internal or returned compositing state sufficient to derive the above maps from the actual ray-binned event ordering,
- `converge`: not a Chunk-5.6 blocker for `debug_multiframe.py`; restore it only if `train.py` is also brought back into scope, otherwise document it as deferred.

Compatibility rule:

- no sonar training loop may claim regularizer support while these outputs remain placeholders.

### 1.5. Learnable-opacity contract

Chunk 5.6 restores learnable opacity in the active `debug_multiframe.py` sonar path.

Important implementation note:

- current code already contains a dormant opacity-policy path that can switch between fixed and learnable opacity,
- but this path has not been validated recently and may be stale or partially broken,
- therefore Chunk 5.6 must treat learnable opacity as an implementation to verify, not a capability to assume.

Required behavior:

- `SONAR_FIXED_OPACITY` must not remain the effective default for Chunk-5.6 validation runs,
- opacity parameters must receive gradients and optimizer updates in the active sonar training path,
- new regularizer behavior must be evaluated with learnable opacity enabled,
- no Chunk-5.6 success claim is valid if the evaluated run still relied on globally fixed opacity unless that run is explicitly labeled as an ablation.

Synthetic-surface opacity check:

- on datasets with known analytic surfaces, especially the synthetic cube dataset, Chunk-5.6 validation must check whether surfels that lie near the true surface become highly opaque,
- for the cube case, surfels near the true cube surface should trend toward near-opaque values rather than remaining diffuse/transparent,
- this check is part of the renderer/training validation because hard-depth and regularization behavior depend on learned opacity being meaningful.

### 2. Sonar normal-consistency target contract

Chunk 5.6 restores normal consistency first.

The active sonar normal target must follow the original camera contract as closely as possible:

- it is connected to the renderer/training path every iteration,
- it does not depend on sparse bright-pixel bank survival,
- it is derived from `surf_depth` rather than from Stage-1 posterior anchors,
- it is sign-agnostic,
- it is dense enough that the loss cannot trivially starve to zero.

Required v1 target construction:

1. Restore a hard `surf_depth` range map from the same occlusion-composited sonar events used to render the frame.
2. Compute `surf_normal` from that `surf_depth` using the sonar geometry path already present in `utils/point_utils.py`.
3. Restore a distinct `rend_normal` map from accumulated surfel quaternion normals.
4. Compare the two using the same sign-agnostic cosine law used by the original training loop:

```python
normal_error = 1.0 - (rend_normal * surf_normal).sum(dim=0)
normal_loss = w_normal * normal_error.mean()
```

Important parity notes from `0d41037`:

- `rend_normal` is an accumulated surfel-normal field and is not alpha-normalized,
- `surf_normal` should be multiplied by detached `rend_alpha` before comparison, matching the original path,
- the first Chunk-5.6 implementation should reuse this structure rather than introducing a new neighborhood-fit loss.

Important semantic note:

- the code path that computes `surf_normal` from `surf_depth` can stay structurally the same,
- but swapping `surf_depth` from support-weighted mean range to hard-depth is still a real semantic change because the resulting `surf_normal` field will change with it,
- this is expected and is part of the restoration, not an incidental side effect.

**Known limitation — `surf_normal` from collapsed-elevation range image (opus4.6 review, 2026-04-05):**

The `surf_normal` target is computed by applying `sonar_ranges_to_points()` then `sonar_points_to_normals()` (central finite differences) to the rendered `surf_depth` range image. This uses the same finite-difference-on-sonar-image approach that Chunk 5.5 flagged as geometrically questionable — but with an important distinction:

- **Chunk 5 (broken):** finite differences on sparse posterior-derived expected-elevation points, gated by confidence, with unreliable elevation estimates. Sparse, gate-dependent, and the underlying geometry was wrong.
- **Chunk 5.6 (proposed):** finite differences on the dense rendered range image. No elevation estimation, no gating. Always available, dense signal.

The collapsed-elevation concern still applies: sonar integrates over the elevation beam, so the range at each pixel is the weighted range to whatever surfaces fall in that beam across all elevations. If multiple surfaces at different elevations project to the same (azimuth, range) pixel, the range doesn't correspond to any real surface, and the finite-difference normal describes a phantom surface.

**Why this is acceptable for Chunk 5.6 v1:**

- For locally smooth scenes with one dominant surface per beam (synthetic cube, simple shapes), the range image normal correctly identifies face/surface orientations. The regularizer will push surfels to align with the apparent surface, which is the right thing.
- The camera path uses the same conceptual approach (finite differences on rendered depth) and it works. The sonar version is noisier due to elevation collapse, but a noisy-but-present signal is strictly better than the current state of zero signal.
- The purpose of `surf_normal` is to serve as a regularization target (pushing surfel orientations toward geometric consistency), not as ground-truth surface normals. An approximate target that covers most pixels every iteration is more useful than a precise target that covers zero pixels.

**Where this may need revisiting:**

- Cluttered scenes with overlapping objects at different elevations within the same beam.
- Geometry near elevation beam edges where the collapse is most distorting.
- If the regularizer drives surfel orientations to converge on phantom-surface normals rather than real-surface normals, the mesh quality could degrade in those regions.

This limitation should be accepted for v1 and revisited only if empirical results show it causing problems on target scenes.

Not acceptable for the primary Chunk-5.6 path:

- introducing a new KNN plane-fit regularizer or other novel normal target before the original-style path is restored,
- resurrecting the old `left/right/up/down` sonar-pixel finite-difference stencil,
- depending on sparse Stage-1 posterior anchors as the only source of normal supervision.

Mandatory implementation note:

- the current `compute_chunk5_normal_for_frame()` sparse-posterior anchor path is not the place to simply inject a new tensor from `render_sonar()`,
- the trainer-side normal construction itself must be rewritten so that the active loss path consumes renderer-connected `surf_depth`, `surf_normal`, `rend_normal`, and `rend_alpha` directly rather than waiting on `_compute_chunk5_local_expected_geometry()`.

### 3. Normal-mode rollout contract

Keep the familiar mode discipline:

- `ELEV_NORMAL_MODE=off|shadow|active`

Mode semantics for Chunk 5.6:

- `off`: do not compute or apply the Chunk-5.6 normal term,
- `shadow`: compute and log normal diagnostics but do not add the term to the objective,
- `active`: add `w_normal * loss_normal` exactly once in the unified loss block.

The weight schedule may reuse the existing Chunk-5 ramp machinery, but the underlying signal source changes back to renderer self-consistency rather than sparse posterior-derived normals.

### 4. Unified-loss integration contract

For the active sonar trainer, the normal term must be added exactly once before `backward()`.

Current-code note:

- `debug_multiframe.py` already contains normal-term integration plumbing in both the Stage-2 and Stage-3 loops,
- Chunk 5.6 should therefore prefer verifying and adapting that existing wiring to the new renderer-connected signal source rather than rewriting the loss accumulation structure from scratch.

Mandatory rules:

- no duplicate normal-term accumulation,
- no silent shadow-mode behavior when `active` is requested,
- zero-valid-sample steps contribute a finite zero term and log why,
- the normal loss must be reported separately from photometric, Chunk-4 coupling, and any depth term.

Primary target:

- `debug_multiframe.py`
- both the Stage-2 and Stage-3 training loops must remain aligned; any loop-local wiring change made for one must be mirrored in the other.

### 5. Sonar unbiased-depth contract

Chunk 5.6 restores unbiased depth only after the normal path is proven alive.

The sonar hard-depth output must be derived from the compositing process itself, not from the support-weighted mean range image.

Original-camera parity requirement:

- the first restored sonar hard-depth path should mirror the camera code's use of a median-like pseudo-surface depth rather than switching to expected depth as the normal-consistency surface.

Allowed v1 definitions:

- per-ray median-transmittance depth analogous to the camera baseline,
- first-surface / front-surface depth chosen from the sorted sonar events with an explicit transmittance or accumulated-opacity rule,
- another per-ray hard-depth rule that is explicitly documented and tested.

Minimum requirements:

- deterministic under fixed event ordering,
- tied to the active ray-binned occlusion semantics,
- evaluated with learnable opacity enabled,
- available to the training loop as a first-class renderer output,
- no placeholder zero tensors.

### 6. Sonar depth-regularization contract

After hard-depth is available, restore one depth regularizer with real semantics.

Original-camera parity requirement:

- restore the `rend_dist` / distortion-style path before considering any new depth term.

Preferred order:

1. restore a depth-spread / distortion-style penalty from the per-ray event distribution,
2. only keep or revisit `converge` after the original-style hard-depth + distortion path is alive.

This tranche should not keep both a dead `rend_dist` path and a dead `converge` path merely for naming continuity.

For Chunk-5.6 scope, `converge` is not the primary restoration target. The primary target is the original-style hard-depth plus distortion contract in `debug_multiframe.py`.

If `train.py` remains out of scope, `converge` may remain deferred as long as the closeout note states that explicitly.

### 7. Chunk-5 sparse posterior path preservation contract

The old sparse expected-elevation Chunk-5 path is preserved as existing code and diagnostics, but it is no longer the primary active route for fixing normal orientations.

Rules:

- keep Chunk-4 and earlier behavior intact,
- do not rip out Chunk-5 sparse-posterior code unless a small compatibility edit is needed,
- do not use the sparse posterior path as the main proof that sonar normal consistency is restored,
- keep its diagnostics only if they still provide useful observability,
- do not block Chunk-5.6 closeout on reviving nonzero signal through the old pixel-neighbor route.

### 8. Diagnostics contract

Normal diagnostics must include:

- `normal_mode`,
- scheduled `w_normal`,
- count/fraction of pixels or surfels receiving valid normal targets,
- count/fraction of pixels or surfels with valid rendered normals,
- applied normal-loss scalar,
- coverage of valid `surf_depth`-derived normals.

Depth diagnostics must include:

- nonzero coverage of hard-depth pixels,
- summary statistics for `surf_depth`,
- summary statistics for `rend_alpha`,
- nonzero depth-regularizer scalar when active,
- skip reasons if depth regularization is disabled for a step.

Opacity diagnostics must include:

- whether opacity was fixed or learnable for the run,
- whether opacity gradients were enabled,
- summary statistics for active surfel opacity values,
- on analytic synthetic datasets, summary statistics for opacity restricted to surfels near the true surface,
- for the cube dataset specifically, an explicit check that near-surface surfels trend toward high opacity.

---

## Step-By-Step Work Order

1. Freeze the current diagnosis in tests.
   - Add failing tests that prove the current sonar path is returning placeholder regularizer outputs.

2. Reconstruct the camera baseline contract from `0d41037` in testable form.
   - Document and test the expected map semantics: `rend_alpha`, `rend_normal`, `surf_depth`, `surf_normal`, `rend_dist`.
   - Validate the existing dormant learnable-opacity warmup/policy path early, before normal-path closeout.
   - If that path fails, repair it here rather than postponing opacity work until after normal-path validation.

3. Repair `render_sonar()` outputs first.
   - Replace binary `rend_alpha` with real accumulated opacity/support semantics.
    - Replace `rend_normal = surf_normal` with a true surfel-normal render.
    - Introduce a hard-depth output derived from compositing, not weighted mean range.
    - Expose enough compositing state to derive hard-depth and distortion from the actual event ordering.
    - Introduce a nonzero depth-regularization signal or explicitly defer it with failing tests still in place.
   - Use learnable opacity while validating these semantics so `rend_alpha` behavior matches the intended restored contract.

4. Rewrite the active Chunk-5 normal path in `debug_multiframe.py`.
   - Remove the dependency on `_compute_chunk5_local_expected_geometry()` as the source of the active normal target.
   - Restore `surf_depth -> surf_normal` semantics for sonar, with the explicit understanding that moving from weighted-mean range to hard-depth changes the target field semantics.
   - Restore `rend_normal` as a distinct accumulated surfel-normal map.
   - Reuse the original comparison structure rather than adding a new target type.

5. Verify the existing normal-term integration wiring in `debug_multiframe.py` with the new signal source.
   - Off/shadow/active behavior must be covered by tests.
   - Confirm the Stage-2 and Stage-3 loops both apply the restored term exactly once.
   - Add this as a new feature on top of Chunk 4, not as a rewrite of Chunk 4 or a removal of existing loss terms.

6. Finish learnable-opacity validation in the active sonar training path.
   - Keep opacity trainable for Chunk-5.6 runs.
   - Verify gradients and optimizer updates are actually flowing to opacity.
   - Treat any failure here as a real implementation bug in the dormant opacity-policy path, not as a simple config issue.
   - Keep fixed-opacity only as an explicit ablation/debug option.

7. Run short controlled synthetic smokes in `shadow`, then `active`.
   - Require nonzero `surf_depth` coverage, nonzero valid `surf_normal` coverage, nonzero applied normal loss, and meaningful learned opacity variation.

8. Measure geometry/orientation improvement before touching unbiased depth.
   - If normal coverage is nonzero but geometry regresses, fix that before depth work.
   - On analytic synthetic datasets, also measure opacity behavior near the true surface.

9. Implement sonar hard-depth / unbiased-depth output in original-style form.
   - Add renderer tests first, then training-loop wiring.

10. Restore the active sonar distortion/depth regularizer.
   - Prefer `rend_dist`-style recovery before revisiting `converge`.

11. Publish a closeout note with clear evidence that Chunk 5.6 restored working sonar regularizers in `debug_multiframe.py` without disturbing Chunk-4 behavior, that opacity was learnable during the validated runs, and that the old Chunk-5 sparse path is no longer the primary dependency.

---

## Quantitative Gate Thresholds

These are Chunk-5.6 default gates. Any override must be documented in the closeout note.

- `GATE_FINITE_NAN_INF = 0`
- `GATE_NORMAL_VALID_COVERAGE_MIN = 0.05` on at least one representative active sonar run
- `GATE_NORMAL_APPLIED_STEPS_MIN = 0.50` fraction of measured Stage-2 steps on that run
- `GATE_NORMAL_LOSS_NONZERO = true` on active runs
- `GATE_ORIENTATION_IMPROVEMENT_REQUIRED = true`
- `GATE_OPACITY_LEARNABLE_REQUIRED = true` on all non-ablation Chunk-5.6 validation runs
- `GATE_NEAR_SURFACE_OPACITY_HIGH_REQUIRED = true` on analytic synthetic validation scenes
- `GATE_SURF_DEPTH_NONZERO_COVERAGE_MIN = 0.05` before any depth term is claimed active
- `GATE_DEPTH_TERM_NONZERO = true` before unbiased-depth restoration is claimed complete
- `GATE_OFFMODE_PARITY_REQUIRED = true` for any mode-gated rollout

Recommended orientation evidence for closeout:

- improvement in `|cos(n_surfel, n_reference)|` on synthetic scenes with known normals,
- or another explicit orientation-quality metric if the scene lacks analytic normals.

Recommended opacity evidence for closeout:

- on analytic synthetic scenes, report opacity statistics for surfels near the true surface,
- on the cube dataset, show that near-surface surfels are predominantly near-opaque by the end of training,
- if this does not happen, treat that as a training/renderer blocker rather than a cosmetic issue.

---

## Suggested Validation Matrix

1. Unit tests
   - renderer-output semantics,
   - original-style `surf_depth -> surf_normal` construction,
   - mode gating,
   - hard-depth rule,
   - depth regularizer non-placeholder behavior,
   - learnable-opacity gradient/update behavior,
   - active debug path consuming ray-binned renderer outputs rather than the old geometry-disabled sparse path.

2. Short synthetic smokes
   - clean cube,
   - one smoother-shape comparator,
   - run them with `SONAR_OCCLUSION_MODE=ray_binned` so the restored path is actually exercised.

3. Active-run comparisons
   - Chunk-4 baseline,
   - Chunk-5.5 Phase-A no-neighbor baseline,
   - Chunk-5.6 normal-only,
   - Chunk-5.6 normal + depth.

4. Artifact review
   - run log,
   - normal/depth diagnostics,
   - surfel orientation metrics,
   - geometry metrics.

---

## Non-Goals And Guardrails

- Do not spend more time trying to rescue the removed sonar pixel-neighbor finite-difference normal path.
- Do not introduce a brand-new normal regularizer before the original-style renderer path is restored.
- Do not claim depth regularization is restored while `rend_dist` or `converge` are still placeholders.
- Do not hide missing renderer semantics behind training-loop zeros.
- Do not treat densification as a substitute for working normal or depth regularization.
- Do not redo Chunk 4. Chunk 5.6 adds new working regularizer features on top of the current Chunk-4-capable pipeline.
- Do not validate Chunk 5.6 on fixed-opacity runs except as explicit ablations.
- Do not call Chunk 5.6 complete if only diagnostics improved but the applied objective stayed silent.

---

## Open Questions

1. For the sonar hard-depth adaptation, should the first implementation use the closest possible analogue of the camera median-like cumulative-opacity rule, even if the exact threshold needs sonar-specific tuning?
2. Once the original-style normal + distortion path is restored, do we want Chunk-5 sparse expected-elevation diagnostics kept in the logs by default, or only behind an explicit debug flag?

---

## 2026-04-06 Status Update / Handoff

### Implementation status

Chunk-5.6 is now partially implemented in the active code path.

Completed in code:

- `compose_ray_binned_occlusion()` now exposes the compositing state required by the new renderer contract:
  - `transmittance_before_event`
  - `alpha_sorted`
  - `range_sorted`
  - `ray_ids_sorted`
  - `segment_starts`
  - `event_sort_order`
- `render_sonar()` no longer returns the old placeholder regularizer outputs.
- `rend_alpha` is now derived from composited support over the splatted row footprint, not a binary occupancy threshold.
- `rend_normal` is now a distinct accumulated surfel-orientation field derived from composited quaternion normals over the splatted row footprint.
- `surf_depth` is now derived from the renderer-connected compositing path rather than the previous placeholder weighted-range image contract.
- `rend_dist` is now nonzero and derived from per-event depth spread against the hard-depth selection.
- `compute_chunk5_normal_for_frame()` was rewritten away from `_compute_chunk5_local_expected_geometry()` for the active loss path.
- The active Chunk-5 normal path now consumes renderer outputs directly:
  - `render_pkg["rend_alpha"]`
  - `render_pkg["rend_normal"]`
  - `render_pkg["surf_depth"]`
  - `render_pkg["surf_normal"]`
  - `render_pkg["rend_dist"]`
- `render_pkg["rend_alpha"]` is detached before normal comparison.
- Stage-2 and Stage-3 unified losses now include a renderer-connected depth term as well as the normal term.
- Chunk-5 diagnostics CSV/header/summary were extended with renderer-connected normal/depth/opacity fields.
- `SONAR_FIXED_OPACITY` default in `debug_multiframe.py` was changed to `False`.

### Important runtime fix discovered during validation

The first implementation passed the new TDD/AST contracts but still produced zero active normal coverage at runtime because the initial renderer-side `rend_alpha` / `rend_normal` / `surf_depth` maps were effectively only populated at hard-hit pixels. That made the finite-difference `surf_normal` path too sparse, so `surf_normal_valid_frac` stayed zero.

This was repaired by changing the renderer regularizer maps so that:

- hard-depth selection still comes from the ray-binned compositing rule,
- but the regularizer maps used for `rend_alpha`, `rend_normal`, `surf_depth`, and `rend_dist` are splatted back over the sonar row footprint using the same support-profile machinery as the intensity render.

That change is what made the Chunk-5.6 normal path become nonzero at runtime.

### Tests completed

Focused TDD tests now pass:

```bash
pytest -q tests/test_elevation_chunk5_6_tdd_contracts.py
```

Result at time of handoff:

- `14 passed, 3 skipped`

Additional adjacent checks run successfully:

```bash
pytest -q tests/test_renderer_baseline_r0_red_occlusion_semantics_contracts.py \
          tests/test_renderer_baseline_smoke_contracts.py \
          tests/test_elevation_chunk5_normals_contracts.py
python -m py_compile gaussian_renderer/__init__.py debug_multiframe.py
```

### Synthetic smoke status

Short synthetic smokes were run with learnable opacity enabled and `SONAR_OCCLUSION_MODE=ray_binned`.

Primary validation outputs:

- `output/chunk5_6_smoke_active_sphere_v2`
- `output/chunk5_6_smoke_shadow_sphere_v2`
- `output/chunk5_6_smoke_active_cube_v2`

#### Active sphere (`output/chunk5_6_smoke_active_sphere_v2`)

Key Chunk-5 diagnostics from `chunk5_gate_log.csv`:

- `surf_depth_valid_frac`: about `0.165-0.166`
- `surf_normal_valid_frac`: about `0.088-0.093`
- `finite_count`: about `3395-3554`
- `loss_mean`: about `0.79-0.88`
- `applied_loss_mean`: nonzero, about `0.0088-0.0877`
- `opacity_fixed=0`, `opacity_grad_enabled=1`

Final train eval from run log:

- `loss_mean=0.011751`
- `ssim_mean=0.9339`

Synthetic sphere evaluator result:

- GT mean radial error: `0.023242 m`
- GT p95 radial error: `0.059307 m`
- pass: `True`

#### Shadow sphere (`output/chunk5_6_smoke_shadow_sphere_v2`)

Key Chunk-5 diagnostics from `chunk5_gate_log.csv`:

- signal exists with similar coverage to active mode,
- but `applied_count = 0`
- and `applied_loss_mean = 0.000000`

This is the desired runtime gating behavior for `shadow`.

Final train eval from run log:

- `loss_mean=0.011091`
- `ssim_mean=0.9370`

Synthetic sphere evaluator result:

- GT mean radial error: `0.022869 m`
- GT p95 radial error: `0.057320 m`
- pass: `True`

#### Active cube (`output/chunk5_6_smoke_active_cube_v2`)

Key Chunk-5 diagnostics from `chunk5_gate_log.csv`:

- `surf_depth_valid_frac`: about `0.230-0.233`
- `surf_normal_valid_frac`: about `0.126-0.130`
- `finite_count`: about `4861-5014`
- `loss_mean`: about `0.766-0.785`
- `applied_loss_mean`: nonzero, about `0.0078-0.0782`
- `opacity_fixed=0`, `opacity_grad_enabled=1`

Final train eval from run log:

- `loss_mean=0.023184`
- `ssim_mean=0.8771`

Synthetic cube evaluator result:

- GT mean surface error: `0.025381 m`
- GT p95 surface error: `0.067874 m`
- pass: `True`

### Opacity status

Learnable opacity is active in the validation runs.

Observed runtime evidence:

- logs show `[Stage 0] Opacity mode: LEARNABLE`
- logs show `[Opacity] Switched to LEARNABLE at iter 1`
- final Poisson opacity filtering thresholds are nontrivial, for example:
  - active sphere v2: `opacity >= 0.0785` kept `787/984`
  - active cube v2: `opacity >= 0.0783` kept `1089/1362`

This is enough to say opacity is no longer globally fixed in the active validation path.

What is **not** yet closed out:

- the plan's explicit analytic near-surface opacity check has not been implemented yet,
- there is not yet a report restricted to surfels near the true sphere/cube surface.

### Current interpretation against Chunk-5.6 gates

What is now satisfied well enough for the next tranche:

- renderer contract is no longer placeholder-only,
- active normal signal is nonzero at runtime,
- `shadow` vs `active` mode behavior is visible at runtime,
- learnable opacity is enabled in the validated runs,
- short synthetic runs complete and basic geometry remains acceptable.

What remains incomplete:

- explicit near-surface opacity validation on analytic synthetic datasets,
- a stronger orientation-improvement claim relative to an off/shadow baseline,
- dedicated closeout evidence for the restored depth regularizer beyond "nonzero and wired in",
- any final decision about whether the current `surf_depth` naming should remain as-is even though its semantics are now splatted hard-depth support rather than the original placeholder weighted-range image.

### Suggested next work for the next session

1. Add the synthetic near-surface opacity diagnostic required by this plan.
2. Compare active vs shadow/off runs using an explicit orientation-quality metric, not only coverage/loss activation.
3. Decide whether the current restored depth term is sufficient as the retained Chunk-5.6 `rend_dist` behavior or whether it needs a closer camera-parity distortion formulation.
4. If preparing for commit, update the progress docs required by repo policy before committing:
     - `plans/progress_overview.md`
     - `plans/scientific_progress.md`

## Post-Closeout Visualization Follow-Up

This sits after the main Chunk-5.6 closeout work and is intended as a lightweight inspection pass, not as a blocker for the closeout itself.

### First visualization target

The first macro-surface visualization target should be the renderer outputs:

- `surf_depth`
- `surf_normal`

Reasoning:

- these are the macro-surface outputs used on the target side of the restored normal-consistency path,
- they already exist in the renderer contract,
- they are simpler and more direct than jumping immediately to a 3D Blender export,
- they should already reveal whether the implied surface on the synthetic cube is face-aligned, smeared, rounded, or phantom.

### Initial export scope

For the first pass, generate only 2D images and generate them only at the end of training.

Suggested artifact types:

- `surf_depth` image per training frame,
- `surf_normal` image per training frame.

Naming should match the existing `comparison_*_<frame>.png` style as closely as practical while still making it clear these are macro-surface artifacts.

### Diagnostic expansion note

If the final-only images are not sufficient to explain failures, extend this to generate more of those images throughout training at saved checkpoints or stage boundaries.

That deeper rollout is explicitly a follow-up diagnostic option, not the default first implementation.

### Future option

If the 2D images indicate that deeper geometric inspection is needed, the next visualization step should be a Blender-friendly 3D macro-surface export:

- backproject `surf_depth` into world-space macro-surface geometry,
- carry `surf_normal` along with it,
- compare that geometry against the synthetic cube ground truth in the same world frame.

## 2026-04-06 Closeout Update

### Additional implementation completed

- Added analytic synthetic-surface diagnostics directly to `debug_multiframe.py`:
  - loads `manifest.json` geometry for synthetic sphere/cube datasets,
  - computes near-surface residual statistics,
  - computes sign-agnostic surfel-vs-ground-truth orientation alignment,
  - computes near-surface opacity statistics and high-opacity fraction,
  - writes `synthetic_surface_diagnostics.json` into the run output,
  - injects per-stage synthetic diagnostics into the visualizer manifest/state artifacts.
- Added explicit opacity-policy labeling to those diagnostics:
  - `opacity_policy = fixed|learnable`,
  - `opacity_fixed`,
  - `opacity_grad_enabled`.
- Hardened `debug_multiframe.py` so baseline comparison runs do not crash when they degenerate:
  - Stage-2 / Stage-3 now skip `backward()` with a warning when a step's loss has no gradient graph,
  - Poisson export now skips cleanly if opacity filtering removes all points.

### New validation outputs

Primary closeout comparison runs:

- active cube, opacity warmup then learnable:
  - `output/chunk5_6_closeout_probe_cube_active_warm`
- shadow cube, same config:
  - `output/chunk5_6_closeout_probe_cube_shadow_warm`
- off cube, same config:
  - `output/chunk5_6_closeout_probe_cube_off_warm`

Exploratory ablation showing why zero warmup is not the retained validation posture:

- active sphere with `SONAR_OPACITY_WARMUP_ITERS=0`:
  - `output/chunk5_6_closeout_probe_sphere_active`

### Closeout interpretation

#### 1. Learnable-opacity gate is now explicitly validated

The retained closeout configuration uses:

- `SONAR_OPACITY_WARMUP_ITERS=200`
- then switches to learnable opacity at iteration `201`

Runtime evidence from the retained active cube run:

- run log shows `[Opacity] Switched to LEARNABLE at iter 201 (stage2/iter201)`
- `synthetic_surface_diagnostics.json` reports for active cube Stage 3:
  - `opacity_policy = learnable`
  - `opacity_grad_enabled = 1.0`
  - `near_surface.count = 1681`
  - `near_surface.opacity.mean = 0.9987`
  - `near_surface.high_opacity_frac = 1.0000`

This satisfies the plan's required analytic near-surface opacity check for the cube case under a run that is learnable during the measured phase.

#### 2. Active normal mode now wins cleanly against shadow/off on the retained cube comparator

Active cube Stage 3 (`output/chunk5_6_closeout_probe_cube_active_warm/synthetic_surface_diagnostics.json`):

- `valid_surfel_count = 1914`
- `near_surface.count = 1681`
- `near_surface.orientation_abs_cos.mean = 0.5089`
- `near_surface.good_orientation_frac = 0.0952`

Shadow cube Stage 3 (`output/chunk5_6_closeout_probe_cube_shadow_warm/synthetic_surface_diagnostics.json`):

- `valid_surfel_count = 0`
- `near_surface.count = 0`
- orientation metric collapses to zero because the learned-opacity baseline loses all valid synthetic surfels.

Off cube Stage 3 (`output/chunk5_6_closeout_probe_cube_off_warm/synthetic_surface_diagnostics.json`):

- `valid_surfel_count = 0`
- `near_surface.count = 0`
- same collapse posture as `shadow`.

Interpretation:

- the restored active Chunk-5.6 path preserves a non-collapsed, high-opacity, evaluable surface state,
- the matched `shadow` / `off` baselines do not,
- therefore the restored active normal path now provides measurable orientation-quality benefit relative to the no-applied-loss baselines on the retained cube comparator.

This is not the same as claiming orientation improved over the raw initialization state; it is specifically a successful active-vs-baseline closeout on the plan's required comparison axis.

#### 3. Depth regularizer is retained as the current Chunk-5.6 `rend_dist` behavior

The retained active cube run shows nonzero renderer-connected depth regularization throughout the measured phase.

Examples from `output/chunk5_6_closeout_probe_cube_active_warm/chunk5_gate_log.csv`:

- Stage 2, iter `220`: `rend_dist_mean = 0.076989`
- Stage 3, iter `221`: `rend_dist_mean = 0.082195`
- Stage 3, iter `240`: `rend_dist_mean = 0.084212`

Current retained interpretation:

- `hard_depth` is selected per ray from the ray-binned cumulative-alpha rule,
- `rend_dist` is the compositing-weighted mean absolute event spread around that hard depth,
- it is non-placeholder and active in the trainer objective,
- no further depth-term rewrite is required to close Chunk 5.6 in the working tree.

#### 4. Zero-warmup learnable-opacity is kept as an ablation, not the closeout posture

The exploratory sphere run with `SONAR_OPACITY_WARMUP_ITERS=0` (`output/chunk5_6_closeout_probe_sphere_active`) showed:

- learnable opacity from the first step,
- but near-surface opacity remained low (`near_surface.opacity.mean ~= 0.0420` at Stage 3),
- and orientation quality degraded (`near_surface.orientation_abs_cos.mean ~= 0.6590`).

So zero warmup is useful as a stress ablation, but it is not the retained validation configuration for Chunk 5.6 closeout.

### Final status for this plan

Chunk-5.6 is now considered closed in the working tree under the retained cube comparator configuration because all of the following are true:

- renderer-connected normal/depth outputs are live,
- active normal loss is applied,
- learnable-opacity validation is explicit and measured,
- near-surface opacity is high on the analytic cube run during the learnable phase,
- active mode preserves evaluable synthetic surface geometry while matched shadow/off baselines collapse,
- the current `rend_dist` path is nonzero and retained as the Chunk-5.6 depth regularizer,
- the active comparison path no longer depends on the old sparse posterior normal route.
