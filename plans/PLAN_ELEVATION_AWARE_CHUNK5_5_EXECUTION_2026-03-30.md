# Plan: Elevation-Aware Chunk 5.5 Execution

**Date:** 2026-03-30  
**Status:** Draft diagnostic follow-on to Chunk 5 (gpt-5.4)  
**Scope:** Chunk-5 zero-signal investigation, gate-by-gate rerun instrumentation, and execution guidance before any further late-normal claims

---

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

## Migrated Investigation Addendum (from Chunk-5 plan)

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
