# Plan: Elevation-Aware Training with Back Projection in Loop

**Date/Time:** 2026-01-28
**Git Commit:** 88b210c
**Status:** Design exploration (not yet implemented)

---

## Problem Statement

### Current Limitation

Back projection assumes `elevation = 0`:
```
pixel (col, row) → single 3D point on the horizontal plane
```

But sonar cannot distinguish elevation. A pixel actually corresponds to an **elevation arc**:
```
pixel (col, row) → arc of points with same (azimuth, range), varying elevation ±10°
```

### The Elevation Arc

```
Side view (looking along azimuth direction):

         elevation = +10°
              ↗ ·
            ·
          ·        ← All these points produce
        ·             the SAME pixel (col, row)
      · ← elevation = 0 (current assumption)
        ·
          ·
            ·
              ↘ ·
         elevation = -10°

         ↑
       sonar origin
```

Sign convention note: the sketch is geometric intuition only. For implementation, elevation uses the camera/view frame convention defined later (`+X right, +Y down, +Z forward`), so `+elevation` points toward `+Y` (down in image coordinates) and `-elevation` points toward `-Y`.

### Projection Asymmetry

**Forward projection is many-to-one:**
- Multiple surfels along an elevation arc all map to the same pixel
- The rendered intensity is the sum/blend of all contributing surfels

**Backward projection is one-to-many:**
- One pixel could correspond to a single surfel OR multiple surfels along the arc
- We need to figure out during training: how many surfels, and where on the arc?

### Multi-View Opportunity

With orbiting camera frames around a central structure:
- Significant FOV overlap between frames
- The same physical surface is seen from different angles
- **Different viewpoints constrain elevation differently**
- Multi-view consistency can resolve the elevation ambiguity

```
Frame A (pose 1)                    Frame B (pose 2)
      │                                   │
      │ back-project                      │ back-project
      ↓                                   ↓
   Arc A                               Arc B
      │                                   │
      └──────────► Intersection ◄─────────┘
                        │
                        ↓
                 True 3D point
              (with correct elevation)
```

---

## Design Constraints

- **When to resolve elevation**: During training only (not initialization)
- **Hardware**: Laptop with 8GB VRAM, 500 frames max
- **Priority**: Make it work first, optimize later

---

## Authorship

This plan was developed collaboratively by **opus4.5** and **gpt-5.2-codex**, with user input. Sections are attributed inline where relevant.

---

## Addendum: Elevation-Aware Alternatives (gpt-5.2-codex)

### 1) Elevation distribution per pixel (bin or latent)

- Treat each sonar pixel as an elevation distribution; approximate the integral with a small set of elevation bins.
- Aggregate bin renders to match observed intensity (sum or softmax-weighted sum) so multi-view consistency sharpens the correct elevation.
- Use entropy annealing: start broad (soft) and gradually sharpen bin weights to resolve ambiguity.
- Add light priors to avoid degenerate solutions (smoothness across neighboring surfels, discourage extreme elevations unless supported by multi-view overlap).

### 2) Stochastic elevation sampling (Monte Carlo)

- Replace fixed bins with a small number of random elevation samples per pixel each iteration.
- Over training, sampled expectations approximate the full elevation integral with less memory overhead.
- Stabilize with an annealed sampling schedule (start wide/high variance, reduce variance and sample count as training converges).
- Optionally keep a deterministic anchor sample at elevation=0 early to reduce drift.

### 3) Two-stage elevation solver

- Phase A (geometry-first): estimate elevation per surfel (or per region) using multi-view arc consistency, with minimal photometric reliance.
- Phase B (photometric): freeze or lightly regularize elevation and run standard training to refine appearance and geometry.
- Benefit: decouples elevation resolution from appearance; risk: needs decent initialization or it can lock in errors.

---

## Option 1: Point-to-Point Loss with Learnable Elevation

### Core Idea

Back-project GT images to get "target" 3D points, but instead of assuming elevation=0, **learn the elevation** for each pixel.

### Sub-Option 1A: Per-Pixel Learnable Elevation

Each valid pixel in each frame gets a learnable elevation parameter.

**Data structure:**
```
elevation_params[frame_idx, row, col] = learnable scalar in [-half_elev_fov, +half_elev_fov]
```

**Back-projection with learned elevation:**
```
pixel (col, row) + elevation_params[frame, row, col]
    → 3D point P(frame, row, col)
```

**Multi-view consistency loss:**

If pixel in frame A and pixel in frame B see the same physical surface:
```
P_A = back_project(frame_A, pixel_A, elevation_A)
P_B = back_project(frame_B, pixel_B, elevation_B)

Loss: ||P_A - P_B||²
```

**Challenge**: How to establish pixel correspondences across frames?
- Use surfels as anchors: if surfel S renders to pixel_A in frame A and pixel_B in frame B, those pixels correspond
- Forward-project surfels to find which pixels they hit in each frame

**Gradient flow:**
```
Multi-view consistency loss
    → gradient on P_A, P_B
    → gradient on elevation_A, elevation_B
    → elevation parameters update to minimize distance
```

### Sub-Option 1B: Surfel-Centric Elevation Constraint

Surfels have true 3D positions (including elevation). Constrain them to lie on the elevation arcs of the pixels they render to.

**For each surfel S visible in frame A:**
```
1. Forward project S → pixel (col, row) in frame A
2. Back-project (col, row) → elevation arc
3. Loss: distance from S to the arc
```

**Interpretation**: "The surfel's position should be consistent with where it renders"

**Multi-view version:**
```
Surfel S renders to:
  - pixel_A in frame A → Arc_A
  - pixel_B in frame B → Arc_B

Loss: S should lie on (or near) the intersection of Arc_A and Arc_B
```

If arcs don't intersect, either:
- Surfel position is wrong
- Scale factor is wrong
- Correspondences are wrong

### Pros/Cons

| Aspect | 1A (Per-Pixel Elevation) | 1B (Surfel-Centric) |
|--------|--------------------------|---------------------|
| Parameters | Many (256×200×num_frames) | None (uses existing surfels) |
| Correspondence | Needs explicit matching | Natural via surfel rendering |
| Gradient signal | Direct to elevation | Indirect via surfel position |
| One-to-many | Can represent multiple elevations | Limited by surfel count |

---

## Option 2: Cycle Consistency with Elevation

### Core Idea

Enforce that forward-then-backward projection recovers the original 3D positions, **including elevation**.

### Single-Frame Cycle (Weak)

```
Surfel S (x, y, z) with true elevation
    │
    ↓ forward project
    │
pixel (col, row) ← elevation information LOST here
    │
    ↓ back project (elevation=0)
    │
Recovered point S' (x', 0, z') ← wrong elevation
    │
    ↓
Cycle loss: ||S - S'||² would penalize correct elevation!
```

**Problem**: Single-frame cycle consistency **punishes** correct elevation because back-projection assumes elevation=0.

### Multi-Frame Cycle (Strong)

Use multiple frames to recover elevation through arc intersection:

```
Surfel S (has true elevation)
    │
    ├──► Forward to Frame A → pixel_A → Arc_A
    │
    └──► Forward to Frame B → pixel_B → Arc_B

Arc_A ∩ Arc_B should recover S's position (including elevation)
```

**Loss formulation:**
```
For surfel S visible in frames A and B:
  Arc_A = elevation_arc(back_project(frame_A, pixel_A))
  Arc_B = elevation_arc(back_project(frame_B, pixel_B))

  intersection_point = closest_point_on_both_arcs(Arc_A, Arc_B)

  Loss: ||S.position - intersection_point||²
```

**Interpretation**: The surfel should be where the arcs from different views intersect.

### Handling Non-Intersecting Arcs

Arcs may not perfectly intersect due to:
- Noise in poses
- Incorrect scale factor
- Surfel not actually visible in both frames

**Soft intersection**: Instead of exact intersection, use minimum distance between arcs:
```
Loss: min_{e_A, e_B} ||point_on_arc_A(e_A) - point_on_arc_B(e_B)||²
```

### Scale Factor Diagnostic

If scale factor is wrong, arcs from different frames will systematically fail to intersect. This could be used as:
- A diagnostic for scale factor quality
- An additional loss term to help scale factor converge

---

## Option 3: Guided Densification with Elevation Search

### Core Idea

When adding new surfels (densification), search along the elevation arc to find the best 3D position using multi-view consistency.

### Procedure

```
1. Identify high-error pixel in frame A:
   - GT is bright (strong return)
   - Rendered is dark (missing geometry)

2. This pixel defines an elevation arc in 3D

3. Find other frames where this arc is visible:
   - Forward-project points along the arc to other frames
   - Check which frames can see these points

4. For candidate elevations e ∈ [-10°, +10°]:
   - Compute 3D point P(e) on the arc
   - Forward-project P(e) to all overlapping frames
   - Score(e) = agreement with GT intensities in those frames

5. Best elevation = argmax_e Score(e)

6. Add new surfel at position P(best_e)
```

### Scoring Function

```
Score(e) = Σ_frames  GT_intensity(forward_project(P(e), frame))
```

High score means: the 3D point at elevation e projects to bright pixels in multiple frames → likely real geometry.

### When to Trigger

- Periodically during training (every N iterations)
- When loss plateaus (not improving)
- When specific regions have persistent high error

### Handling One-to-Many

A single high-error pixel might need **multiple** surfels along the arc (e.g., two surfaces at different elevations).

**Detection**: If Score(e) has multiple peaks along the arc, add surfels at each peak.

```
Score
  │    *           *
  │   * *         * *
  │  *   *       *   *
  │ *     *     *     *
  └─────────────────────── elevation
      ↑           ↑
    peak 1      peak 2

→ Add surfels at both elevations
```

---

## Idea A: Elevation as Latent Variable (Probabilistic)

### Core Idea

Instead of committing to a single elevation, maintain a **probability distribution** over the elevation arc for each pixel.

```
pixel (col, row) → P(elevation | pixel, frame, observations)
```

### Representation

**Per-pixel elevation distribution:**
```
For each valid pixel in each frame:
  μ[frame, row, col] = mean elevation (learnable)
  σ[frame, row, col] = uncertainty (learnable or fixed)

  P(e) = Normal(μ, σ²)  or  P(e) = weights over discrete bins
```

**Discrete version (simpler):**
```
Divide elevation range into K bins (e.g., K=20)
For each pixel: softmax weights over K bins

elevation_logits[frame, row, col, k] = learnable
elevation_probs = softmax(elevation_logits, dim=-1)
```

### Multi-View Belief Update

**The Problem**: A pixel in Frame A could correspond to ANY point along the elevation arc. We don't know which one.

**The Solution**: Use OTHER frames to narrow down which elevation is correct. The correct elevation is the one that's CONSISTENT across all frames.

**Intuition**:
- Wrong elevation → inconsistent across views → low probability
- Correct elevation → consistent across views → high probability

**Concrete Example**:
```
Frame A sees bright pixel at (128, 100)
Frame B is at a different viewing angle

Test elevation e = +5°:
  - Back-project from Frame A with e=+5° → 3D point P
  - Forward-project P to Frame B → lands at pixel (130, 98)
  - Check Frame B's GT: pixel (130, 98) is BRIGHT ✓
  - Likelihood is HIGH

Test elevation e = -5°:
  - Back-project from Frame A with e=-5° → 3D point Q
  - Forward-project Q to Frame B → lands at pixel (126, 102)
  - Check Frame B's GT: pixel (126, 102) is DARK ✗
  - Likelihood is LOW

Conclusion: e = +5° is more likely than e = -5°
```

**Bayesian Framing**:

*Prior*: Before looking at any evidence, all elevations are equally likely
```
P(e) = 1/K for all K bins  (uniform)
```

*Likelihood*: For each overlapping frame, ask "if the true elevation is e, how well does that explain what I see in this frame?"
```
Likelihood(frame B | elevation e) =
    "If I back-project with elevation e and forward-project to frame B,
     do I land on a bright pixel (consistent) or dark pixel (inconsistent)?"
```

*Posterior*: Multiply prior × all likelihoods
```
P(e | frames A,B,C) ∝ P(e) × L(A|e) × L(B|e) × L(C|e)
```

**Update Rule (Pseudocode)**:
```python
prior = uniform(K)  # start with equal probability for all elevations

for frame_B in overlapping_frames:
    for e in elevation_bins:
        # Back-project from frame A with elevation e
        P_e = back_project(frame_A, pixel_A, elevation=e)

        # Forward-project to frame B
        pixel_B = forward_project(P_e, frame_B)

        # How bright is frame B at that pixel?
        intensity_B = GT[frame_B][pixel_B]

        # Update likelihood (bright = consistent = high likelihood)
        likelihood[e] *= intensity_B

posterior = prior * likelihood
posterior = normalize(posterior)  # sum to 1
```

### Training with Distributions

**Expected position:**
```
E[P] = Σ_e P(e) × point_at_elevation(e)
```

**Variance/uncertainty:**
```
Var[P] = Σ_e P(e) × ||point_at_elevation(e) - E[P]||²
```

**Loss terms:**
1. **Reconstruction loss**: Render using expected positions
2. **Consistency loss**: Different frames should agree on the distribution
3. **Sharpness loss**: Encourage distributions to become peaked (reduce uncertainty)

### Distribution Sharpening

As training progresses:
- Multi-view evidence accumulates
- Distributions should become sharper (lower σ or more peaked softmax)
- Eventually converge to near-delta distributions (confident elevation)

**Temperature annealing:**
```
P(e) = softmax(logits / temperature)

Start with temperature = 1.0 (soft)
Anneal to temperature = 0.1 (sharp)
```

### Connection to One-to-Many

The distribution naturally handles multiple surfels:
- **Unimodal distribution** → single surfel on arc
- **Multimodal distribution** → multiple surfels at different elevations

```
Single surface:          Two surfaces:
P(e)                     P(e)
  │    ****                │   **      **
  │   *    *               │  *  *    *  *
  │  *      *              │ *    *  *    *
  └────────────            └────────────────
       ↑                       ↑      ↑
   one peak               two peaks
```

### Rendering with Distributions

**Option A: Sample-based rendering**
```
For each pixel:
  Sample e ~ P(e)
  Compute 3D point at sampled elevation
  Render that point

Repeat with multiple samples, average results
```

**Option B: Expectation rendering**
```
For each pixel:
  Compute expected 3D position E[P]
  Render expected position

Add variance as uncertainty measure
```

**Option C: Mixture rendering**
```
For each pixel with multimodal P(e):
  Identify K modes (peaks)
  Render each mode as separate surfel
  Combine contributions
```

---

## Idea B: Elevation Prediction Network

### Core Idea

Train a small neural network to predict elevation from local image context.

```
Local patch around pixel → CNN → predicted elevation
```

### Architecture

```
Input:
  - Local intensity patch (e.g., 11×11 around pixel)
  - Global context (full image, downsampled)
  - Frame metadata (pose, if useful)

Network:
  - Small CNN encoder (few conv layers)
  - Output: elevation value or distribution parameters

Output:
  - Single elevation (regression)
  - Or: μ, σ for Gaussian
  - Or: K logits for discrete bins
```

### Training Signal

The network is trained end-to-end with the main reconstruction loss:

```
pixel → Network → elevation → back-project → 3D point
                                              ↓
                                      used in multi-view loss
                                              ↓
                                      gradient flows back to network
```

**Intuition**: The network learns that certain intensity patterns correlate with elevation. For example:
- Bright spot with shadow below → surface tilted down (positive elevation)
- Intensity gradient patterns → hints about surface orientation

### Generalization

Unlike per-pixel parameters, the network can **generalize**:
- Across frames (same patterns → same elevation prediction)
- To new frames (inference on unseen data)

### Challenges

- May require significant data to learn meaningful patterns
- Sonar intensity patterns might not contain enough elevation information
- Risk of overfitting to training frames

---

## Idea C: Implicit Elevation via Enhanced Surfel Optimization

### Core Idea

Don't explicitly compute elevation during back-projection. Instead, rely on gradient descent to move surfels to correct elevations, but **enhance the loss to provide stronger elevation gradients**.

### Current Situation

- Surfels are initialized at elevation=0
- Single-frame loss doesn't care about elevation (same pixel regardless)
- Surfels have no gradient signal to move in elevation direction

### Enhanced Multi-View Loss

Add explicit term that creates elevation gradients:

**Reprojection consistency loss:**
```
For surfel S visible in frames A and B:
  pixel_A = forward_project(S, frame_A)
  pixel_B = forward_project(S, frame_B)

  intensity_A = GT_intensity(frame_A, pixel_A)
  intensity_B = GT_intensity(frame_B, pixel_B)

  Loss: (intensity_A - intensity_B)²
```

If S is at wrong elevation:
- It might hit bright pixel in frame A but dark pixel in frame B
- Loss is high
- Gradient pushes S to position where intensities match

**Geometric consistency loss:**
```
For surfel S visible in frames A and B:
  range_A = compute_range(S, frame_A)
  range_B = compute_range(S, frame_B)

  expected_range_A = GT_range_from_row(pixel_A)  # row → range
  expected_range_B = GT_range_from_row(pixel_B)

  Loss: (range_A - expected_range_A)² + (range_B - expected_range_B)²
```

### Elevation Gradient Analysis

For a surfel at position (x, y, z), moving in y (elevation direction):
- Changes which pixel it hits in each frame
- If frames are at different angles, the pixel change is different per frame
- Multi-view loss creates asymmetric gradients → net force in y direction

```
Frame A (viewing from angle α)     Frame B (viewing from angle β)

Moving surfel up (Δy):             Moving surfel up (Δy):
  - pixel shifts by Δrow_A           - pixel shifts by Δrow_B
  - Δrow_A ≠ Δrow_B (different viewing angles)

Multi-view loss creates gradient in y!
```

### Comparison to Explicit Methods

| Aspect | Implicit (Idea C) | Explicit (Ideas A, B) |
|--------|-------------------|----------------------|
| Additional parameters | None | Elevation params or network |
| Computational cost | Low | Medium to high |
| Convergence | May be slow | Could be faster |
| Multimodal handling | Limited | Better (Idea A) |

---

## Comparison Summary

| Option | Elevation Representation | Multi-View Usage | One-to-Many | Complexity |
|--------|-------------------------|------------------|-------------|------------|
| **1A** | Per-pixel learnable scalar | Correspondence loss | Explicit per-pixel | Medium |
| **1B** | Implicit in surfel position | Arc constraint loss | Via surfel count | Low |
| **2** | Recovered via arc intersection | Core mechanism | Via intersection | Medium |
| **3** | Searched during densification | Scoring function | Multi-peak detection | Medium |
| **A** | Probability distribution | Belief update | Multimodal distributions | High |
| **B** | Network prediction | End-to-end training | Network capacity | Medium |
| **C** | Implicit in surfel position | Enhanced loss terms | Via surfel count | Low |

---

## Next Steps

1. **Expand Idea A** (probabilistic elevation) - user preference
2. Design concrete loss functions
3. Determine data structures and memory requirements
4. Plan implementation phases

---

## Note: Missing High-Level Decisions

Two areas are not yet planned at a high level and should be called out explicitly:

1. **Loss formulation**: which elevation-aware loss (arc-intersection, multi-view likelihood, surfel-to-expected-point anchor, variance, etc.) is the primary driver, and how it is staged/weighted.
2. **Normals path**: whether normals remain elevation-agnostic (elevation=0) during early stages, or if/when they should use learned elevation.

---

## Plan Response: Loss + Normals (gpt-5.2-codex)

### Guiding objective

- **End goal**: prioritize geometric correctness of mesh extraction; elevation-aware losses and normals handling should be evaluated by downstream mesh fidelity, not just image loss.

### Loss formulation (high-level)

- **Primary driver in Stage 1**: GT-anchored elevation distribution likelihood (bins) with entropy/temperature annealing.
- **Optional stabilizer**: keep arc loss available only as a low-weight term when extra stabilization is needed.
- **Always-on baseline**: existing photometric + bright-pixel loss stays active, but elevation-aware losses remain separate and weighted to avoid destabilizing scale or appearance.
- **Stage 2 (optional)**: densification uses multi-view agreement scoring rather than adding a new global loss term.

### Normals path (high-level)

- **Early Stage 1**: allow normals to update via photometric loss, but keep any explicit normal-based regularization losses very low-weight and ramp them up gradually; avoid hard coupling to uncertain elevation early.
- **Late Stage 1 onward**: recompute normals using expected elevation (from bins) once distributions are stable; enable stronger normal-based regularizers after convergence.
- **Throughout training**: geometric correctness of normals matters because mesh export is the end goal. Multi-view training naturally constrains normals; the staged approach controls HOW STRONGLY we regularize, not WHETHER normals matter.

### Priority note (gpt-5.2-codex)

- **Mesh quality focus**: prioritize surfel positions and sizes over normals for final mesh fidelity; normals are secondary and should not destabilize geometry.

### Decision: Fixed opacity for sonar (opus4.5, user)

**Default: All surfels are fully opaque (opacity = 1.0).** Configurable toggle to re-enable learnable opacity if needed.

**Physical reasoning**: Sonar images underwater hard surfaces (metal, concrete, rock, seabed). Unlike light interacting with glass or fog, sound waves at these frequencies reflect off hard surfaces — they do not pass through. There is no physical basis for partial opacity in this domain. Every surface is either there (fully opaque) or not there (no surfel).

**Impact on the intensity model**:
```
Before (learnable opacity):
  intensity = opacity × max(0, normal · view_direction)
  Three degrees of freedom: opacity, normal, view_direction(position)
  Optimizer can trade off wrong normal ↔ wrong opacity to match brightness

After (fixed opacity = 1.0):
  intensity = max(0, normal · view_direction)
  Two degrees of freedom: normal, view_direction(position)
  Normal is directly constrained by brightness — less room to cheat
```

**Why this helps geometric correctness**:
- Removes one degree of freedom the optimizer could exploit
- Normal orientation is more tightly constrained by the photometric loss
- If brightness is wrong, the optimizer MUST fix the normal or position — it can't compensate with opacity
- Fewer learnable parameters = simpler optimization

### Normals concern: appearance vs geometric correctness (opus4.5)

**Mesh export is the end goal**, so geometric correctness of normals matters throughout training, not just at export time.

**The problem**: The photometric loss only sees rendered pixel values. With fixed opacity, the current intensity model is:
```
intensity = max(0, normal · view_direction)
```

Note: this model is **physically incomplete** — see "Sonar intensity physics" section below for the full model. But even with a corrected model, the optimizer could still find wrong-position + wrong-normal combos that produce correct brightness from one viewpoint.

**Why multi-view helps**: A "cheated" normal might look right from one viewing angle but wrong from another. With orbiting frames, a surfel is seen from many angles — this naturally constrains both position and normal. This is the primary defense against appearance-geometric divergence.

**Why fixing normals only at export is risky**: If normals are "appearance-correct" but not "geometry-correct" during training, surfel positions may also be wrong (they co-adapted with the wrong normals). Swapping in correct normals at export doesn't fix the positions.

**Implication**: Normal geometric correctness should be considered during training, not deferred to export.

**Possible approaches**:
1. **Neighbor normal consistency (smoothness prior)**: nearby surfels should have similar normals. Needs a tunable smoothness weight. **Edge-aware caveat**: a naive KNN approach would incorrectly smooth normals across sharp edges (e.g., two walls meeting at 90°). Two surfels can be spatially close but on different surfaces with legitimately different normals. Mitigations:
   - **Normal-gated smoothness**: only apply the smoothness loss between neighbors whose normals are already within some angular threshold (e.g., < 45°). If normals differ by > 45°, assume they're on different surfaces and skip the penalty.
   - **Clustering / segmentation**: group surfels into surface patches first, only smooth within patches.
   - **Anisotropic weighting**: weight smoothness by both spatial distance AND normal similarity — large normal difference → low weight.
2. **Multi-view normal consistency**: a surfel seen from many angles naturally constrains its normal. This should always be active — the more frames we train with, the stronger this constraint.
3. **Elevation-derived normals**: once elevation distributions are available, compute expected normal from local surface geometry (finite differences of surfel positions). Penalize deviation from quaternion normal.

### Sonar intensity physics (opus4.5, user)

**Current model** (in render_sonar):
```
intensity = max(0, normal · view_direction)
```

**Target model** — includes **inverse square law** for physically correct distance falloff:
```
intensity = max(0, normal · view_direction) / distance²
```

Where `distance` is the range from the surfel to the sonar origin. This matters because:
- A surfel at 3m returns much less intensity than the same surfel at 0.5m
- The current model treats both equally, which is physically wrong
- This could cause the optimizer to compensate with wrong normals for distant surfels

**Decision**: Use a stabilized inverse-distance family model for training:
`intensity = max(0, normal · view_direction) * gain / (max(distance, r0)^p + eps)`.
For this project, the saved sonar images are treated as raw amplitudes (no range/gain compensation in data), so distance attenuation stays enabled by default. Start with `p=2.0`, keep `r0` (near-range floor) to avoid dynamic-range blow-up, and tune exponent/gain using mesh-first ablations.

**Implementation note**: This requires modifying `render_sonar()` to include distance-based attenuation.

### Surface scattering: deliberately ignored for now (user)

**The physical environment**: The dataset is a **swimming pool**. Surfaces include:
- **Tiles**: smooth — specular reflection (sound bounces like a mirror, strong return only at specific angles)
- **Grout between tiles**: rough — diffuse scattering (sound scatters in all directions, more uniform return)
- **Drain covers**: different material/geometry, likely mixed reflection

The Lambertian model (pure diffuse) is a rough approximation. In reality the pool is dominated by smooth tiles which are more specular than diffuse. The scattering behavior depends on surface roughness relative to the acoustic wavelength.

**Decision**: Ignore scattering physics for now. Use a single Lambertian reflectance model for all surfaces. The structure of interest is a PVC cube so all its surfaces are uniform in texture.

**Known limitation**: Tiles may produce specular highlights (strong returns at mirror angles, weak returns elsewhere) that the Lambertian model cannot explain. The optimizer may compensate by distorting normals to "explain" specular highlights as diffuse returns. This is an acceptable tradeoff for the current stage of development.

**Future importance**: To handle mixed tile/grout surfaces properly, would need per-surfel reflectance parameters (specular vs diffuse ratio, roughness) — similar to BRDF in standard rendering. Could also help with drain covers and other non-uniform features.

---

## Open Questions

1. **Hardware budget**: 8GB VRAM (4070 laptop), Intel i9 13th gen, 32GB RAM. Priority is to make it run first, optimize later.

---

## Correspondence Mechanism (opus4.5)

**Question**: How do we match pixels across frames for multi-view losses?

### Options Considered

**Option A: Surfel-Anchored Correspondence**

Use existing surfels as the bridge between frames.
```
Surfel S at position (x, y, z)
    ├──► Forward project to Frame A → pixel_A
    └──► Forward project to Frame B → pixel_B

These pixels correspond because they both come from surfel S.
```
- Pros: No extra data structures, naturally handles visibility, correspondences update as surfels move
- Cons: Only works where surfels exist, poor correspondences early in training when surfels are sparse/wrong

**Option B: Elevation Arc Projection**

For a pixel in Frame A, project its elevation arc to Frame B.
```
Frame A pixel → back-project with elevation e → 3D point P(e) → forward-project to Frame B
```
Different elevations land on different Frame B pixels — test all K bins, find which has consistent intensity.
- Pros: Works without surfels, naturally finds correct elevation
- Cons: Expensive (K projections per pixel per frame pair), assumes visibility

**Option C: Intensity Matching (Classic Stereo)**

Search Frame B for pixels with similar intensity to Frame A pixel, constrained to the epipolar curve (projected elevation arc).
- Pros: Classic technique
- Cons: Sonar is low-texture (large uniform regions), needs geometric constraint anyway

**Option D: No Explicit Correspondence (Implicit via Loss)**

Don't compute pixel-to-pixel correspondences. Train with all frames — the photometric loss naturally pushes surfels to multi-view consistent positions.
- Pros: Simplest, no overhead, already how current training works
- Cons: Relies on gradient descent, may converge slowly

### Decision

**Use Option D first** — no explicit correspondence. Multi-view consistency emerges implicitly from training with many frames (500 frames with overlapping FOV).

If convergence is slow or geometry is poor, can add **Option A** (surfel-anchored) as an explicit multi-view consistency loss:
```
For surfel S visible in frames A and B:
    pixel_A = forward_project(S, frame_A)
    pixel_B = forward_project(S, frame_B)
    Loss += (GT_A[pixel_A] - GT_B[pixel_B])²
```

**Note**: Option B (arc projection) is essentially what Idea A's "multi-view belief update" does — it's baked into the elevation distribution approach rather than being a separate correspondence step.

---

## Recommended Path (gpt-5.2-codex)

### Summary

- **Stage 0 (initialization)**: Random elevation-aware initialization.
- **Stage 1 (probabilistic elevation)**: Use GT-anchored multi-view likelihood with a small elevation distribution (K≈5–9 bins) and entropy/temperature annealing as the primary driver.
- **Stage 2 (optional densification)**: If specific regions remain high-error, add surfels along elevation arcs at multi-view agreement peaks (aligns with opus4.5 Option 3).

### Rationale

- Low VRAM footprint with a single primary optimization stage after initialization.
- Resolves one-to-many ambiguity without per-pixel parameter explosion.
- Keeps extra complexity targeted to optional densification only where needed.

---

## Feedback to Incorporate (opus4.5)

- **Stage 0 init**: Randomize surfel elevation at initialization instead of setting all to elevation=0.
- **Stage 1 bins**: Keep K small for first pass (K=5–9), but make K configurable for later increases (e.g., K=21).
- **Stage 1 preference**: Start with discrete bins rather than stochastic sampling for stability and debuggability.
- **Stage 2**: Keep densification optional and only use where needed.
- **Schedule triggers**: Define thresholds/schedules for temperature annealing, coupling ramp, and support-pruning strictness.

---

## Integration Decisions (gpt-5.2-codex)

- **Accept random elevation init** with uniform sampling across the full elevation FOV by default (aligns with physical sonar ambiguity; keep range configurable if needed).
- **Use discrete bins first**, keep K configurable from the start; target K=5–9 for dev, higher later.
- **Keep Stage 2 optional**; only enable for persistent high-error regions.
- **Use one primary training stage after initialization** (bin likelihood + mandatory coupling), with optional low-weight arc stabilizer as a config flag.
- **Progression default**: fixed schedules with optional early tightening based on entropy/support metrics (manual override still possible).

---

## Decision Update (2026-02-06): Remove Arc-Only Stage 1, Enforce Multi-View Support

This update supersedes the earlier recommendation that used a geometry-first arc-consistency Stage 1 as the primary entry point.

### Why

- Arc loss built only from `surfel -> project -> arc` can be self-referential: the same surfel that generated the pixel also lies on that pixel's arc.
- In that form, the loss can be minimized without adding real data-driven elevation correction.

### Updated training path

- Keep **Stage 0**: random elevation-aware initialization.
- Make bin-likelihood the first primary training stage (previously Stage 2): GT-anchored, multi-view, with temperature/entropy schedule.
- Keep optional arc loss only as a low-weight stabilizer if needed, not as the primary driver.
- Keep optional densification stage for persistent high-error regions.

### New surfel retention policy

- Do not keep/prune surfels based on FOV overlap alone.
- A surfel is considered supported by a frame only if it is:
  - validly projected (in-FOV, in-front, in-bounds),
  - lands on meaningful GT return,
  - and passes residual consistency checks.
- Require support from multiple frames with **viewpoint diversity** (avoid counting many near-duplicate views as strong support).
- Use warmup then soft-to-hard **ratio thresholds** with count floors and EMA/hysteresis before pruning:
  - `support_ratio = support_count / max(1, diverse_candidate_count)`
  - mid: `support_ratio >= 0.25` with floor `support_count >= 2`
  - late: `support_ratio >= 0.45` with floor `support_count >= 4`

---

## Decision Update (2026-02-06): Mandatory Belief-to-Geometry Coupling

### Problem

- Pixel-level elevation bins/logits can improve (lower entropy, better likelihood) without sufficiently moving surfel geometry.
- That creates a failure mode where the inference head looks good but mesh geometry does not improve enough.

### Decision

- Add a required coupling term from pixel elevation belief to surfel state.
- For selected anchor pixels:
  - compute expected 3D point from bin probabilities,
  - associate expected points to candidate surfels on-the-fly,
  - apply a robust attraction loss to surfel positions.
- Keep associations soft and ephemeral each iteration (no persistent global hard map).

### Practical policy

- Use projection and depth residual gates before coupling.
- Use Huber/robust loss and a warmup ramp for coupling weight.
- Increase coupling strength as elevation entropy falls.
- Track coupling residual metrics and verify mesh gains, not only likelihood gains.

---

## Decision Update (2026-02-06): Robust Normalized Likelihood Model

### Problem

- Raw `log(intensity)` as likelihood evidence is fragile under sonar gain variation, speckle, and dropouts.
- Out-of-FOV samples should not automatically become strong negative evidence.

### Decision

- Use robust normalized amplitude likelihood as default:
  - percentile-normalize intensity per frame on valid returns (default P10-P99),
  - compute clipped log-evidence on normalized intensity,
  - treat invalid/out-of-FOV projections as neutral evidence,
  - apply frame reliability weights.

### Why

- Retains amplitude information needed for elevation disambiguation.
- Improves training stability across frame-to-frame gain/contrast changes.
- Reduces false penalties from visibility limits.
- Keeps implementation lightweight and debuggable.

### Fallbacks (optional)

- Binary hit/miss likelihood.
- Hybrid amplitude + hit/miss likelihood.

---

## Decision Update (2026-02-07): Raw Sonar Data Assumption

### Fact recorded

- Training sonar images are raw amplitude data (no explicit range/gain compensation in stored frames).

### Planning impact

- Keep distance attenuation enabled by default in rendering.
- Treat attenuation-off mode as ablation/diagnostic only.
- Continue using robust per-frame normalization in likelihood for numerical stability, not as a replacement for physics.

---

## Decision Update (2026-02-07): Attenuation Dynamic-Range Stabilization

### Problem

- Raw-data attenuation is required, but plain `1/r^2` with clamp can overemphasize near returns and suppress far returns.
- This can bias training toward near geometry and weaken far-surface elevation recovery.

### Decision

- Use stabilized attenuation in renderer:
  - `I = lambert * gain / (max(r, r0)^p + eps)`
  - default: `p=2.0`, `r0=0.35 m`, `eps=1e-6`, configurable `gain`.
- Keep `SONAR_USE_RANGE_ATTEN=1` as default; attenuation-off remains diagnostic only.

### Why

- Preserves physical distance falloff for raw sonar amplitudes.
- Near-range floor prevents saturation blow-up and improves optimization stability.
- Tunable exponent/gain allows calibration without discarding physical structure.

### Validation policy

- Run fixed-seed exponent ablation on `p in {1.5, 2.0, 2.5}`.
- Track near-range saturation and far-range response statistics in addition to mesh metrics.

---

## Decision Update (2026-02-07): Canonical Sonar-to-Camera Mount

### Fact recorded

- Physical mount on ROV is fixed as:
  - `8 cm` behind camera,
  - `10 cm` above camera,
  - `5 deg` downward pitch.

### Canonical implementation constants

- Camera-frame translation: `[0.0, -0.10, -0.08]` meters.
- Rotation: `+5 deg` pitch about camera X-axis.

### Planning impact

- Treat these as the default extrinsic constants across all training paths.
- Resolve doc/code mismatches against this tuple.
- Include active extrinsic values in run-header logging to prevent silent branch drift.

---

## Decision Update (2026-02-07): Coordinate Convention Contract

### Problem

- Multiple modules use mixed frame descriptions (camera-frame vs sonar-frame notation).
- Translation storage/read conventions (row vs column) can silently diverge across branches.

### Decision

- Canonical compute frame for projection/back-projection: camera/view frame `(+X right, +Y down, +Z forward)`.
- Canonical image convention: azimuth columns (`left=+`, `right=-`), range rows (`top=near`, `bottom=far`).
- Canonical elevation-angle sign: `+elevation -> +Y` (down), `-elevation -> -Y` (up), measured in the camera/view frame.
- Canonical mount tuple remains camera-frame: `[0.0, -0.10, -0.08]` m with `+5 deg` pitch about X.
- Canonical transform layout for `world_view_transform`-compatible paths: translation in row 3 (`T[3, :3]`).

### Why

- Prevents silent sign/axis bugs that mimic optimization failure.
- Removes row/column ambiguity already seen in branch history.
- Makes debug and training paths comparable by construction.

### Enforcement policy

- Add startup logging of active convention and extrinsic tuple.
- Add small known-point tests (azimuth sign, lateral sign, transform roundtrip identity).
- Add elevation sign test at boresight/range anchor (`+elevation` gives `y>0`, `-elevation` gives `y<0` in camera/view frame).
- Keep assertions enabled in debug runs by default.

---

## Decision Update (2026-02-09): Frame Filtering Belongs to Dataset Preparation

### Scope decision

- Filtering of candidate sonar frames is a dataset-preparation step, not a training-loop step.

### Context

- Typical training subsets use about `500` frames sampled from `>2000` matched sonar+pose frames.
- Legacy and R2 datasets did not use the filtering pipeline below when those 500-frame subsets were produced.

### Dataset-prep filtering pipeline

1. **Quality gate**: reject weak/noisy frames using per-frame valid-return ratio and high-percentile intensity.
2. **Pose dedup**: remove near-identical poses using translation + heading thresholds.
3. **Diverse subsample**: choose target count via greedy farthest-point sampling in pose space.
4. **Connectivity check**: ensure selected frames have enough multi-view neighbors; replace isolated frames.

### Output contract

- Save selected frame IDs/names as part of dataset split metadata.
- Training consumes this precomputed list directly.
