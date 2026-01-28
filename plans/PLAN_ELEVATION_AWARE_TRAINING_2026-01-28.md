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

## Authorship Marker

- **opus4.5**: Content above this marker (original plan text)
- **gpt-5.2-codex**: Content below this marker (addendum)

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

- Stage 1 (geometry-first): estimate elevation per surfel (or per region) using multi-view arc consistency, with minimal photometric reliance.
- Stage 2 (photometric): freeze or lightly regularize elevation and run standard training to refine appearance and geometry.
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

## Open Questions

1. **Correspondence mechanism**: How exactly to match pixels across frames for multi-view losses?

2. **Continuous vs discrete**:
   - Optimize elevation as continuous variable?
   - Or use discrete bins (K=20)?

3. **Distribution family**: For Idea A, use Gaussian, categorical, or mixture?

4. **Initialization**: Start distributions as uniform? Or peaked at elevation=0?

5. **Computational budget**: What's acceptable iteration time increase?

---

## Recommended Path (gpt-5.2-codex)

### Summary

- **Stage 1 (geometry-first)**: Use surfel-centric arc constraints or soft arc-intersection loss to inject elevation gradients with minimal extra parameters (aligns with opus4.5 Option 1B / Option 2).
- **Stage 2 (probabilistic elevation)**: Replace fixed elevation with a small distribution (K≈5–9 bins) or 2–4 stochastic samples; apply entropy/temperature annealing to sharpen over time (aligns with opus4.5 Idea A + my stochastic sampling).
- **Stage 3 (optional densification)**: If specific regions remain high-error, add surfels along elevation arcs at multi-view agreement peaks (aligns with opus4.5 Option 3).

### Rationale

- Low VRAM footprint and stable early training (Stage 1).
- Resolves one-to-many ambiguity without per-pixel parameter explosion (Stage 2).
- Targeted complexity only where needed (Stage 3).

---

## Feedback to Incorporate (opus4.5)

- **Stage 1 init**: Randomize surfel elevation at initialization instead of setting all to elevation=0.
- **Stage 2 bins**: Keep K small for first pass (K=5–9), but make K configurable for later increases (e.g., K=21).
- **Stage 2 preference**: Start with discrete bins rather than stochastic sampling for stability and debuggability.
- **Stage 3**: Keep densification optional and only use where needed.
- **Alternative**: Consider running Stage 1 + Stage 2 together from the start (small K), to avoid a hard switch.
- **Stage transition triggers**: Define how to move from Stage 1 → Stage 2 if sequential.

---

## Integration Decisions (gpt-5.2-codex)

- **Accept random elevation init** with uniform sampling across the full elevation FOV by default (aligns with physical sonar ambiguity; keep range configurable if needed).
- **Use discrete bins first**, keep K configurable from the start; target K=5–9 for dev, higher later.
- **Keep Stage 3 optional**; only enable for persistent high-error regions.
- **Start sequential for clarity**, but allow a combined Stage 1+2 mode as a config flag.
- **Stage transition default**: fixed iteration count with optional early switch if arc-residual falls below a threshold (manual override still possible).
