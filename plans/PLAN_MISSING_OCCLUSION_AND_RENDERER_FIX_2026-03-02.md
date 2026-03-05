# Plan: Sonar Renderer Fixes — Occlusion, Normals, and Baseline Correctness

**Date:** 2026-03-02 (revised 2026-03-04)
**Status:** Draft for execution
**Scope:** Sonar renderer correctness fixes, acoustic occlusion, and synthetic re-baseline policy

---

## Background

During Chunk-4 gating, the synthetic cube dataset (Dataset C) consistently failed to converge to correct geometry — surfels formed a torus instead of a cube. Escalating Chunk-4 enforcement (coupling strength, support-prune pressure) either had no effect or collapsed the geometry entirely. Investigation traced the failure to **renderer-level defects** that predate all elevation-aware work.

These defects were first documented in `plans/PLAN_ELEVATION_AWARE_CHUNK4_EXECUTION_2026-02-23.md` starting at line 875. This plan supersedes that section for active implementation and gating.

### Why this blocks everything downstream

The elevation-aware pipeline (Chunks 1–5) trains against rendered sonar images. If the renderer produces physically wrong images, then:
- The photometric loss teaches surfels the wrong geometry
- Stage-1 posterior evidence is contaminated (high entropy, weakly discriminative)
- Chunk-4 coupling reinforces incorrect geometry rather than correcting it

Fixing the renderer is a prerequisite for meaningful evaluation of any elevation-aware feature.

---

## Problem Summary

The sonar renderer (`render_sonar` in `gaussian_renderer/__init__.py`) has six confirmed defects, ordered by impact:

### P0 — Discarded normal initialization + Lambertian dead zone

These two defects compound each other and are described together.

**Discarded normals:** `debug_multiframe.py:3275–3283` computes camera-facing normals for every surfel at init. These normals are passed into `BasicPointCloud` and handed to `GaussianModel.create_from_pcd`. But `create_from_pcd` (`scene/gaussian_model.py:139`) ignores `pcd.normals` entirely and initializes rotation quaternions randomly:

```python
rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
```

Every initialized surfel exists because the sonar received a reflection from that point — by definition, the surface was facing the sonar (`lambertian > 0`). Random quaternions throw away this physical invariant. With uniformly random normals on the unit sphere, ~50% of surfels face away from any given sonar view and become invisible.

**Lambertian dead zone:** `gaussian_renderer/__init__.py:447` computes:

```python
lambertian = torch.clamp(torch.sum(normals_world * dir_to_sonar, dim=-1), min=0)
```

The `clamp(min=0)` is mathematically identical to a ReLU. When `dot(normal, direction) < 0`, the gradient is exactly zero — the well-known "dead neuron" problem applied to surfels.

**How normals affect rendered intensity:** The sonar renderer computes `intensity = opacity * lambertian * attenuation` (`gaussian_renderer/__init__.py:451–462`). The Lambertian term `clamp(dot(normal, dir_to_sonar), min=0)` directly scales the rendered intensity of every surfel. This means surfel normal orientation controls not just whether a surfel is visible, but *how bright* it appears. A surfel perfectly aligned with the sonar beam produces full intensity; one at a grazing angle produces weak intensity; one facing away produces zero. This is the correct physical model (specular reflection off a surface), but combined with the `clamp(min=0)` dead zone, surfels that face away receive zero gradient on rotation and cannot learn to turn around.

**Visible symptom:** Rendered sonar images show "speckled dots" instead of the smooth intensity arcs seen in ground truth.

**Why synthetic cubes fail but synthetic spheres don't:** On the synthetic datasets, a sphere is rotationally symmetric — the "correct" normal at any point is similar to many random orientations, and all views provide consistent gradient signal. A cube has 6 discrete face normals. A surfel on a cube face must find one specific direction, and the dead zone makes this much harder.

**Why streaks form:** Surfels initialized from each training camera develop normal biases toward that camera's direction (the only views giving them gradient). Over training, this creates structured clusters in normal space aligned with individual camera poses, which project as streak patterns in 3D.

**Surfel drift trap:** Even after fixing initialization, surfels can drift into a permanently dead configuration. A surfel in front of a cube face but facing backward is trapped: it's invisible from the front (dead zone) and physically occluded from the rear (the cube blocks the beam). No view provides gradient. With fixed opacity, it cannot fade out. With no detection mechanism, it cannot be pruned.

### P1 — No acoustic occlusion

**What:** `render_sonar` (`gaussian_renderer/__init__.py:516–536`) uses purely additive `scatter_add_` accumulation with no depth ordering. Every surfel that passes the FOV check contributes to the rendered image regardless of whether a closer surface blocks the acoustic beam.

**Physical sonar occlusion model:** A solid surface at range R1 blocks the acoustic beam, so nothing at range R2 > R1 along the **same azimuth AND same elevation angle** should produce a return. The farther surfel projects to a different pixel (different range row), but the return should not exist at all. Occlusion is per-(azimuth, elevation) ray, not per-pixel.

Two surfels at different azimuth or different elevation angles never occlude each other, even at different ranges.

**Why it matters:**
- Back-face surfels (from Poisson reconstruction of closed meshes) produce ghost returns at range rows where the GT has no signal
- The optimizer contorts surfel positions and rotations to suppress ghost intensity, since fixed opacity prevents fading
- This compounds with the dead zone: random normals make front-face surfels invisible while no occlusion makes back-face surfels visible — the renderer simultaneously suppresses correct surfaces and hallucinates incorrect ones

### P2a — Point splatting instead of Gaussian splatting

**What:** `render_sonar` splats each surfel to exactly 4 pixels via bilinear interpolation (`gaussian_renderer/__init__.py:482–536`). `pc.get_scaling` is never used for the spatial footprint. The `scaling_modifier` parameter is accepted (`line 337`) but never referenced in the function body. The standard camera `render()` uses the full CUDA rasterizer with 2D Gaussian covariance covering potentially hundreds of pixels per surfel.

**Why it matters:** Scale parameters receive no useful gradient from the photometric loss and cannot influence surface coverage. This limits the expressiveness of the surfel representation.

### P2b — Densification structurally disabled

**What:** `debug_multiframe.py` Stage-2 loop never calls `densify_and_clone/split`. `render_sonar` returns placeholder zeros for `viewspace_points` (`line 400`) and `radii` (`line 599`), so even if densification were called, it would have no gradient signal to act on.

**Why it matters:** Surfel count is fixed from initialization (~113K) and only decreases via FOV pruning. The model cannot add surfels where coverage is sparse.

**Scope note:** Densification is part of Chunk 5's scope ("Optional Stage 2 densification hook, default disabled"). Only Chunks 1–4 are currently implemented; Chunk 5 has not been started. The renderer fix here is limited to ensuring the plumbing is not structurally broken (meaningful `viewspace_points`/`radii` outputs) so that Chunk 5 can enable densification when it lands. This plan does not implement the densification hook itself.

### P3 — Correctness bugs

**FOV pruning `require_all` is broken** (`debug_multiframe.py:319–326`): Both branches of the `if require_all` / `else` block compute identical logic (`all_masks.any(dim=0)` → `~visible_from_any`). The `require_all` parameter has no effect. The `require_all=True` branch should use `.all(dim=0)` instead of `.any(dim=0)`.

**Coordinate convention divergence** between `utils/sonar_utils.py:29` and `utils/point_utils.py:77`:
- `sonar_utils.py` declares: +X right, +Y down, +Z forward
- `point_utils.py` declares: +X forward, +Y right, +Z down
- The formulas confirm this is a real divergence, not just a comment error. Forward projection (`render_sonar`) and backward projection (`point_utils`) use inconsistent coordinate frames.

**Bilinear splat boundary clamping** (`gaussian_renderer/__init__.py:483–486`): `col_floor` and `row_floor` are not lower-bounded before computing `col_ceil = col_floor + 1`. At the image boundary, ceil clamps to equal floor, producing degenerate weight distributions. Contributions at image edges can be silently lost. *(This defect is confined to the legacy bilinear path; under this revised plan, the legacy path is removed rather than patched.)*

### Why opacity learning is not the answer

In standard 2DGS, opacity learning is the natural self-pruning mechanism: useless surfels fade to zero. For sonar rendering of solid objects, opacity has no physical analog — acoustic surfaces either reflect or they don't. There is no partial acoustic transparency for solid underwater objects. Fixed opacity at ~1.0 is the physically correct model. The dead-surfel problem must be solved through normal initialization, gradient safety, and occlusion, not through opacity.

---

## Fix Plan

### R1 — Normal initialization + Lambertian gradient safety

**Normal-based quaternion initialization:**
- Modify `GaussianModel.create_from_pcd` to consume `pcd.normals` when provided.
- Convert camera-facing normals to quaternions at init time.
- Deterministic fallback for missing/invalid normals (e.g., identity quaternion or random with warning).

**Leaky Lambertian** (recommended default):

```python
alpha = 0.01
dot_val = torch.sum(normals_world * dir_to_sonar, dim=-1)
lambertian = torch.where(dot_val >= 0, dot_val, alpha * dot_val)
```

Wrong-facing surfels contribute 1% intensity. Gradient always flows. `alpha=0.01` is the standard Leaky ReLU default.

**Important physical caveat:** Leaky Lambertian is **not physically correct**. In reality, inner surfaces of a solid object are always enveloped by outer surfaces — the acoustic beam cannot reach them. There is no physical scenario where an inner/back-facing surface produces a sonar return. The leak is a gradient-flow workaround, not a physics model. This means leaky Lambertian **must be paired with acoustic occlusion** (R2): occlusion handles the physics (blocking inner surfaces), while the leak handles the optimization (ensuring gradient flows to wrong-facing surfels so they can correct themselves). Without occlusion, the leak would cause inner surfaces to contribute small but nonzero ghost intensity with no mechanism to suppress it. With occlusion, inner surfaces behind a solid front surface are blocked regardless of the leak value.

**Alternative options** (for ablation, not default):

- *ELU-style:* `torch.where(dot_val >= 0, dot_val, alpha * (torch.exp(dot_val) - 1))` — exponential decay for stronger wrong-facing suppression, smoother gradient near zero.
- *Straight-through estimator:* Forward pass uses hard `clamp(min=0)`, backward pass sees a `clamp(min=0.01)` floor. Zero visual change; gradient still flows. More complex but preserves exact physical model in forward pass.

Gate the Lambertian mode behind an env var (`SONAR_LAMBERTIAN_MODE=leaky|clamp0|elu|ste`) for ablation. Default: `leaky`.

**Diagnostics to add:**
- Per-iteration fraction of surfels with `dot < 0` (dead-zone occupancy)
- Per-iteration fraction where leaky path activates
- Mean/max `|dot|` for wrong-facing surfels (drift monitoring)
- Surfel size distribution telemetry (world-space and image-space) so scale drift/collapse is observable during training

**Surfel size telemetry contract (required):**
- **World-space scales:** log distribution for `s1`, `s2`, and equivalent radius `r_eq = sqrt(s1 * s2)`
- **Image-space support:** log rendered support distribution from rasterizer radii (pixels) when available (`2dgs`/`2dgs_nonlinear`)
- **Summary stats:** at minimum `mean`, `median`, `p01`, `p05`, `p95`, `p99`, `min`, `max`
- **Population splits:** log stats for `(all surfels)` and `(currently valid/in-FOV surfels)` to separate model drift from visibility effects
- **Cadence:** log every N iterations (`SONAR_SURFEL_STATS_EVERY`, default 50) and at stage boundaries/checkpoint saves

### R2 — Restore proper 2D Gaussian splatting + acoustic occlusion

The current `render_sonar` uses ad-hoc 4-pixel bilinear splatting with additive accumulation — neither proper Gaussian footprints nor physically correct occlusion. This step restores proper 2DGS footprints and introduces an explicit occlusion pipeline aligned with sonar physics.

**Approach split: ray-space occlusion first, then image rendering**

Standard graphics visibility logic still applies, but the indexing space differs from pinhole cameras:
- In first-person camera rendering, a pixel typically corresponds to one ray, so image-space depth buffering is enough.
- In this sonar image, a pixel `(azimuth, range)` marginalizes elevation, so one displayed pixel corresponds to many `(azimuth, elevation)` rays.

Therefore, occlusion is resolved in ray space first, before elevation is integrated out.

**R2 occlusion semantic contract (required):**
- Occlusion is evaluated per `(azimuth_bin, elevation_bin)` ray in strict increasing sonar range order (continuous range key, not camera-Z key).
- Only events on the same `(azimuth_bin, elevation_bin)` ray can occlude each other.
- Different elevation bins never occlude each other, even when they later map to nearby/same `(azimuth, range)` image support.
- Final 2D image is produced by elevation marginalization after per-ray occlusion.

**Occlusion vs footprint ownership contract (required):**

This plan separates two questions that were previously conflated:

- **Occlusion question:** "is this return physically visible on this ray?"
- **Footprint question:** "how is a visible return spatially distributed on the image?"

To avoid ray-leak artifacts, this plan adopts **multi-bin participation** (center-bin-only ownership is explicitly rejected):

- A surfel may contribute to multiple `(azimuth_bin, elevation_bin)` rays according to its angular footprint support.
- For each contributing ray `k`, the surfel emits one occlusion event with shared range `r0` and ray-specific alpha/intensity share `alpha_k`.
- Occlusion is then solved independently on each ray with front-to-back compositing.
- This prevents the failure mode where a wide surfel deposits energy on neighboring rays that were never included in occlusion ordering.

**High-level pipeline:**
1. Project surfels to sonar coordinates `(azimuth, elevation, range)` and apply hard geometric validity (`in_front`, FOV, range).
2. Build per-surfel angular footprint support on the `(azimuth_bin, elevation_bin)` lattice and compute per-ray participation weights `{w_k}`.
3. Convert each participating surfel-ray pair into an event tuple `(ray_id, range=r0, alpha_share=q*w_k)`, where `q` is the surfel base return strength (Lambertian * attenuation * opacity).
4. For each `ray_id`, sort events by increasing `range` and apply front-to-back transmittance/compositing.
5. Accumulate the elevation-resolved return volume `R[a, e, r]`.
6. Marginalize elevation to form the displayed image:
   `I[a, r] = sum_e w_e * R[a, e, r]`, where `w_e` is a normalized elevation beam weight.
7. Render footprint mode (`2dgs` or `2dgs_nonlinear`) under this fixed occlusion semantics contract.

**Multi-bin support capping policy (required):**

Multi-bin participation is required for physics fidelity, but event count must be bounded for performance and memory. The cap policy is:

1. Build a local candidate window in `(azimuth_bin, elevation_bin)` around each surfel center using an elliptical support threshold
   `(x - mu)^T Sigma^{-1} (x - mu) <= k_sigma^2` with `SONAR_OCCL_KSIGMA=2.5` (default).
2. Compute raw per-bin weights from the angular Gaussian footprint.
3. Drop bins with negligible contribution using a relative threshold
   `w < w_peak * SONAR_OCCL_WEIGHT_FLOOR_REL`, default `SONAR_OCCL_WEIGHT_FLOOR_REL=1e-3`.
4. Apply hard top-K cap by keeping only the highest-weight bins, default `SONAR_OCCL_TOPK=16`.
5. Renormalize retained weights so `sum_k w_k = 1` before creating ray events.

This cap makes per-iteration event complexity bounded by `O(N * K_max)` instead of unbounded support growth. For example, with ~113k surfels and `K_max=16`, the event budget is ~1.8M surfel-ray events before per-ray sorting/compositing, which is tractable on current GPU targets.

**Mass-loss telemetry contract (required):**

Support capping introduces approximation. Therefore each run must log pre-renormalization truncation mass loss per surfel:

`mass_lost = 1 - sum(raw_retained_weights)`

and summarize it at least as `mean`, `median`, `p95`, `p99`, `max`. This telemetry is required in gate metadata so we can prove caps are not materially distorting ray participation.

**Footprint rendering implementation (2DGS reuse): Jacobian-based baseline**

The existing CUDA rasterizer (`submodules/diff-surfel-rasterization`) is ~85% generic 2D code. The only pinhole-specific part is the projection step (`compute_transmat` in `forward.cu:75–118`), which can be completely bypassed via the `transMat_precomp` input. When `transMat_precomp` is provided, the rasterizer skips all pinhole math and uses the precomputed 2D transforms directly. Everything downstream — tile assignment, Gaussian evaluation (`exp(-0.5 * rho)`), front-to-back alpha compositing with depth ordering — works unchanged on any image layout, including a 256x200 sonar image.

The Jacobian footprint steps:

1. **Project surfel positions to sonar image space** in PyTorch (azimuth→col, range→row).

2. **Compute the Jacobian** of the polar projection at each surfel center. The polar projection is:
   ```
   azimuth = -atan2(x, z)
   range   = sqrt(x² + y² + z²)
   col     = (-azimuth / half_azi_rad + 1) * (W/2)
   row     = (range - range_min) / (range_max - range_min) * H
   ```
   The Jacobian `J = ∂(col, row) / ∂(x, y, z)` is a 2x3 matrix, analytically computable.

3. **Transform 3D surfel covariance to 2D image-space covariance** via `Sigma_2D = J * Sigma_3D * J^T`.

4. **Construct the T matrix** (3x3 per surfel) in the format the rasterizer expects and pass via `transMat_precomp`.

5. **Provide sonar range as depth key** where depth ordering is used, but do not rely on image-space alpha compositing alone as the full acoustic occlusion model.

6. The rasterizer handles Gaussian footprint evaluation/tile rendering; ray-space occlusion semantics remain governed by the contract above.

**Known limitation — Jacobian approximation:**

The polar projection is nonlinear (atan2, sqrt). The Jacobian is a first-order local linear approximation — exact at the surfel center but increasingly inaccurate further from it. For small surfels on a 256x200 image this is likely acceptable, but for surfels spanning many degrees of azimuth or a large range extent, the linear approximation could introduce distortion in the Gaussian footprint shape. This is an **open concern** that needs validation on synthetic data, and alternative approaches (e.g., custom CUDA polar projection kernel, or higher-order correction) should be considered if the approximation proves inadequate.

**What this replaces:**
- The entire bilinear `scatter_add_` accumulation path (`gaussian_renderer/__init__.py:482–536`) is replaced by proper 2DGS rasterization.
- The old additive/no-occlusion behavior is replaced by explicit ray-space occlusion + elevation marginalization.
- P2a (point splatting) and P1 (missing occlusion) are solved together under the ray-space occlusion contract.

**What stays the same:**
- Sonar-specific physics: Lambertian intensity model, range attenuation, FOV checks
- Image layout: 256 cols (azimuth) × 200 rows (range)
- Loss computation against GT sonar images

**Mode flag:** `SONAR_RENDER_MODE=2dgs|2dgs_nonlinear`.
- `2dgs` = Jacobian-based polar projection + CUDA rasterizer path (current R2 baseline)
- `2dgs_nonlinear` = nonlinear covariance pushforward path (R2-alt below)

`legacy` bilinear scatter mode is removed as part of this plan and is no longer a supported execution mode.

**Occlusion flag (orthogonal to render mode):** `SONAR_OCCLUSION_MODE=none|ray_binned`.
- `none` = v2 ablation/debug only (non-physical for solid-object sonar)
- `ray_binned` = required acoustic occlusion contract (default for active runs)

**Compatibility/parity policy after legacy removal (required):**

- "Parity" no longer means matching pre-v2 legacy bilinear outputs.
- "Parity" now means matching a frozen v2 compatibility reference run under:
  `SONAR_RENDER_MODE=2dgs`, `SONAR_OCCLUSION_MODE=none`, `SONAR_LAMBERTIAN_MODE=clamp0`.
- This avoids the previous policy conflict where we tried to both delete/fix legacy behavior and simultaneously require exact equivalence to it.

**Elevation integration controls:**
- `SONAR_ELEV_BINS` (discrete elevation bins for ray-space compositor)
- `SONAR_ELEV_WEIGHT_MODE=uniform|beam_pattern` (marginalization weights `w_e`)

### R2-alt — Sigma-point projection (alternative to Jacobian for footprint accuracy)

This is an alternative to the Jacobian approach in R2, not a replacement. Both produce a 2D Gaussian footprint on the sonar image and feed the same CUDA rasterizer. They differ only in **how** they compute that 2D footprint from the surfel's disc shape.

Occlusion semantics are unchanged by R2-alt: it must use the same `SONAR_OCCLUSION_MODE=ray_binned` contract (per `(azimuth_bin, elevation_bin)` range-ordered compositing, then elevation marginalization).

#### The problem R2-alt solves

R2 uses a Jacobian — a linear approximation of the sonar projection at the surfel center. This is like approximating a curved coastline with a straight ruler placed at one point. For small surfels (small ruler), the straight line is close enough. For large surfels or surfels at steep azimuth angles where the polar projection curves sharply, the straight-line approximation distorts the footprint shape.

R2-alt avoids the linear approximation entirely. Instead, it samples a few representative points from the surfel's disc, projects each one through the **exact** nonlinear sonar projection (atan2, sqrt — no approximation), and fits a 2D Gaussian to the projected results. This is analogous to the Unscented Transform used in navigation/filtering (e.g., UKF in spacecraft attitude estimation) — same idea, different application.

#### How it works, step by step

**Important clarification before Step 1:**

R2-alt does **not** create or fit a new 3D surfel. The trainable surfel parameters remain unchanged:
- center `mu` in 3D
- rotation quaternion
- two in-plane scales
- opacity/features (as already defined by the model)

R2-alt only changes how that existing surfel is converted into a per-frame 2D rendering footprint for the rasterizer.

**Step 1 — Build the surfel's disc axes in world coordinates.**

Each surfel is a flat 2D Gaussian disc floating in 3D space. It has only two meaningful dimensions — like a thin coin. Its shape is defined by:

- A quaternion (4 numbers) that encodes the disc's orientation — which way it faces. This converts to a 3x3 rotation matrix $R$.
- Two scale values $s_1, s_2$ (in meters) that encode the disc's extent along its two in-plane axes — like the semi-major and semi-minor axes of an ellipse drawn on the disc's surface.

The code builds a 3x3 transformation matrix $L$ that maps from the disc's local 2D coordinate frame to world coordinates:

$$L = R \, S$$

where:

$$S = \begin{pmatrix} s_1 & 0 & 0 \\ 0 & s_2 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

- $R$ is the 3x3 rotation matrix from the surfel's quaternion
- $s_1, s_2$ are the surfel's two learnable scale parameters (meters) — these are the only scale values the model optimizes
- The third diagonal entry is fixed at $1.0$ — it is **not a learnable parameter**. The surfel has no thickness; this is just a dummy value needed to make the matrix 3x3 for the math to work.

The two columns of $L$ that matter are:

$$\mathbf{t}_1 = R \begin{pmatrix} s_1 \\ 0 \\ 0 \end{pmatrix}, \quad \mathbf{t}_2 = R \begin{pmatrix} 0 \\ s_2 \\ 0 \end{pmatrix}$$

These are the surfel's two tangent vectors in world coordinates — they point along the disc's two axes, scaled by the disc's extent in each direction. Everything about the surfel's shape on the sonar image is determined by where these two vectors project to.

**Step 2 — Generate sigma points on the disc.**

Sample a small set of deterministic points that capture the disc's extent. These are called "sigma points" (from the Unscented Transform literature). Since the surfel is a 2D disc, we only need to sample along the two tangent directions:

$$\chi_0 = \boldsymbol{\mu}$$
$$\chi_1 = \boldsymbol{\mu} + \sqrt{n + \lambda} \; \mathbf{t}_1, \quad \chi_3 = \boldsymbol{\mu} - \sqrt{n + \lambda} \; \mathbf{t}_1$$
$$\chi_2 = \boldsymbol{\mu} + \sqrt{n + \lambda} \; \mathbf{t}_2, \quad \chi_4 = \boldsymbol{\mu} - \sqrt{n + \lambda} \; \mathbf{t}_2$$

where:
- $\boldsymbol{\mu}$ is the surfel center position in 3D world coordinates (meters)
- $\mathbf{t}_1, \mathbf{t}_2$ are the disc's two tangent vectors from Step 1 (meters) — they encode both direction and extent
- $n = 2$ is the sigma-point dimensionality for the surfel disc
- $\lambda = \alpha^2 (n + \kappa) - n$ is the standard UKF scaling term
- with defaults $\alpha = 1.0$, $\kappa = 0$, the spread factor is $\sqrt{n + \lambda} = \sqrt{2} \approx 1.414$
- This produces $2 \times 2 + 1 = 5$ sigma points total: the center plus two points along each of the disc's two axes

Think of it as placing 4 probe points at the "edges" of the disc (one pair along each axis), plus the center. These 5 points together capture the disc's position, size, and orientation. There is no third axis to sample because the surfel is flat.

**Pinned sigma-point hyperparameters (required in gate metadata):**

| Parameter | Env var | Default | Notes |
|-----------|---------|---------|-------|
| $\kappa$ | `SONAR_SIGMA_KAPPA` | `0` | UKF secondary spread parameter |
| $\alpha$ | `SONAR_SIGMA_ALPHA` | `1.0` | UKF spread scaling |
| $\beta$ | `SONAR_SIGMA_BETA` | `2.0` | Gaussian prior (optimal for Gaussian distributions) |

Derived values (not independently tunable once `alpha`, `beta`, `kappa` are set, $n = 2$):
- Sigma point count: $2n + 1 = 5$ (fixed)
- $\lambda = \alpha^2 (n + \kappa) - n$
- Mean weights: $w_0^{(m)} = \lambda / (n + \lambda)$, $w_i^{(m)} = 1 / (2(n + \lambda))$
- Covariance weights: $w_0^{(c)} = w_0^{(m)} + (1 - \alpha^2 + \beta)$, $w_i^{(c)} = 1 / (2(n + \lambda))$

For pinned defaults (`alpha=1.0`, `beta=2.0`, `kappa=0`):
- $\lambda = 0$
- $w_0^{(m)} = 0$, $w_i^{(m)} = 0.25$
- $w_0^{(c)} = 2.0$, $w_i^{(c)} = 0.25$

These values must be recorded in gate metadata as `sigma_point_config: {kappa, alpha, beta}`.

**Step 3 — Project each sigma point through the exact sonar projection.**

Each sigma point is a 3D point in world coordinates. "Projecting" it means computing where it appears on the sonar image — the same math `render_sonar` already does. Apply the full nonlinear sonar projection to each sigma point:

$$\text{col}_i = f_{\text{col}}(\chi_i), \quad \text{row}_i = f_{\text{row}}(\chi_i)$$

where $f_{\text{col}}$ and $f_{\text{row}}$ are the sonar projection functions (from R2):

$$\text{azimuth} = -\text{atan2}(x, z)$$
$$\text{range} = \sqrt{x^2 + y^2 + z^2}$$
$$\text{col} = \left(\frac{-\text{azimuth}}{\theta_{\text{half}}} + 1\right) \cdot \frac{W}{2}$$
$$\text{row} = \frac{\text{range} - r_{\min}}{r_{\max} - r_{\min}} \cdot H$$

where:
- $(x, y, z)$ are the sigma point's coordinates in the sonar frame (meters)
- $\theta_{\text{half}}$ is the half-azimuth FOV (radians) — for our sonar, $60°$
- $W = 256$, $H = 200$ are the sonar image dimensions (pixels)
- $r_{\min} = 0.2$ m, $r_{\max} = 3.0$ m are the sonar range bounds

No linearization here — each sigma point goes through the actual atan2 and sqrt. This is the key difference from R2. The result is 5 points in image space (col, row) that show where the disc's center and edges actually land on the sonar image.

**Validity flags (per sigma point):**

After projection, each sigma point receives a validity flag based on hard physical constraints:
- `in_front`: $z > 0$ in sonar frame
- `in_range`: $r_{\min} \leq \text{range} \leq r_{\max}$
- `in_azimuth`: $|\text{azimuth}| \leq \theta_{\text{half}}$
- `in_elevation`: $|\text{elevation}| \leq \phi_{\text{half}}$

A sigma point is **valid** only if all four flags are true. Invalid sigma points are **not** clamped or wrapped — they are excluded from the Step 4 fitting and their weights are renormalized. This avoids edge-clamp bias (which would pull centroids inward and shrink covariance) and wrap artifacts (which would create phantom footprints on the opposite side of the image).

**Step 4 — Fit a 2D Gaussian to the projected sigma points.**

We started with a 2D disc in 3D world space. After projection, we have 5 points on the 2D sonar image. We need to describe the surfel's footprint as a 2D Gaussian on the image (an ellipse with a center, size, and orientation in pixels) because that's what the CUDA rasterizer needs to render it.

Compute the weighted mean and covariance of the projected points:

$$\boldsymbol{\mu}_{2D} = \sum_{i=0}^{4} w_i \begin{pmatrix} \text{col}_i \\ \text{row}_i \end{pmatrix}$$

$$\Sigma_{2D} = \sum_{i=0}^{4} w_i \left(\begin{pmatrix} \text{col}_i \\ \text{row}_i \end{pmatrix} - \boldsymbol{\mu}_{2D}\right) \left(\begin{pmatrix} \text{col}_i \\ \text{row}_i \end{pmatrix} - \boldsymbol{\mu}_{2D}\right)^T$$

where:
- $w_i$ are the sigma-point weights (standard UKF weights with $\lambda = \alpha^2 (n + \kappa) - n$: $w_0 = \lambda / (n + \lambda)$ for the center, $w_i = 1 / (2(n + \lambda))$ for the others, with $n = 2$ since the disc is 2D)
- $\boldsymbol{\mu}_{2D}$ is the 2D center of the footprint on the sonar image (pixels)
- $\Sigma_{2D}$ is a 2x2 covariance matrix describing the footprint's shape on the sonar image — an ellipse encoding how many pixels wide and tall the surfel appears, and at what angle

**Why Step 4 exists:**

The projected sigma points describe a nonlinear warped footprint, but the CUDA rasterizer input contract expects a compact Gaussian-style footprint representation. Step 4 is therefore an adapter: it moment-matches the projected sigma points into a single 2D Gaussian `(mu_2D, Sigma_2D)` that the rasterizer can consume.

This 2D Gaussian is **temporary and per-forward-pass**. It is not stored as a new surfel, not persisted in checkpoints, and does not add new learned state.

**Accuracy envelope of Step 4 (training relevance):**

- It is an approximation, not exact physics.
- It is typically accurate when projected surfel footprints are small-to-moderate and away from strong projection curvature regions.
- It degrades for very large surfels, steep azimuth curvature, and near FOV/wrap boundaries where a single ellipse cannot represent the full warped shape.

Because this approximation is used during training, fidelity must be validated explicitly (see `RB-T15`). If large-footprint error is material, fallback options include adaptive surfel subdivision or tightening scale caps in high-curvature regions.

**Boundary-aware fitting policy:**

When some sigma points are invalid (outside FOV/range/elevation per Step 3 validity flags), the fit adapts based on valid count $K$:

| Valid count $K$ | Behavior |
|-----------------|----------|
| $K \geq 3$ | Fit $\mu_{2D}$, $\Sigma_{2D}$ from valid sigma points with renormalized weights |
| $K < 3$ (but center valid) | Fallback to center-based Jacobian footprint (R2 path for this surfel) |
| $K = 0$ or center invalid | Skip rendering this surfel for this view |

**Soft boundary weighting (edge band):**

To avoid gradient discontinuities when sigma points cross FOV boundaries, apply a smooth taper before hard exclusion:

1. Compute distance-to-boundary $d_i$ for each sigma point in image space (pixels to nearest edge).
2. Apply sigmoid weight: $\tau_i = \sigma((d_i - d_{\text{margin}}) / d_{\text{taper}})$ where $d_{\text{margin}} = 4$ px, $d_{\text{taper}} = 2$ px.
3. Multiply sigma-point weights by $\tau_i$ before fitting; renormalize.
4. Hard exclusion applies only for physically invalid points (`in_front` false, outside range/elevation limits).

This replaces the abrupt valid/invalid transition with a smooth falloff near image boundaries while preserving hard physics gating.

**Fallback hysteresis:**

To prevent frame-to-frame jitter when valid count fluctuates near the threshold:
- Enter fallback (Jacobian) mode when $K < 3$.
- Exit fallback (return to sigma-point fit) when $K \geq 4$.
- Per-surfel fallback state is ephemeral (per-frame, not persisted).

**Covariance conditioning:**

After fitting $\Sigma_{2D}$:
1. Symmetrize: $\Sigma_{2D} \leftarrow (\Sigma_{2D} + \Sigma_{2D}^T) / 2$.
2. Eigendecompose; clamp eigenvalues to $[\lambda_{\min}, \lambda_{\max}]$ where $\lambda_{\min} = 0.1$ px$^2$, $\lambda_{\max} = 400$ px$^2$ (20 px radius cap).
3. Optional condition-number cap: $\lambda_{\max} / \lambda_{\min} \leq 100$.
4. Reconstruct $\Sigma_{2D}$ from clamped eigenvalues.

**Loss-side border mask (recommended):**

Downweight or mask a thin outer image border (e.g., 2–4 px) during training loss computation. This prevents edge-approximation artifacts from dominating the optimizer and is orthogonal to the sigma-point boundary policy.

**Boundary diagnostics (must log at `SONAR_SURFEL_STATS_EVERY` cadence):**
- Fraction of surfels using fallback ($K < 3$)
- Histogram of $K$ values (0..5)
- Count of invalid-by-reason (azimuth / range / elevation / in_front)
- Fraction of surfels in edge band ($d < d_{\text{margin}}$ px)

**Step 5 — Convert to rasterizer format and render.**

Convert $\Sigma_{2D}$ and $\boldsymbol{\mu}_{2D}$ into the 3x3 transform matrix $T$ that the CUDA rasterizer expects (via `transMat_precomp`), then render exactly as in R2: the rasterizer evaluates the 2D Gaussian and handles tile-based rendering, while authoritative acoustic occlusion remains the ray-binned compositor contract above.

**`Sigma_2D -> T` conversion contract:**

The CUDA rasterizer expects a 3x3 matrix $T$ per surfel such that the rasterizer evaluates $\exp(-0.5 \, \rho)$ where $\rho$ is the Mahalanobis distance induced by $\Sigma_{2D}$. The conversion path is:

1. Start from conditioned $\Sigma_{2D}$ (post Step 4 covariance conditioning — already symmetric, PSD, eigenvalue-clamped).
2. Compute the Cholesky factor $L$ such that $\Sigma_{2D} = L L^T$, or equivalently eigendecompose $\Sigma_{2D} = V \Lambda V^T$ and set $L = V \sqrt{\Lambda}$.
3. Construct $T$ in the format the rasterizer expects (encodes $\mu_{2D}$ and $L$ into the 3x3 layout used by `transMat_precomp`).

**PSD safeguards (applied before Cholesky):**
- $\Sigma_{2D}$ is already symmetrized and eigenvalue-clamped per Step 4 covariance conditioning.
- If Cholesky fails (numerically degenerate despite conditioning), fall back to eigendecomposition path with $\sqrt{\max(\lambda_i, \lambda_{\min})}$.
- Add jitter $\epsilon I$ ($\epsilon = 10^{-6}$) as last resort if both paths fail. Log a warning if jitter is triggered.

**Gradient flow requirement:** The entire path $\Sigma_{2D} \to L \to T$ must be differentiable. PyTorch's `torch.linalg.cholesky` and eigendecomposition are differentiable; the jitter fallback path must also preserve gradients (add jitter before decomposition, not after).

This conversion is validated by `RB-T19` (forward/backward consistency: a known $\Sigma_{2D}$ produces a $T$ that reconstructs the same covariance, and gradients flow through the full path).

#### Comparison: R2 (Jacobian) vs R2-alt (sigma points)

| Aspect | R2 (Jacobian) | R2-alt (sigma points) |
|--------|---------------|----------------------|
| Approximation | First-order linear (tangent plane) | Captures second-order curvature |
| Accuracy for small surfels | Excellent | Excellent (no meaningful difference) |
| Accuracy for large/oblique surfels | Degrades with surfel size | Remains accurate |
| Compute cost per surfel | 1 matrix multiply | 5 projections + covariance fit |
| Implementation complexity | Lower (analytical Jacobian) | Moderate (batch projection + weighted covariance) |
| Differentiability | Yes (analytical gradients) | Yes (all operations are differentiable in PyTorch) |

#### Why add this mode

- Better footprint fidelity for surfels where the polar projection curves significantly (large surfels, steep azimuth, close range where range-to-row mapping compresses)
- Avoids maintaining a hand-derived Jacobian that must be verified for correctness
- Enables direct A/B comparison against the Jacobian path on synthetic gates to quantify whether the approximation matters in practice

#### Risks / cost

- More compute than Jacobian mode (~5x more projection evaluations per surfel)
- Requires careful handling near azimuth wrap boundaries (±60°) where sigma points could project outside the image
- Still depends on range-based sort order for physically correct acoustic occlusion (same as R2)

#### Recommended policy

Start with `2dgs` (Jacobian) as the initial footprint implementation — it's simpler and faster to validate. Then implement `2dgs_nonlinear` (sigma points) as a required second footprint mode. Keep `SONAR_OCCLUSION_MODE=ray_binned` fixed while comparing footprint modes, so occlusion semantics do not drift across runs. Synthetic gate runs should compare both footprint modes to determine which is the default for production use.

### R3 — Densification signal wiring

With proper 2DGS rendering (R2), `viewspace_points` and `radii` come from the rasterizer with meaningful values rather than placeholder zeros. This step ensures the densification plumbing is correct so Chunk 5 can enable it.

- Verify that the rasterizer's `viewspace_points` and `radii` outputs carry correct gradients under both `2dgs` (Jacobian) and `2dgs_nonlinear` (sigma-point) paths, with `ray_binned` occlusion enabled.
- Densification remains optional but no longer structurally blocked.
- Remove all legacy bilinear placeholder-only outputs; densification signal paths are validated only on `2dgs` and `2dgs_nonlinear`.

### R4 — Consistency fixes

- **FOV pruning:** Fix the `require_all=True` branch to use `all_masks.all(dim=0)` instead of `all_masks.any(dim=0)`.
- **Coordinate conventions:** Standardize `sonar_utils.py` and `point_utils.py` on a single frame convention. Add roundtrip contract tests (forward-project then backward-project recovers original point within tolerance).
- **Legacy-path cleanup:** remove legacy bilinear renderer entry points and config handling so no stale code path remains callable.

### R5 — Documentation and plan hierarchy patches

Patch these upstream plans with v2 renderer contract addenda:

| Document | What to add |
|----------|-------------|
| `plans/PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md` | "Renderer Baseline Contract v2" section; dependency note before chunk progression |
| `plans/PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md` | Executable renderer contracts (normal init, transfer modes, ray occlusion, footprint) |
| `plans/PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md` | Pre-chunk gate: "Renderer remediation gate" with pass criteria |
| `plans/PLAN_ELEVATION_AWARE_TRAINING_FLOW_2026-02-24.md` | Mark pre-v2 evidence as historical; add renderer-version branch |
| `plans/scientific_progress.md` | Update forward model equations to include v2 transfer + occlusion |
| `docs/SYNTHETIC_DATASET_GUIDE.md` | Refresh-trigger policy and mandatory renderer fingerprint fields |
| `plans/progress_overview.md` | Split evidence into pre/post renderer fix sections |

### R6 — Synthetic re-baseline

After any accepted v2 change, pre-existing synthetic gate claims become historical-only.

**Minimum rerun checklist:**
- `C4-S1` (sphere-A guard) and `C4-S2` (cube-C gate)
- `C4-S3` (direct active path) and `C4-S4` (continuation)
- `C4-T11` (off-mode compatibility smoke against frozen v2 reference: `2dgs` + `SONAR_OCCLUSION_MODE=none` + `SONAR_LAMBERTIAN_MODE=clamp0`)
- `C4-T17` (manual artifact verdict panel for Dataset C)

**Required metadata in gate summaries:**
- `renderer_semantics_version`, `normal_init_mode`, `lambertian_transfer`, `occlusion_model`
- `render_sonar_contract_hash` (or commit SHA)
- `dataset_contract_version`, `evaluator_contract_version`, `baseline_group_id`
- `surfel_size_stats_schema_version` and final-iteration surfel size snapshot (`s1/s2/r_eq` + image-space radii percentiles)
- `sonar_render_mode`, `sonar_occlusion_mode`, `occlusion_space` (must be `ray_binned` for active runs)
- `occlusion_footprint_policy` (must be `multi_bin_participation`)
- `occlusion_support_cap_config: {k_sigma, weight_floor_rel, topk}`
- `occlusion_support_cap_mass_loss` summary (`mean`, `median`, `p95`, `p99`, `max`)
- `compat_reference_id` (frozen v2 off-mode reference artifact used by `RB-T13`)
- `elevation_bin_count`, `elevation_bin_policy`, `elevation_weight_mode` (`uniform` or `beam_pattern`)
- `sigma_point_config: {kappa, alpha, beta}` (required when `sonar_render_mode=2dgs_nonlinear`)
- `sigma_point_boundary_policy_version` (hash or version tag of the boundary handling contract)
- `sigma_point_fallback_fraction` (final-iteration fraction of surfels using Jacobian fallback)

**Comparator rule:** Cross-run deltas are valid only when renderer semantic fingerprints match.

---

## Execution Order

Steps must be executed in order. Each step gates the next.

| Step | What | Files touched | Depends on |
|------|------|---------------|------------|
| **R0** | Write failing contract tests (`RB-T01`..`RB-T10`, `RB-T15`, `RB-T16`, `RB-T18`, `RB-T19`, `RB-T20`, `RB-T21`, `RB-T22`) | `tests/test_renderer_baseline_*.py` | — |
| **R1** | Normal init + leaky Lambertian + diagnostics | `scene/gaussian_model.py`, `gaussian_renderer/__init__.py` | R0 |
| **R2a** | Implement ray-binned acoustic occlusion + elevation marginalization with multi-bin participation and support capping (`SONAR_OCCLUSION_MODE=ray_binned`) | `gaussian_renderer/__init__.py` | R1 (leaky Lambertian requires physically valid occlusion before active deployment) |
| **R2b** | Restore 2DGS Jacobian footprint rendering on top of R2a occlusion semantics | `gaussian_renderer/__init__.py`, `submodules/diff-surfel-rasterization/*` (if range-order sort changes needed) | R2a |
| **R2-alt** | Add sigma-point footprint mode (`2dgs_nonlinear`) — **required** | `gaussian_renderer/__init__.py` | R2b |
| **R3** | Densification signal wiring (verify `viewspace_points`/`radii` under both `2dgs` and `2dgs_nonlinear`, with `ray_binned` occlusion active) | `gaussian_renderer/__init__.py` | R2-alt |
| **R4** | Consistency fixes and legacy path removal (FOV prune, coordinates, renderer mode cleanup) | `debug_multiframe.py`, `utils/sonar_utils.py`, `utils/point_utils.py`, `gaussian_renderer/__init__.py` | R2b |
| **R5** | Plan hierarchy patches + synthetic policy | `plans/*.md`, `docs/SYNTHETIC_DATASET_GUIDE.md` | R1, R2a, R2b, R2-alt, R3, R4 |
| **R6** | Synthetic re-baseline and re-gate | output artifacts | R1, R2a, R2b, R2-alt, R3, R4, R5 |

Note: FOV/coordinate fixes are logically independent, but legacy-path removal in R4 depends on R2b being in place.

---

## Test Catalog

### Unit / contract tests (must fail before implementation)

| ID | Contract | What it validates |
|----|----------|-------------------|
| `RB-T01` | Normal-init ingestion | `create_from_pcd` consumes provided normals into quaternions |
| `RB-T02` | Init determinism | Quaternion init from normals is deterministic under fixed seed |
| `RB-T03` | Lambertian gradient safety | Wrong-facing surfels get non-zero gradient under leaky mode |
| `RB-T04` | Compatibility parity | `clamp0` mode reproduces the pre-leaky transfer behavior within the v2 renderer path |
| `RB-T05` | Ray-binned occlusion core | Same `(azimuth_bin, elevation_bin)` closer surfel suppresses farther surfel via front-to-back range ordering |
| `RB-T06` | No cross-ray occlusion | Different azimuth or different elevation bins do not occlude each other |
| `RB-T07` | Gaussian footprint from surfel scale | Changing surfel scale changes rendered spatial support (proper 2D Gaussian, not 4-pixel splat) |
| `RB-T08` | Densification signals | `viewspace_points`/`radii` from rasterizer carry meaningful values under both `2dgs` and `2dgs_nonlinear` |
| `RB-T09` | FOV prune semantics | `require_all=True` and `False` produce distinct expected behavior |
| `RB-T10` | Convention consistency | Forward-project then backward-project roundtrips within tolerance |
| `RB-T15` | Nonlinear footprint fidelity | `2dgs_nonlinear` better matches brute-force nonlinear projection oracle than Jacobian mode for large/oblique surfels |
| `RB-T16` | Range-order occlusion semantics | Occlusion ordering under `ray_binned` follows sonar range (not camera Z) on the same `(azimuth_bin, elevation_bin)` ray |
| `RB-T18` | Elevation marginalization contract | Final image equals weighted sum over elevation bins after per-bin occlusion (`I[a,r] = sum_e w_e * R[a,e,r]`) within tolerance |
| `RB-T19` | `Sigma_2D -> T` conversion fidelity | Forward/backward consistency of `(mu_2D, Sigma_2D) -> T` conversion: rasterizer footprint matches input covariance, gradients flow correctly through PSD conditioning path |
| `RB-T20` | Multi-bin participation semantics | A wide surfel contributes to multiple `(azimuth_bin, elevation_bin)` rays and is occluded independently per participating ray (reject center-bin-only leakage behavior) |
| `RB-T21` | Support-cap mass accounting | Top-K support capping preserves normalized participation weights and emits required truncation mass-loss telemetry |
| `RB-T22` | Legacy mode removal | `SONAR_RENDER_MODE=legacy` is rejected/unavailable and no bilinear scatter branch is reachable |

### Smoke / integration tests

| ID | Contract | What it validates |
|----|----------|-------------------|
| `RB-T11` | Shadow-mode debug run | Finite losses, no NaN/Inf with v2 renderer in shadow mode |
| `RB-T12` | Active-mode debug run | v2 diagnostics emitted and bounded |
| `RB-T13` | Off-mode compatibility | `2dgs` with `SONAR_OCCLUSION_MODE=none` and `SONAR_LAMBERTIAN_MODE=clamp0` matches frozen v2 reference outputs within tolerance |
| `RB-T14` | Checkpoint continuity | Save/load/continue with renderer-mode metadata |
| `RB-T17` | Surfel-size telemetry | Required world/image-space size distribution stats are emitted at configured cadence and remain finite |

---

## Exit Criteria

This plan is closeable when:

1. All contract tests pass (`RB-T01`..`RB-T10`, `RB-T15`, `RB-T16`, `RB-T18`, `RB-T19`, `RB-T20`, `RB-T21`, `RB-T22`).
2. All smoke tests pass (`RB-T11`..`RB-T14`, `RB-T17`).
3. Both `2dgs` (Jacobian) and `2dgs_nonlinear` (sigma-point) modes pass their respective tests.
4. `SONAR_OCCLUSION_MODE=ray_binned` is validated and used for active/gating runs.
5. R3 densification signal validation passes under both `2dgs` and `2dgs_nonlinear` modes.
6. Synthetic rerun checklist completed with renderer fingerprint metadata.
7. Chunk-4 gate posture re-evaluated using post-v2 evidence.
8. Upstream plan documents patched and cross-linked.

---

## Deliverables

- Renderer v2 implementation across `scene/gaussian_model.py`, `gaussian_renderer/__init__.py`, `debug_multiframe.py`, and `submodules/diff-surfel-rasterization/` (if rasterizer changes are needed for range-order depth sorting).
- Ray-binned occlusion compositor with multi-bin participation, support capping, and elevation marginalization (`SONAR_OCCLUSION_MODE=ray_binned`) with explicit metadata/versioning.
- Both `2dgs` and `2dgs_nonlinear` render modes implemented and tested.
- Legacy bilinear render path removed from runtime config and codepath dispatch.
- Renderer contract tests (`tests/test_renderer_baseline_*.py`).
- Updated synthetic gate metadata and versioning policy.
- Plan hierarchy addenda across upstream documents.
- Re-baselined gate report separating pre-v2 and post-v2 evidence.

---

## Appendix — Outstanding Plan Issues (Codex review)

The following items are still ambiguous or inconsistent in the current draft and should be resolved before execution lock:

1. ~~**Exit criteria coverage is stale.**~~ **Resolved** — exit criteria updated to include RB-T15, T16, T17 and require both render modes.
2. ~~**R2 file scope under-specified.**~~ **Resolved** — R2 execution row now includes `submodules/diff-surfel-rasterization/*`.
3. ~~**Deliverables do not match expanded scope.**~~ **Resolved** — deliverables now list rasterizer submodule changes and both render modes.
4. ~~**`R2-alt` requirement status is unclear.**~~ **Resolved** — R2-alt is marked required. Exit criteria explicitly require both modes to pass.
5. ~~**R3 validation only names Jacobian path.**~~ **Resolved** — R3 now validates densification signals under both `2dgs` and `2dgs_nonlinear`.
6. ~~**Sigma-point weighting/hyperparameters not pinned.**~~ **Resolved** — defaults pinned in R2-alt Step 2 (`SONAR_SIGMA_KAPPA=0`, `SONAR_SIGMA_ALPHA=1.0`, `SONAR_SIGMA_BETA=2.0`, standard UKF weights). Required in gate metadata as `sigma_point_config`.
7. ~~**Boundary handling for sigma points is not contract-defined.**~~ **Resolved** — deterministic boundary policy defined in R2-alt Step 3 (validity flags) and Step 4 (boundary-aware fitting with K-based fallback, soft border weighting, hysteresis, covariance conditioning). Validated by RB-T15 edge-case fixtures.
8. ~~**`Sigma_2D -> T` conversion contract is implicit.**~~ **Resolved** — explicit conversion contract added to R2-alt Step 5 (PSD safeguards, eigenvalue floor/jitter, symmetrization). Dedicated contract test `RB-T19` added for forward/backward consistency of the conversion path, supplementing RB-T07 and RB-T15.

---

## Appendix — Verbatim Discussion on Issues 6-8

Historical transcript only. If any statements here conflict with main plan sections above (including test IDs), the main plan sections take precedence.

**User:** check the plan again, opus updated it and also gave you response in the appendix. do you agree with opus?

**Assistant:** Short answer: **partly**.

I agree with Opus on items 1–5 (those are genuinely fixed in the plan).  
I **don’t fully agree** with 6–8 as currently written.

- **Agree (1–5):** exit criteria/test coverage, R2 scope, deliverables scope, R2-alt required status, and R3 wording are now materially better (`plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:439`, `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:393`, `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:451`, `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:394`, `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:395`).
- **#6 (weights/hparams deferred):** I’d call this only **partially acceptable**. Exact numbers can be implementation-time, but the plan should still require they are pinned and logged in metadata before first gate run.
- **#7 (boundary handling deferred):** I **disagree** with deferring at plan level. Boundary policy (drop/clamp/renormalize/wrap) can materially change gradients and outcomes; this should be explicitly contracted before execution.
- **#8 (“no action needed” on `Sigma_2D -> T`):** I **disagree**. In this repo/rasterizer path, that mapping is not trivial boilerplate; a precise conversion contract (or dedicated contract test) is still needed to avoid implementation drift.

One more thing still inconsistent:
- `RB-T08` text still says Jacobian-only (`plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:417`) while R3/exit now require both modes. That should be aligned.

So: mostly improved plan, but I’d ask Opus to harden 6–8 + that `RB-T08` mismatch before locking.

**User:** suggest want you like. ill look into it

**Assistant:** Nice — here’s exactly what I’d tighten before lock.

- Add a **reproducibility contract for sigma points** in `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:193`:
  - Pin defaults in-plan (even if implemented later), e.g. `SONAR_SIGMA_ALPHA`, `SONAR_SIGMA_BETA`, `SONAR_SIGMA_KAPPA`, `SONAR_SIGMA_COUNT`.
  - Require these values in gate metadata.

- Add a **deterministic boundary policy** in `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:330`:
  - Explicitly define one rule for out-of-FOV sigma points (recommended: drop invalid sigma points, renormalize remaining weights, fallback to center-only isotropic footprint if <3 valid points).
  - Make it part of RB-T15 fixtures.

- Add an explicit **`Sigma_2D -> T` conversion contract** near `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:306`:
  - State exact mapping and PSD safeguards (symmetrize, eigenvalue floor/jitter).
  - Add a small contract test (new `RB-T18`) verifying forward/backward consistency of the conversion path.

- Fix the current wording mismatch for densification:
  - `RB-T08` still says Jacobian-only at `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:417`.
  - Update to “under both `2dgs` and `2dgs_nonlinear`” to match R3/Exit Criteria.

- Extend required metadata at `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:375`:
  - Add `sonar_render_mode`, `sigma_point_policy_version`, boundary-policy fingerprint, and sigma hyperparameters.

- Optional schedule decision (worth making explicit):
  - If both modes are required (`plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md:441`), keep that.
  - If you want faster closure, mark `2dgs_nonlinear` as post-close hardening with mandatory ablation report instead.

If you want, I can patch these exact edits directly into the plan in one pass.

**User:** explain to me how you want to deal with the boundary policy

**Assistant:** Here’s the boundary policy I’d use for `2dgs_nonlinear` (sigma-point mode).

- **No wrap behavior:** treat sonar image as bounded FOV, not periodic.  
  - If a sigma point goes past azimuth/range/elevation bounds, it is **invalid**, not wrapped to the opposite side.

- **Do not clamp invalid sigma points into image:**  
  - Clamping to edge pixels biases the fitted footprint and creates fake edge streaks.  
  - Invalid points should be excluded from the fit.

- **Fit rule by valid count (`K`) after exact projection/FOV checks:**
  - `K >= 3`: fit `mu_2D, Sigma_2D` from valid sigma points, with **renormalized weights**.
  - `K < 3`: fallback to **center-based Jacobian footprint** (or a small isotropic fallback if Jacobian is degenerate).
  - `K = 0` or center surfel invalid: skip rendering that surfel for this view (same visibility semantics as normal pipeline).

- **Stability guardrails on fitted covariance:**
  - Symmetrize covariance.
  - Eigenvalue floor/ceiling (`lambda_min`, `lambda_max`) to keep PSD and avoid exploding footprints.
  - Optional condition-number cap.

- **Boundary diagnostics (must log):**
  - Fraction of surfels using fallback (`K<3`).
  - Histogram of `K` (0..5).
  - Count of invalid-by-azimuth / invalid-by-range / invalid-by-elevation.
  - This tells you if boundary behavior is dominating training.

Why this policy:
- It is deterministic and easy to test.
- It avoids artificial edge artifacts from clamping.
- It keeps training stable when sigma points are partially outside FOV.
- It gives you clear telemetry to know when boundary effects are a real problem.

If you want, I can convert this into exact plan text + a concrete `RB-T15` edge-case fixture spec.

**User:** could the boundary handling be on step 4 rather than step 3? physcially seems to make more sense but i dont know if this can be implemented in CUDA or not

**Assistant:** Yes — **partly**.  
Good intuition, but best is a **hybrid**:

- **Step 3 must still do hard geometric validity** (`in_front`, range limits, elevation FOV).  
  - That is physics/visibility, not fitting.
  - Especially elevation: current 2D rasterizer cannot enforce elevation clipping by itself.

- **Step 4 can do boundary-aware fitting** for the surviving points.  
  - This is where you handle partial support near azimuth/range image edges.
  - So your idea is good for the *fitting* part.

Practical recommendation:
- Step 3: compute all projected sigma points + validity flags.
- Step 4: fit `mu_2D, Sigma_2D` with boundary-aware weights (renormalize), and fallback if too few valid points.

On CUDA:  
- You **don’t need CUDA changes** for this policy if you still output one `T` matrix per surfel and use `transMat_precomp`.
- True truncated-footprint physics is not exactly representable by one Gaussian anyway; this remains an approximation.

So: boundary handling in Step 4 is viable and often better numerically, but Step 3 hard gating cannot be removed.

**User:** tell me for step 4, how are you going to do the fitting. explain at a high level (remember that i have a aerospace/ocean engineering background)

**Assistant:** Perfect framing — think of Step 4 as an **equivalent-moment fit** in measurement space, like UKF measurement covariance propagation.

At high level, I’d do this:

- **Measurement model**
  - Treat sonar pixel mapping as a nonlinear measurement function  
    `y = h(x) = [col(x), row(x)]`.
  - Step 3 already gave us projected sigma points `y_i` from the surfel patch.

- **What we fit**
  - We are not fitting new geometry.
  - We fit an **equivalent 2D Gaussian footprint** that preserves:
    1) centroid (first moment), and  
    2) spread/orientation (second moment).

- **Moment match (UKF style)**
  - Mean:
    `mu_2D = Σ w_i y_i`
  - Covariance:
    `Sigma_2D = Σ w_i (y_i - mu_2D)(y_i - mu_2D)^T`
  - This is exactly the “predicted measurement covariance” idea from navigation filters.

- **Boundary-aware variant**
  - If some sigma points are invalid/out-of-FOV, use only valid ones and renormalize weights.
  - If too few remain (e.g., `<3`), use a deterministic fallback (center/Jacobian footprint).
  - This avoids edge-clamp bias.

- **Numerical conditioning**
  - Symmetrize `Sigma_2D`.
  - Eigenvalue floor (and optional ceiling) so it stays PSD and stable.
  - This is like keeping an inertia/covariance tensor physically well-posed.

- **Rasterizer handoff**
  - Convert `mu_2D, Sigma_2D` into the rasterizer’s transform matrix `T` (via factorization, e.g., Cholesky/eigendecomp path).
  - Rasterizer then does Gaussian evaluation + compositing.

Engineering intuition:
- You can view the projected surfel as a warped “energy blob” on the sonar image.
- Step 4 replaces that blob with an ellipse having the same center of energy and second moments (analogous to matching mass center + inertia tensor).
- It is accurate when warp is mild; it loses higher-order shape effects (skew/multimodality).

**User:** is your boundary-award variant method good? what issues do you forsee?

**Assistant:** Short answer: **yes, it’s a good practical method**, but it has known failure modes.

- It’s good because it is deterministic, easy to implement, and usually stable for training.
- Main weakness: it approximates a **truncated/warped** footprint with one full Gaussian, so edge behavior is never exact.

Issues I’d expect:

- **Gradient discontinuities at boundaries**: hard valid/invalid masking causes abrupt changes when sigma points cross FOV limits.
- **Inward bias near edges**: dropping out-of-FOV points + renormalizing tends to pull centroid inward and shrink covariance.
- **Fallback flicker**: switching to fallback when valid-point count drops (e.g., `<3`) can create frame-to-frame jitter in gradients.
- **Ill-conditioned covariance**: with too few/near-collinear valid points, fitted `Sigma_2D` becomes unstable or overly anisotropic.
- **Model mismatch**: fitted Gaussian is untruncated, but actual support near edges is effectively truncated, so rendered tails can be wrong.

What I’d do to control this:

- Use **soft border weights** (smooth taper) before hard exclusion to reduce gradient jumps.
- Keep strict PSD conditioning (`eig floor/ceil`, symmetrize).
- Add **hysteresis** around fallback threshold to reduce mode switching.
- Log boundary stats (`K` histogram, fallback fraction, edge-hit counts) and gate them.
- Add edge-focused contract tests (your `RB-T15` should include these).

So: good baseline, but only if you add these guardrails.

**User:** im worried about the edge behaviour, any suggestions?

**Assistant:** Yes — edge behavior is the hardest part. I’d use a layered policy:

- **1) Define an edge band and treat it differently**
  - Compute distance-to-image-boundary for each surfel footprint center.
  - If far from edges: normal sigma-point fit.
  - If in edge band (e.g. `< 6 px`): use edge-safe fitting path.

- **2) Replace hard valid/invalid jumps with soft gating near boundary**
  - Weight each sigma point by a smooth boundary window (sigmoid/taper) based on margin to FOV/image limits.
  - Keep hard reject only when clearly out (`in_front` false, beyond physical range/elevation).
  - This removes gradient discontinuities.

- **3) Add hysteresis for fallback**
  - If valid support is weak, fallback to Jacobian/center footprint.
  - Use enter/exit thresholds (e.g. switch to fallback at `<3` valid points, leave fallback at `>=4`) to avoid flicker.

- **4) Condition covariance aggressively**
  - Symmetrize `Sigma_2D`, eigenvalue floor/ceiling, cap condition number.
  - This prevents skinny/exploding ellipses at edges.

- **5) Loss-side guardrail (very useful)**
  - Downweight or mask a thin outer image border (e.g. 2–4 px) during training loss.
  - Prevents edge artifacts from dominating optimization.

If you want the **highest fidelity** near edges, add:
- **Edge-only micro-splat mode**: when footprint intersects border, split surfel into a few sub-splats and project exactly, then compose.  
This is slower but much more stable/accurate than forcing one Gaussian to represent a clipped shape.

For now, I’d start with 1–5 (cheap, no major CUDA surgery), then add micro-splat only if RB-T15 edge fixtures still fail.

**User:** explain to me why is it bad to just truncate the 2d gaussian in step 4 if it is on the boundary

**Assistant:** Great question. It’s not “bad” physically — clipping at the sensor boundary is actually realistic.  
What’s bad is doing it **naively in this pipeline**.

Why naive truncation is problematic:

- **Representation mismatch:** Step 4 currently reduces footprint to `(mu_2D, Sigma_2D)`. That represents a full ellipse/Gaussian only. A truncated Gaussian needs extra shape info (clip boundary), which `T` alone cannot encode.
- **Rasterizer contract mismatch:** The CUDA path assumes analytic full Gaussian evaluation (`exp(-0.5 rho)`) over support. It has no native “half-ellipse only” mask unless we modify kernels.
- **Normalization ambiguity:** After truncation, do you preserve peak intensity or total reflected energy?  
  - Preserve peak -> less total energy near edges.  
  - Preserve total energy -> peak inflates near edges.  
  Both change training behavior.
- **Non-smooth gradients:** Hard clipping creates set changes (a sigma point/pixel suddenly in/out), giving boundary gradient jumps and optimizer jitter.
- **Bias near edges:** Truncated support shifts moments inward, so fitted center/covariance can “pull” geometry away from boundary if not handled carefully.

So the issue is not “truncate = wrong physics”; it’s “truncate requires a different rendering contract than single Gaussian + existing rasterizer.”

If you want edge-physical behavior without big CUDA rewrite, best compromise is:
1) hard physical gating (front/range/elevation),  
2) smooth boundary weighting in Step 4,  
3) robust fallback near edges,  
4) optional border loss mask.  

That gets most of the benefit with fewer instability traps.

---

## Appendix -- 2026-03-05 Cross-Plan Gap Audit (codex)

Status: **Not lock-ready yet**. The plan is close, but the following items should be resolved before execution lock.

### Critical

1. **Test-ID namespace collision (`C4-*` vs `RB-*`)**
   - `C4-T11`/`C4-T17` are reused in this plan with meanings that differ from Chunk-4 plan semantics.
   - Why it matters: gate ledgers become ambiguous and cross-run comparators can mix incompatible evidence.
   - Required fix: reserve renderer-remediation checks to `RB-*` IDs only, and add a small mapping note: `RB-T13` supersedes v2 compatibility checks; `C4-T17` remains historical Chunk-4 artifact review.

2. **Execution precedence conflict with upstream flow plan**
   - Upstream flow currently says to close Chunk-3 parity first, while this plan says renderer correction is prerequisite.
   - Why it matters: scheduling ambiguity can produce invalid post-fix comparisons.
   - Required fix: add explicit precedence rule that renderer-v2 remediation gate runs before any new Chunk-3/4 evidence claims; pre-v2 Chunk-3/4 claims are historical-only.

3. **Source-of-truth scope conflict**
   - Upstream implementation plan still declares the detailed elevation plan as sole authority.
   - Why it matters: ownership ambiguity for renderer semantics causes review/implementation drift.
   - Required fix: add explicit scope override: for renderer semantics, this v2 renderer-fix plan is authoritative until upstream patches land; detailed elevation plan remains authoritative for non-renderer elevation logic.

4. **Scientific model still documents pre-v2 renderer behavior**
   - Existing equations still encode legacy bilinear/no-occlusion assumptions and stale extrinsic values.
   - Why it matters: post-v2 evidence interpretation is mathematically inconsistent across project docs.
   - Required fix: patch `plans/scientific_progress.md` with a renderer-semantics-version split (pre-v2 historical vs v2 active), v2 forward model equations, and canonical extrinsic `[0.0, -0.10, -0.08]` with +5 deg pitch.

### Important

1. **R5 patch list is missing one upstream document**
   - `plans/PLAN_TRAINING_PIPELINE_OVERVIEW_2026-01-28.md` still presents bilinear legacy behavior as active.
   - Required fix: include that file in R5 with a v2 addendum and historical banner for pre-v2 renderer behavior.

2. **Synthetic metadata contract is not yet reflected in the guide**
   - `docs/SYNTHETIC_DATASET_GUIDE.md` currently captures commands/thresholds but not mandatory renderer fingerprint fields or refresh triggers.
   - Required fix: add a mandatory gate-metadata block and refresh-trigger policy aligned with R6.

3. **Progress evidence partition not yet explicit**
   - `plans/progress_overview.md` still mixes historical and active evidence streams.
   - Required fix: split into `Pre-v2 Historical Evidence` and `Post-v2 Active Evidence`; disallow cross-group deltas unless renderer fingerprints match.

4. **`compat_reference_id` is named but not operationalized**
   - Required by `RB-T13`, but generation/storage/tolerance procedures are not specified.
   - Required fix: define compatibility reference creation protocol (commit+config fingerprint), artifact path contract, and tolerance source-of-truth.

### Minor

1. **Large verbatim transcript appendix in active execution plan**
   - Why it matters: increases review overhead and leaves room for precedence confusion.
   - Suggested fix: move transcript to archived note; keep short resolved-decisions changelog in this plan.

2. **Baseline implementation-state note (optional but useful)**
   - Current repo baseline does not yet include `tests/test_renderer_baseline_*.py` or v2 env flags (`SONAR_RENDER_MODE`, `SONAR_OCCLUSION_MODE`).
   - Suggested fix: add one-line execution preface that R0 starts from zero for renderer-v2 test/flag scaffolding.

Execution-readiness condition from this audit: resolve all Critical items first; then this plan is ready to lock.

---

## Appendix — 2026-03-05 Codebase & CUDA Rasterizer Verification Audit (claude-opus-4-6)

Status: **All code-level defect claims verified accurate.** Three architectural gaps identified that must be resolved before execution lock.

### Verification Method

Three parallel research passes:
1. **Code claims** — verified all 8 defect claims (P0–P3) against actual source, line by line.
2. **Upstream plans** — checked all 7 upstream plans + `docs/SYNTHETIC_DATASET_GUIDE.md` for consistency.
3. **CUDA rasterizer interface** — verified `transMat_precomp` bypass, `T` matrix format, depth sorting, frustum culling, and return values.

### Code Claims: All Verified

| Claim | Line Match | Accuracy |
|-------|-----------|----------|
| P0 — Discarded normals (`gaussian_model.py:139`) | Exact | 100% — `pcd.normals` exists in `BasicPointCloud` but `create_from_pcd` never references it |
| P0 — Lambertian dead zone (`__init__.py:447`) | Exact | 100% — `clamp(min=0)` confirmed, zero gradient when `dot < 0` |
| P1 — No occlusion (`__init__.py:516-536`) | Exact | 100% — pure additive `scatter_add_`, no depth ordering |
| P2a — Point splatting (`__init__.py:482-536`) | Exact | 100% — fixed 4-pixel bilinear; `pc.get_scaling` never called; `scaling_modifier` dead code |
| P2b — Placeholder outputs (`__init__.py:599`) | Exact | 100% — `radii` explicit zeros; `viewspace_points` initialized but never populated |
| P3 — FOV pruning bug (`debug_multiframe.py:319-326`) | Exact | 100% — both `if`/`else` branches identical |
| P3 — Coordinate divergence (`sonar_utils.py:29` vs `point_utils.py:77`) | Exact | 100% — incompatible frames (+X=right/+Z=forward vs +X=forward/+Y=right) |
| P3 — Bilinear boundary (`__init__.py:483-486`) | Exact | 100% — `col_floor` unbounded; degenerate when `floor==ceil` |

### CUDA Rasterizer Verification

| Plan Claim | Status | Evidence |
|------------|--------|----------|
| `transMat_precomp` bypasses pinhole | **Confirmed** | `forward.cu:195-210`: null check skips `compute_transmat` entirely |
| `compute_transmat` at lines 75-118 | **Confirmed** | Exact location; contains `world2ndc` and `ndc2pix` (pinhole-specific) |
| T matrix is 3x3, 9 floats | **Confirmed** | `forward.cu:349-351`: Tu/Tv/Tw rows, 3 floats each |
| Downstream rendering is projection-agnostic | **Mostly confirmed** | Tile assignment, Gaussian eval, alpha compositing are generic |
| Depth sorting is camera-Z | **Confirmed, NOT configurable** | `forward.cu:249`: `depths[idx] = p_view.z` hardcoded |

### HIGH — Must resolve before execution lock

#### H1. Rendering architecture: who produces the final image?

The plan describes two systems that both produce images:
- **Ray-binned occlusion compositor** (R2a): produces `R[a,e,r]` → elevation marginalization → `I[a,r]`
- **2DGS rasterizer** (R2b): produces footprint-based rendered images via CUDA tile rendering

The plan never states how these combine. Possible architectures:

| Option | Description | Implication |
|--------|-------------|-------------|
| (a) Ray-binned is sole image producer | Rasterizer used only for footprint/gradient computation | Must implement custom differentiable compositor in PyTorch; rasterizer output discarded |
| (b) Rasterizer produces image; ray-binned pre-filters visibility | Ray-binned compositor determines which surfels are visible per ray; rasterizer renders visible set | Requires passing visibility mask to rasterizer; depth ordering mismatch (camera-Z vs sonar range) |
| (c) Hybrid | Ray-binned for occlusion decisions; rasterizer for footprint evaluation within visible set | Most complex; clearest physics |

**Required fix:** State which architecture is intended. This is the single most important missing piece — everything in R2a and R2b depends on it.

#### H2. CUDA rasterizer depth sorting hardcoded to camera-Z

The rasterizer sorts surfels by camera-space Z (`forward.cu:249`), not sonar range. For sonar, two surfels at the same range but different elevation angles have the same acoustic occlusion priority but different camera-Z values. The plan says "Provide sonar range as depth key" (R2 Step 5) but the rasterizer has no mechanism to accept a custom depth key.

**Required fix (choose one):**
- (a) Add `depths_precomp` parameter to the CUDA rasterizer (C++ modification in `rasterizer_impl.cu` and `forward.cu`)
- (b) Explicitly state that rasterizer depth ordering is **not authoritative** — only ray-binned occlusion governs visibility, and rasterizer compositing order is approximate/irrelevant
- (c) Encode sonar range into the Tw row of the T matrix so that rasterizer depth naturally becomes sonar range

If (b), explain what the rasterizer's alpha compositing output actually means and whether it's used.

#### H3. Frustum culling uses `projmatrix` — incompatible with sonar geometry

When `transMat_precomp` bypasses pinhole projection, the rasterizer's `preprocessCUDA` still uses `projmatrix` for frustum culling (`auxiliary.h:187-212`). Surfels valid in sonar space could be incorrectly culled by the pinhole frustum, or surfels outside sonar FOV could pass the pinhole frustum.

**Required fix (choose one):**
- (a) Construct a synthetic `projmatrix` that conservatively contains the sonar FOV (wide frustum that never culls valid surfels)
- (b) Pre-filter surfels on the Python side (sonar FOV check) and pass only valid surfels to the rasterizer, with a synthetic `projmatrix` wide enough to never cull
- (c) Modify the rasterizer to accept a `visible_precomp` mask that overrides frustum culling

### MEDIUM — Should address

#### M1. `sonar_project_points()` not in R4 files-touched

`render_sonar` calls `sonar_project_points()` at `__init__.py:411` for forward projection. The coordinate convention fix (R4) must cover this function. Add to R4 scope.

#### M2. Rasterizer normals are dummy under `transMat_precomp`

`forward.cu:209` sets `normal = make_float3(0.0, 0.0, 1.0)` when precomp is used. The plan computes normals in PyTorch (`__init__.py:436-438`), which is correct. But any downstream code consuming rasterizer-returned normals (e.g., normal consistency loss in `render_sonar` output at line 569) will silently receive dummy values. Document this and ensure rasterizer normals are overridden.

#### M3. Elevation binning connection to existing code

`render_sonar` already computes elevation at line 421 (`elevation = torch.atan2(down, horiz_dist)`). R2a proposes elevation-binned ray-space occlusion but doesn't specify whether this existing computation is reused or recomputed. Clarify to avoid redundant or inconsistent calculations.

#### M4. `viewspace_points` semantics undefined for sonar

R3 says "verify `viewspace_points`/`radii` carry meaningful values" but doesn't define what "meaningful" means for sonar. For pinhole cameras, these are screen-space (x,y) coordinates and pixel radii used by densification. For sonar, should `viewspace_points` be (azimuth_pixel, range_pixel)? Define the contract so Chunk 5 densification can consume them.

#### M5. `compute_fov_margin()` not mentioned

`compute_fov_margin()` (`gaussian_renderer/__init__.py`) is used by `debug_multiframe.py` for size-aware FOV pruning. Under the new projection model, does this function need updating? Not listed in any R-step.

### LOW — Informational

#### L1. Near-plane coupling

Rasterizer hardcodes `near_n = 0.2` (`auxiliary.h:37`). Sonar `range_min` is also 0.2. Works by coincidence. Document as a known assumption.

#### L2. Channel count mismatch

Current renderer outputs 1-channel expanded to 3 via `.expand(3, -1, -1)` at `__init__.py:563`. CUDA rasterizer natively outputs 3-channel RGB. Plan should specify the channel convention for the new path.

#### L3. Bilinear boundary bug slightly overstated

Lines 433-434 pre-clamp `col`/`row` to `[0, W-1]` before bilinear splatting, so `col_floor` going negative requires `valid_col` exactly at 0.0. The primary real issue is `floor == ceil` degenerate splatting at boundaries. Moot since legacy path is being removed.

#### L4. Chunk-4 misattribution

The plan implies but doesn't explicitly state: "Chunk-4 NO-GO should be re-evaluated as potentially misattributed once renderer fixes land." The upstream Chunk-4 plan (`PLAN_ELEVATION_AWARE_CHUNK4_EXECUTION_2026-02-23.md`) attributed failure to coupling/support design, but the defects documented here show the renderer was the root cause. Making this explicit would strengthen the plan's rationale.

### Upstream Plans Status

All 7 upstream plans and `docs/SYNTHETIC_DATASET_GUIDE.md` were audited. None contain the content R5 says should be added:

| Upstream Document | Has Renderer Contract? | Has Renderer Gate? |
|-------------------|----------------------|-------------------|
| `PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md` | No | No |
| `PLAN_ELEVATION_AWARE_TRAINING_detailed_2026-02-01.md` | No | No |
| `PLAN_ELEVATION_AWARE_IMPLEMENTATION_EXECUTION_2026-02-10.md` | No | No |
| `PLAN_ELEVATION_AWARE_TRAINING_FLOW_2026-02-24.md` | No | No |
| `PLAN_ELEVATION_AWARE_CHUNK4_EXECUTION_2026-02-23.md` | Historical only (line 875+) | No |
| `plans/scientific_progress.md` | No (documents `clamp(min=0)` without noting dead zone) | No |
| `plans/progress_overview.md` | No | No |
| `docs/SYNTHETIC_DATASET_GUIDE.md` | No renderer fingerprint; no refresh-trigger policy | No |

This confirms R5 scope is real and necessary. No content duplication risk.

### Execution-readiness condition from this audit

Resolve H1–H3 (rendering architecture, depth sorting, frustum culling) before execution lock. M1–M5 should be addressed in the relevant R-step descriptions. L1–L4 are documentation items that can land with R5.

---

## Appendix — 2026-03-05 R0 Test Suite Review (claude-opus-4-6)

Status: **All plan test IDs covered. Several issues found — some critical, some moderate.**

### Coverage

Every test ID in the plan catalog (RB-T01 through RB-T22, smokes T11–T14, T17) has at least one test across 8 files. Multiple IDs have additional variant tests (T12, T14, T15, T17, T19, T21).

### Already-GREEN tests (not actually RED)

Several R1 features are already implemented. These tests pass against current code, contradicting the "R0 = write failing tests" premise:

| Test | Why it passes now |
|------|-------------------|
| **T01** | `_normals_to_quaternions` exists at `scene/gaussian_model.py:26`; `create_from_pcd:166-177` already uses it |
| **T02** | Same — deterministic init already in place |
| **T03** | `compute_sonar_lambertian` exists at `gaussian_renderer/__init__.py:299` with leaky mode |
| **T04** | Same function, clamp0 mode |
| **T22** | `resolve_sonar_render_contract:324` already rejects `legacy` |

Not a blocking problem (the tests are valid regression coverage), but these are not gating TDD red-light tests — the implementation preceded the test authoring.

### Critical Issues

#### C1. Compositing model is implicit — could trip up implementers

T05, T06, and T16 encode a specific compositing formula where `event_return = value * transmittance` — alpha only reduces transmittance, does NOT scale the event return. This is **not the standard alpha compositing** formula (`alpha * value * transmittance`).

Verification:

- **T05** expects `event_returns[1] == 0.3` with alpha=0.7, value=1.0, one prior event at alpha=0.7:
  - Standard compositing: `0.7 * 1.0 * 0.3 = 0.21` — would FAIL
  - Tests' model: `1.0 * 0.3 = 0.3` — passes

- **T16** expects `ray_returns[0] == 1.08`:
  - Standard compositing: `0.6*1.0 + 0.6*0.2*0.4 = 0.648` — would FAIL
  - Tests' model: `1.0*1.0 + 0.2*0.4 = 1.08` — passes

The plan does not precisely specify this formula. An implementer using standard alpha compositing will get failing tests with no explanation.

**Required fix:** Add a compositing contract docstring/comment at the top of `test_renderer_baseline_r0_red_r2_contracts.py` specifying the exact formula, or add the formula to the plan's R2a section.

#### C2. T08 (densification) will likely NameError at runtime

`_load_renderer_fn` in `test_renderer_baseline_r0_red_densification_contracts.py` loads all function/class definitions from the renderer and `exec`s them. The mocked globals are sparse:

```python
ns = {"torch", "math", "os", "dataclass", "F", "GaussianModel",
      "get_scaled_world_to_view_transform", "sonar_ranges_to_points",
      "sonar_points_to_normals"}
```

But `render_sonar` uses many **imported** names at runtime (`SurfelRasterizationSettings`, `SurfelRasterizer`, `getWorld2View2`, etc.) that are not functions/classes defined in the file — they're module-level imports. The AST extraction skips imports, so these will be missing from the exec namespace.

The test will fail with `NameError` (missing imported dependency), not with a meaningful assertion failure about densification signals.

**Required fix:** Either (a) add the missing imported symbols to the mock namespace, (b) restructure the test to call only the specific sub-functions that produce `radii`/`viewspace_points`, or (c) accept this as a known limitation and document it.

#### C3. T09 expected results depend on function internals beyond the mask logic

The test mocks `apply_row_prune_with_chunk4_state` to return 0 and `ensure_chunk4_state_capacity` to no-op. The expected `pruned_all == 3` assumes the function returns a count that comes directly from `prune_points` call accounting. If the function's return type or internal accounting changes (e.g., returns a dict, or includes row-prune counts), the test breaks for structural reasons rather than contract reasons.

**Recommended:** Add a comment documenting the assumed return-value contract.

### Moderate Issues

#### M1. T20 inconsistently skips `_call_with_supported_kwargs`

T05, T06, T16 all call `compose_ray_binned_occlusion` via `_call_with_supported_kwargs` (lenient about extra kwargs), but T20 calls it directly with a `surfel_ids=` kwarg. If the implementation ignores surfel IDs (since ray isolation is already guaranteed by `ray_ids`), T20 fails with `TypeError` while the others pass.

**Required fix:** Either all tests for `compose_ray_binned_occlusion` should use `_call_with_supported_kwargs`, or all should call directly. Recommend making T20 consistent with T05/T06/T16.

#### M2. Inconsistent `sonar_config` handling across `project_sonar_footprint` tests

- T07 (`remaining_contracts.py`): **does not pass `sonar_config` at all**
- T15 (`remaining_contracts.py`): passes a real `SonarConfig` object (via `_load_sonar_utils_module()`)
- T15 (`nonlinear_boundary_contracts.py`): passes a plain **dict** from `_dummy_sonar_cfg()`

This means the implementation must accept a `SonarConfig` object, a dict, AND treat `sonar_config` as optional — all simultaneously. The tests pull in different directions on the API contract.

**Required fix:** Standardize on one convention. Recommend always passing a `SonarConfig` object (or always a dict with documented keys), and making it required.

#### M3. T13 default occlusion mode will remain RED throughout R1

T13 asserts the default `occlusion_mode` is `"ray_binned"`, but the current default is `"none"` (`resolve_sonar_render_contract:329`). Per the execution order, R2a changes this default. T13 will be RED throughout R1 implementation — expected behavior, but potentially confusing if someone runs the full test suite during R1 and sees unexpected failures.

**Recommended:** Add a comment in T13 noting it gates R2a, not R1.

#### M4. Duplicated test coverage across files

T14 and T17 assertions appear in both `test_renderer_baseline_r0_red_runtime_metadata_contracts.py` and `test_renderer_baseline_smoke_contracts.py` with overlapping (but not identical) formulations. Adds maintenance burden and ambiguity about which is authoritative.

### Minor Issues

#### m1. T10 roundtrip tolerance is generous

0.10m tolerance on a 256x200 image with 2.8m range extent gives ~0.014m/pixel. The threshold allows ~7 pixels of error, which could mask real coordinate issues.

#### m2. T21 AST hardcoded-zero check is bypassable

The test checks for literal `0.0` constants in the AST. An implementer could write `float(0)` or `0.0 + 0` and bypass the check. Reasonable heuristic but not airtight.

#### m3. T15 oracle setup guard could mislead

`assert oracle_pixels.shape[0] >= 20` is a setup guard in `test_renderer_baseline_r0_red_remaining_contracts.py:196`. If the surfel projects fewer than 20 valid samples, the test fails with "insufficient projected samples" — a misleading message unrelated to the actual contract.

### Summary

The test suite is structurally complete and the logic is mostly correct. The main risks for execution are:

1. **Document the compositing formula** (C1) — the tests encode it but it's not written anywhere
2. **T08 will likely NameError** (C2) — the renderer has too many imported dependencies for AST extraction to work cleanly
3. **Normalize the `sonar_config` API** (M2) across T07/T15 variants
4. **Make T20 consistent** (M1) with T05/T06/T16 on `_call_with_supported_kwargs`

The 5 already-GREEN tests (T01, T02, T03, T04, T22) are not a problem — they're valid regression coverage for already-landed R1 work.

---

## Appendix — 2026-03-05 R0 TDD Readiness Assessment (claude-opus-4-6)

Status: **16 of 17 unit/contract tests can be written now.** H1–H3 architectural gaps affect integration wiring, not unit contracts. TDD tests the contract.

### Per-test readiness

| Test | Contract | Blocked? | Writable now? | Notes |
|------|----------|----------|---------------|-------|
| `RB-T01` | Normal-init ingestion | No | **Yes** | Tests `create_from_pcd` consuming `pcd.normals` into quaternions |
| `RB-T02` | Init determinism | No | **Yes** | Pure quaternion-from-normal math under fixed seed |
| `RB-T03` | Lambertian gradient safety | No | **Yes** | Tests leaky vs clamp0 gradient flow on wrong-facing surfels |
| `RB-T04` | Compatibility parity (clamp0) | No | **Yes** | Tests clamp0 mode reproduces current `torch.clamp(min=0)` behavior |
| `RB-T05` | Ray-binned occlusion core | No | **Yes** | Per-ray front-to-back compositing contract is architecture-independent |
| `RB-T06` | No cross-ray occlusion | No | **Yes** | Different `(azimuth_bin, elevation_bin)` rays never occlude each other |
| `RB-T07` | Gaussian footprint from scale | Partially | **Mostly** | Can test `Sigma_2D` computation from surfel params; rasterizer integration deferred |
| `RB-T08` | Densification signals | **Yes (M4)** | **No** | `viewspace_points` semantics for sonar undefined — cannot write meaningful assertion |
| `RB-T09` | FOV prune semantics | No | **Yes** | Tests `require_all=True` uses `.all()` vs `require_all=False` uses `.any()` |
| `RB-T10` | Convention consistency | No | **Yes** | Forward-project then backward-project roundtrip within tolerance |
| `RB-T15` | Nonlinear footprint fidelity | No | **Yes** | Sigma-point math is self-contained; tests against brute-force projection oracle |
| `RB-T16` | Range-order occlusion | No | **Yes** | Tests sonar-range ordering (not camera-Z) on same `(azimuth_bin, elevation_bin)` ray |
| `RB-T18` | Elevation marginalization | No | **Yes** | Tests `I[a,r] = sum_e w_e * R[a,e,r]` within tolerance |
| `RB-T19` | `Sigma_2D -> T` conversion | No | **Yes** | T matrix format confirmed (9 floats, 3 rows); forward/backward consistency testable |
| `RB-T20` | Multi-bin participation | No | **Yes** | Per-ray contract; tests wide surfel contributing to multiple rays with independent occlusion |
| `RB-T21` | Support-cap mass accounting | No | **Yes** | Pure weight normalization and truncation mass-loss telemetry |
| `RB-T22` | Legacy mode removal | No | **Yes** | Tests `SONAR_RENDER_MODE=legacy` is rejected; no bilinear scatter path reachable |

### Why H1–H3 don't block unit tests

The three HIGH architectural gaps (H1: rendering architecture, H2: depth sort key, H3: frustum culling) affect how components **connect** — not what each component's input/output contract is:

- The occlusion compositor has a clear contract: given `(ray_id, range, alpha_share)` events, produce front-to-back composited returns per ray. Whether the compositor's output feeds the rasterizer or replaces it is an integration question.
- The footprint projection has a clear contract: given surfel params + sonar config, produce `(mu_2D, Sigma_2D)`. Whether the rasterizer or a custom renderer consumes it is an integration question.
- Normal init, Lambertian transfer, FOV pruning, coordinate conventions, and legacy removal are all self-contained contracts with no dependency on the final pipeline architecture.

### What is blocked

- **RB-T08** (densification signals): Requires defining what `viewspace_points` means for sonar (M4). Cannot write meaningful assertions until this is specified.
- **Smoke tests (RB-T11 through RB-T14, RB-T17)**: Require the full pipeline wired up. Not R0 scope — these come after R2b/R2-alt.

### Conclusion

R0 can proceed immediately with 16 tests. `RB-T08` should be deferred until M4 is resolved (define sonar `viewspace_points` semantics). Smoke tests are post-integration by design.

---

## Appendix — 2026-03-05 R0 Test Quality Audit (claude-opus-4-6)

Status: **R0 tests must be rewritten.** All 25 tests use source-code string/AST grep instead of behavioral contract testing. This violates TDD methodology and produces tests that are fragile, non-discriminating, and unable to gate implementation correctness.

### Audit Method

- Read all 5 test files (`tests/test_renderer_baseline_r0_red_*.py`, `tests/test_renderer_baseline_smoke_contracts.py`)
- Ran full suite: `pytest tests/test_renderer_baseline_*.py -v`
- Cross-referenced each test against its plan contract (Test Catalog, lines 590–620)
- Verified current source file state against test expectations

### Results: 12 failed, 11 passed, 2 skipped

### Finding 1 — Fundamental TDD violation: no behavioral tests

Every R0 "red" test uses one of two approaches:
- **AST analysis**: Parse source code with `ast` module and check for specific AST node patterns
- **String grep**: `assert "SOME_STRING" in source_text`

**Zero tests** construct inputs, call a function, or assert on outputs. No test file imports `torch`. The R0 TDD readiness assessment (this plan, line 1096) explicitly confirmed 16 tests are writable NOW as behavioral contract tests with synthetic tensor inputs. None were written that way.

### Finding 2 — 9 tests PASS prematurely (should be RED)

In TDD, R0 tests must fail against the pre-implementation codebase. The following 9 tests pass, meaning they cannot gate whether the implementation is correct:

| Test | Why it passes prematurely |
|------|--------------------------|
| `RB-T01` (normal init) | AST finds `pcd.normals` reference in `create_from_pcd` even though normals are not consumed for quaternion init |
| `RB-T02` (init determinism) | Checks `torch.rand` absent from AST — may already be absent without normals being used |
| `RB-T03` (Lambertian gradient) | `"SONAR_LAMBERTIAN_MODE"` string found in working-tree modifications |
| `RB-T04` (clamp0 parity) | Exact source string match found in working-tree modifications |
| `RB-T05/06` (occlusion) | `"SONAR_OCCLUSION_MODE"` + `"ray_binned"` strings found in working-tree modifications |
| `RB-T09` (FOV prune) | AST check passes — either bug already fixed or AST match too loose |
| `RB-T18` (elevation marginalization) | `"SONAR_ELEV_BINS"` string found in working-tree modifications |
| `RB-T22` (render mode, r2 file) | `"2dgs_nonlinear"` string found in working-tree modifications |
| `RB-T22` (legacy rejection, occ file) | Exact error message string found in working-tree modifications |

### Finding 3 — Per-test contract violations

| Test ID | Plan contract | What was written | What should be written |
|---------|--------------|-----------------|----------------------|
| **RB-T01** | `create_from_pcd` consumes normals into quaternions | AST check for `pcd.normals` reference | Create `GaussianModel` with known normals → verify quaternion output aligns with input normal direction |
| **RB-T02** | Quaternion init deterministic under fixed seed | Checks `torch.rand` absent from AST | Call `create_from_pcd` twice with same seed + normals → assert identical quaternion output |
| **RB-T03** | Wrong-facing surfels get non-zero gradient under leaky mode | Checks `"SONAR_LAMBERTIAN_MODE"` string exists | Create wrong-facing surfel tensor → compute leaky Lambertian → backprop → assert `grad != 0` on rotation |
| **RB-T04** | `clamp0` reproduces pre-leaky transfer behavior | Exact source string match | Compute Lambertian both ways on same tensor input → assert outputs match within tolerance |
| **RB-T05** | Closer surfel suppresses farther on same ray | String check for `"ray_binned"` | Two surfels on same `(az_bin, el_bin)` at different ranges → front-to-back compose → assert back is suppressed |
| **RB-T06** | Different rays don't occlude each other | Combined with T05 as single string check | Two surfels on different `(az_bin, el_bin)` → verify both contribute independently |
| **RB-T07** | Scale changes rendered spatial support | AST check for `pc.get_scaling` | Render same surfel at 2 scales → assert different pixel footprint size |
| **RB-T09** | `require_all=True/False` produce distinct results | AST check for `.all()` vs `.any()` | Construct geometry where `all` and `any` diverge → call `prune_outside_fov` both ways → assert different mask outputs |
| **RB-T10** | Forward/backward projection roundtrip within tolerance | Comment string matching in `point_utils.py` | Project 3D point → sonar image → back to 3D → assert Euclidean distance < tolerance |
| **RB-T15** | Sigma-point mode beats Jacobian for large/oblique surfels | Env var string check | Brute-force project surfel grid as oracle → compare sigma-point vs Jacobian footprint error |
| **RB-T16** | Occlusion uses sonar range, not camera-Z | String check for `"argsort"` | Two surfels where range order differs from Z order → verify range order governs occlusion |
| **RB-T18** | `I[a,r] = sum_e w_e * R[a,e,r]` | String check for env vars | Construct synthetic `R[a,e,r]` volume + weights → marginalize → compare to hand-computed `I[a,r]` |
| **RB-T19** | `Sigma_2D → T` roundtrip + gradient flow | String check for `"transMat_precomp"` | Known `Sigma_2D` → convert to T → reconstruct covariance → assert matches; verify `grad` flows through chain |
| **RB-T20** | Wide surfel contributes to multiple rays, occluded independently per ray | String check for `topk(` | Create wide surfel → verify events emitted on multiple ray bins with correct per-ray occlusion |
| **RB-T21** | Mass-loss telemetry from support capping | String check for `"quantile"` | Cap known weights via top-K → compute mass loss → verify summary stats match expected values |
| **RB-T22** | Legacy mode rejected | String check for error message | Set `SONAR_RENDER_MODE=legacy` → call render entry point → assert raises `ValueError` or equivalent |

### Finding 4 — Duplicate test coverage

RB-T20, RB-T21, and RB-T22 each appear in **two files** with different implementations:
- `test_renderer_baseline_r0_red_r2_contracts.py` — string checks for env vars
- `test_renderer_baseline_r0_red_occlusion_semantics_contracts.py` — slightly different string checks

These will conflict when one version passes and the other fails, creating ambiguous gate status.

### Finding 5 — RB-T08 written despite explicit deferral

The R0 TDD readiness assessment (this plan, line 1131) states: "RB-T08 (densification signals): Requires defining what `viewspace_points` means for sonar (M4). Cannot write meaningful assertions until this is specified." The test was written anyway as an AST check on the return dictionary of `render_sonar`.

### Finding 6 — Extreme fragility

Tests like RB-T04 check for exact source code strings:
```python
assert 'if lambertian_mode == "clamp0"' in renderer_source
assert "torch.clamp(dot_val, min=0.0)" in renderer_source
```

A variable rename (`dot_val` → `cos_theta`), whitespace change, or `0` vs `0.0` breaks these without any semantic difference. Multiple other tests share this fragility pattern.

### Required remediation

All R0 tests must be rewritten as **behavioral contract tests**:

1. Import the module under test (or define a standalone contract function when the module API doesn't exist yet)
2. Construct synthetic tensor inputs (surfel positions, normals, scales, sonar config)
3. Call the target function
4. Assert on numerical output behavior (values, shapes, gradient flow, ordering)

Tests for contracts where the implementation function doesn't exist yet (e.g., ray-binned compositor, sigma-point projection) should define the expected function signature and test against it. The implementation fills in the function body to make the test pass — that is standard TDD.

### Tests that can remain as-is (with caveats)

- `test_rb_t11_rb_t12_debug_script_compiles_for_smoke_contracts` — compile check is fine as a smoke gate
- Runtime smoke tests (`RB-T11`, `RB-T12` runtime variants) — correctly gated behind `RUN_RENDERER_BASELINE_RUNTIME_SMOKES=1`

All other tests require rewrite before R0 can be considered complete.

---

## Appendix — 2026-03-05 R0 Test Rewrite Review (claude-opus-4-6)

Status: **Major improvement. Three bugs remain, plus one housekeeping item.** After fixes, R0 is complete.

### What changed

All 25 source-string/AST-grep tests were replaced with behavioral contract tests. Every unit test now constructs synthetic tensor inputs, calls the function under test, and asserts on numerical outputs. The approach uses AST extraction + `exec` to isolate individual functions from CUDA-dependent modules — creative and practical for avoiding heavy import chains.

### Rewrite results: 12 failed, 5 passed, 2 skipped (19 tests total)

| Status | Tests | Notes |
|--------|-------|-------|
| FAIL (correct — function doesn't exist) | T05, T06, T07, T15, T16, T18, T19, T20, T21 | Proper TDD red: `pytest.fail("missing renderer callable")` |
| FAIL (wrong reason — mock bug) | T01, T02 | DummyModel incomplete — see Bug 1 |
| FAIL (wrong reason — assertion bug) | T09 | See Bug 2 |
| PASS (R1 already implemented) | T03, T04, T22 | `compute_sonar_lambertian` and `resolve_sonar_render_contract` exist in working tree |
| PASS (existing infrastructure) | T10 | Roundtrip works with current sonar_utils — correct |
| PASS (compile check) | T11/T12 compile | Expected |
| SKIP (env-gated runtime) | T11, T12 runtime | Correctly gated behind `RUN_RENDERER_BASELINE_RUNTIME_SMOKES=1` |
| Deferred (correct) | T08 | Per readiness assessment (M4 unresolved) |

### What's good

- **10 of 16 unit tests fail for the correct reason** — target function doesn't exist yet
- **No more duplicates** — each test ID appears exactly once
- **Compositing model is consistent across T05/T06/T16/T20** — all use `event_return = T * value` with `T *= (1 - alpha)`, producing verifiable numerical expectations
- **T16 correctly discriminates range-order vs camera-Z order** — range-sorted gives 1.08, Z-sorted would give 0.6
- **T19 tests both roundtrip fidelity AND gradient flow** through `Sigma_2D → T` conversion
- **T21 tests precise weight-cap normalization** with exact expected values for retained weights and mass-loss stats
- **T22 tests runtime rejection** via `pytest.raises(ValueError, match="legacy")`
- **T10 does real forward/backward roundtrip** using existing sonar_utils infrastructure
- **RB-T08 correctly deferred** per readiness assessment

### Bug 1 — T01/T02: DummyModel missing `get_xyz` property

`create_from_pcd` sets `self._xyz = nn.Parameter(...)` at `scene/gaussian_model.py:185`, then accesses `self.get_xyz.shape[0]` at line 191. `get_xyz` is a `@property` on the real `GaussianModel` class, but the test's `DummyModel` doesn't implement it. Both T01 and T02 fail with `AttributeError: 'DummyModel' object has no attribute 'get_xyz'` before the actual contract assertion is reached.

**Fix:** Add to DummyModel:
```python
@property
def get_xyz(self):
    return self._xyz
```

### Bug 2 — T09: Wrong assertion when nothing needs pruning

The `require_all=False` path correctly identifies that all 3 surfels are visible from at least one camera, so `num_to_prune = 0`. The function then skips `prune_points` entirely (`debug_multiframe.py:330`: `if num_to_prune > 0`), which is correct behavior. But the test asserts:

```python
assert torch.equal(g_any.last_prune_mask, torch.tensor([False, False, False]))
```

Since `prune_points` was never called, `g_any.last_prune_mask` is `None`, causing `TypeError`. The first three assertions all pass correctly (`pruned_all == 3`, `pruned_any == 0`, `g_all.last_prune_mask == [True, True, True]`).

**Fix:** Replace the final assertion with:
```python
assert g_any.last_prune_mask is None, "RB-T09: no prune_points call expected when nothing to prune"
```

### Bug 3 — T15: Oracle doesn't use sonar projection

The plan contract for RB-T15 is: "2dgs_nonlinear better matches brute-force nonlinear projection oracle than Jacobian mode for large/oblique surfels." The oracle should project points through the actual nonlinear sonar projection to produce a reference 2D covariance in image space. Instead, the test computes a sample covariance in world XY space:

```python
pts = local * scale_xy.cpu().numpy()[None, :] + mean_3d[:2].cpu().numpy()[None, :]
oracle_cov = _oracle_covariance_from_points(pts)
```

This covariance is in meters², not pixels². It doesn't go through `atan2` / `sqrt` / pixel mapping. Comparing `sigma_2d` (image-space, pixels²) against this oracle is dimensionally wrong and will produce meaningless pass/fail results.

**Fix:** The oracle must:
1. Generate a dense grid of 3D points on the surfel disc (using quaternion and scale to parameterize the disc in world coordinates)
2. Project each point through the full nonlinear sonar projection: `azimuth = -atan2(x, z)`, `range = sqrt(x² + y² + z²)`, then to pixel coordinates via the sonar config
3. Compute sample covariance of the projected 2D pixel positions
4. Compare both `2dgs` and `2dgs_nonlinear` footprints against this pixel-space oracle

### Housekeeping — Smoke tests T12 (active diagnostics), T13, T14, T17 absent

The v1 string-check versions of these tests were correctly removed, and the plan categorizes them as post-integration. However, having no trace of these test IDs in the test files makes the catalog harder to audit. Adding `pytest.skip("post-integration: requires full pipeline wired up")` placeholder functions for T13, T14, and T17 would keep the test catalog traceable without implying they should pass now.

### Conclusion

After fixing Bugs 1–3 and optionally adding smoke placeholders, the R0 test suite is complete. The rewrite addresses all six findings from the prior audit: behavioral testing, correct red/fail status, no duplicates, T08 deferred, and no fragile string matching.
