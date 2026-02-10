# Plan: Elevation-Aware Training — Detailed Implementation (opus4.5)

**Date:** 2026-02-02
**Base plan:** `PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md`
**Git commit:** 61a6fc6 (branch `debug-multiframe-r2`)
**Status:** Detailed design (not yet implemented)
**Author:** opus4.5

---

## Summary

This plan describes the concrete code changes needed to implement elevation-aware
sonar training.  The approach follows the three-stage strategy from the base plan:

1. **Stage E1** — Randomise surfel elevation at initialisation + arc-constraint loss
2. **Stage E2** — Discrete elevation bins with temperature-annealed multi-view likelihood
3. **Stage E3** — (Optional) Elevation-guided densification

These "elevation stages" sit *inside* the existing curriculum (scale → surfels → joint).
They affect Stage 2 and Stage 3 of the existing curriculum only, because Stage 1 is
currently disabled (`STAGE1_ITERATIONS=0`) and only learns the scale factor.

---

## Guiding Principles

- Every change is additive; existing non-elevation paths must still work when
  elevation mode is off (`ELEVATION_BINS=0` or `ELEVATION_MODE=False`).
- Keep K small (5–9 bins) for development; make it configurable.
- Prefer discrete bins over stochastic sampling for debuggability.
- Minimise VRAM impact: 8 GB laptop target, 500 frames max.

---

## 1. New Module: `utils/elevation.py`

Create a dedicated module for elevation logic.  Keeping it separate avoids
polluting sonar_utils.py with a large amount of new code and makes it easier
for multiple reviewers to diff.

### 1.1 `ElevationBins` class

Manages the discrete elevation bin centres and per-pixel log-weights.

```python
class ElevationBins:
    """Discrete elevation distribution for backward projection."""

    def __init__(self, K, half_elev_rad, device="cuda"):
        # K evenly-spaced bin centres in [-half_elev_rad, +half_elev_rad]
        self.centres = torch.linspace(-half_elev_rad, half_elev_rad, K, device=device)
        # Temperature for softmax (annealed during training)
        self.temperature = 1.0

    def get_probs(self, logits):
        """logits: [..., K] -> probs [..., K] via temperature-scaled softmax."""
        return torch.softmax(logits / self.temperature, dim=-1)
```

### 1.2 `back_project_with_elevation()`

A differentiable PyTorch version of `sonar_frame_to_points()` that produces
one 3D point *per bin* per valid pixel, plus weighted combination.

```python
def back_project_with_elevation(
    azimuth,          # [P]  azimuth angles for P valid pixels
    range_vals,       # [P]  metric range values
    elevation_bins,   # ElevationBins (K centres)
    elevation_logits, # [P, K]  learnable logits
    R_c2w,            # [3,3]
    cam_centre,       # [3]
    scale_factor,     # scalar
):
    """
    Returns
    -------
    points_world : [P, 3]
        Weighted (soft) 3D positions for each pixel using the elevation
        distribution.
    arc_points   : [P, K, 3]
        Explicit per-bin 3D positions (world frame). Used for multi-view
        arc-intersection loss.
    probs        : [P, K]
        Elevation probabilities.
    """
    K = elevation_bins.centres.shape[0]
    probs = elevation_bins.get_probs(elevation_logits)  # [P, K]

    # Expand azimuth/range to [P, K]
    az = azimuth[:, None].expand(-1, K)
    r  = range_vals[:, None].expand(-1, K)
    el = elevation_bins.centres[None, :].expand(P, -1)  # [P, K]

    # Polar → camera-frame Cartesian  (camera: +X right, +Y down, +Z forward)
    horiz  = r * torch.cos(el)
    x_cam  = -horiz * torch.sin(az)          # lateral
    y_cam  = r * torch.sin(el)               # elevation (down = +Y)
    z_cam  = horiz * torch.cos(az)           # forward

    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [P, K, 3]

    # Transform to world
    # point_world = R_c2w @ point_cam + camera_centre_metric
    # then divide by scale_factor to get COLMAP scale
    pts_world_metric = torch.einsum("ij,pkj->pki", R_c2w, pts_cam) + cam_centre
    arc_points = pts_world_metric / scale_factor        # [P, K, 3]

    # Soft weighted position
    points_world = (probs[..., None] * arc_points).sum(dim=1)  # [P, 3]

    return points_world, arc_points, probs
```

### 1.3 `arc_intersection_loss()`

Given a surfel position visible in two frames, compute how close it lies to the
intersection of the two elevation arcs.

```python
def arc_intersection_loss(
    surfel_xyz,       # [M, 3]  surfel positions (world, COLMAP scale)
    arc_points_A,     # [M, K, 3]  arc from frame A
    arc_points_B,     # [M, K, 3]  arc from frame B
    probs_A,          # [M, K]
    probs_B,          # [M, K]
):
    """
    Soft intersection: for each surfel pair, compute weighted distance between
    the best-matching bins on each arc.

    Returns scalar loss.
    """
    # Pairwise distance between all bins of A and B: [M, K, K]
    diff = arc_points_A[:, :, None, :] - arc_points_B[:, None, :, :]  # [M, K_A, K_B, 3]
    dist2 = (diff ** 2).sum(dim=-1)                                   # [M, K_A, K_B]

    # Weight by joint probability
    joint_prob = probs_A[:, :, None] * probs_B[:, None, :]            # [M, K_A, K_B]
    loss = (joint_prob * dist2).sum(dim=(1, 2)).mean()                # scalar

    return loss
```

### 1.4 `multi_view_likelihood()`

Given a pixel in frame A, back-project along the elevation arc, forward-project
each bin to frame B, and score against frame B's ground truth.

```python
def multi_view_likelihood(
    arc_points,      # [P, K, 3]  world-frame arc from frame A
    gt_image_B,      # [1, H, W]  ground truth of frame B
    camera_B,        # camera object for frame B
    sonar_config,
    scale_factor,
):
    """
    Returns likelihood [P, K]: how well each elevation bin of frame A's
    pixels explains the observation in frame B.
    """
    P, K, _ = arc_points.shape

    # Forward-project each arc point to frame B's pixel space
    # (reuse the same polar→pixel math from render_sonar lines 278-328)
    pts_flat = arc_points.reshape(P * K, 3)
    # ... transform to frame B sonar coords ...
    # ... compute col_B, row_B ...
    # ... bilinear sample gt_image_B at (col_B, row_B) ...

    intensities = F.grid_sample(...)  # [P*K] → reshape [P, K]
    return intensities  # higher = more consistent
```

---

## 2. Changes to `utils/sonar_utils.py`

### 2.1 `sonar_frame_to_points()` — add `elevation_samples` parameter

**Location:** `sonar_utils.py:323-415`

Current line 388 is `y_cam = np.zeros_like(range_vals_metric)`.

Change: add an optional `elevation_angles` argument (NumPy array, shape `[P]` or
scalar).  Default `None` preserves old behaviour (elevation = 0).

```python
def sonar_frame_to_points(camera, sonar_config, ..., elevation_angles=None):
    ...
    if elevation_angles is None:
        y_cam = np.zeros_like(range_vals_metric)
    else:
        horiz = range_vals_metric * np.cos(elevation_angles)
        x_cam = -horiz * np.sin(azimuth)
        y_cam = range_vals_metric * np.sin(elevation_angles)
        z_cam = horiz * np.cos(azimuth)
    ...
```

This is used at **initialisation** (see Section 4.1).

### 2.2 `SonarConfig` — add elevation bin info (convenience)

Add `K` (number of bins) and `elevation_centres` as optional cached tensors
so downstream code can reference `sonar_config.elevation_centres` directly.

```python
# In SonarConfig.__init__(), after half_elevation_rad:
self.elevation_K = 0  # 0 = elevation-unaware (legacy)

def init_elevation_bins(self, K):
    self.elevation_K = K
    self.elevation_centres = torch.linspace(
        -self.half_elevation_rad, self.half_elevation_rad, K, device=self.device
    )
```

---

## 3. Changes to `utils/point_utils.py`

### 3.1 `sonar_ranges_to_points()` — elevation parameter

**Location:** `point_utils.py:64-146`, specifically line 115:
`z_s = torch.zeros_like(r)`.

Same pattern as Section 2.1 but for the PyTorch/differentiable path.
Add optional `elevation` tensor `[H, W]`.  If `None`, keep
`z_s = torch.zeros_like(r)`.  Otherwise:

```python
if elevation is not None:
    horiz = r * torch.cos(elevation)
    x_s = horiz * torch.cos(azimuth)
    y_s = -horiz * torch.sin(azimuth)
    z_s = r * torch.sin(elevation)   # now non-zero
```

This affects `depth_to_normal()` (line 30-57) which calls `sonar_ranges_to_points()`.
The caller in `debug_multiframe.py` that computes rendered normals would pass
`elevation=None` for now (normals don't need elevation accuracy to be useful).

---

## 4. Changes to `debug_multiframe.py`

This is the largest set of changes.  All changes are gated behind a new
environment variable `ELEVATION_MODE` (default `False`).

### 4.1 Environment variables

Add near the existing env-var block (around line 730):

```python
ELEVATION_MODE     = env_bool("ELEVATION_MODE", False)
ELEVATION_K        = env_int("ELEVATION_K", 7)          # number of bins
ELEVATION_TEMP_START = env_float("ELEVATION_TEMP_START", 2.0)
ELEVATION_TEMP_END   = env_float("ELEVATION_TEMP_END", 0.1)
ELEVATION_ARC_WEIGHT = env_float("ELEVATION_ARC_WEIGHT", 0.1)  # arc loss weight
ELEVATION_INIT_RANDOM = env_bool("ELEVATION_INIT_RANDOM", True)
```

### 4.2 Initialisation — randomise elevation

**Location:** `debug_multiframe.py:895-920` (the backward-projection loop)

Currently calls `sonar_frame_to_points()` which assumes elevation = 0.

When `ELEVATION_MODE` and `ELEVATION_INIT_RANDOM` are true, sample a random
elevation per pixel from `Uniform(-half_el_rad, +half_el_rad)`:

```python
if ELEVATION_MODE and ELEVATION_INIT_RANDOM:
    half_el = math.radians(sonar_config.elevation_fov / 2)
    n_valid = len(rows)   # rows/cols from the thresholding mask
    elevation_angles = np.random.uniform(-half_el, half_el, size=n_valid)
else:
    elevation_angles = None

points, colors = sonar_frame_to_points(
    cam, sonar_config,
    intensity_threshold=INTENSITY_THRESHOLD / 255.0,
    mask_top_rows=10,
    scale_factor=temp_scale_factor.get_scale_value(),
    elevation_angles=elevation_angles,
)
```

This spreads initial surfels across the full elevation fan rather than
concentrating them on the elevation = 0 plane.

### 4.3 Per-frame elevation logits

Create a dictionary of learnable elevation logits, one tensor per training frame.
Only allocate for pixels that pass the intensity threshold.

```python
if ELEVATION_MODE:
    from utils.elevation import ElevationBins, back_project_with_elevation

    elev_bins = ElevationBins(ELEVATION_K, sonar_config.half_elevation_rad)

    # Per-frame logits: dict[frame_idx] → tensor [P_i, K]
    # P_i = number of valid pixels in frame i
    elevation_logits = {}
    for i, cam in enumerate(training_frames):
        mask = preprocess_gt_image(cam.original_image).squeeze(0) > (INTENSITY_THRESHOLD / 255.0)
        mask[:10, :] = False  # top-row mask
        n_valid = mask.sum().item()
        logits = torch.zeros(n_valid, ELEVATION_K, device="cuda", requires_grad=True)
        elevation_logits[i] = logits

    # Single optimizer for all logits
    elev_optimizer = torch.optim.Adam(
        list(elevation_logits.values()),
        lr=1e-2,
    )
```

Memory estimate: 500 frames x ~8000 valid pixels x 7 bins x 4 bytes = ~112 MB.
Well within 8 GB budget.

### 4.4 Temperature annealing

At the start of each iteration, update the temperature based on training progress:

```python
if ELEVATION_MODE:
    progress = iteration / STAGE2_ITERATIONS  # 0..1
    elev_bins.temperature = ELEVATION_TEMP_START * (
        ELEVATION_TEMP_END / ELEVATION_TEMP_START
    ) ** progress
```

### 4.5 Multi-view arc-intersection loss

Added inside the Stage 2 training loop (`debug_multiframe.py:1253-1312`), after
the existing photometric loss computation.

**Strategy**: For each training iteration on frame A, pick one (or a few) other
frame(s) B that overlap with A.  Compute the arc-intersection loss between matched
surfels.

```python
if ELEVATION_MODE and iteration > ELEVATION_WARMUP_ITERS:
    # ---- Build arc for current frame A ----
    logits_A = elevation_logits[frame_idx]
    mask_A   = valid_pixel_mask[frame_idx]   # precomputed boolean [H, W]
    az_A     = sonar_config.azimuth_mesh[mask_A]   # [P_A]
    rng_A    = sonar_config.range_mesh[mask_A]     # [P_A]

    _, arc_pts_A, probs_A = back_project_with_elevation(
        az_A, rng_A, elev_bins, logits_A,
        R_c2w_A, cam_centre_A, scale_factor
    )

    # ---- Pick an overlapping frame B ----
    frame_idx_B = pick_overlapping_frame(frame_idx, training_frames, ...)
    logits_B = elevation_logits[frame_idx_B]
    # ... same arc computation for B ...

    # ---- Find surfel correspondences ----
    # Forward-project all surfels to both frames, find surfels visible in both
    xyz = gaussians.get_xyz
    in_fov_A = is_in_sonar_fov(xyz, training_frames[frame_idx], sonar_config, sonar_scale_factor)
    in_fov_B = is_in_sonar_fov(xyz, training_frames[frame_idx_B], sonar_config, sonar_scale_factor)
    both = in_fov_A & in_fov_B  # [N] mask

    if both.sum() > 0:
        # For each co-visible surfel, look up its pixel in A and B
        # via forward projection (same polar→pixel as render_sonar)
        # Then index into arc_pts_A/B at those pixel positions
        # Compute arc_intersection_loss(...)
        elev_loss = arc_intersection_loss(
            xyz[both], arc_pts_A_matched, arc_pts_B_matched,
            probs_A_matched, probs_B_matched
        )
        loss = loss + ELEVATION_ARC_WEIGHT * elev_loss

    # Step elevation optimizer
    elev_optimizer.step()
    elev_optimizer.zero_grad()
```

### 4.6 Multi-view intensity likelihood (supplementary signal)

For frame A's valid pixels, forward-project each elevation bin to frame B and
sample frame B's ground truth.  Use the sampled intensities as a likelihood to
directly supervise the elevation logits.

```python
if ELEVATION_MODE:
    likelihood_B = multi_view_likelihood(
        arc_pts_A, gt_image_B, training_frames[frame_idx_B],
        sonar_config, sonar_scale_factor
    )
    # Cross-entropy-like: encourage logits to match likelihood
    target_probs = (likelihood_B / (likelihood_B.sum(dim=-1, keepdim=True) + 1e-8))
    elev_ce_loss = -(target_probs.detach() * torch.log(probs_A + 1e-8)).sum(dim=-1).mean()
    loss = loss + ELEVATION_CE_WEIGHT * elev_ce_loss
```

### 4.7 Overlap lookup: `pick_overlapping_frame()`

Simple heuristic using camera positions: pick the frame whose camera centre is
closest to the current frame's centre but at a minimum angular separation (to
ensure different elevation perspectives).

```python
def pick_overlapping_frame(current_idx, frames, sonar_config, scale_factor,
                            min_angle_deg=10.0, rng=None):
    """Return index of a frame that overlaps well with current_idx."""
    # Compute cam centres
    # Compute pairwise angular separation (azimuth of cam-to-cam vector)
    # Filter by min_angle_deg
    # Among remaining, pick closest (or random if many)
    ...
```

This can be precomputed once at startup into an `overlap_table[i] → list[j]`.

### 4.8 Elevation-aware FOV pruning

**Location:** `prune_outside_fov()` at `debug_multiframe.py:225-279`.

No change needed: the existing pruning already checks elevation via
`is_in_sonar_fov()` which uses `atan2(down, horiz_dist)`.  Surfels that drift
to extreme elevations are already pruned.

### 4.9 Logging and diagnostics

Add to the periodic print and TensorBoard logging:

- `elev_temperature`: current temperature
- `elev_arc_loss`: arc-intersection loss value
- `elev_entropy`: mean entropy of elevation distributions (should decrease)
- Histogram of the "winning" (argmax) elevation bin across all frames

---

## 5. Changes to `gaussian_renderer/__init__.py`

### 5.1 `render_sonar()` — no changes to core rendering

The forward projection already handles any 3D surfel position regardless of
elevation.  The row mapping (line 328) correctly uses range, which is
`sqrt(x^2 + y^2 + z^2)` — it does NOT assume elevation = 0.

The only addition is a convenience function to bulk forward-project a tensor of
arbitrary 3D points to pixel coordinates for a given camera, reusing the same
transform.  This avoids duplicating the transform logic in `elevation.py`.

### 5.2 New utility: `forward_project_points()`

Add near `compute_fov_margin()` (around line 190):

```python
def forward_project_points(
    points_world,    # [M, 3]
    viewpoint_camera,
    sonar_config,
    scale_factor,
):
    """
    Project arbitrary world-space points to sonar pixel coordinates.
    Returns col [M], row [M], in_fov [M] boolean mask.

    Reuses the same transform as render_sonar.
    """
    # Same lines 260-328 of render_sonar but for arbitrary points
    ...
    return col, row, in_fov
```

---

## 6. Changes to `scene/gaussian_model.py`

**No changes.**

Surfels keep their existing `_xyz` positions.  Elevation is resolved through
the loss landscape, not through an additional per-surfel parameter.  The
elevation logits live in `debug_multiframe.py` and attach to *pixels*, not
surfels.

If future work wants per-surfel elevation (Option 1B from the base plan), a
new `_elevation` parameter could be added to `GaussianModel.training_setup()`,
but that is out of scope for this plan.

---

## 7. Stage E3: Elevation-Guided Densification (Optional)

Triggered when loss plateaus or at fixed iteration milestones.

**Location:** Inside the training loop, near existing FOV-prune logic
(`debug_multiframe.py:1291`).

### 7.1 High-error pixel identification

```python
if ELEVATION_MODE and iteration % DENSIFY_INTERVAL == 0:
    # Identify pixels where GT is bright but rendered is dark
    error = (gt_image - rendered).abs().squeeze(0)  # [H, W]
    high_error_mask = (error > ERROR_THRESHOLD) & (gt_image.squeeze(0) > BRIGHT_THRESHOLD)
    # These pixels likely need new surfels at correct elevations
```

### 7.2 Multi-view scoring per elevation bin

For each high-error pixel, the arc is already computed (from Section 4.5).
Score each bin by forward-projecting to multiple overlapping frames and summing
GT intensities.

```python
    # arc_pts: [P_err, K, 3] from back_project_with_elevation
    scores = torch.zeros(P_err, K, device="cuda")
    for j in overlap_table[frame_idx]:
        col_j, row_j, valid_j = forward_project_points(
            arc_pts.reshape(-1, 3), training_frames[j],
            sonar_config, sonar_scale_factor
        )
        intensity_j = F.grid_sample(gt_images[j], ...)  # sample at projected coords
        scores += intensity_j.reshape(P_err, K) * valid_j.float().reshape(P_err, K)

    # Pick peak(s)
    best_bin = scores.argmax(dim=-1)  # [P_err]
    # Add new surfels at arc_pts[range(P_err), best_bin, :]
```

### 7.3 Multi-peak detection

If the score profile has two distinct peaks (local maxima separated by a valley
below 50 % of peak height), add surfels at both peaks.  Simple finite-difference
peak finding on the K-element score vector.

---

## 8. Testing Strategy

### 8.1 Unit tests (manual, in-script)

- Back-project with elevation = 0 must produce the same points as the original
  `sonar_frame_to_points()`.
- `back_project_with_elevation()` with a single bin at `el=0` must match the
  legacy path to within float tolerance.
- `forward_project_points()` must agree with `render_sonar`'s pixel coordinates
  for the same input points.

### 8.2 Integration smoke test

Run `debug_multiframe.py` with `ELEVATION_MODE=False` and compare metrics
against a known-good baseline — must be identical (no regression).

### 8.3 Elevation convergence test

Run with `ELEVATION_MODE=True`, `ELEVATION_K=7`, `SONAR_STAGE2_ITERS=2000`.
Monitor:
- Elevation entropy should decrease over iterations.
- Arc-intersection loss should decrease.
- Photometric loss should be equal or better than the non-elevation baseline.
- Peak elevation bin should cluster near ground-truth surface elevation
  (qualitative check via 3D point cloud visualisation).

---

## 9. File Change Summary

| File | Type of change | Scope |
|------|---------------|-------|
| `utils/elevation.py` | **New file** | `ElevationBins`, `back_project_with_elevation`, `arc_intersection_loss`, `multi_view_likelihood` |
| `utils/sonar_utils.py` | Modify `sonar_frame_to_points()` | Add optional `elevation_angles` arg (~10 lines) |
| `utils/point_utils.py` | Modify `sonar_ranges_to_points()` | Add optional `elevation` arg (~8 lines) |
| `gaussian_renderer/__init__.py` | Add `forward_project_points()` | New utility function (~40 lines), no changes to `render_sonar` |
| `debug_multiframe.py` | Major additions | Env vars, elevation logit allocation, arc loss in training loop, temperature annealing, diagnostics, optional densification (~200-300 lines) |
| `scene/gaussian_model.py` | No changes | — |

---

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| VRAM overflow from per-pixel logits | Low (est. 112 MB for 500×8k×7) | Monitor; reduce K or subsample pixels if needed |
| Elevation logits don't converge | Medium | Temperature annealing + direct likelihood supervision (Section 4.6); fall back to E1-only (random init + gradient descent) |
| Arc loss destabilises existing training | Medium | Gate behind `ELEVATION_ARC_WEIGHT`; start very small (0.01); warmup period before activating |
| Overlap table is wrong (bad correspondences) | Low | Validate visually with a few frame pairs; use conservative min-angle filter |
| Scale factor error propagates to arc loss | Medium | Scale factor is frozen at calibrated value; arc loss residual can serve as diagnostic for scale accuracy |

---

## 11. Incremental Delivery Order

Implement and test in this order to get early signal:

1. **`utils/elevation.py`** — core classes and functions, tested standalone.
2. **`sonar_frame_to_points()` elevation arg** — verify backward projection with non-zero elevation produces physically sensible points.
3. **`forward_project_points()`** — verify round-trip: back-project at elevation e, forward-project, lands at same pixel (within tolerance).
4. **Random elevation init** in `debug_multiframe.py` — run training, see if surfels spread in elevation and whether the mesh improves or worsens.
5. **Elevation logits + temperature annealing** — wire up, run, check entropy decreases.
6. **Arc-intersection loss** — add, check loss decreases and doesn't destabilise photometric loss.
7. **Multi-view likelihood** — add direct supervision, check elevation logits converge faster.
8. **Elevation-guided densification** — only if Steps 4-7 show persistent high-error regions.

---

## 12. Open Questions for Discussion

1. **Pixel-to-surfel correspondence**: Section 4.5 finds co-visible surfels via
   FOV check, then forward-projects to get their pixel positions.  Should we
   instead maintain an explicit mapping from pixel to surfel (updated each iteration)?
   The FOV+forward-project approach is simpler but noisier.

2. **Number of overlapping frames per iteration**: Using just 1 other frame is
   fast but may give weak signal.  Using all overlapping frames is expensive.
   Suggest: sample 2-3 per iteration, cycling through the overlap table.

3. **Interaction with bright-pixel loss**: The existing `BRIGHT_WEIGHT` loss
   focuses on bright pixels.  The elevation arc loss also focuses on bright
   pixels (they're the ones with valid returns).  Should these share the same
   pixel mask, or be independent?

4. **When to activate elevation loss**: A warmup period (`ELEVATION_WARMUP_ITERS`)
   lets surfels settle before adding the arc-intersection signal.  Alternative:
   activate from iteration 1 but with very low weight that ramps up.

5. **Elevation bins for normals**: `sonar_ranges_to_points()` in `point_utils.py`
   is used for surface normal computation.  Should normals also use the learned
   elevation, or is elevation = 0 acceptable for normal estimation?
