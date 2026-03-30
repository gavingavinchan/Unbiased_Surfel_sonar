# Chunk-5 Normals Explainer

How the Chunk-5 late-normal supervision path works, what each logged metric means, and why the pipeline can produce zero signal even when training looks healthy.

**Key source files:**

| File | What it contains |
|---|---|
| `debug_multiframe.py` | `_compute_chunk5_local_expected_geometry()`, `compute_chunk5_normal_for_frame()`, pixel bank construction |
| `utils/elevation_chunk5_helpers.py` | `compute_confidence_mask()`, `compute_finite_difference_normals()`, `compute_normal_supervision_loss()`, mode logic |
| `utils/elevation_stage1_helpers.py` | `run_stage1_likelihood_step()`, `masked_softmax()` |
| `utils/elevation_chunk4_helpers.py` | `associate_expected_points_to_surfels()` |
| `utils/point_utils.py` | `sonar_ranges_to_points()`, `sonar_points_to_normals()` |
| `gaussian_renderer/__init__.py` | `render_sonar()` normal output, camera-path `render()` normal output |

---

## What Chunk-5 Does

Chunk-5 estimates local surface normals from elevation posteriors at sparse anchor pixels, then uses those normals to supervise surfel orientations.

A pixel only contributes a normal-loss gradient if it passes every gate in this chain:

```
anchor pixel
  --> supported?           (multi-view evidence exists)
  --> confident?           (posterior entropy low enough)
  --> interior?            (not on image border)
  --> all 4 neighbors confident?
  --> finite normal?       (cross product is numerically valid)
  --> matched to surfel?   (expected point near a visible surfel)
  --> normal_mode=active?  (loss actually added to objective)
```

If any gate fails, that anchor is skipped. No fallbacks, no partial normals.

---

## Step By Step

### 1. Select anchor pixels

Chunk-5 operates on a sparse set of pixels, not the whole image. The pixel bank is built by `select_topk_bright_pixels()` in `debug_multiframe.py`, which picks the top-K brightest valid pixels per frame.

### 2. Get the center pixel's elevation posterior

For each anchor pixel, Stage-1 has already computed a posterior over `K` elevation bins (typically `K=7`). This happens in `run_stage1_likelihood_step()` (`utils/elevation_stage1_helpers.py:314`):

```python
p_post = masked_softmax(
    logits / temp_post + loglik,
    support_mask
)
```

This combines two sources:
- `logits` -- learned per-pixel parameters (divided by temperature)
- `loglik` -- multi-view evidence (back-project into overlapping frames, sample intensity, aggregate log-likelihood)

The result is a probability distribution over elevation bins for each anchor pixel.

### 3. Check support

A pixel is "supported" if at least one of its elevation bins has valid multi-view evidence (`support_mask.any(dim=-1)`).

The logged metric `cov` is the fraction of anchor pixels that are supported. It does **not** mean a usable normal exists -- only that there is some evidence at the center pixel.

### 4. Check confidence

The code computes entropy of the posterior and rejects pixels where the posterior is too spread out (`utils/elevation_chunk5_helpers.py:61-78`):

```python
entropy = -sum(p * log(p))
max_entropy = log(K)
confident = supported and is_finite(entropy) and entropy < confidence_thresh * max_entropy
```

With the default `ELEV_NORMAL_CONFIDENCE_THRESH=0.5` and `K=7`:
- max entropy = `log(7) = 1.946`
- cutoff = `0.5 * 1.946 = 0.973`

So a posterior with entropy 0.8 passes, but entropy 1.2 fails. A nearly uniform distribution (entropy ~1.9) fails by a wide margin.

Concrete examples with `K=7`:

| Posterior | Entropy | Passes? |
|---|---|---|
| `[0, 0, 0.95, 0.05, 0, 0, 0]` | ~0.20 | yes |
| `[.05, .10, .20, .30, .20, .10, .05]` | ~1.70 | no |
| `[.14, .14, .15, .14, .15, .14, .14]` | ~1.95 | no |

### 5. Check neighbors

Chunk-5 needs four adjacent pixels to compute a finite-difference normal:

```
        up (row-1, col)
            |
left (row, col-1) -- CENTER -- right (row, col+1)
            |
       down (row+1, col)
```

The center pixel must not be on the image border (otherwise neighbors don't exist).

Each neighbor goes through its own posterior computation and confidence check. All four must pass. One failing neighbor kills the normal for that anchor.

This is where the `conf_cov` metric comes from: the fraction of anchor pixels that survive the full local-readiness gate, meaning the center pixel is supported and confident, the anchor is interior, and all four neighbors are confident.

### 6. Compute expected 3D points and finite-difference normal

For each surviving pixel (center + 4 neighbors), the code computes an expected elevation as a weighted mean over bins (`utils/elevation_chunk5_helpers.py:35-39`):

```python
e_expected = sum(p[k] * elev_bin[k])
```

Each `(row, col, expected_elevation)` is converted to a 3D world-space point via sonar polar geometry. Then the normal is computed from central differences (`utils/elevation_chunk5_helpers.py:42-58`):

```python
dp_az = pts_right - pts_left      # tangent in azimuth direction
dp_rg = pts_down  - pts_up        # tangent in range direction
normal = normalize(cross(dp_az, dp_rg))
```

If the cross product has zero or non-finite magnitude, the normal is marked invalid. The `finite` metric counts how many anchors produced a valid normal.

### 7. Match expected point to a visible surfel

The expected center point is projected into the current frame's pixel/range coordinates. The code then searches for a nearby visible surfel using `associate_expected_points_to_surfels()` (`utils/elevation_chunk4_helpers.py`):

- Filter: surfel must be within `max_pix_err` pixels and `max_depth_err` range
- Score: `(pix_err / sigma_pix)^2 + (depth_err / sigma_depth)^2`
- Pick the surfel with the lowest score
- Weight: `w = exp(-0.5 * score)`, clamped to `[min_w, 1.0]`

The `match` metric counts how many expected points found a valid surfel.

### 8. Compute the normal loss

The loss compares the surfel's current normal (from its quaternion via `quaternion_to_normal()`) against the expected finite-difference normal (`utils/elevation_chunk5_helpers.py:81-84`):

```python
loss_normal = 1 - abs(dot(n_surfel, n_expected))
```

This is sign-agnostic: a perfectly aligned normal and a perfectly flipped normal both give loss 0. Orthogonal normals give loss 1.

### 9. Apply (or don't) based on mode

Set by `ELEV_NORMAL_MODE` environment variable (default: `"shadow"`):

| Mode | Behavior |
|---|---|
| `off` | Skip all Chunk-5 processing |
| `shadow` | Compute everything, log diagnostics, but do **not** add loss to objective |
| `active` | Add `w_normal * loss_normal` to total loss (backprop enabled) |

The weight `w_normal` ramps linearly from `weight_early` to `weight_late` between configurable iteration bounds.

---

## Log Metrics Reference

All of these appear in the training log line. There is also a separate `match=...` from Chunk-4 coupling earlier in the same line -- that is a different metric.

| Field | Meaning | Does NOT mean |
|---|---|---|
| `cov` | fraction of anchors whose center pixel is supported | that a usable normal exists |
| `conf_cov` | fraction of anchors passing center + 4-neighbor confidence | that matching/loss happened |
| `finite` | count of anchors with numerically valid finite-difference normals | that loss was applied |
| `match` | count of finite normals matched to a visible surfel | that loss affected training (depends on mode) |
| `normal` | scalar Chunk-5 normal loss value | that it was added to the objective (shadow logs it but doesn't backprop) |
| `w_normal` | scheduled weight for the normal term | anything if match=0 or mode is not active |

### Reading a typical failure

```
cov=1.0000  conf_cov=0.0000  finite=0  match=0  normal=0.0
```

This means: center pixels have evidence, but no anchor survived the confidence + neighbor gate. No normals were computed at all. Even switching to `active` mode would not help -- there is nothing to backprop.

---

## The Center-vs-Neighbor Posterior Asymmetry

This is the most important implementation detail for understanding why `conf_cov` can collapse to zero.

**Center pixels** use the full Stage-1 posterior from `run_stage1_likelihood_step()`:

```python
p_post = masked_softmax(logits / temp_post + loglik, support_mask)
```

This combines learned logits with multi-view evidence. The logits provide a prior that sharpens the distribution.

**Neighbor pixels** are computed freshly inside `_compute_chunk5_local_expected_geometry()` (`debug_multiframe.py:1382-1402`):

```python
query_loglik, query_support = build_stage1_multiview_loglik_for_pixels(...)
query_probs = masked_softmax(query_loglik, query_support)
```

Neighbors use **evidence only** -- no learned logits, no temperature scaling. This produces a weaker, flatter posterior that is more likely to fail the entropy confidence gate.

Since all four neighbors must pass, and each is judged by this weaker evidence-only posterior, the probability of the full 5-pixel patch surviving drops steeply. This is the primary mechanism by which `conf_cov` can stay at zero even when `cov` is high.

---

## How The Camera Normal Pipeline Works, Start To Finish

This project descends from a camera-based 2D Gaussian Splatting codebase. The camera path has a dense normal-correction mechanism that Chunk-5 is trying to replace for sonar. Understanding it end-to-end -- from raw photos to correct surfel normals -- explains why the camera path is so much stronger.

### Where surfels come from in the first place

Before any training happens, the input camera images are fed through COLMAP (a photogrammetry tool). COLMAP does two things:

1. **Estimates camera poses** -- where each photo was taken from and which direction it was looking. Same idea as visual odometry on an AUV.
2. **Produces a sparse point cloud** -- by triangulating feature matches across images. If the same rock corner appears in photo 3 and photo 7, and you know both camera poses, you can triangulate its 3D position.

The output is ~thousands of 3D points with estimated positions. The important point for this discussion is that the original `0d41037` training path did not rely on meaningful input normals for surfel orientation initialization.

Each point becomes a surfel. The code sets initial values for every parameter:

- **Position**: from COLMAP's point cloud
- **Color**: from the average observed color at that point
- **Scale**: estimated from nearest-neighbor distances (small if points are dense, large if sparse)
- **Opacity**: some initial value
- **Quaternion/normal**: initialized generically; in the original `0d41037` base commit, `create_from_pcd()` used random quaternions

At this stage the normals are essentially garbage. In the original base commit, the surfels are positioned roughly correctly but oriented randomly.

### How training works: render, compare, update

The entire renderer is differentiable -- you can compute gradients of the output image with respect to every surfel parameter, including the quaternion.

Each training iteration:

1. **Pick a camera** from your training set (you know its pose and have its real photo).
2. **Render** the current surfels from that camera's viewpoint. The CUDA rasterizer projects all surfels onto the image plane, alpha-composites their color contributions, and produces a synthetic image.
3. **Compare** the synthetic image to the real photo. The difference is the photometric loss (combination of L1 pixel error and SSIM structural similarity).
4. **Backpropagate** that loss through the differentiable renderer to get gradients on every surfel parameter.
5. **Update** all parameters with the Adam optimizer.

This is system identification. You have a parameterized forward model (surfels -> rendered image), observations (real photos), and you minimize prediction error by gradient descent.

The photometric loss alone gives some normal signal -- a surfel's orientation affects which pixels it covers and with what shape, so wrong orientation means wrong image. But this signal is weak and indirect. Many orientations can produce similar-looking images, especially for small surfels. The optimizer mostly focuses on getting colors and positions right, leaving orientations under-constrained.

That is where the normal consistency loss comes in.

### What a surfel's normal is

Each surfel stores a quaternion `_rotation` [N,4]. That quaternion defines a 3x3 rotation matrix R. The surfel's normal is **the third column of R** -- the local Z-axis of the surfel's coordinate frame.

In the CUDA rasterizer (`forward.cu:88-90,113`):

```c
R = quat_to_rotmat(rot);        // quaternion -> 3x3 rotation
S = scale_to_mat(scale, mod);   // diag(sx, sy, 1) -- third scale forced to 1
L = R * S;
normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
```

`S` is `diag(sx, sy, 1)` because surfels are flat discs (2D Gaussians), not 3D ellipsoids -- the third axis has no spatial extent. So `L[2] = R * [0, 0, 1]^T` = the third column of R = the surfel normal in world space. The `transformVec4x3` call rotates it into view space.

There is also a Python-side equivalent in `quaternion_to_normal()` (`gaussian_renderer/__init__.py:1556`), which does the same quaternion-to-Z-axis math. The CUDA version is used during rasterization; the Python version is used when you need normals outside the rasterizer (e.g., Chunk-5 matching).

### How `rend_normal` is produced: alpha-blending surfel normals

The CUDA rasterizer (`diff-surfel-rasterization/cuda_rasterizer/forward.cu`) processes surfels in front-to-back order per pixel. For each pixel, it iterates over all surfels that overlap that pixel and composites their contributions.

Per-surfel preprocessing (`forward.cu:171-253`):

1. The surfel's quaternion and scale are used to build the 2D-to-world transform and extract the view-space normal (as described above).
2. The normal and opacity are packed together into a `float4`: `normal_opacity[idx] = {normal.x, normal.y, normal.z, opacity}`.

Per-pixel compositing (`forward.cu:358-457`):

For each surfel overlapping this pixel, the rasterizer computes:
- The intersection point of the pixel ray with the surfel disc
- A Gaussian weight `G = exp(-0.5 * rho)` based on distance from surfel center
- Alpha: `alpha = min(0.99, opacity * G)`
- Compositing weight: `w = alpha * T`, where T is the remaining transmittance (starts at 1.0, decreases as surfels accumulate)

The normal is accumulated exactly like color (`forward.cu:434`):

```c
for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
```

This is standard alpha-compositing. Each surfel contributes its view-space normal weighted by how much "opacity budget" it uses at this pixel. A surfel that is nearly transparent or far behind other surfels contributes little. A surfel that is opaque and in front dominates.

After all surfels are processed, the accumulated normal `N[3]` is written to `out_others` at the `NORMAL_OFFSET` channels (`forward.cu:477`):

```c
for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
```

The layout of `out_others` (defined in `auxiliary.h:23-27`):

| Offset | Channel |
|---|---|
| 0 | depth (expected) |
| 1 | alpha |
| 2,3,4 | normal (view-space, x/y/z) |
| 5 | median depth (unbiased surface depth) |
| 6 | distortion |

Back in Python, `out_others` is returned as `allmap` (the third return value from the rasterizer, called `depth` in the Python binding at `__init__.py:94`). The normal channels are extracted and rotated from view-space to world-space (`gaussian_renderer/__init__.py:128-129`):

```python
render_normal = allmap[2:5]
render_normal = (render_normal.permute(1,2,0) @ viewpoint_camera.world_view_transform[:3,:3].T).permute(2,0,1)
```

So **`rend_normal`** is a [3,H,W] image where each pixel holds the alpha-composited, world-space normal from all surfels contributing to that pixel. It reflects what the surfels currently "believe" the surface orientation is.

### How `surf_normal` is produced: depth-derived pseudo-normals

Completely independently, the rasterizer also produces a depth value per pixel. The "unbiased" surface depth (Eq. 9 in the Unbiased 2DGS paper) ends up in `allmap[5]`, which becomes `surf_depth`.

That depth map is unprojected to 3D points, then finite-difference normals are computed (`gaussian_renderer/__init__.py:143`, calling `depth_to_normal()` in `utils/point_utils.py:57-65`):

```python
points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
dx = points[2:, 1:-1] - points[:-2, 1:-1]   # central diff in row
dy = points[1:-1, 2:] - points[1:-1, :-2]    # central diff in col
normal_map = normalize(cross(dx, dy))
```

`depths_to_points()` (`point_utils.py:13-28`) unprojects each pixel using the camera intrinsics and extrinsics: pixel -> ray direction -> scale by depth -> world point. Then the cross product of row/column differences gives the surface normal at each interior pixel.

The result is masked by alpha to zero out regions with no rendered content (`gaussian_renderer/__init__.py:146`):

```python
surf_normal = surf_normal * render_alpha.detach()
```

So **`surf_normal`** is a [3,H,W] image where each pixel holds the normal implied by the rendered depth geometry. It does not look at any surfel's quaternion -- it only looks at where the surfels ended up in depth.

### Why these two maps disagree

`rend_normal` and `surf_normal` come from the same surfels but through completely different computational paths:

- `rend_normal`: surfel quaternion -> rotation matrix -> third column -> alpha-blend across surfels
- `surf_normal`: surfel positions/opacity -> depth compositing -> unproject to 3D -> finite differences

A surfel can claim its normal points up (via its quaternion) while sitting on a depth surface that slopes sideways. The two maps will then disagree at that pixel.

### What "consistent with depth geometry" means

After rendering, you have a depth value at every pixel. That's a height field -- a surface defined on a regular grid, like bathymetry data.

Take any pixel. Look at its depth and its neighbors' depths. If the pixel to the right is farther away and the pixel below is closer, the depth surface slopes. You compute the slope direction by taking finite differences on the grid -- the same thing you'd do to get terrain gradient from a DEM.

That slope is `surf_normal`. It comes purely from the depth values. It doesn't know or care what any surfel's quaternion says.

Now, the surfels that rendered that pixel also have quaternions that claim the surface points in some direction. That claim is `rend_normal`.

The consistency loss says: **if your surfels produce a depth surface that slopes at 30 degrees to the left, then the surfels at that pixel should also claim their normals point 30 degrees to the left.**

Concretely, suppose you have a flat seafloor at 2m depth, tilted 15 degrees in x:

```
depth at pixel (100, 50) = 2.00m
depth at pixel (102, 50) = 2.01m    (farther -- surface tilts away from camera)
depth at pixel (100, 52) = 2.00m    (same -- no tilt in this direction)
```

Unproject these to 3D, take differences, cross product -- you get a `surf_normal` tilted ~15 degrees. But suppose the surfel at that pixel has a quaternion claiming the normal points straight at the camera (0 degrees). The loss: `1 - cos(15 deg) = 0.034`. The gradient pushes the surfel's quaternion to rotate 15 degrees to match.

Why would they ever disagree? Because depth and normal come from independent computations. Depth is driven mainly by surfel **positions** and **opacities** -- the optimizer adjusts these to match image colors. The quaternion is a separate parameter that the optimizer can leave wrong while getting the depth right. Think of hull plates tack-welded in roughly the right position but tilted at the wrong angle -- the positions are right, the angles are wrong. The consistency loss is the alignment step that corrects the angles to match the as-built surface profile.

### The loss and gradient chain

Training penalizes disagreement at every pixel (`train.py:159-160`):

```python
normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
normal_loss = lambda_normal * normal_error.mean()
```

This is `1 - cos(angle)` per pixel, averaged over the image (range 0 to 2; note Chunk-5 uses the sign-agnostic `1 - |cos|` instead). It is enabled after iteration 7000 (`train.py:143`).

The gradient flows backward through both paths, but the critical one for fixing surfel orientations is through `rend_normal`:

1. `normal_loss` produces a gradient on each pixel of `allmap[2:5]` (the blended normal image).
2. The CUDA backward pass (`backward.cu:491-492`) distributes the per-pixel gradient back to per-surfel normal gradients:
   ```c
   atomicAdd(&dL_dnormal3D[surfel * 3 + ch], alpha * T * dL_dnormal2D[ch]);
   ```
   Each surfel receives gradient proportional to how much it contributed to that pixel (its compositing weight `alpha * T`).
3. The per-surfel normal gradient becomes part of the rotation gradient (`backward.cu:673-676`):
   ```c
   glm::mat3 dL_dRS = glm::mat3(
       dL_dM[0],                                    // from projection transform
       dL_dM[1],                                    // from projection transform
       glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)     // from normal loss
   );
   ```
   The third row of the `dL_dRS` matrix comes directly from the normal gradient.
4. This is converted to a quaternion gradient via `quat_to_rotmat_vjp()` (`backward.cu:684`):
   ```c
   dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
   ```
5. PyTorch's Adam optimizer updates `_rotation` using that quaternion gradient.

So on every iteration after warmup, every visible surfel gets a gradient nudge on its quaternion that pushes its self-reported normal toward the normal implied by the depth surface. Over many iterations, the two converge.

### Why this is strong

- **Dense**: every visible pixel contributes every iteration. Hundreds of thousands of gradient signals per step.
- **Always available**: it only needs a rendered depth map and rendered normals, both free byproducts of rasterization.
- **Self-correcting**: if the depth surface says "slope facing left" but a surfel says "pointing up", the gradient rotates the surfel. As surfels rotate, the depth surface also shifts, and the two converge.
- **No external input**: no multi-view evidence, no posterior, no matching step. Purely internal to one rendered frame.

### Why sonar cannot use this

The sonar renderer (`render_sonar`) does not use the camera path's CUDA rasterizer normal pipeline. It renders in sonar polar geometry and can use the CUDA `GaussianRasterizer` for event-volume accumulation in some configurations, but it does not produce the camera path's alpha-blended surfel-normal channels (`allmap[2:5]`).

What it does produce is a range image, which it converts to 3D points and then to normals via `sonar_points_to_normals()`. But it returns the same tensor for both outputs (`gaussian_renderer/__init__.py:1548,1551`):

```python
"rend_normal": surf_normal,   # same object
"surf_normal": surf_normal,   # same object
```

If you tried to use the camera loss, you'd compute `1 - dot(x, x) = 0` everywhere. No gradient.

The camera-style normal loss is explicitly disabled in sonar mode (`train.py:138-140`):

```python
if dataset.sonar_mode:
    lambda_normal = 0.0
```

### Side-by-side

| Aspect | Camera path | Sonar Chunk-5 |
|---|---|---|
| Density | every visible pixel | sparse anchor pixels only |
| Normal source A | alpha-blended surfel normals from CUDA rasterizer | surfel quaternion normal after point-matching |
| Normal source B | depth-derived finite-difference normals | elevation-posterior finite-difference normals |
| Prerequisites | rendered depth exists | support + confidence + 4-neighbor + finite + match |
| Gradient path to quaternion | CUDA backward through alpha-compositing | Python-side through `compute_normal_supervision_loss` |
| Active after warmup? | always (iter > 7000) | only when all gates succeed |
| Degradation | gradual (blurrier if depth is poor) | binary (one failed gate = zero signal for that anchor) |

The camera path can correct bad initial surfel orientations because the signal is present every iteration. The sonar path can go completely silent if the confidence gates fail broadly, leaving the optimizer with no pressure to fix orientations.

---

## Why This Matters In Practice

The failure mode is not "Chunk-5 is working but shadow mode prevents gradient flow." The failure mode is "Chunk-5 produces no usable normals at all."

If the logs show `conf_cov=0`, then even switching to `active` mode and cranking up `w_normal` would produce no gradient. The problem is upstream of the mode switch: either the posteriors are too uncertain, or the center-vs-neighbor asymmetry is starving the neighbor confidence gate.

The immediate diagnostic question is always: **is `conf_cov` nonzero?** If not, nothing downstream matters.
