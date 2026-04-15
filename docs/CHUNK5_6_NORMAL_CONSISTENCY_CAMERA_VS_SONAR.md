# Normal Consistency: Camera Baseline vs. Current Sonar Path

> **Context:** This document was created while planning Chunk 5.6 (see
> `plans/PLAN_ELEVATION_AWARE_CHUNK5_6_EXECUTION_2026-04-05.md`), to compare how
> the restored sonar normal-consistency regularizer matches the original camera
> baseline and to decide what per-frame visualizations would let us see whether
> it is working.

Reference camera commit: `0d41037`.

## Core idea (shared by both paths)

Both camera and sonar training enforce the same self-consistency principle every iteration:

- **`rend_normal`** — what the surfels *claim* the surface orientation is (accumulated per-surfel normals).
- **`surf_normal`** — what the *rendered depth geometry* says the orientation should be (derived from `surf_depth`).

The loss pushes these two fields to agree, so surfel orientations converge to match the macroscopic surface implied by the composited depth.

## Renderer contract

Both paths return the same five regularization maps from the renderer:

| Field | Meaning |
|---|---|
| `rend_alpha` | Accumulated opacity / support mass |
| `rend_normal` | Rendered surfel-orientation field (from per-surfel normals) |
| `surf_depth` | Pseudo-surface / hard-depth per ray |
| `surf_normal` | Macroscopic surface normal derived from `surf_depth` |
| `rend_dist` | Depth-spread / distortion signal |

### How each path produces these

| Step | Camera (`render()`) | Sonar (`render_sonar()`) |
|---|---|---|
| Source of surfel normals | Rasterizer `allmap` channels, per-surfel frame | `quaternion_to_normal(rotations)` |
| `rend_normal` accumulation | CUDA rasterizer alpha/transmittance compositing, then rotated to world | `transmittance_before_event * alpha_sorted` along ray-binned events, splatted over sonar row footprint |
| `surf_depth` | Rasterizer's pseudo-surface / median-like depth channel | Per-ray hard-depth selection from compositing; returned `surf_depth` is the splatted regularizer version of `hard_depth_image` |
| `surf_normal` | `depth_to_normal()` — backproject depth, finite differences | `sonar_ranges_to_points()` + `sonar_points_to_normals()` on the sonar range image |
| Alpha weighting of target | `surf_normal *= rend_alpha.detach()` | Same: `surf_normal *= rend_alpha.detach()` |

## Loss formulas

### Camera baseline (`train.py`)

```python
normal_error = 1 - (rend_normal * surf_normal).sum(dim=0)
normal_loss  = lambda_normal * normal_error.mean()
```

- `rend_normal` is **not normalized** before the loss.
- Loss is **sign-sensitive** (`1 - dot`), not `1 - abs(dot)`.
- The rasterizer already flips surfel normals camera-facing before accumulation, so sign consistency is guaranteed upstream.

### Current sonar path (`debug_multiframe.py:compute_chunk5_normal_for_frame`)

```python
surf_normal_target = surf_normal * render_pkg["rend_alpha"].detach()
rend_normal_comp   = normalize(rend_normal[..., valid])
surf_normal_comp   = normalize(surf_normal_target[..., valid])
loss_normal = compute_normal_supervision_loss(
    n_quat=rend_normal_comp,
    n_expected=surf_normal_comp,
)
# compute_normal_supervision_loss: loss = 1.0 - torch.abs(cosine)
```

- Both fields are **normalized** before comparison.
- Loss is **sign-agnostic** (`1 - abs(dot)`) — no camera-facing flip exists in the sonar renderer, so the loss absorbs the ambiguity.

### Where it is applied

- **Camera**: added directly into `train.py`'s total loss alongside photometric and depth regularizers.
- **Sonar**: added into the unified loss in **both** the Stage 2 and Stage 3 loops in `debug_multiframe.py`. The depth term from `rend_dist` is also added alongside.

## Key differences to keep in mind

1. **Normalization**: sonar normalizes both fields first; camera does not.
2. **Sign handling**: sonar uses `1 - |dot|`; camera uses `1 - dot` and relies on upstream normal flipping.
3. **`surf_normal` source**: sonar finite-differences a rendered *range* image under collapsed-elevation geometry; camera finite-differences a rendered depth image under pinhole geometry.
4. **`surf_depth` semantics**: sonar returns the *splatted regularizer depth* built from `hard_depth_image`, not the raw per-ray hard-depth map itself.

## Conclusion

The normal-consistency feature **does exist** in the current sonar code and follows the same high-level structure as the camera baseline:

- renderer emits `rend_normal`, `rend_alpha`, `surf_depth`, `surf_normal`,
- trainer compares rendered surfel normals against the depth-derived surface target,
- the term is added to the actual training loss (not just logged).

The differences are confined to the comparison rule (normalized + sign-agnostic for sonar) and the geometry used to build `surf_normal` from `surf_depth`.

## Suggested visualizations

To see whether it's actually working, dump these per-frame maps at stage boundaries:

1. `rend_alpha` — support mass heatmap
2. `surf_depth` — depth/range heatmap
3. `rend_normal` — RGB normal map (xyz → RGB)
4. `surf_normal` — RGB normal map, same encoding
5. **Agreement map**: `abs(dot(normalize(rend_normal), normalize(surf_normal)))` — 0 (red) to 1 (green)

Pairs (3) and (4) side-by-side will show whether surfel orientations are tracking the depth-implied surface. The agreement map in (5) is the most direct visual proxy for what the normal loss actually sees.

## Blender visualization discussion

### Existing Blender-friendly exports

The repo already emits Blender-friendly geometry, and there is already a surfel glyph path with explicit normal stems.

Already Blender-friendly:

- `debug_multiframe.py` already writes `.ply` geometry you can import into Blender:
  - `sonar_init_points.ply`
  - `mesh_before_training.ply`
  - `mesh_after_stage2.ply`
  - `mesh_after_stage3.ply`
  - `visualizer/glyphs_*_sampled.ply`
  - `visualizer/glyphs_*_front_faces.ply`
  - `visualizer/glyphs_*_back_faces.ply`
- The glyph exports are the most useful for normals:
  - they are actual triangle meshes
  - each surfel has a face disk
  - each surfel also gets a normal stem
- That glyph geometry is built in `utils/visualization_utils.py:383-521`.

What that lets you see:

- surfel positions in 3D
- surfel orientation in 3D
- front vs back side of surfels
- how orientation changes across training stages

So if the question is "can I inspect the surfel geometry and normals in Blender now?", the answer is yes.

What you cannot see yet in 3D:

- the actual `rend_normal` field as a dense 3D vector field
- the actual `surf_normal` field as a dense 3D vector field
- their per-pixel agreement directly in Blender

Right now those exist as rendered image-space maps, not exported 3D geometry.

Best current Blender route:

1. Import `mesh_after_stage3.ply` or `mesh_after_stage2.ply`.
2. Import `visualizer/glyphs_stage3_sampled.ply`.
3. Optionally import `visualizer/glyphs_stage3_front_faces.ply` and `...back_faces.ply` separately.
4. Compare against `sonar_init_points.ply` and `pose_pyramids_wireframe.ply`.

If the goal is to see the normal-regularizer itself in Blender, the right next export is:

1. backproject `surf_depth` to 3D points
2. attach `surf_normal` as line segments
3. attach `rend_normal` as line segments at the same points
4. color by agreement `abs(dot(...))`

That would give a true Blender-viewable 3D debug artifact for the normal consistency term.

### What we actually want to see

For the active sonar path:

- `rend_normal` = the orientation the surfels are claiming
- `surf_depth` = the renderer's macro-surface depth estimate
- `surf_normal` = the macro-surface normal field derived from that `surf_depth`

So if the goal is to inspect whether the regularizer is meaningful on the cube, the thing to visualize is the `surf_depth -> surf_normal` macro-surface, not just the surfels.

In Blender, the most informative comparison would be:

1. the rendered macro-surface for one sonar frame
2. the ground-truth cube surface
3. optionally the sampled surfel glyphs on top

That would answer:

- does the renderer's implied surface actually lie on the cube faces?
- does it bow, smear, or form phantom surfaces?
- do the surfel normals align to that same surface?

Best representation:

1. Per-frame macro-surface mesh
- Build a 3D mesh from `surf_depth` by backprojecting it to world points.
- Triangulate neighboring valid pixels.
- This gives the actual macroscopic surface the loss is using.

2. Per-frame macro-surface normals
- Optional but useful.
- Either store vertex normals from `surf_normal`, or export short line segments.

3. Ground-truth cube mesh
- From the dataset manifest.
- For `synthetic_cube_C_tiny`, the GT cube is:
  - center `[0, 0, 0]`
  - half extent `0.8`
- So a simple axis-aligned cube mesh can be exported directly into the same world coordinates.

4. Existing surfel glyph mesh
- Reuse the existing sampled surfel glyph export.
- Then you can overlay:
  - GT cube
  - macro-surface mesh
  - surfel glyphs

That is probably the cleanest Blender view.

Why this is better than only 2D maps:

The 2D `surf_depth` image is useful, but it can hide the actual geometric failure mode.

In 3D, you immediately see:

- whether the macro-surface sits on the true cube faces
- whether it gets rounded off
- whether edges smear across faces
- whether the collapsed-elevation sonar geometry is inventing a false surface

That is exactly the concern that motivated the Blender discussion.

Recommended first pass:

1. Export one macro-surface mesh per training frame from the latest cube run.
2. Export one GT cube mesh once.
3. Overlay those in Blender.
4. Keep the existing `surfels_in_fov` glyph export for the same frame.

That gives the most direct sanity check with minimal ambiguity.

Optional extras:

1. cube residual
- color by distance to GT cube surface

2. agreement with `rend_normal`
- color by `abs(dot(rend_normal, surf_normal))`

3. face identity
- color which cube face each macro-surface point is closest to

If the goal is first-pass debugging, start with:

- plain macro-surface mesh
- GT cube mesh
- surfel glyphs

Then add color overlays only if needed.

One important choice before implementation:

1. per-frame macro-surface exports
- best for understanding what each sonar view is supervising

2. merged macro-surface from all training frames
- better for overall shape comparison
- but less faithful to the per-frame normal loss, which is applied frame-by-frame

Recommendation: `per-frame` first, because that is what the loss actually sees.

## How the macro-surface is determined and used

Per sonar frame, not per surfel.

High level:

For one training step on one sonar image, the code does this:

1. Render the current surfel set into that sonar view.
2. From that render, build a dense `surf_depth` image.
3. Convert `surf_depth` into a dense `surf_normal` image.
4. Compare that dense macro-surface normal image against the dense rendered surfel-normal image `rend_normal`.
5. Average that mismatch into one loss term for that frame.

So the macro-surface is a property of the rendered frame, not something recomputed separately for each surfel.

How the macro-surface is determined:

In the sonar path, the macro-surface starts from the current surfels and the current camera pose.

Roughly:

1. Each surfel projects into the sonar view.
2. The renderer does ray-binned compositing along sonar rays.
3. From that compositing, it chooses a hard or pseudo-surface depth per ray.
4. That depth is then splatted into the image as `surf_depth`.
5. `surf_depth` is backprojected into 3D points.
6. Finite differences on neighboring pixels give `surf_normal`.

So the macro-surface is basically:

- "the surface implied by the current rendered depth image"

not:

- "the true cube surface"
- and not "a separate fitted plane/cube model"

It is self-consistency regularization.

What it means conceptually:

The training is asking:

- "Given all the surfels currently visible in this sonar frame, what macroscopic surface do they collectively imply?"
- "Do the individual surfel normals agree with that collective surface?"

If not, the loss pushes them toward agreement.

Is it computed per surfel?

No.

It is computed once per rendered sonar frame as dense image-space maps:

- `surf_depth`
- `surf_normal`
- `rend_normal`
- `rend_alpha`

Then the loss is evaluated pixelwise over valid pixels.

So the structure is:

- per-frame render
- per-pixel comparison
- mean loss over valid pixels

not:

- loop over surfels and build a separate macro-surface for each surfel

Why this matters:

That means the macro-surface is view-dependent.

Each sonar frame gets its own macro-surface because:

- different surfels are visible
- occlusion changes
- the rendered depth image changes
- therefore the derived `surf_normal` field changes

So if this is inspected in Blender, the most faithful thing is a per-frame macro-surface, not one global surface.

How the loss is formed:

1. Build `rend_normal`
- dense image of what the visible surfels are claiming

2. Build `surf_normal`
- dense image of what the rendered macro-surface says the surface orientation is

3. Compare them only where both are valid

Current sonar code uses a sign-agnostic cosine comparison:

- normalize both
- compute dot product
- take `1 - abs(dot)`
- average over valid pixels

So the loss is pixel-based and frame-based.

Short version:

- The macro-surface is recomputed every render, for each sonar frame.
- It is derived from the rendered depth image, not from ground truth.
- The loss is not computed "per surfel against its own local macro-surface".
- Instead, the renderer builds one dense macro-surface for the frame, and all visible surfels are judged through the rendered normal field against that frame-level surface.

## Multiple surfaces in one sonar frame

Yes and no.

Not one single plane:

It is not "one macro-surface for the whole frame" in the sense of a single fitted plane.

It is:

- one dense `surf_depth` field per sonar frame
- and one dense `surf_normal` field derived from that depth field

So the frame can contain multiple visible surfaces across different pixels.

For a cube edge, that means:

- some pixels can lie on one face
- neighboring pixels can lie on the other face
- the resulting macro-surface can represent both faces in the same frame

But only one depth choice per pixel/ray:

The important limitation is this:

For each rendered pixel or sonar ray, the renderer ultimately produces one `surf_depth` value.

So if a single ray/beam contains competing geometry, the regularizer does not keep two separate surfaces there. It collapses that to one chosen surface depth.

That means:

- across the image: many different surfaces can appear
- at one pixel/ray: only one surface depth is retained

For the cube edge case:

1. Many rays on the left side of the image may hit one cube face.
2. Many rays on the right side may hit the adjacent face.
3. Near the boundary, the depth field can transition sharply.
4. The finite-difference normal field then sees that transition and may produce:
   - a sensible edge transition
   - or a smeared / unstable normal near the edge

So the frame can absolutely encode both cube faces, but only as a single depth map over the image.

Where it breaks down:

The real issue is when multiple surfaces compete within the same sonar beam support, especially because sonar collapses elevation.

Then the chosen `surf_depth` may correspond to:

- the front-most dominant surface
- or an effective composited surface
- not a clean physical face

In that case the derived `surf_normal` can describe a phantom surface.

That is exactly why visualizing the macro-surface on the cube is useful.

Short answer:

- Yes: there is one macro-surface representation per sonar frame, in the form of one dense depth/normal field.
- No: that does not mean one single surface for the whole frame; it can contain multiple cube faces across different pixels.
- But at each individual pixel/ray, it keeps only one depth/surface estimate, not multiple competing surfaces.
