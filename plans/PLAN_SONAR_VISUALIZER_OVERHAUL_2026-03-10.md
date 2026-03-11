# Plan: Sonar Visualizer Overhaul — Oriented Surfels, Per-Frame Wireframes, Full-Range FOV

**Date:** 2026-03-10
**Git Commit:** 82a7d6c
**Status:** Draft for execution
**Scope:** Visualization/debugging only. This plan improves observability of sonar surfels and frame geometry; it does not claim renderer correctness.

**Primary consumer:** Blender-imported `.ply` artifacts (no interactive viewer planned).

---

## Why this is needed now

The latest renderer-focused commit (`82a7d6c`) adds more internal plumbing and failure logging, but it still does not make the geometry visually legible enough to answer the question that matters: *is the model learning the right surfaces, with the right size/orientation, from the right sonar frames?*

Right now the main exported artifacts still leave critical blind spots:

- `sonar_init_points.ply` shows only backward-projected point centers, not initialized surfel footprint/orientation.
- `surfels_after_training.ply` stores latent surfel state, but there is no viewer/export path that turns `scale_*` and `rot_*` into visible geometry.
- `pose_pyramids_wireframe.ply` merges all frames into one file, which makes frame identity hard to read.
- The current wireframe uses a short debug depth (`PYRAMID_DEPTH`) rather than the sonar's full usable range, so FOV reach is visually understated.

Because of those gaps, the occlusion/renderer plan at `plans/PLAN_MISSING_OCCLUSION_AND_RENDERER_FIX_2026-03-02.md` is still hard to visually validate even when code changes land. We need a visualizer that makes failure modes obvious instead of hiding them.

---

## Goals

- Make surfel size and orientation directly visible for both pre-training and post-training states.
- Split sonar frame wireframes into one artifact per frame so frame identity is unambiguous.
- Keep the existing short pose wireframe shape, but add a separate full-range FOV wireframe that reaches `sonar_config.range_max`.
- Make it easy to compare initial vs trained surfels and isolate a single sonar frame.
- **Per-pose render comparison:** For each individual pose, export the surfels within that pose's FOV as a separate artifact, alongside the forward-projected rendered sonar image for that pose. This lets the user compare "what 3D surfels does this frame see?" with "what did the renderer actually produce?" — the key question for diagnosing render failures. Scrub the rendered images to spot a problem, search the frame number in Blender's outliner to inspect the 3D geometry.
- Prioritize Blender-friendly offline artifact export (`.ply`) for all debugging inspection.

## Non-goals

- This plan does not fix sonar rendering, occlusion, or training stability by itself.
- This plan does not replace Poisson tuning or mesh extraction workflows.
- This plan does not require changing the existing renderer contract before visual inspection improves.

---

## Current gaps to close

### G0 — Surfels are exported as points, not surfels

`GaussianModel.save_ply()` writes positions, latent scales, and quaternions, but there is no companion export/view step that reconstructs visible surfel geometry. For initialization, `debug_multiframe.py` currently saves only `sonar_init_points.ply`, so the user cannot inspect the actual initialized Gaussian footprint state at all.

### G1 — Initial and trained states are not comparable in the same representation

The current output mix is inconsistent:

- initial state: point cloud + mesh
- trained state: final Gaussian PLY + mesh

That prevents direct visual comparison of surfel footprint evolution across stages.

### G2 — Frame identity is hidden by a combined wireframe export

`debug_multiframe.py` currently writes a single `pose_pyramids_wireframe.ply` with every training frame packed into one `LineSet`. Even with color differences, it is hard to tell which wireframe belongs to which frame once the geometry overlaps.

### G3 — FOV reach is visually misleading

The current wireframe is drawn at `PYRAMID_DEPTH`, which is useful near the sensor but does not show the actual far limit of the sonar beam. We need an added artifact that shows the same FOV at full range without removing the existing near-field shape.

### G4 — There is no frame-centric visual debugging mode

To judge whether occlusion work is helping, we need to inspect surfels relative to an individual sonar frame: what lies inside that frame's FOV, what faces toward/away from it, and how initialized vs trained surfels differ near that frame.

---

## Proposed output contract

All visualizer outputs for a run should live under a dedicated subfolder, for example:

`<OUTPUT_DIR>/visualizer/`

### UX principle: minimal clicks to full scene

The user should be able to import the entire `visualizer/` folder into Blender in one go (or a small number of batch imports) and see the complete scene immediately. Per-frame inspection happens by clicking on named objects in Blender's outliner, not by navigating filesystem subfolders. Therefore:

- **Flat directory structure** — all PLYs live directly in `visualizer/`, no subfolders
- **Naming encodes identity** — filenames use prefixes like `000_`, `001_`, etc. so Blender's outliner groups them naturally when sorted alphabetically
- **One import = full scene** — select all PLYs in the folder, import, done

### UX principle: rendered images must be scrubbable

The original sonar dataset images live in one flat folder. Opening any image and holding the right-arrow key scrubs through them like a video — critical for quickly reviewing ~500 frames. The rendered sonar outputs must support the same workflow:

- **Separate `rendered/` subfolder** — all rendered sonar PNGs live in `visualizer/rendered/`, one per frame, sequentially named
- **Naming matches source order** — `000_<image_name>.png`, `001_<image_name>.png`, etc. so arrow-key scrubbing goes in frame order
- The rendered images are also referenced from the main `visualizer/` manifest, but they live in their own folder so the image viewer shows only images (not mixed with PLY files)

### Naming convention: shared stem across artifacts

All per-frame artifacts use the same `NNN_<image_name>` stem. When you're scrubbing rendered images and spot something odd at frame 042, you search for `042_` in Blender's outliner and immediately find the matching wireframes and FOV surfels. Zero mental mapping.

Recommended structure:

```text
visualizer/
  manifest.json

  # Per-frame 3D artifacts — stem is NNN_<image_name>
  000_<image_name>_wireframe_near.ply
  000_<image_name>_wireframe_full_range.ply
  000_<image_name>_surfels_in_fov.ply      # size-aware: includes boundary surfels
  001_<image_name>_wireframe_near.ply
  001_<image_name>_wireframe_full_range.ply
  001_<image_name>_surfels_in_fov.ply      # size-aware: includes boundary surfels
  ...

  # Rendered sonar images — same stem, separate folder for scrubbing
  rendered/
    000_<image_name>.png
    001_<image_name>.png
    ...

  # Global surfel state snapshots
  surfels_initial_state.ply
  surfels_after_stage1.ply
  surfels_after_stage2.ply
  surfels_after_stage3.ply

  # Sampled glyph exports (all surfels, not per-frame)
  glyphs_initial_centers_full.ply
  glyphs_initial_sampled.ply
  glyphs_initial_front_faces.ply
  glyphs_initial_back_faces.ply
  glyphs_stage2_sampled.ply
  glyphs_stage2_front_faces.ply
  glyphs_stage2_back_faces.ply
  glyphs_stage3_sampled.ply
  glyphs_stage3_front_faces.ply
  glyphs_stage3_back_faces.ply
```

`manifest.json` should be a **narrow artifact manifest**, not a second general run metadata dump.

It should record at minimum:

- visualizer export root
- commit SHA
- frame index -> image name -> artifact file paths
- stage artifact file paths
- glyph sampling/export policy
- whether scales/rotations are latent or activated in each file
- the exact FOV-membership rule used for `_surfels_in_fov.ply`
- references to existing run metadata files already written elsewhere when present, e.g. checkpoint metadata source, `run.log`, `loss_log.csv`, `cfg_args`, `cameras.json`, synthetic dataset/gate manifest

It should **not** duplicate large run provenance already stored elsewhere unless that data is required to interpret the visualizer artifacts directly.

### Blender-first export policy

Because the only inspection tool is Blender, the artifact contract optimizes for minimum-click importability:

- **Flat folder for 3D** — all PLYs in one directory, no subfolders. Select-all → Import PLY brings the entire 3D scene in at once.
- **Alphabetical grouping via naming** — `000_*`, `001_*`, etc. sort together in the outliner. Click any wireframe to isolate that pose; the corresponding `_surfels_in_fov.ply` is right next to it.
- **Scrubbable rendered images** — rendered sonar PNGs live in `visualizer/rendered/` with sequential numbering (`000_<name>.png`, `001_<name>.png`, ...). Open any image, hold the arrow key, and scrub through all frames like a video — same workflow as browsing the raw sonar dataset folder.
- **Front/back face separation** — exported as distinct files so they can be toggled independently in the outliner.
- **Blender-safe payloads only** — core inspection must rely on filenames, separate files, geometry, and vertex colors; do not depend on custom PLY attributes for anything essential.
- **Deterministic filenames** — stable across repeated runs so visual diffs are easy.

---

## Visualizer design

### V1 — Export actual surfel states at key stages

`debug_multiframe.py` should export Gaussian-state PLYs, not just point clouds/meshes, at the stages that matter for visual debugging:

- immediately after `create_from_pcd(...)`: `surfels_initial_state.ply`
- after Stage 1: `surfels_after_stage1.ply`
- after Stage 2: `surfels_after_stage2.ply`
- after Stage 3 / final: `surfels_after_stage3.ply`

This creates a consistent surfel-state sequence that the visualizer can read without guessing.

### V2 — Add a surfel glyph builder that shows footprint + orientation

Add a visualization utility module that converts stored Gaussian parameters into visible geometry.

Recommended default glyph:

- an oriented ellipse ring lying in the surfel plane
- a short normal stem from the surfel center
- optional tangent cross for debugging local axes

The geometry must be built from the *activated* parameters:

- scales from `exp(scale_i)`
- rotation from normalized quaternion

The glyph builder should derive:

- center
- normal direction
- two in-plane tangent directions
- equivalent radius summary (`sqrt(s1 * s2)`) for filtering/coloring

This gives a readable answer to "where is the surfel, how big is it, and which way is it facing?"

### V2b — Add explicit two-sided surfel visualization

Yes, the visualizer should show both sides of a surfel. But the representation must be described carefully:

- what we can show directly from surfel state is the `+normal` side and the `-normal` side
- that is a reliable local front/back visualization
- it is **not** a guaranteed global object-level "outer vs inner" classification without a separate watertight/topology inference step

Recommended export modes:

- `front_faces`: geometry/color for the `+normal` side of sampled surfels
- `back_faces`: geometry/color for the `-normal` side of sampled surfels
- `double_sided_glyphs`: optional combined artifact with both sides present

Recommended geometry for Blender:

- use a very thin disc/slab or twin offset ellipse surfaces rather than lines only
- offset the front/back faces slightly along the normal so both remain visible when imported
- use distinct colors for front/back sides so flipping errors are obvious

This is important because a line-only ellipse shows footprint, but it does not clearly tell you which side is "front" once you orbit around it in Blender.

### V3 — Handle density with two representations, not one

Rendering every surfel as a multi-segment ellipse can become unreadable and heavy. The plan therefore keeps two complementary representations:

- **Full population:** center points for all surfels
- **Inspectable subset:** oriented glyphs for a deterministic filtered/sample subset

Default sampling policy:

- deterministic seed
- configurable max glyph count
- optional filters by opacity percentile, equivalent radius percentile, and selected-frame FOV membership

This preserves context while keeping the size/orientation view usable.

### V4 — Split wireframes into one file per sonar frame

Replace the current single combined wireframe with per-frame files, all in the flat `visualizer/` directory.

For each training frame, export 3D artifacts into `visualizer/`:

- `<idx>_<name>_wireframe_near.ply` — near-field pose wireframe
- `<idx>_<name>_wireframe_full_range.ply` — full-range FOV wireframe
- `<idx>_<name>_surfels_in_fov.ply` — glyph PLY of surfels that overlap this frame's sonar FOV. Uses **size-aware** FOV membership: a surfel is included if any part of it reaches into the FOV (`fov_margin + surfel_radius > 0`), not just if its center is inside. This catches boundary surfels that are centered outside the FOV but large enough to contribute to the rendered image — exactly the artifacts that cause edge problems.

And the rendered sonar image into `visualizer/rendered/`:

- `<idx>_<name>.png` — forward-projected sonar image from `render_sonar()`. Lives in a dedicated image-only subfolder so the user can open any one and hold the arrow key to scrub through all frames like a video.

All artifacts share the same `NNN_<name>` stem. Spot something at frame 042 while scrubbing renders → search `042_` in Blender's outliner → instantly find the matching wireframes and FOV surfels.

Together these let the user answer: "the renderer produced this image — do the underlying surfels in the FOV actually support it?"

The combined all-frame wireframe can remain as an optional compatibility artifact.

### V5 — Add, do not replace, the full-range FOV wireframe

The existing short wireframe is still useful because it keeps the local pose readable. We keep it.

In addition, create a second wireframe for each frame whose depth equals the sonar's actual far limit.

Per pose there must always be **two** wireframes:

- near wireframe depth: existing `PYRAMID_DEPTH`
- full-range wireframe depth: `sonar_config.range_max`

The full-range artifact should preserve the same FOV shape and orientation, just extended to the full sonar reach.

### V6 — Export frame-relative diagnostic data for occlusion debugging

For each training frame, the glyph export should optionally expose per-surfel diagnostics using Blender-safe channels only:

- separate files when a diagnostic changes object membership
- vertex colors when a diagnostic is best shown as a scalar field on the exported geometry

Do **not** depend on arbitrary custom PLY attributes for core inspection, since Blender import behavior for those is not the primary contract.

Diagnostics to expose:

- in/out of sonar FOV
- range to sonar
- facing score `dot(normal, dir_to_sonar)`

This metadata makes several renderer failure modes visible in Blender:

- wrong-facing surfel clusters (facing score near -1)
- oversized surfels crossing beam boundaries
- geometry that exists only behind the visible front surface
- initialization/training disagreement near a specific sonar view

---

## Implementation plan

### P1 — Factor visualization helpers into a reusable module

Add a utility module, e.g. `utils/visualization_utils.py`, with helpers for:

- quaternion -> rotation matrix / normal / tangent axes
- surfel glyph line or mesh construction
- per-frame wireframe generation
- manifest writing
- deterministic glyph sampling/filtering
- Blender-safe color encoding helpers

This avoids duplicating geometry code across `debug_multiframe.py` and any future consumers.

### P2 — Export stage-aligned surfel states from `debug_multiframe.py`

Patch `debug_multiframe.py` to emit surfel-state PLYs at the key checkpoints listed in V1, and move wireframe export into the new helper module.

Also rename the visual outputs so file intent is obvious:

- `sonar_init_points.ply` remains for raw point initialization context
- new `surfels_initial_state.ply` gives the actual initialized Gaussian state

### P3 — Export per-frame wireframes, FOV surfels, and rendered images

For every training frame, write 3D artifacts into `visualizer/` and rendered images into `visualizer/rendered/`:

- `visualizer/000_sonar_1765233595134_wireframe_near.ply`
- `visualizer/000_sonar_1765233595134_wireframe_full_range.ply`
- `visualizer/000_sonar_1765233595134_surfels_in_fov.ply`
- `visualizer/rendered/000_sonar_1765233595134.png`

All share the `000_sonar_1765233595134` stem. Rendered images live in their own folder for arrow-key scrubbing.

### P4 — Export sampled surfel glyph artifacts for offline inspection

Write sampled glyph PLYs for the main stages so a user can inspect footprint/orientation without launching the GUI.

Recommended first-pass artifacts:

- `glyphs_initial_centers_full.ply` — all surfel centers (full population, no glyphs)
- `glyphs_initial_sampled.ply`
- `glyphs_initial_front_faces.ply`
- `glyphs_initial_back_faces.ply`
- `glyphs_stage2_sampled.ply`
- `glyphs_stage2_front_faces.ply`
- `glyphs_stage2_back_faces.ply`
- `glyphs_stage3_sampled.ply`
- `glyphs_stage3_front_faces.ply`
- `glyphs_stage3_back_faces.ply`

These should be deterministic so repeated runs are comparable.

### P5 — Add minimal contract tests for export correctness

Add lightweight tests for the visualization contract. These are not aesthetics tests; they verify that the generated artifacts actually encode what the plan says they encode.

Recommended test set:

- one near wireframe and one full-range wireframe are produced per selected training frame
- full-range wireframe corner distances match `range_max` within tolerance
- glyph builder uses activated scales and normalized quaternions
- front-face and back-face exports lie on opposite sides of the surfel normal axis
- `surfels_initial_state.ply` exists and uses the same state schema as later stage exports
- manifest entries resolve to real files
- manifest references to existing run metadata resolve when those files are present
- core visualizer interpretation does not require custom PLY attributes

---

## Execution order

| Step | What | Files touched |
|------|------|---------------|
| P1 | Write visualization helpers and glyph builder | `utils/visualization_utils.py` |
| P2 | Export stage surfel states from multi-frame debug run | `debug_multiframe.py` |
| P3 | Export per-frame wireframes, FOV surfels, and rendered images | `debug_multiframe.py`, `utils/visualization_utils.py` |
| P4 | Export sampled glyph PLY artifacts | `debug_multiframe.py`, `utils/visualization_utils.py` |
| P5 | Add tests for export contract | `tests/test_sonar_visualizer_*.py` |

---

## Exit criteria

This plan is closeable when:

1. A run produces stage-aligned surfel-state PLYs for initial and trained checkpoints.
2. A run produces per-frame wireframes, FOV surfel glyphs, and rendered sonar images in a flat `visualizer/` directory.
3. The full-range wireframe visibly extends to `sonar_config.range_max` while the original near wireframe remains available.
4. Sampled surfel glyph PLYs are generated deterministically with visible size/orientation.
5. Front-side and back-side sampled surfel exports are available for Blender inspection.
6. `visualizer/manifest.json` exists as a narrow artifact index and references existing run metadata instead of duplicating it.
7. Manifest + contract tests pass.

---

## Deliverables

- New plan document: `plans/PLAN_SONAR_VISUALIZER_OVERHAUL_2026-03-10.md`
- New reusable visualization helper module: `utils/visualization_utils.py`
- Stage-aligned surfel-state exports from `debug_multiframe.py`
- Per-frame wireframe, FOV surfel, and rendered sonar exports (flat directory)
- Sampled surfel glyph PLY exports
- Separate sampled front-side/back-side surfel-face PLY exports
- Narrow visualization artifact manifest per run, with references to existing metadata sources
- Basic export contract tests

---

## Recommended defaults

- Keep the existing short wireframe exactly as the local pose cue.
- Add a separate full-range wireframe instead of stretching/replacing the current one.
- Flat `visualizer/` directory — all PLYs in one folder, no subfolders. Per-frame files share a `NNN_` prefix so they group alphabetically in Blender's outliner.
- Use ellipse-ring + normal-stem glyphs as the default quick-look surfel representation.
- Export separate front-side and back-side sampled surfel faces for Blender as the default two-sided inspection view.
- Keep the visualizer manifest narrow; link to existing run metadata instead of duplicating it.
- Use filenames, separate files, geometry, and vertex colors as the Blender contract; avoid custom PLY attributes for anything critical.
- Use deterministic sampled glyph exports for readability, while preserving full center-point exports for context.
- Make per-frame files the primary artifact; merged wireframes become optional compatibility output only.
