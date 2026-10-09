# Sonar geometry contract (experiment 003, 2026-10-09)

`Camera.R` is R_camera_to_world. `Camera.T` is conventional world-to-camera t.
`Camera.world_view_transform` stores S = conventional world-to-camera matrix **transposed**.
With row-vector points, p_camera = p_world @ S[:3,:3] + S[3,:3].
Do not transpose that rotation on forward use. The inverse is
p_world = (p_camera - S[3,:3]) @ S[:3,:3].T.
Always use the actual stored transform so Camera scene offsets are respected.

All internal sonar/view points use optical axes: +X right, +Y down, +Z forward.
Azimuth is left-positive: theta = -atan2(x,z). Elevation is down-positive:
phi = atan2(y,sqrt(x*x+z*z)). Range is Euclidean slant range.
For FLU pose sources, world_T_optical = world_T_FLU @ FLU_T_optical, where
FLU_T_optical has columns [(0,-1,0),(0,0,-1),(1,0,0)].
This is a proper rotation, not an image mirror.

World positions and camera translations use the same arbitrary SfM units.
Scale both by `scale_m_per_world_unit` before composing a metric mount transform.
If E = conventional sonar_T_camera, S_sonar = scaled(S_camera) @ E.T.
E contains a proper rotation and metre translation. Invert S_sonar, then divide
world points by scale on echo initialization. Mount offsets must not be scaled twice.
`SonarExtrinsic(camera_to_sonar=E)` accepts a conventional rigid matrix, checks it,
and stores E.T. Its no-argument default is only a historical provisional mounting
assumption, **not calibration**. Historical .65 is likewise not calibrated.

A tensor is H range rows by W azimuth columns (normally 200 by 256).
Pixel centres are r = range_min + (row+.5)/H*(range_max-range_min),
theta = half_fov - (col+.5)/W*full_fov. The inverse subtracts .5.
`range_min` is the bin-grid origin, not a sensor's acoustic usability threshold.
`pixel_center_offset=0` explicitly requests the historical edge convention.
Resized images span the whole aperture: recompute centres at their actual H/W.
Do not slice the first W elements from a 256-column angle grid.
`meshgrid(indexing='xy')` already returns H by W; do not transpose it.

Gaussian tangent vectors are defined in world coordinates. Rotate them into the
sonar frame and scale their lengths to metres before projecting their footprint.
Lambertian normals remain world-space and use the correctly composed sonar origin.
FOV helpers use the same world-to-sonar transform; size margins/radii use metres.

Sonar echo elevation is unresolved by a 2D imaging sonar. Zero or seeded random
initialization is a prior, not a measured 3D surface. Polar projection correctness
and intensity fitting do not validate acoustic footprints or meshing. The old
pinhole TSDF path is not calibrated sonar surface fusion.

Tests: `tests/test_sonar_geometry_conventions.py`. Full CUDA analytic diagnostics,
controlled training/splits/checkpoints, before/after figures and point-to-GT
inspection live in harness experiment `003-surfel-sonar-geometry-fix`.
