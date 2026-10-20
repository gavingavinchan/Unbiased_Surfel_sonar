# Convention gate 019: selected native conventions

Internal sonar axes are optical: X right, Y down, Z forward. Azimuth is
`-atan2(X,Z)` (left positive), elevation is `atan2(Y,hypot(X,Z))` (down positive),
and range is Euclidean slant range in metres. Conventional transforms act on
columns. Camera storage is `S = W2C.T`; row points multiply `S[:3,:3]` directly.
Scale world points and camera translation before composing a metric mount:
`S_sonar = S_camera_metric @ E.T`. Never scale the physical lever arm twice.

| Native selection | Archived sphere/cube | sim_cube_v1 PVC |
|---|---|---|
| `--sonar_range_origin` / `SONAR_RANGE_ORIGIN` | 0.2 m | 0 m |
| `--sonar_range_span` / `SONAR_RANGE_SPAN` | 2.8 m | 3 m |
| `--sonar_pixel_center_offset` / `SONAR_PIXEL_CENTER_OFFSET` | 0 | 0.5 |
| `--sonar_pose_mode` / `SONAR_POSE_MODE` | `poses_are_sonar` | `poses_are_sonar` |
| Resolution: `--resolution` / `SONAR_RESOLUTION` | 1 (native 200x256) | 1 (native 200x256) |

Both native entries use `build_sonar_config`; debug routes environment values via
`build_debug_sonar_config`. Legacy `sonar_range_min/max` remain fallback aliases;
explicit origin/span take precedence. The low-level `SonarConfig` accepts
`range_origin`, `range_span`, `pixel_center_offset`, and `pose_mode` directly.
Native pose mode defaults to `poses_are_sonar`, with exactly no mount. Select
`poses_are_camera` only for camera poses: it applies the recorded provisional
5-degree mount once. Explicit rigid extrinsics remain supported by low-level
geometry calls; supplying one in already-sonar mode raises an error. This is
software routing, not a calibration of the real vehicle.

For H rows, W columns, origin o, span L, offset d and full azimuth A:
`r=o+(row+d)L/H`, `theta=A/2-(col+d)A/W`. Masks do not change this grid.
Physical aperture includes its border cells: +60 degrees maps to col=-0.5 in
a centre grid and is valid. Render sampling clamps to the nearest border sample;
raw projection coordinates are retained. Pixel-centre bounds are not FOV bounds.

**Finite-precision endpoint policy:** azimuth/elevation membership uses signed
clearance `h-|angle|`, rounded nearest-even to ticks of **q=2^-19 radians**
(0.000109283 degree). Tick zero is a closed boundary. Thus unresolved values
within q/2=9.536743e-7 rad of an endpoint belong to its boundary cell, including
values infinitesimally outside the mathematical angle. Values outside that cell
are rejected. This explicit discretization applies equally to both signs,
projection, footprint projection, debug FOV, and visualization membership. Raw
angles/pixels and physical size margins are not rounded. Tests require **zero**
classification disagreements, retain exact endpoints and +/-4q nearby points,
and independently check +/-0.49q and +/-0.51q around the cell edge. A finite-radius
surfel still needs strictly positive physical clearance beyond its radius.
Range membership is the closed [origin,origin+span] interval; Z must be positive.

Camera retains its float64 source pose beside the ordinary float32 render pose.
Point-to-pixel projection uses the source pose and float64 arithmetic, preserving
floor-bin decisions near integer rows without enlarging bin tolerances. Render
values cast back to surfel dtype; no changes to the covariance-gradient repair.

Archived clean PNGs accumulate all 64 elevation rays using
`floor((r-.2)*200/2.8)`. The retained newer generator is a different acoustic
model: centre-sampled azimuth, front-envelope elevation collapse, and linear
floor/ceil range deposition at `(r-origin)H/span` **without subtracting 0.5**.
Its exported conventional R/t now compose/extract consistently from transposed
homogeneous matrices. Diagnostic outputs are new; retained benchmarks are intact.
