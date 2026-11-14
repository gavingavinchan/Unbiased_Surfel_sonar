# Numerical revision 409 (on top of frozen 371)

Two changes, frozen after reproducing the 371 failures and before any rescoring.

## 1. Surrogate gradient removed from the tested sonar path
`render_sonar` added `1e-6*(weight_sum - weight_sum.detach())` to the image. It is
zero in the forward pass but biases every image gradient (371: 2.07% xyz-x error).
It is now off by default; `SONAR_LEGACY_WEIGHT_SURROGATE=1` reproduces the old path.

## 2. Shared-frame float64 geometry + declared representation-error coincidence
`SONAR_VISIBILITY_PRECISION=shared64_409` (default; `stored32_371` = old path).

* Visibility geometry is evaluated once in float64 from the STORED float32 world
  parameters and the stored camera transform (world_view_transform_precise if
  present). The old path re-encoded sonar-frame centres to float32 first, which
  is a second, frame-dependent lossy conversion.
* Planes i,j are one stored surface (tie, reciprocal multiplicity, never
  mutual occlusion) iff unit normals are parallel within eps_i+eps_j and each
  centre lies on the other's plane within
  `delta_c_i + delta_c_j + (eps_i+eps_j) * (|c_i-c_j| + max(|h_i|,|h_j|))`.
  delta_c = 1.01*u32*|x_world|*|R|_2 (u32=2^-24: float32 rounding of each stored
  coordinate), eps = 16*u32 rad (quaternion re-encoding + float32 normalization:
  |dq|/|q|<=4u32, rotation angle <= 2|dq|/|q|, x2 margin).
* The relation is symmetric and plane-level (same decision for all 256 samples),
  uses no identity, label, range bin or acoustic width. At metre-scale
  coordinates the bound is ~1e-6 m or less, so 10 micrometre layers stay ordered
  by the exact 371 predicates (both signs). Declared input-precision limit:
  stored separations below the bound are not representable in float32 at that
  coordinate magnitude and are treated as one surface; the bound grows linearly
  with world-coordinate magnitude (10 um is lost beyond roughly |x|~30 m).
* All other predicates (half-open support, exact rational fallback) are 371's.
* Visibility remains under no_grad: image derivatives certify emission /
  projection / footprint only, never the moving-shadow boundary.

## Densification consequence
The removed surrogate was the ONLY position-gradient signal on fixtures whose image is
all zero (test RB-T08). Default-path densification statistics now come only from real
image gradients; RB-T08 pins the legacy flag and a new test asserts the default path
injects no gradient into an empty image. 409 training keeps densification off.
