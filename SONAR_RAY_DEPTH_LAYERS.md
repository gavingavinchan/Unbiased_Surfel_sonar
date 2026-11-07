# 336 — frozen surface-sampled ray depth layers
Frozen before implementation/results of the new law. One rule; no tolerance, width,
gain or quadrature-resolution sweep. Pinned baseline c969d60 reproduced first.

## Representation and material semantics
Keep 333's area_r4 emission q_i=rho_i*(2*pi*sx_i*sy_i)*max(mu_i,0)^2/r_i^4,
its centre admission, event shares, top-k support cap and actual compiled splat
footprints. Switch is SONAR_VISIBILITY_RULE=ray_depth_layers; default remains legacy.
Only the visibility factor changes. No source IDs, rectangle/pipe membership,
reference mesh or world-space plane grouping enters the implementation.

Declare an opaque finite material cell with half widths sqrt(2*pi)*sx/2 and
sqrt(2*pi)*sy/2 in each patch's quaternion tangent frame. Its area equals the
Gaussian integral, but its support is NOT a Gaussian isodensity ellipse. Gaussian
widths still determine emission and receiver/image deposition. Gaussian tails do
not block rays. rho>0 means opaque material with reflectance rho; zero rho is
nonblocking. Translucency, coincident unequal reflectance and arbitrary unbounded
Gaussian material are unavailable. This is an extra explicit support semantics,
not a fact about the existing learned Gaussian surfaces.

## One frozen rule
For each centre-admitted patch, use a fixed 16x16 equal-area midpoint quadrature
on its finite cell. Trace the ray from the sonar origin through each sample.
Intersect this ray with every centre-admitted positive-rho patch plane, then test
the hit in that patch's finite rectangular tangent support. Compute t=(n.c)/(n.d)
from each plane, never patch-centre distance. The visible layer at a source sample
contains all covering intersections numerically coincident with its depth, provided
no distinct covering intersection is nearer. Its source weight is 1/multiplicity;
a hidden sample has weight zero. Average the 256 weights to obtain v_i. Existing
events receive event_value*v_i and go through the same actual splat deposition.

The ray bundle covers each admitted patch's material cell, not just the centre of
an azimuth/elevation bin. This avoids missing a thin patch between bin centres and
does not merge disjoint objects merely because they share a bin. It deliberately
uses ONE patch-average visibility fraction for all its Gaussian events: angular
variation inside its image footprint is not resolved. This limitation is part of
the rule, not fixed after measuring. The cell is not clipped again at the aperture:
centre admission and Gaussian event aperture semantics remain inherited. Partial
aperture support and receiver-image agreement with 284 remain unmatched diagnostics.

Subdivision of an isolated plane: disjoint cell interiors each see one front
intersection, so v=1 exactly regardless of tile count. Duplicate a material cell:
each sample sees two equal-depth cells, each gets v=1/2; identical splats sum to the
original. Coplanar partial overlap uses 1/2 only over overlap, without defining a
world-space surface group. These are discrete algorithm properties, not independent
physical evidence. Subdivision near visibility discontinuities has finite midpoint
quadrature error; no general 1% accuracy guarantee follows. Centre-evaluated q/A
and splat footprints still vary with tiling; compare isolated output to its OWN
admitted input, not to another finite density.

## Geometry error and tolerances (frozen before outcomes)
Geometry transformed from float32 input is evaluated in float64. Let
L=max(1 metre, max_i(||c_i|| + ||halfwidth_i||)); eta=32*eps_float32*L.
The factor 32 conservatively accounts for quaternion normalization, rotation,
subtraction and plane dot-products in the float32 input pipeline; it is a numerical
engineering allowance, NOT a measured uncertainty or a rigorous total error bound.
Assume each plane residual is within eta. First-order ray depth uncertainty is
eta/|n.d|. Pair depths count coincident if |t_i-t_j|<=tau_i+tau_j,
where tau_k=eta/max(|n_k.d|, 64*eps_float64). Edges have eta coordinate allowance.
A denominator <=64*eps_float64 is parallel and cannot supply an intersection;
the source sample still exists, but invalid-source rays are reported unavailable,
not silently promoted to a confident forward claim. Intersections must have t>0.
Grazing uncertainty can be large: distinct surfaces within the uncertainty band
are unresolved. There is no fixed metres/degrees merge distance or tolerance sweep.
Ray-blocking, finite-cell edges, rho=0 and depth ties are discontinuous boundaries.
No gradient, differentiability, optimization or training validity is claimed.

## Independent expectations, challenges and criteria
Same 27 x {1,4,16,64} x {8,16,32} fixtures on CPU fallback, CUDA fallback,
compiled splats. Save emitted/admitted/event/post-visibility/pre-clamp/final stages,
raw cap losses, arrays and source/binary hashes. Bypass is only an input image ledger.
Expected partial coplanar overlap uses analytic rectangular overlap areas per cell,
computed from fixture coordinates OUTSIDE the implementation: v=1-overlap/(2*A).
Expected images are original splats with independently prescribed v, not 284's
unmatched rectangle response. Do not equate partial_overlap_union's different
finite Gaussian representation to the two-rectangle image.
Additional fixed challenges: reverse/shuffle patch order; whole-scene rigid frame
transform; tilted-plane analytic depth; 5 mm parallel layers, 10 micrometre unresolved
layers, two intersecting planes (opposite tilts); coplanar disjoint cells. Compare
analytic half-plane visible fractions and coarse/fine quadrature discrepancies,
with near-tie unresolved bands disclosed. No new rule if a check fails.

Success requires <=1% isolated admitted-energy loss (front/tilted/grazing/thin/farfield),
nonzero weak return, <=1% duplicate/fully-hidden energy AND image change, <=1%
independent partial-overlap error and CPU/CUDA arithmetic/order/rigid invariance.
Unavailable requirements fail. Inherited aperture, support cap and compiled/fallback
footprint mismatches remain separate. Slot closes on all outcomes, no training or
real reconstruction successor. Request actual-Claude decision-change review.
