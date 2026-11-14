# Frozen numerical revision 371

Frozen after reproducing the three old failures and before any revised scoring.
This is one change of numerical support/depth decisions, not an acoustic model.
area_r4, centre admission, positive-reflectance opaque material, 16x16 area-midpoint
sampling, cell-average event scaling, event caps and the compiled depositor remain.
337's published opt-in intrinsic_screen_mass_v1 fallback is imported unchanged as
local test plumbing, so CPU and compiled paths use a declared identical footprint.

## Stored geometry and predicates
c, u, v are the actual sonar-local coordinates/tangents supplied to visibility,
converted exactly to binary64. h=sqrt(2*pi)*s/2 is evaluated once in binary64 and
its stored binary value defines support. The plane normal is cross(u,v), with no
normalization needed. These decoded numbers, not unknown pre-quantization geometry,
define material. No float32 coordinate uncertainty is reinterpreted as material
coincidence. In particular two distinct stored layers 10 micrometres apart remain
distinct; no range-bin or acoustic merging rule is used.

For a source sample P and blocker centre C, normal N, define A=N.C, B=N.P,
D=N.(P-C). The ray-plane intersection is (A/B)P. It is forward when A,B have the
same nonzero sign, and nearer exactly when sign(B)D>0. A tie requires D=0.
No metre-valued depth allowance is divided by incidence. Support uses
Q=A(P-C)-DC and tests -h|B| <= sign(B)u.Q < h|B| (and v), so neither support nor
ordering divides by B. Tangent support is half open [-h,h) on both axes.

## Floating error treatment
Float64 is a fast sign filter, not the authority at a sign ambiguity. Each normal
component has envelope |u_a v_b|+|u_b v_a|. Sample envelope is |c|+|alpha h0 u|+
|beta h1 v|. Dot-product envelopes Aenv/Benv and the displayed product-sum
expressions in ray_depth_layers.py bound the residual/support expressions.
Use gamma_128=128 eps64/(1-128 eps64): each scalar expression has at most 64
rounded arithmetic operations, including point/normal construction, and the extra
factor allows evaluation of the envelopes. This assumes finite normal-range
floating values (our metre-scale fixtures satisfy it); overflow/subnormal
conditioning is outside this numerical contract. A necessary inequality certainly
false rejects a candidate. Any remaining support, positivity, denominator or depth
comparison within its envelope is recomputed with fractions.Fraction, treating all
stored c/u/v/h numbers and rational midpoint coordinates as exact. Exact plane-pair
coplanarity is cached to avoid repeating point predicates on true duplicates.
The filter never enlarges support or returns an uncertain comparison as a tie.

A source supplies its own constructed sample; a source plane through the sonar
is unavailable. Source identity here is the quadrature's array index only; no
reference identities, parent surface, cube or mesh labels enter visibility.

## Frame invariance and differentiation
Exact predicates on geometry transformed by an exact common rigid transform are
frame invariant geometrically. Float32 transformation/re-encoding changes stored
material; no bitwise invariance is promised at discontinuities. Test actual common
rigid transforms and permutations, disclosing discrepancies. Dot-coordinate
support inherits tiny decoded-frame orthogonality errors; no physical frame-error
bound is claimed. Exact fallback resolves arithmetic ambiguity, not input error.

Visibility is explicitly computed under no_grad. Finite midpoint visibility is
piecewise constant; its derivative is zero away from switches and undefined at
switches. Smooth image tests can establish emission/projection/footprint gradients
only. They cannot certify the continuous moving-shadow boundary derivative.
Support, centre admission, top-k, alpha cutoff, conditioning and clamps remain
separate discrete or piecewise paths. Finite within-cell sampling and assigning one
visibility mean to every event leave shadow position unresolved; the compiled
footprint can paint neighbouring columns. Alpha/depth/normal/regularizer outputs
remain unsupported for this rule. No whole-scene throughput or acoustics claim.
