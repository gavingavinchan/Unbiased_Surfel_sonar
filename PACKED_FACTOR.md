# Frozen packed-factor protocol (before repair)

Exact baseline: PR5 553e4ee35f31d5b010c617968898291d5fc70bdd, parent PR4 b82913cbf1004d809add64c41abb83a5f01dcfb4, base debug-multiframe-r2 d8dea36c5cd5ef691a0ccf5a241a4e53a49861a3. Native 9b6100ca7eefc7f7163bc0ef9b91582b318904b8.

## Derivation independent of Python decoder
Contiguous nine floats are forwarded unchanged by rasterize_points.cu and rasterizer_impl.cu. Native forward and backward load consecutive triples as Tu,Tv,Tw. GLM's three constructor vectors are columns, so GLM T[0],T[1],T[2] are exactly those same triples, despite Python calling them rows.
For homogeneous latent z=(s0,s1,1), the native two planes are k=x*Tw-Tu, l=y*Tw-Tv. Their cross product p satisfies k.p=l.p=0. Therefore s=p.xy/p.z and pixel=(Tu.z/Tw.z,Tv.z/Tw.z) for s=0; generally pixel=((Tu.z+Tu.xy.s)/(Tw.z+Tw.xy.s), (Tv.z+Tv.xy.s)/(Tw.z+Tw.xy.s)).
For affine Tw=(0,0,1), write H=[[A,mu],[0,0,1]]. Then pixel=A*s+mu and rho3d=s.s=(pixel-mu)^T (A A^T)^-1 (pixel-mu). Unit latent Gaussian thus has covariance A A^T. The source is the authority for storage, not the old decoder.

## Chosen representation and immutable targets
Condition raw Sigma with the UNCHANGED existing spectral conditioner C. Let L be lower Cholesky, C=L L^T. Store ROW-MAJOR H=[[L00,0,mu0],[L10,L11,mu1],[0,0,1]]. Both callers use this lower block; decoder returns A A^T (also valid for general affine A). Public arguments, units (pixel / pixel squared), shape, dtype, device and autograd stay intact. Normal outputs intentionally change. No jitter, variance bounds, loss, poses, opacity schedule or density change.
For C=[[2,.6],[.6,4]], L=[[sqrt(2),0],[.6/sqrt(2),sqrt(3.82)]]. Native expected covariance is exactly [[2,.6],[.6,4]]. Old packing gives [[2.18,.8292164976651151],[.8292164976651151,3.82]]. Means are independent translation.

## Native filtering and bounds
Actual rho=min(rho3d,2*||pixel-center||^2), alpha=min(.99,opacity*exp(-rho/2)), discarded below 1/255. This is the union of an elliptic Gaussian and isotropic screen filter, NOT one Gaussian with covariance C. Compare full filtered intensity against an independent covariance-solve expression and separately test unfiltered covariance.
Native compute_aabb uses t=(cutoff^2,cutoff^2,-1), f=t/dot(t,Tw^2), center_i=dot(f,Ti*Tw), extent_i=sqrt(max(1e-4,center_i^2-dot(f,Ti^2))). For affine input this is mu_i and cutoff*sqrt(Cii), except roundoff. Its scalar radius rounds up max(extents,cutoff*FilterSize); getRect expands to 16x16 tiles.
A second inherited issue is that cutoff=3 can exclude samples whose actual alpha exceeds 1/255 (unit-opacity support radius sqrt(2 log255)=3.329...). Freeze the test at the ACTUAL 1/255 threshold, not a conveniently smaller support. Repair only affine-precomputed footprint extents using row norms and that opacity threshold, retaining the existing center calculation and all ray/backward equations. This includes low-pass support and avoids subtracting nearly equal squared means for extent. General projective/camera packing and cutoff remain outside this repair.

## Frozen validation
64 rotated spectra per dtype, both correlation signs, anisotropy, tiny raw spectra and existing conditioning; both normal/forced LinAlgError, actual AST callers, scalar plus batches1/512/12001. 128 fixed pixel offsets per matrix, nonzero means and affine nonidentity transforms. Scaled covariance <=2e-6 float32 /1e-12 float64. Intensity |difference|<= absolute tail tolerance (2e-6 /1e-13) + relative (1e-5 /1e-10)*|expected|. Central FD covariance and mean norm-relative errors <=1e-3 /1e-6, away from conditioning kinks. Native backward plane VJP compared to autograd and FD. Footprint must contain every declared above-1/255 sample, with old failures retained. OOM/unrelated errors propagate. Inherited target assertions retain their strength; only their private storage reader changes to this explicitly versioned contract.

CPU equations are not compiled-kernel execution. A registered changed-source 024c build, analytic kernel and complete state release/restore proof remain mandatory before native use. No reconstruction-gain claim.

## Implementation clarification
The affine radius includes one additional pixel after ceil: getRect truncates its fractional-center upper bound after adding BLOCK-1, so ceil(extent) alone is not a universal integer-pixel guarantee. The original 1/255 support target and tolerances above are unchanged. Source token checks pin the CPU bounds oracle to the native expressions; no compiled CUDA check is claimed. All nine native plane/backward entries are compared separately, including the screen-center VJP.
