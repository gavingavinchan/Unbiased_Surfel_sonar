# Frozen deposition contract: intrinsic_screen_mass_v1

Frozen after baseline reproduction and source trace, before the one opt-in edit.
This is a numerical compatibility contract, NOT a measured acoustic response.
It deliberately adopts the existing compiled affine footprint, making the fallback
implement that law. The legacy fallback's ray-column/range-profile law is different
by design; the whole historical difference is not called a CUDA kernel bug.

Inputs: finite patch centres mu=(column,row), finite symmetric 2x2 covariance in
pixel², physical centre ranges, event patch indices, flattened column/elevation
ray IDs, already composed event returns q and event shares. Integer coordinates
are pixel centres. Projection uses its unchanged half-pixel offset before this
interface; no second offset is applied. Event column influenced admission and
visibility upstream; under this contract it does NOT re-centre/restrict deposition.
Event elevation is ray_id modulo B. All emission, event admission, top-k=16 support,
visibility and return/gain/range parameters stay fixed. Those caps apply upstream.

Condition covariance by symmetric eigendecomposition: clip eigenvalues to
[0.1,400] pixel² and raise the smaller to at least larger/100. No added covariance
low-pass term. For integer pixel p, d=p-mu, define
  rho = min(d^T S^-1 d, 2 d^T d), G = exp(-rho/2).
This is the MAXIMUM of intrinsic and screen Gaussians, not their convolution.
Screen variance is 0.5 pixel² (FilterInvSquare=2). With fixed linear opacity
l=1/16, retain G exactly where min(0.99,l*G) >= 1/255, else zero.
Equality is retained; crossing this hard threshold is discontinuous. Pixel clipping
is to the actual image, then Z=sum_in_image G; K=G/Z if Z>0 else all zero.
No edge-tail energy is lost if any pixel survives: clipping renormalizes the return.
There is no extra 2.5-sigma cutoff or deposition support cap. The existing native
padded affine tile radius conservatively contains this support; it is not a new law.

For event j of patch i in elevation b, deposit q_j*K_i into returns[:,b,:],
q_j*max(range_i,0.21)*K_i into range numerator, and share_j*K_i into support.
Add events linearly; no deposition alpha transmittance/early exit. Negative returns
are allowed algebraically but are not physical energy. Empty events or zero return
have zero return images (support may remain nonzero). No pixel passing cutoff means
zero deposit. Invalid inputs and other footprint normalization/linear-opacity
settings are outside this v1 opt-in, not silently reinterpreted.
Final render preserves elevation summation, gain, [0,1] clamp and near-row mask.
The callable independent scalar oracle is scripts/scalar_contract.py; it imports
no renderer/Torch/kernel helper. The correction is one opt-in fallback depositor;
compiled calls remain actual GaussianRasterizer calls, unchanged.

Frozen matrix: configs/events.json covers integer/quarter/half centres; isotropic,
rotated anisotropic, thin/grazing and conditioning-cap covariances; image edges,
outside/empty/zero returns. Exact inputs are float32 before being supplied to all
paths; the independent oracle evaluates those stored numbers in float64.
Also replay all 27 original fixtures x4 tilings x3 elevation counts; keep inherited
visibility/aperture/retiling failures separate. Full supplied cube: TRAIN33,76,186,
33,792 patches, nominal ranges/original floor/base subdivision, all three original
334 return/visibility settings; same captured event arrays feed every depositor.

Metrics: energy error |sum(A)-sum(O)|/sum|O| and image L1 sum|A-O|/sum|O|.
Zero denominator: both zero gives 0; nonzero candidate gives null plus explicit
zero-denominator failure. Nonfinite values always fail. Centroids use positive
image mass; undefined for zero/nonpositive mass. Peak amplitude relative to max|O|
and row-major argmax displacement are separate, with ties disclosed. Pass requires
<=0.01 energy AND L1 for every supported nonzero row and exact expected-zero return.
No centroid/peak threshold is invented. Three image channels and elevation volumes
are checked as well as final pre/post-mask images. No gradients or acoustic validity
are established. Close this slot on every outcome; no automatic successor.
