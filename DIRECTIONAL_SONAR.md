# Directional Gaussian surface reference (experiment241)

This branch retains PR6 d63fd31 and its rasterizer pin unchanged. The new
`directional_sonar.py` is a separate PyTorch ray-plane forward model; it does
not execute the image-space packed-covariance CUDA path. Neither the old
analytic native proof nor a CPU run certifies this new path on the GPU.

A surfel is a finite ellipse in the plane defined by its quaternion, with
Gaussian alpha within 3 tangent standard deviations. Candidate rays sample
each native azimuth pixel twice and elevation at21 midpoint directions inside
the nominal20degree beam. The angular broadphase encloses the full ellipse.
Ray-plane hits are sorted by distance separately for every azimuth subray and
elevation. Front-to-back alpha transmittance attenuates farther material on
that direction only; it never consults the observed first-bright row. Different
elevations therefore preserve multiple range returns in one stored bearing.
The intensity includes two-sided absolute incidence, gain24 and exp(-.24r),
normalized cos(elevation) quadrature, a normalized7-bin sigma.65 range footprint
and finite9-pixel approximate1.5degree azimuth PSF. No nearest-return column
censor, nominal pipes, photo masks or simulator truth enter inference.

`directional_training.Session` follows `native_sonar_training.Session`'s fixed
density, metric scale, optimizer groups, exponential position schedule,
random-without-replacement sampler, 0.8L1+0.2(1-SSIM), and full-state restricted
atomic serializer. Its device-neutral ParameterDict replaces CUDA-hardcoded
GaussianModel allocation; this is an explicit reference implementation, not
compiled-renderer acceptance. Colour is one grey SH0 coefficient. Background
is a fixed TRAIN per-range mean. Positions stay in the historical1.5m camera-only
box and tangent sigmas are bounded2..100mm. The initializer uses inherited sonar
backprojection with uniform ambiguous elevations and034's unmodified PCA/seeded
normal recipe. Step1000 is fixed from the exposed150 recipe, not selected on
new EVAL or truth. All discs are exported at2sigma with no completion/filter.

Limitations: elevation quadrature convergence on thin real pipes is unproven;
near-grazing intersections/support boundaries/order changes are nonsmooth;
incidence and normals are weakly identifiable and no material scattering is
calibrated. Alpha transparency is an incoherent approximation. No multipath,
phase, beamformed sidelobes, range-zero correction, spreading law, historical
pixel-sign calibration, acoustic background model, or motion correction.
A fitted image or many disconnected discs does not establish a pipe structure.

Experiment-local `scripts/native_adapter.py` is only a payload for the existing
registered owner after scoped registration. The current two-scene/fixed-density
registry cannot express this real-data/strategy change; no alternate native
controller, source-free command, quota exemption or pause override is supplied.
CPU scientific runs, full-state next-update checks and minimal two-reflector
physics checks are separate from native CUDA release/resume acceptance.
