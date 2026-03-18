# Scientific Progress Report: Sonar 2D Gaussian Splatting Extension

## Abstract
We extended 2D Gaussian Splatting to forward-looking multibeam sonar by introducing polar rendering, backward projection, metric scale alignment, and sonar-specific training constraints. The work adds sonar mode data flow, pose interpolation from camera trajectories, a learnable global scale factor, camera-to-sonar extrinsics, differentiable polar rendering with intensity modeling, size-aware field-of-view (FOV) constraints, and loss shaping for bright sonar returns. We also introduced mesh tuning workflows and dataset preparation guidelines to support real-world sonar reconstructions.

## 2026-03-18 Chunk-5 Runtime Addendum
Chunk 5 now has a first concrete late-normal runtime path in `debug_multiframe.py`. For each sparse Stage-1 anchor pixel with a valid posterior, the implementation explicitly queries the image-grid 4-neighborhood, evaluates Stage-1 multi-view evidence on those exact neighbors, forms an expected elevation
$$
e_{\mathrm{exp}}(u,v) = \sum_k p_{\mathrm{post}}(u,v,k)\, e_k,
$$
back-projects the center and four neighbors into world coordinates, and computes a finite-difference normal using
$$
n_{\mathrm{fd}} = \frac{(p_{\mathrm{right}} - p_{\mathrm{left}}) \times (p_{\mathrm{down}} - p_{\mathrm{up}})}{\left\|(p_{\mathrm{right}} - p_{\mathrm{left}}) \times (p_{\mathrm{down}} - p_{\mathrm{up}})\right\| + \varepsilon}.
$$

Those expected normals are then associated to visible surfels using the same projection/error gates already used by Chunk 4, and the supervision term added to the unified Stage-2/Stage-3 objective is the sign-ambiguous cosine loss
$$
\mathcal{L}_{\mathrm{normal}} = 1 - \left| n_{\mathrm{quat}} \cdot n_{\mathrm{expected}} \right|.
$$
The current rollout remains conservative in interpretation but no longer purely shadow-mode in implementation. Optional Chunk-5 densification now has a first active spawn path: persistent high-error candidates are ranked deterministically, a peak elevation bin is chosen from Stage-1 support, and a new surfel is initialized at the resulting world point. When the local 4-neighborhood geometry is valid, the spawn rotation is initialized from the expected finite-difference normal; otherwise it falls back to a deterministic camera-facing normal. The new point-utils contract also now admits an optional per-pixel elevation field so the sonar back-projection equations are consistent between the standalone geometry helpers and the training-loop Chunk-5 path, while `elevation_image=None` still uses the legacy collapsed zero-elevation geometry exactly.

## 2026-03-17 Chunk-5 Planning Clarification Addendum
The current Chunk-5 execution contract is now tightened to match the actual Stage-1 data layout used by `debug_multiframe.py`. Stage-1 posteriors are maintained on a sparse bright-pixel bank, so any expected-elevation finite-difference normal estimate in Chunk 5 must explicitly query the exact local image-grid neighborhood around each anchor pixel instead of assuming a dense posterior image exists. This removes an ambiguity that would otherwise make the late-normal supervision coverage ill-defined.

The planning contract is also now explicit that Chunk 5 adds a new normal-supervision term to the existing debug-training objective rather than reusing a pre-existing normal-loss block, and that optional densification should only be interpreted under the active ray-binned renderer semantics family. Reduced-budget synthetic gates are therefore only scientifically meaningful for Chunk 5 when they state their iteration-threshold overrides directly, so that the exercised late-normal or densification regime is unambiguous in the resulting artifacts.

## 2026-03-11 Visualizer Addendum
The debugging workflow now includes an explicit offline visualizer contract for surfel-state inspection in Blender. Instead of relying on latent Gaussian PLYs and a single merged pose wireframe, the run exports per-stage surfel states, per-frame wireframes, per-frame FOV surfel glyphs, and per-frame rendered sonar images under a shared `visualizer/` root.

Two practical geometry clarifications were required for this contract to become scientifically useful:

1. Near-range and full-range FOV wireframes must share the same angular boundary construction. If one artifact is built as a forward-depth rectangle while the other is built from constant-range beam corners, their apparent elevation envelope disagrees even when they encode the same nominal FOV. The corrected construction uses the same sonar-angle corner parameterization for both, differing only in radius.

2. Per-frame surfel inspection must distinguish between strict center-in-FOV membership and size-aware overlap membership. The current export keeps the size-aware overlap rule for diagnostic completeness,
$$
\operatorname{export}(i) = \mathbb{1}\left[m_i + r_i > 0 \right],
$$
where $m_i$ is the signed margin to the nearest FOV boundary and $r_i = \max(s_{u,i}, s_{v,i})$ is the activated in-plane surfel radius proxy, but the sampled subset is now prioritized by center-in-FOV first so the exported glyphs better reflect the frame's visually dominant supporting geometry.

Empirically, this new visualizer produced the first cube-dataset result that was qualitatively interpretable: in `output/debug_multiframe_synth_c_first6/`, six contiguous frames viewing the same cube face yielded a readable straight surfel band aligned with that face. The orientation field remains noisy, but the artifact is now good enough to reveal that many surfels face toward the observing sonar poses rather than away, which was previously impossible to assess reliably from the old exports.

Separately, the rasterizer backend now needs its additive accumulation path to be treated as part of the scientific contract rather than as an untracked local patch. The active sonar renderer already invokes the rasterizer with `additive_mode=True`, so preserving that backend source change in submodule history is necessary for reproducible behavior. In that mode, color accumulation uses
$$
w_i = \alpha_i
$$
instead of
$$
w_i = \alpha_i T_i,
$$
which intentionally removes front-to-back transmittance attenuation for the sonar-specific accumulation path.

## 2026-03-10 Renderer Stability Addendum
The active renderer-baseline remediation now includes a stability patch to the sonar accumulation path. The practical issue was not just incorrect visibility semantics but broken optimization plumbing: intermediate event volumes in `render_sonar` were being materialized through non-differentiable write patterns, so the photometric objective could become disconnected from surfel parameters during synthetic gate replays.

The current WIP renderer uses two additional numerical safeguards:

1. Ray-binned transmittance is accumulated in log space,
$$
\log T_i = \sum_{j < i} \log(1 - \alpha_j),
\quad
R_i = \exp(\log T_i) \cdot V_i,
$$
where $\alpha_j$ is the per-event opacity share, $V_i$ is the event value, $T_i$ is transmittance before event $i$, and $R_i$ is the visible return contributed by that event.

2. Event-volume outputs are sanitized before loss-side use,
$$
\tilde{X} = \operatorname{nan\_to\_num}(X; 0, 0, 0),
$$
for rendered return, range numerator, and support accumulators, so NaN/Inf values do not poison optimization or downstream mesh extraction.

Empirically, this restores finite training losses and successful smoke-scale synthetic runs for both sphere and cube datasets. However, the visual reconstruction quality has not yet shown a corresponding qualitative jump, so the renderer work remains scientifically incomplete despite the improved stability.

## 2026-03-12 Renderer-Semantics Interpretation Addendum

The scientific interpretation of recent synthetic runs now depends on an explicit renderer-semantic fingerprint. A post-v2 result is comparable to another run only when transfer mode, occlusion semantics, and footprint mode all match.

For active renderer-v2 runs, the forward model should be interpreted as:

1. Normal-based return strength with explicit transfer mode,
$$
q_i = \rho_i \cdot \tau(\mathbf{n}_i, \mathbf{d}_i) \cdot a(r_i),
$$
where $\rho_i$ is fixed-opacity return strength, $\tau$ is the configured Lambertian transfer law, and $a(r_i)$ is range attenuation.

2. Ray-binned acoustic occlusion resolved before elevation marginalization,
$$
R[a,e,r] = \sum_{i \in \mathcal{E}(a,e)} T_i \, q_i \, \delta(r-r_i),
\qquad
I[a,r] = \sum_e w_e \, R[a,e,r].
$$

This means pre-v2 additive no-occlusion results and post-v2 ray-binned results are not part of the same metric-comparison family. They may still be discussed historically, but they should not be used as direct quantitative comparators once the renderer fingerprint changes.

## 1. Problem Setting
The base system renders surfels via a pinhole camera model. Sonar imaging instead measures intensity as a function of azimuth and range with a narrow elevation beam. This introduces two core challenges: the geometry is polar rather than pinhole, and COLMAP camera poses are up-to-scale while sonar ranges are metric.

We use the Sonoptix Echo defaults: image width $W = 256$, height $H = 200$, azimuth FOV $\Theta = 120^\circ$, elevation FOV $\Phi = 20^\circ$, and valid ranges $r_{\min} = 0.2\,\mathrm{m}$, $r_{\max} = 3.0\,\mathrm{m}$.

## 2. Data and Pose Alignment
### 2.1 Pose Interpolation for Sonar Frames
Sonar frames are timestamped and interpolated from camera pose trajectories. For camera poses $(\mathbf{q}_i, \mathbf{t}_i)$ at timestamps $\tau_i$, sonar pose at time $\tau$ is computed by:

Translation (linear interpolation):
$$
\mathbf{t}(\tau) = (1 - \lambda)\,\mathbf{t}_i + \lambda\,\mathbf{t}_{i+1}
$$
Rotation (spherical linear interpolation):
$$
\mathbf{q}(\tau) = \mathrm{slerp}(\mathbf{q}_i, \mathbf{q}_{i+1}, \lambda)
$$
Here $\mathbf{t}(\tau)$ is the interpolated translation, $\mathbf{t}_i$ and $\mathbf{t}_{i+1}$ are adjacent camera translations, $\mathbf{q}(\tau)$ is the interpolated quaternion, $\mathbf{q}_i$ and $\mathbf{q}_{i+1}$ are adjacent camera quaternions, and $\lambda = \frac{\tau - \tau_i}{\tau_{i+1} - \tau_i}$ is the normalized time fraction.

We enforce a temporal threshold $\Delta t = 100\,\mathrm{ms}$, discarding frames with $\min |\tau - \tau_i| > \Delta t$.

### Pseudocode: Pose Interpolation
Algorithm 1 (Pose Interpolation)
1. Load COLMAP camera poses and timestamps.
2. For each sonar timestamp $\tau$:
   - Find bracketing camera timestamps $(\tau_i, \tau_{i+1})$.
   - If $\min |\tau - \tau_i| > \Delta t$, discard.
   - Compute $\lambda$ and interpolate $(\mathbf{q}, \mathbf{t})$.
3. Write a sonar-specific pose set for all accepted frames.

## 3. Sonar Geometry Model
### 3.1 Pixel-to-Polar Mapping
Given pixel $(u, v)$ with $u \in [0, W-1]$ and $v \in [0, H-1]$:
$$
\theta(u) = -\frac{u - W/2}{W/2} \cdot \frac{\Theta}{2},
\quad
r(v) = r_{\min} + \frac{v}{H}(r_{\max} - r_{\min})
$$
Here $\theta(u)$ is the azimuth angle, $r(v)$ is the range value, and $W$, $H$, $\Theta$, $r_{\min}$, $r_{\max}$ use the values in Section 1.

### 3.2 Polar-to-Cartesian (Sonar Frame)
Assuming zero elevation (fan beam):
$$
\mathbf{p}_s = \begin{bmatrix}
 r\cos\theta \\
 -r\sin\theta \\
 0
\end{bmatrix}
$$
Here $\mathbf{p}_s$ is the sonar-frame point, $r$ is range, and $\theta$ is azimuth.

### 3.3 Extrinsic Transform (Camera to Sonar)
Let $\mathbf{T}_{c\rightarrow s}$ be the fixed extrinsic transform. The sonar pose is:
$$
\mathbf{T}_{w\rightarrow s} = \mathbf{T}_{c\rightarrow s} \, \mathbf{T}_{w\rightarrow c}
$$
Here $\mathbf{T}_{w\rightarrow c}$ is the world-to-camera transform from COLMAP and $\mathbf{T}_{w\rightarrow s}$ is the world-to-sonar transform. We use a translation of $\mathbf{t}_{c\rightarrow s} = (0, -0.1, 0)\,\mathrm{m}$ (10 cm up in camera $+Y$-down convention) and a pitch of $\alpha = 5^\circ$ downward about the camera $x$ axis, plus a fixed axis permutation to map camera $(+X,+Y,+Z)$ to sonar $(+Y,+Z,+X)$.

## 4. Metric Scale Alignment
COLMAP produces poses up to an unknown global scale. We learn a scalar $s$ so that:
$$
\mathbf{t}_{\text{metric}} = s \cdot \mathbf{t}_{\text{colmap}}
$$
Here $\mathbf{t}_{\text{metric}}$ is the translation in meters, $\mathbf{t}_{\text{colmap}}$ is the translation in COLMAP units, and $s$ is the learned global scale.

A log-parameterization enforces positivity:
$$
 s = \exp(\alpha)
$$
Here $\alpha$ is the unconstrained log-scale parameter optimized with Adam at learning rate $0.01$ when scale learning is enabled.

In current debug runs, the scale optimizer is frozen and $s$ is held fixed (manual initialization) while other components are tuned. We used the following initial scales:
- Default training init: $s_0 = 1.0$.
- Legacy dataset init: $s_0 = 0.65$.
- R2 dataset init: $s_0 = 0.6127$.

### Curriculum for Scale Learning
We use staged optimization to reduce identifiability between scale and surfel positions.

Algorithm 2 (Curriculum)
1. Stage 1: Freeze surfels, optimize $s$ only (set to $0$ iterations in recent debug runs).
2. Stage 2: Freeze $s$, optimize surfels for $1000$ iterations.
3. Stage 3: Joint refinement for $1$ iteration (short stabilization pass).

## 5. Sonar Forward Rendering
### 5.1 World-to-Sonar Transform
Given a world point $\mathbf{p}_w$, the sonar-frame point is:
$$
\mathbf{p}_s = \mathbf{R}_{w\rightarrow s}\,\mathbf{p}_w + \mathbf{t}_{w\rightarrow s}
$$
Here $\mathbf{R}_{w\rightarrow s}$ and $\mathbf{t}_{w\rightarrow s}$ are the rotation and translation of $\mathbf{T}_{w\rightarrow s}$, with $\mathbf{t}_{w\rightarrow s}$ scaled by $s$ when enabled.

### 5.2 Polar Projection
For each sonar-frame point $\mathbf{p}_s = (x, y, z)$:
$$
\theta = -\arctan2(x, z),
\quad
r = \sqrt{x^2 + y^2 + z^2},
\quad
\phi = \arctan2(y, \sqrt{x^2 + z^2})
$$
Here $\theta$ is azimuth, $r$ is range, $\phi$ is elevation, and $(x, y, z)$ are sonar-frame coordinates.

### 5.3 Intensity Model
Using surfel normal $\mathbf{n}$ and direction to the sonar $\mathbf{d}$:
$$
I = \sigma \cdot \max(0, \mathbf{n}^\top \mathbf{d})
$$
Here $I$ is sonar intensity, $\sigma$ is surfel opacity, and $\mathbf{d}$ is the unit vector from the surfel to the sonar origin.

### 5.4 Differentiable Splatting and Range Normalization
Each surfel contributes to neighboring pixels with bilinear weights $w_k$. The range image is normalized by intensity-weighted contributions:
$$
R(u,v) = \frac{\sum_k w_k I_k r_k}{\sum_k w_k I_k + \epsilon}
$$
Here $R(u,v)$ is the range at pixel $(u,v)$, $I_k$ and $r_k$ are the intensity and range of surfel $k$, $w_k$ are bilinear weights, and $\epsilon$ is a small stabilizer.

### 5.5 Validity and Artifact Masking
A surfel is valid if:
$$
\mathbb{1}[\mathrm{in\_fov}] = \mathbb{1}[|\theta| \leq 60^\circ] \cdot \mathbb{1}[|\phi| \leq 10^\circ] \cdot \mathbb{1}[0.2 \leq r \leq 3.0] \cdot \mathbb{1}[z > 0]
$$
Here $\mathbb{1}[\cdot]$ is the indicator function. We mask the top $m=10$ rows in the range image to suppress near-field artifacts. We also apply an intensity validity threshold $I > 0.01$ for sonar masks, and a preprocessing threshold of $10/255 \approx 0.0392$ for debug ground-truth images.

## 6. Backward Projection and Normal Estimation
### 6.1 Backward Projection
Given a range image $R(u,v)$:
$$
\mathbf{p}_s(u,v) = \begin{bmatrix}
 r(u,v)\cos\theta(u) \\
 -r(u,v)\sin\theta(u) \\
 0
\end{bmatrix}
$$
$$
\mathbf{p}_w = \mathbf{R}_{s\rightarrow w}\,\mathbf{p}_s + \mathbf{t}_{s\rightarrow w}
$$
Here $\mathbf{p}_w$ is the world point, and $\mathbf{R}_{s\rightarrow w}$, $\mathbf{t}_{s\rightarrow w}$ are the inverse of the sonar pose.

### 6.2 Normal Estimation (Finite Differences)
Let $\mathbf{p}_{i,j}$ denote the world point at pixel $(i,j)$:
$$
\partial_r \mathbf{p} = \mathbf{p}_{i+1,j} - \mathbf{p}_{i-1,j},
\quad
\partial_{\theta} \mathbf{p} = \mathbf{p}_{i,j+1} - \mathbf{p}_{i,j-1}
$$
$$
\mathbf{n} = \frac{\partial_{\theta} \mathbf{p} \times \partial_r \mathbf{p}}{\|\partial_{\theta} \mathbf{p} \times \partial_r \mathbf{p}\|}
$$
Here $\partial_r \mathbf{p}$ and $\partial_{\theta} \mathbf{p}$ are central differences along range and azimuth directions, and $\mathbf{n}$ is the unit normal.

## 7. Size-Aware FOV Constraints
Standard FOV checks use surfel centers only. We enforce that the surfel extent lies fully inside the FOV by using a margin:
$$
\mathrm{margin} = \min\big((60^\circ - |\theta|)r, (10^\circ - |\phi|)r, r - r_{\min}, r_{\max} - r\big)
$$
Let $\rho$ be the surfel radius (maximum scaling dimension). A surfel is valid if:
$$
\mathrm{center\_in\_fov} \wedge \mathrm{margin} > \rho
$$
Here $\rho$ is the surfel size proxy, and $\mathrm{center\_in\_fov}$ is the center-based FOV validity indicator. ($\wedge$ means AND)

We prune outside-FOV surfels every $100$ iterations in multi-frame debugging.

## 8. Losses and Training Enhancements
### 8.1 Base Photometric Objective
The base loss blends L1 and SSIM with weights $0.8$ and $0.2$ respectively:
$$
\mathcal{L}_{\text{base}} = 0.8\,\|I - I_{gt}\|_1 + 0.2\,(1 - \mathrm{SSIM}(I, I_{gt}))
$$
Here $I$ is the rendered image and $I_{gt}$ is the ground-truth sonar image.

### 8.2 Bright-Pixel Loss
We focus on the top percentile $p = 95$ of ground-truth intensities. Let $\mathcal{B}$ be the set of pixels above percentile $p$:
$$
\mathcal{L}_{\text{bright}} = \frac{1}{|\mathcal{B}|} \sum_{(u,v)\in\mathcal{B}} \left| I(u,v) - I_{gt}(u,v) \right|
$$
Here $|\mathcal{B}|$ is the number of bright pixels and $I(u,v)$ is the rendered intensity at pixel $(u,v)$. If $|\mathcal{B}| < 32$, the threshold falls back to the 50th percentile. The bright-loss weight is $w_b = 0.5$.

### 8.3 Peak-Aware Loss
We emphasize local maxima in the ground truth. Let $\mathcal{P}$ be the set of top-$K$ local maxima in $I_{gt}$:
$$
\mathcal{L}_{\text{peak}} = \frac{1}{|\mathcal{P}|} \sum_{(u,v)\in\mathcal{P}} \left| I(u,v) - I_{gt}(u,v) \right|
$$
Here $K$ is a configured peak count (treated as a tunable hyperparameter in experiments).

### 8.4 Anti-Collapse Regularizer
We penalize degenerate low-intensity outputs by comparing total predicted intensity with ground truth:
$$
\mathcal{L}_{\text{collapse}} = \max\left(0, \; M_{gt} - M_{pred} \right)
$$
Here $M_{pred} = \sum_{u,v} I(u,v)$ and $M_{gt} = \sum_{u,v} I_{gt}(u,v)$ are the total predicted and ground-truth intensity masses.

### 8.5 Combined Objective
The total objective is a weighted sum:
$$
\mathcal{L} = w_0 \mathcal{L}_{\text{base}} + w_1 \mathcal{L}_{\text{bright}} + w_2 \mathcal{L}_{\text{peak}} + w_3 \mathcal{L}_{\text{collapse}}
$$
Here $w_0, w_1, w_2, w_3$ are nonnegative weights, with $w_1 = 0.5$ in current bright-loss runs.

## 9. Mesh Extraction and Tuning
Mesh extraction uses the TSDF fusion output and we introduced Poisson tuning workflows (including a GUI-driven parameter loop) to address gaps.

### 9.1 TSDF Surface as a Zero Level-Set
We treat the extracted mesh as a zero level-set of the fused signed distance field:
$$
\mathcal{M} = \{ \mathbf{x} \in \Omega \mid \phi(\mathbf{x}) = 0 \}
$$
Here $\phi(\mathbf{x})$ is the TSDF value at point $\mathbf{x}$ and $\Omega$ is the integration volume.

### 9.2 Poisson Reconstruction Objective (Tuned)
For post-processing, we solve a screened Poisson system:
$$
\chi^* = \arg\min_{\chi} \int_{\Omega} \|\nabla \chi(\mathbf{x}) - \mathbf{V}(\mathbf{x})\|^2\, d\mathbf{x} + \lambda \int_{\Omega} \chi(\mathbf{x})^2\, d\mathbf{x}
$$
Here $\chi(\mathbf{x})$ is the implicit indicator function, $\mathbf{V}(\mathbf{x})$ is the oriented normal vector field, $\lambda$ is the screening weight, and $\Omega$ is the reconstruction domain.

We used: Poisson depth $= 9$, density quantile cutoff $= 0.02$, minimum opacity cutoff $= 0.05$, opacity percentile $= 0.2$, and scale percentile $= 0.9$ for point filtering before reconstruction.

### Pseudocode: Mesh Tuning Workflow
Algorithm 4 (Poisson Tuning Loop)
1. Run multi-frame sonar rendering to produce range maps.
2. Fuse range maps into a TSDF and extract $\mathcal{M}$.
3. Solve for $\chi^*$ with tunable depth and smoothing parameters.
4. Iterate parameters and refresh mesh outputs for evaluation.

### 9.3 Sonar TSDF Mismatch (Observation)
The current TSDF extraction path still uses Open3D pinhole intrinsics and pinhole ray integration, while sonar rendering produces a polar range image. This projection mismatch causes extracted meshes to extend beyond the sonar FOV even when surfels remain inside the FOV. A sonar-native TSDF plan (ray integration along sonar azimuth/elevation) is documented in the latest snapshots for future implementation once training stability is improved.

## 10. Diagnostics and Validation
We added diagnostics tied to scale learning, FOV constraints, and projection consistency.

### 10.1 Scale Sensitivity
We evaluate the loss as a function of scale:
$$
\mathcal{L}(s_k) = \mathcal{L}(I(s_k), I_{gt})
$$
Here $s_k$ is a candidate scale, $I(s_k)$ is the rendered sonar image at scale $s_k$, and $I_{gt}$ is the ground-truth sonar image. Candidate scales used in sensitivity checks: $s_k \in \{0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 2.0\}$.

We also monitor the log-scale gradient:
$$
\frac{\partial \mathcal{L}}{\partial \alpha}, \quad s = \exp(\alpha)
$$
Here $\alpha$ is the log-scale parameter and $s$ is the positive scale factor applied to translations.

### 10.2 FOV Coverage Ratio
We track how many surfels are visible under the size-aware FOV check:
$$
C = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\mathrm{in\_fov}_i]
$$
Here $N$ is the total surfel count and $\mathrm{in\_fov}_i$ indicates whether surfel $i$ satisfies the size-aware FOV constraint.

## 11. Limitations and Open Issues
We quantify remaining gaps in scale convergence, dataset alignment, and mesh bounds.

### 11.1 Scale Convergence Error
When calibration suggests a target scale $s^*$, the relative error is:
$$
E_s = \frac{|s - s^*|}{s^*}
$$
Here $s$ is the learned global scale factor and $s^*$ is the calibration-derived scale. In our runs, $s^* \approx 0.66$ while initial values used were $0.65$ (legacy) and $0.6127$ (R2).

### 11.2 R2 Dataset Non-Equivalence
The best-fit similarity alignment between legacy camera centers $\mathbf{x}_i$ and R2 camera centers $\mathbf{y}_i$ is:
$$
\min_{s,\mathbf{R},\mathbf{t}} \sum_i \| s\mathbf{R}\mathbf{x}_i + \mathbf{t} - \mathbf{y}_i \|^2
$$
Here $s$ is global scale, $\mathbf{R}$ is a rotation matrix, $\mathbf{t}$ is translation, and $i$ indexes matched frames.

### 11.3 FOV-Overflow Mesh Ratio
Mesh surface leakage beyond the sonar FOV can be quantified as:
$$
E_{\mathrm{fov}} = \frac{|\mathcal{M}_{\mathrm{out}}|}{|\mathcal{M}|}
$$
Here $\mathcal{M}_{\mathrm{out}}$ are mesh elements with $|\theta| > 60^\circ$ or $|\phi| > 10^\circ$, and $|\mathcal{M}|$ is the total mesh element count.

## 12. Summary of Added Techniques and Features
- Sonar mode pipeline with pose interpolation and sonar image ingestion.
- Sonar polar projection for rendering (forward projection) and backward projection for point initialization.
- Global scale-factor learning with log-parameterization and curriculum training.
- Camera-to-sonar extrinsic transform integration (10 cm offset, 5 degree pitch).
- Lambertian intensity model for sonar returns.
- Differentiable bilinear splatting and range normalization.
- Top-row masking ($m=10$) and intensity thresholding ($I>0.01$, GT preprocessing $I>10/255$).
- Size-aware FOV pruning and visibility constraints (120 degree azimuth, 20 degree elevation).
- Bright-pixel loss (percentile 95, weight 0.5, min pixels 32).
- Multi-frame debug tooling and mesh tuning workflows.
- Poisson mesh tuning with depth 9 and filtering quantiles (0.02, 0.2, 0.9).

## 13. Chunk 2 Cross-View Baseline (for Chunk 3/4 Targets)

To avoid overfitting decisions to qualitative inspection, we recorded a fixed-seed 8-train + 2-holdout baseline protocol and treat its metrics as the floor that Chunk 3/4 must beat.

### 13.1 Baseline runs (higher-budget protocol)

- Fixed opacity (primary comparator):
  - train loss mean: $0.032021$
  - holdout loss mean: $0.048669$
  - train SSIM mean: $0.8775$
  - holdout SSIM mean: $0.8601$
  - support fractions: $\mathrm{support}\ge2 = 0.4615$, $\mathrm{support}\ge3 = 0.0446$
  - median support: $1.0$
  - single-view dominance (train): $0.5606$
- Learnable opacity (secondary ablation):
  - train loss mean: $0.039490$
  - holdout loss mean: $0.060202$
  - train SSIM mean: $0.8455$
  - holdout SSIM mean: $0.8448$
  - support fractions: $\mathrm{support}\ge2 = 0.4630$, $\mathrm{support}\ge3 = 0.0439$
  - median support: $1.0$
  - single-view dominance (train): $0.5643$

### 13.2 Derived indicators

Generalization ratio:
$$
R_{\mathrm{gen}} = \frac{\mathcal{L}_{\mathrm{holdout}}}{\mathcal{L}_{\mathrm{train}}}
$$

For both higher-budget baselines, $R_{\mathrm{gen}} \approx 1.52$, indicating persistent holdout degradation.

Support depth indicators:
$$
S_{\ge k} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\mathrm{support}_i \ge k]
$$

With $k=3$, $S_{\ge3}\approx 0.044$, showing shallow multi-view reinforcement.

### 13.3 Strategic implication

- Stage-0 work (Chunk 2) is considered functionally stabilized.
- Remaining overlap-quality issues are expected to be addressed mainly by:
  - Chunk 3: overlap-aware frame sampling + bin-likelihood evidence,
  - Chunk 4: belief-to-geometry coupling and ID-keyed support retention/pruning.
- Therefore, Chunk 2 metrics above are treated as baseline comparators, not final quality targets.

## 14. Synthetic Dataset A Program Update (2026-02-15 to 2026-02-16)

We added a controlled synthetic sonar benchmark path to isolate geometry/projection correctness from real-data noise.

### 14.1 Dataset-A acceptance gate

The automated gate is evaluated as:
$$
G_A = G_{\mathrm{consistency}} \wedge G_{\mathrm{run1}} \wedge G_{\mathrm{run2}} \wedge G_{\mathrm{drift}}
$$
where:
- $G_{\mathrm{consistency}}$ checks backward-projection sanity (pixel round-trip and radial residual thresholds),
- $G_{\mathrm{run1}}, G_{\mathrm{run2}}$ check sphere residual/center thresholds on two independent training runs,
- $G_{\mathrm{drift}}$ checks repeatability deltas against fixed tolerances.

In canonical mode (`sonar_equivalent`), the latest full-quality gate passes with low radial error and near-zero run-to-run drift.

### 14.2 Pose-mode policy

For acceptance, we use sonar-pose contract mode:
$$
\mathbf{T}_{w\rightarrow c}^{\mathrm{export}} = \mathbf{T}_{w\rightarrow s}
$$
This matches current debug training behavior and avoids introducing extrinsic-path mismatch into the baseline gate.

An optional diagnostic mode is also implemented:
$$
\mathbf{T}_{w\rightarrow c}^{\mathrm{export}} = \mathbf{T}_{w\rightarrow s}\,\mathbf{T}_{s\rightarrow c}
$$
to stress-test camera-to-sonar extrinsic handling when that path is explicitly under test.

### 14.3 Open geometric coverage limitation

Current synthetic pose sampling is predominantly a single rough orbit, so observed views concentrate near an equatorial band. This can bias initialized surfel centers toward a cylindrical shell rather than uniformly covering spherical latitude. Planned mitigation is multi-orbit/random-shell viewpoint sampling with bounded radius and center-looking orientation constraints.

## 15. Synthetic Dataset C (Cube Vacuum) Update (2026-02-16)

We extended the synthetic benchmark from Dataset A (sphere) to Dataset C (cube) using the same acceptance-gate structure, with cube-surface metrics replacing sphere-radial metrics.

### 15.1 Dataset-C acceptance gate

The gate is:
$$
G_C = G_{\mathrm{consistency}} \wedge G_{\mathrm{run1}} \wedge G_{\mathrm{run2}} \wedge G_{\mathrm{drift}}
$$
where $G_{\mathrm{run}i}$ requires:
$$
\bar{d}_{\mathrm{surf}} \le 0.05\,\mathrm{m}, \quad d_{95} \le 0.10\,\mathrm{m}, \quad e_{\mathrm{center}} \le 0.03\,\mathrm{m}
$$
for run $i\in\{1,2\}$.

Observed canonical gate result (`sonar_equivalent`, multi-band poses):
- $G_{\mathrm{consistency}}=\text{true}$ with mean residual $0.098358\,\mathrm{m}$ and $p95=0.244904\,\mathrm{m}$.
- $G_{\mathrm{run1}}=\text{false}$ with $(\bar{d}_{\mathrm{surf}}, d_{95}, e_{\mathrm{center}})=(0.090046, 0.233642, 0.010489)\,\mathrm{m}$.
- $G_{\mathrm{run2}}=\text{false}$ with $(0.090045, 0.233642, 0.010489)\,\mathrm{m}$.
- $G_{\mathrm{drift}}=\text{true}$ (near-zero inter-run deltas).

Hence $G_C=\text{false}$ in the current implementation.

### 15.2 Additional ablations

Post-gate runs tested longer stage budgets and key toggles:
- attenuation off + zero elevation init: $(\bar{d}_{\mathrm{surf}}, d_{95})\approx(0.0853, 0.2139)\,\mathrm{m}$,
- learnable opacity ablation: $(0.0891, 0.2307)\,\mathrm{m}$.

Both remain above acceptance thresholds.

### 15.3 Current scientific interpretation

- Dataset C pipeline reliability is strong (deterministic behavior, reproducible metrics).
- The failure mode is geometric quality, not stochastic instability.
- Under the current Chunk-2-era feature set, results suggest a quality ceiling for cube face/edge recovery.
- The most probable next gain is from Chunk 3/4 components (overlap-aware likelihood evidence and belief-to-geometry coupling) rather than additional Stage-2-only budget increases.

## 16. Chunk 3 Stage-1 Runtime Update (2026-02-20 to 2026-02-21)

### 16.1 Runtime contract fixes applied

The Stage-1 likelihood path in `debug_multiframe.py` now matches the intended invalid-sample neutrality contract:

$$
\text{support\_mask}_{i,k} = \mathbf{1}[\text{sample\_valid}_i]
$$

instead of all-ones support. Invalid samples continue to use neutral evidence (`loglik=0`), but are now excluded from valid-row CE/entropy aggregation.

Device coupling was also removed from elevation bin-center construction:

$$
\text{device}(\text{elev\_bin\_centers}) = \text{device}(\mathbf{x}_{\text{gaussians}})
$$

so CPU/GPU mismatch hazards from hardcoded `"cuda"` are eliminated.

### 16.2 Resume/refresh state behavior

- Resume path now avoids double construction of pixel-logit parameters and `optim_elev`.
- Refresh/remap path is wired for Stage 2/3 with deterministic behavior:
  - trigger: `ELEV_BANK_REFRESH_INTERVAL`,
  - remap policy: `nearest|reset`,
  - rebuild criterion: optimizer is rebuilt iff any refreshed logit tensor shape changes.

### 16.3 Synthetic gate outcomes after Chunk-3 runtime changes

Dataset A (`S1`, canonical gate):

$$
G_A = \text{true}
$$

with run metrics around:

$$
(\bar d_{\mathrm{rad}}, d_{95}, e_{\mathrm{center}}) \approx (0.0103,\ 0.0320,\ 0.01635)\ \text{m}
$$

Dataset C (`S2`, canonical gate):

$$
G_C = \text{false}
$$

while still deterministic/reproducible, with current run metrics approximately:

$$
(\bar d_{\mathrm{surf}}, d_{95}, e_{\mathrm{center}}) \approx (0.08797,\ 0.22886,\ 0.02364)\ \text{m}
$$

Reproducibility repeat (`S3`) remains low-drift relative to `S1` (order `10^{-5}` m deltas in the tracked evaluator metrics).

### 16.4 Resume continuity (`S4`) evidence

Continuation flow on synthetic Dataset C (save checkpoint -> resume -> continue) restores Stage-1 runtime state consistently:

- schema: `checkpoint_schema_version = chunk3_stage1_v1`,
- active-frame fingerprint: restored and matched,
- sampler state: restored (`cursor=303`, `epoch=3` in observed run),
- pixel logits: restored (`restored=500`, `reset=0` in observed run).

This supports the claim that Chunk-3 Stage-1 checkpoint payloads are resume-stable for the tested continuation path.

### 16.5 Dataset-C center-error variance and mode-isolation checks

Let $e_c(s, m)$ denote `center_error_m` on Dataset C for seed $s$ and mode $m \in \{\text{off},\text{shadow}\}$.

Baseline-relative comparison (Chunk-2 comparator vs current Chunk-3 S2 reference) remains:

$$
e_c^{\text{base}} = 0.00654\ \text{m},\quad e_c^{\text{S2}} = 0.02364\ \text{m},\quad
\Delta_{\mathrm{rel}} = \frac{e_c^{\text{S2}}-e_c^{\text{base}}}{e_c^{\text{base}}} \approx 261.18\%.
$$

To isolate Stage-1 runtime influence (without activating Stage-1 weighted losses), paired off-vs-shadow sweeps were run.

Short-budget sweep (`SONAR_STAGE2_ITERS=250`, seeds $\{42,101,202\}$):

$$
\mu_{\text{off}} \approx 0.02802,\ \sigma_{\text{off}} \approx 0.00619,
\quad
\mu_{\text{shadow}} \approx 0.02841,\ \sigma_{\text{shadow}} \approx 0.00668.
$$

Full-budget parity sweep (`SONAR_STAGE2_ITERS=1000`, seeds $\{77,303,404\}$):

$$
\mu_{\text{off}} \approx 0.01850,\ \sigma_{\text{off}} \approx 0.01278,
\quad
\mu_{\text{shadow}} \approx 0.02061,\ \sigma_{\text{shadow}} \approx 0.01199.
$$

Paired differences $\Delta_s = e_c(s,\text{shadow}) - e_c(s,\text{off})$ were mixed-sign in both sweeps, rather than consistently positive.

Interpretation:
- current evidence does not support a strong claim that Stage-1 `shadow` plumbing is a dominant directional regressor by itself;
- Dataset-C error behavior is better characterized as seed-sensitive variance plus known cube-shape difficulty in the current pipeline.

### 16.6 Comparator hygiene correction (mesh vs surfels)

During seed `303/404` runs, a first-pass evaluator call used `mesh_after_stage3.ply`, yielding much larger center errors than historical surfel-based references.

For apples-to-apples comparison with prior Chunk-2/Chunk-3 gate numbers, evaluator input was corrected to `surfels_after_training.ply`. Corrected artifacts:

- `output/chunk3_seed_sweep_full/seed303_off/eval_surfel_surfels/cube_eval.json`
- `output/chunk3_seed_sweep_full/seed303_shadow/eval_surfel_surfels/cube_eval.json`
- `output/chunk3_seed_sweep_full/seed404_off/eval_surfel_surfels/cube_eval.json`
- `output/chunk3_seed_sweep_full/seed404_shadow/eval_surfel_surfels/cube_eval.json`

This correction restores metric comparability for Chunk-3-to-Chunk-4 handoff tracking.

### 16.7 Contract-parity status note (2026-02-23)

Chunk-3 implementation status can be summarized as:

$$
\text{status}_{\text{Chunk3}} = \text{infrastructure complete} \land \text{core-likelihood parity pending}.
$$

Implemented and verified components include Stage-1 mode gating, frame-key/fingerprint/schema resume contracts, frame-keyed pixel-logit registry with optimizer ownership, refresh/remap controls, and checkpoint payload wiring.

The remaining parity item is the Stage-1 likelihood core in the hot training loop: the current path is still an interim per-frame surrogate and has not yet been fully aligned to the overlap-neighbor, `back_project_bins`-driven multi-view evidence contract specified in the detailed plan.

This note is status-only and does not change mathematical intent or acceptance criteria.

Fast test evidence on current codebase:

- `python -m py_compile debug_multiframe.py utils/elevation_stage1_helpers.py`
- `pytest tests/test_elevation_stage1_core_contracts.py tests/test_elevation_stage1_checkpoint_contracts.py tests/test_elevation_stage1_smoke_modes.py -q`
- Result: `37 passed`.

## 17. Chunk-4 contract test formalization (2026-02-24)

Chunk-4 introduces explicit contract tests for belief-to-geometry coupling and support-by-ID lifecycle.

### 17.1 Association and coupling reduction contracts

The implemented helper path enforces per-expected-point nearest gated association:

$$
s_{ij} = \left(\frac{\mathrm{pix\_err}_{ij}}{\sigma_{\mathrm{pix}}}\right)^2 +
\left(\frac{\mathrm{depth\_err}_{ij}}{\sigma_{\mathrm{depth}}}\right)^2,
\quad
w_{ij} = \mathrm{clamp}(\exp(-0.5 s_{ij}), w_{\min}, 1).
$$

Coupling reduction is robust-Huber weighted over matched associations only:

$$
\mathcal{L}_{\mathrm{couple}} =
\frac{\sum_{m \in \mathcal{M}} w_m\,\rho_\delta\!\left(\lVert \mathbf{x}_{\mathrm{surfel},m} - \mathbf{x}_{\mathrm{exp},m} \rVert\right)}{\sum_{m \in \mathcal{M}} w_m + \epsilon},
$$

with zero-match frames constrained to contribute exact zero.

### 17.2 Persistent ID and support schedule contracts

Support state is keyed by persistent surfel ID and remains invariant to row reindexing after prune/reorder:

$$
\mathrm{EMA}_{sid}^{(t+1)} = \beta\,\mathrm{EMA}_{sid}^{(t)} + (1-\beta)\,\mathrm{raw}_{sid}^{(t)}.
$$

Effective count floors are encoded as:

$$
\mathrm{floor}_{\mathrm{eff}} = \min(\mathrm{config\_floor},\, \mathrm{diverse\_candidate\_count}),
$$

with hysteresis and grace checks covered by unit contracts.

### 17.3 Runtime/synthetic gate harness status

- `C4-T11`..`C4-T14` runtime smokes are present as opt-in tests (env-gated) to avoid default heavy execution.
- `C4-T15`..`C4-T16` synthetic matrix and continuation checks are present as opt-in tests (env-gated).
- `C4-T17` remains manual by definition and is represented as an explicit skipped placeholder test requiring artifact review.

Current local fast-suite result:

- `pytest tests/test_elevation_chunk4_coupling_contracts.py tests/test_elevation_chunk4_id_support_lifecycle.py tests/test_elevation_chunk4_checkpoint_contracts.py tests/test_elevation_chunk4_smoke_modes.py tests/test_elevation_chunk4_synthetic_matrix.py -q`
- `29 passed, 7 skipped`.

## 18. Chunk-4 coupling runtime integration checkpoint (2026-02-24)

### 18.1 Coupling term in the training objective

Chunk-4 coupling is now wired in Stage-2 and Stage-3 runtime loops using Stage-1 posterior caches (`p_post`) from the same iteration. For each sampled frame, expected geometry is formed as:

$$
\mathbf{x}_{\mathrm{exp},i} = \sum_{k=1}^{K} p_{i,k}^{\mathrm{post}}\,\mathbf{x}_{i,k}^{\mathrm{bin}},
$$

where $\mathbf{x}_{i,k}^{\mathrm{bin}}$ are back-projected bin points from `back_project_bins` and $p_{i,k}^{\mathrm{post}}$ is normalized per-pixel posterior mass.

The per-frame objective is now:

$$
\mathcal{L}_{\mathrm{frame}} = \mathcal{L}_{\mathrm{photo}} + \mathcal{L}_{\mathrm{stage1}} + w_{\mathrm{couple}}(t)\,\mathcal{L}_{\mathrm{couple}},
$$

with a linear warmup schedule:

$$
w_{\mathrm{couple}}(t) = w_0 + \min\!\left(\frac{t}{T_{\mathrm{warmup}}}, 1\right)(w_1 - w_0).
$$

Here $w_0 = \texttt{ELEV\_COUPLE\_WEIGHT\_START}$, $w_1 = \texttt{ELEV\_COUPLE\_WEIGHT\_END}$, and $T_{\mathrm{warmup}} = \texttt{ELEV\_COUPLE\_WARMUP}$.

### 18.2 Mode-gated runtime semantics

- `off`: coupling disabled.
- `shadow`: coupling diagnostics computed, but $w_{\mathrm{couple}}(t)$ is effectively forced to zero in loss aggregation.
- `active`: diagnostics computed and weighted coupling applied.

This preserves off-mode parity while enabling staged activation of belief-to-geometry enforcement.

### 18.3 Runtime stability and validation evidence

Logging teardown was hardened (stdio restoration before stream close) to avoid subprocess false-fail exits in runtime smoke harnesses.

Post-integration validation in conda env:

- `RUN_CHUNK4_RUNTIME_SMOKES=1 pytest tests/test_elevation_chunk4_smoke_modes.py -q` -> `10 passed`.
- `RUN_CHUNK4_SYNTHETIC_MATRIX=1 pytest tests/test_elevation_chunk4_synthetic_matrix.py -q` -> `2 passed, 1 skipped`.
- Full Chunk-4 set with runtime + synthetic opt-ins -> `33 passed, 3 skipped`.

### 18.4 Closure status update

Runtime integration now includes both coupling and support/prune enforcement paths in `debug_multiframe.py` (persistent IDs, by-ID support state, hysteresis/grace pruning logic, and Chunk-4 checkpoint payload handling).

Chunk-4 remains open only at the gate-evidence level (not wiring completeness).

## 19. Chunk-4 closeout gate results (2026-02-24)

Closeout artifacts were generated under:

- `output/chunk4_closeout/chunk4_gate_closeout_report_2026-02-24.md`
- `output/chunk4_closeout/chunk4_gate_closeout_report_2026-02-24.json`
- `output/chunk4_closeout/gate_logs/`

### 19.1 Contract/smoke ledger

- Fast contract suite: `29 passed, 7 skipped`.
- Runtime smokes (`C4-T11`..`C4-T14`): all pass.
- Synthetic matrix (`C4-T15`, `C4-T16`): pass.
- Manual visual panel (`C4-T17`): intentionally skipped placeholder pending human verdict.

### 19.2 Quantitative gate metrics

From parsed active run logs (`c4_s2_run1`) over the last 20% iterations:

- median match rate: $0.9465$ (passes $\ge 0.01$),
- coupling residual gate metric ($\mathrm{p95}$ over per-iter coupling $\mathrm{p95}$): $0.30195\,\mathrm{m}$,
- assoc weight tail range: $[0.662, 0.772]$ (inside required $[0.10, 1.0]$),
- off-mode parity: relative loss delta $0.0012866$ and absolute SSIM delta $0.0006394$ (both pass configured limits).

Thus the coupling residual gate misses by a small margin:

$$
0.30195 - 0.30 = 0.00195\,\mathrm{m}.
$$

### 19.3 Dataset-C directional movement

Against the Chunk-3 comparator (`output/debug_multiframe_synth_c_run1/eval_surfel/cube_eval.json`) using the closeout run (`output/chunk4_closeout/c4_s2_run1/eval_surfel/cube_eval.json`):

- mean surface error: $0.087973 \rightarrow 0.087923$ (unchanged/slightly improved),
- p95 surface error: $0.228855 \rightarrow 0.227167$ (improved),
- center error: $0.023639 \rightarrow 0.023750$ (slightly regressed).

### 19.4 Gate decision

Current closeout decision is **NO-GO** due to:

1. `C4-S2` synthetic cube gate overall fail,
2. coupling residual threshold miss ($0.30195 > 0.30$),
3. pending manual `C4-T17` visual verdict.
