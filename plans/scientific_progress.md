# Scientific Progress Report: Sonar 2D Gaussian Splatting Extension

## Abstract
We extended 2D Gaussian Splatting to forward-looking multibeam sonar by introducing polar rendering, backward projection, metric scale alignment, and sonar-specific training constraints. The work adds sonar mode data flow, pose interpolation from camera trajectories, a learnable global scale factor, camera-to-sonar extrinsics, differentiable polar rendering with intensity modeling, size-aware field-of-view (FOV) constraints, and loss shaping for bright sonar returns. We also introduced mesh tuning workflows and dataset preparation guidelines to support real-world sonar reconstructions.

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
