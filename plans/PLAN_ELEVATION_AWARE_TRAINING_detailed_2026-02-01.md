# Plan: Elevation-Aware Training with Back Projection (Detailed)

**Date/Time:** 2026-02-01 (updated 2026-02-06)
**Git Commit:** (not yet implemented)
**Status:** Detailed implementation plan aligned to latest base-plan decisions

---

## Goal

Implement elevation-aware sonar training in `debug_multiframe.py` with concrete staged code changes, aligned with the updated high-level decisions in `plans/PLAN_ELEVATION_AWARE_TRAINING_2026-01-28.md`:

- Loss priority: GT-anchored multi-view likelihood with elevation bins and annealing (arc-only Stage 1 removed).
- Normals path: weak explicit normal regularization early, stronger late.
- Correspondence policy: start with implicit multi-view consistency (no heavy explicit pixel matching).
- Physics defaults: fixed opacity for sonar and distance attenuation in rendering.

---

## Scope (Files to Change)

- `debug_multiframe.py` (primary integration, staged losses, schedule, logging)
- `utils/sonar_utils.py` (elevation-aware back-projection utilities)
- `utils/point_utils.py` (optional elevation path for range->points used by normals)
- `gaussian_renderer/__init__.py` (point projection helper, distance attenuation in sonar intensity)
- `arguments/__init__.py` (optional defaults for new elevation/physics flags)

---

## Dataset Preparation: Frame Filtering (Pre-Training)

This filtering is part of dataset preparation, not the training loop.

Context:
- Current `~500`-frame training subsets are sampled from `>2000` matched sonar+pose frames.
- Legacy and R2 datasets did not apply the filtering pipeline below when those subsets were created.

Procedure for future dataset builds:

1. Quality gate (hard reject)
   - Compute per-frame `valid_return_ratio` after top-row mask and intensity threshold.
   - Compute per-frame `p99_intensity` on valid returns.
   - Drop frames below minimum quality thresholds.
2. Pose dedup (remove near-identical frames)
   - For candidate pairs, compute `delta_pos` (center distance) and `delta_yaw`.
   - Cluster/drop near-duplicate poses; keep one representative per cluster.
3. Diverse subsample to target size
   - From remaining frames, select target count (e.g., 500) with greedy farthest-point sampling in pose space.
   - Use distance metric combining translation and heading difference.
4. Connectivity check
   - Ensure each selected frame has enough potential multi-view neighbors under baseline/view-angle constraints.
   - Replace isolated selections with neighbor-rich alternatives.

Output contract:
- Persist selected frame list (indices/names) with the dataset split.
- Training should consume this prepared list directly, without re-running heavy frame filtering logic.

---

## Resolved High-Level Decisions (Carried Into This Detailed Plan)

1. **Loss formulation**
   - Primary stage: elevation-bin likelihood with temperature annealing and GT supervision across overlapping frames.
   - Use robust normalized amplitude likelihood (percentile-normalized per frame, clipped log-evidence, validity masking, frame reliability weighting).
   - Belief-to-geometry coupling is mandatory: expected 3D points from elevation bins must directly update surfel geometry.
   - Arc-only Stage 1 is removed (self-referential risk when built from surfel-projected pixels).
   - Optional arc stabilizer remains available at low weight if needed for late training stability.
   - Baseline photometric and bright-pixel losses remain active.
2. **Normals path**
   - Early: keep explicit normal regularization very weak.
   - Late Stage 1+: compute normals using expected elevation and increase normal regularizer weight.
3. **Correspondence strategy**
   - Start implicit (Option D): do not build a global pixel-pixel correspondence structure.
   - If needed, add surfel-anchored explicit correspondence as a fallback.
4. **Sonar physics for current implementation**
   - Fixed opacity by default for sonar (`opacity=1.0`, learnable toggle optional).
   - Add stabilized distance attenuation in `render_sonar`:
     `intensity *= gain / (max(range, r0)^p + eps)`.
   - Default for raw sonar data: attenuation enabled, `p=2.0`, near-range floor `r0=0.35`.
5. **Dataset assumption (recorded)**
   - Sonar images used for this work are raw amplitudes (no onboard range/gain compensation in the saved data).
   - Therefore range attenuation remains enabled by default; ablation can disable it only for diagnostics.
6. **ROV mount extrinsic constants (recorded)**
   - Sonar is mounted `8 cm` behind camera, `10 cm` above camera, with `5 deg` downward pitch.
   - Canonical camera-frame translation for implementation: `[0.0, -0.10, -0.08]` (meters), with `+5 deg` pitch about camera X.
   - These constants are treated as fixed defaults for elevation-aware training and must be consistent across docs + code paths.
7. **Coordinate convention contract (resolved)**
   - Camera/view frame used for core projection math: `+X right, +Y down, +Z forward`.
   - Sonar image convention: columns encode azimuth (`left=+`, `right=-`), rows encode range (`top=near`, `bottom=far`).
   - If sonar-frame symbols are used (`+X forward, +Y right, +Z down`), conversion to camera/view frame must be explicit and isolated.
   - Transform storage convention: row-major `world_view_transform` semantics with translation in row 3 (`T[3, :3]`) for renderer/debug compatibility.

---

## Stage Plan

### Stage 0: Elevation-Aware Initialization

- Extend back-projection init to support non-zero elevation.
- Default: random elevation per valid pixel, uniform in beam range.
- Keep `zero` mode for ablation.

### Stage 1: Bin-Likelihood Elevation Update (Primary)

- Per selected pixels, optimize logits over K elevation bins (K=5-9 default).
- Build multi-view likelihood by projecting each bin candidate into overlapping frames.
- Anneal temperature to sharpen distributions over time.
- Convert pixel beliefs to expected 3D points and apply robust attraction to associated surfels.
- Retain only surfels with strong multi-view support.

### Stage 2: Optional Arc-Guided Densification

- For persistent high-error bright pixels, search elevation bins and spawn surfels at peak-score bins.
- Off by default.

---

## Data Structures

### Elevation bins

```python
def build_elevation_bins(sonar_config, k_bins, device):
    half = sonar_config.half_elevation_rad
    return torch.linspace(-half, half, k_bins, device=device)
```

### Pixel bank (subsampled)

```python
# Learnable logits are owned by a persistent parameter registry.
pixel_logits = nn.ParameterDict({
    str(frame_idx): nn.Parameter(torch.zeros([N_pix, K], device="cuda"))
    for frame_idx in selected_frames
})

pixel_bank[frame_idx] = {
    "rows": torch.tensor([...], device="cuda"),
    "cols": torch.tensor([...], device="cuda"),
    "logits_key": str(frame_idx),  # lookup key into pixel_logits
}

optim_elev = torch.optim.Adam(pixel_logits.parameters(), lr=ELEV_LOGIT_LR)
```

Use subsampled bright pixels (`ELEV_PIXELS_PER_FRAME`) to cap memory and runtime.

Lifecycle contract for pixel logits:
- Initialize `pixel_logits` once at Stage 1 start for the active frame set.
- If pixel-bank refresh is enabled, rebuild `(rows, cols)` and remap logits using configured mode (`nearest` or `reset`).
- After any add/remove/reshape of pixel-logit parameters, rebuild `optim_elev` param groups to avoid stale optimizer state.
- Save/load both `pixel_bank` metadata and `pixel_logits` + `optim_elev.state_dict()` in checkpoints.
- Default first pass: no refresh (`ELEV_BANK_REFRESH_INTERVAL=0`) for maximum stability/debuggability.

### Coupling/support buffers

```python
# Persistent surfel identity (stable across densify/prune/reorder).
surfel_ids = torch.arange(N_surfels, device="cuda", dtype=torch.long)
next_surfel_id = int(N_surfels)

# Support state is keyed by persistent surfel_id, not row index.
surfel_support = {
    "ema_by_id": torch.zeros([next_surfel_id], device="cuda"),
    "last_raw_by_id": torch.zeros([next_surfel_id], device="cuda"),
}

# Row lookup (rebuild after any topology change).
id_to_row = torch.full([next_surfel_id], -1, device="cuda", dtype=torch.long)
id_to_row[surfel_ids] = torch.arange(N_surfels, device="cuda", dtype=torch.long)
```

Point-to-surfel association is computed on-the-fly per iteration (no persistent global correspondence map).

Lifecycle contract for persistent surfel IDs:
- Densify: allocate fresh IDs `[next_surfel_id, ..., next_surfel_id + n_new - 1]`, append to `surfel_ids`, grow `ema_by_id/last_raw_by_id` with zeros, increment `next_surfel_id`, then rebuild `id_to_row`.
- Prune/reorder: update `surfel_ids = surfel_ids[keep_idx]` (or reordered view), then rebuild `id_to_row`.
- Update support by ID each iteration: `sid = surfel_ids[row]`; write to `ema_by_id[sid]` and `last_raw_by_id[sid]`.
- Checkpoint must persist `surfel_ids`, `next_surfel_id`, and support tensors keyed by ID.
- First pass keeps retired IDs in support tensors (no compaction) for simpler, safer bookkeeping.

---

## File-by-File Implementation Plan

### `utils/sonar_utils.py`

1. Add elevation-aware polar-to-cartesian helper:

```python
def sonar_polar_to_points(azimuth, elevation, range_vals):
    x = -range_vals * torch.sin(azimuth) * torch.cos(elevation)
    y = range_vals * torch.sin(elevation)
    z = range_vals * torch.cos(azimuth) * torch.cos(elevation)
    return torch.stack([x, y, z], dim=-1)
```

2. Extend `sonar_frame_to_points(...)` with elevation mode:
   - `elevation_mode = "zero" | "random" | "values"`
   - Preserve legacy behavior when elevation mode is disabled.
3. Ensure camera-to-sonar extrinsic defaults match recorded mount:
   - translation `[0.0, -0.10, -0.08]` meters in camera frame,
   - pitch `+5 deg` about camera X.
4. Enforce transform layout contract (row-major translation in row 3 for repo-compatible transforms) and avoid mixed row/column translation writes.
5. Add lightweight assertion helpers for convention checks (signs and axis mapping).

### `gaussian_renderer/__init__.py`

1. Add helper for arbitrary point projection into sonar pixel coordinates:

```python
def sonar_project_points(points_world, camera, sonar_config, scale_factor):
    # returns row, col, range_vals, in_fov
```

2. Update sonar intensity model in `render_sonar()` with stabilized distance attenuation:

```python
lambert = torch.clamp((normals * view_dirs).sum(dim=-1), min=0.0)
r_eff = torch.clamp(range_vals, min=SONAR_RANGE_ATTEN_R0)
atten = SONAR_RANGE_ATTEN_GAIN / (r_eff**SONAR_RANGE_ATTEN_EXP + SONAR_RANGE_ATTEN_EPS)
intensity = lambert * atten
```

3. Keep attenuation behind a config flag for ablation (`SONAR_USE_RANGE_ATTEN=1`).
4. Treat attenuation-enabled mode as the default because training data are raw (uncompensated) sonar amplitudes.
5. Add dynamic-range diagnostics (near saturation rate, far-range under-response).
6. Add convention assertions (debug mode):
   - translation read path matches write path,
   - azimuth sign check (`left=+`, `right=-`) at known synthetic points.

### `debug_multiframe.py`

1. Add elevation and physics env flags (see config section).
2. During initialization, call elevation-aware back-projection (`random` default).
3. Add optional sonar fixed-opacity path:
     - when enabled, set surfel opacity to 1 and freeze optimizer for opacity.
4. Add persistent pixel-logit registry (`nn.ParameterDict`) and dedicated optimizer (`optim_elev`) with explicit refresh/remap policy.
5. Implement Stage 1 bin-likelihood loss + annealing.
6. Implement belief-to-geometry coupling loss (`expected-point -> surfel`) with robust weighting.
7. Add persistent surfel-ID management (`surfel_ids`, `next_surfel_id`, `id_to_row`) and ID-keyed support tracking across densify/prune.
8. Implement staged normals regularization schedule.
9. Add optional Stage 2 densification hook.
10. Extend logs with elevation/coupling/support/normals diagnostics.
11. Checkpoint integration: save/restore `pixel_bank`, `pixel_logits`, `optim_elev`, `surfel_ids`, `next_surfel_id`, and ID-keyed support state.

### `utils/point_utils.py`

1. Extend `sonar_ranges_to_points(..., elevation=None)`:
   - `None` keeps legacy zero-elevation behavior.
   - `elevation=[H,W]` supports expected-elevation geometry path.
2. Keep default normals path unchanged early; only pass learned/expected elevation in late Stage 1+.

### `arguments/__init__.py` (optional)

Add defaults for new flags if standard CLI exposure is desired.

---

## Loss Formulation (Concrete)

### Stage 1 primary: Bin likelihood + annealing

```python
loss_lik, loss_ent = 0.0, 0.0
T_model = anneal(iter, T_start, T_end)
T_post = anneal(iter, T_start, T_end)
T_tgt = ELEV_LIK_TGT_TEMP
cached_loglik = {}  # reset every iteration; same-frame cache for coupling
for frame_a in sampled_frames:
    rows, cols, logits = pixel_bank[frame_a]
    pts_bins = back_project_bins(frame_a, rows, cols, elev_bins)   # [P,K,3]
    loglik_sum = torch.zeros_like(logits)
    valid_w = torch.zeros_like(logits)
    valid_any = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    for frame_b in overlap_table[frame_a][:B]:
        pix_b = sonar_project_points(pts_bins.reshape(-1, 3), frame_b, sonar_config, scale)
        i_raw = sample_gt(frame_b, pix_b.row, pix_b.col).reshape(P, K)
        v_b = pix_b.in_fov.reshape(P, K).float()
        valid_any = valid_any | (v_b > 0).any(dim=-1)

        # Frame-robust normalization (computed once per frame_b)
        # i_norm in [0,1], robust to gain/contrast changes
        i_norm = normalize_by_percentiles(i_raw, p_lo=P10_B, p_hi=P99_B)

        # Robust amplitude evidence (valid-only, clipped)
        l_amp = torch.log(i_norm.clamp_min(LOG_EPS)).clamp(min=LOG_FLOOR, max=0.0)
        loglik_sum += REL_B * v_b * l_amp
        valid_w += REL_B * v_b

    # Per-bin valid-count normalization to avoid sparse-support bias.
    loglik = loglik_sum / (valid_w + 1e-8)
    cached_loglik[frame_a] = loglik.detach()

    # Likelihood loss: evidence-target CE (target detached from model logits)
    t = softmax(loglik.detach() / T_tgt, dim=-1)   # evidence-derived target
    q = softmax(logits / T_model, dim=-1)          # logits-derived prediction

    m = valid_any.float()                          # all-invalid pixels get zero weight
    ce = -torch.sum(t * torch.log(q.clamp_min(1e-8)), dim=-1)
    loss_lik += (ce * m).sum() / (m.sum() + 1e-8)

    # Posterior for coupling/debug (not used as CE target)
    p_post = softmax((logits + loglik) / T_post, dim=-1)

    # Sharpness term (negative entropy; positive weight => sharper beliefs)
    ent = -torch.sum(p_post * torch.log(p_post.clamp_min(1e-8)), dim=-1)
    loss_ent += ((-ent) * m).sum() / (m.sum() + 1e-8)

# Defer aggregation: combine weighted terms exactly once after coupling is computed.
```

Notes:
- This likelihood is GT-anchored (not surfel-self-referential).
- CE target comes from detached GT evidence only; prediction comes from learnable logits only.
- Invalid/out-of-FOV projections are neutral evidence to avoid false penalties.
- Per-bin evidence is normalized by valid support (`loglik_sum / valid_w`) to avoid invalid-wins bias.
- Pixels with all-invalid projections are zero-weighted in the reduction.
- Per-frame percentile normalization reduces gain-induced bias across views.
- `cached_loglik` is rebuilt each iteration and stored as detached tensors for coupling use in the same iteration only.
- Arc-consistency can be added only as an optional low-weight stabilizer.

### Belief-to-geometry coupling (mandatory)

```python
loss_couple = 0.0
for frame_a in sampled_frames:  # must match the same sampled_frames used to build cached_loglik
    rows, cols, logits = pixel_bank[frame_a]
    pts_bins = back_project_bins(frame_a, rows, cols, elev_bins)      # [P,K,3]
    p_post = softmax((logits + cached_loglik[frame_a]) / T_post, dim=-1) # [P,K]
    pts_exp = (p_post[..., None] * pts_bins).sum(dim=1)                   # [P,3]

    # On-the-fly association using projection agreement + residual gating
    surf_idx, assoc_w, match_valid = associate_expected_points_to_surfels(
        pts_exp, gaussians, frame_a,
        max_pix_err=ELEV_COUPLE_MAX_PIX_ERR,
        max_depth_err=ELEV_COUPLE_MAX_DEPTH_ERR,
    )

    if match_valid.any():
        idx = surf_idx[match_valid]
        w = assoc_w[match_valid]
        pts_m = pts_exp[match_valid]

        d = torch.norm(gaussians.get_xyz[idx] - pts_m, dim=-1)
        loss_couple += (w * huber(d, delta=ELEV_COUPLE_HUBER_DELTA)).sum() / (w.sum() + 1e-8)
    else:
        # No valid associations in this frame; coupling contributes zero.
        continue

# Single aggregation point (avoid double-counting):
loss += w_lik * loss_lik
loss += w_ent * loss_ent
loss += w_couple(iter) * loss_couple
```

Coupling policy:
- Use soft association weights (`assoc_w`) and robust Huber penalty to reduce sensitivity to bad matches.
- Enable stronger coupling only after warmup and when elevation entropy is decreasing.
- Keep association ephemeral per iteration (no global hard map) to avoid lock-in.
- Coupling consumes only same-iteration detached `cached_loglik` built from the same frame/overlap sample set; no cross-iteration cache reuse.
- Association must return an explicit `match_valid` mask; unmatched points are excluded from gather/reduction.
- If a frame has zero valid matches, skip coupling for that frame and log the event (avoid sentinel-index side effects).

---

## Likelihood Measurement Model (Resolved)

Use robust normalized amplitude likelihood as the default measurement model.

Per-frame preprocessing (once per frame):
- Compute robust percentiles on valid GT sonar returns: `P10_B`, `P99_B`.
- Normalize sampled intensity: `I_norm = clamp((I - P10_B) / (P99_B - P10_B + eps), 0, 1)`.

Per-sample evidence:
- For valid projections: `l_amp = clamp(log(I_norm + eps_log), log_floor, 0)`.
- For invalid projections: contribution is `0` (neutral evidence).
- Apply frame reliability weight `REL_B` to downweight weak/noisy frames.
- Aggregate evidence per bin as weighted mean, not raw sum:
  `loglik_bin = sum(REL_B * valid * l_amp) / (sum(REL_B * valid) + eps)`.

Why this default:
- Preserves amplitude information for elevation disambiguation.
- Improves stability across frame-to-frame dynamic-range variation (raw sonar, speckle, local reflectivity changes).
- Avoids overly harsh penalties for out-of-FOV projections.
- Low overhead and easy to debug compared with heavier probabilistic models.

Fallback modes (optional, not default):
- Binary hit/miss likelihood.
- Hybrid amplitude + hit/miss.

---

## Range Attenuation and Dynamic Range (Resolved)

Use stabilized attenuation as the default physics model for raw sonar amplitudes:

- `I = lambert * gain / (max(r, r0)^p + eps)`
- Default values:
  - `p = 2.0` (physics-first),
  - `r0 = 0.35 m` (near-field floor),
  - `eps = 1e-6`,
  - `gain`: fixed configurable scalar; optionally auto-calibrated from early-iteration intensity stats.

Why this default:
- Raw sonar data requires distance attenuation in the forward model.
- Near-range floor prevents blow-up/saturation from very small `r`.
- Tunable exponent and gain support calibration without removing physical structure.

Calibration/ablation policy:
- Keep attenuation OFF only for diagnostic ablation.
- Run exponent ablation on `{1.5, 2.0, 2.5}` with same seed and compare mesh quality.
- If far geometry is systematically weak, adjust `gain` and/or lower `p` before changing other losses.

---

## Coordinate Convention Contract (Resolved)

Canonical definitions used by this plan:

- **Camera/view frame** (primary compute frame): `+X right`, `+Y down`, `+Z forward`.
- **Sonar image mapping**: columns represent azimuth (`left=+`, `right=-`), rows represent range (`top=near`, `bottom=far`).
- **Elevation-angle sign**: `+elevation -> +Y` (down), `-elevation -> -Y` (up), defined in camera/view frame.
- **Mount translation tuple** is expressed in camera frame: `[x, y, z] = [0.0, -0.10, -0.08]` m.
- **Transform storage contract** for code paths using `world_view_transform`: translation is stored/read from row 3 (`T[3, :3]`).

Reasoning:
- This repository mixes camera-frame and sonar-frame notation in different utilities.
- A single contract prevents silent sign/axis mistakes that otherwise look like "bad training".
- Row/column translation ambiguity has already caused branch divergence; the contract removes that ambiguity.

Required checks before enabling full training:
- Known-point projection test: right-of-boresight point maps to right image side (`negative` azimuth).
- Back-projection sign test: left image column yields `positive` azimuth and correct lateral sign.
- Elevation sign test at boresight/range anchor: `+elevation` yields `y>0`, `-elevation` yields `y<0` in camera/view frame.
- Transform consistency test: composed `camera->sonar->camera` returns identity within tolerance.

---

## Normals Path (Concrete)

### Early phase (Stage 1)

- Keep explicit normal regularization low (`w_normal` small).
- Do not feed learned elevation into normals yet.
- Rely on multi-view photometric + likelihood losses for stable geometry first.

### Late Stage 1+

- Compute expected elevation per selected pixel:

```python
e_exp = (probs * elev_bins[None, :]).sum(dim=-1)  # [P]
```

- Build expected-elevation 3D neighborhood points and derive finite-difference normals.
- Increase normal regularizer gradually and apply where elevation distributions are confident (low entropy mask).

---

## Correspondence Strategy

- **Default:** implicit multi-view consistency (no persistent global pixel-pixel map).
- Use overlap table and projected bins to get per-iteration correspondences on the fly.
- **Fallback:** add explicit surfel-anchored matching only if convergence is too slow.

---

## Multi-View Support Policy (Surfel Retention)

- Keep surfels based on **actual multi-view support**, not FOV presence alone.
- Track support by persistent `surfel_id` (not mutable row index) so densify/prune/reorder cannot mis-attach history.
- A frame contributes support for surfel `s` only if all are true:
  - projected point is valid (`in_fov`, in-front, in-bounds),
  - projected pixel has meaningful GT return (not masked/near-zero),
  - photometric/geometric residual is below threshold for that frame.
- Require viewpoint diversity in the support set (minimum baseline/pose-angle separation) so near-duplicate frames do not overcount support.
- Use warmup + soft-to-hard schedule:
  - warmup: no hard pruning, only accumulate EMA support,
  - mid: require `support_ratio >= 0.25` with floor `support_count >= 2`,
  - late: require `support_ratio >= 0.45` with floor `support_count >= 4` for stable geometry.
  - where `support_ratio = support_count / max(1, diverse_candidate_count)`.
- Prune only when support stays below threshold for multiple checks (hysteresis) to avoid oscillation.

---

## Configuration / Flags

- `ELEVATION_AWARE=1`
- `ELEV_INIT_MODE=random|zero`
- `ELEV_BINS=7`
- `ELEV_PIXELS_PER_FRAME=2000`
- `ELEV_LOGIT_LR`
- `ELEV_BANK_REFRESH_INTERVAL=0`  # 0 disables refresh
- `ELEV_BANK_REMAP_MODE=nearest`  # nearest|reset
- `ELEV_TEMP_START=2.0`, `ELEV_TEMP_END=0.1`
- `ELEV_LIK_TGT_TEMP=1.0`
- `ELEV_LIK_WEIGHT`, `ELEV_ENTROPY_WEIGHT`
- `ELEV_LIK_NORM_P_LO=10`, `ELEV_LIK_NORM_P_HI=99`
- `ELEV_LIK_LOG_EPS=1e-3`, `ELEV_LIK_LOG_FLOOR=-6.9`
- `ELEV_LIK_USE_FRAME_RELIABILITY=1`
- `ELEV_LIK_INVALID_MODE=neutral`  # neutral|penalize
- `ELEV_COUPLE_WEIGHT_START`, `ELEV_COUPLE_WEIGHT_END`, `ELEV_COUPLE_WARMUP`
- `ELEV_COUPLE_MAX_PIX_ERR`, `ELEV_COUPLE_MAX_DEPTH_ERR`, `ELEV_COUPLE_HUBER_DELTA`
- `ELEV_USE_ARC_STABILIZER=0`, `ELEV_ARC_STABILIZER_WEIGHT`
- `ELEV_SUPPORT_WARMUP_ITERS`
- `ELEV_SUPPORT_USE_PERSISTENT_IDS=1`
- `ELEV_SUPPORT_USE_RATIO=1`
- `ELEV_SUPPORT_MIN_RATIO_MID=0.25`, `ELEV_SUPPORT_MIN_RATIO_LATE=0.45`
- `ELEV_SUPPORT_MIN_COUNT_MID=2`, `ELEV_SUPPORT_MIN_COUNT_LATE=4`
- `ELEV_SUPPORT_VIEW_ANGLE_MIN_DEG`
- `ELEV_SUPPORT_RESIDUAL_THRESH`, `ELEV_SUPPORT_EMA_DECAY`, `ELEV_SUPPORT_PRUNE_PATIENCE`
- `ELEV_SURFEL_ID_ASSERTS=1`
- `ELEV_DENSIFY=0`, `ELEV_DENSIFY_INTERVAL`
- `SONAR_FIXED_OPACITY=1`
- `SONAR_USE_RANGE_ATTEN=1`
- `SONAR_RANGE_ATTEN_EXP=2.0`
- `SONAR_RANGE_ATTEN_GAIN=1.0`
- `SONAR_RANGE_ATTEN_R0=0.35`
- `SONAR_RANGE_ATTEN_EPS=1e-6`
- `SONAR_RANGE_ATTEN_AUTO_GAIN=0`
- `SONAR_CONVENTION_ASSERTS=1`

---

## Logging / Diagnostics

 - `loss_lik`, `loss_entropy`, `loss_couple`, temperature
 - per-frame likelihood normalization stats (`p10`, `p99`, reliability)
 - invalid-projection rate in likelihood accumulation
 - elevation-bin entropy stats and argmax histogram
 - expected-point to surfel residual stats (mean/p95)
- per-surfel support count stats (raw + EMA) and prune counts
- viewpoint-diverse support coverage
- surfel-ID integrity stats (active ID count, duplicate-ID count, invalid map entries)
- confidence mask coverage for normals updates
 - opacity mode and attenuation mode flags in run header
 - active frame/sign convention summary in run header
- near-range saturation rate and far-range response statistics
- periodic debug PLY of expected-elevation points

---

## Validation (Mesh-First)

- Compare baseline vs Stage 0/1 on same dataset and seed.
- Primary success signal: improved mesh geometry and reduced out-of-plane artifacts.
- Secondary signals: stable training, decreasing likelihood/coupling losses, narrowing elevation entropy, increasing multi-view support quality.
- Ablations:
  - raw `log(I)` likelihood vs robust normalized likelihood (default)
  - fixed vs learnable opacity
  - with vs without range attenuation
  - attenuation exponent ablation (`p` in 1.5/2.0/2.5)
  - with vs without belief-to-geometry coupling
  - FOV-only pruning vs multi-view support pruning
  - implicit-only vs implicit+explicit correspondence fallback
  - with vs without convention asserts (debug)

---

## Risks and Mitigations

- **Compute overhead:** keep small frame/pixel subsets; vectorize projection.
- **Unstable early normals:** delay strong normal regularization until low-entropy phase.
- **Scale/elevation coupling:** keep scale frozen during Stage 1.
- **Likelihood collapse:** clamp intensities, add entropy term, use temperature schedule.
- **Cross-frame gain drift:** use percentile normalization and frame reliability weighting.
- **Near/far imbalance from attenuation:** use near-range floor `r0`, tune `p`/gain via controlled ablation, monitor saturation diagnostics.
- **Bad associations in coupling:** use projection/depth gates, robust loss, and delayed weight ramp.
- **Over-pruning true geometry:** warmup first, enforce viewpoint diversity, use EMA+hysteresis before pruning.
- **Support-state misalignment after topology edits:** use persistent surfel IDs with post-mutation `id_to_row` rebuild + ID integrity asserts.
- **Extrinsic drift across branches/files:** enforce one canonical mount tuple (`x=0, y=-0.10, z=-0.08, pitch=+5deg`) and add explicit startup logging of active extrinsic.
- **Frame/sign mismatch across modules:** enforce convention contract + startup assertions + known-point tests before long runs.

---

## Incremental Delivery Order

1. Add projection/back-projection helpers + range attenuation.
2. Add elevation-aware initialization and fixed-opacity toggle.
3. Integrate Stage 1 likelihood bins + annealing.
4. Integrate belief-to-geometry coupling loss (expected-point to surfel).
5. Add multi-view support tracking and retention/pruning schedule.
6. Add staged normals update path.
7. Enable optional Stage 2 densification.
8. Add convention assertion checks and run-header convention logging.

---

## Authorship Marker

- **gpt-5.2-codex**: Original draft
- **gpt-5.3-codex**: Alignment update to latest base-plan decisions
