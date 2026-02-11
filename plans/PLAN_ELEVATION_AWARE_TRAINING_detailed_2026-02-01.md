# Plan: Elevation-Aware Training with Back Projection (Detailed)

**Date/Time:** 2026-02-01 (updated 2026-02-10)
**Git Commit:** (not yet implemented)
**Status:** Detailed implementation plan aligned to latest base-plan decisions

---

## Session Note Procedure (2026-02-10)

To keep future sessions consistent and avoid plan drift:

1. Track issues point-by-point in `scratchpad.md` with file references.
2. When resolving an issue, update this detailed plan first with executable contracts (formulas, interfaces, defaults).
3. Add only a short corresponding decision note in the base plan and point back to this file for full implementation details.
4. Preserve this split for all new issues: base plan = decision summary, detailed plan = implementation contract.

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

## Terminology Contract (2026-02-10)

To avoid confusion with the pre-existing `debug_multiframe.py` curriculum labels:

- This document uses **elevation stages**: Stage 0, Stage 1, Stage 2.
- Existing training curriculum labels should be treated as **curriculum phases** (Phase A/B/C), not reused as Stage 1/2/3 in this plan.
- Default mapping for implementation:
  - Elevation Stage 0 runs at the start of curriculum warmup.
  - Elevation Stage 1 is the main optimization phase.
  - Elevation Stage 2 is optional and only enabled after Stage 1 stabilizes.
- For implementation decisions, this detailed plan is authoritative; exploratory sections in the base plan remain historical context unless explicitly carried into "Resolved" sections here.

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

- Per selected pixels, optimize logits over K elevation bins (`K=7` default, within 5-9 dev range; optional later upgrade to 21 after stability).
- Build multi-view likelihood by projecting each bin candidate into overlapping frames.
- Anneal temperature to sharpen distributions over time.
- Convert pixel beliefs to expected 3D points and apply robust attraction to associated surfels.
- Retain only surfels with strong multi-view support.

### Stage 2: Optional Arc-Guided Densification

- For persistent high-error bright pixels, search elevation bins and spawn surfels at peak-score bins.
- Off by default.

### Stage Boundary Contract (v1 Fixed Gates)

- Stage 0 is `init_only`: run once before optimization iteration 0 (random/zero elevation init); no metric gate.
- Stage 1 is the default main optimization phase starting at iteration 0.
- Stage 2 hooks are enabled only when all are true:
  - `ELEV_DENSIFY=1`,
  - `iter >= ELEV_STAGE2_START_ITER`,
  - `iter % ELEV_DENSIFY_INTERVAL == 0`.
- Normals schedule is iteration-gated in v1:
  - early normals regime: `iter < ELEV_NORMAL_RAMP_START_ITER` with `w_normal=ELEV_NORMAL_WEIGHT_EARLY`,
  - linear ramp: `iter in [ELEV_NORMAL_RAMP_START_ITER, ELEV_NORMAL_RAMP_END_ITER]`,
  - late normals regime: `iter > ELEV_NORMAL_RAMP_END_ITER` with `w_normal=ELEV_NORMAL_WEIGHT_LATE`.
- Expected-elevation normals path activates at `iter >= ELEV_NORMAL_ELEV_START_ITER` (default aligns with ramp start).
- v1 transition policy is fixed-iteration only; hybrid metric+iteration gates are deferred to a later revision.

---

## Core Contracts Required Before Coding (Resolved)

### 1) `overlap_table` contract

- Type: `Dict[int, List[int]]` keyed by frame index.
- Built once at startup from the active training frame list (or loaded from checkpoint when resuming with identical frame IDs).
- `overlap_score` v1 is pose-only (no image-content visibility term).
- For each candidate pair `(frame_a, frame_b)` with `frame_b != frame_a`:
  - `d = ||t_a - t_b||` (baseline in meters),
  - `dyaw = wrapped_abs_deg(yaw_a - yaw_b)` in `[0, 180]`.
- Hard gates (reject before scoring):
  - `d >= ELEV_OVERLAP_MIN_BASELINE`,
  - `dyaw <= ELEV_OVERLAP_MAX_YAW_DEG`.
- v1 default for heading gate is `ELEV_OVERLAP_MAX_YAW_DEG=40`; this is expected to be an early tuning knob and may need to increase for wider-orbit coverage.
- Normalized score components:
  - `s_yaw = clamp(1 - dyaw / ELEV_OVERLAP_MAX_YAW_DEG, 0, 1)`,
  - `s_base = clamp((d - ELEV_OVERLAP_MIN_BASELINE) / max(ELEV_OVERLAP_MIN_BASELINE, 1e-6), 0, 1)`.
- Score formula:
  - `overlap_score = ELEV_OVERLAP_SCORE_W_YAW * s_yaw + ELEV_OVERLAP_SCORE_W_BASE * s_base`.
- Keep candidate only if `overlap_score >= ELEV_OVERLAP_MIN_SCORE`.
- Rank candidates by `overlap_score`, keep top `ELEV_OVERLAP_TOPK_BUILD` per `frame_a`.
- Sort deterministically (descending score, then ascending frame index).
- Training loop uses `B_use = min(ELEV_OVERLAP_TOPK_USE, len(overlap_table[frame_a]))`.
- Persist table + build parameters in checkpoints for reproducibility.

### 2) `sampled_frames` policy

- Default policy: deterministic shuffled round-robin over active frames.
- At each iteration, sample `A = min(ELEV_FRAMES_PER_ITER, N_active)` frames from the current cursor window.
- When cursor reaches end, reshuffle with deterministic seed and start a new epoch.
- This is the default because it balances coverage, stability, and reproducibility better than pure random per-iter sampling.

### 3) `back_project_bins` interface contract

```python
def back_project_bins(
    frame_idx: int,
    rows: torch.Tensor,        # [P], long
    cols: torch.Tensor,        # [P], long
    elev_bins: torch.Tensor,   # [K], radians in camera/view convention
    *,
    cameras,
    sonar_config,
    scale_factor,
) -> torch.Tensor:            # [P, K, 3], world-frame XYZ
```

- Applies canonical convention: camera/view frame (`+X right, +Y down, +Z forward`), then transform to world.
- Elevation sign follows plan contract: `+elevation -> +Y`, `-elevation -> -Y` in camera/view frame.
- Returned points are world-frame points compatible with `sonar_project_points` input.

### 4) `sonar_project_points` return contract

Use a typed return object (namedtuple/dataclass), not a positional tuple:

```python
@dataclass
class SonarProjection:
    row: torch.Tensor         # [N], float
    col: torch.Tensor         # [N], float
    range_vals: torch.Tensor  # [N], float
    azimuth: torch.Tensor     # [N], float
    in_fov: torch.Tensor      # [N], bool
    in_front: torch.Tensor    # [N], bool
    in_bounds: torch.Tensor   # [N], bool
    valid: torch.Tensor       # [N], bool = in_fov & in_front & in_bounds
```

- Pseudocode in this plan assumes attribute access (`pix_b.row`, `pix_b.col`, `pix_b.valid`).
- Stage-1 likelihood/support gating uses `pix_b.valid` (not `pix_b.in_fov`) to enforce full projection validity.

### 5) Frame reliability weight `REL_B`

- `REL_B` is precomputed per frame from a startup-frozen `frame_stats` cache (build once per run for the active frame list).
- No periodic `frame_stats` refresh in v1 because training frames and masks are treated as static.
- Rebuild `frame_stats` only if the active frame list/split changes.
- Default formula:
  - `r_valid = norm(valid_return_ratio_b)` using configurable min/max clamps,
  - `r_dyn = norm(p99_b - p10_b)` using configurable min/max dynamic-range clamps,
  - `REL_B = clamp(ELEV_REL_FLOOR + (1 - ELEV_REL_FLOOR) * (0.7 * r_valid + 0.3 * r_dyn), ELEV_REL_FLOOR, 1.0)`.
- If `ELEV_LIK_USE_FRAME_RELIABILITY=0`, force `REL_B = 1.0`.

### 6) `associate_expected_points_to_surfels` contract

- Inputs: expected points `pts_exp_v` in world frame, current surfels, frame index, and gate thresholds.
- Steps per expected point:
  1. Project surfels and expected point into `frame_a` using the same projection helper.
  2. Candidate gate: `pix_err <= ELEV_COUPLE_MAX_PIX_ERR`, `depth_err <= ELEV_COUPLE_MAX_DEPTH_ERR`, both valid.
  3. If no candidate passes, mark unmatched.
  4. Else choose nearest candidate by normalized score
     `s = (pix_err / sigma_pix)^2 + (depth_err / sigma_depth)^2`.
  5. Set scalar match weight
     `assoc_w = exp(-0.5 * s)` (clamped to `[ELEV_COUPLE_MIN_W, 1]`).
- Coupling normalization scales for v1 are fixed:
  - `sigma_pix = ELEV_COUPLE_SIGMA_PIX` (default `2.0` pixels),
  - `sigma_depth = ELEV_COUPLE_SIGMA_DEPTH` (default `0.05` meters).
- `ELEV_COUPLE_SIGMA_MODE=fixed` is the v1 contract for reproducibility.
- Planned follow-up: add `adaptive` sigma mode in a later revision (per-frame/per-epoch residual-based updates) after v1 baseline is stable.
- Return tensors with one slot per expected point:
  - `surf_idx` (long),
  - `assoc_w` (float),
  - `match_valid` (bool).

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

def get_pixel_bank_tensors(frame_idx, pixel_bank, pixel_logits):
    entry = pixel_bank[frame_idx]
    rows = entry["rows"]
    cols = entry["cols"]
    logits = pixel_logits[entry["logits_key"]]
    return rows, cols, logits
```

Use subsampled bright pixels (`ELEV_PIXELS_PER_FRAME`) to cap memory and runtime.

Lifecycle contract for pixel logits:
- Initialize `pixel_logits` once at Stage 1 start for the active frame set.
- If refresh is enabled (`ELEV_BANK_REFRESH_INTERVAL > 0`), trigger on `iter % ELEV_BANK_REFRESH_INTERVAL == 0`.
- Refresh remap modes:
  - `nearest`: for each new pixel `(row, col)`, copy logits from nearest old pixel in the same frame by L1 image distance; if nearest distance exceeds `ELEV_BANK_REMAP_MAX_DIST`, reset that pixel's logits.
  - `reset`: initialize all refreshed pixel logits to zeros.
- Read logits through `logits_key` lookup only; do not store trainable logits directly inside `pixel_bank` entries.
- After any add/remove/reshape of pixel-logit parameters, rebuild `optim_elev` param groups to avoid stale optimizer state.
- Save/load both `pixel_bank` metadata and `pixel_logits` + `optim_elev.state_dict()` in checkpoints.
- Resume mismatch policy (`frame set` or per-frame `N_pix` changed):
  - `strict` (default): fail fast and require explicit reset.
  - `reset_frame`: keep compatible frames, reset mismatched frames.
  - `reset_all`: rebuild full pixel bank/logits and reset `optim_elev` state.
- Default first pass: no refresh (`ELEV_BANK_REFRESH_INTERVAL=0`) for maximum stability/debuggability.

### Frame stats cache (`frame_stats`)

```python
@dataclass
class FrameStats:
    valid_return_ratio: float
    norm_lo_value: float
    norm_hi_value: float
    dyn_range: float
    reliability: float
    is_valid: bool

frame_stats: Dict[int, FrameStats]
```

Lifecycle contract:
- Build once at Stage 1 startup for the active frame list, using fixed GT frames, fixed top-row mask, and configured intensity/percentile settings.
- Do not refresh periodically in v1.
- Rebuild only when active frame IDs/split change (for example, resume with a different frame subset).
- If a frame has insufficient valid returns, use safe fallback stats (`reliability=ELEV_REL_FLOOR`, `is_valid=False`) and keep training deterministic.

### Coupling/support buffers

```python
# Persistent surfel identity (stable across densify/prune/reorder).
surfel_ids = torch.arange(N_surfels, device="cuda", dtype=torch.long)
next_surfel_id = int(N_surfels)

# Support state is keyed by persistent surfel_id, not row index.
surfel_support = {
    "ema_by_id": torch.zeros([next_surfel_id], device="cuda"),
    "last_raw_by_id": torch.zeros([next_surfel_id], device="cuda"),
    "birth_iter_by_id": torch.zeros([next_surfel_id], device="cuda", dtype=torch.long),
}

# Row lookup (rebuild after any topology change).
id_to_row = torch.full([next_surfel_id], -1, device="cuda", dtype=torch.long)
id_to_row[surfel_ids] = torch.arange(N_surfels, device="cuda", dtype=torch.long)
```

Point-to-surfel association is computed on-the-fly per iteration (no persistent global correspondence map).

Lifecycle contract for persistent surfel IDs:
- Densify: allocate fresh IDs `[next_surfel_id, ..., next_surfel_id + n_new - 1]`, append to `surfel_ids`, grow `ema_by_id/last_raw_by_id` with zeros, append `birth_iter_by_id` for new IDs with the current `iter`, increment `next_surfel_id`, then rebuild `id_to_row`.
- Prune/reorder: update `surfel_ids = surfel_ids[keep_idx]` (or reordered view), then rebuild `id_to_row`.
- Update support by ID each iteration: `sid = surfel_ids[row]`; write to `ema_by_id[sid]` and `last_raw_by_id[sid]`.
- Per-surfel grace gate (v1): if `ELEV_SUPPORT_USE_NEW_SURFEL_GRACE=1` and `iter - birth_iter_by_id[sid] < ELEV_SUPPORT_NEW_SURFEL_GRACE_ITERS`, skip hard prune checks for that `sid` (support still accumulates).
- Checkpoint must persist `surfel_ids`, `next_surfel_id`, and support tensors keyed by ID.
- First pass keeps retired IDs in support tensors (no compaction) for simpler, safer bookkeeping.
- ID compaction policy in v1: disabled by design; accept monotonic ID-space growth during current tuning runs.
- Planned follow-up: add threshold-triggered compaction when ID space becomes too sparse (for example, trigger when `next_surfel_id > ratio * active_count`).

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
    # returns SonarProjection(row, col, range_vals, azimuth, in_fov, in_front, in_bounds, valid)
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
def masked_softmax(logits, mask, dim=-1, eps=1e-12):
    x = logits.masked_fill(~mask, float("-inf"))
    any_valid = mask.any(dim=dim, keepdim=True)
    x = torch.where(any_valid, x, torch.zeros_like(x))  # all-masked rows stay finite
    probs = torch.softmax(x, dim=dim)
    probs = probs * mask.float()                        # force zero on masked bins
    z = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    probs = torch.where(any_valid, probs / z, torch.zeros_like(probs))
    return probs, any_valid.squeeze(dim)

loss_lik_acc, loss_ent_acc = 0.0, 0.0
n_lik_frames, n_ent_frames = 0, 0
T_shared = anneal(iter, T_start, T_end)
T_model = T_shared
T_post = T_shared if ELEV_TEMP_POST_MODE == "shared" else anneal(iter, T_post_start, T_post_end)
T_tgt = ELEV_LIK_TGT_TEMP
B_use = ELEV_OVERLAP_TOPK_USE
cached_loglik = {}  # reset every iteration; same-frame cache for coupling
cached_support_mask = {}  # reset every iteration; same-frame support mask for coupling
for frame_a in sampled_frames:
    rows, cols, logits = get_pixel_bank_tensors(frame_a, pixel_bank, pixel_logits)
    P, K = logits.shape
    pts_bins = back_project_bins(frame_a, rows, cols, elev_bins)   # [P,K,3]
    loglik_sum = torch.zeros_like(logits)
    valid_w = torch.zeros_like(logits)
    neighbors = overlap_table[frame_a][:B_use]
    for frame_b in neighbors:
        pix_b = sonar_project_points(pts_bins.reshape(-1, 3), frame_b, sonar_config, scale)
        i_raw = sample_gt(frame_b, pix_b.row, pix_b.col).reshape(P, K)
        v_b = pix_b.valid.reshape(P, K).float()

        stats_b = frame_stats[frame_b]  # startup-frozen per-run cache
        v_lo_b = stats_b["norm_lo_value"]
        v_hi_b = stats_b["norm_hi_value"]
        REL_B = stats_b["reliability"] if ELEV_LIK_USE_FRAME_RELIABILITY else 1.0

        # Frame-robust normalization (computed once per frame_b)
        # i_norm in [0,1], robust to gain/contrast changes
        i_norm = normalize_by_percentiles(i_raw, lo=v_lo_b, hi=v_hi_b)

        # Robust amplitude evidence (valid-only, clipped)
        l_amp = torch.log(i_norm.clamp_min(LOG_EPS)).clamp(min=LOG_FLOOR, max=0.0)
        loglik_sum += REL_B * v_b * l_amp
        valid_w += REL_B * v_b

    # Per-bin valid-count normalization to avoid sparse-support bias.
    # Unsupported bins are set to LOG_FLOOR and then masked out of softmax.
    loglik = torch.where(
        valid_w > 0,
        loglik_sum / (valid_w + 1e-8),
        torch.full_like(loglik_sum, LOG_FLOOR),
    )
    cached_loglik[frame_a] = loglik.detach()
    support_mask = (valid_w > ELEV_LIK_MIN_SUPPORT)
    cached_support_mask[frame_a] = support_mask.detach()

    # Likelihood loss: evidence-target CE (target detached from model logits)
    t, _ = masked_softmax(loglik.detach() / T_tgt, support_mask, dim=-1)   # evidence-derived target
    q, _ = masked_softmax(logits / T_model, support_mask, dim=-1)          # logits-derived prediction

    m = support_mask.any(dim=-1).float()           # all-unsupported pixels get zero weight
    ce = -torch.sum(t * torch.log(q.clamp_min(1e-8)), dim=-1)
    if m.sum() > 0:
        loss_lik_acc += (ce * m).sum() / (m.sum() + 1e-8)
        n_lik_frames += 1

    # Posterior for coupling/debug (not used as CE target)
    p_post, _ = masked_softmax((logits + loglik) / T_post, support_mask, dim=-1)

    # Sharpness term (negative entropy; positive weight => sharper beliefs)
    ent = -torch.sum(p_post * torch.log(p_post.clamp_min(1e-8)), dim=-1)
    if m.sum() > 0:
        loss_ent_acc += ((-ent) * m).sum() / (m.sum() + 1e-8)
        n_ent_frames += 1

loss_lik = loss_lik_acc / max(1, n_lik_frames)
loss_ent = loss_ent_acc / max(1, n_ent_frames)

# Defer aggregation: combine weighted terms exactly once after coupling is computed.
```

Notes:
- This likelihood is GT-anchored (not surfel-self-referential).
- CE target comes from detached GT evidence only; prediction comes from learnable logits only.
- Invalid/out-of-FOV projections are neutral evidence to avoid false penalties.
- Per-bin evidence is normalized by valid support; unsupported bins are forced to `LOG_FLOOR` and masked out.
- Bins with insufficient support are excluded from softmax via `masked_softmax`.
- Pixels with no supported bins are zero-weighted in the reduction.
- Stage-1 losses are normalized by contributing-frame counts (`n_lik_frames`, `n_ent_frames`) to keep effective weights stable when sampled/valid frame counts vary.
- Per-frame percentile normalization reduces gain-induced bias across views.
- `B_use` is the runtime overlap fanout (`min(ELEV_OVERLAP_TOPK_USE, available_neighbors)`).
- `T_model` and `T_post` are intentionally shared by default (`ELEV_TEMP_POST_MODE=shared`); decoupling is opt-in.
- `cached_loglik` and `cached_support_mask` are rebuilt each iteration and stored as detached tensors for same-iteration coupling only.
- Arc-consistency can be added only as an optional low-weight stabilizer.

### Belief-to-geometry coupling (mandatory)

```python
loss_couple_acc = 0.0
n_couple_frames = 0
for frame_a in sampled_frames:  # must match the same sampled_frames used to build cached_loglik
    rows, cols, logits = get_pixel_bank_tensors(frame_a, pixel_bank, pixel_logits)
    pts_bins = back_project_bins(frame_a, rows, cols, elev_bins)      # [P,K,3]
    p_post, pix_valid = masked_softmax(
        (logits + cached_loglik[frame_a]) / T_post,
        cached_support_mask[frame_a],
        dim=-1,
    )                                                                      # [P,K], [P]
    pts_exp = (p_post[..., None] * pts_bins).sum(dim=1)                   # [P,3]

    if not pix_valid.any():
        # No supported bins for this frame in this iteration.
        continue

    pts_exp_v = pts_exp[pix_valid]

    # On-the-fly association using projection agreement + residual gating
    surf_idx, assoc_w, match_valid = associate_expected_points_to_surfels(
        pts_exp_v, gaussians, frame_a,
        max_pix_err=ELEV_COUPLE_MAX_PIX_ERR,
        max_depth_err=ELEV_COUPLE_MAX_DEPTH_ERR,
    )

    if match_valid.any():
        idx = surf_idx[match_valid]
        w = assoc_w[match_valid]
        pts_m = pts_exp_v[match_valid]

        d = torch.norm(gaussians.get_xyz[idx] - pts_m, dim=-1)
        loss_couple_acc += (w * huber(d, delta=ELEV_COUPLE_HUBER_DELTA)).sum() / (w.sum() + 1e-8)
        n_couple_frames += 1
    else:
        # No valid associations in this frame; coupling contributes zero.
        continue

loss_couple = loss_couple_acc / max(1, n_couple_frames)

# Single aggregation point (avoid double-counting):
loss += w_lik * loss_lik
loss += w_ent * loss_ent
loss += w_couple(iter) * loss_couple
```

Coupling policy:
- Use soft association weights (`assoc_w`) and robust Huber penalty to reduce sensitivity to bad matches.
- Enable stronger coupling only after warmup and when elevation entropy is decreasing.
- Keep association ephemeral per iteration (no global hard map) to avoid lock-in.
- Coupling consumes only same-iteration detached `cached_loglik` and `cached_support_mask` from the same frame/overlap sample set; no cross-iteration cache reuse.
- Association must return an explicit `match_valid` mask; unmatched points are excluded from gather/reduction.
- If a frame has zero valid matches, skip coupling for that frame and log the event (avoid sentinel-index side effects).
- If a frame has zero supported bins (`pix_valid` all false), skip coupling for that frame.
- Coupling is normalized by contributing-frame count (`n_couple_frames`) so `w_couple` is stable under changing valid-frame coverage.

---

## Likelihood Measurement Model (Resolved)

Use robust normalized amplitude likelihood as the default measurement model.

Per-frame preprocessing (once per frame):
- Use percentile levels `q_lo=ELEV_LIK_NORM_P_LO` and `q_hi=ELEV_LIK_NORM_P_HI` (defaults `10`, `99`).
- Compute realized per-frame percentile values `v_lo_b`, `v_hi_b` on valid GT returns.
- Normalize sampled intensity: `I_norm = clamp((I - v_lo_b) / (v_hi_b - v_lo_b + eps), 0, 1)`.

Per-sample evidence:
- For valid projections: `l_amp = clamp(log(I_norm + eps_log), log_floor, 0)`.
- For invalid projections: contribution is `0` (neutral evidence).
- Invalid-mode contract for v1: `ELEV_LIK_INVALID_MODE` is fixed to `neutral`; `penalize` is deferred and not implemented in this plan.
- Apply frame reliability weight `REL_B` to downweight weak/noisy frames.
- Aggregate evidence per bin as weighted mean, not raw sum:
  `loglik_bin = sum(REL_B * valid * l_amp) / (sum(REL_B * valid) + eps)`.

Frame reliability definition (default):
- `r_valid = norm(valid_return_ratio_b; ELEV_REL_VALID_MIN, ELEV_REL_VALID_MAX)`.
- `r_dyn = norm((v_hi_b - v_lo_b); ELEV_REL_DYN_MIN, ELEV_REL_DYN_MAX)`.
- `REL_B = clamp(ELEV_REL_FLOOR + (1 - ELEV_REL_FLOOR) * (0.7 * r_valid + 0.3 * r_dyn), ELEV_REL_FLOOR, 1.0)`.
- If `ELEV_LIK_USE_FRAME_RELIABILITY=0`, use `REL_B=1.0`.

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

Parameter precedence contract (deterministic):
- If `SONAR_USE_RANGE_ATTEN=0`, attenuation is OFF and attenuation parameters are ignored.
- Else if `SONAR_RANGE_ATTEN_AUTO_GAIN=1`, auto-gain mode is active and `SONAR_RANGE_ATTEN_GAIN` is used only as an initialization seed.
- Else (`SONAR_RANGE_ATTEN_AUTO_GAIN=0`), manual-gain mode is active and effective gain is `SONAR_RANGE_ATTEN_GAIN`.
- Defaults therefore resolve to: attenuation ON, auto-gain OFF, manual gain `1.0`.

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

- Defined as `iter < ELEV_NORMAL_RAMP_START_ITER` in the v1 fixed-gate schedule.
- Keep explicit normal regularization low (`w_normal` small).
- Do not feed learned elevation into normals yet.
- Rely on multi-view photometric + likelihood losses for stable geometry first.

### Late Stage 1+

- Expected-elevation normals path is active for `iter >= ELEV_NORMAL_ELEV_START_ITER`.
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
- Per-surfel new-birth grace (v1): even after global warmup ends, a newly densified surfel is exempt from hard pruning until its age reaches `ELEV_SUPPORT_NEW_SURFEL_GRACE_ITERS`.
- To avoid unattainable gates on low-visibility surfels, use effective floors:
  - `mid_floor_eff = min(ELEV_SUPPORT_MIN_COUNT_MID, diverse_candidate_count)`
  - `late_floor_eff = min(ELEV_SUPPORT_MIN_COUNT_LATE, diverse_candidate_count)`
  - apply count checks against these effective floors.
- Rationale for strict late floor: this dataset is captured by orbiting while facing a central object of interest, so valid target surfels are expected to receive strong multi-view overlap. The late `support_count >= 4` gate is intentionally precision-biased to keep consistently observed target geometry and remove side/peripheral surfels with weak overlap.
- Prune only when support stays below threshold for multiple checks (hysteresis) to avoid oscillation.

---

## Configuration / Flags

- `ELEVATION_AWARE=1`
- `ELEV_INIT_MODE=random|zero`
- `ELEV_STAGE0_MODE=init_only`
- `ELEV_BINS=7`  # dev default within 5-9; raise to 21 only after Stage 1 is stable
- `ELEV_FRAMES_PER_ITER=3`
- `ELEV_FRAME_SAMPLER=round_robin`  # round_robin|random
- `ELEV_OVERLAP_TOPK_BUILD=24`, `ELEV_OVERLAP_TOPK_USE=6`
- `ELEV_OVERLAP_MIN_BASELINE=0.06`, `ELEV_OVERLAP_MAX_YAW_DEG=40`, `ELEV_OVERLAP_MIN_SCORE=0.30`  # max yaw likely needs upward tuning
- `ELEV_OVERLAP_SCORE_MODE=pose_only`
- `ELEV_OVERLAP_SCORE_W_YAW=0.6`, `ELEV_OVERLAP_SCORE_W_BASE=0.4`
- `ELEV_PIXELS_PER_FRAME=2000`
- `ELEV_LOGIT_LR=2e-3`
- `ELEV_BANK_REFRESH_INTERVAL=0`  # 0 disables refresh
- `ELEV_BANK_REMAP_MODE=nearest`  # nearest|reset
- `ELEV_BANK_REMAP_MAX_DIST=6`
- `ELEV_RESUME_PIXELLOGIT_MISMATCH=strict`  # strict|reset_frame|reset_all
- `ELEV_TEMP_START=2.0`, `ELEV_TEMP_END=0.1`
- `ELEV_TEMP_POST_MODE=shared`  # shared|decoupled
- `ELEV_TEMP_POST_START=2.0`, `ELEV_TEMP_POST_END=0.1`  # used only when decoupled
- `ELEV_LIK_TGT_TEMP=1.0`
- `ELEV_LIK_WEIGHT=1.0`, `ELEV_ENTROPY_WEIGHT=0.01`
- `ELEV_LIK_NORM_P_LO=10`, `ELEV_LIK_NORM_P_HI=99`
- `ELEV_LIK_LOG_EPS=1e-3`, `ELEV_LIK_LOG_FLOOR=-6.9`
- `ELEV_LIK_MIN_SUPPORT=1e-6`
- `ELEV_LIK_USE_FRAME_RELIABILITY=1`
- `ELEV_REL_FLOOR=0.3`
- `ELEV_REL_VALID_MIN=0.03`, `ELEV_REL_VALID_MAX=0.30`
- `ELEV_REL_DYN_MIN=0.08`, `ELEV_REL_DYN_MAX=0.50`
- `ELEV_LIK_INVALID_MODE=neutral`  # v1 fixed mode; penalize deferred
- `ELEV_COUPLE_WEIGHT_START=0.10`, `ELEV_COUPLE_WEIGHT_END=0.50`, `ELEV_COUPLE_WARMUP=2000`
- `ELEV_COUPLE_MAX_PIX_ERR=3.0`, `ELEV_COUPLE_MAX_DEPTH_ERR=0.08`, `ELEV_COUPLE_HUBER_DELTA=0.03`
- `ELEV_COUPLE_SIGMA_MODE=fixed`  # fixed|adaptive (adaptive deferred)
- `ELEV_COUPLE_SIGMA_PIX=2.0`
- `ELEV_COUPLE_SIGMA_DEPTH=0.05`
- `ELEV_COUPLE_MIN_W=0.10`
- `ELEV_USE_ARC_STABILIZER=0`, `ELEV_ARC_STABILIZER_WEIGHT`
- `ELEV_STAGE2_START_ITER=12000`  # used only when ELEV_DENSIFY=1
- `ELEV_NORMAL_RAMP_START_ITER=4000`, `ELEV_NORMAL_RAMP_END_ITER=8000`
- `ELEV_NORMAL_WEIGHT_EARLY=0.01`, `ELEV_NORMAL_WEIGHT_LATE=0.10`
- `ELEV_NORMAL_ELEV_START_ITER=4000`
- `ELEV_SUPPORT_WARMUP_ITERS=4000`
- `ELEV_SUPPORT_USE_PERSISTENT_IDS=1`
- `ELEV_SUPPORT_USE_RATIO=1`
- `ELEV_SUPPORT_USE_NEW_SURFEL_GRACE=1`
- `ELEV_SUPPORT_NEW_SURFEL_GRACE_ITERS=1500`
- `ELEV_SUPPORT_MIN_RATIO_MID=0.25`, `ELEV_SUPPORT_MIN_RATIO_LATE=0.45`
- `ELEV_SUPPORT_MIN_COUNT_MID=2`, `ELEV_SUPPORT_MIN_COUNT_LATE=4`
- `ELEV_SUPPORT_VIEW_ANGLE_MIN_DEG=8`
- `ELEV_SUPPORT_RESIDUAL_THRESH=0.20`, `ELEV_SUPPORT_EMA_DECAY=0.90`, `ELEV_SUPPORT_PRUNE_PATIENCE=4`
- `ELEV_SURFEL_ID_ASSERTS=1`
- `ELEV_DENSIFY=0`, `ELEV_DENSIFY_INTERVAL=1500`
- `SONAR_FIXED_OPACITY=1`
- `SONAR_USE_RANGE_ATTEN=1`
- `SONAR_RANGE_ATTEN_EXP=2.0`
- `SONAR_RANGE_ATTEN_GAIN=1.0`  # effective manual gain; initialization seed in auto-gain mode
- `SONAR_RANGE_ATTEN_R0=0.35`
- `SONAR_RANGE_ATTEN_EPS=1e-6`
- `SONAR_RANGE_ATTEN_AUTO_GAIN=0`  # 0=manual gain mode, 1=auto-gain mode
- `SONAR_CONVENTION_ASSERTS=1`

---

## Logging / Diagnostics

 - `loss_lik`, `loss_entropy`, `loss_couple`, temperature
 - contributing-frame counts (`n_lik_frames`, `n_ent_frames`, `n_couple_frames`) and normalized-vs-raw term values
 - per-frame likelihood normalization stats (`p10`, `p99`, reliability)
 - invalid-projection rate in likelihood accumulation
 - elevation-bin entropy stats and argmax histogram
 - expected-point to surfel residual stats (mean/p95)
- per-surfel support count stats (raw + EMA) and prune counts
- viewpoint-diverse support coverage
- surfel-ID integrity stats (active ID count, duplicate-ID count, invalid map entries)
- confidence mask coverage for normals updates
- opacity mode and attenuation mode flags in run header
- attenuation gain mode (`manual|auto`) and effective gain value in run header
- active frame/sign convention summary in run header
- near-range saturation rate and far-range response statistics
- periodic debug PLY of expected-elevation points

---

## Experiment Run Ledger Workflow (v1)

Use an in-repo run ledger to track tuning cycles beyond git history.

Directory contract:

```text
testruns/
  INDEX.md
  templates/
    run.md
    config.env
    metrics.json
  testrun_<run_id>_<short_commit>_<YYYYMMDD_HHMM>/
    run.md
    config.env
    metrics.json
    artifacts/
```

Naming contract:
- `run_id` is zero-padded and monotonic (`0001`, `0002`, ...).
- `short_commit` is the short git hash for the code being tested.
- Timestamp uses local time in `YYYYMMDD_HHMM` format.

Required file contents per run:
- `run.md`:
  - objective/hypothesis,
  - exact code/plan context (commit hash, branch, relevant plan issue IDs),
  - intended parameter changes,
  - execution command(s),
  - stop criteria,
  - outcome verdict (`improved|neutral|regressed|invalid`),
  - next action.
- `config.env`:
  - fully resolved environment/config values used for the run (not only deltas).
- `metrics.json`:
  - final scalar metrics and key checkpoints (loss terms, mesh metrics, support stats, notable diagnostics).

Run lifecycle contract:
1. Before run: create the run folder from templates, fill objective + planned config.
2. During run: append checkpoint notes in `run.md` and update `metrics.json` with milestone snapshots.
3. After run: finalize verdict and next action; update `testruns/INDEX.md` with one-row summary.
4. Next tuning step must reference at least the latest completed run and one prior comparable run from `testruns/`.

v1 policy:
- This ledger is required for tuning campaigns and is the primary cross-session memory for LLM-driven iteration.
- Git remains the source of code truth; `testruns/` is the source of experiment/tuning truth.

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
- **Unbounded ID-keyed tensor growth:** accepted in v1 (no compaction) for implementation simplicity; add threshold-triggered compaction in a future revision if long-run memory/overhead grows.
- **Extrinsic drift across branches/files:** enforce one canonical mount tuple (`x=0, y=-0.10, z=-0.08, pitch=+5deg`) and add explicit startup logging of active extrinsic.
- **Frame/sign mismatch across modules:** enforce convention contract + startup assertions + known-point tests before long runs.

---

## Incremental Delivery Order

1. Add convention assertion checks and run-header convention logging (must run before long training jobs).
2. Add projection/back-projection helpers + range attenuation.
3. Add elevation-aware initialization and fixed-opacity toggle.
4. Integrate Stage 1 likelihood bins + annealing.
5. Integrate belief-to-geometry coupling loss (expected-point to surfel).
6. Add multi-view support tracking and retention/pruning schedule.
7. Add staged normals update path.
8. Enable optional Stage 2 densification.

---

## Authorship Marker

- **opus4.5**: Base-plan technical decisions incorporated here (physics, normals, correspondence direction)
- **opus4.6**: Consistency-gap review that prompted contract clarifications
- **gpt-5.2-codex**: Original draft
- **gpt-5.3-codex**: Alignment update to latest base-plan decisions
