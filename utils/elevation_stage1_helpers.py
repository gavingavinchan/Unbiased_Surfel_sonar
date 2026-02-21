import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


CHECKPOINT_SCHEMA_VERSION = "chunk3_stage1_v1"


def assert_frame_keys_unique(frame_keys: Sequence[str]) -> None:
    seen = set()
    dups = []
    for key in frame_keys:
        key_str = str(key)
        if key_str in seen and key_str not in dups:
            dups.append(key_str)
        seen.add(key_str)
    if dups:
        raise ValueError(f"Duplicate frame_key values: {dups}")


def combine_pose_overlap_score(
    yaw_score: float,
    baseline_score: float,
    w_yaw: float,
    w_base: float,
) -> float:
    return float(w_yaw) * float(yaw_score) + float(w_base) * float(baseline_score)


def pose_only_hard_gate(
    baseline_m: float,
    yaw_deg: float,
    min_baseline_m: float,
    max_yaw_deg: float,
) -> bool:
    return (float(baseline_m) >= float(min_baseline_m)) and (abs(float(yaw_deg)) <= float(max_yaw_deg))


def rank_overlap_candidates(candidates: Sequence[Tuple[str, float]], topk: int) -> List[Tuple[str, float]]:
    ordered = sorted(candidates, key=lambda kv: (-float(kv[1]), kv[0]))
    return ordered[: max(0, int(topk))]


@dataclass(frozen=True)
class RoundRobinSamplerState:
    frame_keys: List[str]
    cursor: int = 0
    epoch: int = 0


def round_robin_sample(state: RoundRobinSamplerState, frames_per_iter: int) -> Tuple[List[str], RoundRobinSamplerState]:
    keys = list(state.frame_keys)
    if not keys:
        return [], RoundRobinSamplerState(frame_keys=keys, cursor=0, epoch=state.epoch)

    n = len(keys)
    take = max(0, int(frames_per_iter))
    cursor = int(state.cursor) % n
    epoch = int(state.epoch)

    out: List[str] = []
    for _ in range(take):
        out.append(keys[cursor])
        cursor += 1
        if cursor >= n:
            cursor = 0
            epoch += 1

    return out, RoundRobinSamplerState(frame_keys=keys, cursor=cursor, epoch=epoch)


def normalize_by_percentiles(x: torch.Tensor, lo: float, hi: float, eps: float = 1e-6) -> torch.Tensor:
    denom = max(float(hi) - float(lo), float(eps))
    y = (x - float(lo)) / denom
    return y.clamp(0.0, 1.0)


def masked_softmax(logits: torch.Tensor, support_mask: torch.Tensor, dim: int = -1, min_support: float = 1e-6) -> torch.Tensor:
    mask = support_mask.to(dtype=torch.bool)
    logits_masked = logits.masked_fill(~mask, float("-inf"))
    probs = torch.zeros_like(logits)

    support_any = mask.any(dim=dim, keepdim=True)
    if support_any.any():
        soft = torch.softmax(logits_masked, dim=dim)
        soft = torch.where(mask, soft, torch.zeros_like(soft))
        probs = torch.where(support_any, soft, probs)

    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    sums = probs.sum(dim=dim, keepdim=True)
    good = sums > float(min_support)
    probs = torch.where(good, probs / sums.clamp_min(float(min_support)), probs)
    return probs


def linear_anneal(iteration: int, start: float, end: float, horizon: int) -> float:
    h = max(1, int(horizon))
    progress = max(0.0, min(float(iteration) / float(h), 1.0))
    return float(start) + (float(end) - float(start)) * progress


def resolve_temperatures(
    iteration: int,
    temp_start: float,
    temp_end: float,
    temp_post_mode: str,
    temp_post_start: float,
    temp_post_end: float,
    horizon: int,
) -> Tuple[float, float]:
    t_model = linear_anneal(iteration=iteration, start=temp_start, end=temp_end, horizon=horizon)
    mode = str(temp_post_mode)
    if mode == "shared":
        return t_model, t_model
    if mode == "decoupled":
        t_post = linear_anneal(iteration=iteration, start=temp_post_start, end=temp_post_end, horizon=horizon)
        return t_model, t_post
    raise ValueError(f"Invalid temp_post_mode: {temp_post_mode}")


def should_refresh_pixel_bank(iteration: int, refresh_interval: int) -> bool:
    interval = int(refresh_interval)
    if interval <= 0:
        return False
    return int(iteration) % interval == 0


def remap_or_reset_pixel_logits(
    old_rows: torch.Tensor,
    old_cols: torch.Tensor,
    old_logits: torch.Tensor,
    new_rows: torch.Tensor,
    new_cols: torch.Tensor,
    remap_mode: str,
    remap_max_dist: int,
) -> torch.Tensor:
    if old_rows.ndim != 1 or old_cols.ndim != 1 or new_rows.ndim != 1 or new_cols.ndim != 1:
        raise ValueError("rows/cols must be 1D tensors")
    if old_rows.shape[0] != old_cols.shape[0] or new_rows.shape[0] != new_cols.shape[0]:
        raise ValueError("rows/cols length mismatch")
    if old_logits.ndim != 2:
        raise ValueError("old_logits must be [K, B]")
    if old_logits.shape[0] != old_rows.shape[0]:
        raise ValueError("old_logits K must match old pixel count")

    k_new = int(new_rows.shape[0])
    b = int(old_logits.shape[1])
    out = torch.zeros((k_new, b), dtype=old_logits.dtype, device=old_logits.device)

    mode = str(remap_mode)
    if mode == "reset":
        return out
    if mode != "nearest":
        raise ValueError(f"Invalid remap_mode: {remap_mode}")

    if old_rows.numel() == 0 or k_new == 0:
        return out

    old_rows_f = old_rows.to(dtype=torch.float32)
    old_cols_f = old_cols.to(dtype=torch.float32)
    max_dist = float(remap_max_dist)

    for i in range(k_new):
        d_row = torch.abs(old_rows_f - float(new_rows[i].item()))
        d_col = torch.abs(old_cols_f - float(new_cols[i].item()))
        d_l1 = d_row + d_col
        min_dist, idx = torch.min(d_l1, dim=0)
        if float(min_dist.item()) <= max_dist:
            out[i] = old_logits[int(idx.item())]

    return out


def optimizer_rebuild_required(old_logits: torch.Tensor, new_logits: torch.Tensor) -> bool:
    return tuple(old_logits.shape) != tuple(new_logits.shape)


def compute_frame_stats(
    gt_frame: torch.Tensor,
    valid_mask: torch.Tensor,
    p_lo: float,
    p_hi: float,
    rel_floor: float,
    rel_valid_min: float,
    rel_valid_max: float,
    rel_dyn_min: float,
    rel_dyn_max: float,
) -> Dict[str, float]:
    valid = valid_mask.to(dtype=torch.bool)
    numel = int(valid.numel())
    valid_count = int(valid.sum().item())
    valid_ratio = float(valid_count / max(1, numel))

    if valid_count == 0:
        return {
            "p_lo": 0.0,
            "p_hi": 0.0,
            "dyn_range": 0.0,
            "valid_ratio": valid_ratio,
            "reliability": float(rel_floor),
        }

    values = gt_frame[valid].float()
    lo_v = float(torch.quantile(values, float(p_lo) / 100.0).item())
    hi_v = float(torch.quantile(values, float(p_hi) / 100.0).item())
    dyn = max(0.0, hi_v - lo_v)

    if valid_ratio < float(rel_valid_min):
        rel = float(rel_floor)
    else:
        valid_term = (valid_ratio - float(rel_valid_min)) / max(float(rel_valid_max) - float(rel_valid_min), 1e-8)
        dyn_term = (dyn - float(rel_dyn_min)) / max(float(rel_dyn_max) - float(rel_dyn_min), 1e-8)
        valid_term = max(0.0, min(valid_term, 1.0))
        dyn_term = max(0.0, min(dyn_term, 1.0))
        rel = max(float(rel_floor), 0.5 * (valid_term + dyn_term))

    return {
        "p_lo": lo_v,
        "p_hi": hi_v,
        "dyn_range": dyn,
        "valid_ratio": valid_ratio,
        "reliability": float(rel),
    }


def compute_active_frame_fingerprint(active_frame_keys: Sequence[str]) -> str:
    payload = "\n".join(active_frame_keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_stage1_checkpoint_payload(
    overlap_table,
    sampler_state,
    pixel_bank,
    pixel_logits,
    optim_elev_state,
    active_frame_keys: Sequence[str],
):
    keys = list(active_frame_keys)
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "active_frame_keys": keys,
        "active_frame_fingerprint": compute_active_frame_fingerprint(keys),
        "overlap_table": overlap_table,
        "sampler_state": sampler_state,
        "pixel_bank": pixel_bank,
        "pixel_logits": pixel_logits,
        "optim_elev_state": optim_elev_state,
    }


def resolve_stage1_resume_action(
    checkpoint_schema_version: str,
    runtime_schema_version: str,
    frame_fingerprint_matches: bool,
    mismatch_policy: str,
) -> str:
    schema_ok = checkpoint_schema_version == runtime_schema_version
    all_ok = schema_ok and bool(frame_fingerprint_matches)
    if all_ok:
        return "load"

    policy = str(mismatch_policy)
    if policy == "strict":
        if not schema_ok:
            raise ValueError("Stage-1 resume blocked by schema mismatch")
        raise ValueError("Stage-1 resume blocked by active frame-set mismatch")
    if policy == "reset_frame":
        return "reset_frame"
    if policy == "reset_all":
        return "reset_all"
    raise ValueError(f"Unknown mismatch policy: {mismatch_policy}")


def resolve_effective_stage1_mode(elevation_aware: bool, requested_mode: str) -> str:
    mode = str(requested_mode)
    if mode not in {"off", "shadow", "active"}:
        raise ValueError(f"Invalid stage1 mode: {requested_mode}")
    if not bool(elevation_aware):
        return "off"
    return mode


def run_stage1_likelihood_step(
    logits: torch.Tensor,
    loglik: torch.Tensor,
    support_mask: torch.Tensor,
    mode: str,
    lik_weight: float,
    entropy_weight: float,
    temp_model: float,
    temp_post: float,
    lik_tgt_temp: float,
    min_support: float,
):
    zero = logits.new_tensor(0.0)
    if mode == "off":
        return {
            "cached_loglik": None,
            "cached_support_mask": None,
            "p_post": None,
            "loss_lik": zero,
            "loss_ent": zero,
            "stage1_total_loss": zero,
        }

    cached_loglik = loglik.detach()
    cached_support_mask = support_mask.detach()

    p_model = masked_softmax(logits / max(float(temp_model), 1e-8), support_mask, dim=-1, min_support=min_support)
    p_tgt = masked_softmax(loglik / max(float(lik_tgt_temp), 1e-8), support_mask, dim=-1, min_support=min_support).detach()
    p_post = masked_softmax(
        logits / max(float(temp_post), 1e-8) + loglik,
        support_mask,
        dim=-1,
        min_support=min_support,
    )

    valid_rows = support_mask.any(dim=-1)
    ce = -(p_tgt * torch.log(p_model.clamp_min(1e-8))).sum(dim=-1)
    ent = -(p_post * torch.log(p_post.clamp_min(1e-8))).sum(dim=-1)

    if valid_rows.any():
        loss_lik = ce[valid_rows].mean()
        loss_ent = ent[valid_rows].mean()
    else:
        loss_lik = zero
        loss_ent = zero

    if mode == "shadow":
        total = zero
    elif mode == "active":
        total = float(lik_weight) * loss_lik + float(entropy_weight) * loss_ent
    else:
        raise ValueError(f"Invalid stage1 mode: {mode}")

    return {
        "cached_loglik": cached_loglik,
        "cached_support_mask": cached_support_mask,
        "p_post": p_post,
        "loss_lik": loss_lik,
        "loss_ent": loss_ent,
        "stage1_total_loss": total,
    }
