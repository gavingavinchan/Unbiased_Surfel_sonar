import hashlib
from typing import Dict, Sequence, Tuple

import torch


CHECKPOINT_SCHEMA_VERSION = "chunk4_coupling_support_v1"


def resolve_effective_chunk4_modes(
    elevation_aware: bool,
    requested_couple_mode: str,
    requested_support_mode: str,
) -> Tuple[str, str]:
    couple_mode = str(requested_couple_mode)
    support_mode = str(requested_support_mode)
    valid_modes = {"off", "shadow", "active"}

    if couple_mode not in valid_modes:
        raise ValueError(f"Invalid couple mode: {requested_couple_mode}")
    if support_mode not in valid_modes:
        raise ValueError(f"Invalid support mode: {requested_support_mode}")

    if not bool(elevation_aware):
        return "off", "off"
    return couple_mode, support_mode


def mode_enables_weighted_coupling(mode: str) -> bool:
    return str(mode) == "active"


def mode_enables_hard_prune(mode: str) -> bool:
    return str(mode) == "active"


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    d = torch.abs(x)
    delta_v = float(delta)
    quad = 0.5 * d * d
    lin = delta_v * (d - 0.5 * delta_v)
    return torch.where(d <= delta_v, quad, lin)


def associate_expected_points_to_surfels(
    exp_row: torch.Tensor,
    exp_col: torch.Tensor,
    exp_depth: torch.Tensor,
    exp_valid: torch.Tensor,
    surf_row: torch.Tensor,
    surf_col: torch.Tensor,
    surf_depth: torch.Tensor,
    surf_valid: torch.Tensor,
    max_pix_err: float,
    max_depth_err: float,
    sigma_pix: float,
    sigma_depth: float,
    min_w: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = int(exp_row.shape[0])
    n = int(surf_row.shape[0])
    device = exp_row.device

    surf_idx = torch.zeros((p,), dtype=torch.long, device=device)
    assoc_w = torch.zeros((p,), dtype=exp_row.dtype, device=device)
    match_valid = torch.zeros((p,), dtype=torch.bool, device=device)

    if p == 0 or n == 0:
        return surf_idx, assoc_w, match_valid

    max_pix = float(max_pix_err)
    max_depth = float(max_depth_err)
    sig_pix = max(float(sigma_pix), 1e-8)
    sig_depth = max(float(sigma_depth), 1e-8)
    min_w_v = float(min_w)

    surf_valid_b = surf_valid.to(dtype=torch.bool)
    for i in range(p):
        if not bool(exp_valid[i].item()):
            continue

        d_row = torch.abs(surf_row - exp_row[i])
        d_col = torch.abs(surf_col - exp_col[i])
        pix_err = torch.sqrt(d_row * d_row + d_col * d_col)
        depth_err = torch.abs(surf_depth - exp_depth[i])

        gate = surf_valid_b & (pix_err <= max_pix) & (depth_err <= max_depth)
        if not bool(gate.any().item()):
            continue

        score = (pix_err / sig_pix) ** 2 + (depth_err / sig_depth) ** 2
        score_gated = torch.where(gate, score, torch.full_like(score, float("inf")))
        best = int(torch.argmin(score_gated).item())
        s = float(score_gated[best].item())
        if not torch.isfinite(score_gated[best]):
            continue

        w = max(min_w_v, min(1.0, float(torch.exp(torch.tensor(-0.5 * s)).item())))
        surf_idx[i] = best
        assoc_w[i] = w
        match_valid[i] = True

    return surf_idx, assoc_w, match_valid


def reduce_coupling_loss(
    pts_expected: torch.Tensor,
    surfel_xyz: torch.Tensor,
    surf_idx: torch.Tensor,
    assoc_w: torch.Tensor,
    match_valid: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    zero = pts_expected.new_tensor(0.0)
    if not bool(match_valid.any().item()):
        return zero

    idx = surf_idx[match_valid]
    pts = pts_expected[match_valid]
    w = assoc_w[match_valid]

    d = torch.norm(surfel_xyz[idx] - pts, dim=-1)
    robust = _huber(d, delta=float(huber_delta))
    return (w * robust).sum() / w.sum().clamp_min(1e-8)


def _clone_state(state: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def _build_id_to_row(surfel_ids: torch.Tensor, next_surfel_id: int) -> torch.Tensor:
    out = torch.full((int(next_surfel_id),), -1, dtype=torch.long, device=surfel_ids.device)
    if surfel_ids.numel() > 0:
        out[surfel_ids] = torch.arange(surfel_ids.shape[0], dtype=torch.long, device=surfel_ids.device)
    return out


def initialize_persistent_surfel_state(
    num_surfels: int,
    device: str = "cpu",
    init_birth_iter: int = 0,
) -> Dict[str, object]:
    n = int(num_surfels)
    surfel_ids = torch.arange(n, dtype=torch.long, device=device)
    next_surfel_id = n
    return {
        "surfel_ids": surfel_ids,
        "next_surfel_id": next_surfel_id,
        "id_to_row": _build_id_to_row(surfel_ids, next_surfel_id),
        "ema_by_id": torch.zeros((next_surfel_id,), dtype=torch.float32, device=device),
        "last_raw_by_id": torch.zeros((next_surfel_id,), dtype=torch.float32, device=device),
        "birth_iter_by_id": torch.full(
            (next_surfel_id,), int(init_birth_iter), dtype=torch.long, device=device
        ),
        "fail_streak_by_id": torch.zeros((next_surfel_id,), dtype=torch.long, device=device),
    }


def apply_densify_to_surfel_state(state: Dict[str, object], n_new: int, current_iter: int) -> Dict[str, object]:
    out = _clone_state(state)
    add = max(0, int(n_new))
    if add == 0:
        return out

    surfel_ids = out["surfel_ids"]
    next_surfel_id = int(out["next_surfel_id"])
    device = surfel_ids.device

    new_ids = torch.arange(next_surfel_id, next_surfel_id + add, dtype=torch.long, device=device)
    out["surfel_ids"] = torch.cat([surfel_ids, new_ids], dim=0)
    out["next_surfel_id"] = next_surfel_id + add

    out["ema_by_id"] = torch.cat(
        [out["ema_by_id"], torch.zeros((add,), dtype=torch.float32, device=device)], dim=0
    )
    out["last_raw_by_id"] = torch.cat(
        [out["last_raw_by_id"], torch.zeros((add,), dtype=torch.float32, device=device)], dim=0
    )
    out["birth_iter_by_id"] = torch.cat(
        [
            out["birth_iter_by_id"],
            torch.full((add,), int(current_iter), dtype=torch.long, device=device),
        ],
        dim=0,
    )
    out["fail_streak_by_id"] = torch.cat(
        [out["fail_streak_by_id"], torch.zeros((add,), dtype=torch.long, device=device)], dim=0
    )
    out["id_to_row"] = _build_id_to_row(out["surfel_ids"], int(out["next_surfel_id"]))
    return out


def apply_prune_reorder_to_surfel_state(state: Dict[str, object], keep_row_idx: torch.Tensor) -> Dict[str, object]:
    out = _clone_state(state)
    keep = keep_row_idx.to(dtype=torch.long)
    out["surfel_ids"] = out["surfel_ids"][keep]
    out["id_to_row"] = _build_id_to_row(out["surfel_ids"], int(out["next_surfel_id"]))
    return out


def assert_surfel_id_integrity(state: Dict[str, object]) -> None:
    surfel_ids = state["surfel_ids"].to(dtype=torch.long)
    next_surfel_id = int(state["next_surfel_id"])
    id_to_row = state["id_to_row"].to(dtype=torch.long)

    if id_to_row.shape[0] != next_surfel_id:
        raise ValueError("Invalid id_to_row length")

    if surfel_ids.numel() > 0:
        if int(torch.unique(surfel_ids).numel()) != int(surfel_ids.numel()):
            raise ValueError("Duplicate surfel IDs detected")
        if int(surfel_ids.min().item()) < 0 or int(surfel_ids.max().item()) >= next_surfel_id:
            raise ValueError("Surfel IDs out of range")

    expected = _build_id_to_row(surfel_ids, next_surfel_id)
    if not torch.equal(id_to_row, expected):
        raise ValueError("Invalid id_to_row entries")


def update_support_buffers_by_id(
    state: Dict[str, object],
    row_support_raw: torch.Tensor,
    ema_decay: float,
) -> Dict[str, object]:
    out = _clone_state(state)
    surfel_ids = out["surfel_ids"].to(dtype=torch.long)
    raw = row_support_raw.to(dtype=torch.float32)
    if raw.shape[0] != surfel_ids.shape[0]:
        raise ValueError("row_support_raw length mismatch")

    decay = float(ema_decay)
    keep = max(0.0, min(decay, 1.0))
    add = 1.0 - keep

    ema_by_id = out["ema_by_id"]
    last_raw_by_id = out["last_raw_by_id"]
    ema_by_id[surfel_ids] = keep * ema_by_id[surfel_ids] + add * raw
    last_raw_by_id[surfel_ids] = raw
    return out


def compute_support_failure_mask(
    iteration: int,
    support_count_by_id: torch.Tensor,
    diverse_candidate_count_by_id: torch.Tensor,
    warmup_iters: int,
    late_phase_start_iter: int,
    min_ratio_mid: float,
    min_ratio_late: float,
    min_count_mid: int,
    min_count_late: int,
    use_ratio: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    support = support_count_by_id.to(dtype=torch.float32)
    diverse = diverse_candidate_count_by_id.to(dtype=torch.float32)

    if int(iteration) < int(warmup_iters):
        return torch.zeros_like(support, dtype=torch.bool), torch.zeros_like(support)

    late = int(iteration) >= int(late_phase_start_iter)
    ratio_thr = float(min_ratio_late if late else min_ratio_mid)
    count_floor = float(min_count_late if late else min_count_mid)
    floor_eff = torch.minimum(torch.full_like(diverse, count_floor), diverse)

    count_fail = support < floor_eff
    if bool(use_ratio):
        ratio = support / torch.clamp(diverse, min=1.0)
        ratio_fail = ratio < ratio_thr
    else:
        ratio_fail = torch.zeros_like(count_fail)

    fail_mask = count_fail | ratio_fail
    return fail_mask.to(dtype=torch.bool), floor_eff


def apply_prune_hysteresis(
    fail_streak_by_id: torch.Tensor,
    fail_mask_by_id: torch.Tensor,
    patience: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    streak = fail_streak_by_id.to(dtype=torch.long).clone()
    fail = fail_mask_by_id.to(dtype=torch.bool)
    streak = torch.where(fail, streak + 1, torch.zeros_like(streak))
    prune_mask = fail & (streak >= int(patience))
    return streak, prune_mask


def apply_new_surfel_grace(
    prune_mask_by_id: torch.Tensor,
    birth_iter_by_id: torch.Tensor,
    current_iter: int,
    grace_iters: int,
    enabled: bool,
) -> torch.Tensor:
    prune = prune_mask_by_id.to(dtype=torch.bool)
    if not bool(enabled):
        return prune

    age = int(current_iter) - birth_iter_by_id.to(dtype=torch.long)
    grace_active = age < int(grace_iters)
    return prune & (~grace_active)


def _compute_active_frame_fingerprint(active_frame_keys: Sequence[str]) -> str:
    payload = "\n".join(active_frame_keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_chunk4_checkpoint_payload(
    surfel_ids: torch.Tensor,
    next_surfel_id: int,
    id_to_row: torch.Tensor,
    ema_by_id: torch.Tensor,
    last_raw_by_id: torch.Tensor,
    birth_iter_by_id: torch.Tensor,
    fail_streak_by_id: torch.Tensor,
    active_frame_keys: Sequence[str],
    couple_mode: str,
    support_mode: str,
    support_scheduler_state,
    support_count_by_id: torch.Tensor = None,
    diverse_candidate_count_by_id: torch.Tensor = None,
):
    keys = list(active_frame_keys)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "active_frame_keys": keys,
        "active_frame_fingerprint": _compute_active_frame_fingerprint(keys),
        "surfel_ids": surfel_ids,
        "next_surfel_id": int(next_surfel_id),
        "id_to_row": id_to_row,
        "ema_by_id": ema_by_id,
        "last_raw_by_id": last_raw_by_id,
        "birth_iter_by_id": birth_iter_by_id,
        "fail_streak_by_id": fail_streak_by_id,
        "couple_mode": str(couple_mode),
        "support_mode": str(support_mode),
        "support_scheduler_state": support_scheduler_state,
    }
    if support_count_by_id is not None:
        payload["support_count_by_id"] = support_count_by_id
    if diverse_candidate_count_by_id is not None:
        payload["diverse_candidate_count_by_id"] = diverse_candidate_count_by_id
    return payload


def resolve_chunk4_resume_action(
    checkpoint_schema_version: str,
    runtime_schema_version: str,
    frame_fingerprint_matches: bool,
    mismatch_policy: str,
) -> str:
    schema_ok = str(checkpoint_schema_version) == str(runtime_schema_version)
    all_ok = schema_ok and bool(frame_fingerprint_matches)
    if all_ok:
        return "load"

    policy = str(mismatch_policy)
    if policy == "strict":
        if not schema_ok:
            raise ValueError("Chunk-4 resume blocked by schema mismatch")
        raise ValueError("Chunk-4 resume blocked by active frame-set mismatch")
    if policy == "reset_chunk4":
        return "reset_chunk4"
    if policy == "reset_all":
        return "reset_all"
    raise ValueError(f"Unknown mismatch policy: {mismatch_policy}")
