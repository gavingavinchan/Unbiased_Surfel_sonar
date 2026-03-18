import hashlib
import math
from typing import Dict, Optional, Sequence, Tuple

import torch


CHECKPOINT_SCHEMA_VERSION = "chunk5_normals_densify_v1"
CHECKPOINT_PAYLOAD_KEY = "elevation_chunk5_state"


def resolve_normal_weight(
    *,
    iteration: int,
    ramp_start_iter: int,
    ramp_end_iter: int,
    weight_early: float,
    weight_late: float,
) -> float:
    iteration_i = int(iteration)
    start_i = int(ramp_start_iter)
    end_i = int(ramp_end_iter)
    early = float(weight_early)
    late = float(weight_late)

    if iteration_i < start_i:
        return early
    if iteration_i > end_i:
        return late

    progress = float(iteration_i - start_i) / float(max(1, end_i - start_i))
    return early + progress * (late - early)


def compute_expected_elevation(probs: torch.Tensor, elev_bins: torch.Tensor) -> torch.Tensor:
    probs_t = probs.to(dtype=torch.float32)
    bins_t = elev_bins.to(device=probs_t.device, dtype=torch.float32)
    weights = probs_t / probs_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.sum(weights * bins_t.unsqueeze(0), dim=-1)


def compute_finite_difference_normals(
    *,
    pts_left: torch.Tensor,
    pts_right: torch.Tensor,
    pts_up: torch.Tensor,
    pts_down: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    dp_az = pts_right - pts_left
    dp_rg = pts_down - pts_up
    n_fd = torch.cross(dp_az, dp_rg, dim=-1)
    return n_fd / n_fd.norm(dim=-1, keepdim=True).clamp_min(float(eps))


def compute_confidence_mask(
    *,
    probs: torch.Tensor,
    support_mask: torch.Tensor,
    confidence_thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs_t = probs.to(dtype=torch.float32)
    support_t = support_mask.to(dtype=torch.bool)
    entropy = -torch.sum(probs_t * torch.log(probs_t.clamp_min(1e-8)), dim=-1)
    k = int(probs_t.shape[-1]) if probs_t.ndim > 0 else 0
    max_entropy = math.log(max(k, 1))
    supported = support_t.any(dim=-1)

    if max_entropy <= 0.0:
        confident = supported & torch.isfinite(entropy)
    else:
        confident = supported & torch.isfinite(entropy) & (entropy < (float(confidence_thresh) * max_entropy))
    return confident, entropy


def compute_normal_supervision_loss(*, n_quat: torch.Tensor, n_expected: torch.Tensor) -> torch.Tensor:
    cosine = torch.sum(n_quat * n_expected, dim=-1)
    loss = 1.0 - torch.abs(cosine)
    return loss.mean() if loss.numel() > 0 else n_quat.new_tensor(0.0)


def resolve_effective_chunk5_modes(
    *,
    elevation_aware: bool,
    requested_normal_mode: str,
    densify_enabled: bool,
    requested_densify_mode: str,
) -> Tuple[str, str]:
    normal_mode = str(requested_normal_mode)
    densify_mode = str(requested_densify_mode)
    valid_modes = {"off", "shadow", "active"}

    if normal_mode not in valid_modes:
        raise ValueError(f"Invalid normal mode: {requested_normal_mode}")
    if densify_mode not in valid_modes:
        raise ValueError(f"Invalid densify mode: {requested_densify_mode}")

    if not bool(elevation_aware):
        return "off", "off"
    if not bool(densify_enabled):
        densify_mode = "off"
    return normal_mode, densify_mode


def mode_enables_normal_loss(mode: str) -> bool:
    return str(mode) == "active"


def mode_enables_densify_candidates(mode: str) -> bool:
    return str(mode) in {"shadow", "active"}


def mode_enables_densify_spawn(mode: str) -> bool:
    return str(mode) == "active"


def is_densify_iteration_eligible(*, iteration: int, stage2_start_iter: int, densify_interval: int) -> bool:
    interval_i = int(densify_interval)
    if interval_i <= 0:
        return False
    return int(iteration) >= int(stage2_start_iter) and (int(iteration) % interval_i == 0)


def compute_arc_bin_scores(
    *,
    gt_intensity: torch.Tensor,
    valid_mask: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    gt_t = gt_intensity.to(dtype=torch.float32)
    valid_t = valid_mask.to(device=gt_t.device, dtype=torch.float32)
    rel_t = reliability.to(device=gt_t.device, dtype=torch.float32).reshape(-1, 1)
    return torch.sum(rel_t * valid_t * gt_t, dim=0)


def select_arc_peak_bin(*, scores: torch.Tensor, min_score: float) -> Optional[int]:
    scores_t = scores.to(dtype=torch.float32)
    if scores_t.numel() == 0 or not torch.isfinite(scores_t).any():
        return None
    peak = int(torch.argmax(scores_t).item())
    if float(scores_t[peak].item()) < float(min_score):
        return None
    return peak


def _compute_frame_fingerprint(active_frame_keys: Sequence[str]) -> str:
    keys = [str(k) for k in active_frame_keys]
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def build_chunk5_checkpoint_payload(
    *,
    normal_mode: str,
    densify_enabled: bool,
    densify_mode: str,
    densify_event_count: int,
    high_error_tracker: Dict[str, int],
    active_frame_keys: Sequence[str],
) -> Dict[str, object]:
    frame_keys = [str(k) for k in active_frame_keys]
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "active_frame_keys": frame_keys,
        "active_frame_fingerprint": _compute_frame_fingerprint(frame_keys),
        "normal_mode": str(normal_mode),
        "densify_enabled": bool(densify_enabled),
        "densify_mode": str(densify_mode),
        "densify_event_count": int(densify_event_count),
        "high_error_tracker": {str(k): int(v) for k, v in dict(high_error_tracker).items()},
    }


def resolve_chunk5_resume_action(
    *,
    checkpoint_schema_version: str,
    runtime_schema_version: str,
    frame_fingerprint_matches: bool,
    mismatch_policy: str,
) -> str:
    policy = str(mismatch_policy)
    if policy not in {"strict", "reset_frame", "reset_all"}:
        raise ValueError(f"Invalid mismatch_policy: {mismatch_policy}")

    schema_matches = str(checkpoint_schema_version) == str(runtime_schema_version)
    if schema_matches and bool(frame_fingerprint_matches):
        return "load"

    if policy == "strict":
        if not schema_matches:
            raise ValueError("Chunk-5 resume schema mismatch under strict policy")
        raise ValueError("Chunk-5 resume frame-set mismatch under strict policy")
    return policy
