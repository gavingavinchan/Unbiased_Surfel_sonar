import importlib.util
import math
import py_compile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "utils" / "elevation_stage1_helpers.py"
assert HELPER_PATH.exists(), "Missing helper module: utils/elevation_stage1_helpers.py"


def _load_helper_module():
    spec = importlib.util.spec_from_file_location("elevation_stage1_helpers", HELPER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage1 = _load_helper_module()


def _toy_inputs():
    # Keep at least one valid-support row so active-mode loss stays > 0 for this smoke fixture.
    logits = torch.tensor(
        [
            [[0.2, -0.1, 0.0], [0.5, 0.0, -0.2]],
            [[-0.3, 0.1, 0.4], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    loglik = torch.tensor(
        [
            [[-0.1, -0.2, -0.3], [-0.4, -0.1, -0.2]],
            [[-0.2, -0.5, -0.3], [-0.2, -0.2, -0.2]],
        ],
        dtype=torch.float32,
    )
    support_mask = torch.tensor(
        [
            [[1, 1, 1], [1, 0, 1]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    return logits, loglik, support_mask


def test_mode_resolution_elevation_aware_override_contract():
    assert stage1.resolve_effective_stage1_mode(elevation_aware=False, requested_mode="active") == "off"
    assert stage1.resolve_effective_stage1_mode(elevation_aware=True, requested_mode="off") == "off"
    assert stage1.resolve_effective_stage1_mode(elevation_aware=True, requested_mode="shadow") == "shadow"
    assert stage1.resolve_effective_stage1_mode(elevation_aware=True, requested_mode="active") == "active"


@pytest.mark.parametrize("mode", ["off", "shadow", "active"])
def test_stage1_smoke_modes_no_crash_and_finite_outputs(mode):
    logits, loglik, support_mask = _toy_inputs()

    out = stage1.run_stage1_likelihood_step(
        logits=logits,
        loglik=loglik,
        support_mask=support_mask,
        mode=mode,
        lik_weight=1.0,
        entropy_weight=0.01,
        temp_model=1.0,
        temp_post=1.0,
        lik_tgt_temp=1.0,
        min_support=1e-6,
    )

    if mode == "off":
        assert out["cached_loglik"] is None
        assert out["cached_support_mask"] is None
        assert out["p_post"] is None
        assert out["stage1_total_loss"].item() == pytest.approx(0.0)
        return

    assert out["cached_loglik"] is not None
    assert out["cached_support_mask"] is not None
    assert out["p_post"] is not None
    assert out["p_post"].shape == logits.shape
    assert torch.isfinite(out["p_post"]).all()
    assert torch.isfinite(out["cached_loglik"]).all()
    assert torch.isfinite(out["stage1_total_loss"]).all()

    if mode == "shadow":
        assert out["stage1_total_loss"].item() == pytest.approx(0.0)
    else:
        assert out["loss_lik"].item() >= 0.0
        assert out["loss_ent"].item() >= 0.0
        assert out["stage1_total_loss"].item() > 0.0


def test_stage1_all_masked_rows_stay_finite():
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    loglik = torch.zeros((1, 2, 3), dtype=torch.float32)
    support_mask = torch.zeros((1, 2, 3), dtype=torch.bool)

    out = stage1.run_stage1_likelihood_step(
        logits=logits,
        loglik=loglik,
        support_mask=support_mask,
        mode="shadow",
        lik_weight=1.0,
        entropy_weight=0.01,
        temp_model=1.0,
        temp_post=1.0,
        lik_tgt_temp=1.0,
        min_support=1e-6,
    )

    assert torch.isfinite(out["p_post"]).all()
    assert torch.allclose(out["p_post"], torch.zeros_like(out["p_post"]))


def test_stage1_all_masked_rows_active_has_zero_losses():
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    loglik = torch.zeros((1, 2, 3), dtype=torch.float32)
    support_mask = torch.zeros((1, 2, 3), dtype=torch.bool)

    out = stage1.run_stage1_likelihood_step(
        logits=logits,
        loglik=loglik,
        support_mask=support_mask,
        mode="active",
        lik_weight=1.0,
        entropy_weight=1.0,
        temp_model=1.0,
        temp_post=1.0,
        lik_tgt_temp=1.0,
        min_support=1e-6,
    )
    assert out["loss_lik"].item() == pytest.approx(0.0)
    assert out["loss_ent"].item() == pytest.approx(0.0)
    assert out["stage1_total_loss"].item() == pytest.approx(0.0)


def test_entropy_sign_and_magnitude_known_uniform_fixture():
    logits = torch.zeros((1, 1, 2), dtype=torch.float32)
    loglik = torch.zeros((1, 1, 2), dtype=torch.float32)
    support_mask = torch.ones((1, 1, 2), dtype=torch.bool)

    out_uniform = stage1.run_stage1_likelihood_step(
        logits=logits,
        loglik=loglik,
        support_mask=support_mask,
        mode="active",
        lik_weight=0.0,
        entropy_weight=1.0,
        temp_model=1.0,
        temp_post=1.0,
        lik_tgt_temp=1.0,
        min_support=1e-6,
    )

    assert out_uniform["loss_ent"].item() == pytest.approx(math.log(2.0), rel=1e-5)
    assert out_uniform["stage1_total_loss"].item() == pytest.approx(math.log(2.0), rel=1e-5)

    peaked_logits = torch.tensor([[[8.0, 0.0]]], dtype=torch.float32)
    out_peaked = stage1.run_stage1_likelihood_step(
        logits=peaked_logits,
        loglik=loglik,
        support_mask=support_mask,
        mode="active",
        lik_weight=0.0,
        entropy_weight=1.0,
        temp_model=1.0,
        temp_post=1.0,
        lik_tgt_temp=1.0,
        min_support=1e-6,
    )
    assert out_peaked["loss_ent"].item() < out_uniform["loss_ent"].item()


def test_ce_path_backprop_hits_logits_not_evidence_branch():
    logits = torch.nn.Parameter(torch.zeros((1, 2, 2), dtype=torch.float32))
    loglik = torch.tensor([[[-3.0, 0.0], [-2.0, 0.0]]], dtype=torch.float32, requires_grad=True)
    support_mask = torch.ones((1, 2, 2), dtype=torch.bool)

    out = stage1.run_stage1_likelihood_step(
        logits=logits,
        loglik=loglik,
        support_mask=support_mask,
        mode="active",
        lik_weight=1.0,
        entropy_weight=0.0,
        temp_model=1.0,
        temp_post=1.0,
        lik_tgt_temp=1.0,
        min_support=1e-6,
    )
    out["stage1_total_loss"].backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum().item()) > 0.0
    assert (loglik.grad is None) or torch.allclose(loglik.grad, torch.zeros_like(loglik.grad))


def test_debug_multiframe_script_compiles_for_smoke():
    target = REPO_ROOT / "debug_multiframe.py"
    py_compile.compile(str(target), doraise=True)


def test_stage1_helper_module_compiles_for_smoke():
    py_compile.compile(str(HELPER_PATH), doraise=True)
