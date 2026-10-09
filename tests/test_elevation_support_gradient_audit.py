"""Regression of mounted native support routing and detached-gradient contracts."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('support_audit',Path(__file__).resolve().parents[1]/'scripts/audit_elevation_support.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_128_sphere_cube_smooth_surrogate_gradients():
    out=audit.smooth_gradients(128)
    assert out['nonfinite_components']==0
    assert max(out['max_relative_errors'].values())<=1e-2


def test_128_native_mounted_support_cases_match_baked_sonar_poses():
    out=audit.extrinsic_cases(audit.load_native(),128)
    assert out['max_coupling_loss_error']<1e-8
    assert out['max_likelihood_error']<1e-5
    assert out['support_mismatches']==0
    assert out['nonfinite_components']==0
    assert out['max_omitted_mount_loss_error']>1e-5
    for case in out['cases']:
        assert case['match_count']==2
        assert case['gradient_norms']['xyz']>0
        for name in ['posterior','rotation','scaling','opacity']:
            assert case['gradient_norms'][name] is None


def test_association_weight_gradient_is_explicitly_a_frozen_surrogate():
    out=audit.association_audit(audit.load_native())
    assert not out['weights_require_grad']
    assert out['frozen_weight_relative_error']<1e-2
    assert out['full_recomputed_weight_relative_error']>1e-2


def test_active_shadow_off_are_distinct_objective_controls():
    out=audit.modes()
    assert out['active']['likelihood_requires_grad']
    assert out['active']['likelihood_loss']>0
    for mode in ['off','shadow']:
        assert out[mode]['likelihood_loss']==0
        assert not out[mode]['coupling_weight_enabled']
    assert out['active']['support_hard_prune_enabled']


def test_128_actual_tangent_covariance_and_opacity_compositing_gradients():
    out=audit.footprint_gradients(audit.load_native(),128)
    assert out['nonfinite_components']==0
    assert max(out['max_relative_errors'].values())<=1e-2
