"""Spectral covariance clipping must have finite derivatives at repeated roots."""
import pytest
import torch
from test_renderer_baseline_r0_red_densification_contracts import _load_renderer_fn

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('value',[.01,1.,500.])
def test_repeated_eigenvalues_have_the_correct_derivative(dtype,value):
    f=_load_renderer_fn('_condition_sigma_2d')
    x=(torch.eye(2,dtype=dtype)*value).requires_grad_()
    weights=torch.tensor([[1.,.3],[.3,2.]],dtype=dtype)
    output=f(x)
    (output*weights).sum().backward()
    assert torch.isfinite(x.grad).all()
    expected=weights if .1<value<400 else torch.zeros_like(weights)
    torch.testing.assert_close(x.grad,expected)
    torch.testing.assert_close(output,torch.eye(2,dtype=dtype)*min(max(value,.1),400.))

@pytest.mark.parametrize('value',[.01,1.,500.])
def test_repeated_eigenvalue_derivative_matches_finite_difference(value):
    f=_load_renderer_fn('_condition_sigma_2d')
    x=(torch.eye(2,dtype=torch.float64)*value).requires_grad_()
    weights=torch.tensor([[1.,.3],[.3,2.]],dtype=x.dtype)
    (f(x)*weights).sum().backward()
    h=1e-5
    for direction in [torch.eye(2,dtype=x.dtype),torch.tensor([[1.,0.],[0.,-1.]],dtype=x.dtype),torch.tensor([[0.,1.],[1.,0.]],dtype=x.dtype)]:
        fd=((f(x.detach()+h*direction)-f(x.detach()-h*direction))*weights).sum()/(2*h)
        torch.testing.assert_close((x.grad*direction).sum(),fd,atol=1e-8,rtol=1e-6)

@pytest.mark.parametrize('cond_cap',[None,1.,10.,100.])
def test_spectral_clipping_matches_original_eigendecomposition_away_from_ties(cond_cap):
    f=_load_renderer_fn('_condition_sigma_2d')
    generator=torch.Generator().manual_seed(53)
    # Include indefinite input, upper/lower clipping and a condition-number cap.
    a=torch.randn(64,2,2,generator=generator,dtype=torch.float64)
    x=((a+a.transpose(-1,-2))*250).requires_grad_()
    weights=torch.randn(64,2,2,generator=generator,dtype=x.dtype)
    values,vectors=torch.linalg.eigh(x)
    clipped=values.clamp(.1,400.)
    if cond_cap is not None and cond_cap>0:
        clipped=torch.maximum(clipped,(clipped.max(-1,keepdim=True).values/cond_cap).clamp_min(.1)).clamp_max(400.)
    expected=vectors@torch.diag_embed(clipped)@vectors.transpose(-1,-2)
    result=f(x,cond_cap=cond_cap)
    torch.testing.assert_close(result,expected,atol=1e-10,rtol=1e-10)
    actual_grad=torch.autograd.grad((result*weights).sum(),x,retain_graph=True)[0]
    expected_grad=torch.autograd.grad((expected*weights).sum(),x)[0]
    torch.testing.assert_close(actual_grad,expected_grad,atol=1e-9,rtol=1e-9)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('batched',[False,True])
def test_rasterizer_transform_keeps_circular_footprint_gradients_finite(dtype,batched):
    name='_sigma2d_to_transmat_precomp_batch' if batched else 'sigma2d_to_transmat_precomp'
    f=_load_renderer_fn(name)
    x=torch.eye(2,dtype=dtype).requires_grad_()
    mu=torch.tensor([3.,5.],dtype=dtype,requires_grad=True)
    result=f(mu[None],x[None]) if batched else f(mu,x)
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad,torch.eye(2,dtype=dtype))
    torch.testing.assert_close(mu.grad,2*mu)

@pytest.mark.parametrize('gap',[1e-3,1e-6,1e-9,1e-12])
def test_nearly_repeated_unclipped_matrix_has_identity_derivative(gap):
    f=_load_renderer_fn('_condition_sigma_2d')
    x=torch.tensor([[1.,gap],[gap,1.]],dtype=torch.float64,requires_grad=True)
    weights=torch.tensor([[1.,.3],[.3,2.]],dtype=x.dtype)
    y=f(x)
    (y*weights).sum().backward()
    torch.testing.assert_close(y,x,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(x.grad,weights,atol=1e-8,rtol=1e-8)
