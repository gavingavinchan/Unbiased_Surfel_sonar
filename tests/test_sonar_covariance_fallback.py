"""Exercise exact renderer caller paths and the rasterizer's packed rows on CPU.

No renderer/CUDA import: AST extraction executes the actual source definitions.
Raw finite matrices retain the existing spectral conditioning policy; invalid
conditioned matrices and raw nonfinite inputs must not become plausible factors.
"""
import ast
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'_bounded_sonar_linalg', '_condition_sigma_2d',
         '_covariance_cholesky_2x2', 'sigma2d_to_transmat_precomp',
         '_sigma2d_to_transmat_precomp_batch', 'transmat_precomp_to_sigma2d'}
def load_renderer():
    tree = ast.parse((ROOT/'gaussian_renderer/__init__.py').read_text())
    ns = {'torch': torch}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {n.name for n in nodes} == NAMES
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-renderer-paths', 'exec'), ns)
    return ns


def rotated_spd(dtype):
    # Include positive/negative correlations, near-ties and spectra needing the
    # existing min/max/condition cap. No assertion of gradients at a clamp kink.
    spectra = [(2.,4.), (.2,20.), (.10001,.10003), (1e-10,3e-10),
               (1e-8,4.), (1.,400.), (400.,800.), (500.,500.1)]
    matrices = []
    for low, high in spectra:
        for theta in torch.linspace(-1.4, 1.4, 8, dtype=dtype):
            c,s = theta.cos(), theta.sin()
            rotation = torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
            matrices.append(rotation@torch.diag(torch.tensor([low,high],dtype=dtype))@rotation.T)
    return torch.stack(matrices)


def fail_cholesky(matrix):
    raise torch.linalg.LinAlgError('injected factorization failure')


def packed_covariance(t):
    # Existing Python decoder regards the packed upper block as L^T.
    # Read the actual nine stored entries, independently of the decoder.
    # IMPORTANT: native forward consumes these as rows instead. Its separate
    # inherited normal-path mismatch is measured by the experiment audit;
    # this scoped correction must not change that packing convention.
    flat = t.reshape(*t.shape[:-2], 9)
    a,b,c = flat[...,0], flat[...,1], flat[...,4]
    return torch.stack([torch.stack([a*a,a*b],-1),
                        torch.stack([a*b,b*b+c*c],-1)],-2)


def native_covariance(t):
    linear = t[..., :2, :2]
    return linear@linear.transpose(-1,-2)


def caller(ns, batch):
    return ns['_sigma2d_to_transmat_precomp_batch' if batch else 'sigma2d_to_transmat_precomp']


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('batch', [False,True])
def test_counterexample_encoded_covariance(dtype,batch):
    ns=load_renderer();sigma=torch.tensor([[2.,.6],[.6,4.]],dtype=dtype)
    if batch:sigma=sigma[None]
    mu=torch.zeros(sigma.shape[:-1],dtype=dtype)
    with patch.object(torch.linalg,'cholesky',fail_cholesky):t=caller(ns,batch)(mu,sigma)
    tolerance=1e-6 if dtype==torch.float32 else 1e-10
    assert (packed_covariance(t)-ns['_condition_sigma_2d'](sigma)).abs().max()<=tolerance
    # Existing decoder is checked too, independently of direct packed-row math.
    for item,expected in zip(t.reshape(-1,3,3),packed_covariance(t).reshape(-1,2,2)):
        torch.testing.assert_close(ns['transmat_precomp_to_sigma2d'](item),expected,atol=0,rtol=0)


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('count', [1,32,513,1025,12001])
def test_rotated_spd_batched_order_tail_and_scalar(dtype,count):
    ns=load_renderer();matrices=rotated_spd(dtype)
    sigma=matrices[torch.arange(count)%len(matrices)]
    mu=torch.arange(count*2,dtype=dtype).reshape(count,2)*.01
    sizes=[]
    def failing_bounded(x):sizes.append(len(x));return fail_cholesky(x)
    with patch.object(torch.linalg,'cholesky',failing_bounded):
        t=caller(ns,True)(mu,sigma)
    # An injected first-chunk failure need not execute later solver chunks.
    assert sizes==[min(count,512)]
    assert t.dtype==dtype and t.device==sigma.device
    torch.testing.assert_close(t[:,:2,2],mu,atol=0,rtol=0)
    expected=ns['_condition_sigma_2d'](sigma)
    tolerance=1e-6 if dtype==torch.float32 else 1e-10
    scale=expected.abs().amax(dim=(-2,-1)).clamp_min(1)
    assert ((packed_covariance(t)-expected).abs().amax(dim=(-2,-1))/scale).max()<=tolerance
    with patch.object(torch.linalg,'cholesky',fail_cholesky):
        serial=torch.stack([caller(ns,False)(m,s) for m,s in zip(mu[:64],sigma[:64])])
    torch.testing.assert_close(t[:64],serial,atol=0,rtol=0)
    normal=caller(ns,True)(mu,sigma)
    torch.testing.assert_close(t,normal,atol=tolerance,rtol=tolerance)


def gradient_measurements(dtype,batch):
    ns=load_renderer();rows=[]
    for theta in torch.linspace(-1.3,1.3,32,dtype=dtype):
        c,s=theta.cos(),theta.sin();r=torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
        sigma=r@torch.diag(torch.tensor([2.,5.],dtype=dtype))@r.T
        p=torch.tensor([sigma[0,0],sigma[0,1],sigma[1,1],.17,-.2],dtype=dtype,requires_grad=True)
        def loss(q):
            matrix=torch.stack([torch.stack([q[0],q[1]]),torch.stack([q[1],q[2]])])
            mu=q[3:]
            if batch:matrix=matrix[None];mu=mu[None]
            t=caller(ns,batch)(mu,matrix)
            weights=torch.tensor([[.7,.3,.2],[.1,.9,-.4],[.5,.6,.8]],dtype=dtype)
            return (t*weights).sum()
        normal=torch.autograd.grad(loss(p),p)[0]
        with patch.object(torch.linalg,'cholesky',fail_cholesky):
            fallback=torch.autograd.grad(loss(p),p)[0]
            eps=1e-3 if dtype==torch.float32 else 1e-5
            fd=[]
            for j in range(len(p)):
                delta=torch.zeros_like(p);delta[j]=eps
                fd.append((loss(p.detach()+delta)-loss(p.detach()-delta))/(2*eps))
        fd=torch.stack(fd)
        relative=float(torch.linalg.vector_norm(fallback-fd)/torch.linalg.vector_norm(fd))
        normal_relative=float(torch.linalg.vector_norm(fallback-normal)/torch.linalg.vector_norm(normal))
        assert torch.isfinite(fallback).all()
        rows.append({'finite_difference_relative':relative,'normal_relative':normal_relative})
    return rows


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('batch', [False,True])
def test_gradients_against_finite_differences_and_normal(dtype,batch):
    rows=gradient_measurements(dtype,batch)
    assert max(r['finite_difference_relative'] for r in rows)<=1e-3
    assert max(r['normal_relative'] for r in rows)<=(1e-6 if dtype==torch.float32 else 1e-10)


def test_fallback_autograd_gradcheck():
    ns=load_renderer()
    sigma=torch.tensor([[2.,.6],[.6,4.]],dtype=torch.float64,requires_grad=True)
    mu=torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True)
    with patch.object(torch.linalg,'cholesky',fail_cholesky):
        assert torch.autograd.gradcheck(caller(ns,False),(mu,sigma),eps=1e-6,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('batch',[False,True])
@pytest.mark.parametrize('error',[torch.OutOfMemoryError('OOM retained'),RuntimeError('unrelated runtime retained')])
def test_required_errors_propagate_without_fallback(batch,error):
    ns=load_renderer();sigma=torch.eye(2).repeat(1025,1,1) if batch else torch.eye(2)
    mu=torch.zeros(sigma.shape[:-1])
    def forbidden(x):raise AssertionError('error entered numerical fallback')
    with patch.object(torch.linalg,'cholesky',side_effect=error),patch.dict(ns,{'_covariance_cholesky_2x2':forbidden}):
        with pytest.raises(type(error),match=str(error)):caller(ns,batch)(mu,sigma)


@pytest.mark.parametrize('batch',[False,True])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_raw_nonfinite_rejected_before_solver(batch,value):
    ns=load_renderer();sigma=torch.eye(2);sigma[0,0]=value
    if batch:sigma=sigma.repeat(1025,1,1)
    with patch.object(torch.linalg,'cholesky',side_effect=AssertionError('nonfinite reached solver')):
        with pytest.raises(ValueError,match='finite covariance'):caller(ns,batch)(torch.zeros(sigma.shape[:-1]),sigma)


@pytest.mark.parametrize('batch',[False,True])
@pytest.mark.parametrize('invalid',[
    [[0.,0.],[0.,1.]], [[1.,2.],[2.,1.]], [[1.,0.],[0.,0.]],
    [[1.,.2],[.3,1.]], [[float('nan'),0.],[0.,1.]],
    [[float('inf'),0.],[0.,1.]],
])
def test_invalid_conditioned_matrix_fails_actual_fallback(batch,invalid):
    ns=load_renderer();bad=torch.tensor(invalid,dtype=torch.float64)
    sigma=torch.eye(2,dtype=torch.float64)
    if batch:bad=bad.repeat(1025,1,1);sigma=sigma.repeat(1025,1,1)
    with patch.dict(ns,{'_condition_sigma_2d':lambda x:bad}),patch.object(torch.linalg,'cholesky',fail_cholesky):
        with pytest.raises(ValueError,match='conditioned'):caller(ns,batch)(torch.zeros(sigma.shape[:-1],dtype=sigma.dtype),sigma)


def test_conditioned_helper_shape_and_no_hidden_floor():
    ns=load_renderer();helper=ns['_covariance_cholesky_2x2']
    with pytest.raises(ValueError,match='2x2'):helper(torch.eye(3))
    # Below the old fallback's 1e-6 eigen floor; valid input is preserved.
    sigma=torch.tensor([[2e-9,-.6e-9],[-.6e-9,4e-9]],dtype=torch.float64)
    factor=helper(sigma)
    torch.testing.assert_close(factor@factor.T,sigma,atol=1e-23,rtol=1e-14)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_fallback_preserves_native_normal_representation(dtype):
    # Preserve the native convention per scope; do not claim this is the
    # conditioned covariance. The retained audit reports that unmet criterion.
    ns=load_renderer();sigma=rotated_spd(dtype);mu=torch.zeros(len(sigma),2,dtype=dtype)
    normal=caller(ns,True)(mu,sigma)
    with patch.object(torch.linalg,'cholesky',fail_cholesky):actual=caller(ns,True)(mu,sigma)
    tolerance=1e-6 if dtype==torch.float32 else 1e-10
    expected=native_covariance(normal)
    scale=expected.abs().amax(dim=(-2,-1)).clamp_min(1)
    assert ((native_covariance(actual)-expected).abs().amax(dim=(-2,-1))/scale).max()<=tolerance
