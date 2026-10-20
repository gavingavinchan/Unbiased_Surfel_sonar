"""Native-equation contract, independent of the Python diagnostic decoder.

Set PACKED_RENDERER_SOURCE/PACKED_FORWARD_SOURCE to reproduce old exact sources.
No CUDA imports or execution; the real callers are extracted from their AST.
"""
import ast
import math
import os
from pathlib import Path
from unittest.mock import patch
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'_bounded_sonar_linalg', '_condition_sigma_2d', '_covariance_cholesky_2x2',
         'sigma2d_to_transmat_precomp', '_sigma2d_to_transmat_precomp_batch',
         'transmat_precomp_to_sigma2d'}
DTYPES = [torch.float32, torch.float64]

def load_callers():
    path = Path(os.environ.get('PACKED_RENDERER_SOURCE', ROOT/'gaussian_renderer/__init__.py'))
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {n.name for n in nodes} == NAMES
    ns = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns

def fixtures(dtype):
    values = []
    for lo, hi in [(2,4),(.2,20),(.10001,.10003),(1e-10,3e-10),
                   (1e-8,4),(1,400),(400,800),(500,500.1)]:
        for theta in torch.linspace(-1.4,1.4,8,dtype=dtype):
            c,s = theta.cos(),theta.sin()
            r = torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
            values.append(r@torch.diag(torch.tensor([lo,hi],dtype=dtype))@r.T)
    return torch.stack(values)

def forced_failure(x):
    raise torch.linalg.LinAlgError('forced contract fixture')

def call(ns, mu, sigma, batch, fallback):
    fn = ns['_sigma2d_to_transmat_precomp_batch' if batch else 'sigma2d_to_transmat_precomp']
    if fallback:
        with patch.object(torch.linalg,'cholesky',forced_failure):
            return fn(mu,sigma)
    return fn(mu,sigma)

def cov_from_rows(h):
    a = h[..., :2, :2]
    return a@a.transpose(-1,-2)

def ray(h, pixels):
    # Independent native forward: intersections of the two homogeneous planes.
    tu,tv,tw = h.unbind(-2)
    k = pixels[...,0,None]*tw[...,None,:]-tu[...,None,:]
    l = pixels[...,1,None]*tw[...,None,:]-tv[...,None,:]
    p = torch.linalg.cross(k,l,dim=-1)
    s = p[...,:2]/p[...,2,None]
    return k,l,p,s

def aabb_center(h, cutoff=3.):
    t = h.new_tensor([cutoff*cutoff,cutoff*cutoff,-1.])
    tu,tv,tw = h.unbind(-2)
    f = t/(t*tw.square()).sum(-1,keepdim=True)
    return torch.stack(((f*tu*tw).sum(-1),(f*tv*tw).sum(-1)),-1)

def intensity_native(h, pixels, opacity=.7):
    s = ray(h,pixels)[3]
    center = aabb_center(h)
    rho3 = s.square().sum(-1)
    rho2 = 2*(pixels-center[...,None,:]).square().sum(-1)
    alpha = (opacity*torch.exp(-.5*torch.minimum(rho3,rho2))).clamp_max(.99)
    return torch.where(alpha>=1/255,alpha,torch.zeros_like(alpha))

def intensity_independent(cov, mean, pixels, opacity=.7):
    d = pixels-mean[...,None,:]
    q = torch.linalg.solve(cov,d.transpose(-1,-2)).transpose(-1,-2)
    rho = torch.minimum((d*q).sum(-1),2*d.square().sum(-1))
    alpha = (opacity*torch.exp(-.5*rho)).clamp_max(.99)
    return torch.where(alpha>=1/255,alpha,torch.zeros_like(alpha))

def native_backward(h, pixels, weights, opacity=.7):
    # Explicit source VJP for unsaturated additive color=1 contributions.
    k,l,p,s = ray(h,pixels)
    center = aabb_center(h)
    d = center[...,None,:]-pixels
    rho3 = s.square().sum(-1); rho2 = 2*d.square().sum(-1)
    alpha = opacity*torch.exp(-.5*torch.minimum(rho3,rho2))
    active = ((alpha>=1/255)&(alpha<.99)).to(h.dtype)
    on_plane = rho3<=rho2
    g = weights*alpha*active
    ds = -g[...,None]*s*on_plane[...,None]
    dp_xy = ds/p[...,2,None]
    dp_z = -(dp_xy*s).sum(-1,keepdim=True)
    dp = torch.cat((dp_xy,dp_z),-1)
    dk = torch.linalg.cross(l,dp,dim=-1)
    dl = torch.linalg.cross(dp,k,dim=-1)
    dtw = pixels[...,0,None]*dk+pixels[...,1,None]*dl
    direct = torch.stack((-dk.sum(-2),-dl.sum(-2),dtw.sum(-2)),-2)
    # Native compute_transmat_aabb propagates screen-center derivatives.
    dcenter = (-2*g[...,None]*d*(~on_plane)[...,None]).sum(-2)
    center_vjp = torch.autograd.grad(center,h,dcenter,retain_graph=True)[0]
    return direct+center_vjp

def offsets(dtype):
    # Exactly 128 fixed offsets, frozen independently of the producer.
    x,y = torch.meshgrid(torch.linspace(-3.6,3.6,16,dtype=dtype),
                         torch.linspace(-3.5,3.5,8,dtype=dtype),indexing='ij')
    return torch.stack((x.flatten(),y.flatten()),-1)

def affine_transform(dtype):
    return torch.tensor([[.8,-.35,3.2],[.4,1.1,-2.7],[0,0,1]],dtype=dtype)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('fallback',[False,True])
@pytest.mark.parametrize('count',[1,512,12001])
def test_batched_covariance_means_decoder_and_serial(dtype,fallback,count):
    ns=load_callers(); raw=fixtures(dtype)[torch.arange(count)%64]
    mu=torch.stack((torch.arange(count,dtype=dtype)*.003+12.17,
                    torch.arange(count,dtype=dtype)*-.002+34.23),-1)
    h=call(ns,mu,raw,True,fallback); target=ns['_condition_sigma_2d'](raw)
    tol=2e-6 if dtype==torch.float32 else 1e-12
    scale=target.abs().amax((-1,-2)).clamp_min(1)
    assert (((cov_from_rows(h)-target).abs().amax((-1,-2))/scale)<=tol).all()
    assert torch.isfinite(h).all()
    torch.testing.assert_close(h[:,:2,2],mu,atol=0,rtol=0)
    assert torch.equal(h[:,0,1],torch.zeros(count,dtype=dtype))
    for i in range(min(count,64)):
        serial=call(ns,mu[i],raw[i],False,fallback)
        torch.testing.assert_close(serial,h[i],atol=0,rtol=0)
        decoded=ns['transmat_precomp_to_sigma2d'](serial)
        torch.testing.assert_close(decoded,cov_from_rows(serial),atol=0,rtol=0)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('fallback',[False,True])
@pytest.mark.parametrize('transform',[False,True])
def test_independent_native_intensity(dtype,fallback,transform):
    ns=load_callers(); raw=fixtures(dtype); target=ns['_condition_sigma_2d'](raw)
    mu=torch.tensor([12.17,34.23],dtype=dtype).repeat(64,1)
    h=call(ns,mu,raw,True,fallback)
    if transform:
        b=affine_transform(dtype);h=b@h
        target=b[:2,:2]@target@b[:2,:2].T
        mu=(b[:2,:2]@mu.T).T+b[:2,2]
    pixels=mu[:,None,:]+offsets(dtype)[None]*torch.diagonal(target,dim1=-2,dim2=-1).sqrt()[:,None,:]
    actual=intensity_native(h,pixels); expected=intensity_independent(target,mu,pixels)
    torch.testing.assert_close(actual,expected,rtol=1e-5 if dtype==torch.float32 else 1e-10,
                               atol=2e-6 if dtype==torch.float32 else 1e-13)
    assert torch.isfinite(actual).all()


def gradient_row(dtype,batch,fallback,theta):
    ns=load_callers(); c=math.cos(theta);s=math.sin(theta)
    r=torch.tensor([[c,-s],[s,c]],dtype=dtype)
    sigma=r@torch.diag(torch.tensor([2.,5.],dtype=dtype))@r.T
    q=torch.tensor([sigma[0,0],sigma[0,1],sigma[1,1],.17,-.23],dtype=dtype,requires_grad=True)
    # Fixed absolute pixels: perturbing mean must not drag the observation grid.
    pixels=torch.tensor([.17,-.23],dtype=dtype)+offsets(dtype)*1.25
    weights=torch.linspace(.1,.9,128,dtype=dtype)
    b=affine_transform(dtype)
    pixels=(b[:2,:2]@pixels.T).T+b[:2,2]
    def pack(v):
        cov=torch.stack((torch.stack((v[0],v[1])),torch.stack((v[1],v[2]))))
        mean=v[3:]
        if batch: cov=cov[None];mean=mean[None]
        h=call(ns,mean,cov,batch,fallback).reshape(3,3)
        return b@h
    def loss(v):return (intensity_native(pack(v),pixels)*weights).sum()
    h=pack(q)
    actual=torch.autograd.grad(loss(q),q)[0]
    manual=torch.autograd.grad(h,q,native_backward(h,pixels,weights),retain_graph=True)[0]
    def reference_loss(v):
        cov=torch.stack((torch.stack((v[0],v[1])),torch.stack((v[1],v[2]))))
        cov=ns['_condition_sigma_2d'](cov)
        return (intensity_independent(b[:2,:2]@cov@b[:2,:2].T,b[:2,:2]@v[3:]+b[:2,2],pixels)*weights).sum()
    reference=torch.autograd.grad(reference_loss(q),q)[0]
    eps=.004 if dtype==torch.float32 else 1e-5
    finite=[]
    for j in range(5):
        step=torch.zeros_like(q);step[j]=eps
        finite.append((loss(q.detach()+step)-loss(q.detach()-step))/(2*eps))
    finite=torch.stack(finite)
    def rel(a,b):return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-20))
    return {'covariance_fd_relative':rel(actual[:3],finite[:3]),'mean_fd_relative':rel(actual[3:],finite[3:]),
            'native_backward_relative':rel(manual,actual),'independent_gradient_relative':rel(actual,reference)}

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('batch',[False,True])
@pytest.mark.parametrize('fallback',[False,True])
def test_covariance_mean_gradients(dtype,batch,fallback):
    rows=[gradient_row(dtype,batch,fallback,float(theta)) for theta in torch.linspace(-1.3,1.3,8)]
    tol=1e-3 if dtype==torch.float32 else 1e-6
    for row in rows:
        assert row['covariance_fd_relative']<=tol
        assert row['mean_fd_relative']<=tol
        assert row['native_backward_relative']<=(2e-6 if dtype==torch.float32 else 1e-12)
        assert row['independent_gradient_relative']<=(1e-5 if dtype==torch.float32 else 1e-10)


def footprint(h,opacity=1.):
    # Preserve GLM column semantics exactly, including old center computation.
    source=Path(os.environ.get('PACKED_FORWARD_SOURCE',ROOT/'submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu')).read_text()
    center=aabb_center(h)
    t=h.new_tensor([9.,9.,-1.]);tw=h[...,2,:]
    f=t/(t*tw.square()).sum(-1,keepdim=True)
    extent=(center.square()-(f[...,None,:]*h[...,:2,:].square()).sum(-1)).clamp_min(1e-4).sqrt()
    cutoff=3.
    if 'affine_support_cutoff' in source:
        cutoff=math.sqrt(max(2*math.log(max(255*opacity,1)),0))
        extent=cutoff*h[...,:2,:2].square().sum(-1).sqrt()
    radius=torch.maximum(extent.amax(-1),h.new_tensor(cutoff*.707106)).ceil()
    if 'affine_support_cutoff' in source: radius=radius+1
    # Native getRect (positive coordinates): integer truncation, BLOCK=16.
    rect_min=((center-radius[...,None])/16).to(torch.int64).clamp_min(0)
    rect_max=((center+radius[...,None]+15)/16).to(torch.int64).clamp_min(0)
    return center,extent,radius,rect_min,rect_max

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('fallback',[False,True])
def test_above_native_threshold_footprint_containment(dtype,fallback):
    ns=load_callers();raw=fixtures(dtype)
    # Sweep tile phase, opacity, lowpass-dominated and broad ellipses.
    angles=torch.linspace(0,2*math.pi,256,dtype=dtype)
    dirs=torch.stack((angles.cos(),angles.sin()),-1)
    missed=0;tested=0
    for opacity in [.01,.2,1.]:
        support=math.sqrt(2*math.log(255*opacity))
        for phase in [0.,.2,7.7,15.9]:
            mu=torch.tensor([96.+phase,112.+phase],dtype=dtype).repeat(64,1)
            h=call(ns,mu,raw,True,fallback)
            center,extent,radius,lo,hi=footprint(h,opacity)
            # Boundary annulus + isotropic-filter boundary + integer pixels.
            latent=dirs[None]*(support*.999)
            points=(h[:,:2,:2]@latent.transpose(-1,-2)).transpose(-1,-2)+mu[:,None,:]
            filterpoints=mu[:,None,:]+dirs[None]*(support/math.sqrt(2)*.999)
            samples=torch.cat((points,filterpoints,points.round(),filterpoints.round()),1)
            active=intensity_native(h,samples,opacity)>=1/255
            inside=((samples>=lo[:,None,:]*16)&(samples<hi[:,None,:]*16)).all(-1)
            missed+=int((active&~inside).sum());tested+=int(active.sum())
    assert missed==0, f'{missed} above-threshold misses / {tested} samples'


def test_native_sources_match_oracle_and_transport():
    native=ROOT/'submodules/diff-surfel-rasterization'
    forward=(native/'cuda_rasterizer/forward.cu').read_text()
    backward=(native/'cuda_rasterizer/backward.cu').read_text()
    for source in (forward,backward):
        for literal in ['float3 k = pix.x * Tw - Tu;', 'float3 l = pix.y * Tw - Tv;',
                        'float3 p = cross(k, l);','float2 s = {p.x / p.z, p.y / p.z};']:
            assert literal in source
        for j in range(9):assert f'9 * coll_id+{j}' in source
    for literal in ['cross(l, dL_dp)','cross(dL_dp, k)','dsx_pz * s.x + dsy_pz * s.y']:
        assert literal in backward
    assert 'transMat_precomp.contiguous().data<float>()' in (native/'rasterize_points.cu').read_text()
    assert 'transMat_precomp != nullptr ? transMat_precomp : geomState.transMat' in (native/'cuda_rasterizer/rasterizer_impl.cu').read_text()


@pytest.mark.parametrize('dtype',DTYPES)
def test_all_nine_native_backward_entries(dtype):
    ns=load_callers()
    h=call(ns,torch.tensor([.17,-.23],dtype=dtype),
           torch.tensor([[2.,.6],[.6,4.]],dtype=dtype),False,False)
    h=(affine_transform(dtype)@h).detach().requires_grad_()
    pixels=h[:2,2].detach()+offsets(dtype)*1.25
    weights=torch.linspace(.1,.9,128,dtype=dtype)
    expected=torch.autograd.grad((intensity_native(h,pixels)*weights).sum(),h,retain_graph=True)[0]
    actual=native_backward(h,pixels,weights)
    torch.testing.assert_close(actual,expected,atol=2e-6 if dtype==torch.float32 else 1e-12,
                               rtol=2e-6 if dtype==torch.float32 else 1e-12)


def test_affine_bounds_source_equations():
    source=Path(os.environ.get('PACKED_FORWARD_SOURCE',ROOT/'submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu')).read_text()
    for literal in ['T[2].x == 0.0f && T[2].y == 0.0f && T[2].z == 1.0f',
                    '2.0f * logf(max(255.0f * opacities[idx], 1.0f))',
                    'affine_support_cutoff * sqrtf(T[0].x*T[0].x + T[0].y*T[0].y)',
                    'affine_support_cutoff * sqrtf(T[1].x*T[1].x + T[1].y*T[1].y)',
                    'radius = 1.0f + ceil(max(max(extent.x, extent.y),',
                    'affine_support_cutoff * FilterSize)']:
        assert literal in source
