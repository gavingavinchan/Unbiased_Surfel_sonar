"""Conservative ray-support broad phase and batched 409 predicates (416).

No range filter. Every possible blocker, including arbitrarily distant foreground
ranges, survives the angular broad phase. No geometry cache survives a call.
Uncertain predicates use exactly the 371 Fraction decision and 409 coincidence.
"""
import math
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction as F
from pathlib import Path
import torch
from .ray_depth_oracle_409 import _dot, _cross

_guard = ContextVar('visibility_guard', default=None)


class VisibilityInterrupted(RuntimeError):
    """Caller should atomically checkpoint its last complete update and exit."""


@contextmanager
def visibility_execution(guard):
    """Install a per-call execution guard; checked between bounded chunks."""
    token = _guard.set(guard)
    try:
        yield
    finally:
        _guard.reset(token)


def checkpoint_guard():
    callback = _guard.get()
    if callback is not None:
        callback()
    stop = os.environ.get('SONAR_VISIBILITY_STOP_FILE')
    if stop and Path(stop).exists():
        raise VisibilityInterrupted('project-local visibility stop file')
    # No signalling or suspension of unrelated processes. The worker must exit.
    if Path('/proc/meminfo').exists():
        available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        if available < 4*1024**3:
            raise VisibilityInterrupted('available host memory below 4 GiB')
    if Path('/proc/pressure/memory').exists():
        psi = float(Path('/proc/pressure/memory').read_text().split('avg10=')[1].split()[0])
        if psi > 1:
            raise VisibilityInterrupted('host memory pressure')


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def angular_support(c, u, v, half):
    """Outward ball bounds for source samples AND blocker tangent support.

    For blocker support, dual tangents solve u.q=a, v.q=b, n.q=0.
    All four corners fit in the ball with squared radius
    h0²|dual_u|²+h1²|dual_v|²+2h0 h1 |dual_u.dual_v|.
    The source's constructed u/v rectangle is enclosed too. A ball excluding the
    origin subtends a spherical cap; its chord radius bounds every ray direction.
    Degenerate / origin-containing / poorly conditioned cases use the whole sphere.
    The 409 filter's uncertainty tube is enclosed by a 4096-eps64 scale envelope;
    at doubtful conditioning we reject nothing. See SONAR_VISIBILITY_416.md.
    """
    n = torch.linalg.cross(u, v, dim=-1)
    n2 = (n*n).sum(-1)
    dual_u = torch.linalg.cross(v, n, dim=-1)/n2[:, None].clamp_min(1e-300)
    dual_v = torch.linalg.cross(n, u, dim=-1)/n2[:, None].clamp_min(1e-300)
    def radius(a, b):
        return (half[:, 0]**2*(a*a).sum(-1) + half[:, 1]**2*(b*b).sum(-1)
                + 2*half.prod(-1)*(a*b).sum(-1).abs()).clamp_min(0).sqrt()
    rad = torch.maximum(radius(u, v), radius(dual_u, dual_v))
    length = c.norm(dim=-1)
    conditioning = (u.norm(dim=-1)*v.norm(dim=-1))/n2.sqrt().clamp_min(1e-300)
    envelope = 4096*torch.finfo(torch.float64).eps*(c.abs().sum(-1)+rad+1)*conditioning.clamp_min(1)**2
    rad = rad+envelope
    unsafe = (rad >= length) | (conditioning > 1e4) | ~torch.isfinite(rad+length+conditioning) | (n2 < 1e-200)
    ratio = (rad/length.clamp_min(1e-300)).clamp(0, 1)
    # Stable chord radius of alpha=asin(r/d), avoiding cancellation near zero.
    chord = math.sqrt(2)*ratio/(1+(1-ratio*ratio).clamp_min(0).sqrt()).sqrt()
    chord = torch.where(unsafe, torch.full_like(chord, 2), chord)
    direction = c/length[:, None].clamp_min(1e-300)
    return direction, chord, unsafe


@torch.no_grad()
def candidate_pairs(c, u, v, half, rho, chunk=256):
    """Sparse directed (source,blocker) list; rebuilt from current geometry.

    Triangle inequality for unit ray directions is a necessary angular overlap
    test. An outward 1e-11 dot margin covers the normalized dot evaluation and
    the cap arithmetic after the radius envelope. This filter never uses range.
    """
    direction, chord, unsafe = angular_support(c, u, v, half)
    parts=[]
    count=len(c)
    for lo in range(0, count, chunk):
        checkpoint_guard()
        hi=min(lo+chunk,count)
        dots=direction[lo:hi]@direction.T
        limit=1-.5*(chord[lo:hi,None]+chord[None,:])**2
        keep=(dots >= limit-1e-11) | unsafe[lo:hi,None] | unsafe[None,:]
        keep &= rho.reshape(1,-1)>0
        ij=keep.nonzero()
        ij[:,0]+=lo
        parts.append(ij)
    return (torch.cat(parts) if parts else torch.empty((0,2),device=c.device,dtype=torch.long)),int(unsafe.sum())


def _predicates(cj, uj, vj, nj, nabj, hj, aa, ae, p, pe, rho, gamma):
    """Elementwise batch of the original homogeneous expressions, no division."""
    B=(p*nj[:,None,:]).sum(-1)
    be=(pe*nabj[:,None,:]).sum(-1)
    D=p-cj[:,None,:]
    nearval=(D*nj[:,None,:]).sum(-1)
    de=gamma*(be+ae[:,None])
    sg=torch.where(B>=0,1.,-1.)
    A=aa[:,None]
    qnum=A[:,:,None]*D-nearval[:,:,None]*cj[:,None,:]
    U=(qnum*uj[:,None,:]).sum(-1)*sg
    V=(qnum*vj[:,None,:]).sum(-1)*sg
    HB=hj[:,None,:]*B.abs()[:,:,None]
    ue=gamma*(ae[:,None]*(pe*uj[:,None,:].abs()).sum(-1)+(be+2*ae[:,None])*(cj.abs()*uj.abs()).sum(-1)[:,None]+hj[:,0,None]*be)
    ve=gamma*(ae[:,None]*(pe*vj[:,None,:].abs()).sum(-1)+(be+2*ae[:,None])*(cj.abs()*vj.abs()).sum(-1)[:,None]+hj[:,1,None]*be)
    pos=A*sg;dep=nearval*sg
    cover=(B!=0)&(pos>0)&(U>=-HB[:,:,0])&(U<HB[:,:,0])&(V>=-HB[:,:,1])&(V<HB[:,:,1])&(rho[:,None]>0)
    possible=(pos>=-gamma*ae[:,None])&(U+HB[:,:,0]>=-ue)&(HB[:,:,0]-U>=-ue)&(V+HB[:,:,1]>=-ve)&(HB[:,:,1]-V>=-ve)&(rho[:,None]>0)
    uncertain=possible&((B.abs()<=gamma*be)|(pos.abs()<=gamma*ae[:,None])|((U+HB[:,:,0]).abs()<=ue)|((HB[:,:,0]-U).abs()<=ue)|((V+HB[:,:,1]).abs()<=ve)|((HB[:,:,1]-V).abs()<=ve))
    return cover,cover&(dep>de),uncertain,cover&(dep.abs()<=de),de.max()

_compiled=None

def _kernel():
    global _compiled
    mode=os.environ.get('SONAR_VISIBILITY_KERNEL','torch')
    if mode == 'torch':return _predicates
    if mode == 'compiled':
        if _compiled is None:
            _compiled=torch.compile(_predicates,fullgraph=True,dynamic=True)
        return _compiled
    raise ValueError('SONAR_VISIBILITY_KERNEL must be torch or compiled')


@torch.no_grad()
def ray_depth_visibility(centres,tangent_u,tangent_v,scales,reflectance,representation_error=None,out_dtype=None):
    t0=time.monotonic();checkpoint_guard()
    dtype=centres.dtype if out_dtype is None else out_dtype
    c,u,v,s,rho=[a.detach().to(torch.float64) for a in (centres,tangent_u,tangent_v,scales,reflectance)]
    rho=rho.reshape(-1)
    count=len(c)
    if not count:return centres.new_zeros(0),{'samples':0,'invalid_source_samples':0}
    if not bool(torch.isfinite(torch.cat((c,u,v,s),dim=-1)).all()) or bool((s<=0).any()):
        raise ValueError('ray_depth_layers requires finite positive-scale geometry')
    n=torch.linalg.cross(u,v,dim=-1);half=math.sqrt(2*math.pi)*s/2
    eps=torch.finfo(torch.float64).eps;gamma=128*eps/(1-128*eps)
    nab=torch.stack((u[:,1].abs()*v[:,2].abs()+u[:,2].abs()*v[:,1].abs(),u[:,2].abs()*v[:,0].abs()+u[:,0].abs()*v[:,2].abs(),u[:,0].abs()*v[:,1].abs()+u[:,1].abs()*v[:,0].abs()),-1)
    grid=(torch.arange(16,device=c.device,dtype=torch.float64)+.5)/8-1
    gx,gy=torch.meshgrid(grid,grid,indexing='ij');gx,gy=gx.flatten(),gy.flatten()
    du=(gx[None,:]*half[:,0,None])[:,:,None]*u[:,None,:]
    dv=(gy[None,:]*half[:,1,None])[:,:,None]*v[:,None,:]
    points=c[:,None,:]+du+dv;penv=c[:,None,:].abs()+du.abs()+dv.abs()
    aa=(n*c).sum(-1);aenv=(nab*c.abs()).sum(-1)
    pair_chunk=int(os.environ.get('SONAR_VISIBILITY_PAIR_CHUNK','2048'))
    if not 1<=pair_chunk<=8192:raise ValueError('pair chunk must be 1..8192')
    pairs,unsafe=candidate_pairs(c,u,v,half,rho)
    _sync(c.device);construction_s=time.monotonic()-t0;t_eval=time.monotonic()
    hidden=torch.zeros((count,256),device=c.device,dtype=torch.int32)
    mult=torch.zeros_like(hidden)
    exact={};coplanar={};exact_calls=0;coplanar_calls=0
    def cell(i):
        if i not in exact:
            cv,uv,vv,hv=[tuple(F(float(x)) for x in a[i].cpu().tolist()) for a in (c,u,v,half)]
            exact[i]=(cv,uv,vv,hv,_cross(uv,vv))
        return exact[i]
    def same_plane(i,j):
        nonlocal coplanar_calls
        key=(i,j)
        if key not in coplanar:
            ci,ui,vi,_,_=cell(i);cj,_,_,_,nj=cell(j)
            coplanar[key]=(_dot(nj,tuple(x-y for x,y in zip(ci,cj)))==0 and _dot(nj,ui)==0 and _dot(nj,vi)==0)
            coplanar_calls+=1
        return coplanar[key]
    def exact_hit(i,j,k):
        nonlocal exact_calls
        if exact_calls%64==0:checkpoint_guard()
        exact_calls+=1
        ci,ui,vi,hi,_=cell(i);cj,uj,vj,hj,nj=cell(j)
        ax=F(2*(k//16)+1,16)-1;ay=F(2*(k%16)+1,16)-1
        p=tuple(ci[l]+ax*hi[0]*ui[l]+ay*hi[1]*vi[l] for l in range(3))
        A=_dot(nj,cj);B=_dot(nj,p)
        if not B or A*B<=0:return False,False,False
        q=tuple(A*p[l]/B-cj[l] for l in range(3))
        hit=(-hj[0]<=_dot(q,uj)<hj[0] and -hj[1]<=_dot(q,vj)<hj[1])
        return hit,hit and A==B,hit and (B-A)*B>0
    n_unit=n/n.norm(dim=-1,keepdim=True).clamp_min(1e-300)
    rep=None if representation_error is None else [torch.as_tensor(a,device=c.device,dtype=torch.float64).reshape(-1) for a in representation_error]
    rep_pairs=0;rep_inexact=0;max_rep_bound=0.;max_filter=0.;max_chunk_s=0.;sample_ids=torch.arange(256,device=c.device)
    predicate=_kernel()
    for lo in range(0,len(pairs),pair_chunk):
        checkpoint_guard();tc=time.monotonic()
        ii,jj=pairs[lo:lo+pair_chunk].unbind(-1)
        cover,near,uncertain,ambiguous,de=predicate(c[jj],u[jj],v[jj],n[jj],nab[jj],half[jj],aa[jj],aenv[jj],points[ii],penv[ii],rho[jj],gamma)
        self_pair=ii==jj;uncertain[self_pair]=False;ambiguous[self_pair]=False
        same=torch.zeros_like(ii,dtype=torch.bool)
        if rep is not None:
            d=c[jj]-c[ii];ep=rep[1][jj]+rep[1][ii]
            par=torch.linalg.cross(n_unit[jj],n_unit[ii],dim=-1).norm(dim=-1)
            lever=d.norm(dim=-1)+torch.maximum(half[jj].norm(dim=-1),half[ii].norm(dim=-1))
            bound=rep[0][jj]+rep[0][ii]+ep*lever
            bound=bound*(1+1e-9)+64*eps*(c[jj].abs().sum(-1)+c[ii].abs().sum(-1))
            offs=(n_unit[jj]*d).sum(-1).abs()
            same=(par<=ep*(1+1e-9)+1e-15)&(offs<=bound)&((n_unit[ii]*d).sum(-1).abs()<=bound)&~self_pair
            rep_pairs+=int(same.sum());rep_inexact+=int(((offs>0)&same).sum())
            if bool(same.any()):max_rep_bound=max(max_rep_bound,float(bound[same].max()))
        tie=cover&same[:,None];near &= ~same[:,None];ambiguous &= ~same[:,None]
        # Resolve every ambiguity with the unchanged stored-input rational predicates.
        for rr,kk in (uncertain&same[:,None]).nonzero().cpu().tolist():
            h,_,_=exact_hit(int(ii[rr]),int(jj[rr]),kk);cover[rr,kk]=h;tie[rr,kk]=h
        uncertain &= ~same[:,None]
        for rr,kk in (uncertain|ambiguous).nonzero().cpu().tolist():
            i,j=int(ii[rr]),int(jj[rr])
            if not bool(uncertain[rr,kk]) and same_plane(i,j):tie[rr,kk]=True;near[rr,kk]=False
            else:
                h,t,ne=exact_hit(i,j,kk);cover[rr,kk]=h;tie[rr,kk]=t;near[rr,kk]=ne
        tie[self_pair]=rho[ii[self_pair],None]>0;near[self_pair]=False
        hidden.index_add_(0,ii,near.to(torch.int32));mult.index_add_(0,ii,tie.to(torch.int32))
        max_filter=max(max_filter,float(de));_sync(c.device);max_chunk_s=max(max_chunk_s,time.monotonic()-tc)
    bad=aa.abs()<=gamma*aenv
    for i in bad.nonzero().flatten().cpu().tolist():
        ci,_,_,_,ni=cell(i);bad[i]=_dot(ni,ci)==0
    ok=(mult>0)&~bad[:,None];hidden=hidden>0
    weights=torch.where(ok&~hidden,1/mult.clamp_min(1).double(),0.)
    fraction=weights.mean(-1).to(dtype)
    _sync(c.device)
    return fraction,dict(samples=count*256,invalid_source_samples=int((~ok&(rho[:,None]>0)).sum()),hidden_source_samples=int((hidden&(rho[:,None]>0)).sum()),
        coincident_front_samples=int(((mult>1)&~hidden&(rho[:,None]>0)).sum()),eta_m=0.,max_source_depth_tolerance_m=0.,max_residual_filter_m=max_filter,
        exact_point_predicates=exact_calls,exact_plane_pairs=coplanar_calls,support='half_open_opaque_area_equivalent_rectangle',quadrature='16x16_midpoint',
        numerical_rule='filtered_homogeneous_exact_371' if rep is None else 'filtered_homogeneous_exact_371+representation_coincidence_409',
        representation_coincident_pairs=rep_pairs,representation_inexact_coincident_pairs=rep_inexact,max_representation_bound_m=max_rep_bound,
        min_ordered_parallel_gap_m=None,visibility_boundary_gradient=False,ancillary_alpha_depth_available=False,training_validity=False,
        acceleration='conservative_angular_416',candidate_pairs=len(pairs),possible_pairs=count*count,rejected_pairs=count*count-len(pairs),
        unsafe_supports=unsafe,candidate_construction_s=construction_s,blocker_evaluation_s=time.monotonic()-t_eval,largest_pair_chunk_s=max_chunk_s,pair_chunk=pair_chunk,
        diagnostic_scope='coincidence diagnostics count retained angular candidates; min gap not evaluated')
