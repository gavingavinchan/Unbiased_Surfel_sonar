"""Opaque cell-average visibility, revision 371: filtered homogeneous predicates.

The stored decoded geometry is the material definition. Floating evaluation errors
trigger exact rational predicates, never a physical coincidence tolerance. See
SONAR_RAY_DEPTH_371.md. Visibility is discrete and has no boundary derivative.
"""
import math
from fractions import Fraction as F
import torch


def quaternion_tangents(q):
    q = torch.nn.functional.normalize(q.to(torch.float64), dim=-1)
    w, x, y, z = q.unbind(-1)
    u = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), -1)
    v = torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)
    return u, v


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


@torch.no_grad()
def ray_depth_visibility(centres, tangent_u, tangent_v, scales, reflectance):
    """16x16 midpoint mean. Exact predicates on ambiguous float64 evaluations.

    No scene identities or parent-plane labels. Each stored c/u/v/halfwidth is an
    exact binary number; cross(u,v) defines its plane without normalizing/dividing.
    Local tangent support is [-h0,h0) x [-h1,h1), with no support inflation.
    """
    dtype = centres.dtype
    c, u, v, s, rho = [a.detach().to(torch.float64) for a in
                       (centres, tangent_u, tangent_v, scales, reflectance)]
    n = torch.linalg.cross(u, v, dim=-1)
    half = math.sqrt(2*math.pi)*s/2
    count = len(c)
    if not count:
        return centres.new_zeros(0), {'samples': 0, 'invalid_source_samples': 0}
    if not bool(torch.isfinite(torch.cat((c,u,v,s),dim=-1)).all()) or bool((s<=0).any()):
        raise ValueError('ray_depth_layers requires finite positive-scale geometry')
    # Product-sum envelopes: |cross(u,v)| <= nab. p's envelope includes
    # source point construction. gamma_128 bounds the longest scalar expression
    # (<= 64 rounded operations plus envelope evaluation); no division by n.ray.
    eps = torch.finfo(torch.float64).eps
    gamma = 128*eps/(1-128*eps)
    nab = torch.stack((u[:,1].abs()*v[:,2].abs()+u[:,2].abs()*v[:,1].abs(),
                       u[:,2].abs()*v[:,0].abs()+u[:,0].abs()*v[:,2].abs(),
                       u[:,0].abs()*v[:,1].abs()+u[:,1].abs()*v[:,0].abs()), -1)
    grid = (torch.arange(16, device=c.device, dtype=torch.float64)+.5)/8-1
    gx, gy = torch.meshgrid(grid,grid,indexing='ij'); gx,gy=gx.flatten(),gy.flatten()
    du = (gx[None,:]*half[:,0,None])[:,:,None]*u[:,None,:]
    dv = (gy[None,:]*half[:,1,None])[:,:,None]*v[:,None,:]
    points = (c[:,None,:]+du+dv).reshape(-1,3)
    penv = (c[:,None,:].abs()+du.abs()+dv.abs()).reshape(-1,3)
    source = torch.arange(count,device=c.device).repeat_interleave(256)
    aa = (n*c).sum(-1)
    aenv = (nab*c.abs()).sum(-1)
    # Lazily transferred exact inputs. CPU fallback is bounded by one render;
    # it does not retain a CUDA allocation or detach any emission parameter path.
    exact = {}; coplanar = {}; exact_calls=0; coplanar_calls=0
    def cell(i):
        if i not in exact:
            cv,uv,vv,hv=[tuple(F(float(x)) for x in a[i].cpu().tolist()) for a in (c,u,v,half)]
            exact[i]=(cv,uv,vv,hv,_cross(uv,vv))
        return exact[i]
    def same_plane(i,j):
        nonlocal coplanar_calls
        key=(i,j)
        if key not in coplanar:
            ci,ui,vi,_,_=cell(i); cj,_,_,_,nj=cell(j)
            coplanar[key]=(_dot(nj,tuple(x-y for x,y in zip(ci,cj)))==0 and _dot(nj,ui)==0 and _dot(nj,vi)==0)
            coplanar_calls+=1
        return coplanar[key]
    def exact_hit(i,j,k):
        nonlocal exact_calls
        exact_calls+=1
        ci,ui,vi,hi,_=cell(i); cj,uj,vj,hj,nj=cell(j)
        ax=F(2*(k//16)+1,16)-1; ay=F(2*(k%16)+1,16)-1
        p=tuple(ci[l]+ax*hi[0]*ui[l]+ay*hi[1]*vi[l] for l in range(3))
        A=_dot(nj,cj); B=_dot(nj,p)
        if not B or A*B<=0:return False,False,False
        q=tuple(A*p[l]/B-cj[l] for l in range(3))
        hit=(-hj[0]<=_dot(q,uj)<hj[0] and -hj[1]<=_dot(q,vj)<hj[1])
        return hit,hit and A==B,hit and (B-A)*B>0
    weights = torch.zeros(len(points),device=c.device,dtype=torch.float64)
    invalid=0; hidden_count=0; tie_count=0; max_filter=0.
    for start in range(0,len(points),256):
        p=points[start:start+256]; pe=penv[start:start+256]; src=source[start:start+256]
        B=p@n.T; be=pe@nab.T
        D=(p[:,None,:]-c[None,:,:])
        # Direct residual avoids subtracting two almost equal plane offsets.
        nearval=(D*n[None,:,:]).sum(-1)
        de=gamma*(be+aenv[None,:])
        sg=torch.where(B>=0,1.,-1.)
        A=aa[None,:].expand_as(B); ae=aenv[None,:].expand_as(B)
        # Q = A (p-c) - (B-A)c, divided by B. Tangent coordinates
        # are tested after multiplying by positive |B|, avoiding grazing division.
        qnum=A[:,:,None]*D-nearval[:,:,None]*c[None,:,:]
        U=(qnum*u[None,:,:]).sum(-1)*sg
        V=(qnum*v[None,:,:]).sum(-1)*sg
        HB=half[None,:,:]*B.abs()[:,:,None]
        ue=gamma*(ae*(pe@u.abs().T)+(be+2*ae)*(c.abs()*u.abs()).sum(-1)[None,:]+half[None,:,0]*be)
        ve=gamma*(ae*(pe@v.abs().T)+(be+2*ae)*(c.abs()*v.abs()).sum(-1)[None,:]+half[None,:,1]*be)
        pos=A*sg; dep=nearval*sg
        cover=(B!=0)&(pos>0)&(U>=-HB[:,:,0])&(U<HB[:,:,0])&(V>=-HB[:,:,1])&(V<HB[:,:,1])&(rho[None,:]>0)
        # If any necessary inequality is certainly false, no exact work needed.
        possible=(pos>=-gamma*ae)&(U+HB[:,:,0]>=-ue)&(HB[:,:,0]-U>=-ue)&(V+HB[:,:,1]>=-ve)&(HB[:,:,1]-V>=-ve)&(rho[None,:]>0)
        uncertain_support=possible & ((B.abs()<=gamma*be)|(pos.abs()<=gamma*ae)|((U+HB[:,:,0]).abs()<=ue)|((HB[:,:,0]-U).abs()<=ue)|((V+HB[:,:,1]).abs()<=ve)|((HB[:,:,1]-V).abs()<=ve))
        near=cover&(dep>de); tie=torch.zeros_like(cover)
        ambiguous_depth=cover & (dep.abs()<=de)
        # Source identity only supplies its own constructed sample, not a semantic
        # equivalence class; all other candidates are tested geometrically.
        row=torch.arange(len(p),device=c.device)
        uncertain_support[row,src]=False; ambiguous_depth[row,src]=False
        for rr,jj in torch.nonzero(uncertain_support|ambiguous_depth,as_tuple=False).cpu().tolist():
            i=int(src[rr]); k=(start+rr)%256
            if not bool(uncertain_support[rr,jj]) and same_plane(i,jj):
                tie[rr,jj]=True; near[rr,jj]=False
            else:
                h,t,ne=exact_hit(i,jj,k);cover[rr,jj]=h;tie[rr,jj]=t;near[rr,jj]=ne
        cover[row,src]=rho[src]>0; tie[row,src]=rho[src]>0; near[row,src]=False
        # A source plane through the sonar has no positive-area ray-plane model.
        bad=(aa[src].abs()<=gamma*aenv[src])
        for rr in torch.nonzero(bad,as_tuple=False).flatten().cpu().tolist():
            ci,_,_,_,ni=cell(int(src[rr]));bad[rr]=_dot(ni,ci)==0
        mult=tie.sum(-1); hidden=near.any(-1);ok=(mult>0)&~bad
        invalid+=int((~ok&(rho[src]>0)).sum());hidden_count+=int((hidden&(rho[src]>0)).sum())
        tie_count+=int(((mult>1)&~hidden&(rho[src]>0)).sum())
        weights[start:start+len(p)]=torch.where(ok&~hidden,1/mult.clamp_min(1).double(),0.)
        max_filter=max(max_filter,float(de.max()))
    fraction=weights.reshape(count,256).mean(-1).to(dtype)
    return fraction,dict(samples=len(points),invalid_source_samples=invalid,hidden_source_samples=hidden_count,
        coincident_front_samples=tie_count,eta_m=0.,max_source_depth_tolerance_m=0.,
        max_residual_filter_m=max_filter,exact_point_predicates=exact_calls,exact_plane_pairs=coplanar_calls,
        support='half_open_opaque_area_equivalent_rectangle',quadrature='16x16_midpoint',
        numerical_rule='filtered_homogeneous_exact_371',visibility_boundary_gradient=False,
        ancillary_alpha_depth_available=False,training_validity=False)
