"""Opaque area-equivalent cells, surface-sampled per-ray front layers.

Experimental forward-only visibility. See experiment 336 MODEL.md. Gaussian
receiver footprints and event emission are not changed by this module.
"""
import math
import torch


def quaternion_tangents(q):
    q = torch.nn.functional.normalize(q.to(torch.float64), dim=-1)
    w, x, y, z = q.unbind(-1)
    u = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), -1)
    v = torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)
    return u, v


def ray_depth_visibility(centres, tangent_u, tangent_v, scales, reflectance):
    """Return patch fractions and diagnostics; no IDs, meshes or plane groups.

    Coordinates are sonar-local metres, u/v are unit tangent directions, scales
    are Gaussian standard deviations in metres. rho>0 is opaque; rho=0 is absent.
    A fixed 16x16 cell quadrature is deliberately not a tunable argument.
    """
    dtype = centres.dtype
    c, u, v, s, rho = [a.to(torch.float64) for a in
                       (centres, tangent_u, tangent_v, scales, reflectance)]
    n = torch.linalg.cross(u, v, dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)
    half = math.sqrt(2*math.pi)*s/2
    count = len(c)
    if count == 0:
        return centres.new_zeros(0), {'samples': 0, 'invalid_source_samples': 0,
                                    'eta_m': 0., 'max_source_depth_tolerance_m': 0.}
    if not bool(torch.isfinite(torch.cat((c,u,v,s),dim=-1)).all()) or bool((s<=0).any()):
        raise ValueError('ray_depth_layers requires finite positive-scale geometry')
    length = max(1., float((torch.linalg.vector_norm(c,dim=-1)+torch.linalg.vector_norm(half,dim=-1)).max()))
    eta = 32*torch.finfo(torch.float32).eps*length
    parallel = 64*torch.finfo(torch.float64).eps
    grid = (torch.arange(16, device=c.device, dtype=torch.float64)+.5)/8-1
    gx, gy = torch.meshgrid(grid,grid,indexing='ij')
    points = (c[:,None,:]+(gx.flatten()[None,:]*half[:,0,None])[:,:,None]*u[:,None,:]
              +(gy.flatten()[None,:]*half[:,1,None])[:,:,None]*v[:,None,:]).reshape(-1,3)
    source = torch.arange(count,device=c.device).repeat_interleave(256)
    numerator = (n*c).sum(-1)
    cu, cv = (c*u).sum(-1), (c*v).sum(-1)
    weights = torch.zeros(len(points),device=c.device,dtype=torch.float64)
    invalid = 0; max_tau = 0.; hidden_count = 0; tie_count = 0
    # Bounded temporary arrays; the caller checkpoints after every render.
    for start in range(0,len(points),512):
        p = points[start:start+512]; src = source[start:start+512]
        distance = torch.linalg.vector_norm(p,dim=-1)
        d = p/distance.clamp_min(parallel)[:,None]
        denom = d@n.T
        valid_den = denom.abs()>parallel
        safe_den = torch.where(valid_den,denom,torch.ones_like(denom))
        depth = numerator[None,:]/safe_den
        tau = eta/safe_den.abs()
        src_den = (d*n[src]).sum(-1).abs()
        src_tau = eta/src_den.clamp_min(parallel)
        max_tau = max(max_tau,float(src_tau.max()))
        hu = depth*(d@u.T)-cu[None,:]
        hv = depth*(d@v.T)-cv[None,:]
        cover = (valid_den & (depth>0) & (rho[None,:]>0)
                 & (hu.abs()<=half[None,:,0]+eta)
                 & (hv.abs()<=half[None,:,1]+eta))
        delta = depth-distance[:,None]
        tie = cover & (delta.abs()<=tau+src_tau[:,None])
        nearer = (cover & (delta < -(tau+src_tau[:,None]))).any(-1)
        multiplicity = tie.sum(-1)
        ok = (src_den>parallel) & (distance>parallel) & (multiplicity>0)
        invalid += int(((~ok)&(rho[src]>0)).sum())
        hidden_count += int((nearer & (rho[src]>0)).sum())
        tie_count += int(((multiplicity>1)&(~nearer)&(rho[src]>0)).sum())
        weights[start:start+len(p)] = torch.where(ok & ~nearer,
            1/multiplicity.clamp_min(1).to(torch.float64),torch.zeros_like(distance))
    fraction = weights.reshape(count,256).mean(-1).to(dtype)
    return fraction, dict(samples=len(points),invalid_source_samples=invalid,
        hidden_source_samples=hidden_count,coincident_front_samples=tie_count,
        eta_m=eta,max_source_depth_tolerance_m=max_tau,
        support='opaque_area_equivalent_rectangle',quadrature='16x16_midpoint',
        ancillary_alpha_depth_available=False,training_validity=False)
