"""Finite-footprint 2D Gaussian ray/plane sonar reference, CPU or CUDA.

No observed image enters visibility. Sorted transparency is per azimuth SUBRAY
AND elevation, prior to elevation integration. This is incoherent intensity:
no multipath, calibrated beam response, material scattering, or normal evidence.
It is separate from the PR6 compiled image-space covariance code, retained intact.
"""
import math
import torch
import torch.nn.functional as F


def rotations(q):
    w,x,y,z=F.normalize(q,dim=-1).unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)


def render(xyz, quat, scale, opacity, color, pose, *, elevations=21, azimuth_subsamples=2,
           range_sigma=.65, gain=24., attenuation=.12, background=None, return_aux=False):
    """Native 200x256 bins, midpoint angular integration, normalized range PSF.

    The conservative angular broadphase encloses each finite 3sigma disc. All
    intersections inside its ellipse are retained, including farther true echoes.
    Two-sided |n dot ray| is an explicit surrogate, not measured acoustic BRDF.
    """
    dev=xyz.device;dtype=xyz.dtype;H,W=200,256;N=len(xyz)
    R=rotations(quat);local=(xyz-pose[:3,3])@pose[:3,:3];ran=local.norm(dim=-1)
    # Enclosing sphere of ellipse, plus the complete subpixel angular extent.
    angular=torch.asin((3*scale.max(-1).values/ran.clamp_min(.001)).clamp(max=1))
    mean=torch.atan2(local[:,1],local[:,0]);columns=torch.arange(W,device=dev)
    theta0=math.pi/3-(columns.to(dtype)+.5)*2*math.pi/3/W
    # Elevation is integrated over a top-hat +/-10deg aperture. Ray spacing
    # convergence is unproven for thin/small discs; fixture is not that proof.
    ii,cc=torch.where((theta0[None]-mean[:,None]).abs() <= angular[:,None]+2*math.pi/3/W)
    if len(ii)==0:
        out=xyz.sum()*0+xyz.new_zeros(H,W)
        if background is not None:out=out+background[:,None]
        return (out,{'intersections':0}) if return_aux else out
    ne=elevations;na=azimuth_subsamples;K=ne*na
    i=ii.repeat_interleave(K);c=cc.repeat_interleave(K)
    kk=torch.arange(K,device=dev).repeat(len(ii));e=kk%ne;sub=kk//ne
    phi=(-10+(e.to(dtype)+.5)*20/ne)*math.pi/180
    theta=math.pi/3-(c.to(dtype)+(sub.to(dtype)+.5)/na)*2*math.pi/3/W
    dlocal=torch.stack([phi.cos()*theta.cos(),phi.cos()*theta.sin(),phi.sin()],-1)
    direction=dlocal@pose[:3,:3].T
    norm=R[i,:,2];den=(norm*direction).sum(-1);num=(norm*(xyz[i]-pose[:3,3])).sum(-1)
    safe=torch.where(den.abs()>1e-5,den,torch.ones_like(den));t=num/safe
    delta=pose[:3,3]+t[:,None]*direction-xyz[i]
    u=(delta*R[i,:,0]).sum(-1)/scale[i,0];v=(delta*R[i,:,1]).sum(-1)/scale[i,1]
    power=u.square()+v.square()
    good=(den.abs()>1e-5)&(t>=.2)&(t<3)&(power<=9)
    i=i[good];c=c[good];e=e[good];sub=sub[good];t=t[good];power=power[good];den=den[good];phi=phi[good]
    ray=((c*na+sub)*ne+e)
    by_range=torch.argsort(t,stable=True)
    order=by_range[torch.argsort(ray[by_range],stable=True)]
    ray=ray[order];i=i[order];t=t[order];c=c[order];phi=phi[order];den=den[order];power=power[order]
    alpha=(opacity.reshape(-1)[i]*torch.exp(-.5*power)).clamp(max=.999)
    logT=torch.log1p(-alpha).to(torch.float64);prefix=torch.cumsum(logT,0)-logT
    first=torch.cat([torch.ones_like(ray[:1],dtype=torch.bool),ray[1:]!=ray[:-1]])
    starts=torch.where(first,torch.arange(len(ray),device=dev),torch.zeros_like(ray));starts=torch.cummax(starts,0).values
    trans=torch.exp(prefix-prefix[starts]).to(dtype)
    # Solid angle weights normalized over vertical samples, then average subrays.
    elev=(-10+(torch.arange(ne,device=dev,dtype=dtype)+.5)*20/ne)*math.pi/180
    weights=phi.cos()/elev.cos().sum()/na
    val=gain*trans*alpha*color.reshape(-1)[i]*den.abs()*torch.exp(-2*attenuation*t)*weights
    row=t/.015-.5;base=torch.floor(row).long();offset=torch.arange(-3,4,device=dev)
    rows=base[:,None]+offset[None];psf=torch.exp(-.5*((rows.to(dtype)-row[:,None])/range_sigma).square());psf=psf/psf.sum(-1,keepdim=True)
    valid=(rows>=0)&(rows<H);idx=(rows.clamp(0,H-1)*W+c[:,None]).flatten()
    out=xyz.new_zeros(H*W).scatter_add(0,idx,(psf*val[:,None]*valid).flatten()).reshape(H,W)
    # Approximate documented 1.5degree FWHM beam PSF, 3sigma finite support.
    x=torch.arange(-4,5,device=dev,dtype=dtype);kernel=torch.exp(-.5*(x/1.359).square());kernel=kernel/kernel.sum()
    out=F.conv2d(out[None,None],kernel[None,None,None,:],padding=(0,4))[0,0]
    out=out+xyz.sum()*0  # valid differentiable empty intersection result
    if background is not None:out=out+background[:,None]
    if return_aux:return out,{'intersections':len(i),'distinct_rays':len(torch.unique(ray))}
    return out
