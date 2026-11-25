"""One conservative whole-rectangle enclosure of the original 416 trace.
See RECTANGLE_INTERVAL_511.md. Eager float64 only, no compiler or fused ops.
"""
import torch

class Context:
    def __init__(self, ref):
        self.bad = torch.zeros(ref.shape[0], dtype=torch.bool, device=ref.device)
        self.ref = ref
    def mark(self, x):
        t = (~torch.isfinite(x)) | ((x != 0) & (x.abs() < torch.finfo(torch.float64).tiny))
        self.bad |= t.reshape(len(self.bad), -1).any(-1)
    def box(self, x):
        if isinstance(x, Box): return x
        x = torch.as_tensor(x, dtype=torch.float64, device=self.ref.device)
        return Box(self, x, x)
    def rounded(self, a, b, op):
        y = {'add':torch.add, 'sub':torch.sub, 'mul':torch.mul}[op](a,b)
        # Broadcast scalar intermediates to the batch before checking.
        for z in (a,b,y):
            if z.ndim: self.mark(z)
        if op == 'mul':
            exactzero = (a == 0) | (b == 0)
            under = (y == 0) & ~exactzero
            self.bad |= under.reshape(len(self.bad),-1).any(-1)
        else:
            exactzero = (a == (-b if op == 'add' else b))
        lo = torch.nextafter(y, torch.full_like(y, -torch.inf))
        hi = torch.nextafter(y, torch.full_like(y, torch.inf))
        # These zero identities are exact for all IEEE inputs, including signed 0.
        return torch.where(exactzero, y, lo), torch.where(exactzero, y, hi)

class Box:
    def __init__(self, ctx, lo, hi): self.ctx,self.lo,self.hi=ctx,lo,hi
    def __getitem__(self,k): return Box(self.ctx,self.lo[k],self.hi[k])
    def __neg__(self): return Box(self.ctx,-self.hi,-self.lo)
    def __add__(self,b):
        b=self.ctx.box(b);l,_=self.ctx.rounded(self.lo,b.lo,'add');_,h=self.ctx.rounded(self.hi,b.hi,'add');return Box(self.ctx,l,h)
    __radd__=__add__
    def __sub__(self,b):
        b=self.ctx.box(b);l,_=self.ctx.rounded(self.lo,b.hi,'sub');_,h=self.ctx.rounded(self.hi,b.lo,'sub');return Box(self.ctx,l,h)
    def __rsub__(self,b): return self.ctx.box(b).__sub__(self)
    def __mul__(self,b):
        b=self.ctx.box(b);r=[self.ctx.rounded(x,y,'mul') for x,y in ((self.lo,b.lo),(self.lo,b.hi),(self.hi,b.lo),(self.hi,b.hi))]
        return Box(self.ctx,torch.stack([z[0] for z in r]).amin(0),torch.stack([z[1] for z in r]).amax(0))
    __rmul__=__mul__
    def abs(self):
        lo=torch.where((self.lo<=0)&(self.hi>=0),torch.zeros_like(self.lo),torch.minimum(self.lo.abs(),self.hi.abs()))
        return Box(self.ctx,lo,torch.maximum(self.lo.abs(),self.hi.abs()))
    def sum3(self):
        # All three choices of paired terms. IEEE addition is commutative;
        # these enclose all 12 ordered association/permutation trees.
        a,b,c=[self[...,k] for k in range(3)]
        r=[(a+b)+c,(a+c)+b,(b+c)+a]
        return Box(self.ctx,torch.stack([z.lo for z in r]).amin(0),torch.stack([z.hi for z in r]).amax(0))

@torch.no_grad()
def enclose(c,u,v,n,nab,half,aa,ae,rho,pairs,gamma):
    i,j=pairs.unbind(-1);ctx=Context(c[i]);K=ctx.box
    for x in (c[i],u[i],v[i],half[i],c[j],u[j],v[j],n[j],nab[j],half[j],aa[j],ae[j],rho[j]):ctx.mark(x)
    ci,ui,vi,hi=[K(x[i]) for x in (c,u,v,half)]
    cj,uj,vj,nj,nabj,hj,A,AE=[K(x[j]) for x in (c,u,v,n,nab,half,aa,ae)]
    ab=Box(ctx,-torch.ones_like(hi.lo[:,0]),torch.ones_like(hi.lo[:,0]))
    du=(ab*hi[:,0])[:,None]*ui;dv=(ab*hi[:,1])[:,None]*vi
    p=(ci+du)+dv;pe=(ci.abs()+du.abs())+dv.abs()
    B=(p*nj).sum3();be=(pe*nabj).sum3();D=p-cj
    nearval=(D*nj).sum3();de=gamma*(be+AE)
    signok=(B.lo>0)|(B.hi<0)
    sg=K(torch.where(B.lo>0,1.,-1.).double())
    qnum=A[:,None]*D-nearval[:,None]*cj
    U=(qnum*uj).sum3()*sg;V=(qnum*vj).sum3()*sg;HB=hj*B.abs()[:,None]
    ue=gamma*((AE*(pe*uj.abs()).sum3()+(be+2*AE)*(cj.abs()*uj.abs()).sum3())+hj[:,0]*be)
    ve=gamma*((AE*(pe*vj.abs()).sum3()+(be+2*AE)*(cj.abs()*vj.abs()).sum3())+hj[:,1]*be)
    pos=A*sg;dep=nearval*sg
    nh0=-HB[:,0];nh1=-HB[:,1];negae=-(gamma*AE)
    cover=torch.stack([pos.hi<=0,U.hi<nh0.lo,U.lo>=HB.hi[:,0],V.hi<nh1.lo,V.lo>=HB.hi[:,1]],-1)
    uh=U+HB[:,0];hu=HB[:,0]-U;vh=V+HB[:,1];hv=HB[:,1]-V
    possible=torch.stack([pos.hi<negae.lo,uh.hi<(-ue).lo,hu.hi<(-ue).lo,vh.hi<(-ve).lo,hv.hi<(-ve).lo],-1)
    valid=(~ctx.bad)&signok&(i!=j)
    reject=valid&cover.any(-1)&possible.any(-1)
    reasons=(cover.to(torch.int16)*(2**torch.arange(5,device=c.device))).sum(-1)+(possible.to(torch.int16)*(2**torch.arange(5,10,device=c.device))).sum(-1)
    # Independent witnesses include every intermediate, including source rounding.
    trace=locals()
    witnesses={key:torch.stack((trace[key].lo,trace[key].hi),-1) for key in ('du','dv','p','pe','B','be','D','nearval','de','qnum','U','V','HB','ue','ve','pos','dep','uh','hu','vh','hv','negae')}
    return reject,reasons,ctx.bad,signok,witnesses

@torch.no_grad()
def filter_pairs(c,u,v,n,nab,half,aa,ae,rho,pairs,gamma,guard,observer=None):
    masks=[];codes=[];witness=[];badcount=0;signcount=0
    for lo in range(0,len(pairs),65536):
        guard();pp=pairs[lo:lo+65536]
        reject,reason,bad,signok,ww=enclose(c,u,v,n,nab,half,aa,ae,rho,pp,gamma)
        masks.append(reject);codes.append(reason);badcount+=int(bad.sum());signcount+=int((~signok).sum())
        if observer is not None:
            ids=torch.linspace(0,len(pp)-1,min(32,len(pp)),device=c.device).long()
            witness.append(dict(offset=lo,indices=ids,trace={k:v[ids] for k,v in ww.items()}))
    reject=torch.cat(masks) if masks else torch.zeros(0,device=c.device,dtype=torch.bool)
    reason=torch.cat(codes) if codes else torch.zeros(0,device=c.device,dtype=torch.long)
    if observer is not None:observer(pairs,reject,reason,witness,(c,u,v,n,nab,half,aa,ae,rho,gamma))
    return pairs[~reject],dict(interval_rejected=int(reject.sum()),interval_original=len(pairs),interval_exceptional=badcount,interval_sign_uncertain=signcount)
