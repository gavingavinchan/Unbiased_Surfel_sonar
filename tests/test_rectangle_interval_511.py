"""Independent small arithmetic fixtures; no renderer/native extension import."""
import importlib.util
from pathlib import Path
from fractions import Fraction
import itertools
import torch
spec=importlib.util.spec_from_file_location('interval511',Path(__file__).parents[1]/'gaussian_renderer/rectangle_interval_511.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_fraction_endpoint_enclosure():
    values=[0.,-0.,1.,-1.,float.fromhex('0x1.0000000000001p0'),2.**-500,-2.**-500,2.**500]
    ab=list(itertools.product(values,repeat=2));a=torch.tensor([p[0] for p in ab],dtype=torch.float64);b=torch.tensor([p[1] for p in ab],dtype=torch.float64)
    for op in ['add','sub','mul']:
        ctx=m.Context(a);l,h=ctx.rounded(a,b,op)
        for k,(x,y) in enumerate(ab):
            exact={'add':lambda:Fraction(x)+Fraction(y),'sub':lambda:Fraction(x)-Fraction(y),'mul':lambda:Fraction(x)*Fraction(y)}[op]()
            assert Fraction(float(l[k]))<=exact<=Fraction(float(h[k]))
def test_subnormal_and_nan_unsafe():
    a=torch.tensor([float.fromhex('0x0.0000000000001p-1022'),float('nan'),1e-200],dtype=torch.float64);b=torch.tensor([1.,1.,1e-200],dtype=torch.float64)
    ctx=m.Context(a);ctx.rounded(a,b,'mul');assert bool(ctx.bad.all())
def test_all_sum_orders():
    a=torch.tensor([[1.,-1.,2.**-54],[1e30,1.,-1e30]],dtype=torch.float64)
    ctx=m.Context(a);r=ctx.box(a).sum3()
    for k,row in enumerate(a.tolist()):
        for x,y,z in itertools.permutations(row):
            assert float(r.lo[k])<=(x+y)+z<=float(r.hi[k])
            assert float(r.lo[k])<=x+(y+z)<=float(r.hi[k])
