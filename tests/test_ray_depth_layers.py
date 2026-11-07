"""Forward geometry tests loaded without CUDA-extension imports on CPU CI."""
import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('ray_layers',Path(__file__).parents[1]/'gaussian_renderer/ray_depth_layers.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def run(c, u=None):
    c=torch.tensor(c,dtype=torch.float32)
    u=torch.tensor(u if u is not None else [[0.,1.,0.]]*len(c))
    v=torch.tensor([[0.,0.,-1.]]*len(c))
    return mod.ray_depth_visibility(c,u,v,torch.full((len(c),2),.1),torch.ones(len(c)))

def test_duplicate_and_hidden():
    f,d=run([[1.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([.5,.5,0.]))
    assert d['invalid_source_samples']==0

def test_close_resolved_and_unresolved():
    f,_=run([[1.,0.,0.],[1.005,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([1.,0.]))
    f,_=run([[1.,0.,0.],[1.000001,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([.5,.5]))

def test_disjoint_same_ray_bin_not_merged():
    f,_=run([[1.,-.2,0.],[1.,.2,0.]])
    torch.testing.assert_close(f,torch.ones(2))

def test_zero_reflectance_does_not_block():
    c=torch.tensor([[1.,0.,0.],[2.,0.,0.]])
    u=torch.tensor([[0.,1.,0.]]*2);v=torch.tensor([[0.,0.,-1.]]*2)
    f,_=mod.ray_depth_visibility(c,u,v,torch.full((2,2),.1),torch.tensor([0.,1.]))
    assert f[1]==1
