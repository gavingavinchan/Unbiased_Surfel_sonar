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

def test_distinct_stored_layers_are_resolved():
    f,_=run([[1.,0.,0.],[1.005,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([1.,0.]))
    f,_=run([[1.,0.,0.],[1.000001,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([1.,0.]))

def test_disjoint_same_ray_bin_not_merged():
    f,_=run([[1.,-.2,0.],[1.,.2,0.]])
    torch.testing.assert_close(f,torch.ones(2))

def test_zero_reflectance_does_not_block():
    c=torch.tensor([[1.,0.,0.],[2.,0.,0.]])
    u=torch.tensor([[0.,1.,0.]]*2);v=torch.tensor([[0.,0.,-1.]]*2)
    f,_=mod.ray_depth_visibility(c,u,v,torch.full((2,2),.1),torch.tensor([0.,1.]))
    assert f[1]==1


def test_thin_subdivision_and_no_boundary_gradient():
    c=torch.tensor([[1.,0.,0.],[1.,0.,1.25e-5]],requires_grad=True)
    u=torch.tensor([[0.,1.,0.]]*2);v=torch.tensor([[0.,0.,1.]]*2)
    scales=torch.tensor([[.01,1.25e-5/(2*torch.pi)**.5]]*2)
    f,d=mod.ray_depth_visibility(c,u,v,scales,torch.ones(2))
    torch.testing.assert_close(f,torch.ones(2))
    assert not f.requires_grad
    assert d['max_source_depth_tolerance_m']==0


def test_source_plane_at_origin_is_unavailable():
    f,d=run([[0.,1.,0.]])
    assert d['invalid_source_samples']==256
    assert f.item()==0


# --- 409 representation-coincidence rule -------------------------------------
def run_rep(c, rep_scale=1.0):
    c=torch.tensor(c,dtype=torch.float64)
    u=torch.tensor([[0.,1.,0.]]*len(c),dtype=torch.float64)
    v=torch.tensor([[0.,0.,-1.]]*len(c),dtype=torch.float64)
    u32=2.0**-24
    delta=u32*c.norm(dim=-1)*1.01*rep_scale
    eps=torch.full((len(c),),16*u32)
    return mod.ray_depth_visibility(c,u,v,torch.full((len(c),2),.1,dtype=torch.float64),torch.ones(len(c)),
        representation_error=(delta,eps),out_dtype=torch.float32)

def test_rep_exact_duplicate_ties():
    f,d=run_rep([[1.,0.,0.],[1.,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([.5,.5]))
    assert d['representation_coincident_pairs']==2

def test_rep_reencoded_coplanar_offset_ties():
    # 1e-10 m offset is far below float32 representation error at 1 m.
    f,d=run_rep([[1.,0.,0.],[1.+1e-10,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([.5,.5]))
    assert d['representation_inexact_coincident_pairs']==2

def test_rep_10um_layers_stay_ordered_both_signs():
    f,_=run_rep([[1.,0.,0.],[1.00001,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([1.,0.]))
    f,_=run_rep([[1.00001,0.,0.],[1.,0.,0.]])
    torch.testing.assert_close(f,torch.tensor([0.,1.]))

def test_rep_bound_is_far_below_10um_at_metre_scale():
    _,d=run_rep([[1.3,0.,0.],[1.3,0.,1e-11]])
    assert 0<d['max_representation_bound_m']<2e-6

def test_rep_disjoint_coplanar_not_hidden():
    f,_=run_rep([[1.,-.2,0.],[1.,.2,0.]])
    torch.testing.assert_close(f,torch.ones(2))
