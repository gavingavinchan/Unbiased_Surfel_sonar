"""Conservative rejection, current-geometry rebuild and bounded stop contracts."""
import importlib
import sys
import types
from pathlib import Path
import torch
import pytest

pkg=types.ModuleType('_visibility416')
pkg.__path__=[str(Path(__file__).parents[1]/'gaussian_renderer')]
sys.modules.setdefault('_visibility416',pkg)
accel=importlib.import_module('_visibility416.ray_depth_accel')
oracle=importlib.import_module('_visibility416.ray_depth_oracle_409')


def geometry(c):
    c=torch.tensor(c,dtype=torch.float64)
    return [c,torch.tensor([[0.,1.,0.]]*len(c),dtype=torch.float64),torch.tensor([[0.,0.,-1.]]*len(c),dtype=torch.float64),torch.full((len(c),2),.03,dtype=torch.float64),torch.ones(len(c),dtype=torch.float64)]


def test_foreground_blocker_many_range_bins_away():
    args=geometry([[.5,0,0],[2.5,0,0]])
    f,d=accel.ray_depth_visibility(*args)
    assert torch.equal(f,torch.tensor([1.,0.],dtype=torch.float64))
    assert d['candidate_pairs']==4


def test_rebuild_after_blocker_moves():
    args=geometry([[.5,0,0],[2.5,0,0]])
    blocked=accel.ray_depth_visibility(*args)[0]
    args[0][0,1]=.3
    clear=accel.ray_depth_visibility(*args)[0]
    assert blocked[1]==0 and clear[1]==1
    assert torch.equal(clear,oracle.ray_depth_visibility(*args)[0])


def test_unsafe_origin_ball_rejects_nothing():
    args=geometry([[0,0,0],[2,0,0]])
    pairs,n=accel.candidate_pairs(*args[:3],(2*torch.pi)**.5*args[3]/2,args[4])
    assert n==1 and len(pairs)==4
    assert torch.equal(accel.ray_depth_visibility(*args)[0],oracle.ray_depth_visibility(*args)[0])


def test_nonorthonormal_decoded_tangents_use_dual_support():
    args=geometry([[1,0,0],[2,.03,0],[2.1,-.3,.2]])
    args[1]*=2
    args[2][:]=torch.tensor([0,.4,-.7],dtype=torch.float64)
    assert torch.equal(accel.ray_depth_visibility(*args)[0],oracle.ray_depth_visibility(*args)[0])


@pytest.mark.parametrize('seed',range(4))
def test_crowded_random_matches_oracle(seed):
    g=torch.Generator().manual_seed(seed)
    c=torch.randn((24,3),generator=g,dtype=torch.float64)*torch.tensor([.2,.1,.1]);c[:,0]+=1
    q=torch.randn((24,4),generator=g,dtype=torch.float64)
    u,v=oracle.quaternion_tangents(q)
    s=torch.exp(torch.randn((24,2),generator=g,dtype=torch.float64)*.5-3.5)
    args=[c,u,v,s,torch.ones(24,dtype=torch.float64)]
    assert torch.equal(accel.ray_depth_visibility(*args)[0],oracle.ray_depth_visibility(*args)[0])


def test_context_stop_between_chunks_preserves_inputs(monkeypatch):
    args=geometry([[1,0,0],[2,0,0]])
    old=args[0].clone();calls=[]
    def guard():
        calls.append(1)
        if len(calls)==3:raise accel.VisibilityInterrupted('controlled stop')
    monkeypatch.setenv('SONAR_VISIBILITY_PAIR_CHUNK','1')
    with pytest.raises(accel.VisibilityInterrupted),accel.visibility_execution(guard):
        accel.ray_depth_visibility(*args)
    assert torch.equal(old,args[0])
    assert torch.equal(accel.ray_depth_visibility(*args)[0],oracle.ray_depth_visibility(*args)[0])
