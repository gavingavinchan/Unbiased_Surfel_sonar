"""Physical distinction missed by first-bright-row column censoring."""
import math
import torch
from directional_sonar import render

def scene(same=False):
    phi=math.radians(100/21)
    xyz=torch.tensor([[math.cos(phi),0,math.sin(phi)],[2*math.cos(phi),0,(1 if same else -1)*2*math.sin(phi)]],dtype=torch.float64)
    q=torch.tensor([[math.sqrt(.5),0,-math.sqrt(.5),0]]*2,dtype=torch.float64)
    return [xyz,q,torch.full((2,2),.025,dtype=torch.float64),torch.tensor([.999,.8],dtype=torch.float64),torch.ones(2,dtype=torch.float64),torch.eye(4,dtype=torch.float64)]

def test_different_elevation_preserves_far_echo():
    torch.set_num_threads(1);a=scene();both=render(*a);near=render(*[x[:1]for x in a[:5]],a[5]);far=render(*[x[1:]for x in a[:5]],a[5])
    torch.testing.assert_close(both,near+far,atol=1e-10,rtol=1e-10)
    assert float(far[125:142].sum())>0

def test_same_direction_attenuates_far_echo():
    torch.set_num_threads(1);a=scene(True);both=render(*a);near=render(*[x[:1]for x in a[:5]],a[5]);far=render(*[x[1:]for x in a[:5]],a[5]);ratio=float((both-near)[125:142].sum()/far[125:142].sum())
    assert 0<ratio<.3

def test_far_surface_position_gradient_matches_interior_difference():
    torch.set_num_threads(1);a=scene();a[0].requires_grad_(True);w=torch.arange(200,dtype=torch.float64)[:,None]/200
    value=(render(*a)*w).sum();value.backward();grad=float(a[0].grad[1,0]);h=1e-6
    plus=[x.detach().clone()for x in a];minus=[x.detach().clone()for x in a];plus[0][1,0]+=h;minus[0][1,0]-=h
    fd=float(((render(*plus)-render(*minus))*w).sum()/(2*h))
    assert abs(fd-grad)<1e-4*max(1,abs(fd),abs(grad))
