"""Actual fallback checkpoint parity, through a nonuniform image readout."""
import torch
from test_sonar_deposition_contract_337 import ns, args

def test_checkpoint_multichunk_reverse_and_guard(monkeypatch):
    monkeypatch.setenv('SONAR_DEPOSITION_CONTRACT','intrinsic_screen_mass_c1_456_v1')
    monkeypatch.setenv('SONAR_RASTER_FOOTPRINT_NORM','mass')
    monkeypatch.setenv('SONAR_USE_CUDA_RASTERIZER','0')
    outputs=[];gradients=[];calls=[]
    monkeypatch.setitem(ns,'_sonar_chunk_guard_509',lambda:calls.append(1))
    for checkpoint in ['0','1']:
        monkeypatch.setenv('SONAR_C1_CHUNK_CHECKPOINT',checkpoint)
        a=args();a['vis_mu']=torch.tensor([[.2,.3],[4.1,4.2]],dtype=torch.float64,requires_grad=True)
        a['vis_sigma']=torch.eye(2,dtype=torch.float64).repeat(2,1,1).requires_grad_()
        a['vis_range']=torch.tensor([1.3,1.4],dtype=torch.float64,requires_grad=True)
        a['event_surfel_idx']=torch.tensor([0,1]);a['event_ray_ids']=torch.tensor([3,4])
        a['event_returns']=torch.tensor([.7,.6],dtype=torch.float64,requires_grad=True)
        a['event_share']=torch.tensor([.3,.2],dtype=torch.float64,requires_grad=True)
        a['chunk']=1
        out=ns['_rasterize_sonar_event_volume_matched_fallback'](**a)
        loss=sum((v*torch.arange(v.numel(),dtype=v.dtype).reshape(v.shape).sin()).sum() for v in out[:3])
        gs=torch.autograd.grad(loss,[a[k] for k in ['vis_mu','vis_sigma','vis_range','event_returns','event_share']])
        outputs.append(out);gradients.append(gs)
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
    assert all(torch.equal(a,b) for a,b in zip(*gradients))
    assert len(calls)>8
