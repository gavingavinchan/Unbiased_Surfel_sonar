"""CPU tests of the opt-in full footprint; compiled parity lives in experiment337."""
import ast, math, os
from pathlib import Path
import pytest
import torch
p=Path(__file__).resolve().parents[1]/'gaussian_renderer/__init__.py'
names={'_condition_sigma_2d','_sonar_deposition_contract','_rasterize_sonar_event_volume_matched_fallback'}
nodes=[x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names]
ns={'torch':torch,'math':math,'os':os};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)

def args():
    return dict(vis_mu=torch.tensor([[4.,4.]],dtype=torch.float64),vis_sigma=torch.tensor([[[.1,0],[0,.1]]],dtype=torch.float64),vis_range=torch.tensor([1.3],dtype=torch.float64),event_surfel_idx=torch.tensor([0]),event_ray_ids=torch.tensor([3]),event_returns=torch.tensor([.7],dtype=torch.float64),event_share=torch.tensor([.3],dtype=torch.float64),image_width=9,image_height=9,elev_bins=2)

def enabled(monkeypatch):
    monkeypatch.setenv('SONAR_DEPOSITION_CONTRACT','intrinsic_screen_mass_v1');monkeypatch.setenv('SONAR_RASTER_FOOTPRINT_NORM','mass');monkeypatch.delenv('SONAR_RASTER_LINEAR_OPACITY',raising=False)

def test_default_and_fail_closed(monkeypatch):
    monkeypatch.delenv('SONAR_DEPOSITION_CONTRACT',raising=False)
    assert ns['_sonar_deposition_contract']()=='legacy'
    monkeypatch.setenv('SONAR_DEPOSITION_CONTRACT','typo')
    with pytest.raises(ValueError):ns['_sonar_deposition_contract']()
    enabled(monkeypatch);monkeypatch.setenv('SONAR_RASTER_FOOTPRINT_NORM','none')
    with pytest.raises(ValueError):ns['_sonar_deposition_contract']()
    monkeypatch.setenv('SONAR_RASTER_FOOTPRINT_NORM','mass');monkeypatch.setenv('SONAR_RASTER_LINEAR_OPACITY','.1')
    with pytest.raises(ValueError):ns['_sonar_deposition_contract']()

def test_analytic_screen_mass_and_event_elevation(monkeypatch):
    enabled(monkeypatch);a=args();r,d,s,_=ns['_rasterize_sonar_event_volume_matched_fallback'](**a)
    z=1+4*math.exp(-1)+4*math.exp(-2)
    assert r.sum().item()==pytest.approx(.7,abs=1e-14)
    assert r[4,1,4].item()==pytest.approx(.7/z,abs=1e-14)
    assert r[:,0,:].count_nonzero()==0
    assert r[:,1,:].count_nonzero()==9
    assert d.sum().item()==pytest.approx(.91)
    assert s.sum().item()==pytest.approx(.3)
    # Azimuth column is upstream-only; moving it leaves the deposition intact.
    a['event_ray_ids']=torch.tensor([15]);r2=ns['_rasterize_sonar_event_volume_matched_fallback'](**a)[0]
    assert torch.equal(r,r2)

def test_edge_crop_renormalization_and_zero(monkeypatch):
    enabled(monkeypatch);a=args();a['vis_mu']=torch.tensor([[0.,0.]],dtype=torch.float64)
    r=ns['_rasterize_sonar_event_volume_matched_fallback'](**a)[0]
    assert r.sum().item()==pytest.approx(.7,abs=1e-14)
    assert r.count_nonzero()==4
    a['event_returns'].zero_();assert ns['_rasterize_sonar_event_volume_matched_fallback'](**a)[0].count_nonzero()==0
    a['vis_mu'][0,0]=float('nan')
    with pytest.raises(ValueError):ns['_rasterize_sonar_event_volume_matched_fallback'](**a)
