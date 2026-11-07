"""Return-law checks; deliberately make no claim of opaque splat conservation."""
import math
import ast
import os
from pathlib import Path
import pytest
import torch

# As in test_sonar_bounded_linalg: execute the exact source helpers without
# requiring an unavailable compiled CUDA module on the CPU CI runner.
source = Path(__file__).resolve().parents[1] / 'gaussian_renderer/__init__.py'
names = {'sonar_area_received_power', 'resolve_sonar_render_contract'}
functions = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in functions} == names
namespace = {'torch': torch, 'math': math, 'os': os}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
sonar_area_received_power = namespace['sonar_area_received_power']
resolve_sonar_render_contract = namespace['resolve_sonar_render_contract']

def test_area_subdivision_range_and_backface():
    s=torch.tensor([[.1,.2]],dtype=torch.float64)
    mu=torch.tensor([.7],dtype=torch.float64);r=torch.tensor([2.],dtype=torch.float64);rho=torch.tensor([.8],dtype=torch.float64)
    q=sonar_area_received_power(s,mu,r,rho)
    assert q.item()==pytest.approx(.8*2*math.pi*.1*.2*.7**2/2**4,rel=1e-14)
    assert (4*sonar_area_received_power(s/2,mu,r,rho)).item()==pytest.approx(q.item(),rel=1e-14)
    assert sonar_area_received_power(s,mu,2*r,rho).item()==pytest.approx(q.item()/16,rel=1e-14)
    assert sonar_area_received_power(s,-mu,r,rho).item()==0

def test_area_gradients():
    s=torch.tensor([[.1,.2]],dtype=torch.float64,requires_grad=True)
    m=torch.tensor([.7],dtype=torch.float64,requires_grad=True)
    r=torch.tensor([2.],dtype=torch.float64,requires_grad=True)
    rho=torch.tensor([.8],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(sonar_area_received_power,(s,m,r,rho))

def test_opt_in_and_invalid_value(monkeypatch):
    monkeypatch.delenv('SONAR_RETURN_MODEL',raising=False)
    assert resolve_sonar_render_contract()['return_model']=='legacy'
    monkeypatch.setenv('SONAR_RETURN_MODEL','area_r4')
    assert resolve_sonar_render_contract()['return_model']=='area_r4'
    monkeypatch.setenv('SONAR_RETURN_MODEL','silent_fallback')
    with pytest.raises(ValueError,match='SONAR_RETURN_MODEL'):resolve_sonar_render_contract()
