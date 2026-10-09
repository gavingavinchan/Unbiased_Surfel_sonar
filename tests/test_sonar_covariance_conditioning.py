"""Finite gradients at isotropic/clipped covariance; preserve original spectral values."""
import ast
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
root = Path(__file__).resolve().parents[1]
tree = ast.parse((root/'gaussian_renderer/__init__.py').read_text())
fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_condition_sigma_2d')
ns={'torch':torch}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'conditioning','exec'),ns)
condition=ns['_condition_sigma_2d']


def test_matches_original_spectral_clamp_on_anisotropic_matrices():
    generator=torch.Generator().manual_seed(42)
    A=torch.randn(200,2,2,generator=generator,dtype=torch.float64)
    sigma=A@A.transpose(-1,-2)*torch.logspace(-3,4,200,dtype=torch.float64)[:,None,None]
    values,vectors=torch.linalg.eigh(sigma)
    values=values.clamp(.1,400)
    minimum=(values.max(-1,keepdim=True).values/100).clamp_min(.1)
    values=torch.maximum(values,minimum).clamp(max=400)
    reference=vectors@torch.diag_embed(values)@vectors.transpose(-1,-2)
    torch.testing.assert_close(condition(sigma),reference,atol=1e-9,rtol=1e-9)


@pytest.mark.parametrize('variance',[.01,.1,1.,400.,500.])
def test_repeated_eigenvalues_have_finite_backward(variance):
    sigma=(torch.eye(2,dtype=torch.float64)*variance).requires_grad_()
    result=condition(sigma)
    result.sum().backward()
    assert torch.isfinite(sigma.grad).all()
    torch.testing.assert_close(result,torch.eye(2,dtype=torch.float64)*min(max(variance,.1),400.))


def test_gradient_agrees_with_finite_differences_away_from_clamp_boundaries():
    sigma=torch.tensor([[1.,.2],[.2,2.]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(condition,(sigma,),eps=1e-6,atol=1e-5,rtol=1e-5)
