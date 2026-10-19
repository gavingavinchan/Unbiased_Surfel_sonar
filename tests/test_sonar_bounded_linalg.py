"""Full training-size covariance values, gradients and solver workspace batches."""
import ast
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
tree = ast.parse((Path(__file__).resolve().parents[1]/'gaussian_renderer/__init__.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_bounded_sonar_linalg')
ns = {'torch': torch}
exec(compile(ast.Module(body=[fn], type_ignores=[]), 'bounded-linalg', 'exec'), ns)
bounded = ns[fn.name]

@pytest.mark.parametrize('operation', ['eigvalsh', 'cholesky', 'eigh'])
def test_full_size_values_and_gradients(operation):
    generator = torch.Generator().manual_seed(78)
    a = torch.randn(12000, 2, 2, generator=generator, dtype=torch.float64)
    matrix = (a@a.transpose(-1,-2) + torch.diag(torch.tensor([1.,3.],dtype=torch.float64))).requires_grad_()
    other = matrix.detach().clone().requires_grad_()
    op = getattr(torch.linalg, operation)
    expected = op(matrix)
    actual = bounded(op, other)
    if operation == 'eigh':
        for x,y in zip(actual, expected): torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
        actual,expected = actual[0],expected[0]
    else: torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    actual.square().sum().backward(); expected.square().sum().backward()
    torch.testing.assert_close(other.grad,matrix.grad,atol=1e-10,rtol=1e-10)


def test_solver_batches_are_bounded_with_partial_last_batch():
    sizes=[]
    matrix=torch.eye(2).repeat(12001,1,1)
    def solver(x):
        sizes.append(len(x)); return x+1
    torch.testing.assert_close(bounded(solver,matrix),matrix+1)
    assert max(sizes)==512 and sum(sizes)==12001 and sizes[-1]==225


def test_allocation_errors_are_not_hidden():
    def solver(x): raise torch.OutOfMemoryError('retained error')
    with pytest.raises(torch.OutOfMemoryError,match='retained error'):
        bounded(solver,torch.eye(2).repeat(513,1,1))
