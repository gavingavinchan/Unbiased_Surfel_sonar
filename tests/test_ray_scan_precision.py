"""Ray independence must survive large prefixes from other image rays."""
import ast
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')


def compose():
    path = Path(__file__).resolve().parents[1] / 'gaussian_renderer/__init__.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'compose_ray_binned_occlusion')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['compose_ray_binned_occlusion']


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('prior_alpha', [.01, .5, .9, 1.])
@pytest.mark.parametrize('target_first', [False, True])
def test_transmission_and_gradient_independent_of_other_rays(dtype, prior_alpha, target_first):
    count = 262144
    previous_ids = torch.arange(count) // 32 + int(target_first)
    last_id = 0 if target_first else count // 32
    ids = torch.cat([previous_ids, torch.full((3,), last_id)])
    ranges = torch.cat([(torch.arange(count, dtype=dtype) % 32) * .01 + .1,
                        torch.tensor([.2, .4, .6], dtype=dtype)])
    alpha = torch.cat([torch.full((count,), prior_alpha, dtype=dtype),
                       torch.tensor([.01, .2, .5], dtype=dtype)]).requires_grad_()
    result = compose()(ids, ranges, alpha, torch.ones(count + 3, dtype=dtype), count // 32 + 1)
    expected = torch.tensor([1., .99, .792], dtype=dtype)
    assert result['event_returns'].dtype == dtype
    torch.testing.assert_close(result['event_returns'][-3:], expected, atol=2e-7, rtol=0)
    torch.testing.assert_close(result['final_transmittance'][last_id],
                              torch.tensor(.396, dtype=dtype), atol=2e-7, rtol=0)
    gradient = torch.autograd.grad(result['event_returns'][-3:].sum(), alpha)[0]
    torch.testing.assert_close(gradient[-3:], torch.tensor([-1.8, -.99, 0], dtype=dtype),
                              atol=3e-7, rtol=0)
    assert gradient[:-3].abs().max() == 0


def test_empty_rays_stay_transparent():
    result = compose()(torch.empty(0, dtype=torch.long), torch.empty(0),
                       torch.empty(0), torch.empty(0), 4)
    assert result['event_returns'].numel() == 0
    torch.testing.assert_close(result['final_transmittance'], torch.ones(4))
