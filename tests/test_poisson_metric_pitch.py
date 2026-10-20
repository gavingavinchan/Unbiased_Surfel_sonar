"""Exercise native extraction argument routing without optional Open3D/CUDA imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def helper():
    source = Path(__file__).resolve().parents[1] / 'debug_multiframe.py'
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'save_poisson_mesh')
    calls = []
    mesh = SimpleNamespace(vertices=[], triangles=[], remove_vertices_by_mask=lambda x: None)
    def create(pcd, **kwargs):
        calls.append(kwargs)
        return mesh, np.array([])
    o3d = SimpleNamespace(geometry=SimpleNamespace(PointCloud=SimpleNamespace,
        TriangleMesh=SimpleNamespace(create_from_point_cloud_poisson=create)),
        utility=SimpleNamespace(Vector3dVector=lambda x: x),
        io=SimpleNamespace(write_triangle_mesh=lambda *a: None))
    ns = dict(np=np, o3d=o3d, os=__import__('os'), POISSON_DEPTH=8,
              POISSON_DENSITY_QUANTILE=0.)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
    return ns['save_poisson_mesh'], calls


def test_metric_pitch_and_thread_limit():
    fn, calls = helper()
    fn(np.ones((4, 3)), np.ones((4, 3)), '.', 'unused.ply', metric_pitch=.01, n_threads=1)
    assert calls == [dict(depth=0, width=.01, n_threads=1)]


def test_default_depth_unchanged():
    fn, calls = helper()
    fn(np.ones((4, 3)), np.ones((4, 3)), '.', 'unused.ply')
    assert calls == [dict(depth=8)]


@pytest.mark.parametrize('kw', [dict(metric_pitch=0), dict(metric_pitch=float('nan')),
    dict(metric_pitch=-1), dict(n_threads=0), dict(n_threads=-1), dict(n_threads=True)])
def test_invalid_limits_rejected(kw):
    fn, calls = helper()
    with pytest.raises(ValueError):
        fn(np.ones((4, 3)), np.ones((4, 3)), '.', 'unused.ply', **kw)
    assert not calls
