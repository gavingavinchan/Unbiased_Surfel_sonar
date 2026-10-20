"""Native routing and endpoint regressions: independent column-vector expectations."""
import ast
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from utils.sonar_utils import (
    SonarConfig, SonarExtrinsic, build_sonar_config, build_debug_sonar_config,
    resolve_sonar_extrinsic, sonar_frame_to_points, back_project_bins,
    get_scaled_world_to_view_transform, APERTURE_QUANTUM_RAD,
)
from utils.point_utils import sonar_ranges_to_points
from utils.visualization_utils import compute_frame_surfel_membership

ROOT = Path(__file__).resolve().parents[1]
# Exercise real Python geometry without importing the optional CUDA extensions.
functions = {'SonarProjection', '_transform_world_points_to_sonar_frame',
             'sonar_project_points', '_sonar_config_values', '_project_points_to_sonar_batch'}
ns = dict(torch=torch, math=math, dataclass=dataclass,
          get_scaled_world_to_view_transform=get_scaled_world_to_view_transform)
tree = ast.parse((ROOT/'gaussian_renderer/__init__.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                            and n.name in functions], type_ignores=[]), 'renderer_geometry', 'exec'), ns)
project = ns['sonar_project_points']
batch_project = ns['_project_points_to_sonar_batch']


def cfg_for(route, origin, span, offset, mode, width=256, height=200):
    if route == 'debug':
        env = dict(SONAR_RANGE_ORIGIN=str(origin), SONAR_RANGE_SPAN=str(span),
                   SONAR_PIXEL_CENTER_OFFSET=str(offset), SONAR_POSE_MODE=mode)
        return build_debug_sonar_config(env, image_width=width, image_height=height, device='cpu')
    args = SimpleNamespace(sonar_range_origin=origin, sonar_range_span=span,
                           sonar_pixel_center_offset=offset, sonar_pose_mode=mode,
                           sonar_azimuth_fov=120., sonar_elevation_fov=20.,
                           sonar_intensity_threshold=.01)
    return build_sonar_config(args, image_width=width, image_height=height, device='cpu')


def rotation():
    # Noncommuting rotations, independent of the fork's camera/mount helpers.
    x, y = .31, -.47
    Rx = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    return Ry @ Rx


@pytest.mark.parametrize('route', ['train', 'debug'])
@pytest.mark.parametrize('origin,span,offset', [(.2,2.8,0.),(0.,3.,.5)])
@pytest.mark.parametrize('mode', ['poses_are_sonar','poses_are_camera'])
@pytest.mark.parametrize('width,height', [(256,200),(128,100)])
def test_native_grid_pose_routing_across_geometry_paths(route, origin, span, offset, mode, width, height):
    cfg = cfg_for(route,origin,span,offset,mode,width,height)
    C = np.eye(4); C[:3,:3] = rotation(); C[:3,3] = [.3,-.2,.8]
    scale = .65
    metric = C.copy(); metric[:3,3] *= scale
    E = np.eye(4)
    if mode == 'poses_are_camera':
        pitch = math.radians(5)
        E[:3,:3] = [[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]]
        E[:3,3] = -E[:3,:3] @ np.array([0.,-.10,-.08])
    expected = E @ metric
    row, col = height//2, width//2
    r = origin+(row+offset)*span/height
    theta = math.pi/3-(col+offset)*2*math.pi/3/width
    local = np.array([-r*math.sin(theta),0.,r*math.cos(theta)])
    world = (local-expected[:3,3]) @ expected[:3,:3] / scale
    cam = SimpleNamespace(world_view_transform=torch.tensor(C.T,dtype=torch.float64),
                          original_image=torch.zeros(1,height,width),image_height=height,image_width=width)
    sf = SimpleNamespace(scale=scale)
    p = project(torch.tensor(world[None]),cam,cfg,sf)
    assert p.valid.item()
    assert abs(p.row.item()-row) <= 1e-4 and abs(p.col.item()-col) <= 1e-4
    cam.original_image[0,row,col] = 1.
    points,_ = sonar_frame_to_points(cam,cfg,intensity_threshold=.5,mask_top_rows=0,
                                    scale_factor=scale,elevation_mode='zero')
    assert np.max(abs(points[0]-world)) <= 1e-6
    bins = back_project_bins(0,torch.tensor([row]),torch.tensor([col]),torch.tensor([0.]),
                            cameras=[cam],sonar_config=cfg,scale_factor=sf)
    assert np.max(abs(bins.numpy()[0,0]-world)) <= 1e-6
    ranges = torch.full((1,height,width),r,dtype=torch.float64)
    points = sonar_ranges_to_points(cam,ranges,cfg,sf)
    assert np.max(abs(points.numpy()[row,col]-world)) <= 1e-6
    vis = compute_frame_surfel_membership(world[None],np.array([[.01,.02]]),
           np.array([[1.,0,0,0]]),cam,cfg,sf)
    assert np.max(abs(vis['points_view'][0]-local)) <= 1e-6
    assert vis['center_in_fov'][0]


def test_already_sonar_poses_apply_exactly_zero_mount_and_reject_double_mount():
    cfg = cfg_for('train',0,3,.5,'poses_are_sonar')
    point = torch.tensor([[.2,.1,1.6]])
    cam = SimpleNamespace(world_view_transform=torch.eye(4),image_height=200,image_width=256)
    ex = resolve_sonar_extrinsic(cfg)
    actual,_ = ns['_transform_world_points_to_sonar_frame'](point,cam,None,ex)
    assert (actual-point).abs().max().item() == 0.0
    with pytest.raises(ValueError,match='Already-sonar'):
        project(point,cam,cfg,sonar_extrinsic=SonarExtrinsic(device='cpu'))


@pytest.mark.parametrize('offset',[0.,.5])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_exact_and_nearby_aperture_endpoints(offset,dtype):
    cfg = SonarConfig(device='cpu',range_origin=0,range_span=3,pixel_center_offset=offset)
    angles=[];wanted=[]
    # Both signs, both apertures, and corner endpoints; +/-4 ticks is nearby
    # but unambiguously separated from the stated half-tick boundary cell.
    for axis in ('az','el','corner'):
        for sign in (-1,1):
            for delta in (-4*APERTURE_QUANTUM_RAD,0.,4*APERTURE_QUANTUM_RAD):
                az = sign*(math.pi/3+delta) if axis in ('az','corner') else 0.
                el = sign*(math.pi/18+delta) if axis in ('el','corner') else 0.
                angles.append((az,el));wanted.append(delta<=0)
    R=rotation();t=np.array([.31,-.22,.83])
    local=np.array([[-1.6*np.sin(a)*np.cos(e),1.6*np.sin(e),1.6*np.cos(a)*np.cos(e)] for a,e in angles])
    world=(local-t)@R
    C=np.eye(4);C[:3,:3]=R;C[:3,3]=t
    cam=SimpleNamespace(world_view_transform=torch.tensor(C.T,dtype=dtype),image_height=200,image_width=256)
    actual=project(torch.tensor(world,dtype=dtype),cam,cfg)
    assert actual.valid.tolist()==wanted
    assert batch_project(torch.tensor(local,dtype=dtype),cfg)['valid'].tolist()==wanted
    # At +60 degrees a centre grid's raw column is -0.5, but membership is true.
    assert actual.col[4].item() == pytest.approx(-offset,abs=1e-4)
    assert actual.valid[4]


def test_boundary_cell_is_explicit_and_outside_it_is_rejected():
    cfg=SonarConfig(device='cpu')
    q=APERTURE_QUANTUM_RAD
    a=torch.tensor([math.pi/3+d*q for d in [-.51,-.49,0,.49,.51,1]],dtype=torch.float64)
    mask,_=cfg.aperture_masks(a,torch.zeros_like(a))
    assert mask.tolist()==[True,True,True,True,False,False]


def test_generator_row_pose_construction_composition_extraction():
    tree=ast.parse((ROOT/'scripts/generate_synthetic_sonar_dataset.py').read_text())
    names={'build_row_major_pose_matrix','extract_rt_from_row_major_pose','camera_center_from_w2v'}
    local={'np':np,'Tuple':tuple}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                 and n.name in names],type_ignores=[]),'generator_pose','exec'),local)
    R=rotation();t=np.array([.3,-.2,.8]);W2S=np.eye(4);W2S[:3,:3]=R;W2S[:3,3]=t
    E=np.eye(4);E[:3,:3]=rotation().T;E[:3,3]=[.04,-.12,.07]
    stored=local['build_row_major_pose_matrix'](R,t)
    np.testing.assert_array_equal(stored,W2S.T)
    camera=stored@np.linalg.inv(E.T)
    Rc,tc=local['extract_rt_from_row_major_pose'](camera)
    expected=np.linalg.inv(E)@W2S
    assert np.max(abs(Rc-expected[:3,:3]))<=1e-12
    assert np.max(abs(tc-expected[:3,3]))<=1e-12
    assert np.max(abs(local['camera_center_from_w2v'](Rc,tc)-np.linalg.inv(expected)[:3,3]))<=1e-12


def test_native_entries_use_shared_configuration_and_no_unconditional_mount():
    tree=ast.parse((ROOT/'debug_multiframe.py').read_text())
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                 isinstance(n.targets[0],ast.Name) and n.targets[0].id=='sonar_config']
    assert len(assignments)==2
    assert all(isinstance(n.value,ast.Call) and n.value.func.id=='build_debug_sonar_config' for n in assignments)
    tree=ast.parse((ROOT/'train.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and
                   n.func.id=='SonarExtrinsic' for n in ast.walk(tree))


def test_source_pose_precision_preserves_archive_floor_return_bin():
    import json
    from utils.graphics_utils import getWorld2View2
    f=json.loads((ROOT/'tests/fixtures/archive_floor_boundary.json').read_text())
    R=np.array(f['R_w2c']);t=np.array(f['t_w2c'])
    S=getWorld2View2(R.T,t,dtype=np.float64).T
    cam=SimpleNamespace(world_view_transform=torch.tensor(S,dtype=torch.float32),
        world_view_transform_precise=torch.tensor(S,dtype=torch.float64),
        image_height=200,image_width=256)
    point=torch.tensor([f['point_world']],dtype=torch.float32)
    cfg=cfg_for('train',.2,2.8,0.,'poses_are_sonar')
    row=project(point,cam,cfg).row.item()
    assert math.floor(row)==f['expected_floor_bin']
    assert abs(row-f['expected_row'])<1e-6
