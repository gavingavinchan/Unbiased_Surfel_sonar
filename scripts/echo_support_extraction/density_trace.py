"""Trace original density rejection without changing accepted points.
GT distances are computed only after reproducing each retained point set.
"""
import ast,hashlib,importlib.util,json,os
from pathlib import Path
import numpy as np
import torch,trimesh,open3d as o3d
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from original_mesh_adapter import GaussianModel,remake_symmetric,SH2RGB
cfg=json.loads(Path('configs/extraction.json').read_text());srcpath=Path('/home/gavin/sonar-recon/third_party/sonar_splat/scripts/mesh_gaussian.py');src=srcpath.read_text();names={'_batch_mahalanobis','CustomMultivariateNormal','query_gaussians','create_pc_gaussians'};tree=ast.parse(src);tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
for n in ast.walk(tree):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='opacity_reflectance_mask' for t in n.targets):n.value=ast.parse('gaussians.selection_mask[~invalid_mask]',mode='eval').body
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name=='create_pc_gaussians':
  n.body[-1:-1]=ast.parse('gaussians.density_trace = high_density_mask.reshape(num_samples, -1).sum(0).cpu().numpy()').body
ast.fix_missing_locations(tree);o3d.visualization.draw_geometries=lambda *a,**k:None;plt.show=lambda *a,**k:plt.close('all');scope=globals().copy();scope['CHUNK_SIZE']=600;exec(compile(tree,str(srcpath),'exec'),scope)
spec=importlib.util.spec_from_file_location('metrics','/home/gavin/sonar-recon/tools/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);gt=m.load(Path(cfg['dataset'])/'gt_cube.ply');masks=np.load('outputs/support/masks.npz');model=GaussianModel(cfg['checkpoint']);d=m.distance(model.get_xyz.cpu().numpy(),gt);near=d<=.02;rows={}
for name in ['opacity','all','support2','support3']:
 torch.manual_seed(42);np.random.seed(42);mask=masks[name];model.selection_mask=torch.tensor(mask,dtype=torch.bool,device='cuda');pcd=scope['create_pc_gaussians'](model,num_samples=10,threshold=.9);original=o3d.io.read_point_cloud(f'outputs/{name}/samples.ply');assert np.array_equal(np.asarray(pcd.points),np.asarray(original.points)),name
 count=np.zeros(len(mask),int);count[mask]=model.density_trace
 rows[name]={'selected_means':int(mask.sum()),'near_gt_20mm_selected_means':int((mask&near).sum()),'means_with_any_density_sample':int((count>0).sum()),'near_gt_20mm_means_with_any_density_sample':int(((count>0)&near).sum()),'weak_opacity_le_point2_means_with_any_density_sample':int(((count>0)&(model.get_opacity.cpu().numpy().ravel()<=.2)).sum()),'mean_samples_accepted_per_near_gt_selected_mean':float(count[mask&near].mean()),'accepted_samples':int(count.sum()),'sample_points_identical_to_original':True}
 np.savez_compressed(f'outputs/{name}/density_trace.npz',accepted_samples_per_primitive=count);print(name,rows[name],flush=True)
Path('outputs/density_trace.json').write_text(json.dumps({'method':'same original density function, added tracing only; identical accepted point sets; GT labels evaluation only','rows':rows},indent=2)+'\n')
