"""Paired sampling control, not the original013 extraction protocol.
Draw exactly the same world samples for each Gaussian before applying any mask.
Original density/query/MC/smoothing bodies otherwise stay fixed. Uses baseline
metric pitch for every population, so mask is the only population difference.
"""
import argparse,ast,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch,trimesh,open3d as o3d
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from original_mesh_adapter import GaussianModel,remake_symmetric,SH2RGB
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);a=p.parse_args();cfg=json.loads(Path('configs/extraction.json').read_text());srcpath=Path('/home/gavin/sonar-recon/third_party/sonar_splat/scripts/mesh_gaussian.py');src=srcpath.read_text();tree=ast.parse(src);names={'_batch_mahalanobis','CustomMultivariateNormal','query_gaussians','create_pc_gaussians','marching_cubes'};tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
changes=[]
for n in ast.walk(tree):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='opacity_reflectance_mask' for t in n.targets):n.value=ast.parse('gaussians.selection_mask[~invalid_mask]',mode='eval').body;changes.append('primitive_selection')
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='samples' for t in n.targets):n.value=ast.parse('gaussians.common_samples[:, ~invalid_mask, :][:, opacity_reflectance_mask, :]',mode='eval').body;changes.append('paired_primitive_samples')
assert sorted(changes)==['paired_primitive_samples','primitive_selection'];ast.fix_missing_locations(tree);o3d.visualization.draw_geometries=lambda *a,**k:None;plt.show=lambda *a,**k:plt.close('all');scope=globals().copy();scope['CHUNK_SIZE']=600;exec(compile(tree,str(srcpath),'exec'),scope)
model=GaussianModel(cfg['checkpoint']);cov=remake_symmetric(model.get_covariance().detach())+torch.eye(3,device='cuda')[None]*1e-5;assert torch.isfinite(cov).all();torch.manual_seed(a.seed);np.random.seed(a.seed);full_mvn=scope['CustomMultivariateNormal'](loc=model.get_xyz.detach(),covariance_matrix=cov);model.common_samples=full_mvn.sample((10,));samplehash=hashlib.sha256(model.common_samples.cpu().numpy().tobytes()).hexdigest();pitch=json.loads(Path('outputs/opacity/metadata.json').read_text())['pitch_m'];masks=np.load('outputs/support/masks.npz')
for name in ['opacity','all','support3']:
 start=time.perf_counter();out=Path(f'outputs/paired_s{a.seed}')/name;out.mkdir(parents=True,exist_ok=True);mask=masks[name];model.selection_mask=torch.tensor(mask,dtype=torch.bool,device='cuda');pcd=scope['create_pc_gaussians'](model,num_samples=10,threshold=.9)
 if not len(pcd.points):raise ValueError('empty density samples in paired control')
 o3d.io.write_point_cloud(str(out/'samples.ply'),pcd);scale=pitch*150/np.ptp(np.asarray(pcd.points),axis=0).max();shape=np.ceil(np.ptp(np.asarray(pcd.points),axis=0)/pitch).astype(int)+6;assert np.prod(shape)<=35000000
 scope['marching_cubes'](pcd,str(out),-1,'sampling',filtering_iters=2,pitch_scale=scale);mesh=trimesh.load(out/'sampling_mesh_iteration.ply',process=False)
 meta={'population':name,'seed':a.seed,'available':True,'paired_common_sample_sha256':samplehash,'full_gaussian_sample_shape':list(model.common_samples.shape),'sample_points':len(pcd.points),'retained':int(mask.sum()),'pitch_m':pitch,'density_threshold':.9,'num_samples':10,'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'controlled_ast_substitutions':changes,'wall_s':time.perf_counter()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'gt_or_roi_in_extraction':False}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(out,meta,flush=True)
 if a.seed==42 and name=='all':assert (out/'sampling_mesh_iteration.ply').read_bytes()==Path('outputs/common_pitch/all/sampling_mesh_iteration.ply').read_bytes()
