#!/usr/bin/env python3
"""013 adapter with one controlled primitive-selection substitution.
The source function bodies are unchanged except opacity_reflectance_mask when
--population differs from opacity. All input means/scales/rotations/weights,
global reflectance normalization, density, sampling and MC/smoothing are frozen.
"""
import argparse, ast, hashlib, json, time, os
from pathlib import Path
import numpy as np
import torch
import trimesh
import open3d as o3d
from plyfile import PlyData
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from original_mesh_adapter import GaussianModel, remake_symmetric, SH2RGB


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/extraction.json');parser.add_argument('--masks',default='outputs/support/masks.npz');parser.add_argument('--population',choices=['opacity','all','support2','support3'],required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    source_path=Path('/home/gavin/sonar-recon/third_party/sonar_splat/scripts/mesh_gaussian.py');source=source_path.read_text();tree=ast.parse(source)
    names={'_batch_mahalanobis','CustomMultivariateNormal','query_gaussians','create_pc_gaussians','marching_cubes'}
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    substitutions=0
    if args.population!='opacity':
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='opacity_reflectance_mask' for t in node.targets):
                assert ast.unparse(node.value)=='valid_opacities > 0.2'
                node.value=ast.parse('gaussians.selection_mask[~invalid_mask]',mode='eval').body;substitutions+=1
        assert substitutions==1
    ast.fix_missing_locations(tree)
    o3d.visualization.draw_geometries=lambda *a,**k:None;plt.show=lambda *a,**k:plt.close('all')
    scope=globals().copy();scope['CHUNK_SIZE']=600;exec(compile(tree,str(source_path),'exec'),scope)
    model=GaussianModel(cfg['checkpoint']);mask=np.load(args.masks)[args.population];model.selection_mask=torch.tensor(mask,device='cuda',dtype=torch.bool)
    if not mask.any():
        result={'available':False,'reason':'no retained primitives','retained':0}
    else:
        pcd=scope['create_pc_gaussians'](model,num_samples=cfg['num_samples_per_gaussian'],threshold=cfg['density_threshold'])
        result={'sample_points':len(pcd.points),'retained':int(mask.sum()),'available':bool(len(pcd.points))}
        if len(pcd.points):
            # Original global pitch; reject unreasonable transient grids before MC.
            pitch=float(np.ptp(np.asarray(pcd.points),axis=0).max()/150*cfg['pitch_scale'])
            shape=np.ceil(np.ptp(np.asarray(pcd.points),axis=0)/pitch).astype(int)+6
            if np.prod(shape)>35000000: raise MemoryError(f'Voxel grid cap exceeded: {shape}')
            o3d.io.write_point_cloud(str(out/'samples.ply'),pcd)
            scope['marching_cubes'](pcd,str(out),-1,'sampling',filtering_iters=2,pitch_scale=cfg['pitch_scale'])
            mesh=trimesh.load(out/'sampling_mesh_iteration.ply',process=False)
            result.update({'pitch_m':pitch,'voxel_grid_upper_shape':shape.tolist(),'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'resolution_le_5mm':pitch<=.005})
        else:result['reason']='upstream density filtering returned no samples'
    result.update({'population':args.population,'source':str(source_path),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'adapter_original_sha256':hashlib.sha256(Path('scripts/original_mesh_adapter.py').read_bytes()).hexdigest(),'selection_ast_substitutions':substitutions,'config':cfg,'mask_sha256':hashlib.sha256(Path(args.masks).read_bytes()).hexdigest(),'wall_s':time.perf_counter()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'normals_available':False,'opacity_and_covariance_unchanged':True})
    (out/'metadata.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result)
if __name__=='__main__':main()
