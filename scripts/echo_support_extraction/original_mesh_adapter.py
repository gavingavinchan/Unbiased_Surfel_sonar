#!/usr/bin/env python3
"""Run upstream sample-density + marching-cubes functions headlessly.
The released script imports absent 3DGS modules; this adapter supplies only the
GaussianModel data accessors and symmetric matrix/SH utilities it consumes.
Original function bodies are executed unchanged. Includes deterministic seeds.
"""
import argparse
import os
import ast
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import trimesh
import open3d as o3d
from plyfile import PlyData
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class GaussianModel:
    def __init__(self, path):
        ply = PlyData.read(path)['vertex']
        self.get_xyz = torch.tensor(np.column_stack([ply[k] for k in ['x','y','z']]), device='cuda')
        self.scales = torch.exp(torch.tensor(np.column_stack([ply[f'scale_{i}'] for i in range(3)]), device='cuda'))
        self.quats = torch.tensor(np.column_stack([ply[f'rot_{i}'] for i in range(4)]), device='cuda')
        self.get_opacity = torch.sigmoid(torch.tensor(np.asarray(ply['opacity']).copy(), device='cuda')).view(-1,1)
        self.get_features = torch.tensor(np.column_stack([ply[f'f_dc_{i}'] for i in range(3)]), device='cuda').view(-1,1,3)
    def get_covariance(self):
        from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci
        return _quat_scale_to_covar_preci(self.quats, self.scales, True, False, triu=True)[0]

def remake_symmetric(x):
    result = torch.zeros((len(x),3,3),device=x.device,dtype=x.dtype)
    result[:,0,0]=x[:,0];result[:,0,1]=result[:,1,0]=x[:,1];result[:,0,2]=result[:,2,0]=x[:,2]
    result[:,1,1]=x[:,3];result[:,1,2]=result[:,2,1]=x[:,4];result[:,2,2]=x[:,5]
    return result

def SH2RGB(x):
    return 0.28209479177387814*x + .5

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply-path',required=True)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--num-samples',type=int,default=10)
    parser.add_argument('--threshold',type=float,default=.9)
    parser.add_argument('--pitch-scale',type=float,default=.5)
    args=parser.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    source_path = Path('/home/gavin/sonar-recon/third_party/sonar_splat/scripts/mesh_gaussian.py')
    source = source_path.read_text()
    names = {'_batch_mahalanobis','CustomMultivariateNormal','query_gaussians','create_pc_gaussians','marching_cubes'}
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    # Avoid display operations while retaining algorithm bodies.
    o3d.visualization.draw_geometries = lambda *a,**k: None
    plt.show = lambda *a,**k: plt.close('all')
    scope = globals().copy();scope['CHUNK_SIZE']=600
    exec(compile(tree,str(source_path),'exec'),scope)
    out = Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    model = GaussianModel(args.ply_path)
    pcd = scope['create_pc_gaussians'](model,num_samples=args.num_samples,threshold=args.threshold)
    if not len(pcd.points):
        raise ValueError('Upstream density filtering returned no points')
    o3d.io.write_point_cloud(str(out/'samples.ply'),pcd)
    scope['marching_cubes'](pcd,str(out),-1,'sampling',filtering_iters=2,pitch_scale=args.pitch_scale)
    mesh = trimesh.load(out/'sampling_mesh_iteration.ply',process=False)
    (out/'metadata.json').write_text(json.dumps({'source':str(source_path),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'args':vars(args),'sample_points':len(pcd.points),'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'pitch_m':float(np.ptp(np.asarray(pcd.points),axis=0).max()/150*args.pitch_scale),'wall_s':time.time()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3},indent=2))
if __name__=='__main__': main()
