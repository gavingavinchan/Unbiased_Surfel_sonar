"""Resolution control: remesh the same retained density samples at the baseline pitch.
No support thresholds, density weights, sample coordinates, smoothing or GT bounds
change. Uses upstream marching_cubes unchanged, with computed pitch_scale.
"""
import ast,hashlib,json,os
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
baseline=json.loads(Path('outputs/opacity/metadata.json').read_text());pitch=baseline['pitch_m'];source_path=Path('/home/gavin/sonar-recon/third_party/sonar_splat/scripts/mesh_gaussian.py');source=source_path.read_text();tree=ast.parse(source);tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='marching_cubes'];scope=globals().copy();exec(compile(tree,str(source_path),'exec'),scope)
for name in ['opacity','all','support2','support3']:
 inp=Path('outputs')/name/'samples.ply';out=Path('outputs/common_pitch')/name;out.mkdir(parents=True,exist_ok=True);pcd=o3d.io.read_point_cloud(str(inp));scale=pitch*150/np.ptp(np.asarray(pcd.points),axis=0).max()
 scope['marching_cubes'](pcd,str(out),-1,'sampling',filtering_iters=2,pitch_scale=scale)
 mesh=trimesh.load(out/'sampling_mesh_iteration.ply',process=False)
 meta={'population':name,'pitch_m':pitch,'pitch_scale_to_existing_function':scale,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'sample_points':len(pcd.points),'input_samples':str(inp),'input_samples_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'density_threshold':.9,'new_samples_drawn':False,'source_function_unchanged':True,'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'available':True}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 if name=='opacity':assert (out/'sampling_mesh_iteration.ply').read_bytes()==Path('outputs/opacity/sampling_mesh_iteration.ply').read_bytes()
 print(name,meta)
