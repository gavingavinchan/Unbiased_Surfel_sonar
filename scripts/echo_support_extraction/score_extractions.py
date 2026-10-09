#!/usr/bin/env python3
"""Exact retained metrics plus evaluation-only ROI, components and 12-edge coverage."""
import argparse,importlib.util,json,hashlib,time
from pathlib import Path
import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree
from support_filter import echo_hits,observed_image
spec=importlib.util.spec_from_file_location('metrics','/home/gavin/sonar-recon/tools/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def crop(mesh,bounds):
    for axis in range(3):
        for side in range(2):
            n=np.zeros(3);n[axis]=1 if side==0 else -1;p=np.zeros(3);p[axis]=bounds[side,axis];mesh=mesh.slice_plane(p,n,cap=False)
    return mesh

def topology(mesh):
    comps=mesh.split(only_watertight=False)
    return {'components':len(comps),'watertight':bool(mesh.is_watertight),'euler_number':int(mesh.euler_number),'area_m2':float(mesh.area),'component_triangles':sorted([len(c.faces) for c in comps],reverse=True),'watertight_components':sum(c.is_watertight for c in comps)}

def edge_coverage(gp,d,scene):
    corners=np.array(scene['corner_centres_m']);ds=[];ts=[]
    for a,b in scene['edges']:
        axis=corners[b]-corners[a];t=(gp-corners[a])@axis/(axis@axis);q=corners[a]+np.clip(t,0,1)[:,None]*axis
        ds.append(np.linalg.norm(gp-q,axis=1));ts.append(t)
    assignment=np.argmin(ds,axis=0);rows=[]
    for i in range(12):
        mask=(assignment==i)&(ts[i]>=.1)&(ts[i]<=.9)
        vals=d[mask];coverage=float((vals<=.02).mean()) if len(vals) else None
        rows.append({'edge':i,'interior_gt_samples':int(mask.sum()),'coverage_20mm':coverage,'mean_mm':float(vals.mean()*1000) if len(vals) else None,'observed_ge_50pct':bool(coverage is not None and coverage>=.5)})
    return {'definition':'nearest GT edge centreline; only middle80% of edge; observed when >=50% GT surface samples within20mm of actual mesh','edges_observed_ge_50pct':sum(r['observed_ge_50pct'] for r in rows),'edges_with_any_20mm_sample':sum(r['coverage_20mm'] is not None and r['coverage_20mm']>0 for r in rows),'edges':rows}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--population',choices=['opacity','all','support2','support3'],required=True);p.add_argument('--mesh-dir');args=p.parse_args();start=time.perf_counter()
    cfg=json.loads(Path('configs/extraction.json').read_text());data=Path(cfg['dataset']);out=Path(args.mesh_dir) if args.mesh_dir else Path('outputs')/args.population
    mesh=m.load(out/'sampling_mesh_iteration.ply');gt=m.load(data/'gt_cube.ply');scene=json.loads((data/'scene.json').read_text());gp=m.sample(gt,100000,20261010)
    result={}
    for label,pred in [('full',mesh),('roi',crop(mesh,gt.bounds+np.array([[-.05]*3,[.05]*3])))]:
        if not len(pred.faces):result[label]={'available':False,'reason':'empty evaluation crop'};continue
        if label=='roi':pred.export(out/'cube_roi.ply')
        score=m.compare(pred,gt,n=100000,seed=20261009);score['topology']=topology(pred);score['strut_thickness']=m.thickness(pred,scene)
        score['strut_coverage']=edge_coverage(gp,m.distance(gp,pred),scene)
        score['inputs']={k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in [('pred',out/('cube_roi.ply' if label=='roi' else 'sampling_mesh_iteration.ply')),('gt',data/'gt_cube.ply')]}
        if label=='roi':score['roi_definition']='GT bounding box plus50mm; evaluation only; uncapped plane clipping, no alignment'
        (out/f'metrics_{label}.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n');result[label]=score
        print(args.population,label,{k:score[k] for k in ['pred_to_gt_mean_mm','gt_to_pred_mean_mm','chamfer_mean_mm','completeness']},'edges',score['strut_coverage']['edges_observed_ge_50pct'],'components',score['topology']['components'],flush=True)
    # No GT is read by this observed support computation on area samples.
    pp=m.sample(mesh,100000,20261009);poses=np.load(data/'world_T_sonar_flu.npy');ids=np.load('outputs/support/masks.npz')['support_frame_ids'];counts=np.zeros(len(pp),int)
    for i in ids:counts+=echo_hits(pp,poses[i],observed_image(data,int(i)),cfg)[0]
    result['sampled_mesh_echo_support']={'samples':len(pp),'seed':20261009,'fraction_lt_2_distinct_pose_returns':float((counts<2).mean()),'fraction_lt_3_distinct_pose_returns':float((counts<3).mean()),'count_histogram':np.bincount(counts,minlength=len(ids)+1).tolist(),'poisson_closure_quantification':None,'reason':'no validated surfel normals or Poisson comparison available; MC support fraction is not a Poisson closure metric'}
    # Centre scores are separate from surface metrics, without GT-ROI selection.
    ply=PlyData.read(cfg['checkpoint'])['vertex'];xyz=np.column_stack([ply[k] for k in ['x','y','z']]);mask=np.load('outputs/support/masks.npz')[args.population]
    d=m.distance(xyz[mask],gt);back=cKDTree(xyz[mask]).query(gp,workers=4)[0]
    result['centres']={'count':int(mask.sum()),'mean_to_gt_mm':float(d.mean()*1000),'p95_to_gt_mm':float(np.quantile(d,.95)*1000),'accuracy_20mm':float((d<=.02).mean()),'reverse_mean_mm':float(back.mean()*1000),'reverse_coverage_20mm':float((back<=.02).mean()),'metric_kind':'frozen Gaussian means to PVC mesh; not oriented surface accuracy'}
    result['seconds']=time.perf_counter()-start
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
