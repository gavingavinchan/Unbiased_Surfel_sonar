#!/usr/bin/env python3
"""Count independent training-pose echo support, and export frozen selection masks.
No geometry labels, ROI bounds, clean images or covariance normals enter selection.
Supports sim_cube_v1's explicit FLU poses and centre sampled polar image grid.
"""
import argparse, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter, binary_dilation
from scipy.special import expit
from plyfile import PlyData


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def project(xyz, pose, cfg):
    local = (xyz-pose[:3,3]) @ pose[:3,:3]
    r = np.linalg.norm(local, axis=1)
    az = np.arctan2(local[:,1],local[:,0])
    el = np.arctan2(local[:,2],np.linalg.norm(local[:,:2],axis=1))
    eps=cfg['fov_numerical_tolerance_rad']
    valid=(local[:,0]>0)&(r>=cfg['range_origin_m'])&(r<cfg['range_origin_m']+cfg['range_span_m'])&(abs(az)<=np.deg2rad(cfg['azimuth_deg']/2)+eps)&(abs(el)<=np.deg2rad(cfg['elevation_deg']/2)+eps)
    row=np.floor((r-cfg['range_origin_m'])*200/cfg['range_span_m']-cfg['pixel_offset']+.5).astype(int)
    col=np.floor((cfg['azimuth_deg']/2-np.rad2deg(az))*256/cfg['azimuth_deg']-cfg['pixel_offset']+.5).astype(int)
    valid &= (row>=0)&(row<200)&(col>=0)&(col<256)
    return row,col,valid

def separated_training_poses(poses,ids,cfg):
    selected=[]
    for i in ids:
        if selected:
            prior=poses[selected]
            translation=np.linalg.norm(prior[:,:3,3]-poses[i,:3,3],axis=1)
            rr=np.einsum('nij,jk->nik',prior[:,:3,:3].transpose(0,2,1),poses[i,:3,:3])
            angle=np.rad2deg(np.arccos(np.clip((np.trace(rr,axis1=1,axis2=2)-1)/2,-1,1)))
            if np.any(translation<cfg['pose_min_translation_m']) or np.any(angle<cfg['pose_min_rotation_deg']): continue
        selected.append(int(i))
    return selected

def observed_image(data,i):
    with open(data/'Data'/f'{i:06d}.pkl','rb') as f: d=pickle.load(f)
    img=np.asarray(d['ImagingSonar'],dtype=float).copy()
    # Original013 loader masks 10 pixels at each border and range rows below7.
    img[:10]=0; img[-10:]=0; img[:,:10]=0; img[:,-10:]=0
    return img

def sizes(cfg): return (2*cfg['range_tolerance_bins']+1,2*cfg['azimuth_tolerance_columns']+1)

def echo_hits(xyz,pose,img,cfg):
    row,col,valid=project(xyz,pose,cfg)
    bright=maximum_filter(img,size=sizes(cfg),mode='constant',cval=0)>cfg['bright_threshold']
    hit=np.zeros(len(xyz),bool)
    hit[valid]=bright[row[valid],col[valid]]
    return hit,valid

def projection_scores(xyz,mask,poses,ids,data,cfg):
    tp=fp=fn=0; bright_projections=projections=0; rows=[]
    for i in ids:
        img=observed_image(data,i); bright=img>cfg['bright_threshold']
        row,col,valid=project(xyz[mask],poses[i],cfg)
        occupancy=np.zeros((200,256),bool);occupancy[row[valid],col[valid]]=True
        occupancy=binary_dilation(occupancy,structure=np.ones(sizes(cfg),bool))
        usable=np.ones((200,256),bool);usable[:10]=False;usable[-10:]=False;usable[:,:10]=False;usable[:,-10:]=False
        occupancy &= usable
        a=int((occupancy&bright).sum());b=int((occupancy&~bright).sum());c=int((~occupancy&bright).sum())
        tp+=a;fp+=b;fn+=c
        hit,v=echo_hits(xyz[mask],poses[i],img,cfg);bright_projections+=int(hit.sum());projections+=int(v.sum())
        rows.append({'frame':int(i),'tp':a,'fp':b,'fn':c})
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return {'pixel_precision':precision,'pixel_recall':recall,'pixel_f1':2*precision*recall/max(precision+recall,1e-30),'projected_means_bright_fraction':bright_projections/max(projections,1),'valid_mean_projections':projections,'frames':rows,'metric_kind':'binary echo-projection proxy; not rendered intensity or clean held-out score'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/extraction.json');p.add_argument('--out-dir',default='outputs/support');args=p.parse_args()
    start=time.perf_counter();cfg=json.loads(Path(args.config).read_text());data=Path(cfg['dataset']);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    ply=PlyData.read(cfg['checkpoint'])['vertex'];xyz=np.column_stack([ply[k] for k in ['x','y','z']]);opa=expit(np.asarray(ply['opacity']))
    poses=np.load(data/'world_T_sonar_flu.npy');train=np.flatnonzero(np.arange(len(poses))%8!=0);val=np.flatnonzero(np.arange(len(poses))%8==0)
    ids=separated_training_poses(poses,train,cfg);hits=[];visible=[]
    for i in ids:
        h,v=echo_hits(xyz,poses[i],observed_image(data,i),cfg);hits.append(h);visible.append(v)
    hits=np.stack(hits);visible=np.stack(visible);count=hits.sum(0)
    masks={'opacity':opa>.2,'all':np.ones(len(xyz),bool),**{f'support{k}':count>=k for k in cfg['support_settings']}}
    np.savez_compressed(out/'masks.npz',**masks,count=count,visible_count=visible.sum(0),support_frame_ids=np.array(ids),support_hits=hits)
    rows={}
    for name,mask in masks.items():
        rows[name]={'retained':int(mask.sum()),'fraction':float(mask.mean()),'opacity_quantiles':np.quantile(opa[mask],[0,.25,.5,.75,1]).tolist() if mask.any() else None,'retained_opacity_le_0_2':int((mask&(opa<=.2)).sum()),'normal_orientation':None,'orientation_reason':'no validated surface normals in 013; covariance axes not promoted to normals','training_projection':projection_scores(xyz,mask,poses,ids,data,cfg),'validation_projection':projection_scores(xyz,mask,poses,val,data,cfg)}
    selected=max([f'support{k}' for k in cfg['support_settings']],key=lambda x:rows[x]['validation_projection']['pixel_f1'])
    result={'config':cfg,'config_sha256':sha(args.config),'checkpoint_sha256':sha(cfg['checkpoint']),'total':len(xyz),'training_frames':train.tolist(),'validation_frames':val.tolist(),'independent_support_frames':ids,'count_histogram':np.bincount(count,minlength=len(ids)+1).tolist(),'populations':rows,'selected_by_transductive_validation_projection':selected,'seconds':time.perf_counter()-start,'no_gt_in_selection':True,'no_training':True,'normals_available':False}
    (out/'statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:{'retained':v['retained'],'val_f1':v['validation_projection']['pixel_f1']} for k,v in rows.items()});print('independent views',len(ids),'selected',selected)
if __name__=='__main__':main()
