"""Frozen original013 rasterizer image checks; no optimization or pose updates.
Scores are transductive:013 initialization included the validation views.
"""
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/gavin/sonar-recon/third_party/sonar_splat');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'examples'))
spec=importlib.util.spec_from_file_location('ss_trainer',ROOT/'examples/sonar_simple_trainer.py');trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)
cfg=trainer.Config(data_dir='/home/gavin/sonar-recon/data/sim_cube_v1/sonarsplat',result_dir='outputs/image_checks',disable_viewer=True,skip_frames=1,img_threshold=0,init_threshold=.05,init_num_pts=12000,range_clear_start=200,range_clear_end=7,init_type='predefined',init_scale=.01)
# Reuse cached LPIPS weights read-only; no downloads or new pretrained weights.
torch.hub.set_dir('/home/gavin/sonar-recon/experiments/013-sonarsplat-on-sim/.cache/torch/hub')
runner=trainer.Runner(0,0,1,cfg)
ckpt_path='/home/gavin/sonar-recon/experiments/013-sonarsplat-on-sim/outputs/bright_full/ckpts/ckpt_39999_rank0.pt';ckpt=torch.load(ckpt_path,map_location='cuda',weights_only=False)
masks=np.load('outputs/support/masks.npz');n=len(masks['all']);result={};start=time.perf_counter()
# PLY acoustic means and checkpoint ordering must match before using masks.
from plyfile import PlyData
ply=PlyData.read('/home/gavin/sonar-recon/experiments/013-sonarsplat-on-sim/outputs/bright_full/renders/output_step39999.ply')['vertex'];xyz=np.column_stack([ply[k] for k in ['x','y','z']]);assert np.array_equal(ckpt['splats']['means'].cpu().numpy(),xyz)
fixed=[0,144,296,448]
with torch.no_grad():
 for name in ['all','opacity','support2','support3']:
  mask=torch.tensor(masks[name],device='cuda',dtype=torch.bool)
  for k,v in ckpt['splats'].items():runner.splats[k].data=v[mask] if len(v)==n else v
  rows=[]
  for i,data in enumerate(runner.valset):
   real=data['image'].cuda()[None];pred,meta=runner.rasterize_splats(data['camtoworld'].cuda()[None],data['K'].cuda()[None],width=200,height=256,near_plane=data['near_plane'],far_plane=data['far_plane'],sh_degree=3)
   fg=real>.05;bg=~fg;err=abs(pred-real)
   row={'frame':int(runner.valset.indices[i]),'l1':float(err.mean()),'foreground_l1':float(err[fg].mean()),'background_l1':float(err[bg].mean()),'mse':float(((pred-real)**2).mean()),'black_l1':float(real.mean()),'black_foreground_l1':float(real[fg].mean()),'black_background_l1':float(real[bg].mean()),'predicted_mass':float(pred.sum()),'target_mass':float(real.sum()),'background_predicted_mass':float(pred[bg].sum())};rows.append(row)
   if row['frame'] in fixed:
    out=Path('outputs/image_checks')/name;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/f'frame_{row["frame"]:06d}.npz',gt=real.squeeze().T.cpu().numpy(),pred=pred.squeeze().T.cpu().numpy())
  result[name]={'retained':int(mask.sum()),'rows':rows,**{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='frame'}}
  print(name,{k:v for k,v in result[name].items() if k!='rows'},flush=True)
result['metadata']={'checkpoint':ckpt_path,'checkpoint_step':ckpt['step'],'renderer_source':str(ROOT/'examples/sonar_simple_trainer.py'),'metric_population':'75 transductive validation views, float unthresholded data and original border mask','selection_changes_only':True,'training_steps':0,'init_control_available':False,'init_control_reason':'frozen-final-checkpoint extraction; exact initialization checkpoint not provided','wall_s':time.perf_counter()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
Path('outputs/image_checks/metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');runner.writer.close()
