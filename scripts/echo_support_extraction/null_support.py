"""No-GT specificity diagnostic: destroy echo spatial correspondence per training view.
This shuffled-image null is not a new candidate, a physical noise model, or a
threshold-selection score. It preserves observed per-frame usable-pixel values.
"""
import json,time
from pathlib import Path
import numpy as np
from plyfile import PlyData
from support_filter import observed_image,echo_hits
cfg=json.loads(Path('configs/extraction.json').read_text());data=Path(cfg['dataset']);masks=np.load('outputs/support/masks.npz');ids=masks['support_frame_ids'];poses=np.load(data/'world_T_sonar_flu.npy');ply=PlyData.read(cfg['checkpoint'])['vertex'];xyz=np.column_stack([ply[k] for k in ['x','y','z']]);counts=np.zeros(len(xyz),int);rng=np.random.default_rng(20261009);start=time.perf_counter()
usable=np.zeros((200,256),bool);usable[10:-10,10:-10]=True
for i in ids:
 img=observed_image(data,int(i));img[usable]=rng.permutation(img[usable]);counts+=echo_hits(xyz,poses[i],img,cfg)[0]
result={'seed':20261009,'independent_training_views':ids.tolist(),'description':'independent per-frame shuffle of observed intensities inside original usable border; noGT; destroys correspondence but preserves frame histograms','not_a_candidate':True,'seconds':time.perf_counter()-start,'populations':{}}
for k in [2,3]:
 actual=masks[f'support{k}'];null=counts>=k;result['populations'][f'support{k}']={'actual_retained':int(actual.sum()),'null_retained':int(null.sum()),'null_retained_fraction':float(null.mean()),'actual_retained_also_null_supported':int((actual&null).sum()),'null_support_fraction_of_actual_retained':float(null[actual].mean())}
np.savez_compressed('outputs/support/null_counts.npz',count=counts);Path('outputs/support/null_statistics.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
