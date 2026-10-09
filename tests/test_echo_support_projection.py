"""Independent metric-frame echo projection tests, no GPU/data downloads."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
path=Path(__file__).resolve().parents[1]/'scripts/echo_support_extraction/support_filter.py'
spec=importlib.util.spec_from_file_location('echo_support',path);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
CFG={'range_origin_m':0.,'range_span_m':3.,'pixel_offset':.5,'azimuth_deg':120.,'elevation_deg':20.,'fov_numerical_tolerance_rad':1e-6,'pose_min_translation_m':.1,'pose_min_rotation_deg':5.,'range_tolerance_bins':1,'azimuth_tolerance_columns':2,'bright_threshold':.05}

def test_nonidentity_column_vector_projection_at_centres():
 pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[13,-17,31],degrees=True).as_matrix();pose[:3,3]=[.4,-.2,-.8]
 rows=np.array([20,55,120,180]);cols=np.array([30,80,130,220]);elev=np.deg2rad([-10,-3,4,10]);r=(rows+.5)*3/200;az=np.deg2rad(60-(cols+.5)*120/256)
 local=np.stack([r*np.cos(elev)*np.cos(az),r*np.cos(elev)*np.sin(az),r*np.sin(elev)],1);xyz=(pose[:3,:3]@local.T).T+pose[:3,3]
 row,col,valid=s.project(xyz,pose,CFG)
 assert np.array_equal(row,rows) and np.array_equal(col,cols) and valid.all()

def test_elevation_aperture_is_not_zero_elevation_ray():
 el=np.deg2rad([0,10,-10,10.01,-10.01]);xyz=np.stack([np.cos(el),np.zeros(5),np.sin(el)],1)
 assert s.project(xyz,np.eye(4),CFG)[2].tolist()==[True,True,True,False,False]

def test_duplicate_pose_cannot_count_twice():
 poses=np.repeat(np.eye(4)[None],10,0)
 assert s.separated_training_poses(poses,np.arange(10),CFG)==[0]

def test_neighbourhood_and_threshold_fixed():
 row,col=50,100;r=(row+.5)*3/200;az=np.deg2rad(60-(col+.5)*120/256);xyz=np.array([[r*np.cos(az),r*np.sin(az),0]])
 image=np.zeros((200,256));image[row+1,col+2]=.051
 assert s.echo_hits(xyz,np.eye(4),image,CFG)[0].item()
 image[row+1,col+2]=.05
 assert not s.echo_hits(xyz,np.eye(4),image,CFG)[0].item()
 image[row+2,col+3]=.2
 assert not s.echo_hits(xyz,np.eye(4),image,CFG)[0].item()
