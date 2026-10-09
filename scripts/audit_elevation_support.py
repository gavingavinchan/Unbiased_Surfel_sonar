"""Diagnostic-only native function audit; no training or dataset mutation.

AST extraction executes the actual native function bodies without running the
large debug runner's import-time configuration/CUDA setup. Full scalar finite
 differences and the intentionally frozen-association surrogate are separate.
"""
import ast
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import sonar_utils as su
from utils import elevation_stage1_helpers as s1
from utils import elevation_chunk4_helpers as c4
from utils import elevation_chunk5_helpers as c5


def load_native():
    ns = dict(torch=torch, np=np, math=math, F=torch.nn.functional, dataclass=dataclass,
              get_scaled_world_to_view_transform=su.get_scaled_world_to_view_transform,
              back_project_bins=su.back_project_bins)
    for mod in [s1, c4, c5]:
        ns.update({k:v for k,v in vars(mod).items() if not k.startswith('__')})
    for path, names in [
        ('gaussian_renderer/__init__.py', {'SonarProjection','_transform_world_points_to_sonar_frame','sonar_project_points','quaternion_to_normal','_sonar_config_values','_condition_sigma_2d','_quat_to_rotation_matrices','_jacobian_sigma_footprint_batch','compose_ray_binned_occlusion'}),
        ('debug_multiframe.py', {'build_pose_overlap_table','build_frame_stats_cache','sample_gt','resolve_stage1_overlap_neighbors','build_stage1_multiview_loglik','build_stage1_multiview_loglik_for_pixels','build_stage1_multiview_evidence_for_pixels','compute_chunk4_coupling_for_frame','compute_chunk4_support_observations_for_frame'})]:
        tree=ast.parse((ROOT/path).read_text())
        body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert len(body)==len(names)
        exec(compile(ast.Module(body=body,type_ignores=[]),str(ROOT/path),'exec'),ns)
    return ns


def rotation(a,b,c):
    ca,sa=math.cos(a),math.sin(a);cb,sb=math.cos(b),math.sin(b);cc,sc=math.cos(c),math.sin(c)
    return np.array([[cc,-sc,0],[sc,cc,0],[0,0,1.]]) @ np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]]) @ np.array([[1.,0,0],[0,ca,-sa],[0,sa,ca]])


def camera(M):
    return NS(world_view_transform=torch.tensor(M.T,dtype=torch.float64),image_height=200,image_width=256)


def fd(fun,x,h=1e-5):
    vals=[]
    for i in range(x.numel()):
        d=torch.zeros_like(x);d.flatten()[i]=h
        vals.append(float((fun(x+d)-fun(x-d))/(2*h)))
    return torch.tensor(vals,dtype=x.dtype).reshape(x.shape)


def rel(a,b):
    return float(torch.linalg.vector_norm(a-b)/torch.maximum(torch.linalg.vector_norm(a),torch.linalg.vector_norm(b)).clamp_min(1e-10))


def smooth_gradients(count=128):
    torch.manual_seed(20261009);torch.set_num_threads(1)
    records=[];nonfinite=0;native=load_native()
    for i in range(count):
        # Generic smooth surrogate: no visibility, matching, Huber or support boundary.
        pts=torch.randn(3,3,dtype=torch.float64)*.2
        if i%2: pts=.8*pts/pts.norm(dim=-1,keepdim=True)
        else: pts[:,2]=.8
        xyz=(pts+.02+torch.rand_like(pts)*.006).requires_grad_()
        weights=torch.tensor([.45,.73,.89],dtype=torch.float64)
        idx=torch.arange(3);valid=torch.ones(3,dtype=torch.bool)
        fun=lambda v:c4.reduce_coupling_loss(pts,v,idx,weights,valid,.09)
        g=torch.autograd.grad(fun(xyz),xyz)[0];f=fd(fun,xyz.detach())
        logits=torch.randn(3,7,dtype=torch.float64,requires_grad=True)*.2
        ll=-torch.rand(3,7,dtype=torch.float64)*3
        mask=torch.ones(3,7,dtype=torch.bool);mask[-1]=False
        likelihood=lambda v:s1.run_stage1_likelihood_step(v,ll,mask,'active',.4,.03,1.,.8,1.,1e-6)['stage1_total_loss']
        lg=torch.autograd.grad(likelihood(logits),logits)[0];lf=fd(likelihood,logits.detach())
        # Quaternion tangent rotation and normal objective; target detached and fixed.
        q=torch.tensor([.9,.14+.001*i,-.21,.1],dtype=torch.float64,requires_grad=True)
        target=torch.tensor([[.2,.8,.55]],dtype=torch.float64);target=target/target.norm()
        def normal_loss(v):
            n=native['quaternion_to_normal'](v[None])
            return c5.compute_normal_supervision_loss(n_quat=n,n_expected=target)
        ng=torch.autograd.grad(normal_loss(q),q)[0];nf=fd(normal_loss,q.detach())
        nonfinite+=sum(int((~torch.isfinite(v)).sum()) for v in [g,f,lg,lf,ng,nf])
        records.append({'case':i,'coupling_frozen_relative_error':rel(g,f),'likelihood_logits_relative_error':rel(lg,lf),'normal_rotation_relative_error':rel(ng,nf)})
    return {'configurations':count,'nonfinite_components':nonfinite,'max_relative_errors':{k:max(r[k] for r in records) for k in records[0] if k!='case'},'cases':records}


def association_audit(ns):
    # Two matches with different residuals: fresh weights really affect normalization.
    p=torch.tensor([[0.,0.,1.4],[.2,0.,1.6]],dtype=torch.float64)
    x=(p+torch.tensor([[.004,.025,.006],[-.003,.04,-.01]],dtype=torch.float64)).requires_grad_()
    cfg=su.SonarConfig(range_min=.2,pixel_center_offset=0,device='cpu');cam=camera(np.eye(4))
    pp=ns['sonar_project_points'](p,cam,cfg)
    def assoc(v):
        sp=ns['sonar_project_points'](v,cam,cfg)
        return c4.associate_expected_points_to_surfels(pp.row,pp.col,pp.range_vals,pp.valid,sp.row,sp.col,sp.range_vals,sp.valid,5.,.1,2.,.05,.1)
    idx,w,valid=assoc(x)
    def full(v):
        a,b,c=assoc(v);return c4.reduce_coupling_loss(p,v,a,b,c,.03)
    frozen=lambda v:c4.reduce_coupling_loss(p,v,idx,w,valid,.03)
    g=torch.autograd.grad(full(x),x)[0]
    gf=fd(full,x.detach(),1e-4);gg=fd(frozen,x.detach(),1e-4)
    # Identical range/azimuth, competing elevation: association sees an exact tie.
    r=1.4;e=math.radians(6);pair=torch.tensor([[0.,r*math.sin(e),r*math.cos(e)],[0.,-r*math.sin(e),r*math.cos(e)]],dtype=torch.float64)
    pr=ns['sonar_project_points'](pair,cam,cfg)
    ii,ww,vv=c4.associate_expected_points_to_surfels(pr.row[1:],pr.col[1:],pr.range_vals[1:],pr.valid[1:],pr.row,pr.col,pr.range_vals,pr.valid,5.,.1,2.,.05,.1)
    return {'full_recomputed_weight_relative_error':rel(g,gf),'frozen_weight_relative_error':rel(g,gg),'weights_require_grad':w.requires_grad,'match_indices':idx.tolist(),'grad_autograd':g.tolist(),'grad_full_fd':gf.tolist(),'elevation_tie':{'chosen_index':int(ii[0]),'correct_index':1,'wrong_match_distance_m':float((pair[ii[0]]-pair[1]).norm()),'same_range_azimuth':bool(torch.allclose(pr.row[:1],pr.row[1:]) and torch.allclose(pr.col[:1],pr.col[1:]))},'interpretation':'Autograd is a frozen-association/weight surrogate; not derivative of recomputed scalar.'}


def extrinsic_cases(ns,count=128):
    cfg=su.SonarConfig(range_min=.2,pixel_center_offset=0,device='cpu')
    bins=torch.linspace(-.14,.14,7)
    rows=torch.tensor([85,108]);cols=torch.tensor([110,146])
    cfg4=NS(couple_max_candidates=0,couple_max_pix_err=5.,couple_max_depth_err=.1,couple_sigma_pix=2.,couple_sigma_depth=.05,couple_min_w=.1,couple_huber_delta=.03)
    cfg1=NS(lik_invalid_mode='neutral',lik_log_floor=-10.,overlap_topk_use=1,lik_log_eps=1e-6,lik_use_frame_reliability=False,lik_min_support=1e-6)
    yy,xx=torch.meshgrid(torch.arange(200),torch.arange(256),indexing='ij');im=(.1+.6*yy/200+.2*xx/256).float()
    rec=[]
    for i in range(count):
        M=np.eye(4);M[:3,:3]=rotation(.21+.001*i,-.31,.43);M[:3,3]=[.3,-.2,.4]
        E=np.eye(4);E[:3,:3]=rotation(-.09,.11,.06);E[:3,3]=[.04,-.12,.07]
        cam=camera(M);ex=su.SonarExtrinsic(device='cpu',camera_to_sonar=E)
        sonar=camera(E@M)
        bank={'a':{'rows':rows,'cols':cols}}
        probs=torch.tensor([[0.,0.,0.,0.,0.,1.,0.],[0.,1.,0.,0.,0.,0.,0.]],requires_grad=True)
        pts=su.back_project_bins(0,rows,cols,bins,cameras=[sonar],sonar_config=cfg,scale_factor=None)
        expected=(probs.detach()[...,None]*pts).sum(1)
        xyz=(expected+torch.tensor([[.002,.007,.003],[-.004,.005,-.002]])).detach().requires_grad_()
        g=NS(get_xyz=xyz,_rotation=torch.randn(2,4,requires_grad=True),_scaling=torch.zeros(2,2,requires_grad=True),_opacity=torch.zeros(2,1,requires_grad=True))
        args=dict(frame_idx=0,frame_key='a',render_pkg={'visibility_filter':torch.ones(2,dtype=torch.bool)},gaussians=g,sonar_config=cfg,sonar_scale_factor=None,pixel_bank=bank,p_post_frame=probs,elev_angle_bins=bins,chunk4_cfg=cfg4)
        baked=ns['compute_chunk4_coupling_for_frame'](training_frames=[sonar],**args)
        mounted=ns['compute_chunk4_coupling_for_frame'](training_frames=[cam],sonar_extrinsic=ex,**args)
        omitted=ns['compute_chunk4_coupling_for_frame'](training_frames=[cam],**args)
        grad=torch.autograd.grad(mounted['loss'],[xyz,probs,g._rotation,g._scaling,g._opacity],allow_unused=True)
        # Likelihood-equivalence through both public wrappers and neighbor transform.
        common=dict(frame_idx=0,frame_key='a',frame_key_to_index={'a':0,'b':1},overlap_table={'a':{'topk_use':['b']}},pixel_bank=bank,gt_frame_cache={'b':{'gt_gray':im}},frame_stats_cache={'b':{'p_lo':0.,'p_hi':1.,'reliability':1.}},elev_angle_bins=bins,sonar_config=cfg,sonar_scale_factor=None,cfg=cfg1)
        M2=M.copy();M2[:3,3]+=[.025,.012,-.008]
        l1,sup1=ns['build_stage1_multiview_loglik'](training_frames=[sonar,camera(E@M2)],**common)
        l2,sup2=ns['build_stage1_multiview_loglik'](training_frames=[cam,camera(M2)],sonar_extrinsic=ex,**common)
        # Mixed positive/negative elevations are tiny competing-return fixtures;
        # no GT surface is used to construct training targets.
        obs_args=dict(frame_idx=0,gaussians=g,render_pkg=args['render_pkg'],rendered=im[None].repeat(3,1,1),gt_image=im[None].repeat(3,1,1),sonar_config=cfg,sonar_scale_factor=None,support_residual_thresh=.01)
        o1=ns['compute_chunk4_support_observations_for_frame'](training_frames=[sonar],**obs_args)
        o2=ns['compute_chunk4_support_observations_for_frame'](training_frames=[cam],sonar_extrinsic=ex,**obs_args)
        rec.append({'case':i,'coupling_loss_error':abs(float(baked['loss']-mounted['loss'])),'omitted_mount_loss_error':abs(float(baked['loss']-omitted['loss'])),'likelihood_max_error':float((l1-l2).abs().max()),'support_equal':torch.equal(sup1,sup2) and torch.equal(o1['support_idx'],o2['support_idx']),'gradient_norms':{n:None if v is None else float(v.norm()) for n,v in zip(['xyz','posterior','rotation','scaling','opacity'],grad)},'nonfinite_gradients':sum(int((~torch.isfinite(v)).sum()) for v in grad if v is not None),'match_count':mounted['match_count']})
    return {'configurations':count,'max_coupling_loss_error':max(r['coupling_loss_error'] for r in rec),'max_likelihood_error':max(r['likelihood_max_error'] for r in rec),'max_omitted_mount_loss_error':max(r['omitted_mount_loss_error'] for r in rec),'support_mismatches':sum(not r['support_equal'] for r in rec),'nonfinite_components':sum(r['nonfinite_gradients'] for r in rec),'cases':rec}


def footprint_gradients(ns,count=128):
    """Actual tangent covariance + compositing, away from visibility/sort boundaries."""
    cfg=su.SonarConfig(range_min=.2,pixel_center_offset=0,device='cpu')
    M=np.eye(4);M[:3,:3]=rotation(.21,-.31,.43);M[:3,3]=[.3,-.2,.4]
    E=np.eye(4);E[:3,:3]=rotation(-.09,.11,.06);E[:3,3]=[.04,-.12,.07]
    cam=camera(M);ex=su.SonarExtrinsic(device='cpu',camera_to_sonar=E)
    records=[]
    for i in range(count):
        local=np.array([.1+.0002*i,.07,1.8])
        world=(E@M)[:3,:3].T@(local-(E@M)[:3,3])
        x=torch.tensor(list(world)+[.9,.14+.0002*i,-.21,.1]+[math.log(.028),math.log(.041)]+[-.4,.2]+[0.],dtype=torch.float64,requires_grad=True)
        def fun(v):
            sf=NS(scale=v[11].exp())
            loc,w2v=ns['_transform_world_points_to_sonar_frame'](v[:3][None],cam,sf,ex)
            cov=ns['_jacobian_sigma_footprint_batch'](loc,v[7:9].exp()[None]*sf.scale,v[3:7][None],cfg,w2v[:3,:3].T)
            footprint=(cov*cov.new_tensor([[[.7,.11],[.11,.3]]])).sum()
            alpha=v[9:11].sigmoid()
            # Fixed ray/depth order: differentiable opacity attenuation only.
            returns=ns['compose_ray_binned_occlusion'](torch.tensor([0,0]),v.new_tensor([1.7,1.9]),alpha,alpha*v.new_tensor([.6,.8]),1)['ray_returns'].sum()
            return footprint*returns
        ag=torch.autograd.grad(fun(x),x)[0];fg=fd(fun,x.detach(),1e-5)
        groups={'xyz':slice(0,3),'rotation':slice(3,7),'log_surfel_scaling':slice(7,9),'opacity_logits':slice(9,11),'log_metric_scale':slice(11,12)}
        records.append({'case':i,'relative_errors':{n:rel(ag[sl],fg[sl]) for n,sl in groups.items()},'nonfinite':int((~torch.isfinite(ag)).sum()+(~torch.isfinite(fg)).sum())})
    return {'configurations':count,'max_relative_errors':{n:max(r['relative_errors'][n] for r in records) for n in groups},'nonfinite_components':sum(r['nonfinite'] for r in records),'cases':records}


def modes():
    logits=torch.tensor([[.1,.2,-.1]],requires_grad=True);ll=torch.tensor([[-1.,-3.,-.2]]);mask=torch.ones(1,3,dtype=torch.bool)
    out={}
    for mode in ['off','shadow','active']:
        o=s1.run_stage1_likelihood_step(logits,ll,mask,mode,.4,.03,1.,1.,1.,1e-6)
        loss=o['stage1_total_loss'];out[mode]={'likelihood_loss':float(loss),'likelihood_requires_grad':loss.requires_grad,'coupling_weight_enabled':c4.mode_enables_weighted_coupling(mode),'support_hard_prune_enabled':c4.mode_enables_hard_prune(mode)}
    return out


def run():
    ns=load_native()
    return {'scope':'diagnostic only; no training or renderer image gradient certification','smooth_gradients':smooth_gradients(),'footprint_gradients':footprint_gradients(ns),'association':association_audit(ns),'extrinsics':extrinsic_cases(ns),'modes':modes(),'posterior_mean':{'range_m':2.,'elevation_modes_deg':[-8.,8.],'equal_mixture_radial_shortening_m':2*(1-math.cos(math.radians(8))),'meaning':'Cartesian mean lies inside both equal-range hypotheses; not an observed return.'}}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=run();Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='cases'} if isinstance(v,dict) else v for k,v in out.items()},indent=2))
