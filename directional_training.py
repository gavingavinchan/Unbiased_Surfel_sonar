"""Bounded reference Session following native_sonar_training's optimization.

CPU and CUDA use identical new ray-plane equations, not the compiled PR6 splat.
Reuse native recipe, sampler, restricted full-state serializer and loss utilities.
No native controller/admission is provided by this module.
"""
from argparse import ArgumentParser
from pathlib import Path
import random
import numpy as np
import torch
from arguments import OptimizationParams
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import RGB2SH, SH2RGB
from utils import atomic_training as atomic
from directional_sonar import render,rotations


class Session:
    def __init__(self,cfg):
        self.cfg=cfg;self.device=cfg.get('device','cuda');torch.set_num_threads(1)
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        if self.device=='cuda':
            torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        parser=ArgumentParser();op=OptimizationParams(parser);self.opt=op.extract(parser.parse_args([]))
        self.opt.iterations=cfg['steps'];self.opt.position_lr_max_steps=cfg['steps'];self.opt.densify_until_iter=0
        a=np.load(cfg['init_file']);self.model=torch.nn.ParameterDict()
        for k,v in [('xyz',a['xyz']),('rotation',a['quat']),('scaling',np.log(a['scale'])),('opacity',np.log(a['opacity']/(1-a['opacity'])).reshape(-1,1))]:
            self.model[k]=torch.nn.Parameter(torch.tensor(v,device=self.device,dtype=torch.float32))
        self.model['f_dc']=torch.nn.Parameter(RGB2SH(torch.full((len(a['xyz']),1),.5,device=self.device)))
        opt=self.opt
        rates=dict(xyz=opt.position_lr_init,f_dc=opt.feature_lr,opacity=opt.opacity_lr,scaling=opt.scaling_lr,rotation=opt.rotation_lr)
        self.optimizer=torch.optim.Adam([dict(params=[v],lr=rates[k],name=k)for k,v in self.model.items()],lr=0.,eps=1e-15)
        self.schedule=get_expon_lr_func(lr_init=opt.position_lr_init,lr_final=opt.position_lr_final,lr_delay_mult=opt.position_lr_delay_mult,max_steps=cfg['steps'])
        self.data=np.load(cfg['input_file']);self.images=self.data['images'];self.poses=self.data['poses']
        train=cfg['frame_indices'];self.background=torch.tensor(self.images[train,13:].mean((0,2))/255,device=self.device,dtype=torch.float32)
        self.background=torch.cat([self.background.new_zeros(13),self.background])
        self.remaining=[];self.step_number=0;self.last_loss=None

    def predict(self,i,background=True):
        m=self.model
        return render(m['xyz'],m['rotation'],m['scaling'].exp(),m['opacity'].sigmoid(),SH2RGB(m['f_dc']).clamp(0,1),
            torch.tensor(self.poses[i],device=self.device,dtype=torch.float32),
            elevations=self.cfg['elevation_samples'],azimuth_subsamples=2,background=self.background if background else None)

    def step(self):
        if not self.remaining:self.remaining=list(self.cfg['frame_indices'])
        i=self.remaining.pop(random.randrange(len(self.remaining)));self.step_number+=1
        for group in self.optimizer.param_groups:
            if group['name']=='xyz':group['lr']=float(self.schedule(self.step_number))
        target=torch.tensor(self.images[i]/255,device=self.device,dtype=torch.float32)[None]
        pred=self.predict(i)[None];mask=torch.ones_like(pred);mask[:,:13]=0
        pred=pred*mask;target=target*mask
        loss=.8*l1_loss(pred,target)+.2*(1-ssim(pred,target))
        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
        loss.backward()
        for x in self.model.values():
            if x.grad is not None and not bool(torch.isfinite(x.grad).all()):raise ValueError('nonfinite gradient')
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.model['scaling'].clamp_(np.log(.002),np.log(.1))
            center=torch.tensor(self.cfg['center'],device=self.device,dtype=torch.float32)
            self.model['xyz'].copy_(torch.maximum(torch.minimum(self.model['xyz'],center+.75),center-.75))
        self.last_loss=float(loss.detach())
        return self.last_loss

    def snapshot(self):
        return dict(step=self.step_number,model=self.model.state_dict(),optimizers={'gaussians':self.optimizer.state_dict()},
            schedulers={'xyz_max_steps':self.cfg['steps']},sampler={'remaining':self.remaining},
            strategy={'name':'directional-surface-v1','last_loss':self.last_loss},scale={'metric':1.,'learned':False},rng=atomic.rng_state())

    def restore(self,state):
        if state['strategy']['name']!='directional-surface-v1' or state['schedulers']['xyz_max_steps']!=self.cfg['steps']:raise ValueError('strategy/schedule differs')
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizers']['gaussians'])
        self.step_number=state['step'];self.remaining=list(state['sampler']['remaining']);self.last_loss=state['strategy']['last_loss'];atomic.restore_rng(state['rng'])

    def export(self,path):
        m=self.model
        np.savez_compressed(path,xyz=m['xyz'].detach().cpu().numpy(),quat=m['rotation'].detach().cpu().numpy(),
            normal=rotations(m['rotation'])[:,:,2].detach().cpu().numpy(),scale=m['scaling'].exp().detach().cpu().numpy(),
            opacity=m['opacity'].sigmoid().detach().cpu().numpy(),color=SH2RGB(m['f_dc']).clamp(0,1).detach().cpu().numpy())
