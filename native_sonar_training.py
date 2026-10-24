"""Fixed-density sonar training session for the guarded native adapter.

Uses the fork's actual GaussianModel, optimizer schedule and sonar renderer.
No Scene camera loader, GUI, tensorboard, densification or learned metric scale.
Those strategies must get their own state audit before this class admits them.
Every step includes its optimizer update, including the last budgeted step.
cfg['loss'] (optional, default plain 0.8 L1 + 0.2 SSIM): {'name': 'return_weighted', 'threshold': t, 'dilate': 1}
weights return and background pixels equally (utils.loss_utils.return_weighted_loss).
"""
from argparse import ArgumentParser
from types import SimpleNamespace
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_sonar, quaternion_to_normal
from utils.sonar_utils import build_debug_sonar_config
from utils.sh_utils import RGB2SH
from utils.loss_utils import l1_loss, ssim, return_weighted_loss
from utils import atomic_training as atomic


class Session:
    def __init__(self, cfg):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        self.cfg = cfg
        parser = ArgumentParser(); op = OptimizationParams(parser)
        self.opt = op.extract(parser.parse_args([]))
        self.opt.iterations = cfg['steps']
        self.opt.position_lr_max_steps = cfg['steps']
        self.opt.densify_until_iter = 0
        self.model = GaussianModel(0)
        init = np.load(cfg['init_file'], allow_pickle=False)
        # point_limit is an upper bound; size every per-point buffer to the actual init.
        n = min(cfg['point_limit'], len(init['xyz']))
        xyz = torch.as_tensor(init['xyz'][:n], device='cuda').float()
        q = torch.as_tensor(init['quat'][:n], device='cuda').float()
        scales = torch.as_tensor(init['scale'][:n], device='cuda').float()
        opacity = torch.as_tensor(init['opacity'][:n], device='cuda').float().reshape(-1, 1)
        colors = torch.full_like(xyz, .5)
        self.model._xyz = torch.nn.Parameter(xyz.clone())
        self.model._features_dc = torch.nn.Parameter(RGB2SH(colors)[:, None, :])
        self.model._features_rest = torch.nn.Parameter(xyz.new_empty((n, 0, 3)))
        self.model._rotation = torch.nn.Parameter(q.clone())
        self.model._scaling = torch.nn.Parameter(scales.log())
        self.model._opacity = torch.nn.Parameter(torch.logit(opacity))
        self.model.max_radii2D = xyz.new_zeros(n)
        self.model.spatial_lr_scale = 1.
        self.model.training_setup(self.opt)
        self.sonar = build_debug_sonar_config({'SONAR_RANGE_ORIGIN':'0', 'SONAR_RANGE_SPAN':'3',
            'SONAR_PIXEL_CENTER_OFFSET':'.5', 'SONAR_POSE_MODE':'poses_are_sonar'}, device='cuda')
        poses = np.load(Path(cfg['dataset'])/'world_T_sonar_flu.npy', allow_pickle=False)
        basis = np.eye(4); basis[:3,:3] = [[0,0,1],[-1,0,0],[0,-1,0]]
        self.cameras = {}
        for i in cfg['frame_indices']:
            self.cameras[i] = SimpleNamespace(image_height=200, image_width=256,
                world_view_transform=torch.as_tensor(np.linalg.inv(poses[i]@basis).T, device='cuda'))
        self.remaining = []
        self.step_number = 0
        self.last_loss = None

    def step(self):
        if not self.remaining: self.remaining = list(self.cfg['frame_indices'])
        i = self.remaining.pop(random.randrange(len(self.remaining)))
        camera = self.cameras[i]
        image = np.array(Image.open(Path(self.cfg['dataset'])/f'frames/{i:06d}.png'), copy=True)
        target = torch.as_tensor(image, device='cuda').float().permute(2,0,1)/255
        target[:, :10] = 0
        self.step_number += 1
        self.model.update_learning_rate(self.step_number)
        pkg = render_sonar(camera, self.model, torch.zeros(3, device='cuda'), self.sonar)
        prediction = pkg['render']
        spec = self.cfg.get('loss', {'name': 'plain'})
        if spec['name'] == 'plain':
            loss = .8*l1_loss(prediction,target) + .2*(1-ssim(prediction,target))
        elif spec['name'] == 'return_weighted':
            loss, _ = return_weighted_loss(prediction, target, spec['threshold'], spec.get('dilate', 1))
        else:
            raise ValueError(f"unknown loss {spec['name']}")
        if not bool(torch.isfinite(loss)): raise ValueError('nonfinite training loss')
        loss.backward()
        for group in self.model.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                    raise ValueError('nonfinite training gradient')
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
        self.last_loss = float(loss.detach())
        # Diagnostics are not an RNG/strategy input, but per-camera sigma fallback is.
        del pkg, prediction, loss

    def snapshot(self):
        capture = list(self.model.capture())
        optimizer = capture.pop(10)
        # The xyz schedule returns numpy.float64; weights_only checkpoint loading rejects numpy scalars.
        for group in optimizer['param_groups']:
            group['lr'] = float(group['lr'])
        fallback = {str(i): c._sonar_sigma_fallback_state for i,c in self.cameras.items()
                    if hasattr(c,'_sonar_sigma_fallback_state')}
        return {'step':self.step_number, 'model':capture, 'optimizers':{'gaussians':optimizer},
            'schedulers':{'xyz_max_steps':self.opt.position_lr_max_steps},
            'sampler':{'remaining':self.remaining, 'fallback':fallback},
            'strategy':{'name':'fixed-density-v1','last_loss':self.last_loss},
            'scale':{'metric':1.,'learned':False}, 'rng':atomic.rng_state()}

    def restore(self, state):
        if state['strategy']['name'] != 'fixed-density-v1' or state['scale'] != {'metric':1.,'learned':False}:
            raise ValueError('unsupported strategy or metric scale')
        if state['schedulers'] != {'xyz_max_steps':self.cfg['steps']}:
            raise ValueError('schedule differs')
        capture = list(state['model']); capture.insert(10,state['optimizers']['gaussians'])
        # weights_only tensors are promoted back to trainable leaves explicitly.
        for k in range(1,7): capture[k] = torch.nn.Parameter(capture[k].detach().to('cuda'))
        for k in (7,8,9): capture[k] = capture[k].to('cuda')
        self.model.restore(capture,self.opt)
        self.step_number = state['step']
        self.remaining = list(state['sampler']['remaining'])
        for i,v in state['sampler']['fallback'].items():
            self.cameras[int(i)]._sonar_sigma_fallback_state = v.to('cuda')
        self.last_loss = state['strategy']['last_loss']
        atomic.restore_rng(state['rng'])

    def export(self, path):
        """Freeze oriented training points for separately bounded Poisson extraction."""
        m = self.model
        np.savez(path, xyz=m.get_xyz.detach().cpu().numpy(),
            normal=quaternion_to_normal(m.get_rotation).detach().cpu().numpy(),
            scale=m.get_scaling.detach().cpu().numpy(), opacity=m.get_opacity.detach().cpu().numpy())
