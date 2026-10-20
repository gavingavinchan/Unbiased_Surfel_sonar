"""Independent single-Gaussian sonar CUDA forward/gradient check (native only)."""
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'native CUDA required')
class AdditiveGradientTests(unittest.TestCase):
    def test_analytic_footprint_and_gradient(self):
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        eye = torch.eye(4, device='cuda')
        settings = GaussianRasterizationSettings(image_height=32, image_width=32,
            tanfovx=1., tanfovy=1., bg=torch.zeros(3, device='cuda'), ndc2world=eye,
            scale_modifier=1., viewmatrix=eye, projmatrix=eye, sh_degree=0,
            campos=torch.zeros(3, device='cuda'), prefiltered=False, additive_mode=True, debug=False)
        xyz = torch.tensor([[0., 0., 1.]], device='cuda')
        matrix = torch.tensor([[2.5,0.,15.3,0.,1.8,14.7,0.,0.,1.]], device='cuda', requires_grad=True)
        color = torch.tensor([[.7,.5,.2]], device='cuda', requires_grad=True)
        opacity = torch.tensor([[.2]], device='cuda', requires_grad=True)
        image, radii, _, _ = GaussianRasterizer(settings)(means3D=xyz, means2D=torch.zeros_like(xyz),
            opacities=opacity, colors_precomp=color, cov3D_precomp=matrix)
        samples = [(15,15,1.),(14,16,-.5),(16,14,.3)]
        coefficients = np.array([1.,-.2,.4])
        def analytic(flat, c, o):
            tu,tv,tw = np.asarray(flat,dtype=np.float64).reshape(3,3)
            answer = 0.
            for y,x,w in samples:
                p = np.cross(x*tw-tu,y*tw-tv)
                sx,sy = p[:2]/p[2]
                g = np.exp(-.5*(sx*sx+sy*sy))
                answer += w*g*o*np.dot(coefficients,c)
            return answer
        c=color.detach().cpu().numpy()[0].astype(np.float64)
        m=matrix.detach().cpu().numpy()[0].astype(np.float64)
        o=float(opacity.detach().cpu()[0,0])
        loss=sum(w*(image[:,y,x]*torch.as_tensor(coefficients,device='cuda',dtype=torch.float32)).sum()
                 for y,x,w in samples)
        np.testing.assert_allclose(float(loss.detach()),analytic(m,c,o),atol=2e-6,rtol=2e-5)
        self.assertGreater(int(radii[0]),0)
        loss.backward()
        expected_color=np.array([analytic(m,np.eye(3)[i],o) for i in range(3)])
        np.testing.assert_allclose(color.grad.cpu().numpy()[0],expected_color,atol=3e-5,rtol=3e-4)
        np.testing.assert_allclose(float(opacity.grad.cpu()[0,0]),analytic(m,c,1.),atol=3e-5,rtol=3e-4)
        eps=1e-5
        expected_matrix=[]
        for i in range(9):
            delta=np.zeros(9);delta[i]=eps
            expected_matrix.append((analytic(m+delta,c,o)-analytic(m-delta,c,o))/(2*eps))
        np.testing.assert_allclose(matrix.grad.cpu().numpy()[0],expected_matrix,atol=3e-5,rtol=3e-4)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'native CUDA required')
class PackedCovarianceAcceptance(unittest.TestCase):
    """Staged 162 kernel acceptance; runs only in the registered native build.

    Real scalar/batch callers, both factorization paths, correlated Gaussian
    pixels and covariance/mean derivatives. The broad case catches omitted
    above-threshold pixels beyond the inherited 3-sigma tile footprint.
    """
    def test_correlated_packing_gradients_and_tail_containment(self):
        import ast
        from pathlib import Path
        from unittest.mock import patch
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        source=Path(__file__).resolve().parents[1]/'gaussian_renderer/__init__.py'
        names={'_bounded_sonar_linalg','_condition_sigma_2d','_covariance_cholesky_2x2',
               'sigma2d_to_transmat_precomp','_sigma2d_to_transmat_precomp_batch'}
        nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in nodes},names)
        ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
        eye=torch.eye(4,device='cuda')
        size=192
        settings=GaussianRasterizationSettings(image_height=size,image_width=size,
            tanfovx=1.,tanfovy=1.,bg=torch.zeros(3,device='cuda'),ndc2world=eye,
            scale_modifier=1.,viewmatrix=eye,projmatrix=eye,sh_degree=0,
            campos=torch.zeros(3,device='cuda'),prefiltered=False,additive_mode=True,debug=False)
        raster=GaussianRasterizer(settings)
        yy,xx=torch.meshgrid(torch.arange(size,dtype=torch.float64),torch.arange(size,dtype=torch.float64),indexing='ij')
        pixels=torch.stack((xx,yy),-1)
        # Fixed integer observations, away from clipping and branch boundaries.
        sy,sx=torch.meshgrid(torch.arange(90,106,2),torch.arange(88,104),indexing='ij')
        sy=sy.flatten();sx=sx.flatten()
        weights=torch.linspace(.1,.9,128,dtype=torch.float64)
        fixtures=[[[2.,.6],[.6,4.]],[[2.,-.6],[-.6,4.]],
                  [[7.,5.],[5.,6.]],[[390.,5.],[5.,380.]]]
        def failure(x):raise torch.linalg.LinAlgError('forced native acceptance fallback')
        for batch in (False,True):
            for fallback in (False,True):
                for raw in fixtures:
                    sigma=torch.tensor(raw,device='cuda',dtype=torch.float32,requires_grad=True)
                    mean=torch.tensor([97.2,99.3],device='cuda',dtype=torch.float32,requires_grad=True)
                    fn=ns['_sigma2d_to_transmat_precomp_batch' if batch else 'sigma2d_to_transmat_precomp']
                    def pack():return fn(mean[None],sigma[None]) if batch else fn(mean,sigma)
                    if fallback:
                        with patch.object(torch.linalg,'cholesky',failure):h=pack()
                    else:h=pack()
                    image,radii,_,_=raster(means3D=torch.tensor([[0.,0.,1.]],device='cuda'),
                        means2D=torch.zeros(1,3,device='cuda'),opacities=torch.tensor([[.7]],device='cuda'),
                        colors_precomp=torch.ones(1,3,device='cuda'),cov3D_precomp=h.reshape(1,9).contiguous())
                    ref_sigma=sigma.detach().cpu().double().requires_grad_()
                    ref_mean=mean.detach().cpu().double().requires_grad_()
                    target=ns['_condition_sigma_2d'](ref_sigma)
                    delta=pixels-ref_mean
                    solved=torch.linalg.solve(target,delta.reshape(-1,2).T).T.reshape(size,size,2)
                    rho=torch.minimum((delta*solved).sum(-1),2*delta.square().sum(-1))
                    alpha=.7*torch.exp(-.5*rho)
                    expected=torch.where(alpha>=1/255,alpha,torch.zeros_like(alpha))
                    np.testing.assert_allclose(image[0].detach().cpu().numpy(),expected.detach().numpy(),atol=2e-6,rtol=2e-5)
                    self.assertGreater(int(radii[0]),0)
                    gpu_loss=(image[0,sy.to('cuda'),sx.to('cuda')]*weights.to(device='cuda',dtype=torch.float32)).sum()
                    gpu_grads=torch.autograd.grad(gpu_loss,(sigma,mean))
                    ref_loss=(expected[sy,sx]*weights).sum()
                    ref_grads=torch.autograd.grad(ref_loss,(ref_sigma,ref_mean))
                    for actual,reference in zip(gpu_grads,ref_grads):
                        np.testing.assert_allclose(actual.detach().cpu().numpy(),reference.detach().numpy(),atol=3e-5,rtol=3e-4)


if __name__ == '__main__':
    unittest.main()
