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


if __name__ == '__main__':
    unittest.main()
