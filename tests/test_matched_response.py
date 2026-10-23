"""Native image response contracts; do not assert acoustic/geometry validity."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from directional_sonar import deposit, image_response


def test_native_response_empty_edges_saturation():
    for t,c,a in [([],[],[]),([.0001,.1,2.9999],[0,255,255],[1.,2.,3.]),([1.123]*3,[127]*3,[200.]*3)]:
        t=torch.tensor(t,dtype=torch.float64);c=torch.tensor(c,dtype=torch.long);a=torch.tensor(a,dtype=torch.float64)
        raw=deposit(t,c,a,'matched');expected=gaussian_filter(raw.numpy(),(.5,1.5/2.354820045/(120/256)),mode='constant')
        expected[(np.arange(200)+.5)*.015<.1]=0
        np.testing.assert_allclose(image_response(raw,'matched').numpy(),np.clip(expected,0,1),atol=1e-12,rtol=1e-12)


def test_linear_deposition_interior_gradient():
    t=torch.tensor([1.123],dtype=torch.float64,requires_grad=True);c=torch.tensor([100]);a=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
    w=torch.arange(200,dtype=torch.float64)[:,None].expand(200,256)
    (deposit(t,c,a,'matched')*w).sum().backward()
    np.testing.assert_allclose(t.grad.numpy(),[.4/.015],atol=1e-10)
    np.testing.assert_allclose(a.grad.numpy(),[1.123/.015-.5],atol=1e-10)
